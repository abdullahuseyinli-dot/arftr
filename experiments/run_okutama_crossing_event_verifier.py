"""Locked, small CPU gate trial over frozen nested PDI producers; no producer fitting.

Outer outcomes are evaluated only after all 45 fits and independent replay pass.
Existing evidence is read-only. All outputs are exclusive/idempotent writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from hac.crossing_event_verifier import (  # noqa: E402
    VARIANTS,
    build_event_features,
    event_targets,
    fit_common_scaler,
    fit_verifier,
    route_actions,
)

PDI = ROOT / ".runs/research_20260919/paired_detail_innovation_v1_full"
ANCHOR = ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
REVIEW = ROOT / ".runs/research_20260919/post_pdi_aerial_review_v1"
PROTOCOL = ROOT / "experiments/okutama_crossing_event_verifier_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260919/crossing_event_verifier_v1"
READERS = ("coarse_residual", "paired_spatial", "paired_spatial_temporal")
ANCHOR_SHA256 = "ec9957a9393e6f1803d274d281549c20290ed59343b308c5be3446be37cec720"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def write_json(path: Path, value: dict):
    content = json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_text(path, content)


def write_text(path: Path, content: str):
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)


def save_npz(path: Path, **arrays):
    if path.exists():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {"path": path.resolve().relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(condition, explanation):
    if not condition:
        raise RuntimeError(explanation)


def canonical() -> dict:
    require(sha256(ANCHOR) == ANCHOR_SHA256, "retained ARFTR changed")
    result = npz(ANCHOR)
    result["anchor"] = result["mean_probabilities"][result["arms"].tolist().index("r5_arftr_full")]
    require(len(result["labels"]) == 4977, "wrong row count")
    require(len(set(result["sample_ids"].tolist())) == 4977, "duplicate IDs")
    require(set(result["labels"].tolist()) == {0, 1, 2}, "wrong labels")
    require(set(result["folds"].tolist()) == set(range(5)), "wrong folds")
    require(result["scenarios"].dtype.kind in "US", "scenarios must remain strings")
    require(len(np.unique(result["scenarios"])) == 11, "wrong scenario count")
    for scene in np.unique(result["scenarios"]):
        require(len(np.unique(result["folds"][result["scenarios"] == scene])) == 1,
                "scenario crosses outer folds")
    return result


def check_training_rows(rows, held, folds, outer):
    """Exact T/O coverage, not just disjointness or a zero-label-read counter."""
    require(np.array_equal(rows, np.flatnonzero(folds != outer)), "training rows not exact outer T")
    require(np.array_equal(held, np.flatnonzero(folds == outer)), "held rows not exact outer O")
    require(not np.intersect1d(rows, held).size, "outer label contamination")


def frozen_nll(verifier, original_features):
    transformed = (original_features - verifier["scaler_mean"]) / verifier["scaler_scale"]
    return np.column_stack([
        transformed[:, action] @ verifier[f"nll_coef_{action}"]
        + float(verifier[f"nll_intercept_{action}"].item()) for action in range(3)
    ])


def sources(reader, fold, data):
    folder = PDI / reader / f"fold-{fold}"
    source_receipt = read_json(folder / "receipt.json")
    for name, digest_key in (("inner_oof_verifier_inputs.npz", "inner_oof_verifier_inputs_sha256"),
                             ("predictions.npz", "predictions_sha256"), ("verifier.npz", "verifier_sha256")):
        require(sha256(folder / name) == source_receipt[digest_key], "frozen PDI fold output changed")
    inner = npz(folder / "inner_oof_verifier_inputs.npz")
    outer = npz(folder / "predictions.npz")
    verifier = npz(folder / "verifier.npz")
    final = npz(folder / "final/predictions.npz")
    for name in ("actions", "anchor_probabilities", "correction", "scaffold_norms"):
        require(np.array_equal(outer[name], final[name]), f"outer {name} differs from hash-verified final producer")
    rows, held = inner["rows"], outer["held_rows"]
    check_training_rows(rows, held, data["folds"], fold)
    require(np.array_equal(inner["sample_ids"], data["sample_ids"][rows]), "inner IDs differ")
    require(np.array_equal(outer["sample_ids"], data["sample_ids"][held]), "outer IDs differ")
    require(np.array_equal(outer["anchor_probabilities"], data["anchor"][held]), "outer anchor differs")
    names = verifier["feature_names"].astype(str).tolist()
    train_x, feature_names = build_event_features(inner["anchor_probabilities"], inner["actions"], inner["verifier_features"], names)
    held_x, held_names = build_event_features(outer["anchor_probabilities"], outer["actions"], outer["verifier_features"], names)
    require(feature_names == held_names and len(feature_names) == 35, "feature schema mismatch")
    return folder, inner, outer, verifier, train_x, held_x, feature_names


def validate_lock(run: Path) -> dict:
    lock = read_json(run / "execution_lock.json")
    require(lock["status"] == "CROSSING_EVENT_LOCKED_BEFORE_FITS", "invalid lock")
    for entry in lock["dependencies"]:
        path = ROOT / entry["path"]
        require(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
                f"locked dependency changed: {entry['path']}")
    return lock


def prepare(run: Path):
    if (run / "execution_lock.json").exists():
        return validate_lock(run)
    require(not run.exists() or not any(run.iterdir()), "unlocked nonempty run: use a new directory")
    run.mkdir(parents=True, exist_ok=True)
    from crossing_event_ancestry import audit_source_ancestry
    ancestry = audit_source_ancestry()
    require(ancestry["status"] == "CROSSING_EVENT_ANCESTRY_PASS", "ancestry did not pass")
    data, protocol = canonical(), read_json(PROTOCOL)
    original_spec = read_json(REVIEW / "NEXT_TRIAL_PROTOCOL.json")
    for key in ("readers", "new_verifiers", "gate_fits", "feature_count", "promotion", "minimum_training_support"):
        require(protocol[key] == original_spec[key], f"protocol drift: {key}")
    reports = []
    for reader in READERS:
        for fold in range(5):
            _, inner, _, verifier, x, _, _ = sources(reader, fold, data)
            rows = inner["rows"]
            cross, outcomes, _ = event_targets(inner["anchor_probabilities"], inner["actions"], data["labels"][rows])
            nll_positive = frozen_nll(verifier, x[:, :, :23]) > 0
            counts = {"crossing_centers": int(cross.any(1).sum()),
                      "rescue_centers": int((outcomes == 0).any(1).sum()),
                      "harm_centers": int((outcomes == 1).any(1).sum()),
                      "scenarios": len(np.unique(data["scenarios"][rows][cross.any(1)]))}
            for key, minimum in protocol["minimum_training_support"].items():
                require(counts[key] >= minimum, f"insufficient {key} for {reader}/{fold}: {counts[key]}")
            reports.append({"reader": reader, "outer_fold": fold, **counts,
                            "crossing_events": int(cross.sum()),
                            "rescue_centers_after_frozen_nll": int(((outcomes == 0) & nll_positive).any(1).sum()),
                            "harm_centers_after_frozen_nll": int(((outcomes == 1) & nll_positive).any(1).sum()),
                            "outer_labels_used_for_gate_fit_or_selection": False})
    preflight = {"status": "CROSSING_EVENT_PREFLIGHT_PASS", "training_only_capacity": reports,
                 "labels_loaded": "canonical label array is loaded; fitting/targets/scaler use only each T; no outer outcomes released before audit",
                 "new_producer_fits": 0, "planned_gate_fits": 45, "failures": []}
    write_json(run / "ancestry_audit.json", ancestry)
    write_json(run / "preflight.json", preflight)
    paths = [PROTOCOL, ANCHOR, REVIEW / "NEXT_TRIAL_PROTOCOL.md", REVIEW / "NEXT_TRIAL_PROTOCOL.json",
             REVIEW / "knowledge_graph_complete.json", REVIEW / "experiment_ledger_complete.json",
             PDI / "summary_v2.json", PDI / "independent_audit.json",
             ROOT / ".runs/research_20260908/evidence_memory/data/memory_data.npz",
             Path(__file__), ROOT / "src/hac/crossing_event_verifier.py",
             ROOT / "experiments/crossing_event_ancestry.py",
             ROOT / "experiments/audit_okutama_crossing_event_verifier.py",
             ROOT / "tests/test_crossing_event_verifier.py", ROOT / "tests/test_crossing_event_runner.py",
             ROOT / "tests/test_audit_okutama_crossing_event_verifier.py",
             ROOT / "tests/test_crossing_event_ancestry.py", run / "ancestry_audit.json", run / "preflight.json"]
    for reader in READERS:
        for fold in range(5):
            paths.extend(PDI / reader / f"fold-{fold}" / name for name in ("receipt.json", "verifier.npz", "predictions.npz", "inner_oof_verifier_inputs.npz"))
    dependencies = {item["path"]: {"path": item["path"], "bytes": (ROOT / item["path"]).stat().st_size, "sha256": item["sha256"]}
                    for item in ancestry["dependency_receipts"]}
    for path in paths:
        dependencies[path.relative_to(ROOT).as_posix()] = record(path)
    lock = {"status": "CROSSING_EVENT_LOCKED_BEFORE_FITS", "created_utc": datetime.now(UTC).isoformat(),
            "source_run": PDI.relative_to(ROOT).as_posix(), "gate_fits_at_lock": 0,
            "outer_outcomes_released_at_lock": 0, "planned_fits": 45,
            "dependencies": sorted(dependencies.values(), key=lambda x: x["path"]),
            "environment": {"python": sys.version, "platform": platform.platform(),
                            **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "torch")}},
            "fit_manifest": [{"reader": r, "outer_fold": f, "variant": v}
                             for r in READERS for f in range(5) for v in VARIANTS]}
    write_json(run / "execution_lock.json", lock)
    return lock


def fit_all(run: Path):
    validate_lock(run)
    data = canonical()
    start = time.perf_counter()
    count = 0
    with threadpool_limits(limits=1):
        for reader in READERS:
            for fold in range(5):
                folder, inner, outer, verifier, x, held_x, names = sources(reader, fold, data)
                scaler = fit_common_scaler(x)
                nll_scores = frozen_nll(verifier, outer["verifier_features"])
                for variant in VARIANTS:
                    destination = run / reader / f"fold-{fold}" / variant
                    if (destination / "receipt.json").exists():
                        previous = read_json(destination / "receipt.json")
                        require(previous["execution_lock_sha256"] == sha256(run / "execution_lock.json"), "resumed fit lock changed")
                        for name in ("model", "predictions"):
                            require(previous[f"{name}_sha256"] == sha256(destination / f"{name}.npz"), "resumed artifact changed")
                        count += 1
                        continue
                    require(not destination.exists() or not any(destination.iterdir()), "partial fit; do not silently overwrite")
                    destination.mkdir(parents=True, exist_ok=True)
                    fit_start = time.perf_counter()
                    model, fit_receipt = fit_verifier(x, inner["anchor_probabilities"], inner["actions"],
                                                     data["labels"][inner["rows"]], variant, scaler=scaler)
                    require(fit_receipt["converged"], "numerical failure; no scientific conclusion")
                    utility, probabilities = model.scores(held_x)
                    routed, choices, invalid = route_actions(outer["anchor_probabilities"], outer["actions"], held_x, utility, nll_scores)
                    save_npz(destination / "model.npz", **model.to_arrays())
                    save_npz(destination / "predictions.npz", held_rows=outer["held_rows"], sample_ids=outer["sample_ids"],
                             features=held_x, utility=utility, event_probabilities=probabilities,
                             nll_scores=nll_scores, choices=choices, probabilities=routed, invalid_rows=invalid)
                    write_json(destination / "receipt.json", {
                        "status": "GATE_FIT_COMPLETE_OUTER_OUTCOMES_EMBARGOED", "reader": reader,
                        "outer_fold": fold, "variant": variant, "train_rows": len(inner["rows"]),
                        "held_rows": len(outer["held_rows"]), "fit_receipt": fit_receipt,
                        "elapsed_seconds": time.perf_counter() - fit_start, "feature_names": names,
                        "model_sha256": sha256(destination / "model.npz"),
                        "predictions_sha256": sha256(destination / "predictions.npz"),
                        "execution_lock_sha256": sha256(run / "execution_lock.json"),
                        "source_inner_sha256": sha256(folder / "inner_oof_verifier_inputs.npz"),
                        "source_outer_sha256": sha256(folder / "predictions.npz"),
                        "source_verifier_sha256": sha256(folder / "verifier.npz")})
                    count += 1
                    print(f"gate {count}/45 complete: {reader} fold-{fold} {variant}; outcomes embargoed", flush=True)
    require(count == 45, "incomplete fit matrix")
    path = run / "fit_completion.json"
    if not path.exists():
        write_json(path, {"status": "ALL_45_GATE_FITS_COMPLETE", "elapsed_seconds": time.perf_counter() - start,
                          "gate_fits": count, "producer_fits": 0, "device": "CPU, BLAS threads=1"})


def metrics(labels, probability):
    labels, probability = np.asarray(labels), np.asarray(probability)
    predicted = probability.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=[0, 1, 2], zero_division=0)
    return {"rows": len(labels), "macro_f1": float(f1.mean()), "accuracy": float((predicted == labels).mean()),
            "errors": int((predicted != labels).sum()), "nll": float(-np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
            "brier_sum": float(np.square(probability - np.eye(3)[labels]).sum(1).mean()),
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(), "per_class_support": support.tolist(),
            "confusion": confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist()}


def transitions(labels, anchor, proposed):
    old, new = anchor.argmax(1) == labels, proposed.argmax(1) == labels
    rescue, harm = int((~old & new).sum()), int((old & ~new).sum())
    return {"rescues": rescue, "harms": harm, "net": rescue - harm}


def confusion_f1(confusions):
    tp = np.diagonal(confusions, axis1=-2, axis2=-1)
    denominator = confusions.sum(-2) + confusions.sum(-1)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0).mean(-1)


def clustered_interval(labels, anchor, proposed, scenarios, draws=10000, seed=20260919):
    scenes = np.unique(scenarios)
    def counts(p):
        return np.stack([confusion_matrix(labels[scenarios == s], p[scenarios == s].argmax(1), labels=[0, 1, 2]) for s in scenes])
    old, new = counts(anchor), counts(proposed)
    sampled = np.random.default_rng(seed).integers(0, len(scenes), size=(draws, len(scenes)))
    delta = confusion_f1(new[sampled].sum(1)) - confusion_f1(old[sampled].sum(1))
    return {"unit": "scenario", "clusters": len(scenes), "draws": draws, "seed": seed,
            "percentile_95": np.quantile(delta, [.025, .975]).tolist(), "positive_fraction": float((delta > 0).mean())}


def event_diagnostics(run, reader, variant, data):
    scores, targets, probabilities, weights, scenes = [], [], [], [], []
    fold_diagnostics = []
    for fold in range(5):
        source = npz(PDI / reader / f"fold-{fold}" / "predictions.npz")
        saved = npz(run / reader / f"fold-{fold}" / variant / "predictions.npz")
        held = saved["held_rows"]
        cross, outcome, _ = event_targets(source["anchor_probabilities"], source["actions"], data["labels"][held])
        fold_target, fold_score = outcome[cross], saved["utility"][cross]
        fold_binary = fold_target != 2
        scores.extend(saved["utility"][cross].tolist())
        targets.extend(outcome[cross].tolist())
        per_center = np.repeat((1 / np.maximum(cross.sum(1), 1))[:, None], 3, axis=1)
        fold_weight = per_center[cross]
        fold_diagnostics.append({"outer_fold": fold, "crossing_events": int(cross.sum()),
            "crossing_centers": int(cross.any(1).sum()),
            "mean_predicted_utility": float(np.average(fold_score, weights=fold_weight)) if cross.any() else None,
            "rescue_vs_harm_auc": float(roc_auc_score(fold_target[fold_binary] == 0, fold_score[fold_binary], sample_weight=fold_weight[fold_binary])) if len(np.unique(fold_target[fold_binary])) == 2 else None})
        weights.extend(per_center[cross].tolist())
        scenes.extend(np.repeat(data["scenarios"][held, None], 3, axis=1)[cross].tolist())
        if saved["event_probabilities"].size:
            probabilities.extend(saved["event_probabilities"][cross].tolist())
    target, score, weight, scene = map(np.asarray, (targets, scores, weights, scenes))
    binary = target != 2
    auc = float(roc_auc_score(target[binary] == 0, score[binary], sample_weight=weight[binary])) if len(np.unique(target[binary])) == 2 else None
    result = {"crossing_events": len(target), "outcome_counts": np.bincount(target.astype(int), minlength=3).tolist(),
              "rescue_vs_harm_auc": auc, "auc_weighting": "one total event weight per crossing center; exploratory",
              "pooled_auc_limitation": "scores from separately fitted fold gates can differ in location/calibration; inspect per_fold before attributing discrimination gains",
              "per_fold": fold_diagnostics,
              "mean_realized_utility": float(np.average(np.choose(target.astype(int), [1, -2, 0]), weights=weight)),
              "mean_predicted_utility": float(np.average(score, weights=weight))}
    if probabilities:
        p = np.asarray(probabilities)
        result["event_nll"] = float(np.average(-np.log(np.clip(p[np.arange(len(target)), target.astype(int)], 1e-12, 1)), weights=weight))
        result["event_brier_sum"] = float(np.average(np.square(p - np.eye(3)[target.astype(int)]).sum(1), weights=weight))
        for event, name in enumerate(("rescue", "harm", "both_wrong")):
            result[name + "_calibration_bins"] = []
            bin_id = np.minimum((10 * p[:, event]).astype(int), 9)
            for bin_index in range(10):
                take = bin_id == bin_index
                if take.any():
                    result[name + "_calibration_bins"].append({"lower": bin_index / 10, "events": int(take.sum()),
                        "predicted": float(np.average(p[take, event], weights=weight[take])),
                        "observed": float(np.average(target[take] == event, weights=weight[take]))})
    if reader == READERS[2] and variant == VARIANTS[2] and auc is not None:
        rng = np.random.default_rng(20260919)
        unique = np.unique(scene)
        bootstrap = []
        for _ in range(10000):
            sampled = rng.integers(0, len(unique), len(unique))
            multiplier = dict(zip(unique, np.bincount(sampled, minlength=len(unique)).tolist(), strict=True))
            w = weight * np.asarray([multiplier[s] for s in scene])
            use = binary & (w > 0)
            if len(np.unique(target[use])) == 2:
                bootstrap.append(roc_auc_score(target[use] == 0, score[use], sample_weight=w[use]))
        result["auc_scenario_bootstrap_95"] = np.quantile(bootstrap, [.025, .975]).tolist()
        result["auc_valid_draws"] = len(bootstrap)
    return result


def summarize(run: Path):
    validate_lock(run)
    audit = read_json(run / "independent_audit.json")
    require(audit["status"] == "CROSSING_EVENT_INDEPENDENT_NO_FIT_AUDIT_PASS" and audit["gate_replays"] == 45, "no release before complete independent replay")
    for stem in ("execution_lock", "preflight", "ancestry_audit"):
        require(audit[f"{stem}_sha256"] == sha256(run / f"{stem}.json"), "audit provenance changed before release")
    require(len(audit["audited_artifacts"]) >= 135, "missing audit input binding")
    for entry in audit["audited_artifacts"]:
        require(sha256(ROOT / entry["path"]) == entry["sha256"], "audited output changed before release")
    data, protocol = canonical(), read_json(PROTOCOL)
    baseline = metrics(data["labels"], data["anchor"])
    results, pooled = {}, {}
    for reader in READERS:
        for variant in VARIANTS:
            key = reader + "::" + variant
            probabilities = np.zeros_like(data["anchor"])
            choices = np.zeros(len(probabilities), dtype=np.int64)
            eligible_actions, eligible_centers, before_nll_actions, before_nll_centers = 0, 0, 0, 0
            for fold in range(5):
                saved = npz(run / reader / f"fold-{fold}" / variant / "predictions.npz")
                probabilities[saved["held_rows"]] = saved["probabilities"]
                choices[saved["held_rows"]] = saved["choices"]
                crossing = saved["features"][:, :, 22] > 0.5
                before_nll = crossing & (saved["utility"] > 0) & ~saved["invalid_rows"][:, None]
                before_nll_actions += int(before_nll.sum())
                before_nll_centers += int(before_nll.any(1).sum())
                eligible = crossing & (saved["utility"] > 0) & (saved["nll_scores"] > 0) & ~saved["invalid_rows"][:, None]
                eligible_actions += int(eligible.sum())
                eligible_centers += int(eligible.any(1).sum())
            pooled[key] = probabilities
            result = {**metrics(data["labels"], probabilities), **transitions(data["labels"], data["anchor"], probabilities),
                      "reader": reader, "variant": variant, "interventions": int(np.count_nonzero(choices)),
                      "eligible_actions": eligible_actions, "eligible_centers": eligible_centers,
                      "positive_crossing_actions_before_nll": before_nll_actions,
                      "positive_crossing_centers_before_nll": before_nll_centers,
                      "accepted_per_action": np.bincount(choices, minlength=4).tolist(),
                      "invalid_rows": sum(int(npz(run / reader / f"fold-{f}" / variant / "predictions.npz")["invalid_rows"].sum()) for f in range(5)),
                      "paired_scenario_bootstrap": clustered_interval(data["labels"], data["anchor"], probabilities, data["scenarios"]),
                      "per_fold": [], "per_scenario": [], "event_diagnostics": event_diagnostics(run, reader, variant, data)}
            for field, output in (("folds", "per_fold"), ("scenarios", "per_scenario")):
                for value in np.unique(data[field]):
                    mask = data[field] == value
                    result[output].append({"group": native(value), **metrics(data["labels"][mask], probabilities[mask]),
                                           **transitions(data["labels"][mask], data["anchor"][mask], probabilities[mask])})
            result["delta_f1"] = result["macro_f1"] - baseline["macro_f1"]
            results[key] = result
    primary_key = protocol["primary_reader"] + "::" + protocol["primary_verifier"]
    primary = results[primary_key]
    coarse = results[READERS[0] + "::" + VARIANTS[2]]
    spatial = results[READERS[1] + "::" + VARIANTS[2]]
    gates = protocol["promotion"]
    checks = {"f1_effect": primary["macro_f1"] >= gates["macro_f1_minimum"],
              "net_corrections": primary["net"] >= gates["net_corrections_minimum"],
              "positive_every_fold": all(f["net"] > 0 for f in primary["per_fold"]),
              "dense_detail_over_coarse": primary["macro_f1"] - coarse["macro_f1"] >= gates["minimum_f1_delta_vs_coarse_same_verifier"],
              "cluster_interval_positive": primary["paired_scenario_bootstrap"]["percentile_95"][0] > 0,
              "nll_preserved": primary["nll"] <= gates["nll_maximum"],
              "brier_preserved": primary["brier_sum"] <= gates["brier_sum_maximum"],
              "class_f1_preserved": bool(np.all(np.asarray(primary["per_class_f1"]) >= np.asarray(baseline["per_class_f1"]) - gates["maximum_class_f1_drop"])),
              "independent_audit_and_ancestry": True}
    passed = all(checks.values())
    summary = {"status": "CROSSING_EVENT_COMPLETE_GO" if passed else "CROSSING_EVENT_COMPLETE_NO_GO",
               "evaluation_status": protocol["evaluation_status"], "primary": primary_key,
               "baseline": baseline, "results": results, "promotion_checks": checks,
               "promotion_passed": passed, "no_posthoc_secondary_promotion": True,
               "temporal_detail_claim": passed and primary["macro_f1"] > spatial["macro_f1"] and sum(a["macro_f1"] > b["macro_f1"] for a, b in zip(primary["per_fold"], spatial["per_fold"], strict=True)) >= 4,
               "next_action": "conditional_producer_seeds_43_44" if passed else protocol["failure_action"],
               "gate_fits": 45, "new_producer_fits": 0, "execution_lock_sha256": sha256(run / "execution_lock.json"),
               "audit_sha256": sha256(run / "independent_audit.json"), "retained_arftr_sha256": sha256(ANCHOR),
               "original_v0_reference": {"source": str((PDI / "summary_v2.json").relative_to(ROOT)), "interpretation": "all three saved V0 gates retain the ARFTR class decisions, 0 rescues/0 harms; calibration-only probability changes are not composed with new gates"}}
    summary["fixed_mechanism_contrasts"] = {
        reader: {"conditional_ridge_minus_all_event_ridge_f1": results[reader + "::" + VARIANTS[1]]["macro_f1"] - results[reader + "::" + VARIANTS[0]]["macro_f1"],
                 "multinomial_minus_conditional_ridge_f1": results[reader + "::" + VARIANTS[2]]["macro_f1"] - results[reader + "::" + VARIANTS[1]]["macro_f1"],
                 "qualification": "fixed matched contrasts in adaptive development; not proof of a unique causal explanation"}
        for reader in READERS}
    write_json(run / "summary.json", summary)
    lines = ["# Crossing-event verifier: " + ("GO" if passed else "NO-GO"), "",
             "45/45 fixed CPU gate fits; no PDI/ARFTR retraining. Independent replay and recursive ancestry passed.", "",
             "| Frozen reader | Gate | Macro-F1 | Rescue/harm | Net | Fold nets |",
             "|---|---|---:|---:|---:|---|"]
    for result in results.values():
        lines.append(f"| {result['reader']} | {result['variant']} | {100*result['macro_f1']:.6f}% | {result['rescues']}/{result['harms']} | {result['net']:+d} | {[f['net'] for f in result['per_fold']]} |")
    lines += ["", f"Retained ARFTR: {100*baseline['macro_f1']:.6f}% macro-F1, {baseline['errors']} errors; unchanged.",
              f"Primary: {primary_key}. Scenario-bootstrap F1 difference 95% interval: {primary['paired_scenario_bootstrap']['percentile_95']}.",
              "", "Promotion checks: " + json.dumps(checks, sort_keys=True), "", "Next action: " + summary["next_action"], "",
              "Interpretation: these are adaptive internal-development results, not independent generalization confirmation. Failure does not prove temporal information is useless; it closes this fixed producer/feature/gate family without a threshold sweep.", "",
              "All per-class, fold, scenario, calibration and ranking diagnostics are in summary.json. Original locks and probabilities were not modified.", ""]
    write_text(run / "REPORT.md", "\n".join(lines))
    import io
    buffer = io.StringIO(newline="")
    fields = ["reader", "variant", "macro_f1", "accuracy", "errors", "rescues", "harms", "net", "nll", "brier_sum", "interventions"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for result in results.values():
        writer.writerow({k: result[k] for k in fields})
    write_text(run / "compact_results.csv", buffer.getvalue())
    graph = read_json(REVIEW / "knowledge_graph_complete.json")
    ledger = read_json(REVIEW / "experiment_ledger_complete.json")
    artifact = (run / "summary.json").relative_to(ROOT).as_posix()
    for key, result in results.items():
        nid = "experiment:crossing_event_verifier_v1/" + key
        graph["nodes"].append({"id": nid, "kind": "experiment_arm", "label": f"{key}: {100*result['macro_f1']:.4f}% net {result['net']:+d}", "artifact": artifact,
                               "macro_f1": result["macro_f1"], "status": "primary_go" if passed and key == primary_key else "not_promoted"})
        graph["edges"].append({"source": "experiment:arftr_v1/r5_arftr_full", "target": nid, "relation": "evaluated-against",
                               "effect_size_f1_points": 100*result["delta_f1"], "artifact": artifact,
                               "confidence": "paired_scenario_bootstrap", "evidence_strength": "45fits_independent_replay_recursive_ancestry"})
        ledger["records"].append({"experiment_id": "crossing_event_verifier_v1::" + key,
                                  "family": "crossing_event_verifier_v1", "artifact": artifact,
                                  "audit_status": "independent_no_fit_replay_pass", "outcome": "primary_go" if passed and key == primary_key else "not_promoted",
                                  **{k: result[k] for k in fields}, "per_fold_net": [f["net"] for f in result["per_fold"]]})
        graph["edges"].append({"source": "plan:crossing_event_verifier_v1", "target": nid,
                               "relation": "evaluated-on", "artifact": artifact, "evidence_strength": "fixed_preregistered_internal_contrast"})
    primary_node = "experiment:crossing_event_verifier_v1/" + primary_key
    graph["edges"].append({"source": primary_node, "target": "hypothesis:structural_zero_dilution",
                           "relation": "supports-hypothesis" if passed else "contradicts",
                           "confidence": "restricted_to_this_frozen_bank_feature_schema_and_fixed_gate; not_a_universal_falsification",
                           "effect_size_f1_points": 100 * primary["delta_f1"], "artifact": artifact})
    if not passed:
        graph["edges"].append({"source": primary_node, "target": "plan:learned_persistent_tracking_v2",
                               "relation": "suggests-next-test", "artifact": artifact,
                               "evidence_strength": "prespecified_conditional_path"})
    graph["node_count"] = len(graph["nodes"])
    graph["edge_count"] = len(graph["edges"])
    graph["current_primary"] = primary_key
    graph["current_primary_outcome"] = summary["status"]
    graph["current_protocol"] = PROTOCOL.relative_to(ROOT).as_posix()
    graph["next_action"] = summary["next_action"]
    graph["parents"] = [str((REVIEW / "knowledge_graph_complete.json").relative_to(ROOT))]
    ledger["record_count"] = len(ledger["records"])
    ledger["parent_ledgers"] = [str((REVIEW / "experiment_ledger_complete.json").relative_to(ROOT))]
    ledger["supplement"] = {"source": artifact, "records_added": 9, "all_fits": 45, "primary_outcome": summary["status"]}
    write_json(run / "knowledge_graph.json", graph)
    write_json(run / "experiment_ledger.json", ledger)
    import networkx as nx
    network = nx.MultiDiGraph()
    def attributes(item, excluded):
        return {k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                for k, v in item.items() if k not in excluded and v is not None}
    for node in graph["nodes"]:
        network.add_node(node["id"], **attributes(node, {"id"}))
    for edge in graph["edges"]:
        network.add_edge(edge["source"], edge["target"], **attributes(edge, {"source", "target"}))
    graphml = "\n".join(nx.generate_graphml(network)) + "\n"
    write_text(run / "knowledge_graph.graphml", graphml)
    write_text(run / "MIND_MAP.md", "# Updated decision branch\n\n```mermaid\nflowchart TD\n A[ARFTR 85.383648% retained] --> B[Frozen PDI readers: available crossing proposals]\n B --> C[V0: no class interventions]\n C --> D[45 fixed V1/V2/V3 gate fits]\n D --> E[Independent replay and ancestry PASS]\n E --> F[" + ("Primary GO: confirm seeds 43 and 44" if passed else "Primary NO-GO: close frozen bank family") + "]\n F --> G[" + ("All safety gates remain required" if passed else "Corrected persistent geometry plus learned tracking measurement") + "]\n```\n\nSee REPORT.md and summary.json for exact observed effects; no oracle is an achieved model result.\n")
    print(json.dumps({"status": summary["status"], "primary_f1": primary["macro_f1"], "net": primary["net"], "checks": checks}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("prepare", "fit", "audit", "summarize", "all"), default="all")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage in ("prepare", "all"):
        prepare(run)
    if args.stage in ("fit", "all"):
        fit_all(run)
    if args.stage in ("audit", "all"):
        from audit_okutama_crossing_event_verifier import audit
        audit(run)
    if args.stage in ("summarize", "all"):
        summarize(run)


if __name__ == "__main__":
    main()
