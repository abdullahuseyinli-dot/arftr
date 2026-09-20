"""One authorized ten-fit method-fidelity experiment; no router or extraction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for search in (ROOT, ROOT / "src", ROOT / "experiments"):
    sys.path.insert(0, str(search))

from experiments import run_okutama_native_motion_innovation as legacy
from experiments import run_okutama_source_posture as provenance
from hac.actor_memory_base import canonical_hash, file_sha256, group_splits
from hac.motion_null_contrast import MotionNullContrast, exact_retaining_candidate
from hac.native_motion_data import load_native_motion_cache, motion_geometry, phase_inputs
from hac.native_motion_innovation import NativeMotionInnovation, motion_loss

RUN = ROOT / ".runs/research_20260920/motion_null_contrast_v1"
PROTOCOL = ROOT / "experiments/okutama_motion_null_contrast_protocol.json"
ARMS = ("plain", "paired_null")
STATUS = "MOTION_NULL_TEN_FITS_LOCKED_BEFORE_NEW_TASK_OUTCOMES"
read = legacy.read_json
write = legacy.write_json
arrays = legacy.load_predictions
record = legacy.record


def verify_record(item):
    path = ROOT / item["path"]
    if file_sha256(path) != item["sha256"]:
        raise RuntimeError(f"Changed dependency: {path}")
    return path


def check_preservation():
    receipt = read(ROOT / ".runs/research_20260920/research_closeout_v1/preservation_manifest.json")
    for item in receipt["files"]:
        verify_record(item)
    return len(receipt["files"])


def ancestor_scope(d, train, held, ancestry, dependencies, checked, plan, fold):
    """Verify exact reusable populations and hash-bound descendant receipts.

    Neural ancestor inference is inherited from the prior independent audit;
    this check does not pretend to refit/replay those hundreds of checkpoints.
    """
    outer_train = set(train.tolist())
    outer_held = set(held.tolist())
    for entry in ancestry["ancestry"]:
        if "prior_F_id" not in entry:
            continue
        pid = entry["prior_F_id"]
        folder = provenance.ANCESTORS / "populations" / pid
        receipt_path = folder / "receipt.json"
        prediction_path = folder / "predictions.npz"
        if file_sha256(receipt_path) != entry["prior_receipt_sha256"]:
            raise RuntimeError("Training prior receipt no longer matches historical producer")
        receipt = read(receipt_path)
        if file_sha256(prediction_path) != entry["prior_predictions_sha256"]:
            raise RuntimeError("Training prior predictions changed")
        saved = arrays(prediction_path)
        population = set(saved["population_rows"].tolist())
        block = plan["outer_plans"][fold]["blocks"][entry["block"]]
        target = set(np.flatnonzero(np.isin(d["scenarios"], block["target_scenarios"])).tolist())
        if population != outer_train - target or population & outer_held:
            raise RuntimeError("Ancestor population escapes the outer training fold")
        if canonical_hash(d["sample_ids"][sorted(target)].tolist()) != entry["target_sample_ids_sha256"]:
            raise RuntimeError("Training-prior target-block identity mismatch")
        if receipt["outer_prediction_labels_read"] != 0:
            raise RuntimeError("Ancestor used prediction labels")
        dependencies.update((receipt_path, prediction_path, folder / "independent_audit.json"))
        if pid in checked:
            continue
        audit = read(folder / "independent_audit.json")
        if not audit.get("prediction_label_counterfactual_invariance") or not all(audit["exact_recomposition"].values()):
            raise RuntimeError("Historical independent population audit did not pass")
        # Explicitly revisit transitive supervised-fit scopes, not just a pass flag.
        population_rows = saved["population_rows"]
        splits = group_splits(d["labels"], d["scenarios"], population_rows)
        expected_inner = [pair for _config in range(4) for pair in splits]
        for family in ("m4_fit_receipts", "a3_fit_receipts"):
            if len(receipt[family]) != 15 or len(expected_inner) != 12:
                raise RuntimeError("Unexpected ancestor fit inventory")
            for index, item in enumerate(receipt[family]):
                path = verify_record(item)
                dependencies.add(path)
                child = read(path)
                request = child.get("request", {})
                expected_train = expected_inner[index][0] if index < 12 else population_rows
                if "train_rows" in child:
                    if isinstance(child["train_rows"], list):
                        if child["train_rows"] != expected_train.tolist():
                            raise RuntimeError("Descendant training rows differ from expected split")
                    elif child["train_rows"] != len(expected_train):
                        raise RuntimeError("Descendant training row count differs")
                if family == "a3_fit_receipts":
                    if request.get("train_ids_sha256") != canonical_hash(d["sample_ids"][expected_train].tolist()):
                        raise RuntimeError("A3 descendant training identities differ from expected split")
                if child.get("outer_prediction_labels_read", 0) != 0:
                    raise RuntimeError("Descendant read outer labels")
                if request.get("stage") == "outer_refit":
                    if request.get("selection_labels_sha256") is not None or not child.get("outer_held_metrics_embargoed"):
                        raise RuntimeError("A3 outer-refit embargo violated")
        checked.add(pid)


def prepare(run: Path):
    if (run / "execution_lock.json").exists():
        raise RuntimeError("Existing execution lock; do not re-prepare")
    protocol = read(PROTOCOL)
    if protocol["arms"] != list(ARMS) or protocol["updates"] != 256:
        raise RuntimeError("Unexpected experimental contract")
    preserved = check_preservation()
    old_lock = legacy.validate_lock(legacy.DEFAULT_RUN)
    d = legacy.data()
    plan = provenance.fixed_recipe_plan(d["labels"], d["scenarios"], d["folds"], d["sample_ids"], enforce_canonical_counts=True)
    ancestor_paths, _ = provenance._ancestor_population_receipts(plan)
    dependencies = {ROOT / x["path"] for x in old_lock["dependencies"]}
    dependencies.update(ancestor_paths)
    dependencies.update((PROTOCOL, Path(__file__), ROOT / "src/hac/motion_null_contrast.py",
                         ROOT / "tests/test_motion_null_contrast.py",
                         ROOT / "experiments/audit_okutama_motion_null_contrast.py",
                         ROOT / ".runs/research_20260920/reopening_council_v1/REPORT.md"))
    historical_global_audit = read(provenance.ANCESTOR_AUDIT)
    for item in historical_global_audit["population_audits"]:
        dependencies.add(verify_record(item))
    populations = {}
    checked = set()
    for fold in range(5):
        train, held, train_anchor, ancestry = legacy.ancestry.outer_anchors(d, fold)
        if set(d["scenarios"][train]) & set(d["scenarios"][held]):
            raise RuntimeError("Outer scenarios overlap")
        old_folder = legacy.DEFAULT_RUN / "signed_motion" / f"fold-{fold}" / "final"
        old = read(old_folder / "receipt.json")
        if ancestry != old["ancestry"]:
            raise RuntimeError("Ancestry differs from the exact original recipe")
        ancestor_scope(d, train, held, ancestry, dependencies, checked, plan, fold)
        transform, transform_path = legacy.geometry_transform(run, d, train, f"outer-{fold}_final")
        original_transform = ROOT / old["transform_path"]
        old_transform = arrays(original_transform)
        if not np.array_equal(transform.mean, old_transform["mean"]) or not np.array_equal(transform.scale, old_transform["scale"]):
            raise RuntimeError("Geometry normalization differs from historical control")
        fit_hash = canonical_hash(d["sample_ids"][train].tolist())
        held_hash = canonical_hash(d["sample_ids"][held].tolist())
        if fit_hash != old["fit_ids_sha256"] or held_hash != old["held_ids_sha256"]:
            raise RuntimeError("Fit population differs from original")
        path = run / "inputs" / f"fold-{fold}.npz"
        legacy.write_npz(path, train_rows=train, held_rows=held, train_anchor=train_anchor,
                         train_labels=d["labels"][train], fit_ids_sha256=np.asarray(fit_hash),
                         held_ids_sha256=np.asarray(held_hash))
        dependencies.update((path, transform_path, original_transform,
                             old_folder / "receipt.json", old_folder / "checkpoint.pt", old_folder / "predictions.npz"))
        populations[str(fold)] = {"train_rows": len(train), "held_rows": len(held),
                                 "fit_ids_sha256": fit_hash, "held_ids_sha256": held_hash,
                                 "ancestry": ancestry, "training_label_scope": "train_rows_only"}
    cohort_path = run / "inputs" / "cohort.npz"
    legacy.write_npz(cohort_path, sample_ids=d["sample_ids"], folds=d["folds"], scenarios=d["scenarios"], anchor=d["anchor"])
    dependencies.add(cohort_path)
    # Bind the actually imported local dependency code as well as data receipts.
    for module in list(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if path.suffix == ".py" and (path.is_relative_to(ROOT / "src") or path.is_relative_to(ROOT / "experiments")):
                dependencies.add(path)
    lock = {"status": STATUS, "created_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": protocol["authorization"], "scope": protocol["scope"],
            "protocol": protocol, "new_task_fits_at_lock": 0, "rows": len(d["labels"]),
            "ancestry_populations_rechecked": len(checked), "preserved_files_verified": preserved,
            "held_labels_access": "cohort equality and fixed split validation only; no new held scores before ten-fit completion",
            "ancestry_audit_scope": "exact populations, old independent audit hashes, descendant receipt scopes; no ancestor neural refits",
            "populations": populations, "dependencies": [record(p) for p in sorted(dependencies, key=str)]}
    write(run / "execution_lock.json", lock)
    return {"status": STATUS, "planned_head_fits": 10, "ancestors_rechecked": len(checked), "dependencies": len(dependencies)}


def validate(run):
    lock = read(run / "execution_lock.json")
    if lock["status"] != STATUS or lock["new_task_fits_at_lock"] != 0:
        raise RuntimeError("Invalid experiment lock")
    for item in lock["dependencies"]:
        verify_record(item)
    return lock


def preflight(run, device):
    lock = validate(run)
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("This bounded screen requires the checked CUDA environment")
    torch.manual_seed(91)
    x = torch.randn(16, 2, 6, 64, 64, device=device)
    g = torch.randn(16, 12, device=device)
    anchor = torch.tensor([[0.05, 0.5, 0.45]], device=device).repeat(16, 1)
    available = torch.ones(16, 2, dtype=torch.bool, device=device)
    labels = torch.arange(16, device=device) % 3
    model = MotionNullContrast().to(device)
    initial = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    for _ in range(4):
        opt.zero_grad()
        loss, _ = motion_loss(model(x, g), anchor, labels, available)
        loss.backward()
        opt.step()
    if not all(not torch.equal(initial[k], v) for k, v in model.named_parameters() if v.requires_grad):
        raise RuntimeError("An active parameter tensor failed to update")
    model.eval()
    null = x.clone()
    null[:, :, 3:] = 0
    with torch.no_grad():
        delta = model(null, g).cpu().numpy().astype(np.float64)
    if np.count_nonzero(delta):
        raise RuntimeError("Trained null invariant failed")
    p = np.tile(np.asarray([0.05, 0.5, 0.45]), (16, 1))
    candidate, _ = exact_retaining_candidate(p, delta, np.ones((16, 2), bool))
    if not np.array_equal(candidate, p):
        raise RuntimeError("Null prediction did not retain exact anchor")
    receipt = {"status": "MOTION_NULL_PREFLIGHT_PASS", "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
               "synthetic_optimizer_steps": 4, "real_task_fits": 0, "trained_null_exact": True,
               "all_trainable_tensors_updated": True, "exact_anchor_retain": True,
               "plain_active_parameters": 120129, "paired_null_active_parameters": 120128,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "environment": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "torch")},
               "protocol": lock["protocol"]}
    write(run / "preflight.json", receipt)
    return {k: v for k, v in receipt.items() if k not in {"protocol", "environment"}}


def predict(model, cache, geometry, transform, rows, device):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            x = torch.from_numpy(phase_inputs(cache.crops[batch], "signed_motion")).to(device)
            g = torch.from_numpy(transform.apply(geometry[batch])).to(device)
            values.append(model(x, g).float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def fit_one(run, arm, fold, cohort, cache, geometry, device, queue_started):
    protocol = read(PROTOCOL)
    folder = run / arm / f"fold-{fold}"
    if folder.exists():
        raise RuntimeError(f"Refusing to overwrite or resume a fit: {folder}")
    inputs = arrays(run / "inputs" / f"fold-{fold}.npz")
    train, held = inputs["train_rows"], inputs["held_rows"]
    trans = arrays(run / "transforms" / f"outer-{fold}_final.npz")
    transform = legacy.GeometryTransform(trans["mean"], trans["scale"])
    torch.manual_seed(42)
    np.random.seed(42)
    model = (NativeMotionInnovation() if arm == "plain" else MotionNullContrast()).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(42)
    order = np.empty(0, np.int64)
    cursor = 0
    losses = []
    schedule_hash = hashlib.sha256()
    started = time.perf_counter()
    model.train()
    for update in range(1, 257):
        if time.perf_counter() - started > protocol["runtime_caps_seconds"]["per_fit"]:
            raise RuntimeError("Fixed per-fit runtime cap exceeded")
        if time.perf_counter() - queue_started > protocol["runtime_caps_seconds"]["queue"]:
            raise RuntimeError("Fixed queue runtime cap exceeded")
        if cursor + 128 > len(order):
            order = rng.permutation(len(train))
            cursor = 0
        local = order[cursor:cursor + 128]
        cursor += len(local)
        rows = train[local]
        schedule_hash.update(np.asarray(rows, dtype="<i8").tobytes())
        x = torch.from_numpy(phase_inputs(cache.crops[rows], "signed_motion")).to(device)
        g = torch.from_numpy(transform.apply(geometry[rows])).to(device)
        available = torch.from_numpy(cache.phase_available[rows]).to(device)
        anchor = torch.from_numpy(np.asarray(inputs["train_anchor"][local], np.float32)).to(device)
        labels = torch.from_numpy(inputs["train_labels"][local].astype(np.int64)).to(device)
        optimizer.zero_grad(set_to_none=True)
        delta = model(x, g)
        loss, pieces = motion_loss(delta, anchor, labels, available)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
        optimizer.step()
        if update == 1 or update % 16 == 0:
            losses.append({"update": update, "loss": float(loss.detach()), "gradient_norm": gradient,
                           **{k: float(v.detach()) for k, v in pieces.items()}})
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    folder.mkdir(parents=True)
    with (folder / "checkpoint.pt").open("xb") as stream:
        torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": 42}, stream)
    delta = predict(model, cache, geometry, transform, held, device)
    candidate, eligible = exact_retaining_candidate(cohort["anchor"][held], delta, cache.phase_available[held])
    legacy.write_npz(folder / "predictions.npz", sample_ids=cohort["sample_ids"][held], held_rows=held,
                     anchor_probabilities=cohort["anchor"][held], candidate_probabilities=candidate,
                     delta=delta, eligible=eligible)
    parity = None
    if arm == "plain":
        old = arrays(legacy.DEFAULT_RUN / "signed_motion" / f"fold-{fold}" / "final" / "predictions.npz")
        difference = float(np.abs(old["delta"] - delta).max())
        same_classes = np.array_equal(old["candidate_probabilities"].argmax(1), candidate.argmax(1))
        if difference > protocol["historical_plain_parity"]["delta_atol"] or not same_classes:
            raise RuntimeError(f"Historical plain recipe parity failed: delta={difference}")
        parity = {"max_abs_delta": difference, "same_class_predictions": bool(same_classes)}
    receipt = {"status": "MOTION_NULL_FIT_COMPLETE_OUTCOMES_EMBARGOED", "arm": arm, "fold": fold,
               "execution_lock_sha256": file_sha256(run / "execution_lock.json"), "seed": 42, "updates": 256,
               "active_parameters": model.trainable_parameters, "elapsed_training_seconds": elapsed,
               "training_schedule_sha256": schedule_hash.hexdigest(), "training_label_rows": len(train),
               "fit_ids_sha256": str(inputs["fit_ids_sha256"].item()), "held_ids_sha256": str(inputs["held_ids_sha256"].item()),
               "held_labels_passed_to_optimizer": False, "new_held_scores_computed": False,
               "historical_plain_parity": parity, "loss_curve": losses,
               "checkpoint_sha256": file_sha256(folder / "checkpoint.pt"),
               "predictions_sha256": file_sha256(folder / "predictions.npz")}
    write(folder / "receipt.json", receipt)
    del model, optimizer
    torch.cuda.empty_cache()
    print(json.dumps({"event": "fit_complete", "arm": arm, "fold": fold, "seconds": elapsed,
                      "historical_plain_parity": parity}), flush=True)


def queue(run, device):
    validate(run)
    if read(run / "preflight.json")["status"] != "MOTION_NULL_PREFLIGHT_PASS":
        raise RuntimeError("Preflight required")
    if any((run / arm).exists() for arm in ARMS):
        raise RuntimeError("Queue is one-shot; no automatic retraining/resume")
    cohort = arrays(run / "inputs" / "cohort.npz")
    cache = load_native_motion_cache(legacy.CACHE, cohort["sample_ids"])
    geometry = motion_geometry(cache.geometry, cache.camera)
    started = time.perf_counter()
    for fold in range(5):
        for arm in ARMS:
            fit_one(run, arm, fold, cohort, cache, geometry, device, started)
    result = {"status": "MOTION_NULL_TEN_FITS_COMPLETE", "head_fits": 10, "router_fits": 0,
              "new_held_scores_computed": False, "elapsed_queue_seconds": time.perf_counter() - started}
    write(run / "queue_receipt.json", result)
    return result


def transitions(y, anchor, candidate, folds):
    old = anchor.argmax(1) == y
    new = candidate.argmax(1) == y
    return {"rescues": int((new & ~old).sum()), "harms": int((~new & old).sum()),
            "net": int(new.sum() - old.sum()),
            "per_fold_net": [int(new[folds == f].sum() - old[folds == f].sum()) for f in range(5)]}


def summarize(run):
    lock = validate(run)
    audit = read(run / "replay_audit.json")
    if audit["status"] != "MOTION_NULL_SEPARATE_FORMULA_REPLAY_PASS":
        raise RuntimeError("Audit must pass before outcomes are released")
    cohort = arrays(run / "inputs" / "cohort.npz")
    y = arrays(legacy.ARFTR)["labels"].astype(int)
    scores = {"ARFTR": legacy.metrics(y, cohort["anchor"])}
    predictions = {}
    details = {}
    for arm in ARMS:
        p = np.full_like(cohort["anchor"], np.nan)
        for fold in range(5):
            saved = arrays(run / arm / f"fold-{fold}" / "predictions.npz")
            p[saved["held_rows"]] = saved["candidate_probabilities"]
        if not np.isfinite(p).all():
            raise RuntimeError("Incomplete predictions")
        predictions[arm] = p
        scores[arm] = legacy.metrics(y, p)
        details[arm] = transitions(y, cohort["anchor"], p, cohort["folds"])
        details[arm]["per_fold_metrics"] = [legacy.metrics(y[cohort["folds"] == f], p[cohort["folds"] == f]) for f in range(5)]
    d = {"labels": y, "scenarios": cohort["scenarios"]}
    intervals = {"paired_minus_plain": legacy.bootstrap(d, predictions["paired_null"], predictions["plain"]),
                 "paired_minus_ARFTR": legacy.bootstrap(d, predictions["paired_null"], cohort["anchor"])}
    base, paired, plain = scores["ARFTR"], scores["paired_null"], scores["plain"]
    criteria = lock["protocol"]["continuation_gates"]
    gates = {"paired_above_plain": paired["macro_f1"] > plain["macro_f1"],
             "paired_minus_plain_interval": intervals["paired_minus_plain"]["interval"][0] > 0,
             "macro_f1_gain": paired["macro_f1"] - base["macro_f1"] >= criteria["macro_f1_gain_vs_arftr_min"],
             "net": details["paired_null"]["net"] >= criteria["net_corrections_min"],
             "positive_every_fold": all(x > 0 for x in details["paired_null"]["per_fold_net"]),
             "paired_minus_ARFTR_interval": intervals["paired_minus_ARFTR"]["interval"][0] > 0,
             "nll": paired["nll"] <= base["nll"] + criteria["proper_score_tolerance"],
             "brier": paired["brier_sum"] <= base["brier_sum"] + criteria["proper_score_tolerance"],
             "class_f1": min(np.asarray(paired["per_class_f1"]) - np.asarray(base["per_class_f1"])) >= -criteria["maximum_class_f1_drop"]}
    result = {"status": "MOTION_NULL_EXPERIMENT_COMPLETE", "scores": scores, "transitions": details,
              "bootstrap": intervals, "gates": {k: bool(v) for k, v in gates.items()},
              "raw_continuation_gate_pass": bool(all(gates.values())), "promoted": False,
              "retained_ARFTR_changed": False, "head_fits": 10, "router_fits": 0,
              "scope": "adaptive_internal_raw_producer_diagnostic_not_deployable_router",
              "next_action": "request_separate_phase_authorization" if all(gates.values()) else "close_optional_experiment_return_to_repo_work",
              "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
              "replay_audit_sha256": file_sha256(run / "replay_audit.json")}
    write(run / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--stage", choices=("prepare", "preflight", "queue", "summarize"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run.resolve()
    result = prepare(run) if args.stage == "prepare" else preflight(run, args.device) if args.stage == "preflight" else queue(run, args.device) if args.stage == "queue" else summarize(run)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
