"""Independently replay and open the outer-metric embargo for cached ARFTR."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_evidence_memory as evidence

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.arftr import ARFTRParameters, apply_arftr, exact_track_neighbors, shuffled_neighbors
from hac.matr_artifacts import load_cached_study_data, selected_fold_artifacts
from hac.source_swap_data import immutable_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
PROTOCOL = ROOT / "experiments/okutama_arftr_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/arftr_v1"
ARMS = (
    "r0_exact_m4",
    "r1_p6_restoration_only",
    "r2_a3_motion_only",
    "r3_temporal_only",
    "r4_residual_no_temporal",
    "r5_arftr_full",
    "r6_shuffled_neighbor_control",
)
SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def same(left: Any, right: Any, message: str) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        require(np.array_equal(left, right), message)
    else:
        require(left == right, message)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _validate_lock(run: Path, data: dict[str, np.ndarray]) -> tuple[dict[str, Any], str]:
    lock_path = run / "execution_lock.json"
    lock_hash = file_sha256(lock_path)
    lock = _read_json(lock_path)
    require(
        lock.get("status") == "ARFTR_EXECUTION_LOCKED_ADAPTIVE_OUTER_METRICS_EMBARGOED",
        "Execution-lock status changed",
    )
    same(lock.get("sample_ids_sha256"), canonical_hash(data["sample_ids"].tolist()), "Lock IDs")
    same(lock.get("primary_arm"), ARMS[5], "Lock primary")
    same(lock.get("cached_parameter_candidates"), 1500, "Lock candidate count")
    for record in lock["input_files"]:
        path = (ROOT / record["path"]).resolve()
        path.relative_to(ROOT.resolve())
        require(path.is_file(), f"Locked input is missing: {record['path']}")
        same(path.stat().st_size, record["size_bytes"], f"Locked input size: {record['path']}")
        same(file_sha256(path), record["sha256"], f"Locked input hash: {record['path']}")
    return lock, lock_hash


def _cache(data: dict[str, np.ndarray]):
    base_lock = _read_json(SOURCE / "base_execution_lock.json")
    raw = evidence.load_data(SOURCE)
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        same(raw[name], data[name], f"P6 identity: {name}")
    cache = evidence.base_cache(SOURCE, raw, base_lock["original_learning_protocol"])
    preparation = _read_json(SOURCE / "base_preparation.json")
    same(preparation.get("base_identity"), cache.identity, "P6 base identity")
    same(preparation.get("populations"), 58, "P6 population inventory")
    same(preparation.get("estimator_fits"), 3016, "P6 fit inventory")
    return cache


def _grid(protocol: dict[str, Any]) -> tuple[ARFTRParameters, ...]:
    selection = protocol["selection"]
    result = tuple(
        ARFTRParameters(*values)
        for values in itertools.product(
            selection["posture_restoration"],
            selection["motion_restoration"],
            selection["a3_motion_residual"],
            selection["temporal_strength"],
        )
    )
    require(len(result) == 300, "Auditor grid is not the frozen 300 candidates")
    return result


def _selection_key(metrics: dict[str, Any], parameters: ARFTRParameters) -> tuple[Any, ...]:
    values = tuple(float(value) for value in parameters.as_array())
    return (-metrics["macro_f1"], metrics["nll"], sum(values), values)


def _select(
    labels: np.ndarray,
    m4: np.ndarray,
    p6: np.ndarray,
    a3: np.ndarray,
    neighbors: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[ARFTRParameters, dict[str, Any]]:
    evaluated = []
    for parameters in _grid(protocol):
        probabilities = apply_arftr(
            m4,
            p6,
            a3,
            neighbors,
            parameters,
            epsilon=protocol["factorization"]["epsilon"],
        )
        evaluated.append((parameters, probability_metrics(labels, probabilities)))
    return min(evaluated, key=lambda item: _selection_key(item[1], item[0]))


def _arm_parameters(parameters: ARFTRParameters) -> tuple[ARFTRParameters, ...]:
    posture, motion, a3, temporal = parameters.as_array().tolist()
    return (
        ARFTRParameters(0.0, 0.0, 0.0, 0.0),
        ARFTRParameters(posture, motion, 0.0, 0.0),
        ARFTRParameters(0.0, 0.0, a3, 0.0),
        ARFTRParameters(0.0, 0.0, 0.0, temporal),
        ARFTRParameters(posture, motion, a3, 0.0),
        parameters,
        parameters,
    )


def _transitions(labels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> dict[str, int]:
    candidate_label, anchor_label = candidate.argmax(1), anchor.argmax(1)
    candidate_correct, anchor_correct = candidate_label == labels, anchor_label == labels
    rescues = int((candidate_correct & ~anchor_correct).sum())
    harms = int((~candidate_correct & anchor_correct).sum())
    return {
        "rescues": rescues,
        "harms": harms,
        "net_corrections": rescues - harms,
        "prediction_changes": int((candidate_label != anchor_label).sum()),
    }


def exact_scenario_swap(
    labels: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    scenarios: np.ndarray,
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    require(len(groups) == 11, "Exact swap requires the locked eleven scenarios")
    rows = {group: np.flatnonzero(scenarios == group) for group in groups}
    observed = probability_metrics(labels, candidate)["macro_f1"] - probability_metrics(
        labels, anchor
    )["macro_f1"]
    null = np.empty(2 ** len(groups), dtype=np.float64)
    for index, assignment in enumerate(itertools.product((False, True), repeat=len(groups))):
        left, right = candidate.copy(), anchor.copy()
        for keep, group in zip(assignment, groups, strict=True):
            if not keep:
                selected = rows[group]
                left[selected], right[selected] = anchor[selected], candidate[selected]
        null[index] = probability_metrics(labels, left)["macro_f1"] - probability_metrics(
            labels, right
        )["macro_f1"]
    return {
        "scenario_groups": int(len(groups)),
        "assignments": int(len(null)),
        "enumeration_complete": bool(len(null) == 2048),
        "alternative": "ARFTR macro-F1 exceeds exact M4",
        "observed_macro_f1_delta": float(observed),
        "one_sided_p_value": float(np.mean(null >= observed - 1e-15)),
        "null_minimum": float(null.min()),
        "null_maximum": float(null.max()),
        "random_sampling": False,
    }


def paired_scenario_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    scenarios: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    group_rows = [np.flatnonzero(scenarios == group) for group in groups]
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(groups), size=(resamples, len(groups)), dtype=np.int64)
    delta = np.empty(resamples, dtype=np.float64)
    for index, selection in enumerate(draw):
        rows = np.concatenate([group_rows[value] for value in selection])
        delta[index] = probability_metrics(labels[rows], candidate[rows])[
            "macro_f1"
        ] - probability_metrics(labels[rows], anchor[rows])["macro_f1"]
    interval = np.quantile(delta, (0.025, 0.5, 0.975))
    return {
        "unit": "scenario",
        "scenario_groups": int(len(groups)),
        "resamples": int(resamples),
        "seed": int(seed),
        "delta_quantiles_2_5_50_97_5": interval.tolist(),
        "probability_delta_not_positive": float(np.mean(delta <= 0)),
    }


def audit(run: Path) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL)
    require(tuple(protocol["arms"]) == ARMS, "Protocol arms changed")
    require(protocol["primary_arm"] == ARMS[5], "Protocol primary changed")
    require(protocol["statistics"]["exact_scenario_swap_assignments"] == 2048, "Swap design")
    data = load_cached_study_data(SOURCE)
    lock, lock_hash = _validate_lock(run, data)
    cache = _cache(data)
    rows = len(data["labels"])
    seed_probabilities = np.full((len(ARMS), len(SEEDS), rows, 3), np.nan)
    fold_parameters: list[list[float]] = []
    fold_receipts: dict[str, str] = {}
    p6_oof = np.full((rows, 3), np.nan)
    a3_oof = np.full((len(SEEDS), rows, 3), np.nan)
    for fold in range(5):
        directory = run / f"fold-{fold}"
        prediction_path, checkpoint_path, receipt_path = (
            directory / "predictions.npz",
            directory / "checkpoint.npz",
            directory / "receipt.json",
        )
        receipt = _read_json(receipt_path)
        prediction, checkpoint = _load_npz(prediction_path), _load_npz(checkpoint_path)
        held = np.flatnonzero(data["folds"] == fold)
        require(
            receipt.get("status") == "ARFTR_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED",
            f"Fold {fold} status",
        )
        same(receipt.get("execution_lock_sha256"), lock_hash, f"Fold {fold} lock")
        same(receipt.get("outer_held_labels_read"), 0, f"Fold {fold} label embargo")
        same(receipt.get("cached_parameter_candidates"), 300, f"Fold {fold} candidates")
        same(file_sha256(prediction_path), receipt.get("predictions_sha256"), f"Fold {fold} predictions")
        same(file_sha256(checkpoint_path), receipt.get("checkpoint_sha256"), f"Fold {fold} checkpoint")
        same(prediction["held_rows"], held, f"Fold {fold} held rows")
        same(prediction["sample_ids"], data["sample_ids"][held], f"Fold {fold} sample IDs")
        same(tuple(prediction["arms"].tolist()), ARMS, f"Fold {fold} arms")
        same(tuple(prediction["seeds"].tolist()), SEEDS, f"Fold {fold} seeds")
        same(prediction["seed_probabilities"].shape, (len(ARMS), 3, len(held), 3), f"Fold {fold} shape")
        cached = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
        p6_all, ancestry = cache.meta_probabilities(cached.train_rows)
        p6_inner, p6_outer = p6_all[cached.train_rows], cache.held_predictions(cached.train_rows, held)
        same(receipt.get("p6_inner_ancestry"), ancestry, f"Fold {fold} P6 ancestry")
        inner_neighbors = exact_track_neighbors(data, cached.train_rows)
        outer_neighbors = exact_track_neighbors(data, held)
        shuffled = shuffled_neighbors(
            outer_neighbors,
            data["scenarios"][held],
            seed=int(protocol["control_contract"]["shuffle_seed"]) + fold,
        )
        same(checkpoint["inner_neighbor_map"], inner_neighbors, f"Fold {fold} inner edges")
        same(checkpoint["outer_neighbor_map"], outer_neighbors, f"Fold {fold} outer edges")
        same(checkpoint["shuffled_outer_neighbor_map"], shuffled, f"Fold {fold} shuffled edges")
        selected, training_metrics = _select(
            data["labels"][cached.train_rows],
            cached.m4_inner["probabilities"],
            p6_inner,
            cached.a3_inner["probabilities"],
            inner_neighbors,
            protocol,
        )
        same(checkpoint["parameters"], selected.as_array(), f"Fold {fold} selected parameters")
        same(receipt["selected_parameters"], selected.as_array().tolist(), f"Fold {fold} receipt parameters")
        require(
            abs(receipt["training_metrics"]["macro_f1"] - training_metrics["macro_f1"]) <= 1e-15
            and abs(receipt["training_metrics"]["nll"] - training_metrics["nll"]) <= 1e-15,
            f"Fold {fold} training selection replay",
        )
        arm_parameters = _arm_parameters(selected)
        for seed_index in range(3):
            for arm_index, parameters in enumerate(arm_parameters):
                neighbors = shuffled if arm_index == 6 else outer_neighbors
                replay = apply_arftr(
                    cached.m4_outer["probabilities"][seed_index],
                    p6_outer,
                    cached.a3_outer["probabilities"][seed_index],
                    neighbors,
                    parameters,
                    epsilon=protocol["factorization"]["epsilon"],
                )
                same(
                    prediction["seed_probabilities"][arm_index, seed_index],
                    replay,
                    f"Fold {fold} arm {arm_index} seed {seed_index} replay",
                )
        same(
            prediction["seed_probabilities"][0],
            cached.m4_outer["probabilities"],
            f"Fold {fold} exact M4 seed bytes",
        )
        same(prediction["p6_probabilities"], p6_outer, f"Fold {fold} P6 bytes")
        same(prediction["a3_seed_probabilities"], cached.a3_outer["probabilities"], f"Fold {fold} A3 bytes")
        seed_probabilities[:, :, held] = prediction["seed_probabilities"]
        p6_oof[held] = p6_outer
        a3_oof[:, held] = cached.a3_outer["probabilities"]
        fold_parameters.append(selected.as_array().tolist())
        fold_receipts[str(fold)] = file_sha256(receipt_path)
    require(np.isfinite(seed_probabilities).all(), "ARFTR OOF coverage is incomplete")
    source_reference = _load_npz(SOURCE / "results/v0001/oof_probabilities.npz")
    sear_reference = _load_npz(SEAR / "results/v0001/oof_probabilities.npz")
    same(seed_probabilities[0], source_reference["new_source_m4_seeds"], "Official M4 OOF replay")
    same(p6_oof, source_reference["new_source_p6"], "Official P6 OOF replay")
    same(a3_oof, sear_reference["a3_unrestricted_templates_seeds"], "Official A3 OOF replay")
    # Outer labels are first indexed below, after every fold and cached input has replayed.
    labels, scenarios = data["labels"], data["scenarios"]
    averaged = seed_probabilities.mean(1)
    metrics = {
        arm: probability_metrics(labels, averaged[index]) for index, arm in enumerate(ARMS)
    }
    seed_metrics = {
        arm: [probability_metrics(labels, seed_probabilities[index, seed]) for seed in range(3)]
        for index, arm in enumerate(ARMS)
    }
    anchor, primary = averaged[0], averaged[5]
    transitions = _transitions(labels, primary, anchor)
    swap = exact_scenario_swap(labels, primary, anchor, scenarios)
    statistics = protocol["statistics"]
    bootstrap = paired_scenario_bootstrap(
        labels,
        primary,
        anchor,
        scenarios,
        resamples=int(statistics["paired_scenario_bootstrap_resamples"]),
        seed=int(statistics["paired_scenario_bootstrap_seed"]),
    )
    scenario_metrics = {
        str(scenario): {
            arm: probability_metrics(
                labels[scenarios == scenario], averaged[index, scenarios == scenario]
            )
            for index, arm in enumerate(ARMS)
        }
        for scenario in np.unique(scenarios)
    }
    fold_metrics = {
        str(fold): {
            arm: probability_metrics(
                labels[data["folds"] == fold], averaged[index, data["folds"] == fold]
            )
            for index, arm in enumerate(ARMS)
        }
        for fold in range(5)
    }
    strata = {
        name: {
            arm: probability_metrics(labels[mask], averaged[index, mask])
            for index, arm in enumerate(ARMS)
        }
        for name, mask in {
            "node_support_stable": ~data["node_support_boundary"],
            "node_support_boundary": data["node_support_boundary"],
            "node_support_complete": data["node_support_complete"],
            "node_support_unknown": ~data["node_support_complete"],
        }.items()
    }
    criteria = protocol["adaptive_success_criteria"]
    gain = metrics[ARMS[5]]["macro_f1"] - metrics[ARMS[0]]["macro_f1"]
    fold_wins = sum(
        fold_metrics[str(fold)][ARMS[5]]["macro_f1"]
        > fold_metrics[str(fold)][ARMS[0]]["macro_f1"]
        for fold in range(5)
    )
    class_losses = np.asarray(metrics[ARMS[0]]["per_class_f1"]) - np.asarray(
        metrics[ARMS[5]]["per_class_f1"]
    )
    checks = {
        "primary_macro_f1_minimum": metrics[ARMS[5]]["macro_f1"]
        >= criteria["primary_macro_f1_minimum"],
        "primary_minus_exact_m4_macro_f1_minimum": gain
        >= criteria["primary_minus_exact_m4_macro_f1_minimum"],
        "net_corrected_decisions_minimum": transitions["net_corrections"]
        >= criteria["net_corrected_decisions_minimum"],
        "outer_fold_wins_minimum": fold_wins >= criteria["outer_fold_wins_minimum"],
        "nll_must_not_exceed_anchor": metrics[ARMS[5]]["nll"] <= metrics[ARMS[0]]["nll"],
        "brier_must_not_exceed_anchor": metrics[ARMS[5]]["brier"]
        <= metrics[ARMS[0]]["brier"],
        "per_class_f1_loss_cap": bool(class_losses.max() <= criteria["per_class_f1_loss_cap"]),
        "shuffled_control_must_not_exceed_primary": metrics[ARMS[6]]["macro_f1"]
        <= metrics[ARMS[5]]["macro_f1"],
        "exact_scenario_swap_one_sided_p_maximum": swap["one_sided_p_value"]
        <= criteria["exact_scenario_swap_one_sided_p_maximum"],
        "official_m4_replay_required": True,
        "official_p6_replay_required": True,
        "official_a3_replay_required": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result_dir = run / "results/v0001"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError("Stale ARFTR result artifact retained; use a fresh run path")
    result_dir.mkdir(parents=True, exist_ok=True)
    oof_path = result_dir / "oof_probabilities.npz"
    np.savez_compressed(
        oof_path,
        sample_ids=data["sample_ids"],
        labels=labels,
        scenarios=scenarios,
        folds=data["folds"],
        arms=np.asarray(ARMS),
        seeds=np.asarray(SEEDS),
        seed_probabilities=seed_probabilities,
        mean_probabilities=averaged,
        fold_parameters=np.asarray(fold_parameters),
        p6_probabilities=p6_oof,
        a3_seed_probabilities=a3_oof,
    )
    summary = {
        "status": "ARFTR_ADAPTIVE_CACHED_TRIAL_COMPLETE",
        "complete": True,
        "study_id": protocol["study_id"],
        "development_status": protocol["status"],
        "rows": rows,
        "outer_folds": 5,
        "prediction_seeds": 3,
        "cached_parameter_candidates": 1500,
        "new_neural_fits": 0,
        "primary_arm": ARMS[5],
        "execution_lock_sha256": lock_hash,
        "fold_receipt_sha256": fold_receipts,
        "oof_probabilities_sha256": file_sha256(oof_path),
        "fold_parameters": fold_parameters,
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "transitions_vs_exact_m4": transitions,
        "effects": {
            "macro_f1_gain_points": 100.0 * gain,
            "accuracy_gain_points": 100.0
            * (metrics[ARMS[5]]["accuracy"] - metrics[ARMS[0]]["accuracy"]),
            "nll_delta": metrics[ARMS[5]]["nll"] - metrics[ARMS[0]]["nll"],
            "brier_delta": metrics[ARMS[5]]["brier"] - metrics[ARMS[0]]["brier"],
            "outer_fold_wins": int(fold_wins),
        },
        "fold_metrics": fold_metrics,
        "scenario_metrics": scenario_metrics,
        "strata_metrics": strata,
        "paired_scenario_bootstrap": bootstrap,
        "exact_scenario_swap": swap,
        "adaptive_screen": {
            "checks": checks,
            "passed": bool(all(checks.values())),
            "interpretation": "Exploratory/adaptive screen only; independent confirmation remains required.",
        },
        "interpretation": protocol["interpretation"],
        "locked_input_files": len(lock["input_files"]),
    }
    immutable_json(result_dir / "summary.json", summary)
    _validate_lock(run, data)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "primary_macro_f1": metrics[ARMS[5]]["macro_f1"],
                "anchor_macro_f1": metrics[ARMS[0]]["macro_f1"],
                "gain_points": 100.0 * gain,
                "exact_scenario_swap_p": swap["one_sided_p_value"],
                "adaptive_screen_passed": summary["adaptive_screen"]["passed"],
                "summary": str(result_dir / "summary.json"),
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    run.relative_to(ROOT.resolve())
    audit(run)


if __name__ == "__main__":
    main()
