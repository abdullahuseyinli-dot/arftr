"""Run the locked cached screen for a class-diagonal protected M4 residual."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.class_diagonal_residual import (
    apply_class_diagonal_residual,
    fit_class_diagonal_residual,
    sitting_disagreement_protection,
)
from hac.matr_artifacts import (
    load_cached_study_data,
    selected_fold_artifacts,
    selected_input_paths,
)
from hac.source_swap_data import immutable_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
PROTOCOL = ROOT / "experiments/okutama_class_diagonal_residual_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/class_diagonal_residual_v1"
ARMS = ("c0_exact_m4", "c1_class_diagonal_protected_residual")
SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    optimization = protocol["optimization"]
    if (
        protocol["primary_arm"] != ARMS[1]
        or tuple(protocol["splitting"]["outer_folds"]) != tuple(range(5))
        or optimization["coefficients"] != 3
        or optimization["coefficient_minimum"] != 0.0
        or optimization["coefficient_cap"] != 0.25
        or optimization["l2"] != 0.1
        or protocol["inference_contract"]["cached_three_parameter_fits"] != 5
        or protocol["statistics"]["exact_scenario_swap_test"]["significance_alpha"] != 0.05
    ):
        raise RuntimeError("Class-diagonal protocol contract changed")


def prepare(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    protocol = _read_json(PROTOCOL)
    _validate_protocol(protocol)
    data = load_cached_study_data(SOURCE)
    # Fully validate all selected ancestries before freezing their byte hashes.
    for fold in range(5):
        selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    inputs = set(selected_input_paths(ROOT, SOURCE, SEAR, data, seeds=SEEDS))
    inputs.update(
        {
            PROTOCOL,
            Path(__file__),
            ROOT / "src/hac/class_diagonal_residual.py",
            ROOT / "src/hac/matr_artifacts.py",
        }
    )
    lock = {
        "status": "CLASS_DIAGONAL_RESIDUAL_EXECUTION_LOCKED",
        "study_id": protocol["study_id"],
        "primary_arm": protocol["primary_arm"],
        "rows": 4977,
        "outer_folds": 5,
        "base_prediction_seeds": list(SEEDS),
        "new_backbone_fits": 0,
        "new_neural_checkpoint_fits": 0,
        "cached_three_parameter_fits": 5,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "protocol_sha256": file_sha256(PROTOCOL),
        "input_files": [_file_record(path) for path in sorted(inputs, key=str)],
        "outer_metric_embargo": protocol["splitting"]["outer_metric_embargo"],
    }
    run.mkdir(parents=True, exist_ok=True)
    lock_path = run / "execution_lock.json"
    immutable_json(lock_path, lock)
    return protocol, data, file_sha256(lock_path)


def validate_lock(run: Path, data: dict[str, np.ndarray], expected_hash: str) -> dict[str, Any]:
    lock_path = run / "execution_lock.json"
    if file_sha256(lock_path) != expected_hash:
        raise RuntimeError("Class-diagonal execution lock bytes changed")
    lock = _read_json(lock_path)
    if (
        lock["status"] != "CLASS_DIAGONAL_RESIDUAL_EXECUTION_LOCKED"
        or lock["sample_ids_sha256"] != canonical_hash(data["sample_ids"].tolist())
        or lock["cached_three_parameter_fits"] != 5
    ):
        raise RuntimeError("Class-diagonal execution lock identity changed")
    for record in lock["input_files"]:
        path = (ROOT / record["path"]).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError("Locked input path escaped repository") from error
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked class-diagonal input changed: {record['path']}")
    return lock


def _fold_paths(run: Path, fold: int) -> tuple[Path, Path, Path]:
    directory = run / f"fold-{fold}"
    return directory / "predictions.npz", directory / "checkpoint.npz", directory / "receipt.json"


def fit_fold(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    lock_hash: str,
    fold: int,
) -> dict[str, Any]:
    validate_lock(run, data, lock_hash)
    prediction_path, checkpoint_path, receipt_path = _fold_paths(run, fold)
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if (
            receipt.get("status") != "CLASS_DIAGONAL_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED"
            or receipt.get("execution_lock_sha256") != lock_hash
            or not prediction_path.is_file()
            or not checkpoint_path.is_file()
            or file_sha256(prediction_path) != receipt.get("predictions_sha256")
            or file_sha256(checkpoint_path) != receipt.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"Completed class-diagonal fold {fold} is stale or changed")
        return receipt
    directory = receipt_path.parent
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Partial class-diagonal fold {fold} retained; use a fresh run path")
    directory.mkdir(parents=True, exist_ok=True)
    cached = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    train, held = cached.train_rows, cached.held_rows
    config = protocol["optimization"]
    protected_inner = sitting_disagreement_protection(
        cached.m4_inner["probabilities"],
        cached.a3_inner["probabilities"],
        sitting_class=protocol["sitting_protection"]["sitting_class"],
    )
    fit = fit_class_diagonal_residual(
        cached.m4_inner["probabilities"],
        cached.a3_inner["probabilities"],
        data["labels"][train],
        protected=protected_inner,
        coefficient_cap=config["coefficient_cap"],
        l2=config["l2"],
        epsilon=config["epsilon"],
        maximum_iterations=config["maximum_iterations"],
        tolerance=config["tolerance"],
    )
    np.savez_compressed(
        checkpoint_path,
        coefficients=fit.coefficients,
        objective=np.asarray([fit.objective]),
        training_nll=np.asarray([fit.training_nll]),
        anchor_nll=np.asarray([fit.anchor_nll]),
        iterations=np.asarray([fit.iterations], dtype=np.int64),
        function_evaluations=np.asarray([fit.function_evaluations], dtype=np.int64),
        gradient_norm=np.asarray([fit.gradient_norm]),
    )
    seed_probabilities = np.empty((2, 3, len(held), 3), dtype=np.float64)
    seed_protected = np.empty((3, len(held)), dtype=bool)
    for seed_index in range(3):
        m4 = cached.m4_outer["probabilities"][seed_index]
        a3 = cached.a3_outer["probabilities"][seed_index]
        protected = sitting_disagreement_protection(
            m4, a3, sitting_class=protocol["sitting_protection"]["sitting_class"]
        )
        candidate = apply_class_diagonal_residual(
            m4,
            a3,
            fit.coefficients,
            protected=protected,
            epsilon=config["epsilon"],
        )
        if not np.array_equal(candidate[protected], m4[protected]):
            raise RuntimeError("Outer sitting protection did not replay exact M4 bytes")
        seed_probabilities[0, seed_index] = m4
        seed_probabilities[1, seed_index] = candidate
        seed_protected[seed_index] = protected
    np.savez_compressed(
        prediction_path,
        sample_ids=data["sample_ids"][held],
        held_rows=held,
        arms=np.asarray(ARMS),
        seeds=np.asarray(SEEDS),
        seed_probabilities=seed_probabilities,
        a3_seed_probabilities=cached.a3_outer["probabilities"],
        sitting_protected=seed_protected,
    )
    receipt = {
        "status": "CLASS_DIAGONAL_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": lock_hash,
        "fold": fold,
        "outer_train_rows": int(len(train)),
        "outer_held_rows": int(len(held)),
        "outer_held_labels_read": 0,
        "three_parameter_fits": 1,
        "coefficient_count": 3,
        "coefficients": fit.coefficients.tolist(),
        "coefficient_bounds": [0.0, 0.25],
        "training_objective": fit.objective,
        "training_nll": fit.training_nll,
        "training_anchor_nll": fit.anchor_nll,
        "optimizer": {
            "success": fit.success,
            "message": fit.message,
            "iterations": fit.iterations,
            "function_evaluations": fit.function_evaluations,
            "gradient_norm": fit.gradient_norm,
        },
        "training_sitting_protected_rows": int(protected_inner.sum()),
        "outer_sitting_protected_rows_per_seed": seed_protected.sum(1).tolist(),
        "train_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selected_input_sha256": {
            _relative(path): file_sha256(path) for path in cached.input_paths
        },
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "predictions_sha256": file_sha256(prediction_path),
    }
    immutable_json(receipt_path, receipt)
    print(json.dumps({"event": "class_diagonal_fold_complete", "fold": fold, "rows": len(held)}))
    return receipt


def _transitions(labels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> dict[str, int]:
    candidate_correct = candidate.argmax(1) == labels
    anchor_correct = anchor.argmax(1) == labels
    rescues = int((candidate_correct & ~anchor_correct).sum())
    harms = int((~candidate_correct & anchor_correct).sum())
    return {
        "rescues": rescues,
        "harms": harms,
        "net_corrections": rescues - harms,
        "prediction_changes": int((candidate.argmax(1) != anchor.argmax(1)).sum()),
    }


def exact_scenario_swap(
    labels: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    scenarios: np.ndarray,
) -> dict[str, Any]:
    """Exhaust all scenario-level swaps for a directional macro-F1 test."""

    groups = np.unique(scenarios)
    group_rows = {group: np.flatnonzero(scenarios == group) for group in groups}
    observed = (
        probability_metrics(labels, candidate)["macro_f1"]
        - probability_metrics(labels, anchor)["macro_f1"]
    )
    null = np.empty(2 ** len(groups), dtype=np.float64)
    for index, assignment in enumerate(itertools.product((False, True), repeat=len(groups))):
        left, right = candidate.copy(), anchor.copy()
        for keep, group in zip(assignment, groups, strict=True):
            if not keep:
                rows = group_rows[group]
                left[rows], right[rows] = anchor[rows], candidate[rows]
        null[index] = (
            probability_metrics(labels, left)["macro_f1"]
            - probability_metrics(labels, right)["macro_f1"]
        )
    return {
        "scenario_groups": int(len(groups)),
        "assignments": int(len(null)),
        "enumeration_complete": bool(len(null) == 2 ** len(groups)),
        "alternative": "candidate macro-F1 improvement over exact M4",
        "observed_macro_f1_delta": float(observed),
        "one_sided_p_value": float(np.mean(null >= observed - 1e-15)),
        "null_minimum": float(null.min()),
        "null_maximum": float(null.max()),
        "random_sampling": False,
    }


def summarize(
    run: Path, protocol: dict[str, Any], data: dict[str, np.ndarray], lock_hash: str
) -> dict[str, Any]:
    validate_lock(run, data, lock_hash)
    rows = len(data["labels"])
    seed_probabilities = np.full((2, 3, rows, 3), np.nan, dtype=np.float64)
    a3_probabilities = np.full((3, rows, 3), np.nan, dtype=np.float64)
    protected = np.zeros((3, rows), dtype=bool)
    fold_receipts: dict[str, str] = {}
    fold_coefficients = []
    for fold in range(5):
        prediction_path, checkpoint_path, receipt_path = _fold_paths(run, fold)
        receipt = _read_json(receipt_path)
        held = np.flatnonzero(data["folds"] == fold)
        if (
            receipt.get("status") != "CLASS_DIAGONAL_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED"
            or receipt.get("outer_held_labels_read") != 0
            or receipt.get("three_parameter_fits") != 1
            or receipt.get("coefficient_count") != 3
            or file_sha256(prediction_path) != receipt.get("predictions_sha256")
            or file_sha256(checkpoint_path) != receipt.get("checkpoint_sha256")
        ):
            raise RuntimeError("Class-diagonal fold cannot pass the held-metric embargo")
        output = _read_npz(prediction_path)
        checkpoint = _read_npz(checkpoint_path)
        if (
            not np.array_equal(output["held_rows"], held)
            or not np.array_equal(output["sample_ids"], data["sample_ids"][held])
            or tuple(output["arms"].tolist()) != ARMS
            or tuple(output["seeds"].tolist()) != SEEDS
            or output["seed_probabilities"].shape != (2, 3, len(held), 3)
            or checkpoint["coefficients"].shape != (3,)
            or not np.array_equal(checkpoint["coefficients"], np.asarray(receipt["coefficients"]))
        ):
            raise RuntimeError("Class-diagonal fold identity changed")
        seed_probabilities[:, :, held] = output["seed_probabilities"]
        a3_probabilities[:, held] = output["a3_seed_probabilities"]
        protected[:, held] = output["sitting_protected"]
        fold_coefficients.append(checkpoint["coefficients"].tolist())
        fold_receipts[_relative(receipt_path)] = file_sha256(receipt_path)
    if not np.isfinite(seed_probabilities).all() or not np.isfinite(a3_probabilities).all():
        raise RuntimeError("Class-diagonal OOF coverage is incomplete")
    source_reference = _read_npz(SOURCE / "results/v0001/oof_probabilities.npz")
    sear_reference = _read_npz(SEAR / "results/v0001/oof_probabilities.npz")
    if (
        not np.array_equal(source_reference["sample_ids"], data["sample_ids"])
        or not np.array_equal(sear_reference["sample_ids"], data["sample_ids"])
        or not np.array_equal(seed_probabilities[0], source_reference["new_source_m4_seeds"])
        or not np.array_equal(a3_probabilities, sear_reference["a3_unrestricted_templates_seeds"])
    ):
        raise RuntimeError("Official M4/A3 OOF reference replay failed")
    for seed in range(3):
        if not np.array_equal(
            seed_probabilities[1, seed, protected[seed]],
            seed_probabilities[0, seed, protected[seed]],
        ):
            raise RuntimeError("Sitting-protected rows do not exactly replay M4")
    averaged = seed_probabilities.mean(1)
    labels = data["labels"]
    metrics = {arm: probability_metrics(labels, averaged[index]) for index, arm in enumerate(ARMS)}
    seed_metrics = {
        arm: [probability_metrics(labels, seed_probabilities[index, seed]) for seed in range(3)]
        for index, arm in enumerate(ARMS)
    }
    transitions = _transitions(labels, averaged[1], averaged[0])
    swap = exact_scenario_swap(labels, averaged[1], averaged[0], data["scenarios"])
    alpha = protocol["statistics"]["exact_scenario_swap_test"]["significance_alpha"]
    exact_swap_gate = {
        "alpha": alpha,
        **swap,
        "passed": bool(swap["one_sided_p_value"] <= alpha),
    }
    screen = protocol["screen_gate"]
    gain = metrics[ARMS[1]]["macro_f1"] - metrics[ARMS[0]]["macro_f1"]
    nll_delta = metrics[ARMS[1]]["nll"] - metrics[ARMS[0]]["nll"]
    checks = {
        "minimum_macro_f1": metrics[ARMS[1]]["macro_f1"] >= screen["minimum_macro_f1"],
        "minimum_gain_macro_f1_points": 100 * gain >= screen["minimum_gain_macro_f1_points"],
        "minimum_net_corrections": transitions["net_corrections"] >= screen["minimum_net_corrections"],
        "maximum_exact_scenario_swap_p_value": exact_swap_gate["passed"],
        "maximum_nll_delta": nll_delta <= screen["maximum_nll_delta"],
        "exact_m4_replay_required": np.array_equal(
            seed_probabilities[0], source_reference["new_source_m4_seeds"]
        ),
        "exact_sitting_fallback_required": all(
            np.array_equal(seed_probabilities[1, seed, protected[seed]], seed_probabilities[0, seed, protected[seed]])
            for seed in range(3)
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))
    result_dir = run / "results/v0001"
    result_dir.mkdir(parents=True, exist_ok=True)
    oof_path = result_dir / "oof_predictions.npz"
    if oof_path.exists():
        raise RuntimeError("Stale class-diagonal result artifact retained; use a fresh run path")
    np.savez_compressed(
        oof_path,
        sample_ids=data["sample_ids"],
        labels=labels,
        scenarios=data["scenarios"],
        folds=data["folds"],
        arms=np.asarray(ARMS),
        seeds=np.asarray(SEEDS),
        seed_probabilities=seed_probabilities,
        mean_probabilities=averaged,
        a3_seed_probabilities=a3_probabilities,
        sitting_protected=protected,
        fold_coefficients=np.asarray(fold_coefficients),
    )
    summary = {
        "status": "CLASS_DIAGONAL_PROTECTED_RESIDUAL_CACHED_SCREEN_COMPLETE",
        "complete": True,
        "study_id": protocol["study_id"],
        "development_status": protocol["scope"],
        "rows": rows,
        "outer_folds": 5,
        "base_prediction_seeds": 3,
        "new_backbone_fits": 0,
        "new_neural_checkpoint_fits": 0,
        "cached_three_parameter_fits": 5,
        "primary_arm": ARMS[1],
        "execution_lock_sha256": lock_hash,
        "fold_receipt_sha256": fold_receipts,
        "oof_predictions_sha256": file_sha256(oof_path),
        "fold_coefficients": fold_coefficients,
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "transitions_vs_exact_m4": transitions,
        "sitting_protection": {
            "protected_rows_per_seed": protected.sum(1).tolist(),
            "exact_m4_replay": True,
        },
        "scenario_metrics": {
            str(scenario): {
                arm: probability_metrics(
                    labels[data["scenarios"] == scenario], averaged[index, data["scenarios"] == scenario]
                )
                for index, arm in enumerate(ARMS)
            }
            for scenario in np.unique(data["scenarios"])
        },
        "exact_scenario_swap_alpha_gate": exact_swap_gate,
        "effects": {
            "macro_f1_gain_points": 100 * gain,
            "accuracy_gain_points": 100 * (metrics[ARMS[1]]["accuracy"] - metrics[ARMS[0]]["accuracy"]),
            "nll_delta": nll_delta,
            "brier_delta": metrics[ARMS[1]]["brier"] - metrics[ARMS[0]]["brier"],
        },
        "screen_gate": {
            "wording": screen["wording"],
            "checks": checks,
            "passed": passed,
            "decision": screen["pass_decision"] if passed else screen["fail_decision"],
        },
        "interpretation": protocol["interpretation"],
    }
    immutable_json(result_dir / "summary.json", summary)
    validate_lock(run, data, lock_hash)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(result_dir / "summary.json"),
                "candidate_macro_f1": metrics[ARMS[1]]["macro_f1"],
                "anchor_macro_f1": metrics[ARMS[0]]["macro_f1"],
                "gain_points": 100 * gain,
                "exact_swap_p": swap["one_sided_p_value"],
                "screen_decision": summary["screen_gate"]["decision"],
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    protocol, data, lock_hash = prepare(run)
    for fold in protocol["splitting"]["outer_folds"]:
        fit_fold(run, protocol, data, lock_hash, int(fold))
    summarize(run, protocol, data, lock_hash)


if __name__ == "__main__":
    main()
