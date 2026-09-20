"""Independent replay audit of the completed class-diagonal residual screen.

This file imports neither the runner nor any class-diagonal/artifact-loader code.
It independently re-fits five optimizations, replays all 15 outer forward passes,
and exhausts every scenario-level candidate/anchor swap.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
DEFAULT_RUN = ROOT / ".runs/research_20260912/class_diagonal_residual_v1"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260912/class_diagonal_residual_independent_audit_v1"
ARMS = ("c0_exact_m4", "c1_class_diagonal_protected_residual")
SEEDS = (42, 43, 44)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


class Evidence:
    """Bind immutable byte evidence and catch mutations during the audit."""

    def __init__(self, locked: dict[str, dict[str, Any]]):
        self.locked = locked
        self.signatures: dict[str, tuple[int, int]] = {}

    def locked_path(self, path: Path) -> Path:
        path = path.resolve()
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        require(relative in self.locked, f"Input is absent from execution lock: {relative}")
        record = self.locked[relative]
        require(path.is_file(), f"Missing locked input: {relative}")
        stat = path.stat()
        require(stat.st_size == record["size_bytes"], f"Locked input size changed: {relative}")
        require(sha256(path) == record["sha256"], f"Locked input hash changed: {relative}")
        self.signatures[str(path)] = (stat.st_size, stat.st_mtime_ns)
        return path

    def output_path(self, path: Path, expected_hash: str) -> Path:
        path = path.resolve()
        require(path.is_file() and sha256(path) == expected_hash, f"Output hash changed: {path}")
        stat = path.stat()
        self.signatures[str(path)] = (stat.st_size, stat.st_mtime_ns)
        return path

    def finish(self) -> None:
        for name, expected in self.signatures.items():
            stat = Path(name).stat()
            require((stat.st_size, stat.st_mtime_ns) == expected, f"Evidence mutated: {name}")


def probabilities(values: np.ndarray, rows: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (rows, 3), f"Probability shape: {name}")
    require(np.isfinite(values).all() and (values >= 0).all(), f"Probability values: {name}")
    require(np.allclose(values.sum(1), 1, atol=1e-6, rtol=0), f"Probability sum: {name}")
    return values


def protection(m4: np.ndarray, a3: np.ndarray) -> np.ndarray:
    left, right = m4.argmax(1), a3.argmax(1)
    return (left != right) & ((left == 0) | (right == 0))


def forward(m4: np.ndarray, a3: np.ndarray, alpha: np.ndarray, protected: np.ndarray) -> np.ndarray:
    output = m4.copy()
    active = ~protected
    log_anchor = np.log(np.clip(m4[active], 1e-12, 1))
    logits = log_anchor + alpha * (np.log(np.clip(a3[active], 1e-12, 1)) - log_anchor)
    logits -= logits.max(1, keepdims=True)
    normalized = np.exp(logits)
    output[active] = normalized / normalized.sum(1, keepdims=True)
    require(np.array_equal(output[protected], m4[protected]), "Independent sitting fallback failed")
    return output


def independent_fit(m4: np.ndarray, a3: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    protected = protection(m4, a3)
    log_anchor = np.log(np.clip(m4, 1e-12, 1))
    residual = np.log(np.clip(a3, 1e-12, 1)) - log_anchor
    target = np.eye(3)[labels]

    def value(alpha):
        logits = log_anchor.copy()
        logits[~protected] += residual[~protected] * alpha
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        nll = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        gradient = ((p - target) * residual * (~protected)[:, None]).mean(0) + 0.2 * alpha
        return float(nll + 0.1 * np.square(alpha).sum()), gradient

    result = minimize(
        value,
        np.zeros(3),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, 0.25),) * 3,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-12},
    )
    require(result.success, f"Independent optimizer failed: {result.message}")
    return np.asarray(result.x), float(result.fun)


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def metrics(labels: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    probabilities(p, len(labels), "audit metrics")
    matrix = confusion(labels, p.argmax(1))
    denominator = matrix.sum(0) + matrix.sum(1)
    per_class = np.divide(
        2 * matrix.diagonal(), denominator, out=np.zeros(3, dtype=float), where=denominator > 0
    )
    return {
        "rows": len(labels),
        "macro_f1": float(per_class.mean()),
        "accuracy": float(matrix.trace() / len(labels)),
        "nll": float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier": float(np.square(p - np.eye(3)[labels]).sum(1).mean()),
        "per_class_f1": per_class.tolist(),
        "confusion": matrix.tolist(),
    }


def same(expected: Any, actual: Any, name: str, tolerance: float = 1e-10) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Expected mapping: {name}")
        for key, value in expected.items():
            require(key in actual, f"Missing summary field: {name}.{key}")
            same(value, actual[key], f"{name}.{key}", tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"List mismatch: {name}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            same(left, right, f"{name}[{index}]", tolerance)
    elif isinstance(expected, float):
        require(abs(expected - actual) <= tolerance, f"Numeric mismatch: {name}")
    else:
        require(expected == actual, f"Value mismatch: {name}")


def load_inner(
    fold: int,
    data: dict[str, np.ndarray],
    evidence: Evidence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    train = np.flatnonzero(data["folds"] != fold)
    source_fold = SOURCE / f"models/survival_memory/fold-{fold}"
    selection_path = evidence.locked_path(source_fold / "selection.json")
    evidence.locked_path(source_fold / "completion_manifest.json")
    selected = read_json(selection_path)["selected"]
    m4 = np.full((len(data["labels"]), 3), np.nan)
    covered = []
    paths = []
    for inner in range(3):
        directory = source_fold / f"config-{selected['config_index']}/inner-{inner}"
        receipt_path = evidence.locked_path(directory / "receipt.json")
        prediction_path = evidence.locked_path(directory / "predictions.npz")
        receipt, output = read_json(receipt_path), read_npz(prediction_path)
        rows = np.asarray(receipt["held_rows"], dtype=np.int64)
        request = receipt["request"]
        require(request["learning_rate"] == selected["learning_rate"], "M4 selected LR changed")
        require(request["weight_decay"] == selected["weight_decay"], "M4 selected WD changed")
        require(request["selection_labels_sha256"] == canonical(data["labels"][rows].tolist()), "M4 selection labels changed")
        require(receipt["output_sha256"]["predictions.npz"] == sha256(prediction_path), "M4 prediction receipt changed")
        require(np.array_equal(output["held_rows"], rows), "M4 inner rows changed")
        require(np.array_equal(output["sample_ids"], data["sample_ids"][rows]), "M4 inner IDs changed")
        m4[rows] = probabilities(output["probabilities"], len(rows), "M4 inner")
        covered.append(rows)
        paths.extend((receipt_path, prediction_path))
    require(np.array_equal(np.sort(np.concatenate(covered)), train), "M4 inner OOF coverage")

    workload_path = evidence.locked_path(
        SEAR / f"workloads/a3_unrestricted_templates/fold-{fold}/receipt.json"
    )
    workload = read_json(workload_path)
    selected_a3 = workload["selected"]
    a3 = np.full((len(data["labels"]), 3), np.nan)
    covered = []
    selected_receipts = []
    index = {sample_id: row for row, sample_id in enumerate(data["sample_ids"].tolist())}
    for reference in workload["fit_receipts"]:
        receipt_path = Path(reference["path"]).resolve()
        request = read_json(receipt_path)["request"]
        if not (
            request["stage"] == "inner"
            and request["learning_rate"] == selected_a3["learning_rate"]
            and request["weight_decay"] == selected_a3["weight_decay"]
        ):
            continue
        receipt_path = evidence.locked_path(receipt_path)
        receipt = read_json(receipt_path)
        record = receipt["artifacts"]["predictions.npz"]
        prediction_path = evidence.locked_path(Path(record["path"]))
        require(reference["sha256"] == sha256(receipt_path), "A3 selected receipt hash changed")
        require(record["sha256"] == sha256(prediction_path), "A3 selected output hash changed")
        output = read_npz(prediction_path)
        rows = np.asarray([index[value] for value in output["sample_ids"].tolist()])
        require(np.all(data["folds"][rows] != fold), "A3 inner leaks outer fold")
        require(request["selection_labels_sha256"] == canonical(data["labels"][rows].tolist()), "A3 selection labels changed")
        a3[rows] = probabilities(output["probabilities"], len(rows), "A3 inner")
        covered.append(rows)
        selected_receipts.append(request["inner_fold"])
        paths.extend((receipt_path, prediction_path))
    require(sorted(selected_receipts) == [0, 1, 2], "A3 selected inner fits changed")
    require(np.array_equal(np.sort(np.concatenate(covered)), train), "A3 inner OOF coverage")
    return m4[train], a3[train], train, paths


def load_outer(
    fold: int,
    data: dict[str, np.ndarray],
    evidence: Evidence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    held = np.flatnonzero(data["folds"] == fold)
    source_fold = SOURCE / f"models/survival_memory/fold-{fold}"
    selected = read_json(evidence.locked_path(source_fold / "selection.json"))["selected"]
    m4 = []
    for seed in SEEDS:
        directory = source_fold / f"refit-seed-{seed}"
        receipt_path = evidence.locked_path(directory / "receipt.json")
        prediction_path = evidence.locked_path(directory / "predictions.npz")
        receipt, output = read_json(receipt_path), read_npz(prediction_path)
        require(receipt["request"]["seed"] == seed, "M4 outer seed order changed")
        require(receipt["request"]["learning_rate"] == selected["learning_rate"], "M4 refit LR")
        require(receipt["request"]["weight_decay"] == selected["weight_decay"], "M4 refit WD")
        require(receipt["request"]["selection_labels_sha256"] is None, "M4 held labels reached refit")
        require(np.array_equal(output["held_rows"], held), "M4 outer rows changed")
        require(np.array_equal(output["sample_ids"], data["sample_ids"][held]), "M4 outer IDs changed")
        require(receipt["output_sha256"]["predictions.npz"] == sha256(prediction_path), "M4 outer receipt")
        m4.append(probabilities(output["probabilities"], len(held), "M4 outer"))
    workload_path = evidence.locked_path(
        SEAR / f"workloads/a3_unrestricted_templates/fold-{fold}/receipt.json"
    )
    workload = read_json(workload_path)
    prediction_path = evidence.locked_path(
        SEAR / f"workloads/a3_unrestricted_templates/fold-{fold}/predictions.npz"
    )
    require(workload["artifact"]["sha256"] == sha256(prediction_path), "A3 outer receipt")
    output = read_npz(prediction_path)
    require(np.array_equal(output["sample_ids"], data["sample_ids"][held]), "A3 outer IDs changed")
    a3 = output["seed_probabilities"]
    require(a3.shape == (3, len(held), 3), "A3 outer seed shape")
    for index in range(3):
        probabilities(a3[index], len(held), "A3 outer")
    return np.stack(m4), a3.astype(np.float64), held


def audit(run: Path, output: Path) -> dict[str, Any]:
    run, output = run.resolve(), output.resolve()
    require(run != output, "Audit output must differ from execution directory")
    require(not output.exists(), "Audit output already exists; use a fresh path")
    lock_path = run / "execution_lock.json"
    lock = read_json(lock_path)
    require(lock["status"] == "CLASS_DIAGONAL_RESIDUAL_EXECUTION_LOCKED", "Execution lock status")
    locked = {record["path"]: record for record in lock["input_files"]}
    evidence = Evidence(locked)
    evidence.signatures[str(lock_path.resolve())] = (
        lock_path.stat().st_size,
        lock_path.stat().st_mtime_ns,
    )
    protocol_path = evidence.locked_path(ROOT / "experiments/okutama_class_diagonal_residual_protocol.json")
    protocol = read_json(protocol_path)
    require(sha256(protocol_path) == lock["protocol_sha256"], "Protocol hash lock")
    require(protocol["optimization"]["coefficients"] == 3, "Protocol coefficient count")
    require(protocol["optimization"]["coefficient_cap"] == 0.25, "Protocol coefficient cap")
    require(protocol["optimization"]["l2"] == 0.1, "Protocol L2")
    require(protocol["statistics"]["exact_scenario_swap_test"]["significance_alpha"] == 0.05, "Protocol swap alpha")
    data_path = evidence.locked_path(SOURCE / "data/memory_data.npz")
    data = read_npz(data_path)
    require(len(data["labels"]) == 4977, "Canonical row count")
    require(canonical(data["sample_ids"].tolist()) == lock["sample_ids_sha256"], "Sample identity")
    require(np.array_equal(np.unique(data["folds"]), np.arange(5)), "Outer folds")

    replayed = np.full((2, 3, 4977, 3), np.nan)
    replayed_a3 = np.full((3, 4977, 3), np.nan)
    replayed_protected = np.zeros((3, 4977), dtype=bool)
    independent_coefficients = []
    fold_records = []
    for fold in range(5):
        m4_inner, a3_inner, train, _ = load_inner(fold, data, evidence)
        alpha, objective = independent_fit(m4_inner, a3_inner, data["labels"][train])
        prediction_path = run / f"fold-{fold}/predictions.npz"
        checkpoint_path = run / f"fold-{fold}/checkpoint.npz"
        receipt_path = run / f"fold-{fold}/receipt.json"
        receipt = read_json(receipt_path)
        require(receipt["status"] == "CLASS_DIAGONAL_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED", "Fold status")
        require(receipt["outer_held_labels_read"] == 0, "Held labels read before embargo release")
        require(receipt["three_parameter_fits"] == 1 and receipt["coefficient_count"] == 3, "Fit count")
        evidence.output_path(receipt_path, sha256(receipt_path))
        evidence.output_path(prediction_path, receipt["predictions_sha256"])
        evidence.output_path(checkpoint_path, receipt["checkpoint_sha256"])
        stored, checkpoint = read_npz(prediction_path), read_npz(checkpoint_path)
        require(np.allclose(alpha, checkpoint["coefficients"], atol=1e-10, rtol=0), "Coefficient replay")
        require(abs(objective - float(checkpoint["objective"][0])) <= 1e-10, "Objective replay")
        m4_outer, a3_outer, held = load_outer(fold, data, evidence)
        require(np.array_equal(stored["sample_ids"], data["sample_ids"][held]), "Stored fold IDs")
        require(np.array_equal(stored["held_rows"], held), "Stored fold rows")
        require(tuple(stored["arms"].tolist()) == ARMS, "Stored arms")
        require(tuple(stored["seeds"].tolist()) == SEEDS, "Stored seeds")
        for seed in range(3):
            mask = protection(m4_outer[seed], a3_outer[seed])
            candidate = forward(m4_outer[seed], a3_outer[seed], alpha, mask)
            require(np.array_equal(stored["seed_probabilities"][0, seed], m4_outer[seed]), "M4 fold replay")
            require(np.allclose(stored["seed_probabilities"][1, seed], candidate, atol=1e-14, rtol=0), "Candidate fold replay")
            require(np.array_equal(stored["a3_seed_probabilities"][seed], a3_outer[seed]), "A3 fold replay")
            require(np.array_equal(stored["sitting_protected"][seed], mask), "Sitting mask replay")
            replayed[0, seed, held] = m4_outer[seed]
            replayed[1, seed, held] = candidate
            replayed_a3[seed, held] = a3_outer[seed]
            replayed_protected[seed, held] = mask
        independent_coefficients.append(alpha.tolist())
        fold_records.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "held_rows": len(held),
                "coefficients": alpha.tolist(),
                "objective": objective,
                "outer_forward_passes_replayed": 3,
            }
        )
    require(np.isfinite(replayed).all(), "Replayed OOF coverage")
    source_reference = read_npz(evidence.locked_path(SOURCE / "results/v0001/oof_probabilities.npz"))
    sear_reference = read_npz(evidence.locked_path(SEAR / "results/v0001/oof_probabilities.npz"))
    require(np.array_equal(replayed[0], source_reference["new_source_m4_seeds"]), "Official M4 replay")
    require(np.array_equal(replayed_a3, sear_reference["a3_unrestricted_templates_seeds"]), "Official A3 replay")

    result_path = run / "results/v0001/oof_predictions.npz"
    summary_path = run / "results/v0001/summary.json"
    summary = read_json(summary_path)
    evidence.output_path(result_path, summary["oof_predictions_sha256"])
    evidence.output_path(summary_path, sha256(summary_path))
    for fold in range(5):
        relative = f".runs/research_20260912/class_diagonal_residual_v1/fold-{fold}/receipt.json"
        # Custom --run paths retain absolute identity through the observed path.
        receipt_path = run / f"fold-{fold}/receipt.json"
        matched = summary["fold_receipt_sha256"].get(relative)
        if matched is None:
            matched = next(
                (
                    value
                    for name, value in summary["fold_receipt_sha256"].items()
                    if name.endswith(f"/fold-{fold}/receipt.json")
                ),
                None,
            )
        require(matched == sha256(receipt_path), "Fold receipt summary binding")
    result = read_npz(result_path)
    require(np.allclose(result["seed_probabilities"], replayed, atol=1e-14, rtol=0), "Published OOF replay")
    require(np.array_equal(result["sitting_protected"], replayed_protected), "Published protection replay")
    require(np.allclose(result["fold_coefficients"], independent_coefficients, atol=1e-10, rtol=0), "Published coefficients")
    averaged = replayed.mean(1)
    labels = data["labels"]
    computed_metrics = {arm: metrics(labels, averaged[index]) for index, arm in enumerate(ARMS)}
    same(computed_metrics, summary["metrics"], "metrics")

    groups = np.unique(data["scenarios"])
    group_rows = [np.flatnonzero(data["scenarios"] == group) for group in groups]
    observed = computed_metrics[ARMS[1]]["macro_f1"] - computed_metrics[ARMS[0]]["macro_f1"]
    null = []
    for assignment in itertools.product((0, 1), repeat=len(groups)):
        candidate = averaged[1].copy()
        anchor = averaged[0].copy()
        for swap, rows in zip(assignment, group_rows, strict=True):
            if swap:
                candidate[rows], anchor[rows] = averaged[0, rows], averaged[1, rows]
        null.append(metrics(labels, candidate)["macro_f1"] - metrics(labels, anchor)["macro_f1"])
    null = np.asarray(null)
    swap = {
        "scenario_groups": len(groups),
        "assignments": len(null),
        "enumeration_complete": len(null) == 2 ** len(groups),
        "alternative": "candidate macro-F1 improvement over exact M4",
        "observed_macro_f1_delta": float(observed),
        "one_sided_p_value": float(np.mean(null >= observed - 1e-15)),
        "null_minimum": float(null.min()),
        "null_maximum": float(null.max()),
        "random_sampling": False,
    }
    same(swap, {key: summary["exact_scenario_swap_alpha_gate"][key] for key in swap}, "exact swap")
    require(summary["exact_scenario_swap_alpha_gate"]["alpha"] == 0.05, "Swap alpha changed")
    require(
        summary["exact_scenario_swap_alpha_gate"]["passed"]
        == (swap["one_sided_p_value"] <= 0.05),
        "Exact swap alpha gate",
    )
    evidence.finish()
    audit_summary = {
        "status": "CLASS_DIAGONAL_RESIDUAL_INDEPENDENT_AUDIT_PASS",
        "complete": True,
        "execution_run": str(run),
        "execution_lock_sha256": sha256(lock_path),
        "execution_summary_sha256": sha256(summary_path),
        "rows": 4977,
        "outer_folds": 5,
        "three_parameter_fits_independently_replayed": 5,
        "outer_seed_forward_passes_independently_replayed": 15,
        "official_m4_seed_outputs_bit_exact": True,
        "official_a3_seed_outputs_bit_exact": True,
        "protected_sitting_fallback_bit_exact": True,
        "independent_coefficients": independent_coefficients,
        "folds": fold_records,
        "metrics": computed_metrics,
        "exact_scenario_swap": swap,
        "exact_scenario_swap_alpha": 0.05,
        "exact_scenario_swap_alpha_gate_passed": bool(swap["one_sided_p_value"] <= 0.05),
        "screen_gate_decision": summary["screen_gate"]["decision"],
        "limitations": "Adaptive cached-screen evidence on a previously inspected cohort; this audit verifies computation and ancestry, not external generalization.",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    output.mkdir(parents=True)
    output_path = output / "summary.json"
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(audit_summary, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": audit_summary["status"],
                "output": str(output_path),
                "macro_f1": computed_metrics[ARMS[1]]["macro_f1"],
                "gain_points": 100 * observed,
                "exact_swap_p": swap["one_sided_p_value"],
            },
            indent=2,
        )
    )
    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit(args.run, args.output)


if __name__ == "__main__":
    main()
