"""Independent checkpoint/output audit for a completed FSAR v2 cached screen.

The auditor imports no HAC implementation or runner module. It verifies the
locked byte inventory, all 30 semantic fit identities, replays every checkpoint
on held center features, and independently recomputes metrics and exact swaps.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".runs/research_20260912/frame_supervised_anchor_residual_v2"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260912/frame_supervised_anchor_residual_v2_audit"
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
FRAME_DATA = ROOT / ".runs/research_20260912/cached_frame_supervision_v2/data"
ARMS = (
    "f0_exact_m4",
    "f1_fixed_geometric_m4_a3",
    "f2_center_only_supervision",
    "f3_all_cached_frame_supervision",
    "f4_center_only_standalone",
    "f5_all_cached_frame_standalone",
)
TRAINED = ARMS[2:4]
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
        return {name: saved[name] for name in saved.files}


def probability(values: np.ndarray, rows: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (rows, 3), f"Probability shape changed: {name}")
    require(np.isfinite(values).all() and (values >= 0).all(), f"Probability values: {name}")
    require(np.allclose(values.sum(1), 1, atol=1e-6, rtol=0), f"Probability sum: {name}")
    return values


def geometric(m4: np.ndarray, a3: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(m4, 1e-12, 1))
    logits += 0.10 * (np.log(np.clip(a3, 1e-12, 1)) - logits)
    logits -= logits.max(1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(1, keepdims=True)


def anchored(anchor: np.ndarray, logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.mean(1, keepdims=True)
    values = np.log(np.clip(anchor, 1e-12, 1)) + 0.25 * centered
    values -= values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


class IndependentHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(), nn.Dropout(0.1)
        )
        self.local_classifier = nn.Linear(128, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.local_classifier(self.encoder(values))


def load_checkpoint(path: Path) -> IndependentHead:
    saved = read_npz(path)
    expected = {
        "encoder__0__weight": (768,),
        "encoder__0__bias": (768,),
        "encoder__1__weight": (128, 768),
        "encoder__1__bias": (128,),
        "local_classifier__weight": (3, 128),
        "local_classifier__bias": (3,),
    }
    require(set(saved) == set(expected), "Checkpoint field inventory changed")
    require(
        all(saved[name].shape == shape and np.isfinite(saved[name]).all() for name, shape in expected.items()),
        "Checkpoint shape or values changed",
    )
    model = IndependentHead()
    state = {name.replace("__", "."): torch.as_tensor(value) for name, value in saved.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def metrics(labels: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = probability(p, len(labels), "metrics")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels, p.argmax(1)), 1)
    denominator = matrix.sum(0) + matrix.sum(1)
    scores = np.divide(
        2 * matrix.diagonal(), denominator, out=np.zeros(3, dtype=float), where=denominator > 0
    )
    return {
        "rows": len(labels),
        "macro_f1": float(scores.mean()),
        "accuracy": float(matrix.trace() / len(labels)),
        "nll": float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier": float(np.square(p - np.eye(3)[labels]).sum(1).mean()),
        "per_class_f1": scores.tolist(),
        "confusion": matrix.tolist(),
    }


def scenario_statistics(
    labels, candidate, reference, scenarios, *, resamples: int, seed: int
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    rows = [np.flatnonzero(scenarios == group) for group in groups]
    observed = metrics(labels, candidate)["macro_f1"] - metrics(labels, reference)["macro_f1"]
    values = []
    for assignment in itertools.product((False, True), repeat=len(groups)):
        left, right = candidate.copy(), reference.copy()
        for swap, selected in zip(assignment, rows, strict=True):
            if swap:
                left[selected], right[selected] = reference[selected], candidate[selected]
        values.append(metrics(labels, left)["macro_f1"] - metrics(labels, right)["macro_f1"])
    rng = np.random.default_rng(seed)
    deltas = np.empty((resamples, 3), dtype=np.float64)
    group_map = {group: selected for group, selected in zip(groups, rows, strict=True)}
    for index in range(resamples):
        sampled = rng.choice(groups, len(groups), replace=True)
        selected = np.concatenate([group_map[group] for group in sampled])
        candidate_metrics = metrics(labels[selected], candidate[selected])
        reference_metrics = metrics(labels[selected], reference[selected])
        deltas[index] = (
            candidate_metrics["macro_f1"] - reference_metrics["macro_f1"],
            candidate_metrics["nll"] - reference_metrics["nll"],
            candidate_metrics["brier"] - reference_metrics["brier"],
        )
    return {
        "observed_macro_f1_delta": float(observed),
        "scenario_swap_assignments": len(values),
        "scenario_swap_one_sided_p": float(np.mean(np.asarray(values) >= observed - 1e-15)),
        "macro_f1_delta_ci95": np.quantile(deltas[:, 0], [0.025, 0.975]).tolist(),
        "nll_delta_ci95": np.quantile(deltas[:, 1], [0.025, 0.975]).tolist(),
        "nll_delta_one_sided_upper95": float(np.quantile(deltas[:, 1], 0.95)),
        "brier_delta_ci95": np.quantile(deltas[:, 2], [0.025, 0.975]).tolist(),
        "brier_delta_one_sided_upper95": float(np.quantile(deltas[:, 2], 0.95)),
    }


def same(expected: Any, actual: Any, name: str, tolerance: float = 1e-9) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Mapping changed: {name}")
        for key, value in expected.items():
            require(key in actual, f"Missing summary field: {name}.{key}")
            same(value, actual[key], f"{name}.{key}", tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"List changed: {name}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            same(left, right, f"{name}[{index}]", tolerance)
    elif isinstance(expected, float):
        require(abs(expected - actual) <= tolerance, f"Numeric mismatch: {name}")
    else:
        require(expected == actual, f"Value mismatch: {name}")


def audit(run: Path, output: Path) -> dict[str, Any]:
    run, output = run.resolve(), output.resolve()
    require(not output.exists(), "Use a fresh independent-audit output path")
    lock_path = run / "execution_lock.json"
    lock = read_json(lock_path)
    require(lock["status"] == "FSAR_EXECUTION_LOCKED_BEFORE_MODEL_FITTING", "Lock status")
    require(Path(lock["run_directory"]).resolve() == run, "Run identity changed")
    require(lock["model_fits"] == 30, "Locked fit count changed")
    signatures = {}
    locked = {}
    for record in lock["input_files"]:
        path = (ROOT / record["path"]).resolve()
        require(path.is_file() and path.stat().st_size == record["size_bytes"], "Input size changed")
        require(sha256(path) == record["sha256"], f"Input hash changed: {record['path']}")
        signatures[str(path)] = (path.stat().st_size, path.stat().st_mtime_ns)
        locked[record["path"]] = record
    require(
        "experiments/audit_okutama_frame_supervision.py" in locked,
        "Auditor itself was not frozen before fitting",
    )
    data = read_npz(SOURCE / "data/memory_data.npz")
    frames = read_npz(FRAME_DATA / "frame_supervision.npz")
    require(canonical(data["sample_ids"].tolist()) == lock["sample_ids_sha256"], "Sample lock")
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        require(np.array_equal(data[name], frames[name]), f"Frame identity changed: {name}")
    source = read_npz(SOURCE / "results/v0001/oof_probabilities.npz")
    sear = read_npz(SEAR / "results/v0001/oof_probabilities.npz")
    require(np.array_equal(source["sample_ids"], data["sample_ids"]), "M4 sample order")
    require(np.array_equal(sear["sample_ids"], data["sample_ids"]), "A3 sample order")
    protocol = read_json(ROOT / "experiments/okutama_cached_frame_supervision_protocol.json")
    m4, a3 = source["new_source_m4_seeds"], sear["a3_unrestricted_templates_seeds"]
    fixed = np.stack([geometric(m4[index], a3[index]) for index in range(3)])
    seeds = np.full((len(ARMS), 3, len(data["labels"]), 3), np.nan)
    seeds[0], seeds[1] = m4, fixed
    short_features = np.load(
        ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/dinov2_native_frames.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    fit_receipts = {}
    for arm_index, arm in enumerate(TRAINED, start=2):
        for fold in range(5):
            train = np.flatnonzero(data["folds"] != fold)
            held = np.flatnonzero(data["folds"] == fold)
            auxiliary = int(frames["short_valid"][train].sum() + frames["long_valid"][train].sum())
            expected_steps = math.ceil(auxiliary / 2048) * 8
            centers = short_features[held, 8, 0].astype(np.float32)
            for seed_index, seed in enumerate(SEEDS):
                directory = run / f"models/{arm}/fold-{fold}/seed-{seed}"
                receipt_path = directory / "receipt.json"
                receipt = read_json(receipt_path)
                require(
                    receipt["status"] == "FSAR_FIT_COMPLETE_OUTER_METRICS_EMBARGOED"
                    and receipt["execution_lock_sha256"] == sha256(lock_path)
                    and receipt["arm"] == arm
                    and receipt["fold"] == fold
                    and receipt["seed"] == seed
                    and receipt["outer_held_labels_read"] == 0
                    and receipt["optimizer_steps"] == expected_steps
                    and receipt["train_sample_ids_sha256"] == canonical(data["sample_ids"][train].tolist())
                    and receipt["train_labels_sha256"] == canonical(data["labels"][train].tolist())
                    and receipt["held_sample_ids_sha256"] == canonical(data["sample_ids"][held].tolist()),
                    "Fit semantic identity changed",
                )
                prediction_path, checkpoint_path = directory / "predictions.npz", directory / "checkpoint.npz"
                require(sha256(prediction_path) == receipt["predictions_sha256"], "Prediction hash")
                require(sha256(checkpoint_path) == receipt["checkpoint_sha256"], "Checkpoint hash")
                stored = read_npz(prediction_path)
                require(np.array_equal(stored["sample_ids"], data["sample_ids"][held]), "Held IDs")
                expected_anchor = fixed[seed_index, held]
                require(np.array_equal(stored["anchor_probabilities"], expected_anchor), "Anchor replay")
                model = load_checkpoint(checkpoint_path)
                with torch.inference_mode():
                    logits = model(torch.as_tensor(centers)).numpy().astype(np.float64)
                require(np.allclose(logits, stored["local_logits"], atol=2e-5, rtol=1e-6), "Checkpoint forward replay")
                standalone = torch.softmax(torch.as_tensor(stored["local_logits"]), 1).numpy()
                predicted = anchored(expected_anchor, stored["local_logits"])
                require(np.allclose(standalone, stored["standalone_probabilities"], atol=1e-14, rtol=0), "Standalone replay")
                require(np.allclose(predicted, stored["probabilities"], atol=1e-14, rtol=0), "Residual replay")
                seeds[arm_index, seed_index, held] = stored["probabilities"]
                seeds[arm_index + 2, seed_index, held] = stored["standalone_probabilities"]
                fit_receipts[str(receipt_path.relative_to(ROOT)).replace("\\", "/")] = sha256(receipt_path)
    require(len(fit_receipts) == 30 and np.isfinite(seeds).all(), "Exact 30-fit OOF inventory")
    summary_path = run / "results/v0001/summary.json"
    summary = read_json(summary_path)
    require(summary["fit_receipt_sha256"] == fit_receipts, "Published fit receipt inventory")
    averaged = seeds.mean(1)
    oof_path = run / "results/v0001/oof_predictions.npz"
    require(sha256(oof_path) == summary["oof_predictions_sha256"], "Published OOF hash")
    oof = read_npz(oof_path)
    require(np.array_equal(oof["sample_ids"], data["sample_ids"]), "Published OOF IDs")
    require(np.array_equal(oof["arms"], np.asarray(ARMS)), "Published OOF arms")
    require(np.allclose(oof["seed_probabilities"], seeds, atol=1e-14, rtol=0), "Published OOF predictions")
    computed = {arm: metrics(data["labels"], averaged[index]) for index, arm in enumerate(ARMS)}
    same(computed, summary["metrics"], "metrics")
    seed_metrics = {
        arm: [metrics(data["labels"], seeds[index, seed]) for seed in range(3)]
        for index, arm in enumerate(ARMS)
    }
    same(seed_metrics, summary["seed_metrics"], "seed_metrics")
    replayed_statistics = {}
    for reference_index, reference in enumerate(("f0_exact_m4", "f2_center_only_supervision")):
        observed = scenario_statistics(
            data["labels"],
            averaged[3],
            averaged[ARMS.index(reference)],
            data["scenarios"],
            resamples=protocol["statistics"]["scenario_bootstrap_resamples"],
            seed=protocol["statistics"]["seed"] + reference_index,
        )
        published = summary["primary_statistics"][reference]
        same(observed, published, f"statistics.{reference}")
        require(observed["scenario_swap_assignments"] == 2048, "Exact swap count")
        replayed_statistics[reference] = observed
    transitions = {}
    anchor_correct = averaged[0].argmax(1) == data["labels"]
    for index, arm in enumerate(ARMS[1:], start=1):
        candidate_correct = averaged[index].argmax(1) == data["labels"]
        transitions[arm] = {
            "rescues": int((candidate_correct & ~anchor_correct).sum()),
            "harms": int((~candidate_correct & anchor_correct).sum()),
            "net_corrections": int(candidate_correct.sum() - anchor_correct.sum()),
            "prediction_changes": int((averaged[index].argmax(1) != averaged[0].argmax(1)).sum()),
        }
    same(transitions, summary["transitions_vs_m4"], "transitions")
    primary, anchor, control = computed[ARMS[3]], computed[ARMS[0]], computed[ARMS[2]]
    scenario_deltas = []
    for scenario in np.unique(data["scenarios"]):
        selected = data["scenarios"] == scenario
        scenario_deltas.append(
            metrics(data["labels"][selected], averaged[3, selected])["accuracy"]
            - metrics(data["labels"][selected], averaged[0, selected])["accuracy"]
        )
    gates = protocol["screen_gates"]
    class_delta = np.asarray(primary["per_class_f1"]) - np.asarray(anchor["per_class_f1"])
    checks = {
        "minimum_primary_macro_f1": primary["macro_f1"] >= gates["minimum_primary_macro_f1"],
        "minimum_gain_over_m4_points": 100 * (primary["macro_f1"] - anchor["macro_f1"])
        >= gates["minimum_gain_over_m4_points"],
        "minimum_gain_over_center_control_points": 100
        * (primary["macro_f1"] - control["macro_f1"])
        >= gates["minimum_gain_over_center_control_points"],
        "minimum_net_corrections_over_m4": transitions[ARMS[3]]["net_corrections"]
        >= gates["minimum_net_corrections_over_m4"],
        "scenario_swap_vs_m4_one_sided_p_maximum": replayed_statistics[ARMS[0]][
            "scenario_swap_one_sided_p"
        ]
        <= gates["scenario_swap_vs_m4_one_sided_p_maximum"],
        "scenario_swap_vs_center_control_one_sided_p_maximum": replayed_statistics[ARMS[2]][
            "scenario_swap_one_sided_p"
        ]
        <= gates["scenario_swap_vs_center_control_one_sided_p_maximum"],
        "maximum_nll_delta_over_m4_one_sided_upper95": replayed_statistics[ARMS[0]][
            "nll_delta_one_sided_upper95"
        ]
        <= gates["maximum_nll_delta_over_m4_one_sided_upper95"],
        "maximum_brier_delta_over_m4_one_sided_upper95": replayed_statistics[ARMS[0]][
            "brier_delta_one_sided_upper95"
        ]
        <= gates["maximum_brier_delta_over_m4_one_sided_upper95"],
        "maximum_worst_class_f1_loss_points": 100 * class_delta.min()
        >= -gates["maximum_worst_class_f1_loss_points"],
        "minimum_improved_scenarios": sum(delta > 0 for delta in scenario_deltas)
        >= gates["minimum_improved_scenarios"],
        "maximum_scenario_accuracy_loss_points": 100 * min(0.0, *scenario_deltas)
        >= -gates["maximum_scenario_accuracy_loss_points"],
        "maximum_seed_macro_f1_sd_points": 100
        * np.std([value["macro_f1"] for value in seed_metrics[ARMS[3]]])
        <= gates["maximum_seed_macro_f1_sd_points"],
        "exact_m4_replay": bool(np.array_equal(seeds[0], m4)),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    require(checks == summary["screen_checks"], "Published screen checks changed")
    require(bool(all(checks.values())) == summary["screen_gate_passed"], "Screen decision changed")
    for name, expected in signatures.items():
        stat = Path(name).stat()
        require((stat.st_size, stat.st_mtime_ns) == expected, f"Input mutated during audit: {name}")
    result = {
        "status": "FSAR_V2_INDEPENDENT_30_FIT_AUDIT_PASS",
        "complete": True,
        "execution_run": str(run),
        "execution_lock_sha256": sha256(lock_path),
        "execution_summary_sha256": sha256(summary_path),
        "fits_verified": 30,
        "checkpoint_forward_passes_replayed": 30,
        "outer_held_labels_read_before_complete_inventory": 0,
        "metrics": computed,
        "screen_gate_passed": summary["screen_gate_passed"],
        "limitations": "Adaptive repeated-development cohort; computation and ancestry audit is not external confirmation.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    }
    output.mkdir(parents=True)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "output": str(output / "summary.json")}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit(args.run, args.output)


if __name__ == "__main__":
    main()
