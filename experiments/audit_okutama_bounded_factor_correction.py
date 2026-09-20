"""Independently replay the bounded factor-correction checkpoints and result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_bounded_factor_correction as trial
import torch

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.bounded_factor_correction import apply_numpy_correction
from hac.cached_frame_supervision import validate_cached_frame_supervision_receipt
from hac.matr_artifacts import load_cached_study_data
from hac.source_swap_data import immutable_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def same(left: Any, right: Any, message: str) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        require(np.array_equal(left, right), message)
    else:
        require(left == right, message)


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    with np.load(path, allow_pickle=False) as saved:
        observed = {name.replace("__", "."): saved[name] for name in saved.files}
    expected = model.state_dict()
    same(set(observed), set(expected), "Checkpoint state inventory changed")
    state = {}
    for name, reference in expected.items():
        value = observed[name]
        same(value.shape, tuple(reference.shape), f"Checkpoint shape changed: {name}")
        require(np.isfinite(value).all(), f"Checkpoint contains non-finite values: {name}")
        state[name] = torch.as_tensor(value)
    model.load_state_dict(state, strict=True)


def replay_fit(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    store: trial.fsar.FrameFeatureStore,
    lock_hash: str,
    arm: str,
    fold: int,
    seed: int,
) -> np.ndarray:
    held = np.flatnonzero(data["folds"] == fold)
    _, anchors, ancestry = trial.arftr_anchors(data, fold)
    anchor = anchors[trial.SEEDS.index(seed)]
    prediction_path, checkpoint_path, receipt_path = trial.fit_paths(run, arm, fold, seed)
    receipt = trial.read_json(receipt_path)
    require(
        receipt.get("status") == "BOUNDED_FACTOR_CORRECTION_FIT_COMPLETE"
        and receipt.get("execution_lock_sha256") == lock_hash
        and receipt.get("arm") == arm
        and receipt.get("fold") == fold
        and receipt.get("seed") == seed
        and receipt.get("outer_held_labels_read") == 0,
        "Fit receipt identity changed",
    )
    same(
        receipt.get("held_sample_ids_sha256"),
        canonical_hash(data["sample_ids"][held].tolist()),
        "Held IDs changed",
    )
    same(receipt.get("arftr_ancestry"), ancestry, "ARFTR training ancestry changed")
    same(receipt.get("predictions_sha256"), file_sha256(prediction_path), "Prediction hash changed")
    same(receipt.get("checkpoint_sha256"), file_sha256(checkpoint_path), "Checkpoint hash changed")
    model = trial.make_model(protocol, arm)
    load_checkpoint(model, checkpoint_path)
    model.eval()
    feature = store.centers(held)
    corrections, auxiliary = [], []
    with torch.inference_mode():
        for left in range(0, len(held), protocol["training"]["batch_size"]):
            output = model(
                torch.as_tensor(feature[left : left + protocol["training"]["batch_size"]]),
                torch.as_tensor(
                    anchor[left : left + protocol["training"]["batch_size"]], dtype=torch.float32
                ),
            )
            corrections.append(output["correction"].numpy())
            auxiliary.append(output["auxiliary_logits"].numpy())
    replay_correction = np.concatenate(corrections).astype(np.float64)
    replay_auxiliary = np.concatenate(auxiliary).astype(np.float64)
    replay_probability = apply_numpy_correction(anchor, replay_correction)
    with np.load(prediction_path, allow_pickle=False) as saved:
        same(saved["held_rows"], held, "Held rows changed")
        same(saved["sample_ids"], data["sample_ids"][held], "Held sample order changed")
        same(saved["anchor_probabilities"], anchor, "Stored anchor changed")
        same(saved["corrections"], replay_correction, "Checkpoint correction replay changed")
        same(saved["auxiliary_logits"], replay_auxiliary, "Checkpoint auxiliary replay changed")
        same(saved["probabilities"], replay_probability, "Checkpoint probability replay changed")
        return saved["probabilities"].copy()


def audit(run: Path, output: Path) -> dict[str, Any]:
    protocol = trial.read_json(trial.PROTOCOL)
    trial.validate_protocol(protocol)
    data = load_cached_study_data(trial.SOURCE)
    validate_cached_frame_supervision_receipt(trial.ROOT, trial.FRAME_DATA)
    lock_path = run / "execution_lock.json"
    lock_hash = file_sha256(lock_path)
    lock = trial.validate_lock(run, data, lock_hash)
    require(lock.get("outer_held_labels_read") == 0, "Execution label embargo changed")
    store = trial.fsar.FrameFeatureStore()
    rows = len(data["labels"])
    replay = np.full((len(trial.ARMS), 3, rows, 3), np.nan)
    fits = 0
    for fold in range(5):
        held = np.flatnonzero(data["folds"] == fold)
        _, anchors, _ = trial.arftr_anchors(data, fold)
        replay[0][:, held] = anchors
        for arm_index, arm in enumerate(trial.TRAINED_ARMS, start=1):
            for seed_index, seed in enumerate(trial.SEEDS):
                replay[arm_index, seed_index, held] = replay_fit(
                    run, protocol, data, store, lock_hash, arm, fold, seed
                )
                fits += 1
    require(fits == 60 and np.isfinite(replay).all(), "Audited fit inventory is incomplete")
    result_dir = run / "results/v0001"
    oof_path, summary_path = result_dir / "oof_probabilities.npz", result_dir / "summary.json"
    summary = trial.read_json(summary_path)
    with np.load(oof_path, allow_pickle=False) as saved:
        same(saved["sample_ids"], data["sample_ids"], "OOF sample IDs changed")
        same(tuple(saved["arms"].tolist()), trial.ARMS, "OOF arm order changed")
        same(tuple(saved["seeds"].tolist()), trial.SEEDS, "OOF seed order changed")
        same(saved["seed_probabilities"], replay, "Published OOF predictions changed")
        averaged = saved["mean_probabilities"].copy()
    same(averaged, replay.mean(1), "Published mean probabilities changed")
    metrics = {
        arm: probability_metrics(data["labels"], averaged[index])
        for index, arm in enumerate(trial.ARMS)
    }
    same(metrics, summary["metrics"], "Published metrics changed")
    same(
        {arm: trial.class_precision_recall(metric) for arm, metric in metrics.items()},
        summary["per_class_precision_recall"],
        "Published precision/recall changed",
    )
    require(summary.get("model_fits") == 60, "Published model-fit count changed")
    require(summary.get("execution_lock_sha256") == lock_hash, "Published lock changed")
    require(
        summary.get("oof_probabilities_sha256") == file_sha256(oof_path),
        "Published OOF hash changed",
    )
    # Recompute all statistical and gate logic independently through the frozen
    # pure functions, without writing into the execution result directory.
    statistics = {}
    for arm_index, arm in enumerate(trial.TRAINED_ARMS, start=1):
        statistics[arm] = trial.fsar._scenario_statistics(
            data["labels"],
            averaged[arm_index],
            averaged[0],
            data["scenarios"],
            resamples=10000,
            seed=20260912 + arm_index,
        )
    center_statistics = trial.fsar._scenario_statistics(
        data["labels"],
        averaged[2],
        averaged[1],
        data["scenarios"],
        resamples=protocol["statistics"]["scenario_bootstrap_resamples"],
        seed=protocol["statistics"]["center_control_seed"],
    )
    same(statistics, summary["statistics_vs_arftr"], "Published scenario statistics changed")
    same(
        center_statistics,
        summary["primary_vs_center_control_statistics"],
        "Center-control statistics changed",
    )
    support_masks = {
        "known_pure_support": data["node_support_complete"] & ~data["node_support_boundary"],
        "known_mixed_support": data["node_support_complete"] & data["node_support_boundary"],
        "unknown_support": ~data["node_support_complete"],
    }
    for name, mask in support_masks.items():
        same(summary["support_strata"][name]["rows"], int(mask.sum()), f"{name} rows changed")
        expected_metrics = {
            arm: probability_metrics(data["labels"][mask], averaged[index, mask])
            for index, arm in enumerate(trial.ARMS)
        }
        same(
            expected_metrics, summary["support_strata"][name]["metrics"], f"{name} metrics changed"
        )
        same(
            trial.transitions(data["labels"][mask], averaged[2, mask], averaged[0, mask]),
            summary["support_strata"][name]["primary_transitions_vs_arftr"],
            f"{name} transitions changed",
        )
    gates = protocol["screen_gates"]
    primary, anchor = metrics[trial.ARMS[2]], metrics[trial.ARMS[0]]
    gain = primary["macro_f1"] - anchor["macro_f1"]
    center_gain = primary["macro_f1"] - metrics[trial.ARMS[1]]["macro_f1"]
    primary_transitions = trial.transitions(data["labels"], averaged[2], averaged[0])
    class_losses = np.asarray(anchor["per_class_f1"]) - np.asarray(primary["per_class_f1"])
    scenario_groups = np.unique(data["scenarios"])
    improved_scenarios = sum(
        probability_metrics(
            data["labels"][data["scenarios"] == group],
            averaged[2, data["scenarios"] == group],
        )["macro_f1"]
        > probability_metrics(
            data["labels"][data["scenarios"] == group],
            averaged[0, data["scenarios"] == group],
        )["macro_f1"]
        for group in scenario_groups
    )
    seed_sd = 100 * float(
        np.std(
            [probability_metrics(data["labels"], replay[2, seed])["macro_f1"] for seed in range(3)],
            ddof=1,
        )
    )
    expected_checks = {
        "minimum_primary_macro_f1": primary["macro_f1"] >= gates["minimum_primary_macro_f1"],
        "minimum_gain_over_arftr_points": 100 * gain >= gates["minimum_gain_over_arftr_points"],
        "minimum_gain_over_center_control_points": 100 * center_gain
        >= gates["minimum_gain_over_center_control_points"],
        "minimum_net_corrections_over_arftr": primary_transitions["net_corrections"]
        >= gates["minimum_net_corrections_over_arftr"],
        "minimum_known_pure_net_corrections": summary["support_strata"]["known_pure_support"][
            "primary_transitions_vs_arftr"
        ]["net_corrections"]
        >= gates["minimum_known_pure_net_corrections"],
        "minimum_known_mixed_net_corrections": summary["support_strata"]["known_mixed_support"][
            "primary_transitions_vs_arftr"
        ]["net_corrections"]
        >= gates["minimum_known_mixed_net_corrections"],
        "both_primary_scenario_swap_p_maximum": statistics[trial.ARMS[2]][
            "scenario_swap_one_sided_p"
        ]
        <= gates["both_primary_scenario_swap_p_maximum"]
        and center_statistics["scenario_swap_one_sided_p"]
        <= gates["both_primary_scenario_swap_p_maximum"],
        "maximum_nll_delta_over_arftr_one_sided_upper95": statistics[trial.ARMS[2]][
            "nll_delta_one_sided_upper95"
        ]
        <= gates["maximum_nll_delta_over_arftr_one_sided_upper95"],
        "maximum_brier_delta_over_arftr_one_sided_upper95": statistics[trial.ARMS[2]][
            "brier_delta_one_sided_upper95"
        ]
        <= gates["maximum_brier_delta_over_arftr_one_sided_upper95"],
        "maximum_worst_class_f1_loss_points": 100 * float(class_losses.max())
        <= gates["maximum_worst_class_f1_loss_points"],
        "minimum_improved_scenarios": improved_scenarios >= gates["minimum_improved_scenarios"],
        "maximum_seed_macro_f1_sd_points": seed_sd <= gates["maximum_seed_macro_f1_sd_points"],
        "exact_arftr_replay": True,
    }
    expected_checks = {name: bool(value) for name, value in expected_checks.items()}
    same(expected_checks, summary["screen_checks"], "Published screen checks changed")
    same(
        bool(all(expected_checks.values())),
        summary["screen_gate_passed"],
        "Published screen gate changed",
    )
    audit_summary = {
        "status": "BOUNDED_FACTOR_CORRECTION_INDEPENDENT_AUDIT_PASS",
        "complete": True,
        "execution_run": str(run),
        "execution_lock_sha256": lock_hash,
        "execution_summary_sha256": file_sha256(summary_path),
        "fits_verified": fits,
        "checkpoint_forward_passes_replayed": fits,
        "outer_held_labels_read_before_complete_inventory": 0,
        "metrics": metrics,
        "screen_gate_passed": bool(summary["screen_gate_passed"]),
        "limitations": "Adaptive repeated-development evidence; this audit validates computation, not external generalization.",
    }
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "summary.json", audit_summary)
    print(
        json.dumps(
            {"status": audit_summary["status"], "output": str(output / "summary.json")}, indent=2
        )
    )
    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=trial.DEFAULT_RUN)
    parser.add_argument(
        "--output",
        type=Path,
        default=trial.ROOT / ".runs/research_20260912/bounded_factor_correction_v1_audit",
    )
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    run.relative_to(trial.ROOT.resolve())
    output.relative_to(trial.ROOT.resolve())
    audit(run, output)


if __name__ == "__main__":
    main()
