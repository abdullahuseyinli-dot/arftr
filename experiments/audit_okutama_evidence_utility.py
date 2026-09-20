"""Independently replay and evaluate the locked20-fit evidence-utility screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_okutama_evidence_utility as trial

from hac.actor_memory_base import file_sha256
from hac.evidence_decomposition import (
    ARM_NAMES,
    EvidenceUtilityReader,
    apply_standardizer,
    decode_factor_logits,
    fit_standardizer,
)

ROOT = Path(__file__).resolve().parents[1]
ARFTR = ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
ARFTR_SHA256 = "ec9957a9393e6f1803d274d281549c20290ed59343b308c5be3446be37cec720"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/evidence_utility_screen_v1_audit"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fixed_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels, probabilities = (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )
    require(probabilities.shape == (len(labels), 3), "Probability shape changed")
    require(
        np.isfinite(probabilities).all()
        and np.min(probabilities) >= 0
        and np.allclose(probabilities.sum(1), 1, atol=1e-6),
        "Probability simplex is invalid",
    )
    predictions = probabilities.argmax(1)
    confusion = np.bincount(3 * labels + predictions, minlength=9).reshape(3, 3)
    true_count, predicted_count = confusion.sum(1), confusion.sum(0)
    tp = confusion.diagonal()
    precision = np.divide(
        tp, predicted_count, out=np.zeros(3, dtype=float), where=predicted_count > 0
    )
    recall = np.divide(tp, true_count, out=np.zeros(3, dtype=float), where=true_count > 0)
    f1 = np.divide(
        2 * tp,
        true_count + predicted_count,
        out=np.zeros(3, dtype=float),
        where=(true_count + predicted_count) > 0,
    )
    confidence = probabilities.max(1)
    correct = predictions == labels
    bins = np.minimum(np.floor(confidence * 15).astype(np.int64), 14)
    ece = 0.0
    for bin_index in range(15):
        selected = bins == bin_index
        if selected.any():
            ece += selected.mean() * abs(
                float(confidence[selected].mean()) - float(correct[selected].mean())
            )
    return {
        "rows": len(labels),
        "macro_f1": float(f1.mean()),
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()
        ),
        "brier": float(np.square(probabilities - np.eye(3)[labels]).sum(1).mean()),
        "ece15": float(ece),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion": confusion.tolist(),
    }


def _macro_from_confusion(confusion: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(-2) + confusion.sum(-1)
    return np.divide(
        2 * diagonal,
        denominator,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=denominator > 0,
    ).mean(-1)


def paired_scenario_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    scenarios: np.ndarray,
    *,
    resamples: int = 100000,
    seed: int = 20260913,
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    require(len(groups) == 11, "Scenario population changed")
    matrices = np.zeros((2, len(groups), 3, 3), dtype=np.int64)
    for index, scenario in enumerate(groups):
        selected = scenarios == scenario
        matrices[0, index] = np.bincount(
            3 * labels[selected] + candidate[selected].argmax(1), minlength=9
        ).reshape(3, 3)
        matrices[1, index] = np.bincount(
            3 * labels[selected] + reference[selected].argmax(1), minlength=9
        ).reshape(3, 3)
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 10000):
        stop = min(start + 10000, resamples)
        counts = np.stack(
            [
                np.bincount(draw, minlength=len(groups))
                for draw in rng.integers(0, len(groups), size=(stop - start, len(groups)))
            ]
        )
        candidate_confusion = np.einsum("bg,gij->bij", counts, matrices[0], optimize=True)
        reference_confusion = np.einsum("bg,gij->bij", counts, matrices[1], optimize=True)
        differences[start:stop] = _macro_from_confusion(
            candidate_confusion
        ) - _macro_from_confusion(reference_confusion)
    observed = (
        fixed_metrics(labels, candidate)["macro_f1"] - fixed_metrics(labels, reference)["macro_f1"]
    )
    return {
        "observed": observed,
        "resamples": resamples,
        "seed": seed,
        "quantile_method": "linear",
        "interval95": np.quantile(differences, [0.025, 0.975], method="linear").tolist(),
        "probability_positive": float(np.mean(differences > 0)),
    }


def exact_scenario_signs(
    labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray, scenarios: np.ndarray
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    observed = (
        fixed_metrics(labels, candidate)["macro_f1"] - fixed_metrics(labels, reference)["macro_f1"]
    )
    values = np.empty(2 ** len(groups), dtype=np.float64)
    for assignment in range(len(values)):
        left, right = candidate.copy(), reference.copy()
        for bit, group in enumerate(groups):
            if not ((assignment >> bit) & 1):
                selected = scenarios == group
                left[selected], right[selected] = reference[selected], candidate[selected]
        values[assignment] = (
            fixed_metrics(labels, left)["macro_f1"] - fixed_metrics(labels, right)["macro_f1"]
        )
    return {
        "observed_pooled_macro_f1_delta": observed,
        "assignments": len(values),
        "one_sided_p": float(np.mean(values >= observed - 1e-15)),
        "two_sided_p": float(np.mean(np.abs(values) >= abs(observed) - 1e-15)),
    }


def _load_checkpoint(
    path: Path,
) -> tuple[EvidenceUtilityReader, np.ndarray, np.ndarray, np.ndarray]:
    model = EvidenceUtilityReader()
    with np.load(path, allow_pickle=False) as saved:
        mean, scale, weights = (
            saved["scaler_mean"],
            saved["scaler_scale"],
            saved["class_weight"],
        )
        state = {
            name[len("state__") :]: torch.as_tensor(saved[name])
            for name in saved.files
            if name.startswith("state__")
        }
    require(set(state) == set(model.state_dict()), "Checkpoint state inventory changed")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mean, scale, weights


def _replay_fit(
    run: Path,
    arm_features: np.ndarray,
    data: dict[str, np.ndarray],
    arm_index: int,
    fold: int,
    lock_sha: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    arm = ARM_NAMES[arm_index]
    directory = run / "models" / arm / f"fold-{fold}" / "seed-42"
    receipt = trial.read_json(directory / "receipt.json")
    train = np.flatnonzero(data["folds"] != fold)
    held = np.flatnonzero(data["folds"] == fold)
    prediction_path, checkpoint_path = directory / "predictions.npz", directory / "checkpoint.npz"
    expected_schedule = trial.batch_schedule(
        np.arange(len(train), dtype=np.int64),
        steps=config["steps"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    trial.seed_training(config["seed"])
    expected_initial_state = trial.model_state_sha256(EvidenceUtilityReader())
    require(
        receipt.get("status") == "EVIDENCE_UTILITY_FIT_COMPLETE_METRICS_EMBARGOED"
        and receipt.get("arm") == arm
        and receipt.get("fold") == fold
        and receipt.get("seed") == config["seed"]
        and receipt.get("execution_lock_sha256") == lock_sha
        and receipt.get("train_ids_sha256") == trial.array_sha256(data["sample_ids"][train])
        and receipt.get("held_ids_sha256") == trial.array_sha256(data["sample_ids"][held])
        and receipt.get("held_rows") == held.tolist()
        and receipt.get("held_sample_ids") == data["sample_ids"][held].tolist()
        and receipt.get("held_labels_passed_to_fit") == 0
        and receipt.get("train_rows") == len(train)
        and receipt.get("held_row_count") == len(held)
        and receipt.get("optimizer_updates") == config["steps"]
        and receipt.get("schedule_sha256") == trial.array_sha256(expected_schedule)
        and receipt.get("initial_state_sha256") == expected_initial_state
        and receipt.get("predictions_sha256") == file_sha256(prediction_path)
        and receipt.get("checkpoint_sha256") == file_sha256(checkpoint_path),
        "Fit receipt changed",
    )
    with np.load(prediction_path, allow_pickle=False) as saved:
        require(
            np.array_equal(saved["sample_ids"], data["sample_ids"][held])
            and np.array_equal(saved["row_indices"], held),
            "Held prediction identity changed",
        )
        require(
            set(saved.files) == {"sample_ids", "row_indices", "probabilities"},
            "Prediction inventory changed",
        )
        published = saved["probabilities"]
    model, mean, scale, weights = _load_checkpoint(checkpoint_path)
    expected_mean, expected_scale = fit_standardizer(arm_features[train])
    expected_weights = trial.class_weights(data["labels"][train])
    require(
        np.array_equal(mean, expected_mean)
        and np.array_equal(scale, expected_scale)
        and np.array_equal(weights, expected_weights),
        "Checkpoint contains non-training-only scaler or class weights",
    )
    features = apply_standardizer(arm_features[held], mean, scale)
    replay = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            replay.append(
                decode_factor_logits(model(torch.from_numpy(features[start : start + 256]))).numpy()
            )
    replay = np.concatenate(replay).astype(np.float64)
    maximum_absolute_drift = float(np.max(np.abs(replay - published)))
    require(
        np.allclose(replay, published, rtol=1e-4, atol=1e-5),
        "CPU checkpoint replay drift exceeded tolerance",
    )
    require(
        np.array_equal(replay.argmax(1), published.argmax(1)),
        "CPU checkpoint replay changed decisions",
    )
    return held, published, maximum_absolute_drift


def audit(run: Path, cache_root: Path, output: Path) -> dict[str, Any]:
    inventory_path = run / "fit_inventory.json"
    inventory = trial.read_json(inventory_path)
    require(
        inventory.get("status") == "EVIDENCE_UTILITY_INITIAL_20_FITS_COMPLETE_METRICS_EMBARGOED"
        and inventory.get("fits") == 20
        and len(inventory.get("inventory", [])) == 20,
        "Twenty-fit embargoed inventory is incomplete",
    )
    protocol, data, store, lock_sha = trial.prepare(cache_root, run)
    require(inventory["execution_lock_sha256"] == lock_sha, "Inventory execution ancestry changed")
    expected_inventory = {
        (arm, fold): f"models/{arm}/fold-{fold}/seed-42" for arm in ARM_NAMES for fold in range(5)
    }
    actual_inventory = {}
    for record in inventory["inventory"]:
        key = (record.get("arm"), record.get("fold"))
        require(key not in actual_inventory, "Duplicate fit inventory member")
        actual_inventory[key] = record.get("directory")
        expected_directory = expected_inventory.get(key)
        require(
            expected_directory is not None
            and record.get("directory") == expected_directory
            and record.get("receipt_sha256")
            == file_sha256(run / expected_directory / "receipt.json"),
            "Fit inventory member or receipt hash changed",
        )
    require(actual_inventory == expected_inventory, "Four-arm five-fold inventory changed")
    probabilities = np.full((4, 4977, 3), np.nan, dtype=np.float64)
    replay_drifts = {}
    for arm_index in range(4):
        arm_features = store.arm(arm_index)
        for fold in range(5):
            held, prediction, drift = _replay_fit(
                run,
                arm_features,
                data,
                arm_index,
                fold,
                lock_sha,
                protocol["reader"],
            )
            probabilities[arm_index, held] = prediction
            replay_drifts[f"{ARM_NAMES[arm_index]}/fold-{fold}"] = drift
        del arm_features
    require(np.isfinite(probabilities).all(), "OOF prediction inventory is incomplete")
    metrics = {
        arm: fixed_metrics(data["labels"], probabilities[index])
        for index, arm in enumerate(ARM_NAMES)
    }
    folds = {
        arm: {
            str(fold): fixed_metrics(
                data["labels"][data["folds"] == fold], probabilities[index, data["folds"] == fold]
            )
            for fold in range(5)
        }
        for index, arm in enumerate(ARM_NAMES)
    }
    scenario_metrics = {
        arm: {
            str(scenario): fixed_metrics(
                data["labels"][data["scenarios"] == scenario],
                probabilities[index, data["scenarios"] == scenario],
            )
            for scenario in np.unique(data["scenarios"])
        }
        for index, arm in enumerate(ARM_NAMES)
    }
    primary_statistics = paired_scenario_bootstrap(
        data["labels"], probabilities[3], probabilities[2], data["scenarios"]
    )
    exact = exact_scenario_signs(
        data["labels"], probabilities[3], probabilities[2], data["scenarios"]
    )
    require(file_sha256(ARFTR) == ARFTR_SHA256, "Retained ARFTR bytes changed")
    with np.load(ARFTR, allow_pickle=False) as saved:
        require(
            np.array_equal(saved["sample_ids"], data["sample_ids"])
            and np.array_equal(saved["labels"], data["labels"])
            and tuple(saved["arms"].tolist())[5] == "r5_arftr_full",
            "Retained ARFTR identity changed",
        )
        arftr = saved["mean_probabilities"][5]
    truth, retained_prediction = data["labels"], arftr.argmax(1)
    transitions = {}
    rescued_masks = []
    for index, arm in enumerate(ARM_NAMES):
        prediction = probabilities[index].argmax(1)
        rescued = (retained_prediction != truth) & (prediction == truth)
        harmed = (retained_prediction == truth) & (prediction != truth)
        rescued_masks.append(rescued)
        transitions[arm] = {
            "rescues": int(rescued.sum()),
            "harms": int(harmed.sum()),
            "net": int(rescued.sum() - harmed.sum()),
            "folds": {
                str(fold): {
                    "rescues": int((rescued & (data["folds"] == fold)).sum()),
                    "harms": int((harmed & (data["folds"] == fold)).sum()),
                    "net": int(
                        (rescued & (data["folds"] == fold)).sum()
                        - (harmed & (data["folds"] == fold)).sum()
                    ),
                }
                for fold in range(5)
            },
        }
        require(
            transitions[arm]["net"]
            == int((retained_prediction != truth).sum() - (prediction != truth).sum()),
            "Rescue/harm accounting identity failed",
        )
    unique_rescues = int(
        (rescued_masks[3] & ~rescued_masks[0] & ~rescued_masks[1] & ~rescued_masks[2]).sum()
    )
    gains = {
        reference: metrics[ARM_NAMES[3]]["macro_f1"] - metrics[reference]["macro_f1"]
        for reference in ARM_NAMES[:3]
    }
    discovery = bool(
        gains[ARM_NAMES[2]] >= 0.005
        and gains[ARM_NAMES[0]] >= 0.005
        and gains[ARM_NAMES[1]] >= 0.005
        and all(
            folds[ARM_NAMES[3]][str(fold)]["macro_f1"] > folds[ARM_NAMES[2]][str(fold)]["macro_f1"]
            for fold in range(5)
        )
        and primary_statistics["interval95"][0] > 0
    )
    b3, b2 = transitions[ARM_NAMES[3]], transitions[ARM_NAMES[2]]
    safer = (
        b3["rescues"] >= b2["rescues"]
        and b3["harms"] <= b2["harms"] - 10
        and all(b3["folds"][str(fold)]["net"] > b2["folds"][str(fold)]["net"] for fold in range(5))
    )
    opportunity = bool(b3["rescues"] >= 40 and (unique_rescues >= 10 or safer))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "oof_probabilities.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=data["sample_ids"],
            labels=truth,
            scenarios=data["scenarios"],
            folds=data["folds"],
            arms=np.asarray(ARM_NAMES),
            probabilities=probabilities,
            arftr_probabilities=arftr,
        )
    summary = {
        "status": "EVIDENCE_UTILITY_INDEPENDENT_AUDIT_COMPLETE",
        "fits_replayed": 20,
        "nominal_parameters_all_arms": 737858,
        "active_parameters": {
            ARM_NAMES[0]: 442946,
            ARM_NAMES[1]: 737858,
            ARM_NAMES[2]: 737858,
            ARM_NAMES[3]: 737858,
        },
        "metrics": metrics,
        "fold_metrics": folds,
        "scenario_metrics": scenario_metrics,
        "checkpoint_replay": {
            "rtol": 1e-4,
            "atol": 1e-5,
            "maximum_absolute_drift": max(replay_drifts.values()),
            "per_fit_maximum_absolute_drift": replay_drifts,
            "all_decisions_exact": True,
            "training_scalers_and_class_weights_independently_recomputed": True,
        },
        "primary_B3_minus_B2": primary_statistics,
        "exact_scenario_signs_descriptive": exact,
        "macro_f1_gains": gains,
        "ARFTR_metrics": fixed_metrics(truth, arftr),
        "ARFTR_transitions": transitions,
        "B3_unique_ARFTR_error_rescues_beyond_all_controls": unique_rescues,
        "safer_shared_path": safer,
        "discovery_gate_passed": discovery,
        "opportunity_gate_passed": opportunity,
        "confirmation_authorized": discovery,
        "intervention_design_authorized": discovery and opportunity,
        "limitations": "Repeated development scenarios; no external confirmation. ARFTR overlap is retrospective and was loaded only after the sealed20-fit inventory.",
        "artifacts": {
            "fit_inventory_sha256": file_sha256(inventory_path),
            "execution_lock_sha256": lock_sha,
            "ARFTR_sha256": file_sha256(ARFTR),
            "oof_probabilities_sha256": file_sha256(output / "oof_probabilities.npz"),
        },
    }
    trial.write_json_exclusive(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=trial.DEFAULT_RUN)
    parser.add_argument("--cache-root", type=Path, default=trial.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.run, args.cache_root, args.output):
        path.resolve().relative_to(ROOT.resolve())
    print(
        json.dumps(
            audit(args.run.resolve(), args.cache_root.resolve(), args.output.resolve()), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
