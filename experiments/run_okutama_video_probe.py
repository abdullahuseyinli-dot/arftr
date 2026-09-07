"""Run a prospectively locked P1 screen on role-safe frozen feature caches only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import random
import sys
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from hac.vcoco_v3_neural import decode_factorized_logits
from hac.vcoco_v3_temporal import temporal_teacher_loss
from hac.vcoco_v3_temporal_training import hierarchical_class_weights
from hac.video_probe import SpatiotemporalFactorizedProbe

ARMS = ("vjepa21_real_clip", "vjepa21_repeated_center", "dinov2_native_frames")
PROBES = ("linear", "attentive")
CLASS_NAMES = ("sitting", "standing", "walking_running")
FEATURE_SHAPES = {
    "vjepa21_real_clip": (8, 9, 768),
    "vjepa21_repeated_center": (8, 9, 768),
    "dinov2_native_frames": (16, 1, 768),
}


@dataclass
class PrimaryData:
    sample_ids: np.ndarray
    labels: np.ndarray
    scenarios: np.ndarray
    folds: np.ndarray
    features: dict[str, np.ndarray]
    validity: dict[str, np.ndarray]
    baseline: np.ndarray
    occluded: np.ndarray
    transition: np.ndarray


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_bytes(path: Path, raw: bytes) -> None:
    """Publish a new artifact without overwriting retained or partial evidence."""
    if path.exists():
        raise FileExistsError(f"Retained artifact cannot be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # Every run owns its output directory; callers reject other existing artifacts.
    if path.exists():
        raise FileExistsError(f"Artifact appeared while publishing: {path}")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )


def npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def validate_probability_array(probabilities: np.ndarray, rows: int) -> None:
    if probabilities.shape != (rows, 3) or not np.issubdtype(probabilities.dtype, np.floating):
        raise RuntimeError("Probability array must have shape [rows,3] and floating dtype")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(1), 1.0, atol=2e-6, rtol=0)
    ):
        raise RuntimeError("Probability array is nonfinite or not normalized")


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("Labels and predictions must be aligned vectors")
    if np.any((labels < 0) | (labels > 2)) or np.any((predictions < 0) | (predictions > 2)):
        raise ValueError("Confusion inputs must use the unchanged three classes")
    return np.bincount(labels * 3 + predictions, minlength=9).reshape(3, 3)


def f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    validate_probability_array(probabilities, len(labels))
    if not len(labels):
        raise ValueError("Metrics require at least one retained row")
    predictions = probabilities.argmax(1)
    matrix = confusion(labels, predictions)
    per_class = f1_from_confusion(matrix)
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
    brier = np.square(probabilities - np.eye(3)[labels]).sum(1).mean()
    return {
        "rows": len(labels),
        "macro_f1": float(per_class.mean()),
        "accuracy": float(np.trace(matrix) / len(labels)),
        "nll": float(nll),
        "brier": float(brier),
        "per_class_f1": dict(zip(CLASS_NAMES, per_class.tolist(), strict=True)),
        "class_support": dict(zip(CLASS_NAMES, matrix.sum(1).tolist(), strict=True)),
        "confusion": matrix.tolist(),
    }


def grouped_inner_splits(
    labels: np.ndarray,
    scenarios: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    outer_train = np.flatnonzero(folds != outer_fold)
    outer_held = np.flatnonzero(folds == outer_fold)
    if not len(outer_train) or not len(outer_held):
        raise RuntimeError("An outer fold has an empty fit or held partition")
    if set(scenarios[outer_train]) & set(scenarios[outer_held]):
        raise RuntimeError("A scenario crosses the outer fit/held boundary")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fit, held in splitter.split(
        np.zeros(len(outer_train)), labels[outer_train], scenarios[outer_train]
    ):
        fit_rows, held_rows = outer_train[fit], outer_train[held]
        if set(scenarios[fit_rows]) & set(scenarios[held_rows]):
            raise RuntimeError("A scenario crosses an inner fit/held boundary")
        splits.append((fit_rows, held_rows))
    coverage = np.concatenate([held for _, held in splits])
    if not np.array_equal(np.sort(coverage), outer_train):
        raise RuntimeError("Inner held partitions do not cover outer training exactly once")
    return splits


def validate_feature_array(arm: str, values: np.ndarray, valid: np.ndarray, rows: int) -> None:
    if arm not in FEATURE_SHAPES or values.shape != (rows, *FEATURE_SHAPES[arm]):
        raise RuntimeError(f"Feature shape changed for {arm}")
    if values.dtype != np.float16 or valid.shape != (rows,) or valid.dtype != np.bool_:
        raise RuntimeError("Feature dtype or row-validity contract changed")
    for start in range(0, rows, 64):
        selected = np.asarray(values[start : start + 64])[valid[start : start + 64]]
        if not np.isfinite(selected).all():
            raise RuntimeError("A valid frozen feature row contains nonfinite values")


def _checked_path(root: Path, receipt: Mapping[str, Any]) -> Path:
    path = (root / str(receipt["path"])).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("A fitting input is outside the locked repository")
    forbidden = {"development_metadata.csv", "development_manifest.csv", "development_centres.csv"}
    if path.name.lower() in forbidden or any(
        part.lower() in {"calibration", "confirmation", "test"} for part in path.parts
    ):
        raise RuntimeError("A forbidden mixed/protected input path entered P1")
    if path.stat().st_size != int(receipt["size_bytes"]) or sha256_file(path) != receipt["sha256"]:
        raise RuntimeError("Locked artifact bytes changed before decoding")
    return path


def _bool(value: str) -> bool:
    if value not in {"0", "1", "True", "False", "true", "false"}:
        raise RuntimeError("Invalid manifest Boolean")
    return value in {"1", "True", "true"}


def load_primary_data(root: Path, lock: dict[str, Any]) -> PrimaryData:
    metadata_path = _checked_path(root, lock["inputs"]["clip_index"])
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError("Invalid primary metadata columns")
        rows = list(reader)
    if len(rows) != 4977 or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise RuntimeError("Primary metadata must retain all 4977 original rows")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    labels = np.asarray([int(row["label_index"]) for row in rows], dtype=np.int64)
    scenarios = np.asarray([row["recording_id"] for row in rows])
    folds = np.asarray([int(row["fold"].removeprefix("fold-")) for row in rows], dtype=np.int64)
    expected_folds = {
        scenario: int(fold.removeprefix("fold-"))
        for fold, members in lock["protocol"]["fold_contract"].items()
        for scenario in members
    }
    if (
        len(set(sample_ids)) != 4977
        or len(set(scenarios)) != 11
        or set(folds) != set(range(5))
        or set(scenarios) != set(expected_folds)
        or set(labels) != {0, 1, 2}
        or any(expected_folds[group] != fold for group, fold in zip(scenarios, folds, strict=True))
    ):
        raise RuntimeError("Primary identity, classes or historical scenario folds changed")
    if any(
        row.get("scope") != "grouped_crossfit_oof" or row.get("development_role") != "train"
        for row in rows
    ):
        raise RuntimeError("Only the primary development cohort may enter fitting")
    map_rows = lock["fold_map_rows"]
    if len(map_rows) != len(rows):
        raise RuntimeError("Locked fold-map row count changed")
    for index, mapping in enumerate(map_rows):
        if (
            mapping["sample_id"] != sample_ids[index]
            or mapping["recording_id"] != scenarios[index]
            or int(mapping["outer_fold"]) != folds[index]
        ):
            raise RuntimeError("Locked fold-map identity/order differs from primary metadata")
    baseline_path = _checked_path(root, lock["baseline"])
    with np.load(baseline_path, allow_pickle=False) as source:
        for key, expected in (
            ("sample_ids", sample_ids),
            ("recording_ids", scenarios),
            ("labels", labels),
        ):
            if not np.array_equal(source[key], expected):
                raise RuntimeError("Locked baseline identity/order/labels differ from the cohort")
        baseline = source[lock["baseline"]["probabilities_member"]].astype(np.float64)
    validate_probability_array(baseline, len(rows))
    features, validity = {}, {}
    for arm in ARMS:
        receipt = lock["features"][arm]
        path = _checked_path(root, receipt)
        valid_path = _checked_path(root, receipt["validity"])
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        valid_values = np.load(valid_path, allow_pickle=False)
        column = int(receipt["validity_column"])
        if (
            valid_values.ndim != 2
            or valid_values.shape[0] != len(rows)
            or column not in range(valid_values.shape[1])
        ):
            raise RuntimeError("Locked validity array dimensions changed")
        valid = valid_values[:, column]
        validate_feature_array(arm, values, valid, len(rows))
        features[arm], validity[arm] = values, valid
    return PrimaryData(
        sample_ids,
        labels,
        scenarios,
        folds,
        features,
        validity,
        baseline,
        np.asarray([_bool(row["window_any_occluded"]) for row in rows]),
        np.asarray([_bool(row["transition_window"]) for row in rows]),
    )


def mean_features(values: np.ndarray) -> np.ndarray:
    result = np.empty((len(values), values.shape[-1]), dtype=np.float64)
    for start in range(0, len(values), 64):
        result[start : start + 64] = np.asarray(values[start : start + 64], dtype=np.float32).mean(
            (1, 2)
        )
    return result


def _require_training_classes(labels: np.ndarray) -> None:
    if set(labels.tolist()) != {0, 1, 2}:
        raise RuntimeError("A valid fitting partition is missing one of the three classes")


def fit_linear(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    c: float,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    _require_training_classes(labels)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    model = LogisticRegression(
        C=c,
        solver=config["solver"],
        class_weight=config["class_weight"],
        max_iter=int(config["max_iter"]),
        tol=float(config["tolerance"]),
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(standardized, labels)
        except ConvergenceWarning as error:
            raise RuntimeError("Linear probe did not converge; P1 fails closed") from error
    if int(np.max(model.n_iter_)) >= int(config["max_iter"]):
        raise RuntimeError("Linear probe reached its locked iteration ceiling")
    if not np.array_equal(model.classes_, np.arange(3)):
        raise RuntimeError("Linear probability class order changed")
    return scaler, model


def linear_workload(
    data: PrimaryData,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    config = protocol["probes"]["linear"]
    seed = int(protocol["seed"])
    values = mean_features(data.features[arm])
    valid = data.validity[arm]
    splits = grouped_inner_splits(
        data.labels,
        data.scenarios,
        data.folds,
        outer_fold,
        n_splits=int(config["inner_folds"]),
        seed=int(config["inner_seed"]),
    )
    inner_rows = np.flatnonzero((data.folds != outer_fold) & valid)
    candidates = []
    for c in config["C_values"]:
        probabilities = np.full((len(data.labels), 3), np.nan)
        iterations = []
        for fit_indices, held_indices in splits:
            fit_indices, held_indices = (
                fit_indices[valid[fit_indices]],
                held_indices[valid[held_indices]],
            )
            if not len(held_indices):
                raise RuntimeError("A valid inner-held partition is empty")
            scaler, model = fit_linear(
                values[fit_indices], data.labels[fit_indices], config, float(c), seed
            )
            probabilities[held_indices] = model.predict_proba(
                scaler.transform(values[held_indices])
            )
            iterations.append(int(np.max(model.n_iter_)))
        score = metrics(data.labels[inner_rows], probabilities[inner_rows])
        candidates.append({"C": float(c), "inner_metrics": score, "inner_iterations": iterations})
    selected = min(
        candidates,
        key=lambda row: (-row["inner_metrics"]["macro_f1"], row["inner_metrics"]["nll"], row["C"]),
    )
    fit_indices = np.flatnonzero((data.folds != outer_fold) & valid)
    held_indices = np.flatnonzero(data.folds == outer_fold)
    scaler, model = fit_linear(
        values[fit_indices], data.labels[fit_indices], config, selected["C"], seed
    )
    probabilities = data.baseline[held_indices].copy()
    usable = valid[held_indices]
    if np.any(usable):
        probabilities[usable] = model.predict_proba(scaler.transform(values[held_indices[usable]]))
    checkpoint = npz_bytes(
        coefficients=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        scaler_variance=scaler.var_,
    )
    return (
        probabilities,
        {
            "selection": selected,
            "candidates": candidates,
            "inner_selection_rows": len(inner_rows),
            "inner_baseline_access": False,
            "fit_rows": len(fit_indices),
            "fallback_rows": int((~usable).sum()),
            "final_iterations": int(np.max(model.n_iter_)),
        },
        checkpoint,
    )


def configure_determinism(seed: int) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("Configure P1 determinism before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def attentive_workload(
    data: PrimaryData,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    config = protocol["probes"]["attentive"]
    seed = int(protocol["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Attentive probe requires CUDA BF16; no CPU/precision fallback")
    torch.cuda.reset_peak_memory_stats()
    valid = data.validity[arm]
    fit_indices = np.flatnonzero((data.folds != outer_fold) & valid)
    held_indices = np.flatnonzero(data.folds == outer_fold)
    _require_training_classes(data.labels[fit_indices])
    model = SpatiotemporalFactorizedProbe(
        768,
        **{
            name: config[name]
            for name in ("model_dim", "layers", "attention_heads", "feedforward_dim", "dropout")
        },
        maximum_times=int(config["maximum_times"]),
        maximum_regions=int(config["maximum_regions"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    posture_weight, motion_weight = hierarchical_class_weights(data.labels[fit_indices], device)
    generator = np.random.default_rng(seed)
    batch_size = int(config["batch_size"])
    losses = []
    for epoch in range(int(config["epochs"])):
        model.train()
        order = generator.permutation(fit_indices)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            features = torch.from_numpy(
                np.asarray(data.features[arm][indices], dtype=np.float32)
            ).to(device)
            labels = torch.as_tensor(data.labels[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(features)
                loss = temporal_teacher_loss(
                    output,
                    labels,
                    label_smoothing=float(config["label_smoothing"]),
                    posture_weight=posture_weight,
                    motion_weight=motion_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Attentive probe produced nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"]), error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        losses.append(loss_sum / len(order))
        print(
            f"{arm}/attentive/fold-{outer_fold} epoch {epoch + 1}/{config['epochs']} complete",
            flush=True,
        )
    model.eval()
    probabilities = data.baseline[held_indices].copy()
    usable_positions = np.flatnonzero(valid[held_indices])
    with torch.inference_mode():
        for start in range(0, len(usable_positions), batch_size):
            positions = usable_positions[start : start + batch_size]
            features = torch.from_numpy(
                np.asarray(data.features[arm][held_indices[positions]], dtype=np.float32)
            ).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(features)
            # Decode in FP32, avoiding BF16 softmax rounding in probability receipts.
            probabilities[positions] = (
                decode_factorized_logits(
                    output.posture_logits.float(), output.motion_logits.float()
                )
                .cpu()
                .numpy()
            )
    torch.cuda.synchronize()
    memory = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    buffer = io.BytesIO()
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, buffer)
    details = {
        "fit_rows": len(fit_indices),
        "fallback_rows": int((~valid[held_indices]).sum()),
        "epochs": len(losses),
        "training_loss_per_epoch": losses,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "posture_class_weights": posture_weight.cpu().tolist(),
        "motion_class_weights": motion_weight.cpu().tolist(),
        "peak_cuda_bytes": memory,
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return probabilities, details, buffer.getvalue()


def paired_statistics(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    scenarios: np.ndarray,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    validate_probability_array(candidate, len(labels))
    validate_probability_array(reference, len(labels))
    group_order = np.unique(scenarios)
    cand_pred, ref_pred = candidate.argmax(1), reference.argmax(1)
    cand_cm = np.stack(
        [
            confusion(labels[scenarios == group], cand_pred[scenarios == group])
            for group in group_order
        ]
    )
    ref_cm = np.stack(
        [
            confusion(labels[scenarios == group], ref_pred[scenarios == group])
            for group in group_order
        ]
    )
    observed = float(
        f1_from_confusion(cand_cm.sum(0)).mean() - f1_from_confusion(ref_cm.sum(0)).mean()
    )
    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(0, len(group_order), size=(bootstrap_resamples, len(group_order)))
    boot_cand, boot_ref = cand_cm[draws].sum(1), ref_cm[draws].sum(1)
    supported = (boot_cand.sum(2) > 0).all(1)
    # The P1 protocol fixes all three classes and zero_division=0. Keep every
    # draw, including draws missing a class; report their support diagnostically.
    deltas = f1_from_confusion(boot_cand).mean(1) - f1_from_confusion(boot_ref).mean(1)
    if not len(deltas):
        raise RuntimeError("No scenario bootstrap draw retained all classes")
    swaps = ((np.arange(2 ** len(group_order))[:, None] >> np.arange(len(group_order))) & 1).astype(
        bool
    )
    swapped_cand = np.where(swaps[:, :, None, None], ref_cm, cand_cm).sum(1)
    swapped_ref = np.where(swaps[:, :, None, None], cand_cm, ref_cm).sum(1)
    swap_deltas = f1_from_confusion(swapped_cand).mean(1) - f1_from_confusion(swapped_ref).mean(1)
    rescued, harmed = (
        (cand_pred == labels) & (ref_pred != labels),
        (cand_pred != labels) & (ref_pred == labels),
    )
    group_counts = np.asarray([(scenarios == group).sum() for group in group_order])
    sample_indices = np.arange(len(labels))
    nll_difference = -np.log(np.clip(candidate[sample_indices, labels], 1e-12, 1)) + np.log(
        np.clip(reference[sample_indices, labels], 1e-12, 1)
    )
    targets = np.eye(3)[labels]
    brier_difference = np.square(candidate - targets).sum(1) - np.square(reference - targets).sum(1)
    loss_statistics = {}
    for name, difference in (("nll", nll_difference), ("brier", brier_difference)):
        sums = np.asarray([difference[scenarios == group].sum() for group in group_order])
        sampled = sums[draws].sum(1) / group_counts[draws].sum(1)
        loss_statistics[f"{name}_delta"] = float(difference.mean())
        loss_statistics[f"{name}_delta_two_sided_95pct"] = np.quantile(
            sampled, [0.025, 0.975]
        ).tolist()
        loss_statistics[f"{name}_delta_one_sided_95pct_upper"] = float(np.quantile(sampled, 0.95))
    return {
        **loss_statistics,
        "macro_f1_delta": observed,
        "macro_f1_delta_two_sided_95pct": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "macro_f1_delta_one_sided_95pct_lower": float(np.quantile(deltas, 0.05)),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_valid_fraction": float(supported.mean()),
        "bootstrap_draws_used": len(deltas),
        "bootstrap_missing_class_policy": "retain_fixed_three_classes_zero_division_0",
        "scenario_order": group_order.tolist(),
        "exact_swap_assignments": len(swaps),
        "one_sided_exact_swap_pvalue": float(np.mean(swap_deltas >= observed - 1e-12)),
        "two_sided_exact_swap_pvalue": float(np.mean(np.abs(swap_deltas) >= abs(observed) - 1e-12)),
        "rescued_errors": int(rescued.sum()),
        "new_errors": int(harmed.sum()),
        "shared_errors": int(((cand_pred != labels) & (ref_pred != labels)).sum()),
        "both_correct": int(((cand_pred == labels) & (ref_pred == labels)).sum()),
        "net_correct_change": int(rescued.sum() - harmed.sum()),
        "rescue_harm_per_class": {
            name: {
                "rescued": int(rescued[labels == label].sum()),
                "harmed": int(harmed[labels == label].sum()),
            }
            for label, name in enumerate(CLASS_NAMES)
        },
        "per_scenario": {
            str(group): {
                "macro_f1_delta": float(
                    f1_from_confusion(cand_cm[index]).mean()
                    - f1_from_confusion(ref_cm[index]).mean()
                ),
                "rescued": int(rescued[scenarios == group].sum()),
                "harmed": int(harmed[scenarios == group].sum()),
            }
            for index, group in enumerate(group_order)
        },
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    result, running = {}, 0.0
    for position, key in enumerate(ordered):
        running = max(running, (len(ordered) - position) * pvalues[key])
        result[key] = min(1.0, running)
    return result


def _validate_inner_map(data: PrimaryData, lock: dict[str, Any]) -> None:
    config = lock["protocol"]["probes"]["linear"]
    for fold in range(5):
        assignments = np.full(len(data.labels), -1, dtype=int)
        for inner, (_, held) in enumerate(
            grouped_inner_splits(
                data.labels,
                data.scenarios,
                data.folds,
                fold,
                n_splits=int(config["inner_folds"]),
                seed=int(config["inner_seed"]),
            )
        ):
            assignments[held] = inner
        expected = np.asarray([int(row[f"inner_fold_o{fold}"]) for row in lock["fold_map_rows"]])
        if not np.array_equal(expected, assignments):
            raise RuntimeError("Inner scenario assignment does not reproduce the fitting lock")


def validate_randomization(lock: dict[str, Any]) -> None:
    design = lock["protocol"]["statistics"]
    receipt = lock["randomization"]
    groups = design["scenario_order"]
    if len(groups) != 11 or groups != sorted(set(groups)):
        raise RuntimeError("Expected the sorted eleven-scenario randomization design")
    draws = np.random.default_rng(int(design["bootstrap_seed"])).integers(
        0, 11, size=(int(design["bootstrap_resamples"]), 11), dtype=np.int64
    )
    signs = (2 * ((np.arange(2048)[:, None] >> np.arange(11)) & 1) - 1).astype(np.int8)
    if (
        int(design["bootstrap_resamples"]) != 10_000
        or int(design["exact_scenario_swaps"]) != 2048
        or receipt["scenario_order"] != groups
        or hashlib.sha256(draws.tobytes()).hexdigest() != receipt["bootstrap_indices_bytes_sha256"]
        or hashlib.sha256(signs.tobytes()).hexdigest() != receipt["swap_signs_bytes_sha256"]
    ):
        raise RuntimeError("Scenario randomization does not reproduce its locked byte hashes")


def _workload(
    data: PrimaryData,
    arm: str,
    probe: str,
    fold: int,
    protocol: dict[str, Any],
    output: Path,
    request_sha256: str,
) -> tuple[np.ndarray, bool]:
    directory = output / "workloads" / arm / probe / f"fold-{fold}"
    held = np.flatnonzero(data.folds == fold)
    request = {
        "request_sha256": request_sha256,
        "arm": arm,
        "probe": probe,
        "outer_fold": fold,
        "held_sample_ids": data.sample_ids[held].tolist(),
        "seed": int(protocol["seed"]),
    }
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("request") != request or receipt.get("status") != "P1_WORKLOAD_COMPLETE":
            raise RuntimeError("Retained workload request/status changed")
        checkpoint_name = "checkpoint.npz" if probe == "linear" else "checkpoint.pt"
        if set(receipt.get("artifacts", {})) != {checkpoint_name, "predictions.npz"}:
            raise RuntimeError("Retained workload artifact inventory is incomplete")
        for filename, digest in receipt["artifacts"].items():
            if sha256_file(directory / filename) != digest:
                raise RuntimeError("Retained workload artifact bytes changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], data.sample_ids[held]) or not np.array_equal(
                saved["row_indices"], held
            ):
                raise RuntimeError("Retained workload prediction order changed")
            probabilities = saved["probabilities"].copy()
        validate_probability_array(probabilities, len(held))
        return probabilities, False
    request_path = directory / "request.json"
    if directory.exists():
        unexpected = [path for path in directory.iterdir() if path.name != "request.json"]
        if unexpected:
            raise RuntimeError("Incomplete workload artifacts retained; refusing to overwrite them")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Interrupted workload belongs to a different request")
    else:
        atomic_json(request_path, request)
    begin = time.perf_counter()
    operation = linear_workload if probe == "linear" else attentive_workload
    probabilities, details, checkpoint = operation(data, arm, fold, protocol)
    validate_probability_array(probabilities, len(held))
    checkpoint_name = "checkpoint.npz" if probe == "linear" else "checkpoint.pt"
    atomic_bytes(directory / checkpoint_name, checkpoint)
    atomic_bytes(
        directory / "predictions.npz",
        npz_bytes(
            sample_ids=data.sample_ids[held],
            row_indices=held,
            probabilities=probabilities,
        ),
    )
    atomic_json(
        receipt_path,
        {
            "status": "P1_WORKLOAD_COMPLETE",
            "request": request,
            "details": details,
            "seconds": time.perf_counter() - begin,
            "held_metrics": metrics(data.labels[held], probabilities),
            "artifacts": {
                name: sha256_file(directory / name) for name in (checkpoint_name, "predictions.npz")
            },
        },
    )
    print(f"Completed {arm}/{probe}/fold-{fold}", flush=True)
    return probabilities, True


def _summarize(
    data: PrimaryData, probabilities: dict[str, np.ndarray], protocol: dict[str, Any]
) -> dict:
    design = protocol["statistics"]
    arguments = {
        "bootstrap_resamples": int(design["bootstrap_resamples"]),
        "bootstrap_seed": int(design["bootstrap_seed"]),
    }
    results, contrasts = {}, {}
    for name, values in probabilities.items():
        arm = name.rsplit("__", 1)[0]
        results[name] = {
            "metrics": metrics(data.labels, values),
            "fallback_rows": int((~data.validity[arm]).sum()),
            "per_scenario": {
                str(group): metrics(
                    data.labels[data.scenarios == group], values[data.scenarios == group]
                )
                for group in np.unique(data.scenarios)
            },
            "diagnostic_strata": {
                stratum: metrics(data.labels[selected], values[selected])
                for stratum, selected in (
                    ("occluded", data.occluded),
                    ("clear", ~data.occluded),
                    ("transition", data.transition),
                    ("stable", ~data.transition),
                )
                if selected.any()
            },
        }
        contrasts[f"{name}__vs_baseline"] = paired_statistics(
            data.labels, values, data.baseline, data.scenarios, **arguments
        )
    for probe in PROBES:
        candidate = f"vjepa21_real_clip__{probe}"
        for reference in (f"vjepa21_repeated_center__{probe}", f"dinov2_native_frames__{probe}"):
            name = f"{candidate}__vs_{reference}"
            contrasts[name] = paired_statistics(
                data.labels,
                probabilities[candidate],
                probabilities[reference],
                data.scenarios,
                **arguments,
            )
    primary = sorted(contrasts)
    adjusted = holm_adjust(
        {name: contrasts[name]["one_sided_exact_swap_pvalue"] for name in primary}
    )
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_one_sided_pvalue"] = value
    return {
        "status": "OKUTAMA_VIDEO_P1_EXPLORATORY_CROSSFIT_COMPLETE",
        "rows": len(data.labels),
        "scenarios": len(np.unique(data.scenarios)),
        "seed": int(protocol["seed"]),
        "baseline": metrics(data.labels, data.baseline),
        "models": results,
        "contrasts": contrasts,
        "holm_family": primary,
        "baseline_contrasts": "historical_five_seed_reference_vs_single_seed_screen_descriptive",
        "scope": "reused_development_scenarios_not_independent_confirmation",
        "denominator_policy": "all_original_rows_with_locked_baseline_outer_held_fallback",
        "inner_selection_baseline_access": False,
        "target_images_read": 0,
        "protected_manifest_reads": 0,
    }


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p1")
    if not hasattr(locker, "validate_p1_lock"):
        raise RuntimeError("P1 execution-lock validator is not available; fitting is forbidden")
    return locker.validate_p1_lock(root, path.resolve())


def run(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P1 output must use a dedicated .runs directory")
    configure_determinism(42)
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    if int(protocol["seed"]) != 42 or tuple(protocol["arms"]) != ARMS:
        raise RuntimeError("Locked P1 seed or arm order changed")
    data = load_primary_data(root, lock)
    _validate_inner_map(data, lock)
    validate_randomization(lock)
    request = {
        "status": "P1_REQUEST_BEFORE_FITTING",
        "lock_sha256": sha256_file(args.protocol_lock.resolve()),
        "protocol": protocol,
        "sample_ids_sha256": canonical_digest(data.sample_ids.tolist()),
        "source_sha256": sha256_file(Path(__file__)),
        "output": str(output),
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Output belongs to a different locked P1 request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty output has no matching P1 request")
    else:
        atomic_json(request_path, request)
    request_sha256 = canonical_digest(request)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("request_sha256") != request_sha256:
            raise RuntimeError("Retained summary belongs to a different request")
        for filename, digest in summary["artifacts"].items():
            if sha256_file(output / filename) != digest:
                raise RuntimeError("Retained complete P1 artifact changed")
        return summary
    predictions = {
        f"{arm}__{probe}": np.full((len(data.labels), 3), np.nan)
        for arm in ARMS
        for probe in PROBES
    }
    new_workloads = 0
    for probe in PROBES:
        for arm in ARMS:
            for fold in range(5):
                receipt = output / "workloads" / arm / probe / f"fold-{fold}" / "receipt.json"
                if (
                    args.max_new_workloads is not None
                    and new_workloads >= args.max_new_workloads
                    and not receipt.exists()
                ):
                    return {
                        "status": "P1_BOUNDED_RUN_PAUSED",
                        "new_workloads": new_workloads,
                        "output": str(output),
                    }
                values, fitted = _workload(data, arm, probe, fold, protocol, output, request_sha256)
                predictions[f"{arm}__{probe}"][data.folds == fold] = values
                new_workloads += int(fitted)
    for values in predictions.values():
        validate_probability_array(values, len(data.labels))
    summary = _summarize(data, predictions, protocol)
    _load_lock(root, args.protocol_lock)  # Reject source/input mutations during the run.
    artifact = output / "oof_probabilities.npz"
    atomic_bytes(
        artifact,
        npz_bytes(
            sample_ids=data.sample_ids,
            recording_ids=data.scenarios,
            labels=data.labels,
            folds=data.folds,
            baseline_probabilities=data.baseline,
            **predictions,
        ),
    )
    table = io.StringIO(newline="")
    writer = csv.DictWriter(
        table, fieldnames=["model", "macro_f1", "accuracy", "nll", "brier", "fallback_rows"]
    )
    writer.writeheader()
    for name, result in summary["models"].items():
        writer.writerow(
            {
                "model": name,
                "fallback_rows": result["fallback_rows"],
                **{key: result["metrics"][key] for key in ("macro_f1", "accuracy", "nll", "brier")},
            }
        )
    atomic_bytes(output / "metrics.csv", table.getvalue().encode())
    atomic_json(
        output / "paired_statistics.json",
        {"contrasts": summary["contrasts"], "holm_family": summary["holm_family"]},
    )
    inventory = [artifact, output / "metrics.csv", output / "paired_statistics.json"]
    inventory.extend(sorted((output / "workloads").rglob("receipt.json")))
    inventory.extend(sorted((output / "workloads").rglob("predictions.npz")))
    inventory.extend(sorted((output / "workloads").rglob("checkpoint.npz")))
    inventory.extend(sorted((output / "workloads").rglob("checkpoint.pt")))
    summary.update(
        {
            "request_sha256": request_sha256,
            "artifacts": {
                path.relative_to(output).as_posix(): sha256_file(path) for path in inventory
            },
        }
    )
    atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-workloads", type=int)
    args = parser.parse_args()
    if args.max_new_workloads is not None and args.max_new_workloads < 1:
        parser.error("--max-new-workloads must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
