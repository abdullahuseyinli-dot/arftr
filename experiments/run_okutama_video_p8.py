"""Run the locked Camera-Compensated Actor Correspondence (CCAC) trial."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_video_p5 as p5

from hac.ccac_features import (
    ARMS,
    CCACFeatures,
    arm_inputs,
    validate_feature_blocks,
)
from hac.video_consensus import uniform_consensus
from hac.video_multiscale import derive_multiscale_features
from hac.video_token_moments import TokenMomentFeatures, derive_token_moments

p4, p3, p2, p1 = p5.p4, p5.p3, p5.p2, p5.p1
C_VALUES = (1e-5, 1e-4, 1e-3, 1e-2)
SYSTEMS = ("p6_reference", "ccac_replacement_triad")
STATUS = "OKUTAMA_VIDEO_P8_ADAPTIVE_CROSSFIT_COMPLETE"


@dataclass(frozen=True)
class P8Features:
    moments: TokenMomentFeatures
    ccac: CCACFeatures

    @property
    def long_valid(self) -> np.ndarray:
        return self.moments.long_valid


def _align_ccac_arrays(
    arrays: dict[str, np.ndarray], sample_ids: np.ndarray, long_valid: np.ndarray
) -> CCACFeatures:
    """Validate every fixed block and join by an exact unique-ID bijection."""
    expected = {
        "sample_ids",
        "long_valid",
        "quality",
        "raw_translation",
        "compensated_translation",
        "within_actor",
        "translation_feature_valid",
        "articulation_feature_valid",
        "requested_pair_counts",
        "available_pair_counts",
        "camera_pair_counts",
        "translation_pair_counts",
        "articulation_pair_counts",
    }
    rows = len(sample_ids)
    if set(arrays) != expected:
        raise RuntimeError("CCAC feature array inventory changed")
    ids = arrays["sample_ids"]
    if (
        ids.shape != (rows,)
        or ids.dtype.kind not in "US"
        or sample_ids.shape != (rows,)
        or sample_ids.dtype.kind not in "US"
        or len(set(ids.tolist())) != rows
        or len(set(sample_ids.tolist())) != rows
        or set(ids.tolist()) != set(sample_ids.tolist())
    ):
        raise RuntimeError("CCAC sample identities are not an exact unique-ID bijection")
    lookup = {value: index for index, value in enumerate(ids.tolist())}
    order = np.asarray([lookup[value] for value in sample_ids.tolist()], dtype=np.int64)
    for name, values in arrays.items():
        if values.ndim < 1 or values.shape[0] != rows:
            raise RuntimeError(f"CCAC array row count changed: {name}")
    aligned = {name: values[order] for name, values in arrays.items()}
    if (
        arrays["long_valid"].shape != (rows,)
        or arrays["long_valid"].dtype != np.bool_
        or long_valid.shape != (rows,)
        or long_valid.dtype != np.bool_
        or not np.array_equal(aligned["long_valid"], long_valid)
    ):
        raise RuntimeError("CCAC long-valid identity changed")
    ccac = CCACFeatures(
        aligned["quality"],
        aligned["raw_translation"],
        aligned["compensated_translation"],
        aligned["within_actor"],
        aligned["translation_feature_valid"],
        aligned["articulation_feature_valid"],
    )
    validate_feature_blocks(ccac, rows)
    counts = []
    for name in ("requested", "available", "camera", "translation", "articulation"):
        values = aligned[f"{name}_pair_counts"]
        if values.shape != (rows,) or values.dtype != np.int16:
            raise RuntimeError("CCAC pair-count dtype/shape changed")
        counts.append(values)
    requested, available, camera, translation, articulation = counts
    if (
        np.any(requested != 15)
        or np.any(articulation < 0)
        or any(np.any(left < right) for left, right in zip(counts, counts[1:], strict=False))
        or not np.array_equal(ccac.translation_valid, translation >= 10)
        or not np.array_equal(ccac.articulation_valid, articulation >= 10)
    ):
        raise RuntimeError("CCAC count/mask hierarchy changed")
    quality = ccac.quality
    for column, count in enumerate((available, camera, translation, articulation)):
        if not np.array_equal(quality[:, column], (count / 15).astype(np.float32)):
            raise RuntimeError("CCAC quality fractions disagree with counts")
    if (
        not np.array_equal(quality[:, 4], ccac.translation_valid.astype(np.float32))
        or not np.array_equal(quality[:, 5], ccac.articulation_valid.astype(np.float32))
        or np.any(quality < 0)
        or np.any(quality[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15]] > 1)
        or np.any(~np.isin(quality[:, [10, 12]], [0, 1]))
        or np.any(quality[quality[:, 10] == 0, 9] != 0)
        or np.any(quality[quality[:, 12] == 0, 11] != 0)
    ):
        raise RuntimeError("CCAC quality bounds/observation masks changed")
    for values in (ccac.raw_translation, ccac.compensated_translation, ccac.within_actor):
        channels = values.reshape(rows, -1, 3)
        if np.any(np.diff(channels, axis=2) < 0):
            raise RuntimeError("CCAC quantile order changed")
    if (
        np.any(ccac.raw_translation[:, 6:] < 0)
        or np.any(ccac.compensated_translation[:, 6:] < 0)
        or np.any(ccac.within_actor < 0)
    ):
        raise RuntimeError("CCAC magnitude features are negative")
    return ccac


def _fit_frozen_posture(
    data: p1.PrimaryData,
    inputs: Any,
    outer_fold: int,
    config: dict[str, Any],
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
    posture_c: float,
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    """Exactly17 fits; re-fit fixed A0 posture, select only four motion Cs."""
    if (
        posture_c not in C_VALUES
        or len(inputs.motion_references) != 1
        or len(splits) != 3
        or tuple(config["C_values"]) != C_VALUES
    ):
        raise RuntimeError("Invalid frozen A0 posture C or motion-head count")
    posture_x, motion_x = inputs.posture_or_direct, inputs.motion_references[0]
    fit_rows = np.flatnonzero(data.folds != outer_fold)
    held_rows = np.flatnonzero(data.folds == outer_fold)
    sitting_inner = np.full(len(data.labels), np.nan)
    posture_iterations = []
    for train, held in splits:
        fitted = p2.fit_logistic(
            posture_x[train],
            (data.labels[train] != 0).astype(int),
            config,
            posture_c,
            seed,
            binary=True,
        )
        sitting_inner[held] = 1 - p2._predict(fitted, posture_x[held])[:, 1]
        posture_iterations.append(int(np.max(fitted[1].n_iter_)))
    candidates = []
    for c_value in config["C_values"]:
        motion_c = float(c_value)
        motion_inner = np.full(len(data.labels), np.nan)
        iterations = []
        for train, held in splits:
            upright = train[data.labels[train] != 0]
            fitted = p2.fit_logistic(
                motion_x[upright],
                (data.labels[upright] == 2).astype(int),
                config,
                motion_c,
                seed,
                binary=True,
            )
            motion_inner[held] = p2._predict(fitted, motion_x[held])[:, 1]
            iterations.append(int(np.max(fitted[1].n_iter_)))
        decoded = p2.decode_factorized_probabilities(
            sitting_inner[fit_rows], motion_inner[fit_rows]
        )
        candidates.append(
            {
                "posture_C": posture_c,
                "motion_C": motion_c,
                "inner_metrics": p1.metrics(data.labels[fit_rows], decoded),
                "posture_inner_iterations": posture_iterations,
                "motion_reference_inner_iterations": [iterations],
            }
        )
    selected = p2.select_candidate(candidates, factorized=True)
    upright = fit_rows[data.labels[fit_rows] != 0]
    posture_fit = p2.fit_logistic(
        posture_x[fit_rows],
        (data.labels[fit_rows] != 0).astype(int),
        config,
        posture_c,
        seed,
        binary=True,
    )
    motion_fit = p2.fit_logistic(
        motion_x[upright],
        (data.labels[upright] == 2).astype(int),
        config,
        selected["motion_C"],
        seed,
        binary=True,
    )
    probabilities = p2.decode_factorized_probabilities(
        1 - p2._predict(posture_fit, posture_x[held_rows])[:, 1],
        p2._predict(motion_fit, motion_x[held_rows])[:, 1],
    )
    checkpoint = {
        **p2._checkpoint_arrays(posture_fit, "posture"),
        **p2._checkpoint_arrays(motion_fit, "motion_reference_0"),
    }
    return (
        probabilities,
        {
            "selection": selected,
            "candidates": candidates,
            "fit_rows": len(fit_rows),
            "motion_fit_rows": len(upright),
            "motion_references": 1,
            "probability_reduction": "identity",
            "posture_C_frozen_from_current_a0": posture_c,
            "final_iterations": {
                "posture": int(np.max(posture_fit[1].n_iter_)),
                "motion_reference_0": int(np.max(motion_fit[1].n_iter_)),
            },
            "feature_dimensions": {
                "posture": posture_x.shape[1],
                "motion_reference_0": motion_x.shape[1],
            },
            "unique_estimator_fits": 17,
            "estimator_fit_invocations": 17,
        },
        p1.npz_bytes(**checkpoint),
    )


@dataclass(frozen=True)
class PostFitEvidence:
    p3_long: np.ndarray
    p3_dual: np.ndarray
    p5_spatial: np.ndarray
    p5_orthogonal: np.ndarray
    p6_reference: np.ndarray


def nested_workload(
    data: p1.PrimaryData,
    moments: TokenMomentFeatures,
    ccac: CCACFeatures,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
    posture_c: float | None = None,
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    """Fit one fold; no historical predictions or outer-held labels guide fitting."""
    if arm not in ARMS:
        raise RuntimeError("Unknown P8 arm")
    inputs = arm_inputs(moments, ccac, arm)
    if any(
        not np.isfinite(value).all()
        for value in (inputs.posture_or_direct, *inputs.motion_references)
    ):
        raise RuntimeError("Nonfinite P8 feature")
    config, seed = protocol["probe"], int(protocol["seed"])
    splits = p1.grouped_inner_splits(
        data.labels,
        data.scenarios,
        data.folds,
        outer_fold,
        n_splits=int(config["inner_folds"]),
        seed=int(config["inner_seed"]),
    )
    if arm == "spatial_refit":
        if posture_c is not None:
            raise RuntimeError("A0 must independently select both Cs")
        probabilities, details, checkpoint = p4._fit_factorized(
            data, inputs, outer_fold, config, seed, splits
        )
        details["estimator_fit_invocations"] = 26
    else:
        if posture_c is None:
            raise RuntimeError("CCAC add-on lacks current-fold A0 posture selection")
        probabilities, details, checkpoint = _fit_frozen_posture(
            data, inputs, outer_fold, config, seed, splits, posture_c
        )
    p1.validate_probability_array(probabilities, int((data.folds == outer_fold).sum()))
    details.update(
        inner_selection_rows=int((data.folds != outer_fold).sum()),
        baseline_access_during_fit=False,
        historical_oof_access_during_fit=False,
        probability_fallback_rows=0,
        long_to_short_feature_fallback_rows=int((~moments.long_valid).sum()),
        ccac_short_stride_substitutions=0,
    )
    return probabilities, details, checkpoint


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p8")
    return locker.validate_lock(root, path.resolve())


def _load_data(root: Path, lock: dict[str, Any]) -> tuple[p1.PrimaryData, P8Features]:
    """Read only frozen features and derive the locked label-free CCAC tensors."""
    p2_path = p1._checked_path(root, lock["p3_reference"]["p2_execution_lock"])
    p2_lock = json.loads(p2_path.read_text(encoding="utf-8"))
    data = p1.load_primary_data(root, p2_lock)
    p1._validate_inner_map(data, p2_lock)
    p1.validate_randomization(p2_lock)
    if p1.canonical_digest(data.sample_ids.tolist()) != lock["sample_ids_sha256"]:
        raise RuntimeError("P8 sample identity changed")
    long_values: dict[str, np.ndarray] = {}
    long_validity: dict[str, np.ndarray] = {}
    for cache in lock["p3_long_caches"].values():
        arm = cache["arm"]
        long_values[arm] = np.load(
            p1._checked_path(root, cache["artifacts"][f"{arm}.npy"]),
            allow_pickle=False,
            mmap_mode="r",
        )
        long_validity[arm] = np.load(
            p1._checked_path(root, cache["artifacts"]["validity.npy"]),
            allow_pickle=False,
        )
    short_v = data.features["vjepa21_real_clip"]
    short_d = data.features["dinov2_native_frames"]
    long_v = long_values["vjepa21_long16_real_clip"]
    long_d = long_values["dinov2_long16_native_frames"]
    multiscale = derive_multiscale_features(
        short_v,
        long_v,
        short_d,
        long_d,
        long_validity["vjepa21_long16_real_clip"],
        long_validity["dinov2_long16_native_frames"],
    )
    moments = derive_token_moments(
        multiscale,
        short_v,
        long_v,
        short_d,
        long_d,
    )
    receipt = lock["full_extraction"]["receipts"]["features"]
    with np.load(p1._checked_path(root, receipt), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if len(data.labels) != 4977:
        raise RuntimeError("P8 primary cohort size changed")
    if (
        p1.canonical_digest(arrays["sample_ids"].tolist())
        != lock["full_extraction"]["sample_ids_sha256"]
    ):
        raise RuntimeError("CCAC extraction-order sample identity changed")
    ccac = _align_ccac_arrays(arrays, data.sample_ids, moments.long_valid)
    return data, P8Features(moments, ccac)


def _a0_dependency(output: Path, fold: int, request_sha: str) -> dict[str, Any]:
    directory = output / "workloads" / "spatial_refit" / f"fold-{fold}"
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "P8_WORKLOAD_COMPLETE"
        or receipt["request"]["request_sha256"] != request_sha
        or receipt["request"]["arm"] != "spatial_refit"
        or receipt["request"]["outer_fold"] != fold
    ):
        raise RuntimeError("A0 dependency belongs to another workload")
    expected = {"checkpoint.npz", "predictions.npz", "fit_started.json"}
    if (
        set(receipt.get("artifacts", {})) != expected
        or {path.name for path in directory.iterdir() if path.is_file()}
        != expected | {"request.json", "receipt.json"}
        or json.loads((directory / "request.json").read_text()) != receipt["request"]
        or receipt["request"].get("a0_dependency") is not None
    ):
        raise RuntimeError("A0 dependency inventory or request changed")
    _validate_fit_details(receipt["details"], "spatial_refit", None)
    for name, digest in receipt["artifacts"].items():
        if p1.sha256_file(directory / name) != digest:
            raise RuntimeError("A0 dependency artifact changed")
    c = receipt["details"]["selection"]["posture_C"]
    if c not in C_VALUES:
        raise RuntimeError("A0 selected undeclared posture C")
    return {
        "receipt_sha256": p1.sha256_file(receipt_path),
        "checkpoint_sha256": receipt["artifacts"]["checkpoint.npz"],
        "predictions_sha256": receipt["artifacts"]["predictions.npz"],
        "posture_C": c,
        "outer_fold": fold,
    }


def _assert_frozen_posture(
    output: Path, fold: int, probabilities: np.ndarray, checkpoint: bytes
) -> None:
    directory = output / "workloads" / "spatial_refit" / f"fold-{fold}"
    with np.load(directory / "predictions.npz", allow_pickle=False) as reference:
        if not np.array_equal(probabilities[:, 0], reference["probabilities"][:, 0]):
            raise RuntimeError("P8 sitting probabilities differ from A0")
    with np.load(directory / "checkpoint.npz", allow_pickle=False) as reference:
        with np.load(io.BytesIO(checkpoint), allow_pickle=False) as candidate:
            names = {name for name in reference.files if name.startswith("posture_")}
            if (
                names != {name for name in candidate.files if name.startswith("posture_")}
                or not names
            ):
                raise RuntimeError("P8 posture checkpoint inventory changed")
            for name in names:
                if not np.array_equal(reference[name], candidate[name]):
                    raise RuntimeError(f"P8 posture checkpoint differs from A0: {name}")


def _workload(
    data: p1.PrimaryData,
    features: P8Features,
    arm: str,
    fold: int,
    protocol: dict[str, Any],
    output: Path,
    request_sha256: str,
) -> tuple[np.ndarray, bool]:
    directory = output / "workloads" / arm / f"fold-{fold}"
    held = np.flatnonzero(data.folds == fold)
    dependency = None if arm == "spatial_refit" else _a0_dependency(output, fold, request_sha256)
    request = {
        "request_sha256": request_sha256,
        "arm": arm,
        "outer_fold": fold,
        "held_sample_ids": data.sample_ids[held].tolist(),
        "seed": int(protocol["seed"]),
        "a0_dependency": dependency,
    }
    expected_fits = 26 if arm == "spatial_refit" else 17
    started_marker = {
        "status": "P8_FITS_STARTED_NO_AUTOMATIC_RETRY",
        "request": request,
        "authorized_fit_invocations": expected_fits,
    }
    request_path, receipt_path = directory / "request.json", directory / "receipt.json"
    if receipt_path.exists():
        retained = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {"checkpoint.npz", "predictions.npz", "fit_started.json"}
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        if (
            retained.get("status") != "P8_WORKLOAD_COMPLETE"
            or retained.get("request") != request
            or set(retained.get("artifacts", {})) != expected
            or observed != expected | {"request.json", "receipt.json"}
            or json.loads(request_path.read_text()) != request
            or json.loads((directory / "fit_started.json").read_text()) != started_marker
        ):
            raise RuntimeError("Retained P8 workload identity/inventory changed")
        for name, digest in retained["artifacts"].items():
            if p1.sha256_file(directory / name) != digest:
                raise RuntimeError("Retained P8 workload artifact changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if (
                set(saved.files) != {"sample_ids", "row_indices", "probabilities"}
                or not np.array_equal(saved["sample_ids"], data.sample_ids[held])
                or not np.array_equal(saved["row_indices"], held)
            ):
                raise RuntimeError("Retained P8 prediction identity/order changed")
            probabilities = saved["probabilities"].copy()
        p1.validate_probability_array(probabilities, len(held))
        _validate_fit_details(retained["details"], arm, dependency)
        if dependency is not None:
            _assert_frozen_posture(
                output, fold, probabilities, (directory / "checkpoint.npz").read_bytes()
            )
        return probabilities, False
    if directory.exists() and any(path.name != "request.json" for path in directory.iterdir()):
        raise RuntimeError("Incomplete or started P8 workload retained; no automatic refitting")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Interrupted P8 workload belongs to another request")
    else:
        p1.atomic_json(request_path, request)
    # Exclusive start marker prevents two processes from fitting the same workload.
    with (directory / "fit_started.json").open("x", encoding="utf-8") as stream:
        json.dump(started_marker, stream, sort_keys=True, allow_nan=False)
    started = time.perf_counter()
    probabilities, details, checkpoint = nested_workload(
        data,
        features.moments,
        features.ccac,
        arm,
        fold,
        protocol,
        None if dependency is None else dependency["posture_C"],
    )
    p1.validate_probability_array(probabilities, len(held))
    _validate_fit_details(details, arm, dependency)
    if dependency is not None:
        _assert_frozen_posture(output, fold, probabilities, checkpoint)
        details["exact_a0_posture_checkpoint_and_sitting_probabilities"] = True
    p1.atomic_bytes(directory / "checkpoint.npz", checkpoint)
    p1.atomic_bytes(
        directory / "predictions.npz",
        p1.npz_bytes(
            sample_ids=data.sample_ids[held],
            row_indices=held,
            probabilities=probabilities,
        ),
    )
    p1.atomic_json(
        receipt_path,
        {
            "status": "P8_WORKLOAD_COMPLETE",
            "request": request,
            "details": details,
            "held_metrics": p1.metrics(data.labels[held], probabilities),
            "seconds": time.perf_counter() - started,
            "artifacts": {
                name: p1.sha256_file(directory / name)
                for name in (
                    "checkpoint.npz",
                    "predictions.npz",
                    "fit_started.json",
                )
            },
        },
    )
    print(f"Completed P8 {arm}/fold-{fold}", flush=True)
    return probabilities, True


def _validate_fit_details(
    details: dict[str, Any], arm: str, dependency: dict[str, Any] | None
) -> None:
    expected = 26 if arm == "spatial_refit" else 17
    candidates = details["candidates"]
    if (
        details.get("unique_estimator_fits") != expected
        or details.get("estimator_fit_invocations") != expected
        or len(candidates) != (16 if arm == "spatial_refit" else 4)
        or details["selection"] != p2.select_candidate(candidates, factorized=True)
    ):
        raise RuntimeError("P8 fit/candidate/selection audit changed")
    observed = {(row["posture_C"], row["motion_C"]) for row in candidates}
    expected_grid = {
        (posture_c, motion_c)
        for posture_c in (C_VALUES if dependency is None else (dependency["posture_C"],))
        for motion_c in C_VALUES
    }
    if observed != expected_grid:
        raise RuntimeError("P8 candidate C grid changed")
    if dependency is not None and details["selection"]["posture_C"] != dependency["posture_C"]:
        raise RuntimeError("P8 selected posture C differs from its A0 dependency")


def _read_oof(root: Path, entry: dict[str, Any]) -> dict[str, np.ndarray]:
    path = p1._checked_path(root, entry["oof"])
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _load_postfit_references(
    root: Path,
    lock: dict[str, Any],
    data: p1.PrimaryData,
    features: P8Features,
) -> PostFitEvidence:
    """Decode historical probabilities only after all P8 workloads complete."""
    arrays = {phase: _read_oof(root, entry) for phase, entry in lock["postfit_references"].items()}
    for phase, values in arrays.items():
        for name, expected in (
            ("sample_ids", data.sample_ids),
            ("recording_ids", data.scenarios),
            ("labels", data.labels),
            ("folds", data.folds),
            ("long_valid", features.long_valid),
        ):
            if name not in values or not np.array_equal(values[name], expected):
                raise RuntimeError(f"P8/{phase.upper()} OOF identity changed: {name}")
    protocol = lock["protocol"]
    p3_arrays = protocol["references"]["p3"]["post_fit_arrays"]
    p5_arrays = protocol["references"]["p5"]["post_fit_arrays"]
    p6_name = protocol["references"]["p6"]["reference_array"]
    required = (
        arrays["p3"][p3_arrays[0]],
        arrays["p3"][p3_arrays[1]],
        arrays["p5"][p5_arrays[0]],
        arrays["p5"][p5_arrays[1]],
        arrays["p6"][p6_name],
    )
    for values in required:
        if values.dtype != np.float64:
            raise RuntimeError("P8 historical OOF probability dtype changed")
        p1.validate_probability_array(values, len(data.labels))
    p6_metrics = p1.metrics(data.labels, required[-1])
    for key, declared_key in (
        ("macro_f1", "reference_macro_f1"),
        ("nll", "reference_nll"),
        ("brier", "reference_brier"),
    ):
        if p6_metrics[key] != protocol["references"]["p6"][declared_key]:
            raise RuntimeError(f"Locked P6 {key} changed")
    reconstructed = uniform_consensus([required[0], required[1], required[3]])
    if not np.array_equal(reconstructed, required[-1]):
        raise RuntimeError("P8 exact P6 component reconstruction failed")
    return PostFitEvidence(*required)


def _feature_norm(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    norms = np.linalg.norm(values[mask], axis=1)
    if not np.isfinite(norms).all():
        raise RuntimeError("P8 feature norm contains nonfinite values")
    return {
        "rows": int(mask.sum()),
        "mean_l2": float(norms.mean()),
        "median_l2": float(np.median(norms)),
        "p95_l2": float(np.quantile(norms, 0.95)),
        "max_l2": float(norms.max()),
    }


def _feature_norms(features: P8Features) -> dict[str, Any]:
    masks = {
        "all": np.ones(len(features.long_valid), dtype=bool),
        "long_valid": features.long_valid,
        "short_fallback": ~features.long_valid,
    }
    values = {
        "quality": features.ccac.quality,
        "raw_translation": features.ccac.raw_translation,
        "compensated_translation": features.ccac.compensated_translation,
        "within_actor": features.ccac.within_actor,
    }
    return {
        name: {
            stratum: _feature_norm(matrix, mask) for stratum, mask in masks.items() if mask.any()
        }
        for name, matrix in values.items()
    }


def _error_correlations(
    labels: np.ndarray, probabilities: dict[str, np.ndarray]
) -> dict[str, float]:
    names = sorted(probabilities)
    output: dict[str, float] = {}
    for index, left in enumerate(names):
        left_error = (probabilities[left].argmax(axis=1) != labels).astype(np.float64)
        for right in names[index + 1 :]:
            right_error = (probabilities[right].argmax(axis=1) != labels).astype(np.float64)
            if left_error.std() == 0 or right_error.std() == 0:
                value = float(np.array_equal(left_error, right_error))
            else:
                value = float(np.corrcoef(left_error, right_error)[0, 1])
            output[f"{left}|{right}"] = value
    return output


def _change_counts(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    candidate_prediction = candidate.argmax(axis=1)
    reference_prediction = reference.argmax(axis=1)
    candidate_correct = candidate_prediction == labels
    reference_correct = reference_prediction == labels
    rescue = mask & ~reference_correct & candidate_correct
    harm = mask & reference_correct & ~candidate_correct
    both_wrong_changed = (
        mask
        & ~reference_correct
        & ~candidate_correct
        & (candidate_prediction != reference_prediction)
    )
    return {
        "rows": int(mask.sum()),
        "reference_errors": int((mask & ~reference_correct).sum()),
        "candidate_errors": int((mask & ~candidate_correct).sum()),
        "rescues": int(rescue.sum()),
        "harms": int(harm.sum()),
        "net_correct_change": int(rescue.sum() - harm.sum()),
        "changed_but_still_wrong": int(both_wrong_changed.sum()),
    }


def _mechanism_diagnostics(
    data: p1.PrimaryData,
    features: P8Features,
    predictions: dict[str, np.ndarray],
    systems: dict[str, np.ndarray],
    evidence: PostFitEvidence,
) -> dict[str, Any]:
    labels = data.labels
    p6 = systems["p6_reference"]
    candidate = systems["ccac_replacement_triad"]
    component_predictions = np.stack(
        [
            evidence.p3_long.argmax(axis=1),
            evidence.p3_dual.argmax(axis=1),
            evidence.p5_orthogonal.argmax(axis=1),
        ]
    )
    all_three_wrong = np.all(component_predictions != labels, axis=0)
    unanimous = np.all(component_predictions == component_predictions[0], axis=0)
    p6_error = p6.argmax(axis=1) != labels
    shared = all_three_wrong & p6_error
    routable = ~all_three_wrong & p6_error
    strata = {
        "all": np.ones(len(labels), dtype=bool),
        "clear_stable": ~data.occluded & ~data.transition,
        "occluded": data.occluded,
        "clear": ~data.occluded,
        "transition": data.transition,
        "stable": ~data.transition,
        "long_valid": features.long_valid,
        "short_fallback": ~features.long_valid,
        "p6_unanimous_components": unanimous,
        "p6_disagreeing_components": ~unanimous,
        "p6_all_three_wrong": all_three_wrong,
        "p6_shared_failure_errors": shared,
        "p6_potentially_routable_errors": routable,
        "p6_shared_upright_errors": shared & (labels != 0),
        "p6_shared_motion_boundary_errors": shared & (labels != 0) & (p6.argmax(axis=1) != 0),
        "translation_valid": features.ccac.translation_valid,
        "translation_invalid": ~features.ccac.translation_valid,
        "articulation_valid": features.ccac.articulation_valid,
        "articulation_invalid": ~features.ccac.articulation_valid,
    }
    comparison_counts = {
        name: _change_counts(labels, candidate, p6, mask)
        for name, mask in strata.items()
        if mask.any()
    }
    class_names = ("sitting", "standing", "walking_running")
    candidates = {**predictions, "ccac_replacement_triad": candidate}
    expert_counts = {
        expert: {
            "by_stratum": {
                name: _change_counts(labels, values, p6, mask)
                for name, mask in strata.items()
                if mask.any()
            },
            "by_class": {
                name: _change_counts(labels, values, p6, labels == index)
                for index, name in enumerate(class_names)
            },
        }
        for expert, values in candidates.items()
    }
    return {
        "replacement_vs_p6_change_counts": comparison_counts,
        "p6_component_partition": {
            "all_three_wrong_rows": int(all_three_wrong.sum()),
            "p6_errors_all_three_wrong": int(shared.sum()),
            "p6_errors_any_component_correct": int(routable.sum()),
            "shared_failure_rescues": int((shared & (candidate.argmax(axis=1) == labels)).sum()),
            "potentially_routable_rescues": int(
                (routable & (candidate.argmax(axis=1) == labels)).sum()
            ),
        },
        "all_experts_vs_p6_change_counts": expert_counts,
        "feature_norms": _feature_norms(features),
    }


def _model_result(
    data: p1.PrimaryData,
    values: np.ndarray,
    long_valid: np.ndarray,
    *,
    model_fits: int,
    fallback_rows: int,
) -> dict[str, Any]:
    strata = (
        ("occluded", data.occluded),
        ("clear", ~data.occluded),
        ("transition", data.transition),
        ("stable", ~data.transition),
        ("clear_stable", ~data.occluded & ~data.transition),
        ("long_valid", long_valid),
        ("short_fallback", ~long_valid),
    )
    return {
        "metrics": p1.metrics(data.labels, values),
        "model_fits": model_fits,
        "feature_fallback_rows": fallback_rows,
        "probability_fallback_rows": 0,
        "per_scenario": {
            str(group): p1.metrics(data.labels[mask], values[mask])
            for group in np.unique(data.scenarios)
            for mask in [data.scenarios == group]
        },
        "per_fold": {
            str(fold): p1.metrics(data.labels[mask], values[mask])
            for fold in np.unique(data.folds)
            for mask in [data.folds == fold]
        },
        "diagnostic_strata": {
            name: p1.metrics(data.labels[mask], values[mask]) for name, mask in strata if mask.any()
        },
    }


def summarize(
    data: p1.PrimaryData,
    features: P8Features,
    predictions: dict[str, np.ndarray],
    systems: dict[str, np.ndarray],
    evidence: PostFitEvidence,
    protocol: dict[str, Any],
    model_fits: int,
) -> dict[str, Any]:
    all_probabilities = {**predictions, **systems}
    fallback_rows = int((~features.long_valid).sum())
    fits_per_arm = {arm: 130 if arm == "spatial_refit" else 85 for arm in ARMS}
    models = {
        arm: _model_result(
            data,
            values,
            features.long_valid,
            model_fits=fits_per_arm[arm],
            fallback_rows=fallback_rows,
        )
        for arm, values in predictions.items()
    }
    models.update(
        {
            name: _model_result(
                data,
                values,
                features.long_valid,
                model_fits=0,
                fallback_rows=fallback_rows,
            )
            for name, values in systems.items()
        }
    )
    quality_strata = (
        ("translation_valid", features.ccac.translation_valid),
        ("translation_invalid", ~features.ccac.translation_valid),
        ("articulation_valid", features.ccac.articulation_valid),
        ("articulation_invalid", ~features.ccac.articulation_valid),
    )
    for name, values in all_probabilities.items():
        models[name]["diagnostic_strata"].update(
            {
                stratum: p1.metrics(data.labels[mask], values[mask])
                for stratum, mask in quality_strata
                if mask.any()
            }
        )
    statistics = protocol["statistics"]
    strata = (
        ("occluded", data.occluded),
        ("clear", ~data.occluded),
        ("transition", data.transition),
        ("stable", ~data.transition),
        ("clear_stable", ~data.occluded & ~data.transition),
        ("long_valid", features.long_valid),
        ("short_fallback", ~features.long_valid),
    )
    contrasts: dict[str, Any] = {}
    for item in statistics["comparisons"]:
        candidate, reference = item["candidate"], item["reference"]
        name = f"{candidate}_vs_{reference}"
        contrasts[name] = p1.paired_statistics(
            data.labels,
            all_probabilities[candidate],
            all_probabilities[reference],
            data.scenarios,
            bootstrap_resamples=int(statistics["bootstrap_resamples"]),
            bootstrap_seed=int(statistics["bootstrap_seed"]),
        )
        contrasts[name]["diagnostic_strata"] = {
            stratum: p2.subgroup_statistics(
                data.labels,
                all_probabilities[candidate],
                all_probabilities[reference],
                data.scenarios,
                mask,
                bootstrap_resamples=int(statistics["bootstrap_resamples"]),
                bootstrap_seed=int(statistics["bootstrap_seed"]),
            )
            for stratum, mask in strata
            if mask.any()
        }
    family = sorted(contrasts)
    adjusted = p1.holm_adjust(
        {name: contrasts[name]["one_sided_exact_swap_pvalue"] for name in family}
    )
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_one_sided_pvalue"] = value
    diagnostics = _mechanism_diagnostics(data, features, predictions, systems, evidence)
    candidate = models["ccac_replacement_triad"]
    reference = models["p6_reference"]
    scenario_deltas = {
        group: candidate["per_scenario"][group]["macro_f1"]
        - reference["per_scenario"][group]["macro_f1"]
        for group in reference["per_scenario"]
    }
    changes = diagnostics["replacement_vs_p6_change_counts"]
    guardrails = protocol["engineering_guardrails"]
    guardrail_results = {
        "macro_f1_at_least_0_835": candidate["metrics"]["macro_f1"]
        >= float(guardrails["primary_system_macro_f1_at_least"]),
        "nll_strictly_below_p6": candidate["metrics"]["nll"] < reference["metrics"]["nll"],
        "brier_strictly_below_p6": candidate["metrics"]["brier"] < reference["metrics"]["brier"],
        "at_least_7_scenarios_improved": sum(value > 0 for value in scenario_deltas.values())
        >= int(guardrails["minimum_scenarios_improved_over_p6"]),
        "no_scenario_decline_over_0_02": min(scenario_deltas.values())
        >= -float(guardrails["maximum_scenario_macro_f1_decline"]),
        "positive_net_correct_overall": changes["all"]["net_correct_change"] > 0,
        "positive_net_correct_clear_stable": changes["clear_stable"]["net_correct_change"] > 0,
    }
    primary_macro = candidate["metrics"]["macro_f1"]
    engineering_passed = all(guardrail_results.values())
    primary_holm = contrasts["ccac_replacement_triad_vs_p6_reference"][
        "holm_adjusted_one_sided_pvalue"
    ]
    return {
        "status": STATUS,
        "method": protocol["method_name"],
        "rows": len(data.labels),
        "scenarios": len(np.unique(data.scenarios)),
        "seed": int(protocol["seed"]),
        "baseline": p1.metrics(data.labels, data.baseline),
        "models": models,
        "contrasts": contrasts,
        "holm_family": family,
        "a0_exact_p5_spatial_reproduction": True,
        "p6_exact_component_reconstruction": True,
        "all_addon_posture_checkpoints_and_sitting_probabilities_exact_a0": True,
        "estimator_fit_invocations": model_fits,
        "fit_count_semantics": protocol["probe"]["fit_count_semantics"],
        "ccac_translation_valid_rows": int(features.ccac.translation_valid.sum()),
        "ccac_articulation_valid_rows": int(features.ccac.articulation_valid.sum()),
        "primary_expert": protocol["primary_expert"],
        "primary_system": "ccac_replacement_triad",
        "primary_system_macro_f1": primary_macro,
        "primary_system_delta_vs_p6": primary_macro - reference["metrics"]["macro_f1"],
        "scenario_macro_f1_deltas_vs_p6": scenario_deltas,
        "scenarios_improved_over_p6": sum(value > 0 for value in scenario_deltas.values()),
        "guardrail_results": guardrail_results,
        "all_engineering_guardrails_passed": all(guardrail_results.values()),
        "primary_system_holm_significant_at_0_05": primary_holm <= 0.05,
        "decision": (
            "CCAC_ENGINEERING_SCREEN_PASS_REQUIRES_INDEPENDENT_CONFIRMATION"
            if engineering_passed
            else "RETAIN_P6_CCAC_DID_NOT_PASS_ALL_ENGINEERING_GATES"
        ),
        "independent_confirmation_established": False,
        "milestone_0_84_crossed": primary_macro >= float(guardrails["milestone_macro_f1"]),
        "stretch_0_85_crossed": primary_macro >= float(guardrails["stretch_macro_f1"]),
        "mechanism_diagnostics": diagnostics,
        "error_correlations": _error_correlations(data.labels, all_probabilities),
        "long_valid_rows": int(features.long_valid.sum()),
        "short_fallback_rows": fallback_rows,
        "model_fits": model_fits,
        "maximum_authorized_model_fits": int(protocol["probe"]["maximum_unique_estimator_fits"]),
        "workloads": int(protocol["probe"]["maximum_workloads"]),
        "adaptation_disclosure": protocol["adaptation_disclosure"],
        "scope": "adaptive_same_development_scenarios_not_independent_confirmation",
        "historical_reference_predictions_used_for_fitting": False,
        "labels_used_in_feature_derivation": False,
        "learned_fusion": False,
        "raw_images_read": 0,
        "protected_rows_read": 0,
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    locker = importlib.import_module("tools.lock_okutama_video_p8")
    locker.validate_spec(protocol)


def _model_fit_audit(output: Path) -> int:
    total = 0
    for arm in ARMS:
        for fold in range(5):
            path = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            dependency = receipt["request"]["a0_dependency"]
            _validate_fit_details(receipt["details"], arm, dependency)
            total += int(receipt["details"]["estimator_fit_invocations"])
    return total


def _validate_output(
    output: Path, summary: dict[str, Any], request_sha: str, *, published: bool = True
) -> None:
    if summary.get("status") != STATUS or summary.get("request_sha256") != request_sha:
        raise RuntimeError("Retained P8 summary belongs to another request")
    expected = {
        "oof_probabilities.npz",
        "metrics.csv",
        "paired_statistics.json",
        "diagnostics.json",
    }
    expected.update(
        f"workloads/{arm}/fold-{fold}/{name}"
        for arm in ARMS
        for fold in range(5)
        for name in (
            "request.json",
            "receipt.json",
            "checkpoint.npz",
            "predictions.npz",
            "fit_started.json",
        )
    )
    if set(summary.get("artifacts", {})) != expected:
        raise RuntimeError("Retained P8 aggregate inventory changed")
    for name, digest in summary["artifacts"].items():
        if p1.sha256_file(output / name) != digest:
            raise RuntimeError("Retained P8 artifact bytes changed")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    markers = {"request.json", "summary.json", "completion.json"} if published else {"request.json"}
    if observed != expected | markers:
        raise RuntimeError("Unexpected files in completed P8 output")
    if published:
        completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
        if completion != {
            "status": "P8_PUBLICATION_COMPLETE",
            "request_sha256": request_sha,
            "summary_sha256": p1.sha256_file(output / "summary.json"),
        }:
            raise RuntimeError("Retained P8 summary receipt changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P8 output must use a dedicated directory below .runs")
    if args.max_new_workloads is not None and args.max_new_workloads < 1:
        raise ValueError("max_new_workloads must be positive")
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    _validate_protocol(protocol)
    request = {
        "status": "P8_REQUEST_BEFORE_FEATURE_DERIVATION_OR_FITTING",
        "lock_sha256": p1.sha256_file(args.protocol_lock.resolve()),
        "protocol": protocol,
        "sample_ids_sha256": lock["sample_ids_sha256"],
        "source_sha256": p1.sha256_file(Path(__file__)),
        "output": str(output),
        "blas_thread_limit": 1,
        "named_p3_p5_p6_oof_access_before_all_fits": False,
        "ccac_features_sha256": lock["full_extraction"]["receipts"]["features"]["sha256"],
        "legacy_baseline_loaded_read_only_but_not_used_for_fitting": True,
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("P8 output belongs to another locked request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty P8 output lacks its matching request")
    else:
        p1.atomic_json(request_path, request)
    request_sha = p1.canonical_digest(request)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        _validate_output(output, summary, request_sha)
        return summary
    aggregate_names = (
        "oof_probabilities.npz",
        "metrics.csv",
        "paired_statistics.json",
        "diagnostics.json",
    )
    if any((output / name).exists() for name in aggregate_names):
        raise RuntimeError("Incomplete P8 aggregate evidence retained; refusing overwrite")
    data, features = _load_data(root, lock)
    predictions = {arm: np.full((len(data.labels), 3), np.nan, dtype=np.float64) for arm in ARMS}
    new_workloads = 0
    for arm in ARMS:
        for fold in range(5):
            receipt_path = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            if (
                args.max_new_workloads is not None
                and new_workloads >= args.max_new_workloads
                and not receipt_path.exists()
            ):
                return {"status": "P8_BOUNDED_RUN_PAUSED", "new_workloads": new_workloads}
            values, fitted = _workload(data, features, arm, fold, protocol, output, request_sha)
            predictions[arm][data.folds == fold] = values
            new_workloads += int(fitted)
    for values in predictions.values():
        p1.validate_probability_array(values, len(data.labels))
    model_fits = _model_fit_audit(output)
    if model_fits != int(protocol["probe"]["maximum_unique_estimator_fits"]):
        raise RuntimeError("P8 estimator-fit audit changed")
    _load_lock(root, args.protocol_lock)
    evidence = _load_postfit_references(root, lock, data, features)
    if not np.array_equal(predictions["spatial_refit"], evidence.p5_spatial):
        raise RuntimeError("P8 A0 failed exact P5 spatial OOF reproduction")
    systems = {
        "p6_reference": evidence.p6_reference,
        "ccac_replacement_triad": uniform_consensus(
            [
                evidence.p3_long,
                evidence.p3_dual,
                predictions["ccac_full"],
            ]
        ),
    }
    summary = summarize(
        data,
        features,
        predictions,
        systems,
        evidence,
        protocol,
        model_fits,
    )
    p1.atomic_bytes(
        output / "oof_probabilities.npz",
        p1.npz_bytes(
            sample_ids=data.sample_ids,
            recording_ids=data.scenarios,
            labels=data.labels,
            folds=data.folds,
            long_valid=features.long_valid,
            ccac_translation_valid=features.ccac.translation_valid,
            ccac_articulation_valid=features.ccac.articulation_valid,
            baseline_probabilities=data.baseline,
            p3_long_vjepa_mean=evidence.p3_long,
            p3_dual_scale_vjepa_dino=evidence.p3_dual,
            p5_spatial_contrast_factorized=evidence.p5_spatial,
            p5_orthogonal_moments_factorized=evidence.p5_orthogonal,
            **predictions,
            **systems,
        ),
    )
    table = io.StringIO(newline="")
    writer = csv.DictWriter(
        table,
        fieldnames=[
            "model",
            "macro_f1",
            "accuracy",
            "nll",
            "brier",
            "model_fits",
            "feature_fallback_rows",
        ],
    )
    writer.writeheader()
    for name, result in summary["models"].items():
        writer.writerow(
            {
                "model": name,
                "model_fits": result["model_fits"],
                "feature_fallback_rows": result["feature_fallback_rows"],
                **{key: result["metrics"][key] for key in ("macro_f1", "accuracy", "nll", "brier")},
            }
        )
    p1.atomic_bytes(output / "metrics.csv", table.getvalue().encode())
    p1.atomic_json(
        output / "paired_statistics.json",
        {"contrasts": summary["contrasts"], "holm_family": summary["holm_family"]},
    )
    p1.atomic_json(output / "diagnostics.json", summary["mechanism_diagnostics"])
    inventory = [output / name for name in aggregate_names]
    inventory.extend(sorted(path for path in (output / "workloads").rglob("*") if path.is_file()))
    summary.update(
        workload_audit={
            arm: {
                str(fold): {key: receipt[key] for key in ("seconds", "details")}
                for fold in range(5)
                for receipt in [
                    json.loads(
                        (output / "workloads" / arm / f"fold-{fold}" / "receipt.json").read_text()
                    )
                ]
            }
            for arm in ARMS
        },
        request_sha256=request_sha,
        artifacts={path.relative_to(output).as_posix(): p1.sha256_file(path) for path in inventory},
    )
    _validate_output(output, summary, request_sha, published=False)
    p1.atomic_json(summary_path, summary)
    p1.atomic_json(
        output / "completion.json",
        {
            "status": "P8_PUBLICATION_COMPLETE",
            "request_sha256": request_sha,
            "summary_sha256": p1.sha256_file(summary_path),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-workloads", type=int)
    result = run(parser.parse_args())
    concise = {
        "status": result["status"],
        "rows": result.get("rows"),
        "models": {name: values["metrics"] for name, values in result.get("models", {}).items()},
        "guardrail_results": result.get("guardrail_results", {}),
    }
    print(json.dumps(concise, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
