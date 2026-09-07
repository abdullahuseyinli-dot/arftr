"""Run the locked P4 visual--kinematic camera-reference experiment."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_video_p3 as p3

from hac.video_kinematic import ARMS, ArmInputs, KinematicFeatures, arm_inputs
from hac.video_multiscale import ARMS as P3_ARMS
from hac.video_multiscale import derive_multiscale_features

p2, p1 = p3.p2, p3.p1
C_VALUES = (1e-5, 1e-4, 1e-3, 1e-2)


def _fit_direct(
    data: p1.PrimaryData,
    inputs: ArmInputs,
    outer_fold: int,
    config: dict[str, Any],
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    x = inputs.posture_or_direct
    fit_rows = np.flatnonzero(data.folds != outer_fold)
    held_rows = np.flatnonzero(data.folds == outer_fold)
    candidates = []
    for c_value in config["C_values"]:
        c = float(c_value)
        inner = np.full((len(data.labels), 3), np.nan)
        iterations = []
        for train, held in splits:
            fitted = p2.fit_logistic(
                x[train], data.labels[train], config, c, seed, binary=False
            )
            inner[held] = p2._predict(fitted, x[held])
            iterations.append(int(np.max(fitted[1].n_iter_)))
        candidates.append(
            {
                "C": c,
                "inner_metrics": p1.metrics(data.labels[fit_rows], inner[fit_rows]),
                "inner_iterations": iterations,
            }
        )
    selected = p2.select_candidate(candidates, factorized=False)
    fitted = p2.fit_logistic(
        x[fit_rows], data.labels[fit_rows], config, selected["C"], seed, binary=False
    )
    probabilities = p2._predict(fitted, x[held_rows])
    return probabilities, {
        "selection": selected,
        "candidates": candidates,
        "fit_rows": len(fit_rows),
        "motion_fit_rows": None,
        "final_iterations": {"multinomial": int(np.max(fitted[1].n_iter_))},
        "feature_dimensions": {"multinomial": x.shape[1]},
        "unique_estimator_fits": 13,
    }, p1.npz_bytes(**p2._checkpoint_arrays(fitted, "multinomial"))


def _fit_factorized(
    data: p1.PrimaryData,
    inputs: ArmInputs,
    outer_fold: int,
    config: dict[str, Any],
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    posture_x, motion_xs = inputs.posture_or_direct, inputs.motion_references
    if len(motion_xs) not in (1, 2):
        raise RuntimeError("P4 factorized arm has an undeclared reference count")
    fit_rows = np.flatnonzero(data.folds != outer_fold)
    held_rows = np.flatnonzero(data.folds == outer_fold)
    posture_predictions: dict[float, np.ndarray] = {}
    motion_predictions: dict[float, np.ndarray] = {}
    posture_iterations: dict[float, list[int]] = {}
    motion_iterations: dict[float, list[list[int]]] = {}
    for c_value in config["C_values"]:
        c = float(c_value)
        posture_predictions[c] = np.full(len(data.labels), np.nan)
        motion_predictions[c] = np.full(len(data.labels), np.nan)
        posture_iterations[c] = []
        motion_iterations[c] = [[] for _ in motion_xs]
        for train, held in splits:
            upright = train[data.labels[train] != 0]
            posture_fit = p2.fit_logistic(
                posture_x[train],
                (data.labels[train] != 0).astype(int),
                config,
                c,
                seed,
                binary=True,
            )
            reference_values = []
            for reference_index, motion_x in enumerate(motion_xs):
                motion_fit = p2.fit_logistic(
                    motion_x[upright],
                    (data.labels[upright] == 2).astype(int),
                    config,
                    c,
                    seed,
                    binary=True,
                )
                reference_values.append(p2._predict(motion_fit, motion_x[held])[:, 1])
                motion_iterations[c][reference_index].append(
                    int(np.max(motion_fit[1].n_iter_))
                )
            posture_predictions[c][held] = 1 - p2._predict(
                posture_fit, posture_x[held]
            )[:, 1]
            motion_predictions[c][held] = np.mean(reference_values, axis=0)
            posture_iterations[c].append(int(np.max(posture_fit[1].n_iter_)))

    candidates = []
    for posture_c_value in config["C_values"]:
        for motion_c_value in config["C_values"]:
            posture_c, motion_c = float(posture_c_value), float(motion_c_value)
            decoded = p2.decode_factorized_probabilities(
                posture_predictions[posture_c][fit_rows],
                motion_predictions[motion_c][fit_rows],
            )
            candidates.append(
                {
                    "posture_C": posture_c,
                    "motion_C": motion_c,
                    "inner_metrics": p1.metrics(data.labels[fit_rows], decoded),
                    "posture_inner_iterations": posture_iterations[posture_c],
                    "motion_reference_inner_iterations": motion_iterations[motion_c],
                }
            )
    selected = p2.select_candidate(candidates, factorized=True)
    upright = fit_rows[data.labels[fit_rows] != 0]
    posture_fit = p2.fit_logistic(
        posture_x[fit_rows],
        (data.labels[fit_rows] != 0).astype(int),
        config,
        selected["posture_C"],
        seed,
        binary=True,
    )
    motion_fits = [
        p2.fit_logistic(
            motion_x[upright],
            (data.labels[upright] == 2).astype(int),
            config,
            selected["motion_C"],
            seed,
            binary=True,
        )
        for motion_x in motion_xs
    ]
    sitting = 1 - p2._predict(posture_fit, posture_x[held_rows])[:, 1]
    moving = np.mean(
        [
            p2._predict(fitted, motion_x[held_rows])[:, 1]
            for fitted, motion_x in zip(motion_fits, motion_xs, strict=True)
        ],
        axis=0,
    )
    probabilities = p2.decode_factorized_probabilities(sitting, moving)
    checkpoint = p2._checkpoint_arrays(posture_fit, "posture")
    for index, fitted in enumerate(motion_fits):
        checkpoint.update(p2._checkpoint_arrays(fitted, f"motion_reference_{index}"))
    return probabilities, {
        "selection": selected,
        "candidates": candidates,
        "fit_rows": len(fit_rows),
        "motion_fit_rows": len(upright),
        "motion_references": len(motion_xs),
        "probability_reduction": (
            "identity" if len(motion_xs) == 1 else "unweighted_arithmetic_mean"
        ),
        "final_iterations": {
            "posture": int(np.max(posture_fit[1].n_iter_)),
            **{
                f"motion_reference_{index}": int(np.max(fitted[1].n_iter_))
                for index, fitted in enumerate(motion_fits)
            },
        },
        "feature_dimensions": {
            "posture": posture_x.shape[1],
            **{
                f"motion_reference_{index}": value.shape[1]
                for index, value in enumerate(motion_xs)
            },
        },
        "unique_estimator_fits": 26 if len(motion_xs) == 1 else 39,
    }, p1.npz_bytes(**checkpoint)


def nested_workload(
    data: p1.PrimaryData,
    features: KinematicFeatures,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    if arm not in ARMS:
        raise RuntimeError("Unknown P4 arm")
    inputs = arm_inputs(features, arm)
    matrices = (inputs.posture_or_direct, *inputs.motion_references)
    if any(not np.isfinite(value).all() for value in matrices):
        raise RuntimeError("P4 fixed feature contains nonfinite values")
    config, seed = protocol["probe"], int(protocol["seed"])
    splits = p1.grouped_inner_splits(
        data.labels,
        data.scenarios,
        data.folds,
        outer_fold,
        n_splits=int(config["inner_folds"]),
        seed=int(config["inner_seed"]),
    )
    if inputs.kind == "multinomial":
        probabilities, details, checkpoint = _fit_direct(
            data, inputs, outer_fold, config, seed, splits
        )
    else:
        probabilities, details, checkpoint = _fit_factorized(
            data, inputs, outer_fold, config, seed, splits
        )
    p1.validate_probability_array(
        probabilities, int((data.folds == outer_fold).sum())
    )
    details.update(
        inner_selection_rows=int((data.folds != outer_fold).sum()),
        baseline_access_during_fit=False,
        p3_oof_access_during_fit=False,
        probability_fallback_rows=0,
    )
    return probabilities, details, checkpoint


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p4_probe")
    return locker.validate_probe_lock(root, path.resolve())


def _load_data(
    root: Path, lock: dict[str, Any]
) -> tuple[p1.PrimaryData, KinematicFeatures, np.ndarray]:
    p2_path = p1._checked_path(root, lock["p3_reference"]["p2_execution_lock"])
    p2_lock = json.loads(p2_path.read_text(encoding="utf-8"))
    data = p1.load_primary_data(root, p2_lock)
    p1._validate_inner_map(data, p2_lock)
    p1.validate_randomization(p2_lock)
    long_values, long_validity = {}, {}
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
    multiscale = derive_multiscale_features(
        data.features["vjepa21_real_clip"],
        long_values["vjepa21_long16_real_clip"],
        data.features["dinov2_native_frames"],
        long_values["dinov2_long16_native_frames"],
        long_validity["vjepa21_long16_real_clip"],
        long_validity["dinov2_long16_native_frames"],
    )
    path = p1._checked_path(root, lock["features"]["artifacts"]["features.npz"])
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {
            "sample_ids",
            "legacy_static_features",
            "legacy_quality_features",
            "raw_sequence",
            "raw_summary",
            "compensated_sequence",
            "compensated_summary",
            "camera_quality",
        } or not np.array_equal(source["sample_ids"], data.sample_ids):
            raise RuntimeError("P4 feature identity/member inventory changed")
        values = {name: source[name] for name in source.files if name != "sample_ids"}
    from hac.video_kinematic import derive_kinematic_features

    features = derive_kinematic_features(multiscale, **values)
    return data, features, multiscale.long_valid


def _workload(
    data: p1.PrimaryData,
    features: KinematicFeatures,
    arm: str,
    fold: int,
    protocol: dict[str, Any],
    output: Path,
    request_sha256: str,
) -> tuple[np.ndarray, bool]:
    directory = output / "workloads" / arm / f"fold-{fold}"
    held = np.flatnonzero(data.folds == fold)
    request = {
        "request_sha256": request_sha256,
        "arm": arm,
        "outer_fold": fold,
        "held_sample_ids": data.sample_ids[held].tolist(),
        "seed": int(protocol["seed"]),
    }
    request_path, receipt_path = directory / "request.json", directory / "receipt.json"
    if receipt_path.exists():
        retained = json.loads(receipt_path.read_text(encoding="utf-8"))
        if retained.get("status") != "P4_WORKLOAD_COMPLETE" or retained.get("request") != request:
            raise RuntimeError("Retained P4 workload request/status changed")
        expected = {"checkpoint.npz", "predictions.npz"}
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        if set(retained.get("artifacts", {})) != expected or observed != expected | {
            "request.json",
            "receipt.json",
        }:
            raise RuntimeError("Retained P4 workload inventory changed")
        if not request_path.exists() or json.loads(request_path.read_text()) != request:
            raise RuntimeError("Retained P4 pre-fit request changed")
        for name, digest in retained["artifacts"].items():
            if p1.sha256_file(directory / name) != digest:
                raise RuntimeError("Retained P4 workload artifact changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], data.sample_ids[held]) or not np.array_equal(
                saved["row_indices"], held
            ):
                raise RuntimeError("Retained P4 prediction identity/order changed")
            probabilities = saved["probabilities"].copy()
        p1.validate_probability_array(probabilities, len(held))
        return probabilities, False
    if directory.exists() and any(path.name != "request.json" for path in directory.iterdir()):
        raise RuntimeError("Incomplete P4 workload evidence retained; refusing overwrite")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Interrupted P4 workload belongs to another request")
    else:
        p1.atomic_json(request_path, request)
    started = time.perf_counter()
    probabilities, details, checkpoint = nested_workload(
        data, features, arm, fold, protocol
    )
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
            "status": "P4_WORKLOAD_COMPLETE",
            "request": request,
            "details": details,
            "held_metrics": p1.metrics(data.labels[held], probabilities),
            "seconds": time.perf_counter() - started,
            "artifacts": {
                name: p1.sha256_file(directory / name)
                for name in ("checkpoint.npz", "predictions.npz")
            },
        },
    )
    print(f"Completed P4 {arm}/fold-{fold}", flush=True)
    return probabilities, True


def _load_p3_reference(
    root: Path, lock: dict[str, Any], data: p1.PrimaryData
) -> np.ndarray:
    path = p1._checked_path(root, lock["p3_reference"]["oof"])
    with np.load(path, allow_pickle=False) as source:
        expected_members = {
            "sample_ids",
            "recording_ids",
            "labels",
            "folds",
            "long_valid",
            "baseline_probabilities",
            "p2a_best_probabilities",
            *P3_ARMS,
        }
        if set(source.files) != expected_members:
            raise RuntimeError("P3 comparison member inventory changed")
        for name, expected in (
            ("sample_ids", data.sample_ids),
            ("recording_ids", data.scenarios),
            ("labels", data.labels),
            ("folds", data.folds),
        ):
            if not np.array_equal(source[name], expected):
                raise RuntimeError("P3 comparison identity/order changed")
        values = source[lock["p3_reference"]["arm"]].astype(np.float64)
    p1.validate_probability_array(values, len(data.labels))
    if p1.metrics(data.labels, values)["macro_f1"] != lock["p3_reference"]["macro_f1"]:
        raise RuntimeError("P3 comparison metric changed")
    return values


def _strata(
    data: p1.PrimaryData, features: KinematicFeatures, long_valid: np.ndarray
) -> tuple[tuple[str, np.ndarray], ...]:
    quality = features.camera_quality_mean
    edges = np.quantile(quality, (0.25, 0.5, 0.75))
    bins = np.searchsorted(edges, quality, side="right")
    base = [
        ("occluded", data.occluded),
        ("clear", ~data.occluded),
        ("transition", data.transition),
        ("stable", ~data.transition),
        ("long_valid", long_valid),
        ("long_invalid_short_anchor", ~long_valid),
    ]
    base.extend((f"camera_quality_q{index + 1}", bins == index) for index in range(4))
    return tuple(base)


def summarize(
    data: p1.PrimaryData,
    features: KinematicFeatures,
    long_valid: np.ndarray,
    probabilities: dict[str, np.ndarray],
    protocol: dict[str, Any],
    p3_reference: np.ndarray,
) -> dict[str, Any]:
    references = {**probabilities, "baseline": data.baseline, "p3_best": p3_reference}
    strata = _strata(data, features, long_valid)
    models = {
        arm: {
            "metrics": p1.metrics(data.labels, values),
            "probability_fallback_rows": 0,
            "per_scenario": {
                str(group): p1.metrics(data.labels[mask], values[mask])
                for group in np.unique(data.scenarios)
                for mask in [data.scenarios == group]
            },
            "diagnostic_strata": {
                name: p1.metrics(data.labels[mask], values[mask])
                for name, mask in strata
                if mask.any()
            },
        }
        for arm, values in probabilities.items()
    }
    contrasts = {}
    statistics = protocol["statistics"]
    for item in statistics["comparisons"]:
        candidate, reference = item["candidate"], item["reference"]
        name = f"{candidate}_vs_{reference}"
        contrasts[name] = p1.paired_statistics(
            data.labels,
            probabilities[candidate],
            references[reference],
            data.scenarios,
            bootstrap_resamples=int(statistics["bootstrap_resamples"]),
            bootstrap_seed=int(statistics["bootstrap_seed"]),
        )
        contrasts[name]["diagnostic_strata"] = {
            stratum: p2.subgroup_statistics(
                data.labels,
                probabilities[candidate],
                references[reference],
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
    all_predictions = {**probabilities, "baseline": data.baseline, "p3_best": p3_reference}
    camera_edges = np.quantile(features.camera_quality_mean, (0.25, 0.5, 0.75))
    return {
        "status": "OKUTAMA_VIDEO_P4_ADAPTIVE_CROSSFIT_COMPLETE",
        "rows": len(data.labels),
        "scenarios": len(np.unique(data.scenarios)),
        "seed": int(protocol["seed"]),
        "baseline": p1.metrics(data.labels, data.baseline),
        "p3_best_reference": p1.metrics(data.labels, p3_reference),
        "models": models,
        "contrasts": contrasts,
        "holm_family": family,
        "error_correlations": p3._error_correlations(data.labels, all_predictions),
        "camera_quality_quartile_edges": camera_edges.tolist(),
        "long_valid_rows": int(long_valid.sum()),
        "short_anchor_rows": int((~long_valid).sum()),
        "breakthrough_target": float(statistics["breakthrough_target"]),
        "target_crossed_by": [
            arm for arm, result in models.items() if result["metrics"]["macro_f1"] >= 0.82
        ],
        "scope": "adaptive_same_development_scenarios_not_independent_confirmation",
        "reference_predictions_used_for_fitting": False,
        "raw_images_read": 0,
        "protected_rows_read": 0,
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    locker = importlib.import_module("tools.lock_okutama_video_p4_materialization")
    probe = protocol["probe"]
    if (
        tuple(protocol["arms"]) != ARMS
        or int(protocol["seed"]) != 42
        or tuple(probe["C_values"]) != C_VALUES
        or probe["solver"] != "lbfgs"
        or probe["class_weight"] != "balanced"
        or int(probe["max_iter"]) != 2000
        or float(probe["tolerance"]) != 0.0001
        or int(probe["inner_folds"]) != 3
        or int(probe["inner_seed"]) != 42
        or not probe["standardize"]
        or not probe["inner_shuffle"]
        or int(probe["maximum_unique_estimator_fits"]) != 910
        or protocol["statistics"]["comparisons"] != locker.expected_comparisons()
    ):
        raise RuntimeError("P4 protocol fitting contract changed")


def _validate_output(
    output: Path, summary: dict[str, Any], request_sha: str, *, published: bool = True
) -> None:
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P4_ADAPTIVE_CROSSFIT_COMPLETE"
        or summary.get("request_sha256") != request_sha
    ):
        raise RuntimeError("Retained P4 summary belongs to another request")
    expected = {"oof_probabilities.npz", "metrics.csv", "paired_statistics.json"}
    expected.update(
        f"workloads/{arm}/fold-{fold}/{name}"
        for arm in ARMS
        for fold in range(5)
        for name in ("request.json", "receipt.json", "checkpoint.npz", "predictions.npz")
    )
    if set(summary.get("artifacts", {})) != expected:
        raise RuntimeError("Retained P4 aggregate inventory changed")
    for name, digest in summary["artifacts"].items():
        if p1.sha256_file(output / name) != digest:
            raise RuntimeError("Retained P4 artifact bytes changed")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    markers = {"request.json", "summary.json"} if published else {"request.json"}
    if observed != expected | markers:
        raise RuntimeError("Unexpected files in completed P4 output")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P4 output must use a dedicated directory below .runs")
    if args.max_new_workloads is not None and args.max_new_workloads < 1:
        raise ValueError("max_new_workloads must be positive")
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    _validate_protocol(protocol)
    data, features, long_valid = _load_data(root, lock)
    request = {
        "status": "P4_REQUEST_BEFORE_FITTING",
        "lock_sha256": p1.sha256_file(args.protocol_lock.resolve()),
        "protocol": protocol,
        "sample_ids_sha256": p1.canonical_digest(data.sample_ids.tolist()),
        "source_sha256": p1.sha256_file(Path(__file__)),
        "output": str(output),
        "blas_thread_limit": 1,
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("P4 output belongs to another locked request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty P4 output lacks its matching request")
    else:
        p1.atomic_json(request_path, request)
    request_sha = p1.canonical_digest(request)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        _validate_output(output, summary, request_sha)
        return summary
    if any(
        (output / name).exists()
        for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")
    ):
        raise RuntimeError("Incomplete P4 aggregate evidence retained; refusing overwrite")
    predictions = {arm: np.full((len(data.labels), 3), np.nan) for arm in ARMS}
    new_workloads = 0
    for arm in ARMS:
        for fold in range(5):
            receipt_path = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            if (
                args.max_new_workloads is not None
                and new_workloads >= args.max_new_workloads
                and not receipt_path.exists()
            ):
                return {"status": "P4_BOUNDED_RUN_PAUSED", "new_workloads": new_workloads}
            values, fitted = _workload(
                data, features, arm, fold, protocol, output, request_sha
            )
            predictions[arm][data.folds == fold] = values
            new_workloads += int(fitted)
    for values in predictions.values():
        p1.validate_probability_array(values, len(data.labels))
    _load_lock(root, args.protocol_lock)
    p3_reference = _load_p3_reference(root, lock, data)
    summary = summarize(data, features, long_valid, predictions, protocol, p3_reference)
    p1.atomic_bytes(
        output / "oof_probabilities.npz",
        p1.npz_bytes(
            sample_ids=data.sample_ids,
            recording_ids=data.scenarios,
            labels=data.labels,
            folds=data.folds,
            long_valid=long_valid,
            camera_quality_mean=features.camera_quality_mean,
            baseline_probabilities=data.baseline,
            p3_best_probabilities=p3_reference,
            **predictions,
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
            "probability_fallback_rows",
        ],
    )
    writer.writeheader()
    for arm, result in summary["models"].items():
        writer.writerow(
            {
                "model": arm,
                "probability_fallback_rows": 0,
                **{
                    key: result["metrics"][key]
                    for key in ("macro_f1", "accuracy", "nll", "brier")
                },
            }
        )
    p1.atomic_bytes(output / "metrics.csv", table.getvalue().encode())
    p1.atomic_json(
        output / "paired_statistics.json",
        {"contrasts": summary["contrasts"], "holm_family": summary["holm_family"]},
    )
    inventory = [
        output / name
        for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")
    ]
    inventory.extend(sorted(path for path in (output / "workloads").rglob("*") if path.is_file()))
    summary.update(
        request_sha256=request_sha,
        artifacts={path.relative_to(output).as_posix(): p1.sha256_file(path) for path in inventory},
    )
    _validate_output(output, summary, request_sha, published=False)
    p1.atomic_json(summary_path, summary)
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
        "models": {
            arm: values["metrics"] for arm, values in result.get("models", {}).items()
        },
        "target_crossed_by": result.get("target_crossed_by", []),
    }
    print(json.dumps(concise, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
