"""Run the prospectively declared P3 short/long multiscale cross-fit."""

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
import run_okutama_video_p2a as p2

from hac.video_multiscale import (
    ARMS,
    MultiScaleFeatures,
    arm_features,
    derive_multiscale_features,
)

p1 = p2.p1
C_VALUES = (1e-5, 1e-4, 1e-3, 1e-2)


def nested_workload(
    data: p1.PrimaryData,
    features: MultiScaleFeatures,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    if arm not in ARMS:
        raise RuntimeError("Unknown P3 arm")
    x, motion_x = arm_features(features, arm)
    if not np.isfinite(x).all() or (motion_x is not None and not np.isfinite(motion_x).all()):
        raise RuntimeError("P3 fixed feature contains nonfinite values")
    config, seed = protocol["probe"], int(protocol["seed"])
    splits = p1.grouped_inner_splits(
        data.labels,
        data.scenarios,
        data.folds,
        outer_fold,
        n_splits=int(config["inner_folds"]),
        seed=int(config["inner_seed"]),
    )
    fit_rows = np.flatnonzero(data.folds != outer_fold)
    held_rows = np.flatnonzero(data.folds == outer_fold)
    candidates: list[dict[str, Any]] = []
    factorized = motion_x is not None
    if not factorized:
        for c in config["C_values"]:
            inner = np.full((len(data.labels), 3), np.nan)
            iterations = []
            for train, held in splits:
                fitted = p2.fit_logistic(
                    x[train], data.labels[train], config, float(c), seed, binary=False
                )
                inner[held] = p2._predict(fitted, x[held])
                iterations.append(int(np.max(fitted[1].n_iter_)))
            candidates.append(
                {
                    "C": float(c),
                    "inner_metrics": p1.metrics(data.labels[fit_rows], inner[fit_rows]),
                    "inner_iterations": iterations,
                }
            )
        selected = p2.select_candidate(candidates, factorized=False)
        fitted = p2.fit_logistic(
            x[fit_rows], data.labels[fit_rows], config, selected["C"], seed, binary=False
        )
        probabilities = p2._predict(fitted, x[held_rows])
        checkpoint = p2._checkpoint_arrays(fitted, "multinomial")
        final_iterations = {"multinomial": int(np.max(fitted[1].n_iter_))}
    else:
        posture_predictions: dict[float, np.ndarray] = {}
        motion_predictions: dict[float, np.ndarray] = {}
        posture_iterations: dict[float, list[int]] = {}
        motion_iterations: dict[float, list[int]] = {}
        for c_value in config["C_values"]:
            c = float(c_value)
            posture_predictions[c] = np.full(len(data.labels), np.nan)
            motion_predictions[c] = np.full(len(data.labels), np.nan)
            posture_iterations[c], motion_iterations[c] = [], []
            for train, held in splits:
                upright = train[data.labels[train] != 0]
                posture_fit = p2.fit_logistic(
                    x[train],
                    (data.labels[train] != 0).astype(int),
                    config,
                    c,
                    seed,
                    binary=True,
                )
                motion_fit = p2.fit_logistic(
                    motion_x[upright],
                    (data.labels[upright] == 2).astype(int),
                    config,
                    c,
                    seed,
                    binary=True,
                )
                posture_predictions[c][held] = 1 - p2._predict(posture_fit, x[held])[:, 1]
                motion_predictions[c][held] = p2._predict(motion_fit, motion_x[held])[:, 1]
                posture_iterations[c].append(int(np.max(posture_fit[1].n_iter_)))
                motion_iterations[c].append(int(np.max(motion_fit[1].n_iter_)))
        for posture_c in config["C_values"]:
            for motion_c in config["C_values"]:
                decoded = p2.decode_factorized_probabilities(
                    posture_predictions[posture_c][fit_rows],
                    motion_predictions[motion_c][fit_rows],
                )
                candidates.append(
                    {
                        "posture_C": float(posture_c),
                        "motion_C": float(motion_c),
                        "inner_metrics": p1.metrics(data.labels[fit_rows], decoded),
                        "posture_inner_iterations": posture_iterations[posture_c],
                        "motion_inner_iterations": motion_iterations[motion_c],
                    }
                )
        selected = p2.select_candidate(candidates, factorized=True)
        upright = fit_rows[data.labels[fit_rows] != 0]
        posture_fit = p2.fit_logistic(
            x[fit_rows],
            (data.labels[fit_rows] != 0).astype(int),
            config,
            selected["posture_C"],
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
            1 - p2._predict(posture_fit, x[held_rows])[:, 1],
            p2._predict(motion_fit, motion_x[held_rows])[:, 1],
        )
        checkpoint = {
            **p2._checkpoint_arrays(posture_fit, "posture"),
            **p2._checkpoint_arrays(motion_fit, "motion"),
        }
        final_iterations = {
            "posture": int(np.max(posture_fit[1].n_iter_)),
            "motion": int(np.max(motion_fit[1].n_iter_)),
        }
    p1.validate_probability_array(probabilities, len(held_rows))
    return probabilities, {
        "selection": selected,
        "candidates": candidates,
        "fit_rows": len(fit_rows),
        "inner_selection_rows": len(fit_rows),
        "motion_fit_rows": int((data.labels[fit_rows] != 0).sum()) if factorized else None,
        "baseline_access_during_fit": False,
        "p2a_oof_access_during_fit": False,
        "feature_fallback_rows_total": int((~features.long_valid).sum()),
        "probability_fallback_rows": 0,
        "final_iterations": final_iterations,
        "feature_dimensions": {
            "posture" if factorized else "multinomial": x.shape[1],
            **({"motion": motion_x.shape[1]} if factorized else {}),
        },
    }, p1.npz_bytes(**checkpoint)


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p3_probe")
    return locker.validate_probe_lock(root, path.resolve())


def _load_data(root: Path, lock: dict[str, Any]):
    p2_path = p1._checked_path(root, lock["p2a_reference"]["execution_lock"])
    p2_lock = json.loads(p2_path.read_text(encoding="utf-8"))
    data = p1.load_primary_data(root, p2_lock)
    p1._validate_inner_map(data, p2_lock)
    p1.validate_randomization(p2_lock)
    long_values, long_validity = {}, {}
    for cache in lock["long_caches"].values():
        arm = cache["arm"]
        feature = cache["artifacts"][f"{arm}.npy"]
        validity = cache["artifacts"]["validity.npy"]
        long_values[arm] = np.load(
            p1._checked_path(root, feature), allow_pickle=False, mmap_mode="r"
        )
        long_validity[arm] = np.load(
            p1._checked_path(root, validity), allow_pickle=False
        )
    features = derive_multiscale_features(
        data.features["vjepa21_real_clip"],
        long_values["vjepa21_long16_real_clip"],
        data.features["dinov2_native_frames"],
        long_values["dinov2_long16_native_frames"],
        long_validity["vjepa21_long16_real_clip"],
        long_validity["dinov2_long16_native_frames"],
    )
    return data, features


def _workload(
    data: p1.PrimaryData,
    features: MultiScaleFeatures,
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
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "P3_WORKLOAD_COMPLETE" or receipt.get("request") != request:
            raise RuntimeError("Retained P3 workload request/status changed")
        if not request_path.exists() or json.loads(request_path.read_text()) != request:
            raise RuntimeError("Retained P3 pre-fit request changed")
        expected = {"checkpoint.npz", "predictions.npz"}
        if set(receipt.get("artifacts", {})) != expected:
            raise RuntimeError("Retained P3 workload inventory changed")
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        if observed != expected | {"request.json", "receipt.json"}:
            raise RuntimeError("Retained P3 workload inventory changed")
        for name, digest in receipt["artifacts"].items():
            if p1.sha256_file(directory / name) != digest:
                raise RuntimeError("Retained P3 workload artifact changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], data.sample_ids[held]) or not np.array_equal(
                saved["row_indices"], held
            ):
                raise RuntimeError("Retained P3 prediction identity/order changed")
            probabilities = saved["probabilities"].copy()
        p1.validate_probability_array(probabilities, len(held))
        return probabilities, False
    if directory.exists() and any(path.name != "request.json" for path in directory.iterdir()):
        raise RuntimeError("Incomplete P3 workload evidence retained; refusing overwrite")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Interrupted P3 workload belongs to another request")
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
            sample_ids=data.sample_ids[held], row_indices=held, probabilities=probabilities
        ),
    )
    p1.atomic_json(
        receipt_path,
        {
            "status": "P3_WORKLOAD_COMPLETE",
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
    print(f"Completed P3 {arm}/fold-{fold}", flush=True)
    return probabilities, True


def _load_p2_reference(
    root: Path, lock: dict[str, Any], data: p1.PrimaryData
) -> np.ndarray:
    path = p1._checked_path(root, lock["p2a_reference"]["oof"])
    with np.load(path, allow_pickle=False) as source:
        expected_members = {
            "sample_ids",
            "recording_ids",
            "labels",
            "folds",
            "baseline_probabilities",
            "p1_real_linear_probabilities",
            *lock["p2a_lock_protocol"]["arms"],
        }
        if set(source.files) != expected_members:
            raise RuntimeError("P2a comparison inventory changed")
        for name, expected in (
            ("sample_ids", data.sample_ids),
            ("recording_ids", data.scenarios),
            ("labels", data.labels),
            ("folds", data.folds),
        ):
            if not np.array_equal(source[name], expected):
                raise RuntimeError("P2a comparison identity/order changed")
        values = source[lock["p2a_reference"]["arm"]].astype(np.float64)
    p1.validate_probability_array(values, len(data.labels))
    observed = p1.metrics(data.labels, values)["macro_f1"]
    if observed != lock["p2a_reference"]["macro_f1"]:
        raise RuntimeError("P2a comparison metric changed")
    return values


def _error_correlations(
    labels: np.ndarray, probabilities: dict[str, np.ndarray]
) -> dict[str, float]:
    errors = {name: values.argmax(1) != labels for name, values in probabilities.items()}
    result = {}
    names = sorted(errors)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = np.corrcoef(errors[left].astype(float), errors[right].astype(float))[0, 1]
            result[f"{left}__{right}"] = float(value)
    return result


def summarize(
    data: p1.PrimaryData,
    features: MultiScaleFeatures,
    probabilities: dict[str, np.ndarray],
    protocol: dict[str, Any],
    p2_reference: np.ndarray,
) -> dict[str, Any]:
    references = {**probabilities, "baseline": data.baseline, "p2a_best": p2_reference}
    strata = (
        ("occluded", data.occluded),
        ("clear", ~data.occluded),
        ("transition", data.transition),
        ("stable", ~data.transition),
        ("long_valid", features.long_valid),
        ("long_invalid_short_anchor", ~features.long_valid),
    )
    models = {
        arm: {
            "metrics": p1.metrics(data.labels, values),
            "feature_fallback_rows": int((~features.long_valid).sum()),
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
    for item in protocol["statistics"]["comparisons"]:
        candidate, reference = item["candidate"], item["reference"]
        name = f"{candidate}_vs_{reference}"
        contrasts[name] = p1.paired_statistics(
            data.labels,
            probabilities[candidate],
            references[reference],
            data.scenarios,
            bootstrap_resamples=int(protocol["statistics"]["bootstrap_resamples"]),
            bootstrap_seed=int(protocol["statistics"]["bootstrap_seed"]),
        )
        contrasts[name]["diagnostic_strata"] = {
            stratum: p2.subgroup_statistics(
                data.labels,
                probabilities[candidate],
                references[reference],
                data.scenarios,
                mask,
                bootstrap_resamples=int(protocol["statistics"]["bootstrap_resamples"]),
                bootstrap_seed=int(protocol["statistics"]["bootstrap_seed"]),
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
    all_predictions = {**probabilities, "baseline": data.baseline, "p2a_best": p2_reference}
    return {
        "status": "OKUTAMA_VIDEO_P3_EXPLORATORY_CROSSFIT_COMPLETE",
        "rows": len(data.labels),
        "scenarios": len(np.unique(data.scenarios)),
        "seed": int(protocol["seed"]),
        "baseline": p1.metrics(data.labels, data.baseline),
        "p2a_best_reference": p1.metrics(data.labels, p2_reference),
        "models": models,
        "contrasts": contrasts,
        "holm_family": family,
        "error_correlations": _error_correlations(data.labels, all_predictions),
        "long_valid_rows": int(features.long_valid.sum()),
        "short_anchor_rows": int((~features.long_valid).sum()),
        "scope": "same_development_scenarios_exploratory_not_independent_confirmation",
        "reference_predictions_used_for_fitting": False,
        "raw_images_read": 0,
        "protected_manifest_reads": 0,
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    probe = protocol["probe"]
    expected_comparisons = [
        *({"candidate": arm, "reference": "baseline"} for arm in ARMS),
        *({"candidate": arm, "reference": "p2a_best"} for arm in ARMS),
        {"candidate": "dual_scale_vjepa", "reference": "long_vjepa_mean"},
        {"candidate": "dual_scale_vjepa_motion", "reference": "dual_scale_vjepa"},
        {"candidate": "dual_scale_vjepa_dino", "reference": "dual_scale_vjepa"},
        {
            "candidate": "dual_scale_factorized",
            "reference": "dual_scale_vjepa_dino",
        },
    ]
    if (
        tuple(protocol["arms"]) != ARMS
        or int(protocol["seed"]) != 42
        or tuple(probe["C_values"]) != C_VALUES
        or probe["solver"] != "lbfgs"
        or probe["class_weight"] != "balanced"
        or int(probe["inner_folds"]) != 3
        or int(probe["inner_seed"]) != 42
        or not probe["standardize"]
        or not probe["inner_shuffle"]
        or int(probe["max_iter"]) != 2000
        or float(probe["tolerance"]) != 0.0001
        or int(probe["maximum_unique_estimator_fits"]) != 455
        or protocol["statistics"]["comparisons"] != expected_comparisons
    ):
        raise RuntimeError("P3 protocol fitting contract changed")


def _validate_output(
    output: Path, summary: dict[str, Any], request_sha: str, *, published: bool = True
) -> None:
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P3_EXPLORATORY_CROSSFIT_COMPLETE"
        or summary.get("request_sha256") != request_sha
    ):
        raise RuntimeError("Retained P3 summary belongs to another request")
    expected = {"oof_probabilities.npz", "metrics.csv", "paired_statistics.json"}
    expected.update(
        f"workloads/{arm}/fold-{fold}/{name}"
        for arm in ARMS
        for fold in range(5)
        for name in ("request.json", "receipt.json", "checkpoint.npz", "predictions.npz")
    )
    if set(summary.get("artifacts", {})) != expected:
        raise RuntimeError("Retained P3 aggregate inventory changed")
    for name, digest in summary["artifacts"].items():
        if p1.sha256_file(output / name) != digest:
            raise RuntimeError("Retained P3 artifact bytes changed")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    markers = {"request.json", "summary.json"} if published else {"request.json"}
    if observed != expected | markers:
        raise RuntimeError("Unexpected files in completed P3 output")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P3 output must use a dedicated directory below .runs")
    if args.max_new_workloads is not None and args.max_new_workloads < 1:
        raise ValueError("max_new_workloads must be positive")
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    _validate_protocol(protocol)
    data, features = _load_data(root, lock)
    request = {
        "status": "P3_REQUEST_BEFORE_FITTING",
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
            raise RuntimeError("P3 output belongs to another locked request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty P3 output lacks its matching request")
    else:
        p1.atomic_json(request_path, request)
    request_sha = p1.canonical_digest(request)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        _validate_output(output, summary, request_sha)
        return summary
    if any((output / name).exists() for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")):
        raise RuntimeError("Incomplete P3 aggregate evidence retained; refusing overwrite")
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
                return {"status": "P3_BOUNDED_RUN_PAUSED", "new_workloads": new_workloads}
            values, fitted = _workload(
                data, features, arm, fold, protocol, output, request_sha
            )
            predictions[arm][data.folds == fold] = values
            new_workloads += int(fitted)
    for values in predictions.values():
        p1.validate_probability_array(values, len(data.labels))
    _load_lock(root, args.protocol_lock)
    p2_reference = _load_p2_reference(root, lock, data)
    summary = summarize(data, features, predictions, protocol, p2_reference)
    p1.atomic_bytes(
        output / "oof_probabilities.npz",
        p1.npz_bytes(
            sample_ids=data.sample_ids,
            recording_ids=data.scenarios,
            labels=data.labels,
            folds=data.folds,
            long_valid=features.long_valid,
            baseline_probabilities=data.baseline,
            p2a_best_probabilities=p2_reference,
            **predictions,
        ),
    )
    table = io.StringIO(newline="")
    writer = csv.DictWriter(
        table,
        fieldnames=[
            "model", "macro_f1", "accuracy", "nll", "brier",
            "feature_fallback_rows", "probability_fallback_rows",
        ],
    )
    writer.writeheader()
    for arm, result in summary["models"].items():
        writer.writerow(
            {
                "model": arm,
                "feature_fallback_rows": result["feature_fallback_rows"],
                "probability_fallback_rows": 0,
                **{key: result["metrics"][key] for key in ("macro_f1", "accuracy", "nll", "brier")},
            }
        )
    p1.atomic_bytes(output / "metrics.csv", table.getvalue().encode())
    p1.atomic_json(
        output / "paired_statistics.json",
        {"contrasts": summary["contrasts"], "holm_family": summary["holm_family"]},
    )
    inventory = [output / name for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")]
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
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
