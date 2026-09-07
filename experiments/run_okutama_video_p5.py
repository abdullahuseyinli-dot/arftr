"""Run the locked Orthogonal Space-Time Moment (OSTM) video trial."""

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
import run_okutama_video_p4 as p4

from hac.video_multiscale import derive_multiscale_features
from hac.video_token_moments import (
    ARMS,
    TokenMomentFeatures,
    arm_inputs,
    derive_token_moments,
)

p3, p2, p1 = p4.p3, p4.p2, p4.p1
C_VALUES = (1e-5, 1e-4, 1e-3, 1e-2)


def nested_workload(
    data: p1.PrimaryData,
    features: TokenMomentFeatures,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    if arm not in ARMS:
        raise RuntimeError("Unknown P5 arm")
    inputs = arm_inputs(features, arm)
    matrices = (inputs.posture_or_direct, *inputs.motion_references)
    if any(not np.isfinite(value).all() for value in matrices):
        raise RuntimeError("P5 fixed moment feature contains nonfinite values")
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
        probabilities, details, checkpoint = p4._fit_direct(
            data, inputs, outer_fold, config, seed, splits
        )
    else:
        probabilities, details, checkpoint = p4._fit_factorized(
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
        long_to_short_feature_fallback_rows=int((~features.long_valid).sum()),
    )
    return probabilities, details, checkpoint


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p5")
    return locker.validate_lock(root, path.resolve())


def _load_data(
    root: Path, lock: dict[str, Any]
) -> tuple[p1.PrimaryData, TokenMomentFeatures]:
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
    features = derive_token_moments(
        multiscale,
        short_v,
        long_v,
        short_d,
        long_d,
    )
    return data, features


def _workload(
    data: p1.PrimaryData,
    features: TokenMomentFeatures,
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
        if retained.get("status") != "P5_WORKLOAD_COMPLETE" or retained.get("request") != request:
            raise RuntimeError("Retained P5 workload request/status changed")
        expected = {"checkpoint.npz", "predictions.npz"}
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        if set(retained.get("artifacts", {})) != expected or observed != expected | {
            "request.json",
            "receipt.json",
        }:
            raise RuntimeError("Retained P5 workload inventory changed")
        if not request_path.exists() or json.loads(request_path.read_text()) != request:
            raise RuntimeError("Retained P5 pre-fit request changed")
        for name, digest in retained["artifacts"].items():
            if p1.sha256_file(directory / name) != digest:
                raise RuntimeError("Retained P5 workload artifact changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], data.sample_ids[held]) or not np.array_equal(
                saved["row_indices"], held
            ):
                raise RuntimeError("Retained P5 prediction identity/order changed")
            probabilities = saved["probabilities"].copy()
        p1.validate_probability_array(probabilities, len(held))
        return probabilities, False
    if directory.exists() and any(path.name != "request.json" for path in directory.iterdir()):
        raise RuntimeError("Incomplete P5 workload evidence retained; refusing overwrite")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Interrupted P5 workload belongs to another request")
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
            "status": "P5_WORKLOAD_COMPLETE",
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
    print(f"Completed P5 {arm}/fold-{fold}", flush=True)
    return probabilities, True


def summarize(
    data: p1.PrimaryData,
    features: TokenMomentFeatures,
    probabilities: dict[str, np.ndarray],
    protocol: dict[str, Any],
    p3_reference: np.ndarray,
) -> dict[str, Any]:
    references = {**probabilities, "baseline": data.baseline, "p3_best": p3_reference}
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
    return {
        "status": "OKUTAMA_VIDEO_P5_ADAPTIVE_CROSSFIT_COMPLETE",
        "method": "Orthogonal Space-Time Moment probe (OSTM)",
        "rows": len(data.labels),
        "scenarios": len(np.unique(data.scenarios)),
        "seed": int(protocol["seed"]),
        "baseline": p1.metrics(data.labels, data.baseline),
        "p3_best_reference": p1.metrics(data.labels, p3_reference),
        "models": models,
        "contrasts": contrasts,
        "holm_family": family,
        "error_correlations": p3._error_correlations(data.labels, all_predictions),
        "long_valid_rows": int(features.long_valid.sum()),
        "short_anchor_rows": int((~features.long_valid).sum()),
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
    locker = importlib.import_module("tools.lock_okutama_video_p5")
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
        or int(probe["maximum_unique_estimator_fits"]) != 585
        or protocol["statistics"]["comparisons"] != locker.expected_comparisons()
    ):
        raise RuntimeError("P5 protocol fitting contract changed")


def _validate_output(
    output: Path, summary: dict[str, Any], request_sha: str, *, published: bool = True
) -> None:
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P5_ADAPTIVE_CROSSFIT_COMPLETE"
        or summary.get("request_sha256") != request_sha
    ):
        raise RuntimeError("Retained P5 summary belongs to another request")
    expected = {"oof_probabilities.npz", "metrics.csv", "paired_statistics.json"}
    expected.update(
        f"workloads/{arm}/fold-{fold}/{name}"
        for arm in ARMS
        for fold in range(5)
        for name in ("request.json", "receipt.json", "checkpoint.npz", "predictions.npz")
    )
    if set(summary.get("artifacts", {})) != expected:
        raise RuntimeError("Retained P5 aggregate inventory changed")
    for name, digest in summary["artifacts"].items():
        if p1.sha256_file(output / name) != digest:
            raise RuntimeError("Retained P5 artifact bytes changed")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    markers = {"request.json", "summary.json"} if published else {"request.json"}
    if observed != expected | markers:
        raise RuntimeError("Unexpected files in completed P5 output")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P5 output must use a dedicated directory below .runs")
    if args.max_new_workloads is not None and args.max_new_workloads < 1:
        raise ValueError("max_new_workloads must be positive")
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    _validate_protocol(protocol)
    data, features = _load_data(root, lock)
    request = {
        "status": "P5_REQUEST_BEFORE_FITTING",
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
            raise RuntimeError("P5 output belongs to another locked request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty P5 output lacks its matching request")
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
        raise RuntimeError("Incomplete P5 aggregate evidence retained; refusing overwrite")
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
                return {"status": "P5_BOUNDED_RUN_PAUSED", "new_workloads": new_workloads}
            values, fitted = _workload(
                data, features, arm, fold, protocol, output, request_sha
            )
            predictions[arm][data.folds == fold] = values
            new_workloads += int(fitted)
    for values in predictions.values():
        p1.validate_probability_array(values, len(data.labels))
    _load_lock(root, args.protocol_lock)
    p3_reference = p4._load_p3_reference(root, lock, data)
    summary = summarize(data, features, predictions, protocol, p3_reference)
    p1.atomic_bytes(
        output / "oof_probabilities.npz",
        p1.npz_bytes(
            sample_ids=data.sample_ids,
            recording_ids=data.scenarios,
            labels=data.labels,
            folds=data.folds,
            long_valid=features.long_valid,
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
            "feature_fallback_rows",
            "probability_fallback_rows",
        ],
    )
    writer.writeheader()
    for arm, result in summary["models"].items():
        writer.writerow(
            {
                "model": arm,
                "feature_fallback_rows": result["feature_fallback_rows"],
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
