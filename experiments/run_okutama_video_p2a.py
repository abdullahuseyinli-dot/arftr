"""Execute the prospectively locked P2a frozen-feature fusion screen."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_video_probe as p1
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from hac.video_fusion import (
    ARMS,
    SOURCE_ARMS,
    DerivedFeatures,
    arm_features,
    decode_factorized_probabilities,
    derive_features,
)

C_VALUES = (0.00001, 0.0001, 0.001, 0.01)


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    c: float,
    seed: int,
    *,
    binary: bool,
) -> tuple[StandardScaler, LogisticRegression]:
    expected = np.arange(2 if binary else 3)
    if not np.array_equal(np.unique(labels), expected):
        raise RuntimeError("A fitting partition is missing required classes")
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all():
        raise RuntimeError("Fitting features are not aligned finite vectors")
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
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(standardized, labels)
        except ConvergenceWarning as error:
            raise RuntimeError(
                "P2a logistic probe did not converge; refusing to continue"
            ) from error
    if int(np.max(model.n_iter_)) >= int(config["max_iter"]):
        raise RuntimeError("P2a logistic probe reached its locked iteration ceiling")
    if not np.array_equal(model.classes_, expected):
        raise RuntimeError("Fitted probability class order changed")
    return scaler, model


def _predict(fit: tuple[StandardScaler, LogisticRegression], values: np.ndarray) -> np.ndarray:
    scaler, model = fit
    with threadpool_limits(limits=1):
        return model.predict_proba(scaler.transform(values))


def _checkpoint_arrays(
    fit: tuple[StandardScaler, LogisticRegression], prefix: str
) -> dict[str, np.ndarray]:
    scaler, model = fit
    return {
        f"{prefix}_{key}": value
        for key, value in {
            "coefficients": model.coef_,
            "intercept": model.intercept_,
            "classes": model.classes_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "scaler_variance": scaler.var_,
        }.items()
    }


def select_candidate(candidates: list[dict[str, Any]], *, factorized: bool) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No locked inner candidate was evaluated")
    return min(
        candidates,
        key=lambda row: (
            -row["inner_metrics"]["macro_f1"],
            row["inner_metrics"]["nll"],
            *([row["posture_C"], row["motion_C"]] if factorized else [row["C"]]),
        ),
    )


def nested_workload(
    data: p1.PrimaryData,
    features: DerivedFeatures,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    """Fit only outer-training groups; neither reference's probabilities are read."""
    if arm not in ARMS or not features.validity[arm].all():
        raise RuntimeError("P2a requires a declared arm and every original feature row valid")
    x, motion_x = arm_features(features, arm)
    config = protocol["probes"]["linear"]
    seed = int(protocol["seed"])
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
    candidates = []
    factorized = motion_x is not None
    if not factorized:
        for c in config["C_values"]:
            inner = np.full((len(data.labels), 3), np.nan)
            iterations = []
            for train, held in splits:
                fit = fit_logistic(
                    x[train], data.labels[train], config, float(c), seed, binary=False
                )
                inner[held] = _predict(fit, x[held])
                iterations.append(int(np.max(fit[1].n_iter_)))
            candidates.append(
                {
                    "C": float(c),
                    "inner_metrics": p1.metrics(data.labels[fit_rows], inner[fit_rows]),
                    "inner_iterations": iterations,
                }
            )
        selected = select_candidate(candidates, factorized=False)
        fitted = fit_logistic(
            x[fit_rows], data.labels[fit_rows], config, selected["C"], seed, binary=False
        )
        probabilities = _predict(fitted, x[held_rows])
        checkpoint = _checkpoint_arrays(fitted, "multinomial")
        final_iterations = {"multinomial": int(np.max(fitted[1].n_iter_))}
    else:
        # Cache the 4+4 fitted binary candidate predictions within each inner fold;
        # evaluate the predeclared Cartesian C grid without fitting 16 duplicate pairs.
        posture_predictions, motion_predictions = {}, {}
        posture_iterations, motion_iterations = {}, {}
        for c in config["C_values"]:
            c = float(c)
            posture_predictions[c] = np.full(len(data.labels), np.nan)
            motion_predictions[c] = np.full(len(data.labels), np.nan)
            posture_iterations[c], motion_iterations[c] = [], []
            for train, held in splits:
                upright = train[data.labels[train] != 0]
                posture_fit = fit_logistic(
                    x[train], (data.labels[train] != 0).astype(int), config, c, seed, binary=True
                )
                motion_fit = fit_logistic(
                    motion_x[upright],
                    (data.labels[upright] == 2).astype(int),
                    config,
                    c,
                    seed,
                    binary=True,
                )
                posture_predictions[c][held] = 1 - _predict(posture_fit, x[held])[:, 1]
                motion_predictions[c][held] = _predict(motion_fit, motion_x[held])[:, 1]
                posture_iterations[c].append(int(np.max(posture_fit[1].n_iter_)))
                motion_iterations[c].append(int(np.max(motion_fit[1].n_iter_)))
        for posture_c in config["C_values"]:
            for motion_c in config["C_values"]:
                decoded = decode_factorized_probabilities(
                    posture_predictions[posture_c][fit_rows], motion_predictions[motion_c][fit_rows]
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
        selected = select_candidate(candidates, factorized=True)
        upright = fit_rows[data.labels[fit_rows] != 0]
        posture_fit = fit_logistic(
            x[fit_rows],
            (data.labels[fit_rows] != 0).astype(int),
            config,
            selected["posture_C"],
            seed,
            binary=True,
        )
        motion_fit = fit_logistic(
            motion_x[upright],
            (data.labels[upright] == 2).astype(int),
            config,
            selected["motion_C"],
            seed,
            binary=True,
        )
        probabilities = decode_factorized_probabilities(
            1 - _predict(posture_fit, x[held_rows])[:, 1],
            _predict(motion_fit, motion_x[held_rows])[:, 1],
        )
        checkpoint = {
            **_checkpoint_arrays(posture_fit, "posture"),
            **_checkpoint_arrays(motion_fit, "motion"),
        }
        final_iterations = {
            "posture": int(np.max(posture_fit[1].n_iter_)),
            "motion": int(np.max(motion_fit[1].n_iter_)),
        }
    p1.validate_probability_array(probabilities, len(held_rows))
    return (
        probabilities,
        {
            "selection": selected,
            "candidates": candidates,
            "fit_rows": len(fit_rows),
            "motion_fit_rows": int((data.labels[fit_rows] != 0).sum()) if factorized else None,
            "inner_selection_rows": len(fit_rows),
            "baseline_access_during_fit": False,
            "p1_oof_access_during_fit": False,
            "fallback_rows": 0,
            "final_iterations": final_iterations,
            "feature_dimensions": {
                "posture" if factorized else "multinomial": x.shape[1],
                **({"motion": motion_x.shape[1]} if factorized else {}),
            },
        },
        p1.npz_bytes(**checkpoint),
    )


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p2a")
    if not hasattr(locker, "validate_p2a_lock"):
        raise RuntimeError("P2a validator unavailable; fitting is forbidden")
    return locker.validate_p2a_lock(root, path.resolve())


def _workload(
    data: p1.PrimaryData,
    features: DerivedFeatures,
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
    receipt_path, request_path = directory / "receipt.json", directory / "request.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("request") != request or receipt.get("status") != "P2A_WORKLOAD_COMPLETE":
            raise RuntimeError("Retained P2a workload request/status changed")
        if not request_path.exists() or json.loads(request_path.read_text()) != request:
            raise RuntimeError("Retained P2a workload pre-fit request changed")
        if set(receipt.get("artifacts", {})) != {"checkpoint.npz", "predictions.npz"}:
            raise RuntimeError("Retained P2a workload artifact inventory changed")
        if {path.name for path in directory.iterdir()} != {
            "request.json",
            "receipt.json",
            "checkpoint.npz",
            "predictions.npz",
        }:
            raise RuntimeError("Unexpected files in retained P2a workload")
        for filename, digest in receipt["artifacts"].items():
            if p1.sha256_file(directory / filename) != digest:
                raise RuntimeError("Retained P2a artifact bytes changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], data.sample_ids[held]) or not np.array_equal(
                saved["row_indices"], held
            ):
                raise RuntimeError("Retained P2a prediction identity/order changed")
            probabilities = saved["probabilities"].copy()
        p1.validate_probability_array(probabilities, len(held))
        return probabilities, False
    if directory.exists() and any(path.name != "request.json" for path in directory.iterdir()):
        raise RuntimeError("Incomplete P2a workload evidence retained; cannot overwrite")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Interrupted P2a workload belongs to a different request")
    else:
        p1.atomic_json(request_path, request)
    begin = time.perf_counter()
    probabilities, details, checkpoint = nested_workload(data, features, arm, fold, protocol)
    p1.validate_probability_array(probabilities, len(held))
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
            "status": "P2A_WORKLOAD_COMPLETE",
            "request": request,
            "details": details,
            "held_metrics": p1.metrics(data.labels[held], probabilities),
            "seconds": time.perf_counter() - begin,
            "artifacts": {
                name: p1.sha256_file(directory / name)
                for name in ("checkpoint.npz", "predictions.npz")
            },
        },
    )
    print(f"Completed P2a {arm}/fold-{fold}", flush=True)
    return probabilities, True


def _load_p1_reference(root: Path, lock: dict[str, Any], data: p1.PrimaryData) -> np.ndarray:
    path = p1._checked_path(root, lock["p1_reference"])
    with np.load(path, allow_pickle=False) as source:
        for name, expected in (
            ("sample_ids", data.sample_ids),
            ("recording_ids", data.scenarios),
            ("labels", data.labels),
            ("folds", data.folds),
        ):
            if not np.array_equal(source[name], expected):
                raise RuntimeError("P1 comparison reference identity/order differs from P2a")
        values = source[lock["p1_reference"]["probabilities_member"]].astype(np.float64)
    p1.validate_probability_array(values, len(data.labels))
    return values


def summarize(
    data: p1.PrimaryData,
    probabilities: dict[str, np.ndarray],
    protocol: dict[str, Any],
    p1_reference: np.ndarray,
) -> dict[str, Any]:
    design = protocol["statistics"]
    references = {**probabilities, "baseline": data.baseline, "p1_real_linear": p1_reference}
    results = {}
    for name, values in probabilities.items():
        results[name] = {
            "metrics": p1.metrics(data.labels, values),
            "fallback_rows": 0,
            "per_scenario": {
                str(group): p1.metrics(data.labels[mask], values[mask])
                for group in np.unique(data.scenarios)
                for mask in [data.scenarios == group]
            },
            "diagnostic_strata": {
                name: p1.metrics(data.labels[mask], values[mask])
                for name, mask in (
                    ("occluded", data.occluded),
                    ("clear", ~data.occluded),
                    ("transition", data.transition),
                    ("stable", ~data.transition),
                )
                if mask.any()
            },
        }
    contrasts = {}
    for comparison in design["comparisons"]:
        name = comparison["name"]
        candidate, reference = comparison["candidate"], comparison["reference"]
        if name in contrasts or candidate not in probabilities or reference not in references:
            raise RuntimeError("Invalid or duplicate locked P2a comparison")
        contrasts[name] = p1.paired_statistics(
            data.labels,
            probabilities[candidate],
            references[reference],
            data.scenarios,
            bootstrap_resamples=int(design["bootstrap_resamples"]),
            bootstrap_seed=int(design["bootstrap_seed"]),
        )
        contrasts[name]["diagnostic_strata"] = {
            stratum: subgroup_statistics(
                data.labels,
                probabilities[candidate],
                references[reference],
                data.scenarios,
                mask,
                bootstrap_resamples=int(design["bootstrap_resamples"]),
                bootstrap_seed=int(design["bootstrap_seed"]),
            )
            for stratum, mask in (
                ("occluded", data.occluded),
                ("clear", ~data.occluded),
                ("transition", data.transition),
                ("stable", ~data.transition),
            )
            if mask.any()
        }
    family = sorted(contrasts)
    adjusted = p1.holm_adjust(
        {name: contrasts[name]["one_sided_exact_swap_pvalue"] for name in family}
    )
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_one_sided_pvalue"] = value
    return {
        "status": "OKUTAMA_VIDEO_P2A_EXPLORATORY_CROSSFIT_COMPLETE",
        "rows": len(data.labels),
        "scenarios": len(np.unique(data.scenarios)),
        "seed": int(protocol["seed"]),
        "baseline": p1.metrics(data.labels, data.baseline),
        "p1_real_linear_reference": p1.metrics(data.labels, p1_reference),
        "models": results,
        "contrasts": contrasts,
        "holm_family": family,
        "scope": "reused_development_scenarios_post_P1_hypothesis_not_independent_confirmation",
        "denominator_policy": "all_4977_original_rows_all_features_valid_no_fallback",
        "baseline_comparison": "historical_five_seed_reference_vs_single_seed_screen",
        "reference_predictions_used_for_fitting": False,
        "target_images_read": 0,
        "protected_manifest_reads": 0,
    }


def subgroup_statistics(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    scenarios: np.ndarray,
    mask: np.ndarray,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Diagnostic subgroup intervals retain the original full scenario draw design."""
    if mask.shape != labels.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Subgroup mask must select retained rows")
    group_order = np.unique(scenarios)
    cand_pred, ref_pred = candidate.argmax(1), reference.argmax(1)
    matrices = []
    for predictions in (cand_pred, ref_pred):
        matrices.append(
            np.stack(
                [
                    p1.confusion(labels[selected], predictions[selected])
                    for group in group_order
                    for selected in [mask & (scenarios == group)]
                ]
            )
        )
    cand_cm, ref_cm = matrices
    draws = np.random.default_rng(bootstrap_seed).integers(
        0, len(group_order), size=(bootstrap_resamples, len(group_order))
    )
    boot_cand, boot_ref = cand_cm[draws].sum(1), ref_cm[draws].sum(1)
    differences = p1.f1_from_confusion(boot_cand).mean(1) - p1.f1_from_confusion(boot_ref).mean(1)
    supported = (boot_cand.sum(2) > 0).all(1)
    rescued = mask & (cand_pred == labels) & (ref_pred != labels)
    harmed = mask & (cand_pred != labels) & (ref_pred == labels)
    return {
        "rows": int(mask.sum()),
        "class_support": cand_cm.sum((0, 2)).tolist(),
        "macro_f1_delta": float(
            p1.f1_from_confusion(cand_cm.sum(0)).mean() - p1.f1_from_confusion(ref_cm.sum(0)).mean()
        ),
        "macro_f1_delta_two_sided_95pct": np.quantile(differences, [0.025, 0.975]).tolist(),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws_used": len(differences),
        "bootstrap_all_classes_supported_fraction": float(supported.mean()),
        "bootstrap_empty_draw_fraction": float((boot_cand.sum((1, 2)) == 0).mean()),
        "bootstrap_missing_class_policy": "retain_fixed_three_classes_zero_division_0",
        "scenario_order": group_order.tolist(),
        "rescued_errors": int(rescued.sum()),
        "new_errors": int(harmed.sum()),
        "rescue_harm_per_class": {
            name: {
                "rescued": int(rescued[labels == label].sum()),
                "harmed": int(harmed[labels == label].sum()),
            }
            for label, name in enumerate(p1.CLASS_NAMES)
        },
        "scope": "exploratory_subgroup_diagnostic_no_additional_significance_claim",
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    linear = protocol["probes"]["linear"]
    if (
        tuple(protocol["arms"]) != ARMS
        or int(protocol["seed"]) != 42
        or tuple(linear["C_values"]) != C_VALUES
        or linear["solver"] != "lbfgs"
        or linear["class_weight"] != "balanced"
        or int(linear["inner_folds"]) != 3
        or int(linear["inner_seed"]) != 42
        or not linear["inner_shuffle"]
        or not linear["standardize"]
    ):
        raise RuntimeError("P2a arm, seed, preprocessing or fitting contract changed")


def _validate_output_files(
    output: Path, summary: dict[str, Any], request_digest: str, *, published: bool = True
) -> None:
    if (
        summary.get("request_sha256") != request_digest
        or summary.get("status") != "OKUTAMA_VIDEO_P2A_EXPLORATORY_CROSSFIT_COMPLETE"
    ):
        raise RuntimeError("Retained P2a summary belongs to a different request")
    expected = {"oof_probabilities.npz", "metrics.csv", "paired_statistics.json"}
    expected.update(
        f"workloads/{arm}/fold-{fold}/{filename}"
        for arm in ARMS
        for fold in range(5)
        for filename in ("request.json", "receipt.json", "predictions.npz", "checkpoint.npz")
    )
    if set(summary.get("artifacts", {})) != expected:
        raise RuntimeError("Retained P2a summary artifact inventory changed")
    for filename, digest in summary["artifacts"].items():
        if p1.sha256_file(output / filename) != digest:
            raise RuntimeError("Retained P2a result artifact changed")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    markers = {"request.json", "summary.json"} if published else {"request.json"}
    if observed != expected | markers:
        raise RuntimeError("Unexpected evidence files in complete P2a output")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P2a output must use a dedicated directory below .runs")
    if args.max_new_workloads is not None and args.max_new_workloads < 1:
        raise ValueError("max_new_workloads must be positive")
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    _validate_protocol(protocol)
    data = p1.load_primary_data(root, lock)
    if any(not data.validity[arm].all() for arm in SOURCE_ARMS):
        raise RuntimeError("P2a requires all original feature rows valid; no baseline fallback")
    p1._validate_inner_map(data, lock)
    p1.validate_randomization(lock)
    request = {
        "status": "P2A_REQUEST_BEFORE_FITTING",
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
            raise RuntimeError("Output belongs to a different locked P2a request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty output has no matching P2a request")
    else:
        p1.atomic_json(request_path, request)
    request_digest = p1.canonical_digest(request)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        _validate_output_files(output, summary, request_digest)
        return summary
    if any(
        (output / name).exists()
        for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")
    ):
        raise RuntimeError("Incomplete aggregate evidence retained; cannot overwrite")
    features = derive_features(*(data.features[arm] for arm in SOURCE_ARMS), data.validity)
    predictions = {arm: np.full((len(data.labels), 3), np.nan) for arm in ARMS}
    new_workloads = 0
    for arm in ARMS:
        for fold in range(5):
            receipt = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            if (
                args.max_new_workloads is not None
                and new_workloads >= args.max_new_workloads
                and not receipt.exists()
            ):
                return {
                    "status": "P2A_BOUNDED_RUN_PAUSED",
                    "new_workloads": new_workloads,
                    "output": str(output),
                }
            values, fitted = _workload(data, features, arm, fold, protocol, output, request_digest)
            predictions[arm][data.folds == fold] = values
            new_workloads += int(fitted)
    for values in predictions.values():
        p1.validate_probability_array(values, len(data.labels))
    _load_lock(root, args.protocol_lock)
    p1_reference = _load_p1_reference(root, lock, data)  # Comparisons only, after every fit.
    summary = summarize(data, predictions, protocol, p1_reference)
    p1.atomic_bytes(
        output / "oof_probabilities.npz",
        p1.npz_bytes(
            sample_ids=data.sample_ids,
            recording_ids=data.scenarios,
            labels=data.labels,
            folds=data.folds,
            baseline_probabilities=data.baseline,
            p1_real_linear_probabilities=p1_reference,
            **predictions,
        ),
    )
    table = io.StringIO(newline="")
    writer = csv.DictWriter(
        table, fieldnames=["model", "macro_f1", "accuracy", "nll", "brier", "fallback_rows"]
    )
    writer.writeheader()
    for arm, result in summary["models"].items():
        writer.writerow(
            {
                "model": arm,
                "fallback_rows": 0,
                **{
                    name: result["metrics"][name]
                    for name in ("macro_f1", "accuracy", "nll", "brier")
                },
            }
        )
    p1.atomic_bytes(output / "metrics.csv", table.getvalue().encode())
    p1.atomic_json(
        output / "paired_statistics.json",
        {"contrasts": summary["contrasts"], "holm_family": summary["holm_family"]},
    )
    inventory = [
        output / name for name in ("oof_probabilities.npz", "metrics.csv", "paired_statistics.json")
    ]
    inventory.extend(sorted(path for path in (output / "workloads").rglob("*") if path.is_file()))
    summary.update(
        request_sha256=request_digest,
        artifacts={path.relative_to(output).as_posix(): p1.sha256_file(path) for path in inventory},
    )
    _validate_output_files(output, summary, request_digest, published=False)
    p1.atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-workloads", type=int)
    print(
        json.dumps(run(parser.parse_args()), indent=2, sort_keys=True, allow_nan=False), flush=True
    )


if __name__ == "__main__":
    main()
