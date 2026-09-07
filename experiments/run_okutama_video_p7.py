"""Run the locked Center-Aware Phase-Preserving Expert (CAPE) trial."""

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

from hac.video_center import (
    ARMS,
    EXPECTED_WIDTHS,
    CenterAwareFeatures,
    arm_inputs,
    derive_center_aware_features,
)
from hac.video_consensus import uniform_consensus
from hac.video_multiscale import derive_multiscale_features
from hac.video_token_moments import derive_token_moments

p4, p3, p2, p1 = p5.p4, p5.p3, p5.p2, p5.p1
C_VALUES = (1e-5, 1e-4, 1e-3, 1e-2)
SYSTEMS = ("p6_reference", "center_signed_replacement_triad")
STATUS = "OKUTAMA_VIDEO_P7_ADAPTIVE_CROSSFIT_COMPLETE"


@dataclass(frozen=True)
class PostFitEvidence:
    p3_long: np.ndarray
    p3_dual: np.ndarray
    p5_spatial: np.ndarray
    p5_orthogonal: np.ndarray
    p6_reference: np.ndarray


def nested_workload(
    data: p1.PrimaryData,
    features: CenterAwareFeatures,
    arm: str,
    outer_fold: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    """Fit exactly one outer-fold workload without historical predictions."""
    if arm not in ARMS:
        raise RuntimeError("Unknown P7 arm")
    inputs = arm_inputs(features, arm)
    matrices = (inputs.posture_or_direct, *inputs.motion_references)
    if any(not np.isfinite(value).all() for value in matrices):
        raise RuntimeError("P7 fixed CAPE feature contains nonfinite values")
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
        historical_oof_access_during_fit=False,
        probability_fallback_rows=0,
        long_to_short_feature_fallback_rows=int((~features.long_valid).sum()),
    )
    return probabilities, details, checkpoint


def _load_lock(root: Path, path: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_video_p7")
    return locker.validate_lock(root, path.resolve())


def _load_data(
    root: Path, lock: dict[str, Any]
) -> tuple[p1.PrimaryData, CenterAwareFeatures]:
    """Read only frozen features and derive the locked label-free CAPE tensors."""
    p2_path = p1._checked_path(root, lock["p3_reference"]["p2_execution_lock"])
    p2_lock = json.loads(p2_path.read_text(encoding="utf-8"))
    data = p1.load_primary_data(root, p2_lock)
    p1._validate_inner_map(data, p2_lock)
    p1.validate_randomization(p2_lock)
    if p1.canonical_digest(data.sample_ids.tolist()) != lock["sample_ids_sha256"]:
        raise RuntimeError("P7 sample identity changed")
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
    features = derive_center_aware_features(
        moments,
        short_v,
        long_v,
        short_d,
        long_d,
    )
    return data, features


def _workload(
    data: p1.PrimaryData,
    features: CenterAwareFeatures,
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
        if retained.get("status") != "P7_WORKLOAD_COMPLETE" or retained.get("request") != request:
            raise RuntimeError("Retained P7 workload request/status changed")
        expected = {"checkpoint.npz", "predictions.npz"}
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        if set(retained.get("artifacts", {})) != expected or observed != expected | {
            "request.json",
            "receipt.json",
        }:
            raise RuntimeError("Retained P7 workload inventory changed")
        if not request_path.exists() or json.loads(request_path.read_text()) != request:
            raise RuntimeError("Retained P7 pre-fit request changed")
        for name, digest in retained["artifacts"].items():
            if p1.sha256_file(directory / name) != digest:
                raise RuntimeError("Retained P7 workload artifact changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(
                saved["sample_ids"], data.sample_ids[held]
            ) or not np.array_equal(saved["row_indices"], held):
                raise RuntimeError("Retained P7 prediction identity/order changed")
            probabilities = saved["probabilities"].copy()
        p1.validate_probability_array(probabilities, len(held))
        return probabilities, False
    if directory.exists() and any(path.name != "request.json" for path in directory.iterdir()):
        raise RuntimeError("Incomplete P7 workload evidence retained; refusing overwrite")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Interrupted P7 workload belongs to another request")
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
            "status": "P7_WORKLOAD_COMPLETE",
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
    print(f"Completed P7 {arm}/fold-{fold}", flush=True)
    return probabilities, True


def _read_oof(root: Path, entry: dict[str, Any]) -> dict[str, np.ndarray]:
    path = p1._checked_path(root, entry["oof"])
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _load_postfit_references(
    root: Path,
    lock: dict[str, Any],
    data: p1.PrimaryData,
    features: CenterAwareFeatures,
) -> PostFitEvidence:
    """Decode historical probabilities only after all P7 workloads complete."""
    arrays = {
        phase: _read_oof(root, entry)
        for phase, entry in lock["postfit_references"].items()
    }
    for phase, values in arrays.items():
        for name, expected in (
            ("sample_ids", data.sample_ids),
            ("recording_ids", data.scenarios),
            ("labels", data.labels),
            ("folds", data.folds),
            ("long_valid", features.long_valid),
        ):
            if name not in values or not np.array_equal(values[name], expected):
                raise RuntimeError(f"P7/{phase.upper()} OOF identity changed: {name}")
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
        p1.validate_probability_array(values, len(data.labels))
    p6_metrics = p1.metrics(data.labels, required[-1])
    for key, declared_key in (
        ("macro_f1", "reference_macro_f1"),
        ("nll", "reference_nll"),
        ("brier", "reference_brier"),
    ):
        if p6_metrics[key] != protocol["references"]["p6"][declared_key]:
            raise RuntimeError(f"Locked P6 {key} changed")
    return PostFitEvidence(*required)


def _feature_norm(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    norms = np.linalg.norm(values[mask], axis=1)
    if not np.isfinite(norms).all():
        raise RuntimeError("P7 feature norm contains nonfinite values")
    return {
        "rows": int(mask.sum()),
        "mean_l2": float(norms.mean()),
        "median_l2": float(np.median(norms)),
        "p95_l2": float(np.quantile(norms, 0.95)),
        "max_l2": float(norms.max()),
    }


def _feature_norms(features: CenterAwareFeatures) -> dict[str, Any]:
    masks = {
        "all": np.ones(len(features.long_valid), dtype=bool),
        "long_valid": features.long_valid,
        "short_fallback": ~features.long_valid,
    }
    values = {
        "center_anchor": features.center_anchor,
        "center_signed_change": features.center_signed_change,
        "offcenter_anchor": features.offcenter_anchor,
        "offcenter_signed_change": features.offcenter_signed_change,
        "center_unsigned_change": features.center_unsigned_change,
    }
    return {
        name: {stratum: _feature_norm(matrix, mask) for stratum, mask in masks.items()}
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
            right_error = (probabilities[right].argmax(axis=1) != labels).astype(
                np.float64
            )
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
    features: CenterAwareFeatures,
    predictions: dict[str, np.ndarray],
    systems: dict[str, np.ndarray],
    evidence: PostFitEvidence,
) -> dict[str, Any]:
    labels = data.labels
    p6 = systems["p6_reference"]
    candidate = systems["center_signed_replacement_triad"]
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
    }
    comparison_counts = {
        name: _change_counts(labels, candidate, p6, mask)
        for name, mask in strata.items()
        if mask.any()
    }
    class_names = ("sitting", "standing", "walking_running")
    candidates = {**predictions, "center_signed_replacement_triad": candidate}
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
            "shared_failure_rescues": int(
                (shared & (candidate.argmax(axis=1) == labels)).sum()
            ),
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
            name: p1.metrics(data.labels[mask], values[mask])
            for name, mask in strata
            if mask.any()
        },
    }


def summarize(
    data: p1.PrimaryData,
    features: CenterAwareFeatures,
    predictions: dict[str, np.ndarray],
    systems: dict[str, np.ndarray],
    evidence: PostFitEvidence,
    protocol: dict[str, Any],
    model_fits: int,
) -> dict[str, Any]:
    all_probabilities = {**predictions, **systems}
    fallback_rows = int((~features.long_valid).sum())
    fits_per_arm = {
        arm: 65 if EXPECTED_WIDTHS[arm][1] is None else 130 for arm in ARMS
    }
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
    diagnostics = _mechanism_diagnostics(
        data, features, predictions, systems, evidence
    )
    candidate = models["center_signed_replacement_triad"]
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
        "nll_strictly_below_p6": candidate["metrics"]["nll"]
        < reference["metrics"]["nll"],
        "brier_strictly_below_p6": candidate["metrics"]["brier"]
        < reference["metrics"]["brier"],
        "at_least_7_scenarios_improved": sum(value > 0 for value in scenario_deltas.values())
        >= int(guardrails["minimum_scenarios_improved_over_p6"]),
        "no_scenario_decline_over_0_02": min(scenario_deltas.values())
        >= -float(guardrails["maximum_scenario_macro_f1_decline"]),
        "positive_net_correct_overall": changes["all"]["net_correct_change"] > 0,
        "positive_net_correct_clear_stable": changes["clear_stable"][
            "net_correct_change"
        ]
        > 0,
    }
    primary_macro = candidate["metrics"]["macro_f1"]
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
        "primary_expert": protocol["primary_expert"],
        "primary_system": "center_signed_replacement_triad",
        "primary_system_macro_f1": primary_macro,
        "primary_system_delta_vs_p6": primary_macro
        - reference["metrics"]["macro_f1"],
        "scenario_macro_f1_deltas_vs_p6": scenario_deltas,
        "scenarios_improved_over_p6": sum(value > 0 for value in scenario_deltas.values()),
        "guardrail_results": guardrail_results,
        "all_engineering_guardrails_passed": all(guardrail_results.values()),
        "milestone_0_84_crossed": primary_macro
        >= float(guardrails["milestone_macro_f1"]),
        "stretch_0_85_crossed": primary_macro
        >= float(guardrails["stretch_macro_f1"]),
        "mechanism_diagnostics": diagnostics,
        "error_correlations": _error_correlations(data.labels, all_probabilities),
        "long_valid_rows": int(features.long_valid.sum()),
        "short_fallback_rows": fallback_rows,
        "model_fits": model_fits,
        "maximum_authorized_model_fits": int(
            protocol["probe"]["maximum_unique_estimator_fits"]
        ),
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
    locker = importlib.import_module("tools.lock_okutama_video_p7")
    probe = protocol["probe"]
    if (
        tuple(protocol["arms"]) != ARMS
        or protocol["primary_expert"] != "center_signed_factorized"
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
        or int(probe["maximum_unique_estimator_fits"]) != 715
        or int(probe["maximum_workloads"]) != 30
        or protocol["statistics"]["comparisons"] != locker.expected_comparisons()
        or int(protocol["statistics"]["bootstrap_resamples"]) != 10000
    ):
        raise RuntimeError("P7 protocol fitting contract changed")


def _model_fit_audit(output: Path) -> int:
    total = 0
    for arm in ARMS:
        for fold in range(5):
            path = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            total += int(receipt["details"]["unique_estimator_fits"])
    return total


def _validate_output(
    output: Path, summary: dict[str, Any], request_sha: str, *, published: bool = True
) -> None:
    if summary.get("status") != STATUS or summary.get("request_sha256") != request_sha:
        raise RuntimeError("Retained P7 summary belongs to another request")
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
        for name in ("request.json", "receipt.json", "checkpoint.npz", "predictions.npz")
    )
    if set(summary.get("artifacts", {})) != expected:
        raise RuntimeError("Retained P7 aggregate inventory changed")
    for name, digest in summary["artifacts"].items():
        if p1.sha256_file(output / name) != digest:
            raise RuntimeError("Retained P7 artifact bytes changed")
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    markers = (
        {"request.json", "summary.json", "completion.json"}
        if published
        else {"request.json"}
    )
    if observed != expected | markers:
        raise RuntimeError("Unexpected files in completed P7 output")
    if published:
        completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
        if completion != {
            "status": "P7_PUBLICATION_COMPLETE",
            "request_sha256": request_sha,
            "summary_sha256": p1.sha256_file(output / "summary.json"),
        }:
            raise RuntimeError("Retained P7 summary receipt changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P7 output must use a dedicated directory below .runs")
    if args.max_new_workloads is not None and args.max_new_workloads < 1:
        raise ValueError("max_new_workloads must be positive")
    lock = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    _validate_protocol(protocol)
    request = {
        "status": "P7_REQUEST_BEFORE_FEATURE_DERIVATION_OR_FITTING",
        "lock_sha256": p1.sha256_file(args.protocol_lock.resolve()),
        "protocol": protocol,
        "sample_ids_sha256": lock["sample_ids_sha256"],
        "source_sha256": p1.sha256_file(Path(__file__)),
        "output": str(output),
        "blas_thread_limit": 1,
        "named_p3_p5_p6_oof_access_before_all_fits": False,
        "legacy_baseline_loaded_read_only_but_not_used_for_fitting": True,
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("P7 output belongs to another locked request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty P7 output lacks its matching request")
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
        raise RuntimeError("Incomplete P7 aggregate evidence retained; refusing overwrite")
    data, features = _load_data(root, lock)
    predictions = {
        arm: np.full((len(data.labels), 3), np.nan, dtype=np.float64) for arm in ARMS
    }
    new_workloads = 0
    for arm in ARMS:
        for fold in range(5):
            receipt_path = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            if (
                args.max_new_workloads is not None
                and new_workloads >= args.max_new_workloads
                and not receipt_path.exists()
            ):
                return {"status": "P7_BOUNDED_RUN_PAUSED", "new_workloads": new_workloads}
            values, fitted = _workload(
                data, features, arm, fold, protocol, output, request_sha
            )
            predictions[arm][data.folds == fold] = values
            new_workloads += int(fitted)
    for values in predictions.values():
        p1.validate_probability_array(values, len(data.labels))
    model_fits = _model_fit_audit(output)
    if model_fits != int(protocol["probe"]["maximum_unique_estimator_fits"]):
        raise RuntimeError("P7 estimator-fit audit changed")
    _load_lock(root, args.protocol_lock)
    evidence = _load_postfit_references(root, lock, data, features)
    if not np.array_equal(predictions["spatial_refit"], evidence.p5_spatial):
        raise RuntimeError("P7 A0 failed exact P5 spatial OOF reproduction")
    systems = {
        "p6_reference": evidence.p6_reference,
        "center_signed_replacement_triad": uniform_consensus(
            [
                evidence.p3_long,
                evidence.p3_dual,
                predictions["center_signed_factorized"],
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
    p1.atomic_json(output / "diagnostics.json", summary["mechanism_diagnostics"])
    inventory = [output / name for name in aggregate_names]
    inventory.extend(
        sorted(path for path in (output / "workloads").rglob("*") if path.is_file())
    )
    summary.update(
        request_sha256=request_sha,
        artifacts={
            path.relative_to(output).as_posix(): p1.sha256_file(path)
            for path in inventory
        },
    )
    _validate_output(output, summary, request_sha, published=False)
    p1.atomic_json(summary_path, summary)
    p1.atomic_json(
        output / "completion.json",
        {
            "status": "P7_PUBLICATION_COMPLETE",
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
        "models": {
            name: values["metrics"] for name, values in result.get("models", {}).items()
        },
        "guardrail_results": result.get("guardrail_results", {}),
    }
    print(json.dumps(concise, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
