"""No-fit, retrospective diagnostics for the next HAC information experiment.

Oracles use true labels and are explicitly nondeployable. This program does not
choose, fit, promote, or alter a classifier. Existing artifacts are read-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".runs/research_20260912/next_phase_diagnostic_v1/summary.json"
PATHS = {
    "bounded": ROOT / ".runs/research_20260912/bounded_factor_correction_v1/results/v0001/oof_probabilities.npz",
    "arftr": ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz",
    "fsar": ROOT / ".runs/research_20260912/frame_supervised_anchor_residual_v2/results/v0001/oof_predictions.npz",
    "data": ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz",
}


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def read_npz(path: Path, keys: tuple[str, ...] | None = None) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in (saved.files if keys is None else keys)}


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "rows": int(len(labels)),
        "macro_f1_percent": float(100 * f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "accuracy_percent": float(100 * np.mean(labels == predictions)),
        "errors": int(np.sum(labels != predictions)),
        "confusion": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }


def main() -> None:
    if OUTPUT.parent.exists():
        raise RuntimeError("Unique diagnostic output already exists; refusing overwrite")
    bounded, arftr, fsar = (read_npz(PATHS[name]) for name in ("bounded", "arftr", "fsar"))
    data = read_npz(PATHS["data"], (
        "sample_ids", "labels", "scenarios", "folds", "recordings", "tracks", "frames",
        "node_support_complete", "node_support_boundary", "neighbor_indices", "valid",
    ))
    for other in (arftr, fsar, data):
        for key in ("sample_ids", "labels", "scenarios", "folds"):
            if not np.array_equal(other[key], bounded[key]):
                raise RuntimeError(f"Identity mismatch: {key}")
    labels = bounded["labels"]
    if len(labels) != 4977 or len(np.unique(bounded["sample_ids"])) != 4977:
        raise RuntimeError("Unexpected canonical cohort")
    if str(bounded["arms"][0]) != "b0_exact_arftr" or str(arftr["arms"][5]) != "r5_arftr_full":
        raise RuntimeError("ARFTR arm ordering changed")
    reference = bounded["mean_probabilities"][0]
    if not np.array_equal(reference, arftr["mean_probabilities"][5]):
        raise RuntimeError("Exact ARFTR reference mismatch")
    prediction = reference.argmax(1)
    errors = prediction != labels
    masks = {
        "known_pure": data["node_support_complete"] & ~data["node_support_boundary"],
        "known_mixed": data["node_support_complete"] & data["node_support_boundary"],
        "unknown_annotation_support": ~data["node_support_complete"],
    }
    if not np.all(np.stack(list(masks.values())).sum(0) == 1):
        raise RuntimeError("Support strata do not partition the cohort")

    def oracle(mask: np.ndarray) -> dict:
        if np.any(mask & ~errors):
            raise RuntimeError("Oracle rescue mask contains anchor-correct rows")
        changed = prediction.copy()
        changed[mask] = labels[mask]
        return {
            "label_oracle_non_deployable": True,
            "rescues": int(mask.sum()),
            "remaining_errors": int(np.sum(errors & ~mask)),
            "macro_f1_percent": metrics(labels, changed)["macro_f1_percent"],
            "rescues_by_true_class": np.bincount(labels[mask], minlength=3).tolist(),
            "rescues_by_support_stratum": {name: int(np.sum(mask & subset)) for name, subset in masks.items()},
        }

    def transitions(probabilities: np.ndarray) -> dict:
        current = probabilities.argmax(1)
        rescues = errors & (current == labels)
        harms = ~errors & (current != labels)
        correlation = np.corrcoef(errors.astype(float), (current != labels).astype(float))[0, 1]
        return {
            "metrics": metrics(labels, current),
            "rescues": int(rescues.sum()),
            "harms": int(harms.sum()),
            "net_corrections": int(rescues.sum() - harms.sum()),
            "shared_errors": int(np.sum(errors & (current != labels))),
            "binary_error_correlation": float(correlation),
            "support_transitions": {
                name: {"rescues": int(np.sum(rescues & subset)), "harms": int(np.sum(harms & subset))}
                for name, subset in masks.items()
            },
            "oracle": oracle(rescues),
        }

    sources = {str(name): p for name, p in zip(bounded["arms"][1:], bounded["mean_probabilities"][1:])}
    sources.update({str(name): p for name, p in zip(fsar["arms"], fsar["mean_probabilities"])})
    sources.update({str(name): p for name, p in zip(arftr["arms"], arftr["mean_probabilities"])})
    sources["p6"] = arftr["p6_probabilities"]
    sources["a3"] = arftr["a3_seed_probabilities"].mean(0)
    meaningful_names = ["p6", "a3", "f3_all_cached_frame_supervision", "b2_bounded_separate_all_frames"]
    meaningful_union = np.zeros(len(labels), dtype=bool)
    for name in meaningful_names:
        meaningful_union |= errors & (sources[name].argmax(1) == labels)
    all_union = np.zeros(len(labels), dtype=bool)
    for probability in sources.values():
        all_union |= errors & (probability.argmax(1) == labels)

    neighbor_results = {}
    for horizon, slots in ((1, [1, 3]), (2, [0, 1, 3, 4])):
        raw_indices = data["neighbor_indices"][:, slots]
        valid = data["valid"][:, slots] & (raw_indices >= 0)
        indices = np.maximum(raw_indices, 0)
        for key in ("scenarios", "folds", "recordings", "tracks"):
            if np.any(valid & (data[key][indices] != data[key][:, None])):
                raise RuntimeError(f"Temporal neighbor crosses {key}")
        expected = np.asarray([-60, -30, 0, 30, 60])[slots]
        if np.any(valid & (data["frames"][indices] - data["frames"][:, None] != expected)):
            raise RuntimeError("Temporal neighbor timestamps changed")
        correct_neighbor = prediction[indices] == labels[:, None]
        same_label = labels[indices] == labels[:, None]
        rescues = errors & np.any(valid & correct_neighbor, axis=1)
        same_rescues = errors & np.any(valid & correct_neighbor & same_label, axis=1)
        averaged = (reference + np.sum(reference[indices] * valid[:, :, None], axis=1)) / (1 + valid.sum(1))[:, None]
        neighbor_results[str(horizon)] = {
            "neighbor_center_horizon_seconds": horizon,
            "label_oracle": oracle(rescues),
            "same_true_label_neighbor_oracle": oracle(same_rescues),
            "meaningful_sources_plus_neighbor_oracle": oracle(meaningful_union | rescues),
            "fixed_uniform_probability_mean_including_center": transitions(averaged),
            "timing_caveat": "Nominal source frame/30; future access. Neighbor ARFTR already aggregates contextual clips and adjacent centers, so visual receptive field extends beyond the center horizon.",
        }

    # For standing, reducing both factors by the box radius maximizes its
    # advantage over both competitors; for walking, reduce posture/increase
    # motion. For sitting, maximize posture then balance the two upright
    # probabilities as closely as the motion interval permits. Bisection
    # chooses a common motion delta across seed anchors, not trained outputs.
    seed_anchor = bounded["seed_probabilities"][0]
    posture = np.log(seed_anchor[:, :, 0] / seed_anchor[:, :, 1:].sum(2))
    motion = np.log(seed_anchor[:, :, 2] / seed_anchor[:, :, 1])

    def decode(delta_posture: float, delta_motion: float | np.ndarray) -> np.ndarray:
        sitting = 1 / (1 + np.exp(-(posture + delta_posture)))
        walking = 1 / (1 + np.exp(-(motion + delta_motion)))
        return np.stack([sitting, (1 - sitting) * (1 - walking), (1 - sitting) * walking], axis=-1).mean(0)

    low, high = np.full(len(labels), -0.5), np.full(len(labels), 0.5)
    for _ in range(60):
        midpoint = (low + high) / 2
        probabilities = decode(0.5, midpoint)
        walk_larger = probabilities[:, 2] > probabilities[:, 1]
        high = np.where(walk_larger, midpoint, high)
        low = np.where(walk_larger, low, midpoint)
    possible = np.column_stack([
        decode(0.5, (low + high) / 2).argmax(1) == 0,
        decode(-0.5, -0.5).argmax(1) == 1,
        decode(-0.5, 0.5).argmax(1) == 2,
    ])
    box_rescues = errors & possible[np.arange(len(labels)), labels]

    saturation, correction_inputs = {}, []
    for arm in bounded["arms"][1:4]:
        directory = ROOT / ".runs/research_20260912/bounded_factor_correction_v1/models" / str(arm)
        paths = sorted(directory.glob("fold-*/seed-*/predictions.npz"))
        if len(paths) != 15:
            raise RuntimeError("Expected all 15 correction predictions for each bounded arm")
        corrections = []
        for path in paths:
            saved = read_npz(path)
            if not np.array_equal(saved["sample_ids"], bounded["sample_ids"][saved["held_rows"]]):
                raise RuntimeError("Correction prediction row identity mismatch")
            values = saved["corrections"]
            if values.shape != (len(saved["held_rows"]), 2) or not np.isfinite(values).all() or np.any(np.abs(values) > 0.500001):
                raise RuntimeError("Invalid bounded correction values")
            corrections.append(values)
            correction_inputs.append(file_record(path))
        values = np.abs(np.concatenate(corrections))
        saturation[str(arm)] = {
            "seed_row_observations": len(values),
            "factor_order": ["posture", "motion"],
            "absolute_delta_quantiles": {str(q): np.quantile(values, q, axis=0).tolist() for q in (0.5, 0.9, 0.95, 0.99, 1.0)},
            "fraction_absolute_delta_at_least_0_49": np.mean(values >= 0.49, axis=0).tolist(),
        }

    confidence = reference.max(1)
    result = {
        "status": "RETROSPECTIVE_NO_FIT_NEXT_PHASE_DIAGNOSTICS_COMPLETE",
        "model_fits": 0,
        "training_launched": False,
        "existing_artifacts_modified": False,
        "interpretation": "Adaptive development diagnostics only. Label oracles are not achievable forecasts or deployable routing rules. Fixed means are descriptive unselected controls. No promotion decision or threshold selection was performed.",
        "inputs": {name: file_record(path) for name, path in PATHS.items()},
        "correction_prediction_inputs": correction_inputs,
        "analysis_script": file_record(Path(__file__).resolve()),
        "class_order": ["sitting", "standing", "walking_running"],
        "reference": metrics(labels, prediction),
        "support_strata_are_label_derived": True,
        "support_stratum_caveat": "Pure/mixed use true action labels at sampled support frames; unknown describes missing/ambiguous action annotations. None is a deployable gate input. Observable missing image frames/tracking quality are different features.",
        "reference_by_support_stratum": {
            name: {**metrics(labels[subset], prediction[subset]), "errors_by_true_class": np.bincount(labels[errors & subset], minlength=3).tolist()}
            for name, subset in masks.items()
        },
        "reference_by_scenario": {
            str(scenario): metrics(labels[bounded["scenarios"] == scenario], prediction[bounded["scenarios"] == scenario])
            for scenario in np.unique(bounded["scenarios"])
        },
        "confidence_bands": [
            {"lower_inclusive": lower, "upper_exclusive": upper, **metrics(labels[(confidence >= lower) & (confidence < upper)], prediction[(confidence >= lower) & (confidence < upper)])}
            for lower, upper in ((0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))
        ],
        "source_comparisons": {name: transitions(probability) for name, probability in sources.items()},
        "meaningful_source_union": {"sources": meaningful_names, **oracle(meaningful_union)},
        "all_inspected_sources_union_including_weak_and_shuffle_controls": {"sources": list(sources), **oracle(all_union)},
        "temporal_neighbors": neighbor_results,
        "factor_box_reachability": {
            "absolute_bound_per_factor": 0.5,
            "coordinate_system": "sitting-vs-upright log odds; walking-vs-standing log odds",
            "calculation": "Fixed seed anchors; label-informed extremal common factor delta across seeds for each row, followed by arithmetic mean of seed probabilities. Sitting motion balanced by 60-step monotone bisection. Equality uses NumPy argmax class order.",
            "caveat": "Actual trained models have different per-seed corrections; this calculation constrains a common per-row delta. It diagnoses box reachability, not evidence quality or learnability. Closed endpoints are included although finite tanh only approaches them.",
            **oracle(box_rescues),
            "unreachable_errors_by_true_class": np.bincount(labels[errors & ~box_rescues], minlength=3).tolist(),
        },
        "bounded_delta_saturation": saturation,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    reloaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if reloaded["reference"]["errors"] != 702 or reloaded["factor_box_reachability"]["rescues"] != 216:
        raise RuntimeError("Diagnostic replay differs from the initial read-only calculations")
    print(json.dumps({"status": reloaded["status"], "output": str(OUTPUT), "reference_macro_f1_percent": reloaded["reference"]["macro_f1_percent"], "box_reachable_errors": reloaded["factor_box_reachability"]["rescues"], "meaningful_union_rescues": reloaded["meaningful_source_union"]["rescues"]}, indent=2))


if __name__ == "__main__":
    main()
