"""Connect completed phase results without fitting or choosing a new ensemble.

Every candidate must cover the same complete cohort. Error correlations and
pairwise correction opportunities are retrospective development descriptions,
not a deployable label-oracle score or an independent confirmation experiment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_okutama_memory_results import (  # noqa: E402
    compare,
    load_arm,
    load_inputs,
    read_json,
    write_csv,
)
from run_okutama_video_probe import (  # noqa: E402
    holm_adjust,
    paired_statistics,
    validate_probability_array,
)

from hac.actor_memory_base import file_sha256  # noqa: E402

RUNS = ROOT / ".runs/research_20260908"
MEMORY_ARMS = ("temporal_conv", "query_attention", "survival_memory", "corroborated_memory")
ORDERED_ARMS = ("pooled_mlp", "temporal_adapter", "ordered_relational", "correspondence_relational")
EPISODE_ARMS = ("log_survival_control", "expected_episode")
SOURCE_HEADS = ("global_mean", "global_plus_spatial_contrast")
SOURCE_TYPES = ("existing720", "native4k", "exact4k_downsample720")


def require_family_binding(summary, key, path, provenance):
    actual = file_sha256(path)
    if summary[key] != actual:
        raise RuntimeError(f"Result is not bound to its intended execution: {path}")
    provenance[str(path)] = actual


def verify_ordered_workloads(summary, directory, predictions, data, provenance):
    lock_path = directory.parent / "execution_lock.json"
    execution_hash = file_sha256(lock_path)
    provenance[str(lock_path)] = execution_hash
    expected = {
        (directory.parent / "workloads" / arm / f"fold-{fold}" / "receipt.json").resolve(): (
            arm,
            fold,
        )
        for arm in ORDERED_ARMS
        for fold in range(5)
    }
    entries = summary["workloads"]
    if len(entries) != 20 or {Path(entry["path"]).resolve() for entry in entries} != set(expected):
        raise RuntimeError("Ordered summary does not bind exactly the intended20 workloads")
    for entry in entries:
        path = Path(entry["path"]).resolve()
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError("Ordered workload receipt changed")
        provenance[str(path)] = entry["sha256"]
        arm, fold = expected[path]
        saved = read_json(path)
        if (
            saved["request"]
            != {"execution_lock_sha256": execution_hash, "arm": arm, "outer_fold": fold}
            or saved["unique_fits"] != 15
        ):
            raise RuntimeError("Ordered workload request or fit budget changed")
        for artifact in saved["fit_prediction_receipts"]:
            if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise RuntimeError("An ordered fit prediction changed")
        artifact = saved["artifacts"]["predictions.npz"]
        if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("An ordered workload prediction changed")
        held = np.flatnonzero(data["folds"] == fold)
        with np.load(artifact["path"], allow_pickle=False) as archive:
            if not np.array_equal(
                archive["sample_ids"], data["sample_ids"][held]
            ) or not np.array_equal(archive["probabilities"], predictions[arm][held]):
                raise RuntimeError("Ordered full-cohort predictions differ from verified workload")


def checked_archive(path, expected_hash, data, keys, *, group_key="scenarios"):
    if file_sha256(path) != expected_hash:
        raise RuntimeError(f"Completed probability archive changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        for source, target in (
            ("sample_ids", "sample_ids"),
            ("labels", "labels"),
            ("folds", "folds"),
            (group_key, "scenarios"),
        ):
            if not np.array_equal(archive[source], data[target]):
                raise RuntimeError(f"Cross-family row identity differs: {path}/{source}")
        predictions = {}
        for key in keys:
            values = archive[key]
            if values.shape != (len(data["labels"]), 3) or not np.isfinite(values).all():
                raise RuntimeError(f"Incomplete or invalid probability values: {path}/{key}")
            # Reuse the original2e-6 float32 normalization contract; do not
            # silently renormalize/rewrite any family's saved probabilities.
            validate_probability_array(values, len(data["labels"]))
            predictions[key] = values.copy()
    return predictions


def error_connection(labels, left, right, shared):
    left_correct, right_correct = left.argmax(1) == labels, right.argmax(1) == labels
    correlation = None
    if np.unique(left_correct).size == 2 and np.unique(right_correct).size == 2:
        correlation = float(np.corrcoef(~left_correct, ~right_correct)[0, 1])
    return {
        "rows": len(labels),
        "both_correct": int((left_correct & right_correct).sum()),
        "left_only_correct": int((left_correct & ~right_correct).sum()),
        "right_only_correct": int((~left_correct & right_correct).sum()),
        "both_wrong": int((~left_correct & ~right_correct).sum()),
        "error_indicator_pearson_correlation": correlation,
        "historical_shared_errors_left_only_repairs": int(
            (shared & left_correct & ~right_correct).sum()
        ),
        "historical_shared_errors_right_only_repairs": int(
            (shared & ~left_correct & right_correct).sum()
        ),
        "historical_shared_errors_unresolved_by_either": int(
            (shared & ~left_correct & ~right_correct).sum()
        ),
        "interpretation": "Label-dependent diagnostic counts; not an implemented selector or ensemble",
    }


def run(output, episode_results):
    if output.exists():
        raise RuntimeError("Choose a fresh output directory; prior connection audits are immutable")
    provenance = {str(Path(__file__)): file_sha256(Path(__file__))}
    for relative in (
        "experiments/analyze_okutama_memory_results.py",
        "experiments/run_okutama_video_probe.py",
    ):
        provenance[relative] = file_sha256(ROOT / relative)
    data, baseline, legacy, lock = load_inputs(RUNS / "evidence_memory", provenance)
    predictions = {"p6": baseline["probabilities"]}
    for arm in MEMORY_ARMS:
        result, coverage = load_arm(RUNS / "evidence_memory", data, lock, arm, provenance)
        if not coverage["complete_primary_population"]:
            raise RuntimeError("Memory family is not complete")
        predictions[f"memory/{arm}"] = result["probabilities"].mean(0)

    ordered_directory = RUNS / "ordered_motion_v2/results"
    ordered_summary = read_json(ordered_directory / "summary.json")
    if (
        ordered_summary["status"] != "ORDERED_MOTION_ADAPTIVE_CROSSFIT_COMPLETE"
        or ordered_summary["unique_fits"] != 300
    ):
        raise RuntimeError("Ordered matrix is not complete")
    ordered = checked_archive(
        ordered_directory / "oof_probabilities.npz",
        ordered_summary["artifacts"]["oof_probabilities.npz"]["sha256"],
        data,
        ("p6", *ORDERED_ARMS, *(f"{arm}_blend" for arm in ORDERED_ARMS)),
    )
    if not np.array_equal(ordered.pop("p6"), predictions["p6"]):
        raise RuntimeError("Ordered P6 reference differs")
    verify_ordered_workloads(ordered_summary, ordered_directory, ordered, data, provenance)
    predictions.update({f"ordered/{key}": value for key, value in ordered.items()})

    episode_summary = read_json(episode_results / "summary.json")
    if not episode_summary["complete"] or episode_summary["verified_fit_receipts"] != 150:
        raise RuntimeError("Episode matrix is not complete")
    require_family_binding(
        episode_summary,
        "execution_lock_sha256",
        RUNS / "center_episode_v2/execution_lock.json",
        provenance,
    )
    episode = checked_archive(
        episode_results / "oof_probabilities.npz",
        episode_summary["artifacts_sha256"]["oof_probabilities.npz"],
        data,
        EPISODE_ARMS,
    )
    predictions.update({f"episode/{key}": value for key, value in episode.items()})

    source_directory = RUNS / "native4k_source_effect/results"
    source_summary = read_json(source_directory / "summary.json")
    if (
        source_summary["status"] != "FIXED_NESTED_SOURCE_EFFECT_SCREEN_COMPLETE"
        or source_summary["estimator_fits"] != 390
    ):
        raise RuntimeError("Controlled source matrix is not complete")
    require_family_binding(
        source_summary, "protocol_sha256", RUNS / "native4k_source_effect/protocol.json", provenance
    )
    sources = checked_archive(
        source_directory / "oof_probabilities.npz",
        source_summary["oof_sha256"],
        data,
        tuple(f"{source}__{head}" for source in SOURCE_TYPES for head in SOURCE_HEADS),
        group_key="recording_ids",
    )
    predictions.update({f"source/{key}": value for key, value in sources.items()})
    for directory in (ordered_directory, episode_results, source_directory):
        for filename in ("summary.json", "oof_probabilities.npz"):
            provenance[str(directory / filename)] = file_sha256(directory / filename)
    shared = np.all(baseline["components"].argmax(-1) != data["labels"][:, None], axis=1)
    if int(shared.sum()) != 508:
        raise RuntimeError("Historical shared failure definition changed")
    height = data["quality"][:, 0].astype(float) * 720
    masks = {
        "all": np.ones(len(height), dtype=bool),
        "native_height_le32": height <= 32 + 1e-4,
        "native_height_32to64": (height > 32 + 1e-4) & (height <= 64 + 1e-4),
        "native_height_gt64": height > 64 + 1e-4,
        "long_fallback": ~data["long_valid"],
        **legacy,
    }
    masks.update(
        {f"scenario/{group}": data["scenarios"] == group for group in np.unique(data["scenarios"])}
    )
    masks.update({f"class/{label}": data["labels"] == label for label in range(3)})
    metrics, table = {}, []
    for name, values in predictions.items():
        metrics[name] = {}
        for stratum, mask in masks.items():
            result = compare(data["labels"], values, predictions["p6"], shared, mask)
            metrics[name][stratum] = result
            table.append(
                {
                    "model": name,
                    "stratum": stratum,
                    **{key: value for key, value in result.items() if not isinstance(value, dict)},
                    **{
                        key: result.get("metrics", {}).get(key)
                        for key in ("macro_f1", "accuracy", "nll", "brier")
                    },
                }
            )
    connections = [
        {
            "left": left,
            "right": right,
            **error_connection(data["labels"], predictions[left], predictions[right], shared),
        }
        for left, right in itertools.combinations(predictions, 2)
    ]
    source_statistics = {}
    for head in SOURCE_HEADS:
        for candidate, reference in (
            ("native4k", "exact4k_downsample720"),
            ("native4k", "existing720"),
            ("exact4k_downsample720", "existing720"),
        ):
            name = f"{candidate}_minus_{reference}/{head}"
            source_statistics[name] = paired_statistics(
                data["labels"],
                sources[f"{candidate}__{head}"],
                sources[f"{reference}__{head}"],
                data["scenarios"],
                bootstrap_resamples=10000,
                bootstrap_seed=20260908,
            )
    adjusted = holm_adjust(
        {name: value["one_sided_exact_swap_pvalue"] for name, value in source_statistics.items()}
    )
    for name, value in adjusted.items():
        source_statistics[name]["holm_adjusted_over_six_fixed_source_contrasts"] = value
    output.mkdir(parents=True)
    write_csv(output / "metrics_by_stratum.csv", table)
    write_csv(output / "pairwise_error_connections.csv", connections)
    summary = {
        "complete": True,
        "evaluation_only": True,
        "new_fits": 0,
        "new_fusion_rules": 0,
        "rows": len(height),
        "scenarios": len(np.unique(data["scenarios"])),
        "model_count": len(predictions),
        "historical_shared_p6_errors": int(shared.sum()),
        "height_band_note": "Same stored float32 geometry and0.0001pixel tolerance as final memory diagnostics; exact-boundary counts can differ from the original float64 audit",
        "class_subset_note": "Class-subset accuracy is that class recall; fixed-three-class macro-F1 inside a one-class subset is not ordinary per-class F1. Use the all-row per_class_f1 values for class F1 comparisons.",
        "scope": "Retrospective repeated-use development connections; raw source contrasts were fixed before their fits, but this additional uncertainty analysis is not independent confirmation",
        "metrics": metrics,
        "pairwise_error_connections": connections,
        "source_scenario_block_uncertainty": source_statistics,
        "provenance": provenance,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps({"complete": True, "models": len(predictions), "output": str(output)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-results", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output.resolve(), arguments.episode_results.resolve())
