"""Run the locked, label-blind, decoder-free center-completion factorial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.center_completion_factorial import (  # noqa: E402
    cosine_errors,
    locked_factorial_decision,
    relative_reduction,
    robust_consensus,
    uniform_consensus,
)
from hac.center_evidence_completion import (  # noqa: E402
    bilinear_transport,
    donor_confidence_logits,
    fit_visible_anchor_transport,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_center_completion_matched_factorial_protocol.json"
LOCK = ROOT / "experiments/okutama_center_completion_matched_factorial_execution_lock.json"
CACHE = ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1"
CACHE_AUDIT = ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1_audit/audit_receipt.json"
TRANSPORT = ROOT / ".runs/research_20260913/center_evidence_completion_transport_v1"
TRANSPORT_AUDIT = ROOT / ".runs/research_20260913/center_evidence_completion_transport_v1_audit/audit_receipt.json"
SELECTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/center_completion_matched_factorial_v1"
ARMS = ("A", "B", "C", "D")
ARM_NAMES = {
    "A": "same_grid_uniform",
    "B": "same_grid_robust",
    "C": "affine_uniform",
    "D": "affine_robust",
}
OFFSETS = (-8, -4, 4, 7)
ROWS = 128
MASKS = 2
DONORS = 4
DIMENSION = 768
EXPECTED_TARGETS = 27407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-of", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_lock() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (
        lock["status"] != "MATCHED_FACTORIAL_EXECUTION_LOCKED_BEFORE_RESULTS"
        or protocol["status"]
        != "LABEL_BLIND_DECODER_FREE_FACTORIAL_PROTOCOL_LOCKED_BEFORE_RESULTS"
    ):
        raise RuntimeError("The prospective factorial authorization changed")
    for relative, expected in lock["pinned_sources"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Locked source changed: {relative}")
    for relative, expected in lock["pinned_inputs"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Locked input changed: {relative}")

    cache_summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
    cache_audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    transport_summary = json.loads((TRANSPORT / "summary.json").read_text(encoding="utf-8"))
    transport_audit = json.loads(TRANSPORT_AUDIT.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if (
        cache_summary["status"] != "CENTER_EVIDENCE_COMPLETION_FULL_CACHE_COMPLETE"
        or cache_audit["status"] != "INDEPENDENT_FULL_CACHE_REPLAY_AUDIT_PASS"
        or transport_summary["status"] != "CENTER_EVIDENCE_COMPLETION_TRANSPORT_CACHE_COMPLETE"
        or transport_audit["status"] != "INDEPENDENT_TRANSPORT_CACHE_REPLAY_AUDIT_PASS"
        or len(selection) != ROWS
        or len({row["sample_id"] for row in selection}) != ROWS
    ):
        raise RuntimeError("A prerequisite cache or replay audit is not valid")
    for name, receipt in cache_summary["artifacts"].items():
        path = CACHE / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != receipt["shape"] or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"Full-cache artifact changed: {name}")
    for name, receipt in transport_summary["artifacts"].items():
        path = TRANSPORT / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != receipt["shape"] or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"Transport artifact changed: {name}")
    return lock, protocol, selection


def _paired_bootstrap(
    errors: dict[str, np.ndarray], replicates: int = 100_000, seed: int = 20_260_913
) -> dict[str, dict[str, float | list[float]]]:
    generator = np.random.default_rng(seed)
    contrasts = {"C_vs_A": ("A", "C"), "D_vs_B": ("B", "D"), "B_vs_A": ("A", "B"), "D_vs_C": ("C", "D"), "D_vs_A": ("A", "D")}
    values = {name: np.empty(replicates, dtype=np.float64) for name in contrasts}
    cursor = 0
    while cursor < replicates:
        count = min(2_000, replicates - cursor)
        indices = generator.integers(0, ROWS, size=(count, ROWS))
        for name, (reference, candidate) in contrasts.items():
            base = errors[reference][indices].mean(axis=1)
            trial = errors[candidate][indices].mean(axis=1)
            values[name][cursor : cursor + count] = (base - trial) / base
        cursor += count
    return {
        name: {
            "relative_reduction": relative_reduction(
                float(errors[reference].mean()), float(errors[candidate].mean())
            ),
            "paired_center_bootstrap_95_percent": [
                float(np.quantile(values[name], 0.025)),
                float(np.quantile(values[name], 0.975)),
            ],
            "probability_reduction_positive": float(np.mean(values[name] > 0)),
        }
        for name, (reference, candidate) in contrasts.items()
    }


def _stratum_metrics(
    labels: np.ndarray, errors: dict[str, np.ndarray], sums: dict[str, np.ndarray], counts: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in sorted(set(labels.tolist()), key=str):
        rows = labels == label
        result[str(label)] = {
            "centers": int(rows.sum()),
            "target_tokens": int(counts[rows].sum()),
            "equal_center_cosine_error": {
                arm: float(errors[arm][rows].mean()) for arm in ARMS
            },
            "token_weighted_cosine_error": {
                arm: float(sums[arm][rows].sum() / counts[rows].sum()) for arm in ARMS
            },
        }
    return result


def _save_csv(
    path: Path,
    selection: list[dict[str, Any]],
    folds: np.ndarray,
    errors: dict[str, np.ndarray],
    sums: dict[str, np.ndarray],
    counts: np.ndarray,
) -> None:
    fields = ["center_index", "sample_id", "scenario", "held_fold", "target_tokens"]
    for arm in ARMS:
        fields.extend([f"{arm}_{ARM_NAMES[arm]}_equal_center_cosine", f"{arm}_{ARM_NAMES[arm]}_token_error_sum"])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selection):
            record: dict[str, Any] = {
                "center_index": index,
                "sample_id": row["sample_id"],
                "scenario": row["scenario"],
                "held_fold": int(folds[index]),
                "target_tokens": int(counts[index]),
            }
            for arm in ARMS:
                record[f"{arm}_{ARM_NAMES[arm]}_equal_center_cosine"] = float(errors[arm][index])
                record[f"{arm}_{ARM_NAMES[arm]}_token_error_sum"] = float(sums[arm][index])
            writer.writerow(record)


def run(output: Path, audit_of: Path | None = None) -> dict[str, Any]:
    lock, protocol, selection = verify_lock()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for relative in lock["pinned_sources"]:
        source = ROOT / relative
        shutil.copy2(source, snapshot / source.name)

    request = {
        "status": "MATCHED_FACTORIAL_REQUEST_WRITTEN_BEFORE_TEACHER_EVALUATION",
        "study_id": protocol["study_id"],
        "execution_lock_sha256": sha256_file(LOCK),
        "protocol_sha256": sha256_file(PROTOCOL),
        "sample_ids": [row["sample_id"] for row in selection],
        "zero_access_contract": protocol["authorization"],
    }
    write_json_exclusive(output / "request.json", request)

    teacher = np.load(CACHE / "teacher_tokens.npy", mmap_mode="r", allow_pickle=False)
    masked = np.load(CACHE / "masked_tokens.npy", mmap_mode="r", allow_pickle=False)
    neighbors = np.load(CACHE / "neighbor_tokens.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(CACHE / "target_masks.npy", mmap_mode="r", allow_pickle=False)
    visible = np.load(CACHE / "visible_masks.npy", mmap_mode="r", allow_pickle=False)
    neighbor_valid = np.load(CACHE / "neighbor_valid.npy", mmap_mode="r", allow_pickle=False)
    cached_p3 = np.load(TRANSPORT / "true_transport_tokens.npy", mmap_mode="r", allow_pickle=False)
    cached_p3_available = np.load(TRANSPORT / "true_transport_available.npy", mmap_mode="r", allow_pickle=False)

    errors = {arm: np.zeros(ROWS, dtype=np.float64) for arm in ARMS}
    sums = {arm: np.zeros(ROWS, dtype=np.float64) for arm in ARMS}
    counts = np.zeros(ROWS, dtype=np.int64)
    donor_histogram = np.zeros(5, dtype=np.int64)
    affine_observations = 0
    common_observations = 0
    excluded_same_grid_invalid = 0
    affected_targets = 0
    cache_crosscheck_targets = 0
    cache_crosscheck_exact_fp16 = 0
    fit_receipts: list[dict[str, Any]] = []

    for center in range(ROWS):
        center_errors = {arm: [] for arm in ARMS}
        for mask_id in range(MASKS):
            target_mask = np.asarray(targets[center, mask_id], dtype=bool)
            positions = np.argwhere(target_mask)
            target_count = len(positions)
            donor_values_affine = np.zeros((DONORS, target_count, DIMENSION), dtype=np.float32)
            donor_values_same = np.zeros_like(donor_values_affine)
            affine_observed = np.zeros((DONORS, target_count), dtype=bool)
            same_valid = np.zeros_like(affine_observed)
            donor_logits = np.full((DONORS, target_count), -np.inf, dtype=np.float64)
            row_receipts = []
            center_tokens = np.asarray(masked[center, mask_id], dtype=np.float32)
            center_valid = np.asarray(visible[center, mask_id], dtype=bool)
            for donor in range(DONORS):
                donor_tokens = np.asarray(neighbors[center, donor], dtype=np.float32)
                valid = np.asarray(neighbor_valid[center, donor], dtype=bool)
                transform, matches = fit_visible_anchor_transport(
                    center_tokens,
                    donor_tokens,
                    target_mask,
                    center_valid=center_valid,
                    neighbor_valid=valid,
                )
                values, observed = bilinear_transport(
                    donor_tokens, target_mask, transform, neighbor_valid=valid
                )
                logits = donor_confidence_logits(matches, transform, target_mask, observed)
                y, x = positions[:, 0], positions[:, 1]
                donor_values_affine[donor] = values
                donor_values_same[donor] = donor_tokens[y, x]
                affine_observed[donor] = observed
                same_valid[donor] = valid[y, x]
                donor_logits[donor] = logits
                row_receipts.append(
                    {
                        "offset": OFFSETS[donor],
                        "valid_affine": transform.valid,
                        "match_count": transform.match_count,
                        "weighted_residual": float(transform.weighted_residual) if np.isfinite(transform.weighted_residual) else None,
                        "affine_observed_targets": int(observed.sum()),
                        "same_grid_valid_targets": int(same_valid[donor].sum()),
                    }
                )
            common = affine_observed & same_valid
            affine_observations += int(affine_observed.sum())
            common_observations += int(common.sum())
            excluded_same_grid_invalid += int((affine_observed & ~same_valid).sum())
            affected = np.any(affine_observed != common, axis=0)
            affected_targets += int(affected.sum())
            donor_histogram += np.bincount(common.sum(axis=0), minlength=5)
            if not np.all(common.any(axis=0)):
                raise RuntimeError("At least one target lacks a common matched donor")

            cells = {
                "A": uniform_consensus(donor_values_same, common, donor_logits),
                "B": robust_consensus(donor_values_same, common, donor_logits),
                "C": uniform_consensus(donor_values_affine, common, donor_logits),
                "D": robust_consensus(donor_values_affine, common, donor_logits),
            }
            if not all(np.array_equal(cell.available, cells["A"].available) for cell in cells.values()):
                raise RuntimeError("Factorial cells do not share target availability")
            teacher_values = np.asarray(teacher[center][target_mask], dtype=np.float32)
            for arm, cell in cells.items():
                values = cosine_errors(cell.features, teacher_values)
                center_errors[arm].append(values)
                sums[arm][center] += values.sum(dtype=np.float64)

            unaffected = ~affected
            cached_values = np.asarray(cached_p3[center, mask_id][target_mask])
            cached_available = np.asarray(cached_p3_available[center, mask_id][target_mask])
            if not cached_available.all():
                raise RuntimeError("Audited P3 cache unexpectedly lacks a target")
            cache_crosscheck_targets += int(unaffected.sum())
            cache_crosscheck_exact_fp16 += int(
                np.all(cells["D"].features[unaffected].astype(np.float16) == cached_values[unaffected], axis=1).sum()
            )
            counts[center] += target_count
            fit_receipts.append(
                {
                    "center_index": center,
                    "mask_id": mask_id,
                    "target_tokens": target_count,
                    "common_donor_observations": int(common.sum()),
                    "affected_targets_from_intersection": int(affected.sum()),
                    "donors": row_receipts,
                }
            )
        for arm in ARMS:
            combined = np.concatenate(center_errors[arm])
            errors[arm][center] = combined.mean()
            if len(combined) != counts[center] or not np.isclose(sums[arm][center], combined.sum()):
                raise RuntimeError("Per-center metric accounting changed")

    if int(counts.sum()) != EXPECTED_TARGETS or np.any(counts <= 0):
        raise RuntimeError("Matched target population changed")
    if cache_crosscheck_exact_fp16 != cache_crosscheck_targets:
        raise RuntimeError("Recomputed D differs from cached P3 on unaffected targets")

    scenarios = np.asarray([row["scenario"] for row in selection])
    fold_map = {key: int(value) for key, value in protocol["scenario_fold_map"].items()}
    folds = np.asarray([fold_map[scenario] for scenario in scenarios], dtype=np.int64)
    means = {arm: float(errors[arm].mean()) for arm in ARMS}
    token_weighted = {
        arm: float(sums[arm].sum() / counts.sum()) for arm in ARMS
    }
    by_fold = _stratum_metrics(folds, errors, sums, counts)
    by_scenario = _stratum_metrics(scenarios, errors, sums, counts)
    fold_means = {
        arm: [by_fold[str(fold)]["equal_center_cosine_error"][arm] for fold in range(5)]
        for arm in ARMS
    }
    decision = locked_factorial_decision(means, fold_means)
    bootstrap = _paired_bootstrap(errors)

    per_center = np.stack([errors[arm] for arm in ARMS])
    per_center_sums = np.stack([sums[arm] for arm in ARMS])
    np.save(output / "per_center_cosine_error.npy", per_center, allow_pickle=False)
    np.save(output / "per_center_token_error_sum.npy", per_center_sums, allow_pickle=False)
    np.save(output / "target_count.npy", counts, allow_pickle=False)
    write_json_exclusive(output / "fit_receipts.json", {"rows": fit_receipts})
    write_json_exclusive(output / "by_fold_and_scenario.json", {"by_fold": by_fold, "by_scenario": by_scenario})
    _save_csv(output / "per_center.csv", selection, folds, errors, sums, counts)

    data_artifacts = (
        "per_center_cosine_error.npy",
        "per_center_token_error_sum.npy",
        "target_count.npy",
        "fit_receipts.json",
        "by_fold_and_scenario.json",
        "per_center.csv",
    )
    receipts = {
        name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
        for name in data_artifacts
    }
    write_json_exclusive(output / "artifact_receipts.json", receipts)
    summary = {
        "status": "CENTER_COMPLETION_MATCHED_FACTORIAL_COMPLETE",
        "scope": "label-blind decoder-free reconstruction diagnostic; not task performance",
        "population": {
            "centers": ROWS,
            "mask_rows": ROWS * MASKS,
            "target_tokens": int(counts.sum()),
            "scenarios": len(set(scenarios.tolist())),
            "folds": len(set(folds.tolist())),
        },
        "cells": ARM_NAMES,
        "mean_equal_center_cosine_error": means,
        "mean_token_weighted_cosine_error": token_weighted,
        "paired_bootstrap": bootstrap,
        "locked_decision": decision,
        "matching_diagnostics": {
            "affine_donor_observations": affine_observations,
            "common_donor_observations": common_observations,
            "excluded_affine_observations_due_to_same_grid_invalidity": excluded_same_grid_invalid,
            "targets_affected_by_intersection": affected_targets,
            "donor_count_histogram_target_tokens_0_to_4": donor_histogram.tolist(),
            "all_target_tokens_have_common_support": bool(donor_histogram[0] == 0),
            "unaffected_D_targets_exactly_match_audited_P3_cache_after_fp16_rounding": cache_crosscheck_exact_fp16,
            "unaffected_D_targets_checked": cache_crosscheck_targets,
        },
        "zero_access_counters": {
            "task_labels": 0,
            "arftr_probability_arrays": 0,
            "pose_outputs": 0,
            "support_categories": 0,
            "optimizer_updates": 0,
            "task_fits": 0,
            "router_fits": 0,
        },
        "interpretive_limits": protocol["interpretation_limits"],
        "request_sha256": sha256_file(output / "request.json"),
        "artifact_receipts_sha256": sha256_file(output / "artifact_receipts.json"),
        "artifacts": receipts,
    }
    write_json_exclusive(output / "summary.json", summary)

    if audit_of is not None:
        expected_files = (*data_artifacts, "artifact_receipts.json", "summary.json", "request.json")
        comparisons = {}
        for name in expected_files:
            observed_hash = sha256_file(output / name)
            expected_hash = sha256_file(audit_of / name)
            comparisons[name] = {
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "exact": observed_hash == expected_hash,
            }
        audit_receipt = {
            "status": "CENTER_COMPLETION_MATCHED_FACTORIAL_EXACT_REPLAY_PASS"
            if all(row["exact"] for row in comparisons.values())
            else "CENTER_COMPLETION_MATCHED_FACTORIAL_EXACT_REPLAY_FAIL",
            "audited_output": str(audit_of.resolve()),
            "files": comparisons,
            "zero_access_counters": summary["zero_access_counters"],
        }
        write_json_exclusive(output / "audit_receipt.json", audit_receipt)
        if "FAIL" in audit_receipt["status"]:
            raise RuntimeError("Independent matched-factorial replay differed")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args.output, args.audit_of)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
