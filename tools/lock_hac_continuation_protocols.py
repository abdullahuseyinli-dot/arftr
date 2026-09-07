"""Prepare, create, or validate the 2026-09-07 HAC continuation locks.

The source-lock phase binds committed protocol and execution sources plus only inputs
whose development roles are independently established.  The execution-lock phase
binds role-safe artifacts produced after that source lock.  Neither phase discovers
inputs by scanning a directory, and protected paths are rejected before file access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from hac.retained_lock import validate_retained_lock

SHA256_RE = re.compile(r"[0-9a-f]{64}")

DATASET_CONFIG = {
    "okutama": {
        "spec": "experiments/okutama_cptr_continuation_protocol.json",
        "document": "docs/HAC_OKUTAMA_CPTR_CONTINUATION_PROTOCOL_20260907.md",
        "source_status": "OKUTAMA_CPTR_REPLAY_MATERIALIZATION_LOCKED_BEFORE_FEATURE_ACCESS",
        "execution_status": "OKUTAMA_CPTR_FROZEN_REPLAY_LOCKED_BEFORE_EXECUTION",
        "temporal_controls_execution_status": ("OKUTAMA_TEMPORAL_CONTROLS_LOCKED_BEFORE_FITTING"),
        "default_source_output": (
            ".runs/research_20260907/protocol_locks/okutama_materialization_lock.json"
        ),
        "default_execution_output": (
            ".runs/research_20260907/protocol_locks/okutama_r1_execution_lock.json"
        ),
        "temporal_controls_execution_output": (
            ".runs/research_20260907/protocol_locks/okutama_temporal_controls_execution_lock.json"
        ),
        "sources": {
            "protocol_spec": "experiments/okutama_cptr_continuation_protocol.json",
            "protocol_document": "docs/HAC_OKUTAMA_CPTR_CONTINUATION_PROTOCOL_20260907.md",
            "next_run_plan": "docs/HAC_NEXT_RUN_PLAN_20260906.md",
            "engineering_review": "docs/HAC_ENGINEERING_RESEARCH_REVIEW_20260906.md",
            "continuation_provenance": "docs/HAC_CONTINUATION_PROVENANCE_20260906.md",
            "bundle_builder": "experiments/build_okutama_cptr_replay_bundle.py",
            "replay_runner": "experiments/replay_okutama_cptr_interventions.py",
            "temporal_controls_runner": "experiments/run_okutama_temporal_controls.py",
            "cptr_model_module": "src/hac/cptr.py",
            "cptr_feature_module": "src/hac/cptr_features.py",
            "cptr_training_module": "src/hac/cptr_training.py",
            "temporal_model_module": "src/hac/vcoco_v3_temporal.py",
            "temporal_training_module": "src/hac/vcoco_v3_temporal_training.py",
            "neural_module": "src/hac/vcoco_v3_neural.py",
            "polar_training_module": "src/hac/polar_training.py",
            "historical_crossfit_plan": "experiments/okutama_cptr_crossfit_plan.json",
            "historical_candidate_grid": "experiments/okutama_cptr_adaptive_grid.json",
            "locker": "tools/lock_hac_continuation_protocols.py",
            "requirements": "requirements-v3-lock.txt",
        },
    },
    "vcoco": {
        "spec": "experiments/vcoco_continuation_protocol.json",
        "document": "docs/HAC_VCOCO_CONTINUATION_PROTOCOL_20260907.md",
        "source_status": "VCOCO_CONTINUATION_SOURCE_LOCKED_BEFORE_V0_FITTING",
        "execution_status": "VCOCO_CONTINUATION_EXECUTION_LOCKED_BEFORE_FITTING",
        "default_source_output": ".runs/research_20260907/protocol_locks/vcoco_source_lock.json",
        "default_execution_output": (
            ".runs/research_20260907/protocol_locks/vcoco_execution_lock.json"
        ),
        "sources": {
            "protocol_spec": "experiments/vcoco_continuation_protocol.json",
            "protocol_document": "docs/HAC_VCOCO_CONTINUATION_PROTOCOL_20260907.md",
            "next_run_plan": "docs/HAC_NEXT_RUN_PLAN_20260906.md",
            "engineering_review": "docs/HAC_ENGINEERING_RESEARCH_REVIEW_20260906.md",
            "continuation_provenance": "docs/HAC_CONTINUATION_PROVENANCE_20260906.md",
            "nested_model_module": "src/hac/vcoco_v3_models.py",
            "v0_runner": "experiments/run_vcoco_continuation_v0.py",
            "locker": "tools/lock_hac_continuation_protocols.py",
            "requirements": "requirements-v3-lock.txt",
        },
    },
}

FORBIDDEN_INPUT_PATHS = {
    ".runs/vcoco_v3/temporal/development_manifest.csv",
    ".runs/vcoco_v3/okutama/features/dinov2_base/development_metadata.csv",
    ".runs/polar_v2/locked_protocol/vcoco_test_clean.csv",
}
PROTECTED_COMPONENTS = {"calibration", "confirmation", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    parser.add_argument("--phase", choices=("source", "execution"), default="source")
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), default="prepare")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument(
        "--source-archive",
        type=Path,
        help=(
            "External Okutama TrainSetFrames.zip. It is hashed as an opaque byte stream "
            "and checked against the independently audited receipt."
        ),
    )
    parser.add_argument("--input-inventory", type=Path)
    parser.add_argument("--eligible-index", type=Path)
    parser.add_argument("--feature-bundle", type=Path)
    parser.add_argument("--bundle-summary", type=Path)
    parser.add_argument("--eligible-window-masks", type=Path)
    parser.add_argument("--window-mask-summary", type=Path)
    parser.add_argument("--fold-map", type=Path)
    parser.add_argument("--bootstrap-group-indices", type=Path)
    parser.add_argument("--swap-signs-packbits", type=Path)
    parser.add_argument("--statistics-randomization-receipt", type=Path)
    parser.add_argument("--scenario-resampling", type=Path)
    parser.add_argument("--prepare-summary", type=Path)
    parser.add_argument("--execution-runner", type=Path)
    parser.add_argument(
        "--stage",
        choices=("r1b", "temporal-controls", "v0", "v1"),
        help="Required for an execution lock; source locks cover the complete protocol.",
    )
    return parser.parse_args()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        content = handle.read()
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while it was captured: {path}")
    return content


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    content = _read_bytes(path)
    payload = json.loads(content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload, content


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while it was hashed: {path}")
    return digest.hexdigest(), int(before.st_size)


def _relative_path(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Input is outside the repository: {path}") from error


def assert_role_safe_input_path(root: Path, path: Path) -> str:
    """Reject known mixed manifests and any lexical protected-role input."""

    relative = _relative_path(root, path)
    normalized = str(PurePosixPath(relative)).lower()
    if normalized in FORBIDDEN_INPUT_PATHS:
        raise RuntimeError(f"Forbidden role-mixed or protected input path: {relative}")
    components = {component.lower() for component in PurePosixPath(normalized).parts}
    if components.intersection(PROTECTED_COMPONENTS):
        raise RuntimeError(f"Protected-role input path: {relative}")
    return relative


def validate_protocol_spec(dataset: str, spec: dict[str, Any]) -> None:
    expected_status = {
        "okutama": "DECLARED_BEFORE_OKUTAMA_CPTR_CONTINUATION_EXECUTION",
        "vcoco": "DECLARED_BEFORE_VCOCO_CONTINUATION_FITTING",
    }[dataset]
    if spec.get("status") != expected_status or spec.get("protocol_version") != "1.0.0":
        raise RuntimeError(f"Unexpected {dataset} continuation protocol status or version")
    if spec.get("declared_on") != "2026-09-07":
        raise RuntimeError("Continuation protocol declaration date changed")
    for item in spec.get("historical_input_receipts", []):
        if SHA256_RE.fullmatch(str(item.get("sha256", ""))) is None:
            raise RuntimeError(f"Invalid historical input digest: {item.get('name')}")
    if dataset == "okutama":
        data = spec["data_contract"]
        if (data.get("eligible_total_rows"), data.get("eligible_total_scenarios")) != (6360, 14):
            raise RuntimeError("Okutama eligible development scope changed")
        if spec["checkpoint_inventory"].get("folds") != 5:
            raise RuntimeError("Okutama fold count changed")
        if spec["checkpoint_inventory"].get("seeds") != [42, 43, 44, 45, 46]:
            raise RuntimeError("Okutama seed set changed")
        if spec["execution_stages"]["R1a"].get("seeds") != [43]:
            raise RuntimeError("R1a must remain the five-fold seed-43 replay")
        reproduction = spec.get("replay_reproduction_contract", {})
        if (
            reproduction.get("current_gpu_teacher_max_abs_tolerance") != 0.0005
            or reproduction.get("F3_statistical_probabilities")
            != "exact_retained_teacher_probabilities"
        ):
            raise RuntimeError("Okutama retained-teacher reproduction contract changed")
        retained = spec.get("role_safe_retained_teacher_bundle_contract", {})
        if (
            retained.get("probabilities_shape") != [4977, 5, 3]
            or retained.get("probabilities_dtype") != "float32"
            or retained.get("seed_values") != [42, 43, 44, 45, 46]
        ):
            raise RuntimeError("Okutama retained-teacher bundle contract changed")
        if (
            spec["execution_stages"]["R1b"].get("requires_exact_role_specific_window_masks")
            is not True
        ):
            raise RuntimeError("R1b must require exact role-specific window masks")
        controls = spec.get("temporal_controls", {})
        if controls.get("data_scope") != {
            "rows": 4977,
            "scenarios": 11,
            "scope_value": "grouped_crossfit_oof",
            "folds": 5,
            "seeds": [42, 43, 44, 45, 46],
        }:
            raise RuntimeError("Okutama temporal-control data scope changed")
        if controls.get("training", {}).get("fixed_epochs_by_seed") != {
            "42": 5,
            "43": 5,
            "44": 5,
            "45": 5,
            "46": 3,
        }:
            raise RuntimeError("Okutama temporal-control epoch schedule changed")
        if controls.get("T2_architecture", {}).get("distinct_indices") != [
            4,
            5,
            6,
            7,
            8,
            10,
            11,
            12,
        ]:
            raise RuntimeError("Okutama T2 sampler contract changed")
    else:
        data = spec["data_contract"]
        if spec.get("execution_determinism") != {
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }:
            raise RuntimeError("V0 deterministic execution contract changed")
        if (data.get("combined_development_people"), data.get("combined_development_images")) != (
            6640,
            4123,
        ):
            raise RuntimeError("V-COCO combined development scope changed")
        cross_validation = spec["cross_validation"]
        if (
            cross_validation.get("outer_folds"),
            cross_validation.get("inner_folds"),
            cross_validation.get("stack_folds"),
            cross_validation.get("random_seed"),
        ) != (5, 3, 3, 20260827):
            raise RuntimeError("V-COCO shared-fold contract changed")
        if spec["V0"].get("candidate_selection_budget_per_family") != 8:
            raise RuntimeError("V0 equal candidate budget changed")
        svm_grid = spec["V0"].get("linear_svm_grid", {})
        if (
            svm_grid.get("maximum_iterations") != 2000
            or svm_grid.get("tolerance") != 0.0001
            or svm_grid.get("require_iteration_limit_not_reached") is not True
        ):
            raise RuntimeError("V0 CUDA-SVM convergence contract changed")
        if (
            spec["V0"].get("inner_candidate_seed")
            != "20260907 + 10000 * outer_fold + candidate_index"
            or spec["V0"].get("inner_fit_estimator_seed")
            != "20260907 + 10000 * outer_fold + candidate_index + 10000 * (inner_fold + 1)"
        ):
            raise RuntimeError("V0 inner estimator seed contract changed")
        if spec["V0"].get("candidate_selection") != {
            "scope": "shared_inner_folds_only",
            "labels": "source_tag_labels_only",
            "aggregation": "metrics_on_concatenated_inner_oof_predictions_not_mean_fold_scores",
            "zero_division": 0,
            "ranking": [
                {"metric": "macro_f1", "direction": "descending"},
                {"metric": "locomotion_f1", "direction": "descending"},
                {"metric": "log_loss", "direction": "ascending"},
                {"metric": "candidate_id", "direction": "ascending"},
            ],
        }:
            raise RuntimeError("V0 candidate selection rule changed")
        geometry = spec["V0"].get("shared_geometry_features", {})
        if (
            geometry.get("dimensions") != 6
            or geometry.get("preprocessing") != "hac.vcoco_v3_models.geometry_features"
            or geometry.get("identical_across_all_families") is not True
        ):
            raise RuntimeError("V0 shared geometry contract changed")
        if cross_validation.get("selection_stack_seed_by_outer_and_inner_fold") != (
            "20260827 + 200000 + 1000 * outer_fold + inner_fold"
        ):
            raise RuntimeError("V-COCO inner-selection stack seed contract changed")
        if spec["V1"].get("status") != "BLOCKED_PENDING_SPATIAL_AND_AUXILIARY_INPUTS":
            raise RuntimeError("V1 must remain blocked until its missing inputs are bound")
        statistics = spec.get("statistics", {})
        if (
            statistics.get("bootstrap_resamples"),
            statistics.get("bootstrap_seed"),
            statistics.get("swap_monte_carlo_draws"),
            statistics.get("pre_fit_lock_required"),
            statistics.get("shared_across_all_declared_comparisons"),
        ) != (10000, 20260906, 100000, True, True):
            raise RuntimeError("V-COCO shared statistical-randomization contract changed")
        if statistics.get("bootstrap_indices_artifact", {}).get("shape") != [10000, 4123]:
            raise RuntimeError("V-COCO bootstrap-index shape changed")
        if statistics.get("swap_signs_artifact", {}).get("shape") != [100000, 516]:
            raise RuntimeError("V-COCO packed-swap-sign shape changed")
        resources = spec.get("resource_contract", {})
        if (
            resources.get("V0_scope")
            != "locked_cached_development_features_through_nested_heads"
            or resources.get("end_to_end_pipeline_resource_claim_authorized") is not False
            or resources.get("V0_measures")
            != [
                "bounded_outer_fold_wall_time",
                "complete_nested_run_wall_time",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "cuda_svm_iteration_audit",
            ]
        ):
            raise RuntimeError("V0 resource-scope contract changed")


def git_source_receipt(root: Path, relative: str, commit: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    canonical = _relative_path(root, path)
    try:
        blob = _git(root, "rev-parse", f"{commit}:{canonical}")
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Source is not committed at {commit}: {canonical}") from error
    working_blob = _git(root, "hash-object", canonical)
    if working_blob != blob:
        raise RuntimeError(f"Working source differs from committed blob: {canonical}")
    digest, size = _sha256_file(path)
    return {
        "path": canonical,
        "git_blob_oid": blob,
        "sha256": digest,
        "size_bytes": size,
    }


def committed_source_preconditions(
    root: Path, config: dict[str, Any], commit: str
) -> list[dict[str, str]]:
    """Report source commit gaps without reading any experiment input artifact."""

    issues: list[dict[str, str]] = []
    for name, relative in config["sources"].items():
        path = (root / relative).resolve()
        if not path.is_file():
            issues.append({"name": name, "path": relative, "issue": "missing"})
            continue
        try:
            committed_blob = _git(root, "rev-parse", f"{commit}:{relative}")
        except subprocess.CalledProcessError:
            issues.append({"name": name, "path": relative, "issue": "not_committed"})
            continue
        working_blob = _git(root, "hash-object", relative)
        if working_blob != committed_blob:
            issues.append({"name": name, "path": relative, "issue": "differs_from_committed_blob"})
    return issues


def _artifact_receipt(root: Path, relative: str, expected: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    safe_relative = assert_role_safe_input_path(root, path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual, size = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Locked SHA-256 mismatch for {safe_relative}: expected {expected}, got {actual}"
        )
    return {"path": safe_relative, "sha256": actual, "size_bytes": size}


def _okutama_input_receipts(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for item in spec["historical_input_receipts"]:
        receipt = _artifact_receipt(root, item["path"], item["sha256"])
        receipt["role"] = item["role"]
        receipts[item["name"]] = receipt

    epoch_proof_path = root / ".runs/vcoco_v3/temporal/development_final/summary.json"
    epoch_proof, _ = _read_json(epoch_proof_path)
    if (
        epoch_proof.get("status") != "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE"
        or epoch_proof.get("seeds") != [42, 43, 44, 45, 46]
        or epoch_proof.get("fixed_epochs", {}).get("teacher")
        != {"42": 5, "43": 5, "44": 5, "45": 5, "46": 3}
        or int(epoch_proof.get("calibration_samples_read", -1)) != 0
        or int(epoch_proof.get("confirmation_samples_read", -1)) != 0
    ):
        raise RuntimeError("Okutama temporal fixed-epoch proof is unsafe or changed")

    summary_path = root / ".runs/cptr/baseline_preservation_diagnostic/summary.json"
    summary, _ = _read_json(summary_path)
    if summary.get("status") != "OKUTAMA_CPTR_BASELINE_PRESERVATION_DIAGNOSTIC_COMPLETE":
        raise RuntimeError("Okutama role-safe row receipt is incomplete")

    audit_path = root / ".runs/vcoco_v3/okutama/development_audit/summary.json"
    audit, _ = _read_json(audit_path)
    if audit.get("status") != "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED":
        raise RuntimeError("Okutama aggregate development audit is incomplete")
    if audit.get("confirmation_archive_opened") is not False:
        raise RuntimeError("Okutama audit reports confirmation access")
    archive = audit.get("development_archive", {})
    receipts["source_archive_attestation"] = {
        "file_name": "TrainSetFrames.zip",
        "bytes": int(archive.get("bytes", -1)),
        "sha256": str(archive.get("sha256", "")),
        "verification_policy": (
            "the supplied archive is opaque-hashed at source-lock creation and must match "
            "this independently established receipt"
        ),
    }
    if receipts["source_archive_attestation"]["bytes"] != 5_770_432_522:
        raise RuntimeError("Okutama source archive byte attestation changed")
    if SHA256_RE.fullmatch(receipts["source_archive_attestation"]["sha256"]) is None:
        raise RuntimeError("Okutama source archive SHA-256 attestation is invalid")

    checkpoints = []
    inventory = spec["checkpoint_inventory"]
    fixed_epochs = {42: 2, 43: 0, 44: 1, 45: 9, 46: 1}
    for family, template in inventory["families"].items():
        for fold in range(int(inventory["folds"])):
            for seed in map(int, inventory["seeds"]):
                directory = template.format(fold=fold, seed=seed)
                summary_relative = f"{directory}/summary.json"
                checkpoint_relative = f"{directory}/checkpoint.pt"
                summary_file = (root / summary_relative).resolve()
                assert_role_safe_input_path(root, summary_file)
                payload, content = _read_json(summary_file)
                expected_status = (
                    "OKUTAMA_CPTR_CROSSFIT_RUN_COMPLETE"
                    if family == "cptr"
                    else "VCOCO_V3_TEMPORAL_CROSSFIT_RUN_COMPLETE"
                )
                if payload.get("status") != expected_status:
                    raise RuntimeError(
                        f"Incomplete development checkpoint receipt: {summary_relative}"
                    )
                if (int(payload.get("fold", -1)), int(payload.get("seed", -1))) != (fold, seed):
                    raise RuntimeError(f"Fold/seed mismatch in {summary_relative}")
                for counter in (
                    "calibration_samples_read",
                    "confirmation_samples_read",
                    "validation_samples_read",
                ):
                    if counter not in payload or type(payload[counter]) is not int or payload[counter] != 0:
                        raise RuntimeError(
                            f"Missing or nonzero protected-access receipt in "
                            f"{summary_relative}: {counter}"
                        )
                if family == "cptr" and int(payload.get("fixed_epochs", -1)) != fixed_epochs[seed]:
                    raise RuntimeError(f"CPTR fixed epoch mismatch in {summary_relative}")
                if family != "cptr" and payload.get("model_role") != family:
                    raise RuntimeError(f"Baseline role mismatch in {summary_relative}")
                expected_checkpoint = str(
                    payload.get("artifact_sha256", {}).get("checkpoint.pt", "")
                )
                if SHA256_RE.fullmatch(expected_checkpoint) is None:
                    raise RuntimeError(f"Missing checkpoint digest in {summary_relative}")
                checkpoint = _artifact_receipt(root, checkpoint_relative, expected_checkpoint)
                checkpoints.append(
                    {
                        "family": family,
                        "fold": fold,
                        "seed": seed,
                        "fixed_epochs": fixed_epochs[seed] if family == "cptr" else None,
                        "summary": {
                            "path": summary_relative,
                            "sha256": _sha256(content),
                            "size_bytes": len(content),
                        },
                        "checkpoint": checkpoint,
                    }
                )
    if len(checkpoints) != 75:
        raise RuntimeError("Okutama checkpoint inventory is incomplete")
    return {"receipts": receipts, "checkpoints": checkpoints}


def _opaque_okutama_feature_array_receipts(
    root: Path, inputs: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Hash declared mixed-store files without decoding or indexing array values."""

    store_contracts = {
        "base": ("base_store_declaration", {"tight", "context", "geometry"}),
        "part": ("part_store_declaration", {"part_tokens", "part_confidence"}),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for store_name, (receipt_name, expected_names) in store_contracts.items():
        store_receipt = inputs.get("receipts", {}).get(receipt_name, {})
        declaration_path = (root / str(store_receipt.get("path", ""))).resolve()
        declaration, _ = _read_json(declaration_path)
        declared_arrays = declaration.get("arrays")
        if not isinstance(declared_arrays, dict) or set(declared_arrays) != expected_names:
            raise RuntimeError(f"Unexpected {store_name} feature-array declaration")
        for array_name in sorted(expected_names):
            item = declared_arrays[array_name]
            relative = Path(str(item.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe declared feature-array path: {array_name}")
            path = (declaration_path.parent / relative).resolve()
            safe_path = assert_role_safe_input_path(root, path)
            digest, size = _sha256_file(path)
            if digest != item.get("sha256"):
                raise RuntimeError(
                    f"Declared feature-array digest mismatch: {store_name}/{array_name}"
                )
            receipts[f"{store_name}_{array_name}"] = {
                "path": safe_path,
                "sha256": digest,
                "size_bytes": size,
                "verification": "opaque_byte_stream_no_array_decoding_or_indexing",
            }
    return receipts


def _vcoco_input_receipts(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for item in spec["historical_input_receipts"]:
        receipt = _artifact_receipt(root, item["path"], item["sha256"])
        receipt["role"] = item["role"]
        receipts[item["name"]] = receipt

    features = []
    for item in spec["pooled_feature_inputs"]:
        directory = item["directory"]
        provenance_relative = f"{directory}/provenance.json"
        provenance_path = (root / provenance_relative).resolve()
        provenance_receipt = _artifact_receipt(root, provenance_relative, item["receipt_sha256"])
        provenance, _ = _read_json(provenance_path)
        if provenance.get("status") != "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE":
            raise RuntimeError(f"Incomplete V-COCO development feature cache: {directory}")
        if (
            provenance.get("test_rows_read") != 0
            or provenance.get("test_predictions_run") is not False
        ):
            raise RuntimeError(f"V-COCO feature cache reports official-test access: {directory}")
        if int(provenance.get("rows", -1)) != 6640:
            raise RuntimeError(f"Unexpected V-COCO development row count: {directory}")
        if provenance.get("artifact_sha256", {}).get("features.npy") != item["features_sha256"]:
            raise RuntimeError(f"V-COCO feature digest receipt changed: {directory}")
        if provenance.get("artifact_sha256", {}).get("rows.csv") != item["rows_sha256"]:
            raise RuntimeError(f"V-COCO row-map digest receipt changed: {directory}")
        feature_receipt = _artifact_receipt(
            root, f"{directory}/features.npy", item["features_sha256"]
        )
        rows_receipt = _artifact_receipt(root, f"{directory}/rows.csv", item["rows_sha256"])
        features.append(
            {
                "name": item["name"],
                "representation": item["representation"],
                "provenance": provenance_receipt,
                "features": feature_receipt,
                "rows": rows_receipt,
            }
        )
    return {"receipts": receipts, "pooled_features": features}


def collect_environment() -> dict[str, Any]:
    distributions = sorted(
        (
            {
                "name": distribution.metadata.get("Name", "").lower().replace("_", "-"),
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=lambda item: (item["name"], item["version"]),
    )
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": distributions,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        environment["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device": torch.cuda.get_device_name(0) if cuda_available else None,
            "compute_capability": (
                ".".join(map(str, torch.cuda.get_device_capability(0))) if cuda_available else None
            ),
        }
    except ImportError:
        environment["torch"] = {"available": False}
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    environment["canonical_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return environment


def _source_payload(
    root: Path, dataset: str, *, commit: str, args: argparse.Namespace
) -> dict[str, Any]:
    config = DATASET_CONFIG[dataset]
    spec_path = root / config["spec"]
    spec, _ = _read_json(spec_path)
    validate_protocol_spec(dataset, spec)
    sources = {
        name: git_source_receipt(root, relative, commit)
        for name, relative in config["sources"].items()
    }
    inputs = (
        _okutama_input_receipts(root, spec)
        if dataset == "okutama"
        else _vcoco_input_receipts(root, spec)
    )
    source_sha256 = {name: item["sha256"] for name, item in sources.items()}
    materialization_inputs: dict[str, Any] = {}
    if dataset == "okutama":
        required = {
            "input_inventory": args.input_inventory,
            "eligible_index": args.eligible_index,
            "eligible_window_masks": args.eligible_window_masks,
            "window_mask_summary": args.window_mask_summary,
        }
        missing_arguments = [name for name, path in required.items() if path is None]
        if missing_arguments:
            raise ValueError(
                "Okutama materialization locks require: " + ", ".join(missing_arguments)
            )
        inventory_path = required["input_inventory"].resolve()  # type: ignore[union-attr]
        assert_role_safe_input_path(root, inventory_path)
        inventory, _ = _read_json(inventory_path)
        if inventory.get("status") != "OKUTAMA_CPTR_REPLAY_INPUT_INVENTORY_COMPLETE":
            raise RuntimeError("Okutama replay input inventory is incomplete")
        if _eligible_scope(inventory) != (6360, 14):
            raise RuntimeError("Okutama role-safe inventory scope changed")
        for name, path in required.items():
            resolved = path.resolve()  # type: ignore[union-attr]
            relative = assert_role_safe_input_path(root, resolved)
            digest, size = _sha256_file(resolved)
            expected = _inventory_digest(inventory, name, Path(relative).name)
            if name != "input_inventory" and expected is not None and digest != expected:
                raise RuntimeError(f"Okutama inventory digest mismatch for {name}")
            materialization_inputs[name] = {
                "path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
            source_sha256[name] = digest
        mask_summary_path = required["window_mask_summary"].resolve()  # type: ignore[union-attr]
        mask_summary, _ = _read_json(mask_summary_path)
        if mask_summary.get("status") != "OKUTAMA_CPTR_ROLE_SAFE_EXACT_WINDOW_MASKS_COMPLETE":
            raise RuntimeError("Okutama exact-window-mask summary is incomplete")
        if _eligible_scope(mask_summary) != (6360, 14):
            raise RuntimeError("Okutama exact-window-mask scope changed")
        expected_mask = _inventory_digest(
            mask_summary,
            "eligible_window_masks",
            Path(required["eligible_window_masks"]).name,  # type: ignore[arg-type]
        )
        if expected_mask != materialization_inputs["eligible_window_masks"]["sha256"]:
            raise RuntimeError("Okutama exact-window-mask summary does not bind its artifact")
        expected_mask_sources = {
            "input_inventory": materialization_inputs["input_inventory"]["sha256"],
            "eligible_index": materialization_inputs["eligible_index"]["sha256"],
            "extractor": source_sha256["bundle_builder"],
        }
        for name, expected in expected_mask_sources.items():
            if mask_summary.get("source_sha256", {}).get(name) != expected:
                raise RuntimeError(f"Okutama exact-window-mask source mismatch: {name}")
        mask_access = mask_summary.get("access_accounting", {})
        for name in (
            "disallowed_annotation_members_read",
            "image_members_read",
            "mixed_development_manifest_rows_read",
            "broad_development_metadata_rows_read",
            "confirmation_rows_read",
            "test_rows_read",
            "feature_array_values_read",
            "checkpoints_loaded",
        ):
            if int(mask_access.get(name, -1)) != 0:
                raise RuntimeError(f"Unsafe access reported by exact-window-mask summary: {name}")
        feature_arrays = _opaque_okutama_feature_array_receipts(root, inputs)
        materialization_inputs["feature_arrays"] = feature_arrays
        for name, receipt in feature_arrays.items():
            source_sha256[f"feature_array_{name}"] = receipt["sha256"]
        if args.source_archive is None:
            raise ValueError("Okutama materialization locks require --source-archive")
        archive_path = args.source_archive.resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive_attestation = inputs["receipts"]["source_archive_attestation"]
        if archive_path.name != archive_attestation["file_name"]:
            raise RuntimeError("Okutama source archive filename differs from its receipt")
        if archive_path.stat().st_size != int(archive_attestation["bytes"]):
            raise RuntimeError("Okutama source archive byte count differs from its receipt")
        archive_digest, archive_size = _sha256_file(archive_path)
        if archive_digest != archive_attestation["sha256"]:
            raise RuntimeError("Okutama source archive SHA-256 differs from its receipt")
        if mask_summary.get("source_sha256", {}).get("archive") != archive_digest:
            raise RuntimeError("Okutama exact-window masks came from another source archive")
        materialization_inputs["source_archive"] = {
            "file_name": archive_path.name,
            "size_bytes": archive_size,
            "sha256": archive_digest,
            "path_recorded": False,
            "digest_source": "current_opaque_byte_hash_matches_independently_audited_receipt",
        }
        source_sha256["source_archive"] = archive_attestation["sha256"]

    return {
        "schema_version": 1,
        "status": config["source_status"],
        "dataset": dataset,
        "protocol_version": spec["protocol_version"],
        "repository_commit": commit,
        "repository_tree": _git(root, "rev-parse", f"{commit}^{{tree}}"),
        "source_sha256": source_sha256,
        "source_blobs": sources,
        "inputs": inputs,
        "materialization_inputs": materialization_inputs,
        "environment": collect_environment(),
        "protected_access": {
            "mixed_manifest_rows_read": 0,
            "mixed_development_metadata_rows_read": 0,
            "calibration_rows_or_arrays_read": 0,
            "confirmation_rows_or_arrays_read": 0,
            "test_rows_or_arrays_read": 0,
        },
        "authorization": (
            {
                "eligible_feature_bundle_construction": True,
                "temporal_controls_artifact_construction": True,
                "R1a_execution": False,
                "R1b_execution": False,
                "R2_fitting": False,
                "T1_fitting": False,
                "T2_fitting": False,
            }
            if dataset == "okutama"
            else {
                "shared_fold_map_construction": True,
                "shared_statistics_randomization_construction": True,
                "V0_fitting": False,
                "V1_fitting": False,
            }
        ),
        "disclosures": spec["access_disclosures"],
        "disclosed_prelock_access": spec.get("prelock_access_accounting", {}),
    }


def _inventory_digest(payload: dict[str, Any], *names: str) -> str | None:
    for container_name in ("artifact_sha256", "artifacts", "source_sha256"):
        container = payload.get(container_name, {})
        if not isinstance(container, dict):
            continue
        for name in names:
            value = container.get(name)
            if isinstance(value, dict):
                value = value.get("sha256")
            if isinstance(value, str) and SHA256_RE.fullmatch(value):
                return value
    return None


def _eligible_scope(payload: dict[str, Any]) -> tuple[int, int]:
    rows = payload.get("eligible_rows", payload.get("rows", -1))
    scenarios = payload.get("eligible_scenarios", payload.get("scenarios", -1))
    return int(rows), int(scenarios)


def _csv_rows(content: bytes, *, name: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise RuntimeError(f"Invalid CSV input: {name}") from error
    if not fieldnames or any(None in row for row in rows):
        raise RuntimeError(f"Malformed CSV schema: {name}")
    return fieldnames, rows


def _integer_field(row: dict[str, str], field: str, *, row_index: int) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {field} in V-COCO fold-map row {row_index}") from error
    return value


def validate_vcoco_fold_map(root: Path, fold_map_path: Path) -> None:
    spec, _ = _read_json(root / DATASET_CONFIG["vcoco"]["spec"])
    expected_columns = spec["cross_validation"]["fold_map"]["columns"]
    fold_content = _read_bytes(fold_map_path)
    columns, rows = _csv_rows(fold_content, name="V-COCO fold map")
    if columns != expected_columns:
        raise RuntimeError("V-COCO fold-map columns or order changed")
    if len(rows) != 6640:
        raise RuntimeError("V-COCO fold map must contain exactly 6640 people")

    source_path = root / spec["pooled_feature_inputs"][0]["directory"] / "rows.csv"
    source_content = _read_bytes(source_path)
    if _sha256(source_content) != spec["pooled_feature_inputs"][0]["rows_sha256"]:
        raise RuntimeError("V-COCO canonical feature rows changed")
    _, source_rows = _csv_rows(source_content, name="V-COCO canonical feature rows")
    if len(source_rows) != len(rows):
        raise RuntimeError("V-COCO fold map and feature rows differ in length")

    label_to_index = {"sitting": 0, "standing": 1, "walking_running": 2}
    person_ids: set[str] = set()
    image_outer: dict[str, int] = {}
    image_nested: dict[tuple[str, str], int] = {}
    outer_class_support = {fold: set() for fold in range(5)}
    nested_support = {
        (kind, outer, fold): set()
        for kind in ("inner", "stack")
        for outer in range(5)
        for fold in range(3)
    }
    selection_support = {
        (outer, inner, fold): set() for outer in range(5) for inner in range(3) for fold in range(3)
    }
    for index, (row, source) in enumerate(zip(rows, source_rows, strict=True)):
        expected_identity = (
            str(index),
            str(source.get("person_id", "")),
            str(source.get("image_id", "")),
            str(label_to_index.get(str(source.get("label_3", "")), -1)),
        )
        observed_identity = (
            row["row_index"],
            row["person_id"],
            row["image_id"],
            row["label_index"],
        )
        if observed_identity != expected_identity:
            raise RuntimeError(f"V-COCO fold-map identity mismatch at row {index}")
        if row["person_id"] in person_ids:
            raise RuntimeError("V-COCO fold map contains a duplicate person ID")
        person_ids.add(row["person_id"])
        outer = _integer_field(row, "outer_fold", row_index=index)
        label = _integer_field(row, "label_index", row_index=index)
        if outer not in range(5) or label not in range(3):
            raise RuntimeError(f"Invalid outer fold or label at V-COCO row {index}")
        previous_outer = image_outer.setdefault(row["image_id"], outer)
        if previous_outer != outer:
            raise RuntimeError("A V-COCO image crosses an outer-fold boundary")
        outer_class_support[outer].add(label)
        for held_outer in range(5):
            for kind in ("inner", "stack"):
                field = f"{kind}_fold_o{held_outer}"
                value = _integer_field(row, field, row_index=index)
                if outer == held_outer:
                    if value != -1:
                        raise RuntimeError("An outer-held V-COCO row enters a nested fold")
                    continue
                if value not in range(3):
                    raise RuntimeError(f"Invalid V-COCO nested fold at row {index}: {field}")
                key = (row["image_id"], field)
                previous = image_nested.setdefault(key, value)
                if previous != value:
                    raise RuntimeError("A V-COCO image crosses a nested-fold boundary")
                nested_support[(kind, held_outer, value)].add(label)
        for held_outer in range(5):
            inner_value = _integer_field(row, f"inner_fold_o{held_outer}", row_index=index)
            for held_inner in range(3):
                field = f"selection_stack_fold_o{held_outer}_i{held_inner}"
                value = _integer_field(row, field, row_index=index)
                eligible = outer != held_outer and inner_value != held_inner
                if not eligible:
                    if value != -1:
                        raise RuntimeError(
                            "An ineligible V-COCO row enters an inner-selection stack fold"
                        )
                    continue
                if value not in range(3):
                    raise RuntimeError(
                        f"Invalid V-COCO inner-selection stack fold at row {index}: {field}"
                    )
                key = (row["image_id"], field)
                previous = image_nested.setdefault(key, value)
                if previous != value:
                    raise RuntimeError(
                        "A V-COCO image crosses an inner-selection stack-fold boundary"
                    )
                selection_support[(held_outer, held_inner, value)].add(label)
    if any(support != {0, 1, 2} for support in outer_class_support.values()):
        raise RuntimeError("A V-COCO outer fold lacks a class")
    if any(support != {0, 1, 2} for support in nested_support.values()):
        raise RuntimeError("A V-COCO inner or stack fold lacks a class")
    if any(support != {0, 1, 2} for support in selection_support.values()):
        raise RuntimeError("A V-COCO inner-selection stack fold lacks a class")


def validate_vcoco_randomization_artifacts(
    root: Path,
    *,
    bootstrap_path: Path,
    swap_path: Path,
    receipt_path: Path,
    artifact_receipts: dict[str, dict[str, Any]],
) -> None:
    """Validate shared pre-fit statistical draws without materializing their values."""

    receipt, _ = _read_json(receipt_path)
    if receipt.get("status") != "VCOCO_CONTINUATION_SHARED_RANDOMIZATION_COMPLETE":
        raise RuntimeError("V-COCO shared-randomization receipt is incomplete")
    if (
        int(receipt.get("group_count", -1)),
        int(receipt.get("root_seed", -1)),
        str(receipt.get("bit_generator", "")),
    ) != (4123, 20260906, "PCG64"):
        raise RuntimeError("V-COCO shared-randomization contract changed")

    spec, _ = _read_json(root / DATASET_CONFIG["vcoco"]["spec"])
    canonical_rows_path = root / spec["pooled_feature_inputs"][0]["directory"] / "rows.csv"
    _, canonical_rows = _csv_rows(_read_bytes(canonical_rows_path), name="V-COCO rows")
    canonical_group_ids = sorted({str(row.get("image_id", "")) for row in canonical_rows})
    if len(canonical_group_ids) != 4123 or receipt.get("group_ids") != canonical_group_ids:
        raise RuntimeError("V-COCO randomization group IDs differ from canonical development")
    group_content = ("\n".join(canonical_group_ids) + "\n").encode()
    if receipt.get("group_ids_sha256") != _sha256(group_content):
        raise RuntimeError("V-COCO randomization group-ID digest changed")

    expected = {
        "bootstrap_group_indices.npy": (bootstrap_path, (10000, 4123), np.dtype("uint16")),
        "swap_signs_packbits.npy": (swap_path, (100000, 516), np.dtype("uint8")),
    }
    declared = receipt.get("artifacts", {})
    if not isinstance(declared, dict) or set(declared) != set(expected):
        raise RuntimeError("V-COCO randomization artifact receipt set changed")
    for name, (path, shape, dtype) in expected.items():
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        observed_shape = tuple(map(int, values.shape))
        observed_dtype = values.dtype
        del values
        if observed_shape != shape or observed_dtype != dtype:
            raise RuntimeError(f"V-COCO randomization array shape/dtype changed: {name}")
        declaration = declared[name]
        actual = artifact_receipts[
            "bootstrap_group_indices" if path == bootstrap_path else "swap_signs_packbits"
        ]["sha256"]
        if (
            declaration.get("sha256") != actual
            or declaration.get("shape") != list(shape)
            or declaration.get("dtype") != dtype.name
        ):
            raise RuntimeError(f"V-COCO randomization receipt mismatch: {name}")


def validate_okutama_retained_teacher_bundle(bundle_path: Path) -> None:
    """Enforce the exact role-safe retained-teacher member contract."""

    with np.load(bundle_path, allow_pickle=False) as bundle:
        required = {
            "sample_ids",
            "scope",
            "primary_retained_teacher_sample_ids",
            "retained_teacher_seeds",
            "primary_retained_teacher_probabilities",
        }
        if not required.issubset(bundle.files):
            raise RuntimeError("Okutama bundle lacks retained-teacher members")
        sample_ids = np.asarray(bundle["sample_ids"]).astype(str)
        scope = np.asarray(bundle["scope"]).astype(str)
        teacher_ids_raw = np.asarray(bundle["primary_retained_teacher_sample_ids"])
        seeds = np.asarray(bundle["retained_teacher_seeds"])
        probabilities = np.asarray(bundle["primary_retained_teacher_probabilities"])
    primary_ids = sample_ids[scope == "grouped_crossfit_oof"]
    if (
        teacher_ids_raw.shape != (4977,)
        or teacher_ids_raw.dtype.kind != "U"
        or not np.array_equal(teacher_ids_raw.astype(str), primary_ids)
        or seeds.shape != (5,)
        or seeds.dtype != np.dtype("int64")
        or not np.array_equal(seeds, np.asarray([42, 43, 44, 45, 46], dtype=np.int64))
        or probabilities.shape != (4977, 5, 3)
        or probabilities.dtype != np.dtype("float32")
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=2), 1.0, rtol=0.0, atol=1e-5)
    ):
        raise RuntimeError("Okutama retained-teacher bundle member contract changed")


def validate_okutama_temporal_control_artifacts(
    root: Path,
    *,
    bundle_path: Path,
    bundle_summary_path: Path,
    fold_map_path: Path,
    scenario_resampling_path: Path,
    prepare_summary_path: Path,
    source_lock_digest: str,
    source_hashes: dict[str, str],
    artifact_receipts: dict[str, dict[str, Any]],
) -> None:
    """Validate the complete deterministic, role-safe T1/T2 pre-fit package."""

    bundle_summary, _ = _read_json(bundle_summary_path)
    if (
        bundle_summary.get("status") != "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE"
        or _eligible_scope(bundle_summary) != (6360, 14)
        or _inventory_digest(bundle_summary, "eligible_feature_bundle", bundle_path.name)
        != artifact_receipts["eligible_feature_bundle"]["sha256"]
        or bundle_summary.get("source_sha256", {}).get("protocol_lock") != source_lock_digest
    ):
        raise RuntimeError("Okutama temporal controls received an incompatible feature bundle")

    with np.load(bundle_path, allow_pickle=False) as bundle:
        required = {"sample_ids", "recording_ids", "scope", "fold"}
        if not required.issubset(bundle.files):
            raise RuntimeError("Okutama role-safe bundle lacks temporal-control row identity")
        scope = np.asarray(bundle["scope"]).astype(str)
        primary = np.flatnonzero(scope == "grouped_crossfit_oof")
        sample_ids = np.asarray(bundle["sample_ids"])[primary].astype(str)
        recording_ids = np.asarray(bundle["recording_ids"])[primary].astype(str)
        folds = np.asarray(bundle["fold"])[primary].astype(str)
    if len(primary) != 4977 or len(set(recording_ids)) != 11:
        raise RuntimeError("Okutama temporal-control primary scope changed")

    columns, rows = _csv_rows(_read_bytes(fold_map_path), name="Okutama temporal fold map")
    expected_columns = ["bundle_row_index", "sample_id", "recording_id", "fold"]
    if columns != expected_columns or len(rows) != 4977:
        raise RuntimeError("Okutama temporal-control fold-map schema or size changed")
    for position, (row, bundle_index, sample_id, recording_id, fold) in enumerate(
        zip(rows, primary, sample_ids, recording_ids, folds, strict=True)
    ):
        if (
            row["bundle_row_index"],
            row["sample_id"],
            row["recording_id"],
            row["fold"],
        ) != (str(bundle_index), sample_id, recording_id, fold):
            raise RuntimeError(f"Okutama temporal fold-map mismatch at row {position}")
    spec, _ = _read_json(root / DATASET_CONFIG["okutama"]["spec"])
    expected_group_folds = {
        scenario: fold
        for fold, scenarios in spec["fold_contract"].items()
        for scenario in scenarios
    }
    observed_group_folds = {
        recording_id: set(folds[recording_ids == recording_id])
        for recording_id in sorted(set(recording_ids))
    }
    if any(
        fold_set != {expected_group_folds.get(recording_id)}
        for recording_id, fold_set in observed_group_folds.items()
    ):
        raise RuntimeError("An Okutama scenario crosses or changes its historical fold")

    with np.load(scenario_resampling_path, allow_pickle=False) as values:
        expected_keys = {
            "group_order",
            "group_draw_indices",
            "exact_swap_assignments",
            "bootstrap_seed",
            "numpy_version",
            "bit_generator",
            "initial_rng_state_json",
            "final_rng_state_json",
        }
        if set(values.files) != expected_keys:
            raise RuntimeError("Okutama temporal-control resampling keys changed")
        group_order = np.asarray(values["group_order"]).astype(str)
        draws = np.asarray(values["group_draw_indices"])
        swaps = np.asarray(values["exact_swap_assignments"])
        bootstrap_seed = int(np.asarray(values["bootstrap_seed"]).item())
        numpy_version = str(np.asarray(values["numpy_version"]).item())
        bit_generator = str(np.asarray(values["bit_generator"]).item())
        initial_state = str(np.asarray(values["initial_rng_state_json"]).item())
        final_state = str(np.asarray(values["final_rng_state_json"]).item())
    expected_groups = np.asarray(sorted(set(recording_ids)), dtype=str)
    generator = np.random.default_rng(20260906)
    expected_initial_state = generator.bit_generator.state
    expected_draws = generator.integers(0, 11, size=(10000, 11), dtype=np.int16)
    expected_final_state = generator.bit_generator.state
    expected_swaps = (
        (np.arange(2048, dtype=np.uint16)[:, None] >> np.arange(11, dtype=np.uint16)) & 1
    ).astype(bool)
    if (
        not np.array_equal(group_order, expected_groups)
        or draws.dtype != np.dtype("int16")
        or not np.array_equal(draws, expected_draws)
        or swaps.dtype != np.dtype("bool")
        or not np.array_equal(swaps, expected_swaps)
        or bootstrap_seed != 20260906
        or numpy_version != np.__version__
        or bit_generator != "PCG64"
    ):
        raise RuntimeError("Okutama temporal-control resampling contract changed")
    try:
        if json.loads(initial_state) != expected_initial_state:
            raise RuntimeError("Okutama temporal-control initial RNG state changed")
        if json.loads(final_state) != expected_final_state:
            raise RuntimeError("Okutama temporal-control final RNG state changed")
    except json.JSONDecodeError as error:
        raise RuntimeError("Invalid Okutama temporal-control RNG state JSON") from error

    prepare, _ = _read_json(prepare_summary_path)
    if prepare.get("status") != "OKUTAMA_TEMPORAL_CONTROLS_PREPARE_COMPLETE":
        raise RuntimeError("Okutama temporal-control prepare receipt is incomplete")
    if (int(prepare.get("rows", -1)), int(prepare.get("scenarios", -1))) != (4977, 11):
        raise RuntimeError("Okutama temporal-control prepare scope changed")
    for name in ("model_fits", "feature_values_read"):
        if int(prepare.get(name, -1)) != 0:
            raise RuntimeError(f"Unsafe Okutama temporal-control prepare receipt: {name}")
    prepare_access = prepare.get("access_accounting", {})
    if int(prepare_access.get("role_safe_bundle_metadata_rows_read", -1)) != 6360:
        raise RuntimeError("Okutama temporal prepare metadata accounting changed")
    for name in (
        "role_safe_feature_values_read",
        "mixed_role_files_opened",
        "protected_calibration_rows_read",
        "confirmation_rows_read",
        "test_rows_read",
    ):
        if int(prepare_access.get(name, -1)) != 0:
            raise RuntimeError(f"Unsafe Okutama temporal prepare access: {name}")
    expected_prepare_sources = {
        "source_lock": source_lock_digest,
        "protocol": source_hashes["protocol_spec"],
        "eligible_feature_bundle": artifact_receipts["eligible_feature_bundle"]["sha256"],
        "bundle_summary": artifact_receipts["bundle_summary"]["sha256"],
        "runner": source_hashes["temporal_controls_runner"],
    }
    for name, digest in expected_prepare_sources.items():
        if prepare.get("source_sha256", {}).get(name) != digest:
            raise RuntimeError(f"Okutama temporal prepare source mismatch: {name}")
    for name in ("fold_map", "scenario_resampling"):
        if (
            prepare.get("artifact_sha256", {}).get(Path(artifact_receipts[name]["path"]).name)
            != (artifact_receipts[name]["sha256"])
        ):
            raise RuntimeError(f"Okutama temporal prepare artifact mismatch: {name}")


def _execution_payload(root: Path, dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.stage is None:
        raise ValueError("Execution locks require --stage")
    if dataset == "okutama" and args.stage == "r1a":
        raise ValueError(
            "Create the stage-r1b lock up front; it authorizes the resumable seed-43 R1a replay"
        )
    if dataset == "okutama" and args.stage not in {"r1b", "temporal-controls"}:
        raise ValueError("Okutama execution lock stage must be r1b or temporal-controls")
    if dataset == "vcoco" and args.stage not in {"v0", "v1"}:
        raise ValueError("V-COCO execution lock stage must be v0 or v1")
    if args.source_lock is None:
        raise ValueError("Execution locks require --source-lock")
    source_lock_path = args.source_lock.resolve()
    assert_role_safe_input_path(root, source_lock_path)
    source_lock, source_content = _read_json(source_lock_path)
    expected_source_status = DATASET_CONFIG[dataset]["source_status"]
    if source_lock.get("status") != expected_source_status:
        raise RuntimeError("Execution lock references an incompatible source lock")
    if source_lock.get("dataset") != dataset:
        raise RuntimeError("Execution and source lock datasets differ")
    current_head = _git(root, "rev-parse", "HEAD")
    current_tree = _git(root, "rev-parse", "HEAD^{tree}")
    if source_lock.get("repository_commit") != current_head:
        raise RuntimeError("Execution-lock HEAD differs from its source lock")
    if source_lock.get("repository_tree") != current_tree:
        raise RuntimeError("Execution-lock tree differs from its source lock")

    artifacts: dict[str, Any] = {}
    source_blobs: dict[str, Any]
    if dataset == "okutama" and args.stage in {"r1a", "r1b"}:
        required = {
            "input_inventory": args.input_inventory,
            "eligible_index": args.eligible_index,
            "eligible_feature_bundle": args.feature_bundle,
            "eligible_window_masks": args.eligible_window_masks,
            "bundle_summary": args.bundle_summary,
        }
        missing_arguments = [name for name, path in required.items() if path is None]
        if missing_arguments:
            raise ValueError(f"Missing Okutama execution inputs: {', '.join(missing_arguments)}")
        inventory_path = required["input_inventory"].resolve()  # type: ignore[union-attr]
        assert_role_safe_input_path(root, inventory_path)
        inventory, _ = _read_json(inventory_path)
        if inventory.get("status") != "OKUTAMA_CPTR_REPLAY_INPUT_INVENTORY_COMPLETE":
            raise RuntimeError("Okutama replay input inventory is incomplete")
        if _eligible_scope(inventory) != (6360, 14):
            raise RuntimeError("Okutama role-safe inventory scope changed")
        for name, path in required.items():
            resolved = path.resolve()  # type: ignore[union-attr]
            relative = assert_role_safe_input_path(root, resolved)
            digest, size = _sha256_file(resolved)
            expected = _inventory_digest(inventory, name, Path(relative).name)
            if name != "input_inventory" and expected is not None and digest != expected:
                raise RuntimeError(f"Okutama inventory digest mismatch for {name}")
            artifacts[name] = {"path": relative, "sha256": digest, "size_bytes": size}
        bundle_summary, _ = _read_json(required["bundle_summary"].resolve())  # type: ignore[union-attr]
        if bundle_summary.get("status") != "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE":
            raise RuntimeError("Okutama role-safe feature bundle is incomplete")
        if (
            int(bundle_summary.get("rows", -1)),
            int(bundle_summary.get("scenarios", -1)),
        ) != (6360, 14):
            raise RuntimeError("Okutama role-safe feature bundle scope changed")
        expected_bundle = _inventory_digest(
            bundle_summary, "eligible_feature_bundle", "eligible_feature_bundle.npz"
        )
        if expected_bundle != artifacts["eligible_feature_bundle"]["sha256"]:
            raise RuntimeError("Okutama bundle summary does not bind the feature bundle")
        validate_okutama_retained_teacher_bundle(required["eligible_feature_bundle"].resolve())  # type: ignore[union-attr]
        expected_bundle_sources = {
            "protocol_lock": _sha256(source_content),
            "input_inventory": artifacts["input_inventory"]["sha256"],
            "eligible_index": artifacts["eligible_index"]["sha256"],
            "eligible_window_masks": artifacts["eligible_window_masks"]["sha256"],
            "window_mask_summary": source_lock["source_sha256"]["window_mask_summary"],
            "bundle_builder": source_lock["source_sha256"]["bundle_builder"],
            "cptr_feature_module": source_lock["source_sha256"]["cptr_feature_module"],
            **{
                name: digest
                for name, digest in source_lock["source_sha256"].items()
                if name.startswith("feature_array_")
            },
        }
        bundle_sources = bundle_summary.get("source_sha256", {})
        for name, expected in expected_bundle_sources.items():
            if bundle_sources.get(name) != expected:
                raise RuntimeError(f"Okutama bundle summary source mismatch: {name}")
        source_sha256 = {
            "source_lock": _sha256(source_content),
            "input_inventory": artifacts["input_inventory"]["sha256"],
            "eligible_index": artifacts["eligible_index"]["sha256"],
            "eligible_feature_bundle": artifacts["eligible_feature_bundle"]["sha256"],
            "bundle_builder": source_lock["source_sha256"]["bundle_builder"],
            "replay_runner": source_lock["source_sha256"]["replay_runner"],
            "cptr_model_module": source_lock["source_sha256"]["cptr_model_module"],
            "cptr_feature_module": source_lock["source_sha256"]["cptr_feature_module"],
            "cptr_training_module": source_lock["source_sha256"]["cptr_training_module"],
            "temporal_model_module": source_lock["source_sha256"]["temporal_model_module"],
            "neural_module": source_lock["source_sha256"]["neural_module"],
            "locker": _sha256_file(Path(__file__).resolve())[0],
        }
        source_blobs = {
            name: source_lock["source_blobs"][name]
            for name in (
                "replay_runner",
                "cptr_model_module",
                "cptr_feature_module",
                "cptr_training_module",
                "temporal_model_module",
                "neural_module",
                "locker",
            )
        }
        source_sha256["eligible_window_masks"] = artifacts["eligible_window_masks"]["sha256"]
        source_sha256["bundle_summary"] = artifacts["bundle_summary"]["sha256"]
        authorization = {
            "R1a_execution": True,
            "R1b_execution": args.stage == "r1b",
            "model_updates": 0,
            "fitting": False,
        }
    elif dataset == "okutama":
        required = {
            "eligible_feature_bundle": args.feature_bundle,
            "bundle_summary": args.bundle_summary,
            "fold_map": args.fold_map,
            "scenario_resampling": args.scenario_resampling,
            "prepare_summary": args.prepare_summary,
            "runner": args.execution_runner,
        }
        missing_arguments = [name for name, path in required.items() if path is None]
        if missing_arguments:
            raise ValueError(
                "Missing Okutama temporal-control inputs: " + ", ".join(missing_arguments)
            )
        source_authorization = source_lock.get("authorization", {})
        if source_authorization.get("temporal_controls_artifact_construction") is not True:
            raise RuntimeError("Source lock did not authorize temporal-control preparation")
        for name, path in required.items():
            resolved = path.resolve()  # type: ignore[union-attr]
            relative = assert_role_safe_input_path(root, resolved)
            digest, size = _sha256_file(resolved)
            artifacts[name] = {"path": relative, "sha256": digest, "size_bytes": size}
        validate_okutama_retained_teacher_bundle(args.feature_bundle.resolve())
        validate_okutama_temporal_control_artifacts(
            root,
            bundle_path=args.feature_bundle.resolve(),
            bundle_summary_path=args.bundle_summary.resolve(),
            fold_map_path=args.fold_map.resolve(),
            scenario_resampling_path=args.scenario_resampling.resolve(),
            prepare_summary_path=args.prepare_summary.resolve(),
            source_lock_digest=_sha256(source_content),
            source_hashes=source_lock["source_sha256"],
            artifact_receipts=artifacts,
        )
        commit = str(source_lock["repository_commit"])
        committed_sources = {
            name: git_source_receipt(root, DATASET_CONFIG["okutama"]["sources"][name], commit)
            for name in (
                "protocol_spec",
                "temporal_controls_runner",
                "temporal_model_module",
                "temporal_training_module",
                "neural_module",
                "polar_training_module",
            )
        }
        for name, receipt in committed_sources.items():
            if receipt["sha256"] != source_lock["source_sha256"].get(name):
                raise RuntimeError(f"Temporal-control source differs from source lock: {name}")
        if artifacts["runner"]["sha256"] != committed_sources["temporal_controls_runner"]["sha256"]:
            raise RuntimeError("Temporal-control execution runner path changed")
        protocol, _ = _read_json(root / DATASET_CONFIG["okutama"]["spec"])
        epoch_declaration = next(
            item
            for item in protocol["historical_input_receipts"]
            if item["name"] == "temporal_epoch_proof_source"
        )
        epoch_receipt = _artifact_receipt(
            root, epoch_declaration["path"], epoch_declaration["sha256"]
        )
        locked_epoch = (
            source_lock.get("inputs", {}).get("receipts", {}).get("temporal_epoch_proof_source", {})
        )
        if locked_epoch.get("sha256") != epoch_receipt["sha256"]:
            raise RuntimeError("Temporal fixed-epoch proof differs from its source lock")
        source_sha256 = {
            "source_lock": _sha256(source_content),
            "protocol": committed_sources["protocol_spec"]["sha256"],
            "eligible_feature_bundle": artifacts["eligible_feature_bundle"]["sha256"],
            "bundle_summary": artifacts["bundle_summary"]["sha256"],
            "fold_map": artifacts["fold_map"]["sha256"],
            "scenario_resampling": artifacts["scenario_resampling"]["sha256"],
            "prepare_summary": artifacts["prepare_summary"]["sha256"],
            "runner": committed_sources["temporal_controls_runner"]["sha256"],
            "temporal_model_module": committed_sources["temporal_model_module"]["sha256"],
            "temporal_training_module": committed_sources["temporal_training_module"]["sha256"],
            "neural_module": committed_sources["neural_module"]["sha256"],
            "polar_training_module": committed_sources["polar_training_module"]["sha256"],
            "epoch_proof_source": epoch_receipt["sha256"],
        }
        source_blobs = {
            name: source_lock["source_blobs"][name]
            for name in (
                "protocol_spec",
                "temporal_controls_runner",
                "temporal_model_module",
                "temporal_training_module",
                "neural_module",
                "polar_training_module",
                "locker",
            )
        }
        authorization = {
            "T1_execution": True,
            "T2_execution": True,
            "benchmark_then_resume": True,
        }
    else:
        required = {
            "fold_map": args.fold_map,
            "bootstrap_group_indices": args.bootstrap_group_indices,
            "swap_signs_packbits": args.swap_signs_packbits,
            "statistics_randomization_receipt": args.statistics_randomization_receipt,
            "execution_runner": args.execution_runner,
        }
        missing_arguments = [name for name, path in required.items() if path is None]
        if missing_arguments:
            raise ValueError("Missing V-COCO execution inputs: " + ", ".join(missing_arguments))
        if args.stage == "v1":
            raise RuntimeError(
                "V1 is blocked until every missing spatial and auxiliary input is bound"
            )
        validate_vcoco_fold_map(root, args.fold_map.resolve())
        for name, path in required.items():
            resolved = path.resolve()  # type: ignore[union-attr]
            relative = assert_role_safe_input_path(root, resolved)
            digest, size = _sha256_file(resolved)
            artifacts[name] = {"path": relative, "sha256": digest, "size_bytes": size}
        validate_vcoco_randomization_artifacts(
            root,
            bootstrap_path=args.bootstrap_group_indices.resolve(),
            swap_path=args.swap_signs_packbits.resolve(),
            receipt_path=args.statistics_randomization_receipt.resolve(),
            artifact_receipts=artifacts,
        )
        commit = str(source_lock["repository_commit"])
        runner_source = git_source_receipt(root, artifacts["execution_runner"]["path"], commit)
        if runner_source["sha256"] != source_lock["source_sha256"].get("v0_runner"):
            raise RuntimeError("V0 execution runner differs from the source-locked Git blob")
        source_sha256 = {
            "source_lock": _sha256(source_content),
            "fold_map": artifacts["fold_map"]["sha256"],
            "bootstrap_group_indices": artifacts["bootstrap_group_indices"]["sha256"],
            "swap_signs_packbits": artifacts["swap_signs_packbits"]["sha256"],
            "statistics_randomization_receipt": artifacts["statistics_randomization_receipt"][
                "sha256"
            ],
            "execution_runner": runner_source["sha256"],
            "locker": _sha256_file(Path(__file__).resolve())[0],
        }
        source_blobs = {
            name: source_lock["source_blobs"][name]
            for name in ("protocol_spec", "nested_model_module", "v0_runner", "locker")
        }
        authorization = {"V0_fitting": True, "V1_fitting": False}

    return {
        "schema_version": 1,
        "status": (
            DATASET_CONFIG[dataset]["temporal_controls_execution_status"]
            if dataset == "okutama" and args.stage == "temporal-controls"
            else DATASET_CONFIG[dataset]["execution_status"]
        ),
        "dataset": dataset,
        "stage": args.stage,
        "repository_commit": source_lock["repository_commit"],
        "repository_tree": source_lock.get("repository_tree"),
        "source_lock_sha256": _sha256(source_content),
        "source_sha256": source_sha256,
        "source_blobs": source_blobs,
        "artifacts": artifacts,
        "environment": collect_environment(),
        "authorization": authorization,
        "protected_access": {
            "mixed_manifest_rows_read": 0,
            "mixed_development_metadata_rows_read": 0,
            "calibration_rows_or_arrays_read": 0,
            "confirmation_rows_or_arrays_read": 0,
            "test_rows_or_arrays_read": 0,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _without_creation_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("locked_at_utc", None)
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Not a Git checkout: {root}")
    config = DATASET_CONFIG[args.dataset]
    if args.phase == "source":
        default_output = config["default_source_output"]
    elif args.dataset == "okutama" and args.stage == "temporal-controls":
        default_output = config["temporal_controls_execution_output"]
    else:
        default_output = config["default_execution_output"]
    output = (args.output or root / default_output).resolve()

    if args.mode == "check":
        if not output.is_file():
            raise FileNotFoundError(output)
        retained, _ = _read_json(output)
        commit = str(retained.get("repository_commit", ""))
        if not commit:
            raise RuntimeError("Retained lock has no repository commit")
        current = (
            _source_payload(root, args.dataset, commit=commit, args=args)
            if args.phase == "source"
            else _execution_payload(root, args.dataset, args)
        )
        current["locked_at_utc"] = retained.get("locked_at_utc")
        validate_retained_lock(retained, current)
        print(json.dumps({"status": "LOCK_VALID", "path": str(output)}, indent=2))
        return

    head = _git(root, "rev-parse", "HEAD")
    worktree_changes = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    source_preconditions = (
        committed_source_preconditions(root, config, head) if args.phase == "source" else []
    )
    if args.mode == "prepare" and source_preconditions:
        readiness = {
            "status": "COMMIT_REQUIRED_BEFORE_LOCK",
            "dataset": args.dataset,
            "phase": args.phase,
            "repository_commit": head,
            "worktree_clean": not bool(worktree_changes),
            "changed_paths": [line[3:] for line in worktree_changes.splitlines() if len(line) > 3],
            "source_preconditions": source_preconditions,
            "candidate": None,
            "next_action": (
                "Commit every listed source, then rerun prepare with the exact role-safe "
                "artifact arguments. No lock was created."
            ),
        }
        print(json.dumps(readiness, indent=2, sort_keys=True, allow_nan=False))
        return
    payload = (
        _source_payload(root, args.dataset, commit=head, args=args)
        if args.phase == "source"
        else _execution_payload(root, args.dataset, args)
    )
    readiness = {
        "status": "READY_TO_LOCK" if not worktree_changes else "COMMIT_REQUIRED_BEFORE_LOCK",
        "dataset": args.dataset,
        "phase": args.phase,
        "repository_commit": head,
        "worktree_clean": not bool(worktree_changes),
        "changed_paths": [line[3:] for line in worktree_changes.splitlines() if len(line) > 3],
        "candidate": payload,
    }
    if args.mode == "prepare":
        print(json.dumps(readiness, indent=2, sort_keys=True, allow_nan=False))
        return
    if worktree_changes:
        raise RuntimeError("A clean committed worktree is required before creating a lock")
    payload["locked_at_utc"] = datetime.now(UTC).isoformat()
    if not (
        args.phase == "execution"
        and args.dataset == "okutama"
        and args.stage == "temporal-controls"
    ):
        payload["source_sha256"]["locker"] = payload["source_sha256"].get(
            "locker", _sha256_file(Path(__file__).resolve())[0]
        )
    if output.exists():
        retained, _ = _read_json(output)
        if _without_creation_metadata(retained) != _without_creation_metadata(payload):
            raise RuntimeError(f"An incompatible continuation lock already exists: {output}")
    else:
        _write_json(output, payload)
    print(json.dumps({"status": payload["status"], "path": str(output)}, indent=2))


if __name__ == "__main__":
    main()
