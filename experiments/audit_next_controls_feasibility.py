"""Audit T1/T2/V0/V1 feasibility without opening protected or mixed-role rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FOLDS = tuple(range(5))
SEEDS = tuple(range(42, 47))
LEGACY_SHORT_INDICES = (4, 6, 6, 8, 8, 10, 10, 12)
DISTINCT_SHORT_INDICES = (4, 5, 6, 7, 8, 10, 11, 12)
SHORT_CENTER_SLOT = 4

V0_CACHES = {
    "dinov2_tight": ".runs/polar_v2/features/dinov2_base/person_tight/aspect_pad_224",
    "dinov2_context": (".runs/polar_v2/features/dinov2_base/person_context_25/aspect_pad_224"),
    "siglip2_tight": ".runs/polar_v2/features/siglip2_base/person_tight/aspect_pad_224",
    "siglip2_context": (".runs/polar_v2/features/siglip2_base/person_context_25/aspect_pad_224"),
}

SAFE_JSON_EXACT = {
    "experiments/okutama_cptr_crossfit_plan.json",
    ".runs/vcoco_v3/okutama/features/dinov2_base/store.json",
    ".runs/vcoco_v3/okutama/features/dinov2_base/summary.json",
    ".runs/vcoco_v3/representations/evaluation/summary.json",
    ".runs/vcoco_v3/nested_stacks/summary.json",
    ".runs/vcoco_v3/spatial/summary.json",
    *(f"{path}/provenance.json" for path in V0_CACHES.values()),
}


def _normalized_relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the audited repository: {resolved}") from exc
    return relative.as_posix()


def _is_checkpoint_summary(relative: str) -> bool:
    parts = relative.split("/")
    if len(parts) == 7 and parts[-1] == "summary.json":
        return (
            parts[:4] == [".runs", "cptr", "crossfit", "centre_short_parts"]
            and parts[4] in {f"fold-{fold}" for fold in FOLDS}
            and parts[5] in {f"seed-{seed}" for seed in SEEDS}
        )
    if len(parts) == 8 and parts[-1] == "summary.json":
        return (
            parts[:4] == [".runs", "vcoco_v3", "temporal", "crossfit"]
            and parts[4] in {"static", "teacher"}
            and parts[5] in {f"fold-{fold}" for fold in FOLDS}
            and parts[6] in {f"seed-{seed}" for seed in SEEDS}
        )
    return False


def _is_checkpoint(relative: str) -> bool:
    return _is_checkpoint_summary(relative.removesuffix("checkpoint.pt") + "summary.json")


def _is_v0_feature(relative: str) -> bool:
    return any(relative == f"{cache}/features.npy" for cache in V0_CACHES.values())


@dataclass
class AccessAudit:
    """Enforce and record the narrow content-read allowlist used by this audit."""

    root: Path
    reads: list[dict[str, Any]] = field(default_factory=list)

    def _allow_json(self, path: Path) -> str:
        relative = _normalized_relative(self.root, path)
        if relative not in SAFE_JSON_EXACT and not _is_checkpoint_summary(relative):
            raise PermissionError(f"JSON path is outside the feasibility allowlist: {relative}")
        return relative

    def _allow_binary(self, path: Path) -> str:
        relative = _normalized_relative(self.root, path)
        if not (_is_checkpoint(relative) or _is_v0_feature(relative)):
            raise PermissionError(f"Binary path is outside the feasibility allowlist: {relative}")
        return relative

    def read_json(self, path: Path, *, purpose: str) -> dict[str, Any]:
        relative = self._allow_json(path)
        payload = path.read_bytes()
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object: {relative}")
        self.reads.append(
            {
                "path": relative,
                "method": "json_bytes",
                "purpose": purpose,
                "bytes": len(payload),
            }
        )
        return result

    def hash_binary(self, path: Path, *, purpose: str) -> str:
        relative = self._allow_binary(path)
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
        self.reads.append(
            {
                "path": relative,
                "method": "sha256_stream",
                "purpose": purpose,
                "bytes": total,
            }
        )
        return digest.hexdigest()

    def record_weights_only_load(self, path: Path, *, purpose: str) -> str:
        relative = self._allow_binary(path)
        self.reads.append(
            {
                "path": relative,
                "method": "torch_load_weights_only",
                "purpose": purpose,
                "bytes": path.stat().st_size,
            }
        )
        return relative


def t1_all_valid_mean_pool(sequence: np.ndarray) -> np.ndarray:
    """Pool every sampled slot, preserving the legacy sampler's repeated weighting."""

    values = np.asarray(sequence)
    if values.ndim not in {2, 3} or values.shape[-2] != len(LEGACY_SHORT_INDICES):
        raise ValueError("T1 expects [8, features] or [batch, 8, features]")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("T1 inputs must be finite numeric values")
    return values.mean(axis=-2)


def temporal_control_contract() -> dict[str, Any]:
    """Return the fixed no-data T1/T2 contract selected for the next run."""

    return {
        "frame_count": 17,
        "center_frame_index": 8,
        "span_endpoints_inclusive": [4, 12],
        "center_slot_zero_based": SHORT_CENTER_SLOT,
        "t1": {
            "input_indices": list(LEGACY_SHORT_INDICES),
            "unique_frame_count": len(set(LEGACY_SHORT_INDICES)),
            "pool": "arithmetic_mean_over_all_eight_slots",
            "duplicate_weighting": "preserved",
            "inference_mask": "all_valid",
            "positional_or_timestamp_signal": False,
            "macro_f1_noninferiority_margin": 0.005,
            "nll_difference_upper_bound_maximum": 0.0,
            "latency_must_improve": True,
        },
        "t2": {
            "legacy_indices": list(LEGACY_SHORT_INDICES),
            "distinct_indices": list(DISTINCT_SHORT_INDICES),
            "distinct_unique_frame_count": len(set(DISTINCT_SHORT_INDICES)),
            "omitted_relative_offset": 1,
            "asymmetry_reason": (
                "Eight distinct samples cannot include both endpoints, the center, and "
                "symmetric pairs; center remains in historical zero-based slot four."
            ),
            "refit_both_sampler_arms": True,
        },
    }


def _checkpoint_paths(family: str, fold: int, seed: int) -> tuple[str, str]:
    if family == "cptr":
        directory = f".runs/cptr/crossfit/centre_short_parts/fold-{fold}/seed-{seed}"
    else:
        directory = f".runs/vcoco_v3/temporal/crossfit/{family}/fold-{fold}/seed-{seed}"
    return f"{directory}/summary.json", f"{directory}/checkpoint.pt"


def _validate_weights_only(path: Path, audit: AccessAudit) -> dict[str, Any]:
    import torch

    relative = audit.record_weights_only_load(path, purpose="development checkpoint schema")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected_top = {"fixed_epochs", "model_state_dict", "request_sha256"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_top:
        raise RuntimeError(f"Unexpected checkpoint envelope: {relative}")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint has no model state: {relative}")
    schema = []
    all_finite = True
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Non-tensor model state entry in {relative}: {name}")
        schema.append([name, str(tensor.dtype), list(tensor.shape)])
        if tensor.is_floating_point() or tensor.is_complex():
            all_finite = all_finite and bool(torch.isfinite(tensor).all())
    if not all_finite:
        raise RuntimeError(f"Non-finite checkpoint tensor: {relative}")
    schema_bytes = json.dumps(schema, separators=(",", ":"), ensure_ascii=True).encode()
    return {
        "state_entries": len(state),
        "state_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "all_floating_tensors_finite": True,
        "fixed_epochs": int(checkpoint["fixed_epochs"]),
        "request_sha256": str(checkpoint["request_sha256"]),
    }


def _inventory_checkpoints(
    root: Path, audit: AccessAudit, *, validate_checkpoints: bool
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    identities: dict[tuple[str, int, int], str] = {}
    for family in ("static", "teacher", "cptr"):
        records = []
        schema_hashes: set[str] = set()
        for fold in FOLDS:
            for seed in SEEDS:
                summary_relative, checkpoint_relative = _checkpoint_paths(family, fold, seed)
                summary_path = root / summary_relative
                checkpoint_path = root / checkpoint_relative
                summary = audit.read_json(
                    summary_path, purpose=f"{family} development checkpoint receipt"
                )
                if int(summary.get("fold", -1)) != fold or int(summary.get("seed", -1)) != seed:
                    raise RuntimeError(f"Fold/seed receipt mismatch: {summary_relative}")
                for counter in (
                    "calibration_samples_read",
                    "confirmation_samples_read",
                    "validation_samples_read",
                ):
                    if int(summary.get(counter, -1)) != 0:
                        raise RuntimeError(f"Protected-role counter is nonzero: {summary_relative}")
                actual_hash = audit.hash_binary(
                    checkpoint_path, purpose=f"{family} development checkpoint identity"
                )
                if actual_hash != summary["artifact_sha256"]["checkpoint.pt"]:
                    raise RuntimeError(f"Checkpoint hash drift: {checkpoint_relative}")
                identities[(family, fold, seed)] = actual_hash
                record: dict[str, Any] = {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_path": checkpoint_relative,
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "checkpoint_sha256": actual_hash,
                    "fixed_epochs": int(summary["fixed_epochs"]),
                    "runtime_seconds": float(summary["runtime_seconds"]),
                    "reference_device": str(summary["training_device"]),
                }
                if validate_checkpoints:
                    validation = _validate_weights_only(checkpoint_path, audit)
                    if validation["request_sha256"] != summary["request_sha256"]:
                        raise RuntimeError(f"Checkpoint request drift: {checkpoint_relative}")
                    if validation["fixed_epochs"] != record["fixed_epochs"]:
                        raise RuntimeError(f"Checkpoint epoch drift: {checkpoint_relative}")
                    schema_hashes.add(validation["state_schema_sha256"])
                    record["weights_only_validation"] = validation
                if family == "cptr":
                    expected = summary.get("baseline_checkpoint_sha256", {})
                    record["declared_static_sha256"] = expected.get("static")
                    record["declared_teacher_sha256"] = expected.get("teacher")
                records.append(record)
        if family == "cptr":
            for record in records:
                key = (record["fold"], record["seed"])
                if record["declared_static_sha256"] != identities[("static", *key)]:
                    raise RuntimeError(f"CPTR static anchor mismatch: fold/seed {key}")
                if record["declared_teacher_sha256"] != identities[("teacher", *key)]:
                    raise RuntimeError(f"CPTR teacher anchor mismatch: fold/seed {key}")
        families[family] = {
            "expected_sets": len(FOLDS) * len(SEEDS),
            "verified_sets": len(records),
            "total_checkpoint_bytes": sum(row["checkpoint_bytes"] for row in records),
            "historical_runtime_seconds": sum(row["runtime_seconds"] for row in records),
            "historical_runtime_role": "reference_measurement_not_a_new_compute_forecast",
            "reference_devices": sorted({row["reference_device"] for row in records}),
            "weights_only_validated_sets": len(records) if validate_checkpoints else 0,
            "unique_state_schemas": len(schema_hashes) if validate_checkpoints else None,
            "records": records,
        }
    return families


def _file_receipt(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _normalized_relative(root, path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "content_read": False,
    }


def _inventory_temporal_store(root: Path, audit: AccessAudit) -> dict[str, Any]:
    directory = root / ".runs/vcoco_v3/okutama/features/dinov2_base"
    store = audit.read_json(directory / "store.json", purpose="temporal store metadata")
    summary = audit.read_json(directory / "summary.json", purpose="temporal store receipt")
    required = {
        "status": "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE",
        "samples": 8339,
        "frames_per_sample": 17,
        "feature_dimensions": 768,
        "center_frame_index": 8,
    }
    for key, expected in required.items():
        if store.get(key) != expected:
            raise RuntimeError(f"Unexpected temporal store {key}: {store.get(key)!r}")
    if summary.get("confirmation_archive_opened") is not False:
        raise RuntimeError("Temporal store receipt says confirmation was opened")
    stat_only = [
        _file_receipt(directory / name, root)
        for name in (
            "tight.npy",
            "context.npy",
            "geometry.npy",
            "development_metadata.csv",
        )
    ]
    if not all(item["present"] for item in stat_only):
        raise FileNotFoundError("The packed temporal store is incomplete")
    return {
        "status": store["status"],
        "samples": store["samples"],
        "frames_per_sample": store["frames_per_sample"],
        "center_frame_index": store["center_frame_index"],
        "frame_feature_dimensions": {
            "tight": store["feature_dimensions"],
            "context": store["feature_dimensions"],
            "geometry": 6,
            "combined": 2 * int(store["feature_dimensions"]) + 6,
        },
        "stat_only_files": stat_only,
        "role_boundary": (
            "The packed provider-train store spans historical train, validation, and "
            "sealed calibration indices. A fresh development-only row-to-feature index "
            "is required before any array value is loaded."
        ),
        "array_values_loaded": 0,
        "mixed_role_metadata_rows_read": 0,
    }


def _inventory_v0(root: Path, audit: AccessAudit) -> dict[str, Any]:
    caches = {}
    rows_hashes = set()
    for name, relative in V0_CACHES.items():
        directory = root / relative
        provenance = audit.read_json(directory / "provenance.json", purpose=f"V0 {name} receipt")
        if provenance.get("status") != "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE":
            raise RuntimeError(f"V0 cache status mismatch: {name}")
        if int(provenance.get("test_rows_read", -1)) != 0:
            raise RuntimeError(f"V0 cache crossed the test boundary: {name}")
        if int(provenance.get("rows", -1)) != 6640:
            raise RuntimeError(f"V0 row count mismatch: {name}")
        feature_path = directory / "features.npy"
        feature_hash = audit.hash_binary(feature_path, purpose=f"V0 {name} feature identity")
        if feature_hash != provenance["artifact_sha256"]["features.npy"]:
            raise RuntimeError(f"V0 feature hash drift: {name}")
        rows_hashes.add(str(provenance["artifact_sha256"]["rows.csv"]))
        caches[name] = {
            "directory": relative,
            "rows": int(provenance["rows"]),
            "dimensions": int(provenance["feature_dimensions"]),
            "representation": str(provenance["representation"]),
            "feature_bytes": feature_path.stat().st_size,
            "feature_sha256": feature_hash,
            "rows_sidecar": _file_receipt(directory / "rows.csv", root),
            "historical_extraction_runtime_seconds": float(provenance["runtime_seconds"]),
        }
    if len(rows_hashes) != 1:
        raise RuntimeError("V0 cache row identities differ")
    return {
        "cache_reuse_feasible": True,
        "requires_new_feature_extraction": False,
        "rows": 6640,
        "shared_rows_sha256_from_receipts": rows_hashes.pop(),
        "rows_sidecar_content_reads": 0,
        "caches": caches,
        "missing_arm_contracts": [
            "mixed_flat",
            "mixed_factorized_without_reliability",
        ],
        "available_historical_arm_contracts": [
            "dino_only_flat",
            "dino_only_factorized_without_reliability",
            "mixed_factorized_with_reliability",
            "mixed_linear_svm",
        ],
        "execution_gate": (
            "Commit a new V-COCO protocol that authorizes old train+val as development, "
            "bind shared image folds, then lock a matched four-arm V0 grid."
        ),
    }


def _historical_costs(
    root: Path, audit: AccessAudit, checkpoints: dict[str, Any]
) -> dict[str, Any]:
    representation = audit.read_json(
        root / ".runs/vcoco_v3/representations/evaluation/summary.json",
        purpose="historical VCOCO representation runtime",
    )
    nested = audit.read_json(
        root / ".runs/vcoco_v3/nested_stacks/summary.json",
        purpose="historical VCOCO nested runtime",
    )
    spatial = audit.read_json(
        root / ".runs/vcoco_v3/spatial/summary.json",
        purpose="historical VCOCO spatial runtime",
    )
    return {
        "interpretation": (
            "Observed historical wall times on the recorded RTX 4060 Laptop GPU. They "
            "are reference measurements, not forecasts for the current GPU or new arms."
        ),
        "okutama_static_25_runs_seconds": checkpoints["static"]["historical_runtime_seconds"],
        "okutama_teacher_25_runs_seconds": checkpoints["teacher"]["historical_runtime_seconds"],
        "okutama_cptr_25_runs_seconds": checkpoints["cptr"]["historical_runtime_seconds"],
        "vcoco_representation_6750_fits_seconds": float(representation["runtime_seconds"]),
        "vcoco_nested_four_families_seconds": float(nested["runtime_seconds"]),
        "vcoco_spatial_25380_fits_seconds": float(spatial["runtime_seconds"]),
        "required_next_measurement": (
            "Benchmark one representative fit for each new family before scheduling it."
        ),
    }


def _v1_gap_report(root: Path) -> dict[str, Any]:
    rows = 6640
    grid_tokens = 16 * 16
    dimensions = 768
    one_view_float16 = rows * grid_tokens * dimensions * 2
    image_files = []
    for directory in (root / "data", root / ".runs/external/vcoco"):
        if directory.is_dir():
            image_files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
    pose_oracle = (
        root
        / ".runs/polar_v2/features/coco_gt_pose_oracle/ground_truth_person_keypoints/normalized"
    )
    default_dino_snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--facebook--dinov2-base/snapshots"
        / "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    )
    return {
        "executable": False,
        "retained_feature_scope": "pooled_or_cls_only",
        "spatial_dino_patch_grid_present": False,
        "predicted_k4_pose_hypotheses_present": False,
        "ground_truth_pose_oracle_present_but_ineligible": pose_oracle.is_dir(),
        "provenance_clean_pose_auxiliary_pinned": False,
        "default_pinned_dinov2_snapshot_present": default_dino_snapshot.is_dir(),
        "repository_or_external_vcoco_image_files": len(image_files),
        "required_inputs": [
            "rights-safe development image receipt and local source bytes",
            "pinned DINOv2-base revision and exact weight hashes",
            "new 16x16 patch-token grid extraction with coordinate transforms",
            "K=4 label-blind predicted pose hypotheses with confidence and dispersion",
            "auxiliary training-corpus overlap audit",
            "matched B0/B1/B2/B3 arm configuration and resource contract",
        ],
        "minimum_grid_storage": {
            "assumptions": {
                "rows": rows,
                "tokens_per_view": grid_tokens,
                "dimensions": dimensions,
                "dtype": "float16",
            },
            "one_view_bytes": one_view_float16,
            "tight_plus_context_bytes": 2 * one_view_float16,
            "tight_plus_context_gib": (2 * one_view_float16) / (1024**3),
            "float32_tight_plus_context_gib": (4 * one_view_float16) / (1024**3),
        },
    }


def build_report(root: Path, *, validate_checkpoints: bool = False) -> dict[str, Any]:
    root = root.resolve()
    audit = AccessAudit(root)
    plan = audit.read_json(
        root / "experiments/okutama_cptr_crossfit_plan.json",
        purpose="declared fold and seed inventory",
    )
    if int(plan.get("folds", -1)) != len(FOLDS):
        raise RuntimeError("Unexpected CPTR fold count")
    if [int(row["seed"]) for row in plan.get("seeds", [])] != list(SEEDS):
        raise RuntimeError("Unexpected CPTR seed set")
    checkpoints = _inventory_checkpoints(root, audit, validate_checkpoints=validate_checkpoints)
    temporal_store = _inventory_temporal_store(root, audit)
    v0 = _inventory_v0(root, audit)
    costs = _historical_costs(root, audit, checkpoints)
    controls = temporal_control_contract()
    report = {
        "status": "HAC_NEXT_CONTROLS_FEASIBILITY_AUDIT_COMPLETE",
        "scope": "T1, T2, V0, and V1; metadata and development artifacts only; no fits",
        "model_fits": 0,
        "dataset_array_values_loaded": 0,
        "protected_data_reads": 0,
        "mixed_role_manifest_reads": 0,
        "checkpoints": checkpoints,
        "temporal_store": temporal_store,
        "temporal_control_contract": controls,
        "v0": v0,
        "v1": _v1_gap_report(root),
        "historical_cost_references": costs,
        "stage_decisions": {
            "current_state": {
                "T1": "blocked_on_fresh_role_safe_index_protocol_lock_and_fit_runner",
                "T2": "blocked_on_fresh_role_safe_index_protocol_lock_and_fit_runner",
                "V0": "blocked_on_fresh_protocol_shared_folds_and_four_arm_grid_lock",
                "V1": "blocked_on_images_spatial_grids_pose_auxiliary_and_provenance",
            },
            "after_R0_contracts_are_complete": {
                "T1": "executable_independent_of_R1_and_R2",
                "T2": "executable_independent_of_R1_and_R2",
                "V0": "executable_independent_of_R1_and_R2",
                "V1": "still_blocked_until_V0_and_new_input_extraction",
            },
            "R1_gate_controls_only_R2": True,
        },
        "access_audit": {
            "content_reads": audit.reads,
            "content_read_count": len(audit.reads),
            "calibration_reads": 0,
            "confirmation_reads": 0,
            "test_reads": 0,
            "manifest_reads": 0,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validate-checkpoints",
        action="store_true",
        help="Load every development checkpoint with torch weights_only=True and validate schema.",
    )
    args = parser.parse_args()
    report = build_report(args.root, validate_checkpoints=args.validate_checkpoints)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
