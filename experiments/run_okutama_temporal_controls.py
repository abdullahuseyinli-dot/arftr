"""Run the locked Okutama T1 pooling and T2 sampler controls.

Preparation is deliberately separate from fitting.  It derives a primary-only fold
map and the one shared scenario-resampling artifact from the role-safe bundle.  A new
execution lock must bind those artifacts before ``benchmark`` or ``full`` may run.
No historical mixed-role manifest or feature store is an input to this executable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from hac.polar_training import warmup_cosine_scheduler
from hac.training import seed_everything
from hac.vcoco_v3_neural import FactorizedClassifier, decode_factorized_logits
from hac.vcoco_v3_temporal import (
    TemporalFactorizedTeacher,
    TemporalOutput,
    temporal_teacher_loss,
)
from hac.vcoco_v3_temporal_training import hierarchical_class_weights

CLASS_NAMES = ("sitting", "standing", "walking_running")
CLASS_COUNT = len(CLASS_NAMES)
EXPECTED_PRIMARY_ROWS = 4_977
EXPECTED_GROUPS = 11
EXPECTED_FOLDS = tuple(f"fold-{index}" for index in range(5))
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
ARMS = ("t1_arithmetic_mean", "t2_legacy_repeated", "t2_fixed_distinct")
LEGACY_INDICES = np.asarray([4, 6, 6, 8, 8, 10, 10, 12], dtype=np.int64)
DISTINCT_INDICES = np.asarray([4, 5, 6, 7, 8, 10, 11, 12], dtype=np.int64)
LEGACY_FROM_DISTINCT = np.asarray([0, 2, 2, 4, 4, 5, 5, 7], dtype=np.int64)
CENTER_SLOT = 4
BOOTSTRAP_SEED = 20_260_906
BOOTSTRAP_RESAMPLES = 10_000
PREPARE_STATUS = "OKUTAMA_TEMPORAL_CONTROLS_PREPARE_COMPLETE"
LOCK_STATUS = "OKUTAMA_TEMPORAL_CONTROLS_LOCKED_BEFORE_FITTING"
RUN_STATUS = "OKUTAMA_TEMPORAL_CONTROL_WORKLOAD_COMPLETE"
BENCHMARK_STATUS = "OKUTAMA_TEMPORAL_CONTROL_BENCHMARK_COMPLETE"
FULL_STATUS = "OKUTAMA_TEMPORAL_CONTROLS_COMPLETE"
JOINT_LATENCY_STATUS = "OKUTAMA_TEMPORAL_CONTROLS_JOINT_LATENCY_COMPLETE"
MATERIALIZATION_LOCK_STATUS = "OKUTAMA_CPTR_REPLAY_MATERIALIZATION_LOCKED_BEFORE_FEATURE_ACCESS"


@dataclass(frozen=True)
class PrimaryData:
    sample_ids: np.ndarray
    recording_ids: np.ndarray
    labels: np.ndarray
    folds: np.ndarray
    bundle_row_indices: np.ndarray
    short_features: np.ndarray
    distinct_features: np.ndarray
    occluded: np.ndarray
    transition: np.ndarray
    retained_teacher_probabilities: np.ndarray


class ArithmeticMeanFactorizedHead(nn.Module):
    """A fixed, permutation-invariant control with no time or position signal."""

    def __init__(self, input_dim: int, *, model_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim < 1 or model_dim < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid arithmetic-mean head dimensions")
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = FactorizedClassifier(model_dim, dropout)

    def forward(
        self,
        frame_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> TemporalOutput:
        if frame_features.ndim != 3 or frame_features.shape[1] != 8:
            raise ValueError("Pooling input must have shape [batch, 8, feature_dim]")
        if valid_mask is not None:
            if valid_mask.shape != frame_features.shape[:2] or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be Boolean with shape [batch, 8]")
            if not torch.all(valid_mask):
                raise ValueError("T1 is locked to the teacher's all-valid semantics")
        pooled_input = frame_features.mean(dim=1)
        pooled = self.input_projection(pooled_input)
        posture_logits, motion_logits = self.classifier(pooled)
        attention = torch.full(
            frame_features.shape[:2],
            1.0 / frame_features.shape[1],
            dtype=frame_features.dtype,
            device=frame_features.device,
        )
        return TemporalOutput(
            probabilities=decode_factorized_logits(posture_logits, motion_logits),
            posture_logits=posture_logits,
            motion_logits=motion_logits,
            pooled_features=pooled,
            attention_weights=attention,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def validate_temporal_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    controls = protocol.get("temporal_controls")
    if not isinstance(controls, dict):
        raise RuntimeError("Protocol has no temporal_controls declaration")
    data_scope = controls.get("data_scope", {})
    expected_scope = {
        "rows": EXPECTED_PRIMARY_ROWS,
        "scenarios": EXPECTED_GROUPS,
        "scope_value": "grouped_crossfit_oof",
        "folds": 5,
        "seeds": list(EXPECTED_SEEDS),
    }
    if data_scope != expected_scope:
        raise RuntimeError("Temporal-control data scope differs from the frozen declaration")
    fold_spec = controls.get("generated_fold_map", {})
    if fold_spec != {
        "columns": ["bundle_row_index", "sample_id", "recording_id", "fold"],
        "row_order": "eligible_feature_bundle_primary_row_order",
        "group_isolation": True,
    }:
        raise RuntimeError("Temporal-control fold-map contract changed")
    resampling = controls.get("scenario_resampling", {})
    required_resampling = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_shape": [BOOTSTRAP_RESAMPLES, EXPECTED_GROUPS],
        "exact_swap_shape": [2**EXPECTED_GROUPS, EXPECTED_GROUPS],
        "canonical_group_order": "sorted_recording_id",
    }
    if resampling != required_resampling:
        raise RuntimeError("Temporal-control resampling contract changed")
    t1 = controls.get("T1_architecture", {})
    if t1 != {
        "arm_id": "t1_arithmetic_mean",
        "pooling": "unweighted_arithmetic_mean_all_eight_legacy_slots",
        "validity_mask": "all_valid",
        "positional_or_timestamp_signal": False,
        "input_projection": [
            "LayerNorm(input_dim)",
            "Linear(input_dim,256)",
            "GELU",
            "Dropout(0.1)",
        ],
        "classifier": "FactorizedClassifier(256,0.1)",
    }:
        raise RuntimeError("T1 architecture differs from the frozen declaration")
    t2 = controls.get("T2_architecture", {})
    if t2 != {
        "arms": ["t2_legacy_repeated", "t2_fixed_distinct"],
        "model": "TemporalFactorizedTeacher",
        "model_dim": 256,
        "layers": 2,
        "attention_heads": 4,
        "feedforward_dim": 512,
        "dropout": 0.1,
        "maximum_length": 8,
        "validity_mask": "all_valid",
        "legacy_indices": LEGACY_INDICES.tolist(),
        "distinct_indices": DISTINCT_INDICES.tolist(),
        "center_slot": CENTER_SLOT,
    }:
        raise RuntimeError("T2 architecture or sampler contract changed")
    training = controls.get("training", {})
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "batch_size": 64,
        "fixed_epochs_by_seed": {"42": 5, "43": 5, "44": 5, "45": 5, "46": 3},
        "epoch_proof_source": ".runs/vcoco_v3/temporal/development_final/summary.json",
        "warmup_fraction": 0.1,
        "gradient_clip_norm": 1.0,
        "label_smoothing": 0.02,
        "class_balance": "fit_partition_inverse_frequency",
        "automatic_mixed_precision": True,
        "execution_backend": "cuda",
        "candidate_selection": "none",
    }
    if training != expected_training:
        raise RuntimeError("Temporal-control training contract changed")
    joint_latency = controls.get("joint_latency_benchmark", {})
    if joint_latency != {
        "status": JOINT_LATENCY_STATUS,
        "candidate_arm": "t1_arithmetic_mean",
        "reference_arm": "t2_legacy_repeated",
        "checkpoint_fold": "fold-0",
        "checkpoint_seed": 42,
        "cohort_selection": "first_64_fold-0_rows_in_locked_primary_bundle_order",
        "cohort_rows": 64,
        "input_contract": "identical_legacy_short_features_all_valid_mask",
        "warmup_paired_rounds": 20,
        "timed_paired_rounds": 200,
        "order_schedule": (
            "even_round_candidate_then_reference_odd_round_reference_then_candidate"
        ),
        "synchronization": (
            "device_synchronize_before_and_after_each_individual_arm_timing"
        ),
        "timed_scope": "cached_CPU_feature_batch_to_CPU_probability_temporal_head",
        "paired_bootstrap_resamples": 10_000,
        "paired_bootstrap_seed": 20_260_907,
        "authoritative_p95_rule": (
            "candidate_observed_p95_below_reference_and_paired_round_bootstrap_"
            "p95_delta_one_sided_95pct_upper_below_zero"
        ),
        "isolated_per_workload_timings": "diagnostic_only",
    }:
        raise RuntimeError("Joint temporal-control latency benchmark contract changed")
    if controls.get("representative_benchmark") != {
        "workloads": [
            "t1_arithmetic_mean/fold-0/seed-42",
            "t2_legacy_repeated/fold-0/seed-42",
        ],
        "purpose": "measure_simple_and_expensive_family_fit_cost_before_full_schedule",
        "required_resource_fields": [
            "runtime_seconds",
            "peak_cuda_memory_allocated_bytes",
            "peak_cuda_memory_reserved_bytes",
        ],
        "must_complete_before_full": True,
    }:
        raise RuntimeError("Representative temporal-control benchmark contract changed")
    decisions = protocol.get("decision_rules", {})
    if decisions.get("T1_noninferiority") != {
        "accuracy_reference": "exact_retained_five_seed_teacher_probability_ensemble",
        "latency_reference": "t2_legacy_repeated_fold-0_seed-42",
        "latency_cohort": "same_fold-0_seed-42_cached_feature_cohort",
        "authoritative_latency_artifact": "joint_latency_benchmark.json",
        "authoritative_latency_measurement": "same_process_order_balanced_interleaved_AB_BA",
        "pooling_minus_teacher_macro_f1_one_sided_95pct_lower_bound_minimum": -0.005,
        "pooling_minus_teacher_nll_one_sided_95pct_upper_bound_maximum": 0.0,
        "same_cohort_p95_latency_must_improve": True,
        "paired_bootstrap_p95_delta_one_sided_95pct_upper_must_be_below": 0.0,
        "promotion_guardrails_required": True,
        "cached_head_latency_role": "provisional_component_evidence",
        "end_to_end_latency_required_before_preference": True,
        "isolated_per_workload_timings_role": "diagnostic_only",
    }:
        raise RuntimeError("T1 decision rule changed")
    if decisions.get("T2_sampler_benefit") != {
        "comparison": "t2_fixed_distinct_minus_newly_matched_t2_legacy_repeated",
        "aggregate_macro_f1_two_sided_95pct_lower_bound_must_exceed": 0.0,
        "aggregate_nll_one_sided_95pct_upper_bound_maximum": 0.0,
        "one_sided_exact_whole_scenario_swap_pvalue_maximum": 0.05,
        "promotion_guardrails_required": True,
        "point_size_threshold": None,
        "attribution": "sampling_only_never_gate_repair",
    }:
        raise RuntimeError("T2 decision rule changed")
    return controls


def expected_fold_for_group(protocol: dict[str, Any]) -> dict[str, str]:
    contract = protocol.get("fold_contract", {})
    if set(contract) != set(EXPECTED_FOLDS):
        raise RuntimeError("Historical five-fold contract changed")
    result: dict[str, str] = {}
    for fold, groups in contract.items():
        for group in groups:
            group = str(group)
            if group in result:
                raise RuntimeError(f"Scenario appears in multiple folds: {group}")
            result[group] = fold
    if len(result) != EXPECTED_GROUPS:
        raise RuntimeError("Historical fold contract must contain exactly 11 scenarios")
    return result


def validate_fold_frame(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    required = ["bundle_row_index", "sample_id", "recording_id", "fold"]
    if list(frame.columns) != required:
        raise RuntimeError("Fold-map columns or order changed")
    if len(frame) != EXPECTED_PRIMARY_ROWS or frame["sample_id"].duplicated().any():
        raise RuntimeError("Fold-map primary cardinality or row identity changed")
    if frame["bundle_row_index"].duplicated().any():
        raise RuntimeError("Fold-map bundle positions are not unique")
    group_to_fold = expected_fold_for_group(protocol)
    observed_groups = set(frame["recording_id"].astype(str))
    if observed_groups != set(group_to_fold):
        raise RuntimeError("Fold-map scenarios differ from the locked historical scenarios")
    observed = frame.groupby("recording_id", observed=True)["fold"].nunique()
    if not (observed == 1).all():
        raise RuntimeError("A scenario crosses a fit/held fold boundary")
    expected = frame["recording_id"].astype(str).map(group_to_fold)
    if expected.isna().any() or not np.array_equal(expected.to_numpy(), frame["fold"].to_numpy()):
        raise RuntimeError("Fold-map assignments differ from the historical fold contract")
    return frame


def make_resampling(group_order: np.ndarray) -> dict[str, np.ndarray]:
    groups = np.asarray(group_order, dtype=str)
    if len(groups) != EXPECTED_GROUPS or groups.tolist() != sorted(groups.tolist()):
        raise RuntimeError("Scenario resampling requires 11 canonically sorted groups")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    initial = json.dumps(generator.bit_generator.state, sort_keys=True)
    draws = generator.integers(
        0,
        len(groups),
        size=(BOOTSTRAP_RESAMPLES, len(groups)),
        dtype=np.int16,
    )
    final = json.dumps(generator.bit_generator.state, sort_keys=True)
    integers = np.arange(1 << len(groups), dtype=np.uint32)[:, None]
    swaps = ((integers >> np.arange(len(groups), dtype=np.uint32)) & 1).astype(bool)
    return {
        "group_order": groups,
        "group_draw_indices": draws,
        "exact_swap_assignments": swaps,
        "bootstrap_seed": np.asarray(BOOTSTRAP_SEED, dtype=np.int64),
        "numpy_version": np.asarray(np.__version__),
        "bit_generator": np.asarray(type(generator.bit_generator).__name__),
        "initial_rng_state_json": np.asarray(initial),
        "final_rng_state_json": np.asarray(final),
    }


def validate_resampling(path: Path, expected_groups: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "group_order",
            "group_draw_indices",
            "exact_swap_assignments",
            "bootstrap_seed",
            "numpy_version",
            "bit_generator",
            "initial_rng_state_json",
            "final_rng_state_json",
        }
        if set(archive.files) != required:
            raise RuntimeError("Scenario-resampling members changed")
        observed = {name: archive[name] for name in archive.files}
    expected = make_resampling(np.asarray(expected_groups, dtype=str))
    for name, value in expected.items():
        if not np.array_equal(observed[name], value):
            raise RuntimeError(f"Scenario-resampling value changed: {name}")
    return observed


def _validate_prepare_source(
    *, source_lock_path: Path, bundle_summary_path: Path, bundle_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_lock = load_json(source_lock_path)
    if source_lock.get("status") != MATERIALIZATION_LOCK_STATUS:
        raise RuntimeError("Temporal preparation requires the CPTR materialization source lock")
    if source_lock.get("authorization", {}).get("temporal_controls_artifact_construction") is not True:
        raise RuntimeError("Source lock does not authorize temporal-control artifact construction")
    summary = load_json(bundle_summary_path)
    if summary.get("status") != "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE":
        raise RuntimeError("Role-safe feature bundle is incomplete")
    if summary.get("artifact_sha256", {}).get(bundle_path.name) != sha256_file(bundle_path):
        raise RuntimeError("Role-safe bundle differs from its summary")
    if summary.get("source_sha256", {}).get("protocol_lock") != sha256_file(source_lock_path):
        raise RuntimeError("Role-safe bundle belongs to a different materialization lock")
    return source_lock, summary


def prepare_artifacts(
    *,
    protocol_path: Path,
    source_lock_path: Path,
    bundle_path: Path,
    bundle_summary_path: Path,
    fold_map_path: Path,
    resampling_path: Path,
    prepare_summary_path: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    validate_temporal_contract(protocol)
    _validate_prepare_source(
        source_lock_path=source_lock_path,
        bundle_summary_path=bundle_summary_path,
        bundle_path=bundle_path,
    )
    sources = {
        "source_lock": sha256_file(source_lock_path),
        "protocol": sha256_file(protocol_path),
        "eligible_feature_bundle": sha256_file(bundle_path),
        "bundle_summary": sha256_file(bundle_summary_path),
        "runner": sha256_file(Path(__file__).resolve()),
    }
    if fold_map_path.exists() or resampling_path.exists() or prepare_summary_path.exists():
        if not (fold_map_path.is_file() and resampling_path.is_file() and prepare_summary_path.is_file()):
            raise RuntimeError("Partial temporal-control preparation artifacts found; failed closed")
        previous = load_json(prepare_summary_path)
        if previous.get("status") != PREPARE_STATUS or previous.get("source_sha256") != sources:
            raise RuntimeError("Existing temporal-control preparation belongs to different sources")
        artifacts = previous.get("artifact_sha256", {})
        for path in (fold_map_path, resampling_path):
            if artifacts.get(path.name) != sha256_file(path):
                raise RuntimeError("Existing temporal-control preparation artifact changed")
        frame = pd.read_csv(fold_map_path, dtype=str, keep_default_na=False)
        validate_fold_frame(frame, protocol)
        validate_resampling(
            resampling_path, np.asarray(sorted(frame["recording_id"].unique()), dtype=str)
        )
        return {**previous, "resume_action": "verified_existing_preparation_reused"}

    with np.load(bundle_path, allow_pickle=False) as bundle:
        required = {"sample_ids", "recording_ids", "scope", "fold"}
        if not required <= set(bundle.files):
            raise RuntimeError(f"Role-safe bundle lacks preparation members: {sorted(required)}")
        sample_ids = bundle["sample_ids"].astype(str)
        recording_ids = bundle["recording_ids"].astype(str)
        scope = bundle["scope"].astype(str)
        folds = bundle["fold"].astype(str)
    if not (len(sample_ids) == len(recording_ids) == len(scope) == len(folds)):
        raise RuntimeError("Role-safe bundle metadata rows do not align")
    primary = np.flatnonzero(scope == "grouped_crossfit_oof")
    if len(primary) != EXPECTED_PRIMARY_ROWS:
        raise RuntimeError("Role-safe bundle primary row count changed")
    frame = pd.DataFrame(
        {
            "bundle_row_index": primary,
            "sample_id": sample_ids[primary],
            "recording_id": recording_ids[primary],
            "fold": folds[primary],
        }
    )
    validate_fold_frame(frame, protocol)
    fold_map_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = fold_map_path.with_name(fold_map_path.name + ".tmp")
    frame.to_csv(temporary_csv, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    temporary_csv.replace(fold_map_path)
    resampling = make_resampling(np.asarray(sorted(frame["recording_id"].unique()), dtype=str))
    write_npz_atomic(resampling_path, **resampling)
    summary = {
        "status": PREPARE_STATUS,
        "rows": EXPECTED_PRIMARY_ROWS,
        "scenarios": EXPECTED_GROUPS,
        "folds": len(EXPECTED_FOLDS),
        "model_fits": 0,
        "fits_performed": 0,
        "feature_values_read": 0,
        "protected_rows_read": 0,
        "source_sha256": sources,
        "artifact_sha256": {
            fold_map_path.name: sha256_file(fold_map_path),
            resampling_path.name: sha256_file(resampling_path),
        },
        "access_accounting": {
            "role_safe_bundle_metadata_rows_read": int(len(sample_ids)),
            "role_safe_feature_values_read": 0,
            "mixed_role_files_opened": 0,
            "protected_calibration_rows_read": 0,
            "confirmation_rows_read": 0,
            "test_rows_read": 0,
        },
    }
    write_json_atomic(prepare_summary_path, summary)
    return summary


def validate_execution_lock(
    *,
    execution_lock_path: Path,
    source_lock_path: Path,
    protocol_path: Path,
    bundle_path: Path,
    bundle_summary_path: Path,
    fold_map_path: Path,
    resampling_path: Path,
    prepare_summary_path: Path,
) -> dict[str, Any]:
    lock = load_json(execution_lock_path)
    if lock.get("status") != LOCK_STATUS:
        raise RuntimeError("Temporal-control execution lock is absent or has the wrong status")
    if lock.get("dataset") != "okutama" or lock.get("stage") != "temporal-controls":
        raise RuntimeError("Temporal-control execution lock dataset or stage changed")
    authorization = lock.get("authorization", {})
    if authorization != {
        "T1_execution": True,
        "T2_execution": True,
        "benchmark_then_resume": True,
    }:
        raise RuntimeError("Temporal-control execution authorization changed")
    root = Path(__file__).resolve().parents[1]
    sources = {
        "source_lock": sha256_file(source_lock_path),
        "protocol": sha256_file(protocol_path),
        "eligible_feature_bundle": sha256_file(bundle_path),
        "bundle_summary": sha256_file(bundle_summary_path),
        "fold_map": sha256_file(fold_map_path),
        "scenario_resampling": sha256_file(resampling_path),
        "prepare_summary": sha256_file(prepare_summary_path),
        "runner": sha256_file(Path(__file__).resolve()),
        "temporal_model_module": sha256_file(root / "src/hac/vcoco_v3_temporal.py"),
        "temporal_training_module": sha256_file(
            root / "src/hac/vcoco_v3_temporal_training.py"
        ),
        "neural_module": sha256_file(root / "src/hac/vcoco_v3_neural.py"),
        "polar_training_module": sha256_file(root / "src/hac/polar_training.py"),
        "epoch_proof_source": sha256_file(
            root / ".runs/vcoco_v3/temporal/development_final/summary.json"
        ),
    }
    locked = lock.get("source_sha256", {})
    for name, digest in sources.items():
        if locked.get(name) != digest:
            raise RuntimeError(f"Temporal-control locked source changed: {name}")
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError("Temporal controls require a clean committed worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if lock.get("repository_commit") != head or lock.get("repository_tree") != tree:
        raise RuntimeError("Temporal-control repository commit/tree differs from the lock")
    if lock.get("environment") != collect_environment():
        raise RuntimeError("Temporal-control execution environment differs from the lock")
    protected = lock.get("protected_access", {})
    expected_protected = {
        "mixed_manifest_rows_read": 0,
        "mixed_development_metadata_rows_read": 0,
        "calibration_rows_or_arrays_read": 0,
        "confirmation_rows_or_arrays_read": 0,
        "test_rows_or_arrays_read": 0,
    }
    if protected != expected_protected:
        raise RuntimeError("Temporal-control execution lock reports protected access")
    prepare = load_json(prepare_summary_path)
    if prepare.get("status") != PREPARE_STATUS:
        raise RuntimeError("Temporal-control preparation receipt is incomplete")
    if prepare.get("artifact_sha256", {}).get(fold_map_path.name) != sources["fold_map"]:
        raise RuntimeError("Preparation receipt does not bind the fold map")
    if (
        prepare.get("artifact_sha256", {}).get(resampling_path.name)
        != sources["scenario_resampling"]
    ):
        raise RuntimeError("Preparation receipt does not bind scenario resampling")
    return {**lock, "validated_source_sha256": sources}


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
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    environment["canonical_sha256"] = sha256_text(canonical)
    return environment


def validate_sampler_arrays(
    short_features: np.ndarray,
    distinct_features: np.ndarray,
    *,
    short_centre_index: int,
    distinct_centre_index: int,
    distinct_indices: np.ndarray,
) -> None:
    if short_features.ndim != 3 or distinct_features.ndim != 3:
        raise RuntimeError("Temporal bundle features must be rank-three")
    if short_features.shape != distinct_features.shape or short_features.shape[1] != 8:
        raise RuntimeError("Repeated and distinct temporal features must have matched [N,8,D] shape")
    if short_features.dtype != np.float32 or distinct_features.dtype != np.float32:
        raise RuntimeError("Temporal bundle features must remain float32")
    if not np.isfinite(short_features).all() or not np.isfinite(distinct_features).all():
        raise RuntimeError("Temporal bundle features contain non-finite values")
    if int(short_centre_index) != CENTER_SLOT or int(distinct_centre_index) != CENTER_SLOT:
        raise RuntimeError("Temporal center slot changed")
    if not np.array_equal(np.asarray(distinct_indices, dtype=np.int64), DISTINCT_INDICES):
        raise RuntimeError("Distinct temporal sampler indices changed")
    reconstructed = distinct_features[:, LEGACY_FROM_DISTINCT]
    if not np.array_equal(short_features, reconstructed):
        raise RuntimeError("Legacy repeated slots are not an exact view of the distinct source slots")
    if LEGACY_INDICES[CENTER_SLOT] != 8 or DISTINCT_INDICES[CENTER_SLOT] != 8:
        raise RuntimeError("Declared center frame is not retained at slot four")


def _load_fold_map(path: Path, protocol: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["bundle_row_index"] = pd.to_numeric(
        frame["bundle_row_index"], errors="raise"
    ).astype(np.int64)
    return validate_fold_frame(frame, protocol)


def load_primary_data(
    *,
    bundle_path: Path,
    bundle_summary_path: Path,
    fold_map_path: Path,
    protocol: dict[str, Any],
) -> PrimaryData:
    summary = load_json(bundle_summary_path)
    if summary.get("status") != "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE":
        raise RuntimeError("Role-safe feature bundle receipt is incomplete")
    if summary.get("artifact_sha256", {}).get(bundle_path.name) != sha256_file(bundle_path):
        raise RuntimeError("Role-safe feature bundle hash changed")
    required = {
        "sample_ids",
        "recording_ids",
        "scope",
        "fold",
        "labels",
        "transition_targets",
        "occlusion_targets",
        "short_features",
        "short_valid_mask",
        "short_centre_index",
        "distinct_short_features",
        "distinct_short_valid_mask",
        "distinct_short_indices",
        "distinct_short_centre_index",
        "primary_retained_teacher_probabilities",
        "primary_retained_teacher_sample_ids",
        "retained_teacher_seeds",
    }
    with np.load(bundle_path, allow_pickle=False) as bundle:
        if not required <= set(bundle.files):
            raise RuntimeError(f"Role-safe bundle lacks temporal-control members: {sorted(required - set(bundle.files))}")
        sample_ids_all = bundle["sample_ids"].astype(str)
        recording_ids_all = bundle["recording_ids"].astype(str)
        scopes = bundle["scope"].astype(str)
        folds_all = bundle["fold"].astype(str)
        labels_all = bundle["labels"].astype(np.int64)
        transition_all = bundle["transition_targets"]
        occluded_all = bundle["occlusion_targets"]
        short_all = bundle["short_features"]
        distinct_all = bundle["distinct_short_features"]
        short_mask = bundle["short_valid_mask"]
        distinct_mask = bundle["distinct_short_valid_mask"]
        short_center = int(bundle["short_centre_index"].item())
        distinct_center = int(bundle["distinct_short_centre_index"].item())
        distinct_indices = bundle["distinct_short_indices"]
        teacher = bundle["primary_retained_teacher_probabilities"]
        teacher_ids = bundle["primary_retained_teacher_sample_ids"].astype(str)
        teacher_seeds = bundle["retained_teacher_seeds"].astype(np.int64)
    rows = len(sample_ids_all)
    aligned = (
        recording_ids_all,
        scopes,
        folds_all,
        labels_all,
        transition_all,
        occluded_all,
        short_all,
        distinct_all,
        short_mask,
        distinct_mask,
    )
    if any(len(value) != rows for value in aligned):
        raise RuntimeError("Role-safe bundle temporal rows do not align")
    if len(np.unique(sample_ids_all)) != rows:
        raise RuntimeError("Role-safe bundle sample identities are not unique")
    if labels_all.dtype != np.int64 or set(labels_all.tolist()) != {0, 1, 2}:
        raise RuntimeError("Role-safe bundle labels changed")
    if transition_all.dtype != np.bool_ or occluded_all.dtype != np.bool_:
        raise RuntimeError("Role-safe temporal selectors must remain Boolean")
    if short_mask.dtype != np.bool_ or distinct_mask.dtype != np.bool_:
        raise RuntimeError("Role-safe temporal masks must remain Boolean")
    if short_mask.shape != short_all.shape[:2] or distinct_mask.shape != distinct_all.shape[:2]:
        raise RuntimeError("Role-safe temporal masks do not align with their slots")
    validate_sampler_arrays(
        short_all,
        distinct_all,
        short_centre_index=short_center,
        distinct_centre_index=distinct_center,
        distinct_indices=distinct_indices,
    )
    primary = np.flatnonzero(scopes == "grouped_crossfit_oof")
    if len(primary) != EXPECTED_PRIMARY_ROWS:
        raise RuntimeError("Role-safe bundle primary row count changed")
    if teacher.shape != (EXPECTED_PRIMARY_ROWS, len(EXPECTED_SEEDS), CLASS_COUNT):
        raise RuntimeError("Retained teacher probability tensor shape changed")
    if teacher.dtype != np.dtype("float32"):
        raise RuntimeError("Retained teacher probabilities must remain float32")
    if not np.isfinite(teacher).all() or (teacher < 0.0).any():
        raise RuntimeError("Retained teacher probabilities must be finite and nonnegative")
    if not np.allclose(teacher.sum(axis=2), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Retained teacher probability rows are not normalized")
    if not np.array_equal(teacher_ids, sample_ids_all[primary]):
        raise RuntimeError("Retained teacher IDs differ from primary bundle row order")
    if not np.array_equal(teacher_seeds, np.asarray(EXPECTED_SEEDS)):
        raise RuntimeError("Retained teacher seed order changed")
    fold_map = _load_fold_map(fold_map_path, protocol)
    expected_frame = pd.DataFrame(
        {
            "bundle_row_index": primary,
            "sample_id": sample_ids_all[primary],
            "recording_id": recording_ids_all[primary],
            "fold": folds_all[primary],
        }
    )
    for column in expected_frame:
        if not np.array_equal(fold_map[column].to_numpy(), expected_frame[column].to_numpy()):
            raise RuntimeError(f"Fold map differs from role-safe bundle: {column}")
    return PrimaryData(
        sample_ids=sample_ids_all[primary],
        recording_ids=recording_ids_all[primary],
        labels=labels_all[primary],
        folds=folds_all[primary],
        bundle_row_indices=primary,
        short_features=np.asarray(short_all[primary], dtype=np.float32),
        distinct_features=np.asarray(distinct_all[primary], dtype=np.float32),
        occluded=np.asarray(occluded_all[primary], dtype=bool),
        transition=np.asarray(transition_all[primary], dtype=bool),
        retained_teacher_probabilities=np.asarray(teacher, dtype=np.float64),
    )


def build_model(arm: str, input_dim: int) -> nn.Module:
    if arm == "t1_arithmetic_mean":
        return ArithmeticMeanFactorizedHead(input_dim, model_dim=256, dropout=0.1)
    if arm in {"t2_legacy_repeated", "t2_fixed_distinct"}:
        return TemporalFactorizedTeacher(
            input_dim,
            model_dim=256,
            layers=2,
            attention_heads=4,
            feedforward_dim=512,
            dropout=0.1,
            maximum_length=8,
        )
    raise ValueError(f"Unknown temporal-control arm: {arm}")


def features_for_arm(data: PrimaryData, arm: str) -> np.ndarray:
    if arm in {"t1_arithmetic_mean", "t2_legacy_repeated"}:
        return data.short_features
    if arm == "t2_fixed_distinct":
        return data.distinct_features
    raise ValueError(f"Unknown temporal-control arm: {arm}")


def _confusion(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(matrix, (labels, probabilities.argmax(axis=1)), 1)
    return matrix


def _f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1).astype(np.float64)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    return np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator != 0,
    )


def classification_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), CLASS_COUNT):
        raise ValueError("Probability matrix and labels do not align")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    matrix = _confusion(labels, probabilities)
    class_f1 = _f1_from_confusion(matrix)
    support = matrix.sum(axis=1)
    recall = np.divide(
        np.diagonal(matrix),
        support,
        out=np.zeros(CLASS_COUNT, dtype=np.float64),
        where=support != 0,
    )
    true_probability = probabilities[np.arange(len(labels)), labels]
    targets = np.eye(CLASS_COUNT, dtype=np.float64)[labels]
    return {
        "rows": int(len(labels)),
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "macro_f1": float(class_f1.mean()),
        "nll": float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "confusion": matrix.tolist(),
        "class_f1": {name: float(class_f1[index]) for index, name in enumerate(CLASS_NAMES)},
        "true_class_recall": {
            name: float(recall[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "class_support": {
            name: int(support[index]) for index, name in enumerate(CLASS_NAMES)
        },
    }


def _make_loader(
    dataset: TensorDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=generator,
    )


def _forward(model: nn.Module, features: torch.Tensor) -> TemporalOutput:
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    return model(features, mask)  # type: ignore[no-any-return]


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    posture_logits: list[np.ndarray] = []
    motion_logits: list[np.ndarray] = []
    for feature_batch, label_batch, index_batch in loader:
        values = feature_batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = _forward(model, values)
        labels.append(label_batch.numpy())
        row_indices.append(index_batch.numpy())
        probabilities.append(output.probabilities.float().cpu().numpy())
        posture_logits.append(output.posture_logits.float().cpu().numpy())
        motion_logits.append(output.motion_logits.float().cpu().numpy())
    return {
        "labels": np.concatenate(labels).astype(np.int64),
        "primary_row_indices": np.concatenate(row_indices).astype(np.int64),
        "probabilities": np.concatenate(probabilities).astype(np.float32),
        "posture_logits": np.concatenate(posture_logits).astype(np.float32),
        "motion_logits": np.concatenate(motion_logits).astype(np.float32),
    }


@torch.inference_mode()
def benchmark_model_latency(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if warmup < 1 or iterations < 20:
        raise ValueError("Latency benchmark requires >=1 warmup and >=20 timed iterations")
    cohort = torch.from_numpy(np.ascontiguousarray(features)).to(dtype=torch.float32)
    model.eval()

    def execute() -> None:
        values = cohort.to(device, non_blocking=False)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = _forward(model, values)
        _ = output.probabilities.float().cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        execute()
    durations = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        execute()
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    values = np.asarray(durations, dtype=np.float64)
    return {
        "scope": "cached_CPU_feature_batch_to_CPU_probability_temporal_head",
        "cohort_rows": int(len(features)),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(iterations),
        "batch_latency_ms_p50": float(np.quantile(values, 0.50)),
        "batch_latency_ms_p95": float(np.quantile(values, 0.95)),
        "per_row_latency_ms_p50": float(np.quantile(values, 0.50) / len(features)),
        "per_row_latency_ms_p95": float(np.quantile(values, 0.95) / len(features)),
        "throughput_rows_per_second_median": float(len(features) * 1000 / np.median(values)),
        "raw_batch_latency_ms": values.tolist(),
    }


def workload_paths(output_dir: Path, arm: str, fold: str, seed: int) -> dict[str, Path]:
    root = output_dir / "workloads" / arm / fold / f"seed-{seed}"
    return {
        "root": root,
        "request": root / "request.json",
        "predictions": root / "held_predictions.npz",
        "checkpoint": root / "checkpoint.pt",
        "history": root / "history.json",
        "summary": root / "summary.json",
    }


def validate_existing_workload(
    *,
    paths: dict[str, Path],
    request_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    if not paths["summary"].is_file():
        raise RuntimeError("Existing workload is partial; failed closed")
    summary = load_json(paths["summary"])
    if summary.get("status") != RUN_STATUS or summary.get("request_sha256") != request_sha256:
        raise RuntimeError("Existing workload request/source changed; failed closed")
    request = load_json(paths["request"])
    retained_request_sha256 = request.pop("request_sha256", None)
    if (
        retained_request_sha256 != request_sha256
        or canonical_json_sha256(request) != request_sha256
    ):
        raise RuntimeError("Existing workload request receipt changed; failed closed")
    for name in ("predictions", "checkpoint", "history"):
        path = paths[name]
        if not path.is_file() or summary.get("artifact_sha256", {}).get(path.name) != sha256_file(
            path
        ):
            raise RuntimeError(f"Existing workload artifact changed; failed closed: {name}")
    resource = summary.get("resource_measurement", {})
    if set(resource) != {
        "scope",
        "initial_cuda_memory_allocated_bytes",
        "initial_cuda_memory_reserved_bytes",
        "peak_cuda_memory_allocated_bytes",
        "peak_cuda_memory_reserved_bytes",
    }:
        raise RuntimeError("Existing workload resource schema changed; failed closed")
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        required = {
            "sample_ids",
            "recording_ids",
            "bundle_row_indices",
            "primary_row_indices",
            "labels",
            "posture_logits",
            "motion_logits",
            "probabilities",
        }
        if set(archive.files) != required or len(archive["sample_ids"]) != expected_rows:
            raise RuntimeError("Existing workload prediction schema/cardinality changed")
        probabilities = archive["probabilities"]
        if probabilities.shape != (expected_rows, CLASS_COUNT) or not np.isfinite(
            probabilities
        ).all():
            raise RuntimeError("Existing workload predictions are invalid")
    return summary


def workload_request(
    *,
    arm: str,
    fold: str,
    seed: int,
    data: PrimaryData,
    training: dict[str, Any],
    source_sha256: dict[str, str],
    latency_warmup: int,
    latency_iterations: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if arm not in ARMS or fold not in EXPECTED_FOLDS or seed not in EXPECTED_SEEDS:
        raise ValueError("Workload is outside the locked arm/fold/seed design")
    features = features_for_arm(data, arm)
    fit_indices = np.flatnonzero(data.folds != fold)
    held_indices = np.flatnonzero(data.folds == fold)
    if not len(fit_indices) or not len(held_indices):
        raise RuntimeError("Temporal-control fold has an empty fit or held partition")
    if set(data.recording_ids[fit_indices]).intersection(data.recording_ids[held_indices]):
        raise RuntimeError("A scenario crossed the temporal-control fit/held boundary")
    fixed_epochs = int(training["fixed_epochs_by_seed"][str(seed)])
    latency_designated = fold == "fold-0" and seed == 42
    configuration = {
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "fixed_epochs": fixed_epochs,
        "fit_rows": int(len(fit_indices)),
        "held_rows": int(len(held_indices)),
        "input_shape": list(features.shape[1:]),
        "all_valid_mask": True,
        "latency_designated": latency_designated,
        "latency_warmup": int(latency_warmup) if latency_designated else 0,
        "latency_iterations": int(latency_iterations) if latency_designated else 0,
    }
    return (
        {
            "status": "OKUTAMA_TEMPORAL_CONTROL_WORKLOAD_REQUEST",
            "configuration": configuration,
            "training": training,
            "source_sha256": source_sha256,
            "protected_access": {
                "mixed_role_files_opened": 0,
                "calibration_rows_or_arrays_read": 0,
                "confirmation_rows_or_arrays_read": 0,
                "test_rows_or_arrays_read": 0,
            },
        },
        fit_indices,
        held_indices,
    )


def run_workload(
    *,
    arm: str,
    fold: str,
    seed: int,
    data: PrimaryData,
    training: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    source_sha256: dict[str, str],
    latency_warmup: int,
    latency_iterations: int,
) -> dict[str, Any]:
    features = features_for_arm(data, arm)
    request_core, fit_indices, held_indices = workload_request(
        arm=arm,
        fold=fold,
        seed=seed,
        data=data,
        training=training,
        source_sha256=source_sha256,
        latency_warmup=latency_warmup,
        latency_iterations=latency_iterations,
    )
    configuration = request_core["configuration"]
    fixed_epochs = int(configuration["fixed_epochs"])
    latency_designated = bool(configuration["latency_designated"])
    request_sha256 = canonical_json_sha256(request_core)
    paths = workload_paths(output_dir, arm, fold, seed)
    existing = [path.exists() for name, path in paths.items() if name != "root"]
    if any(existing):
        summary = validate_existing_workload(
            paths=paths,
            request_sha256=request_sha256,
            expected_rows=len(held_indices),
        )
        if (
            summary.get("configuration") != request_core["configuration"]
            or summary.get("source_sha256") != source_sha256
            or summary.get("protected_access") != request_core["protected_access"]
        ):
            raise RuntimeError("Existing workload summary differs from its request; failed closed")
        return {**summary, "resume_action": "verified_existing_workload_reused"}
    paths["root"].mkdir(parents=True, exist_ok=False)
    write_json_atomic(paths["request"], {**request_core, "request_sha256": request_sha256})

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial_cuda_allocated = int(torch.cuda.memory_allocated(device))
        initial_cuda_reserved = int(torch.cuda.memory_reserved(device))
    else:
        initial_cuda_allocated = 0
        initial_cuda_reserved = 0
    started = time.perf_counter()
    seed_everything(seed)
    model = build_model(arm, int(features.shape[2])).to(device)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(data.labels),
        torch.arange(len(data.labels), dtype=torch.int64),
    )
    fit_loader = _make_loader(
        dataset,
        fit_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    held_loader = _make_loader(
        dataset,
        held_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=seed,
    )
    posture_weight, motion_weight = hierarchical_class_weights(data.labels[fit_indices], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=max(1, len(fit_loader) * fixed_epochs),
        warmup_fraction=float(training["warmup_fraction"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    history: list[dict[str, Any]] = []
    for epoch in range(fixed_epochs):
        model.train()
        losses = []
        for feature_batch, label_batch, _row_batch in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            values = feature_batch.to(device, non_blocking=True)
            labels = label_batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = _forward(model, values)
                loss = temporal_teacher_loss(
                    output,
                    labels,
                    label_smoothing=float(training["label_smoothing"]),
                    posture_weight=posture_weight,
                    motion_weight=motion_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().item()))
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps({"arm": arm, "fold": fold, "seed": seed, **row}), flush=True)

    held = evaluate_model(model, held_loader, device)
    if not np.array_equal(held["primary_row_indices"], held_indices):
        raise RuntimeError("Held prediction row order changed")
    latency = None
    if latency_designated:
        latency_indices = held_indices[: min(int(training["batch_size"]), len(held_indices))]
        latency = benchmark_model_latency(
            model,
            features[latency_indices],
            device=device,
            warmup=latency_warmup,
            iterations=latency_iterations,
        )
        latency["sample_id_sha256"] = sha256_text(
            "\n".join(data.sample_ids[latency_indices].tolist()) + "\n"
        )
    write_npz_atomic(
        paths["predictions"],
        sample_ids=data.sample_ids[held_indices],
        recording_ids=data.recording_ids[held_indices],
        bundle_row_indices=data.bundle_row_indices[held_indices],
        primary_row_indices=held["primary_row_indices"],
        labels=held["labels"],
        posture_logits=held["posture_logits"],
        motion_logits=held["motion_logits"],
        probabilities=held["probabilities"],
    )
    checkpoint_temporary = paths["checkpoint"].with_name("checkpoint.tmp.pt")
    torch.save(
        {
            "request_sha256": request_sha256,
            "arm": arm,
            "fold": fold,
            "seed": seed,
            "fixed_epochs": fixed_epochs,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_temporary,
    )
    checkpoint_temporary.replace(paths["checkpoint"])
    write_json_atomic(paths["history"], {"epochs": history})
    probabilities = held["probabilities"].astype(np.float64)
    resource_measurement = {
        "scope": "model_construction_fixed_epoch_fit_held_inference_and_diagnostic_latency",
        "initial_cuda_memory_allocated_bytes": initial_cuda_allocated,
        "initial_cuda_memory_reserved_bytes": initial_cuda_reserved,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }
    summary = {
        "status": RUN_STATUS,
        "configuration": configuration,
        "request_sha256": request_sha256,
        "held_metrics": classification_summary(held["labels"], probabilities),
        "latency_benchmark": latency,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "resource_measurement": resource_measurement,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "source_sha256": source_sha256,
        "artifact_sha256": {
            paths["predictions"].name: sha256_file(paths["predictions"]),
            paths["checkpoint"].name: sha256_file(paths["checkpoint"]),
            paths["history"].name: sha256_file(paths["history"]),
        },
        "protected_access": request_core["protected_access"],
    }
    write_json_atomic(paths["summary"], summary)
    return summary


def load_completed_workload_model(
    *,
    arm: str,
    fold: str,
    seed: int,
    data: PrimaryData,
    training: dict[str, Any],
    output_dir: Path,
    source_sha256: dict[str, str],
    latency_warmup: int,
    latency_iterations: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], np.ndarray]:
    request_core, _fit_indices, held_indices = workload_request(
        arm=arm,
        fold=fold,
        seed=seed,
        data=data,
        training=training,
        source_sha256=source_sha256,
        latency_warmup=latency_warmup,
        latency_iterations=latency_iterations,
    )
    request_sha256 = canonical_json_sha256(request_core)
    paths = workload_paths(output_dir, arm, fold, seed)
    summary = validate_existing_workload(
        paths=paths,
        request_sha256=request_sha256,
        expected_rows=len(held_indices),
    )
    if (
        summary.get("configuration") != request_core["configuration"]
        or summary.get("source_sha256") != source_sha256
        or summary.get("protected_access") != request_core["protected_access"]
    ):
        raise RuntimeError("Temporal-control workload summary differs from its request")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "request_sha256",
        "arm",
        "fold",
        "seed",
        "fixed_epochs",
        "model_state_dict",
    }:
        raise RuntimeError("Temporal-control checkpoint top-level schema changed")
    expected_scalars = {
        "request_sha256": request_sha256,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "fixed_epochs": int(training["fixed_epochs_by_seed"][str(seed)]),
    }
    for name, expected in expected_scalars.items():
        if checkpoint.get(name) != expected:
            raise RuntimeError(f"Temporal-control checkpoint identity changed: {name}")
    model = build_model(arm, int(features_for_arm(data, arm).shape[2]))
    expected_state = model.state_dict()
    observed_state = checkpoint["model_state_dict"]
    if not isinstance(observed_state, dict) or set(observed_state) != set(expected_state):
        raise RuntimeError("Temporal-control checkpoint state keys changed")
    for name, expected in expected_state.items():
        observed = observed_state[name]
        if (
            not isinstance(observed, torch.Tensor)
            or observed.shape != expected.shape
            or observed.dtype != expected.dtype
        ):
            raise RuntimeError(f"Temporal-control checkpoint tensor schema changed: {name}")
        if not torch.isfinite(observed).all():
            raise RuntimeError(f"Temporal-control checkpoint contains non-finite tensor: {name}")
    model.load_state_dict(observed_state, strict=True)
    model.to(device).eval()
    return model, summary, held_indices


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@torch.inference_mode()
def measure_interleaved_latency(
    candidate: nn.Module,
    reference: nn.Module,
    feature_batch: np.ndarray,
    *,
    device: torch.device,
    warmup_paired_rounds: int,
    timed_paired_rounds: int,
) -> dict[str, Any]:
    if warmup_paired_rounds < 1 or timed_paired_rounds < 2 or timed_paired_rounds % 2:
        raise ValueError("Joint latency requires warmup and an even number of paired rounds")
    if feature_batch.ndim != 3 or feature_batch.shape[1] != 8:
        raise ValueError("Joint latency feature batch must have shape [rows,8,features]")
    shared_cpu_batch = torch.from_numpy(np.ascontiguousarray(feature_batch)).float()
    candidate.eval()
    reference.eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def execute(model: nn.Module) -> None:
        values = shared_cpu_batch.to(device, non_blocking=False)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = _forward(model, values).probabilities
        _ = probabilities.float().cpu()

    def execute_timed(model: nn.Module) -> float:
        synchronize()
        started = time.perf_counter_ns()
        execute(model)
        synchronize()
        return (time.perf_counter_ns() - started) / 1_000_000.0

    models = {"candidate": candidate, "reference": reference}
    for round_index in range(warmup_paired_rounds):
        order = ("candidate", "reference") if round_index % 2 == 0 else (
            "reference",
            "candidate",
        )
        for name in order:
            execute(models[name])
        synchronize()

    order_labels: list[str] = []
    candidate_ms = np.empty(timed_paired_rounds, dtype=np.float64)
    reference_ms = np.empty(timed_paired_rounds, dtype=np.float64)
    for round_index in range(timed_paired_rounds):
        order = ("candidate", "reference") if round_index % 2 == 0 else (
            "reference",
            "candidate",
        )
        order_labels.append("AB" if round_index % 2 == 0 else "BA")
        for name in order:
            duration = execute_timed(models[name])
            if name == "candidate":
                candidate_ms[round_index] = duration
            else:
                reference_ms[round_index] = duration
    if order_labels.count("AB") != order_labels.count("BA"):
        raise RuntimeError("Joint latency order schedule is not exactly balanced")
    return {
        "order": order_labels,
        "candidate_batch_latency_ms": candidate_ms.tolist(),
        "reference_batch_latency_ms": reference_ms.tolist(),
        "paired_candidate_minus_reference_ms": (candidate_ms - reference_ms).tolist(),
        "paired_candidate_over_reference_ratio": (candidate_ms / reference_ms).tolist(),
    }


def summarize_joint_latency(
    measurements: dict[str, Any],
    *,
    cohort_rows: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    candidate = np.asarray(measurements["candidate_batch_latency_ms"], dtype=np.float64)
    reference = np.asarray(measurements["reference_batch_latency_ms"], dtype=np.float64)
    order = np.asarray(measurements["order"], dtype=str)
    if (
        candidate.shape != reference.shape
        or len(candidate) != len(order)
        or not len(candidate)
        or not np.isfinite(candidate).all()
        or not np.isfinite(reference).all()
        or np.any(candidate <= 0.0)
        or np.any(reference <= 0.0)
    ):
        raise RuntimeError("Joint latency measurements are invalid")
    if np.count_nonzero(order == "AB") != np.count_nonzero(order == "BA"):
        raise RuntimeError("Joint latency measurements are not order balanced")
    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(
        0,
        len(candidate),
        size=(bootstrap_resamples, len(candidate)),
        dtype=np.int16,
    )
    candidate_p95 = np.quantile(candidate[draws], 0.95, axis=1)
    reference_p95 = np.quantile(reference[draws], 0.95, axis=1)
    delta_p95 = candidate_p95 - reference_p95
    ratio_p95 = candidate_p95 / reference_p95

    def arm_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "batch_latency_ms_p50": float(np.quantile(values, 0.50)),
            "batch_latency_ms_p95": float(np.quantile(values, 0.95)),
            "per_row_latency_ms_p50": float(np.quantile(values, 0.50) / cohort_rows),
            "per_row_latency_ms_p95": float(np.quantile(values, 0.95) / cohort_rows),
            "throughput_rows_per_second_median": float(cohort_rows * 1000 / np.median(values)),
        }

    candidate_summary = arm_summary(candidate)
    reference_summary = arm_summary(reference)
    return {
        "candidate": candidate_summary,
        "reference": reference_summary,
        "paired_round_delta_ms": {
            "p50": float(np.quantile(candidate - reference, 0.50)),
            "two_sided_95pct": [
                float(np.quantile(candidate - reference, 0.025)),
                float(np.quantile(candidate - reference, 0.975)),
            ],
        },
        "paired_round_ratio": {
            "p50": float(np.quantile(candidate / reference, 0.50)),
            "two_sided_95pct": [
                float(np.quantile(candidate / reference, 0.025)),
                float(np.quantile(candidate / reference, 0.975)),
            ],
        },
        "order_strata": {
            name: {
                "rounds": int(np.count_nonzero(order == name)),
                "candidate_ms_p50": float(np.median(candidate[order == name])),
                "reference_ms_p50": float(np.median(reference[order == name])),
                "paired_delta_ms_p50": float(np.median((candidate - reference)[order == name])),
            }
            for name in ("AB", "BA")
        },
        "paired_p95_bootstrap": {
            "resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed),
            "candidate_minus_reference_ms_observed": float(
                candidate_summary["batch_latency_ms_p95"]
                - reference_summary["batch_latency_ms_p95"]
            ),
            "candidate_minus_reference_ms_one_sided_95pct_upper": float(
                np.quantile(delta_p95, 0.95)
            ),
            "candidate_over_reference_observed": float(
                candidate_summary["batch_latency_ms_p95"]
                / reference_summary["batch_latency_ms_p95"]
            ),
            "candidate_over_reference_one_sided_95pct_upper": float(
                np.quantile(ratio_p95, 0.95)
            ),
        },
    }


def validate_joint_measurement_design(
    measurements: dict[str, Any], design: dict[str, Any]
) -> None:
    rounds = int(design["timed_paired_rounds"])
    expected_order = ["AB" if index % 2 == 0 else "BA" for index in range(rounds)]
    if measurements.get("order") != expected_order:
        raise RuntimeError("Joint latency AB/BA order differs from the frozen schedule")
    for name in (
        "candidate_batch_latency_ms",
        "reference_batch_latency_ms",
        "paired_candidate_minus_reference_ms",
        "paired_candidate_over_reference_ratio",
    ):
        values = measurements.get(name)
        if not isinstance(values, list) or len(values) != rounds:
            raise RuntimeError(f"Joint latency paired measurement length changed: {name}")
    candidate = np.asarray(measurements["candidate_batch_latency_ms"], dtype=np.float64)
    reference = np.asarray(measurements["reference_batch_latency_ms"], dtype=np.float64)
    stored_delta = np.asarray(
        measurements["paired_candidate_minus_reference_ms"], dtype=np.float64
    )
    stored_ratio = np.asarray(
        measurements["paired_candidate_over_reference_ratio"], dtype=np.float64
    )
    if not np.array_equal(stored_delta, candidate - reference) or not np.array_equal(
        stored_ratio, candidate / reference
    ):
        raise RuntimeError("Joint latency paired deltas/ratios differ from raw arm timings")


def run_joint_latency_benchmark(
    *,
    data: PrimaryData,
    training: dict[str, Any],
    design: dict[str, Any],
    output_dir: Path,
    source_sha256: dict[str, str],
    latency_warmup: int,
    latency_iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    candidate_arm = str(design["candidate_arm"])
    reference_arm = str(design["reference_arm"])
    fold = str(design["checkpoint_fold"])
    seed = int(design["checkpoint_seed"])
    candidate, candidate_summary, candidate_held = load_completed_workload_model(
        arm=candidate_arm,
        fold=fold,
        seed=seed,
        data=data,
        training=training,
        output_dir=output_dir,
        source_sha256=source_sha256,
        latency_warmup=latency_warmup,
        latency_iterations=latency_iterations,
        device=device,
    )
    reference, reference_summary, reference_held = load_completed_workload_model(
        arm=reference_arm,
        fold=fold,
        seed=seed,
        data=data,
        training=training,
        output_dir=output_dir,
        source_sha256=source_sha256,
        latency_warmup=latency_warmup,
        latency_iterations=latency_iterations,
        device=device,
    )
    if not np.array_equal(candidate_held, reference_held):
        raise RuntimeError("Joint latency workload held cohorts differ")
    cohort_indices = candidate_held[: int(design["cohort_rows"])]
    if len(cohort_indices) != int(design["cohort_rows"]):
        raise RuntimeError("Joint latency cohort is smaller than the frozen batch")
    feature_batch = data.short_features[cohort_indices]
    sample_id_sha256 = sha256_text("\n".join(data.sample_ids[cohort_indices].tolist()) + "\n")
    checkpoint_hashes = {
        candidate_arm: candidate_summary["artifact_sha256"]["checkpoint.pt"],
        reference_arm: reference_summary["artifact_sha256"]["checkpoint.pt"],
    }
    request_core = {
        "status": "OKUTAMA_TEMPORAL_CONTROLS_JOINT_LATENCY_REQUEST",
        "design": design,
        "source_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "workload_request_sha256": {
            candidate_arm: candidate_summary["request_sha256"],
            reference_arm: reference_summary["request_sha256"],
        },
        "cohort": {
            "rows": int(len(cohort_indices)),
            "sample_id_sha256": sample_id_sha256,
            "feature_array_sha256": sha256_array(feature_batch),
            "input": "identical_shared_CPU_tensor_for_both_arms",
        },
    }
    request_sha256 = canonical_json_sha256(request_core)
    output_path = output_dir / "joint_latency_benchmark.json"
    if output_path.exists():
        previous = load_json(output_path)
        if set(previous) != {
            "status",
            "request",
            "request_sha256",
            "raw_paired_measurements",
            "statistics",
            "authoritative_latency_checks",
            "authoritative_latency_passed",
            "isolated_per_workload_timings_role",
            "device",
        }:
            raise RuntimeError("Existing joint latency artifact schema changed; failed closed")
        if (
            previous.get("status") != JOINT_LATENCY_STATUS
            or previous.get("request_sha256") != request_sha256
            or previous.get("request") != request_core
            or previous.get("isolated_per_workload_timings_role") != "diagnostic_only"
            or previous.get("device")
            != (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
        ):
            raise RuntimeError("Existing joint latency benchmark changed; failed closed")
        raw_measurements = previous.get("raw_paired_measurements", {})
        validate_joint_measurement_design(raw_measurements, design)
        recalculated = summarize_joint_latency(
            raw_measurements,
            cohort_rows=int(design["cohort_rows"]),
            bootstrap_resamples=int(design["paired_bootstrap_resamples"]),
            bootstrap_seed=int(design["paired_bootstrap_seed"]),
        )
        checks = {
            "candidate_observed_p95_below_reference": float(
                recalculated["candidate"]["batch_latency_ms_p95"]
            )
            < float(recalculated["reference"]["batch_latency_ms_p95"]),
            "paired_bootstrap_p95_delta_upper_below_zero": float(
                recalculated["paired_p95_bootstrap"][
                    "candidate_minus_reference_ms_one_sided_95pct_upper"
                ]
            )
            < 0.0,
        }
        if (
            previous.get("statistics") != recalculated
            or previous.get("authoritative_latency_checks") != checks
            or previous.get("authoritative_latency_passed") != all(checks.values())
        ):
            raise RuntimeError("Existing joint latency statistics changed; failed closed")
        return {**previous, "resume_action": "verified_existing_joint_benchmark_reused"}

    measurements = measure_interleaved_latency(
        candidate,
        reference,
        feature_batch,
        device=device,
        warmup_paired_rounds=int(design["warmup_paired_rounds"]),
        timed_paired_rounds=int(design["timed_paired_rounds"]),
    )
    validate_joint_measurement_design(measurements, design)
    statistics = summarize_joint_latency(
        measurements,
        cohort_rows=int(design["cohort_rows"]),
        bootstrap_resamples=int(design["paired_bootstrap_resamples"]),
        bootstrap_seed=int(design["paired_bootstrap_seed"]),
    )
    checks = {
        "candidate_observed_p95_below_reference": float(
            statistics["candidate"]["batch_latency_ms_p95"]
        )
        < float(statistics["reference"]["batch_latency_ms_p95"]),
        "paired_bootstrap_p95_delta_upper_below_zero": float(
            statistics["paired_p95_bootstrap"][
                "candidate_minus_reference_ms_one_sided_95pct_upper"
            ]
        )
        < 0.0,
    }
    result = {
        "status": JOINT_LATENCY_STATUS,
        "request": request_core,
        "request_sha256": request_sha256,
        "raw_paired_measurements": measurements,
        "statistics": statistics,
        "authoritative_latency_checks": checks,
        "authoritative_latency_passed": all(checks.values()),
        "isolated_per_workload_timings_role": "diagnostic_only",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    write_json_atomic(output_path, result)
    return result


def _group_statistics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    group_order: np.ndarray,
    selector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    confusion = np.zeros((len(group_order), CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    loss_sum = np.zeros(len(group_order), dtype=np.float64)
    rows = np.zeros(len(group_order), dtype=np.int64)
    true_counts = np.zeros((len(group_order), CLASS_COUNT), dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    for group_index, group in enumerate(group_order):
        selected = selector & (groups == group)
        rows[group_index] = int(selected.sum())
        if not rows[group_index]:
            continue
        np.add.at(confusion[group_index], (labels[selected], predicted[selected]), 1)
        true_counts[group_index] = np.bincount(labels[selected], minlength=CLASS_COUNT)
        loss_sum[group_index] = float(losses[selected].sum())
    return confusion, loss_sum, rows, true_counts


def bootstrap_contrast(
    *,
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    group_order: np.ndarray,
    selector: np.ndarray,
    draws: np.ndarray,
) -> dict[str, Any]:
    ref_conf, ref_loss, ref_rows, ref_true = _group_statistics(
        labels, reference, groups, group_order, selector
    )
    cand_conf, cand_loss, cand_rows, cand_true = _group_statistics(
        labels, candidate, groups, group_order, selector
    )
    if not np.array_equal(ref_rows, cand_rows) or not np.array_equal(ref_true, cand_true):
        raise RuntimeError("Paired bootstrap support differs between temporal controls")
    sampled_rows = ref_rows[draws].sum(axis=1)
    sampled_true = ref_true[draws].sum(axis=1)
    valid = (sampled_rows > 0) & np.all(sampled_true > 0, axis=1)
    ref_f1 = _f1_from_confusion(ref_conf[draws].sum(axis=1)).mean(axis=1)
    cand_f1 = _f1_from_confusion(cand_conf[draws].sum(axis=1)).mean(axis=1)
    f1_delta = cand_f1 - ref_f1
    nll_delta = np.divide(
        (cand_loss[draws] - ref_loss[draws]).sum(axis=1),
        sampled_rows,
        out=np.full(len(draws), np.nan, dtype=np.float64),
        where=sampled_rows > 0,
    )
    selected_f1 = f1_delta[valid]
    selected_nll = nll_delta[valid]
    if not len(selected_f1):
        raise RuntimeError("No bootstrap draw retained all three fixed classes")
    observed_rows = int(ref_rows.sum())
    observed_f1 = float(
        _f1_from_confusion(cand_conf.sum(axis=0)).mean()
        - _f1_from_confusion(ref_conf.sum(axis=0)).mean()
    )
    return {
        "observed_rows": observed_rows,
        "observed_macro_f1_delta": observed_f1,
        "observed_nll_delta": float((cand_loss.sum() - ref_loss.sum()) / observed_rows),
        "valid_resamples": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "macro_f1_delta_one_sided_95pct_lower": float(np.quantile(selected_f1, 0.05)),
        "macro_f1_delta_two_sided_95pct": [
            float(np.quantile(selected_f1, 0.025)),
            float(np.quantile(selected_f1, 0.975)),
        ],
        "nll_delta_one_sided_95pct_upper": float(np.quantile(selected_nll, 0.95)),
        "nll_delta_two_sided_95pct": [
            float(np.quantile(selected_nll, 0.025)),
            float(np.quantile(selected_nll, 0.975)),
        ],
    }


def exact_group_swap(
    *,
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    group_order: np.ndarray,
    assignments: np.ndarray,
) -> dict[str, Any]:
    selector = np.ones(len(labels), dtype=bool)
    ref_conf, _, _, _ = _group_statistics(labels, reference, groups, group_order, selector)
    cand_conf, _, _, _ = _group_statistics(labels, candidate, groups, group_order, selector)
    if assignments.shape != (2 ** len(group_order), len(group_order)):
        raise RuntimeError("Exact scenario-swap assignment shape changed")
    bits = assignments.astype(bool, copy=False)
    candidate_null = np.where(bits[:, :, None, None], ref_conf[None], cand_conf[None]).sum(
        axis=1
    )
    reference_null = np.where(bits[:, :, None, None], cand_conf[None], ref_conf[None]).sum(
        axis=1
    )
    null_delta = _f1_from_confusion(candidate_null).mean(axis=1) - _f1_from_confusion(
        reference_null
    ).mean(axis=1)
    observed = float(
        _f1_from_confusion(cand_conf.sum(axis=0)).mean()
        - _f1_from_confusion(ref_conf.sum(axis=0)).mean()
    )
    return {
        "observed_macro_f1_delta": observed,
        "one_sided_pvalue": float(np.mean(null_delta >= observed - 1e-15)),
        "assignments": int(len(assignments)),
    }


def _error_flow(
    labels: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, int | float | None]:
    reference_correct = reference.argmax(axis=1) == labels
    candidate_correct = candidate.argmax(axis=1) == labels
    rescued = int((~reference_correct & candidate_correct).sum())
    harmed = int((reference_correct & ~candidate_correct).sum())
    errors = int((~reference_correct).sum())
    correct = int(reference_correct.sum())
    return {
        "both_correct": int((reference_correct & candidate_correct).sum()),
        "rescued": rescued,
        "harmed": harmed,
        "both_wrong": int((~reference_correct & ~candidate_correct).sum()),
        "rescue_fraction_of_reference_errors": float(rescued / errors) if errors else None,
        "harm_fraction_of_reference_correct": float(harmed / correct) if correct else None,
    }


def _slice_report(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    selector: np.ndarray,
) -> dict[str, Any]:
    selected_labels = labels[selector]
    if not len(selected_labels):
        return {"rows": 0, "supported_classes": 0, "status": "EMPTY"}
    ref = classification_summary(selected_labels, reference[selector])
    cand = classification_summary(selected_labels, candidate[selector])
    counts = np.bincount(selected_labels, minlength=CLASS_COUNT)
    return {
        "rows": int(len(selected_labels)),
        "supported_classes": int(np.count_nonzero(counts)),
        "class_counts": {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
        "reference": ref,
        "candidate": cand,
        "candidate_minus_reference": {
            "macro_f1": float(cand["macro_f1"] - ref["macro_f1"]),
            "accuracy": float(cand["accuracy"] - ref["accuracy"]),
            "nll": float(cand["nll"] - ref["nll"]),
            "brier": float(cand["brier"] - ref["brier"]),
            "class_f1": {
                name: float(cand["class_f1"][name] - ref["class_f1"][name])
                for name in CLASS_NAMES
            },
        },
        "error_flow": _error_flow(selected_labels, reference[selector], candidate[selector]),
    }


def comparison_statistics(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    occluded: np.ndarray,
    transition: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    resampling: dict[str, np.ndarray],
) -> dict[str, Any]:
    group_order = resampling["group_order"].astype(str)
    draws = resampling["group_draw_indices"].astype(np.int64)
    swaps = resampling["exact_swap_assignments"].astype(bool)
    selectors = {
        "aggregate": np.ones(len(labels), dtype=bool),
        "window_occluded": occluded,
        "window_clear": ~occluded,
        "transition": transition,
        "non_transition": ~transition,
    }
    bootstrap: dict[str, Any] = {}
    slices: dict[str, Any] = {}
    for name, selector in selectors.items():
        slices[name] = _slice_report(labels, reference, candidate, selector)
        if np.all(np.bincount(labels[selector], minlength=CLASS_COUNT) > 0):
            bootstrap[name] = bootstrap_contrast(
                labels=labels,
                reference=reference,
                candidate=candidate,
                groups=groups,
                group_order=group_order,
                selector=selector,
                draws=draws,
            )
        else:
            bootstrap[name] = {
                "status": "INSUFFICIENT_FIXED_CLASS_SUPPORT",
                "class_counts": np.bincount(
                    labels[selector], minlength=CLASS_COUNT
                ).tolist(),
            }
    scenarios = {
        str(group): _slice_report(labels, reference, candidate, groups == group)
        for group in group_order
    }
    aggregate_delta = slices["aggregate"]["candidate_minus_reference"]
    supported_bootstraps = [
        value for value in bootstrap.values() if "valid_fraction" in value
    ]
    adequate_scenarios = [
        report
        for report in scenarios.values()
        if int(report.get("supported_classes", 0)) == CLASS_COUNT
    ]
    adequate_strata = [
        report
        for name, report in slices.items()
        if name != "aggregate" and int(report.get("supported_classes", 0)) == CLASS_COUNT
    ]
    occlusion_bootstrap = bootstrap["window_occluded"]
    checks = {
        "no_full_cohort_class_f1_regression_below_minus_0_010": all(
            float(aggregate_delta["class_f1"][name]) >= -0.010 for name in CLASS_NAMES
        ),
        "occlusion_macro_f1_lower_at_or_above_minus_0_010": (
            "macro_f1_delta_one_sided_95pct_lower" in occlusion_bootstrap
            and float(occlusion_bootstrap["macro_f1_delta_one_sided_95pct_lower"]) >= -0.010
        ),
        "all_supported_subgroup_bootstraps_at_least_95pct_valid": all(
            float(report["valid_fraction"]) >= 0.95 for report in supported_bootstraps
        ),
        "no_adequately_supported_scenario_below_minus_0_010": all(
            float(report["candidate_minus_reference"]["macro_f1"]) >= -0.010
            for report in adequate_scenarios
        ),
        "no_adequately_supported_stratum_below_minus_0_010": all(
            float(report["candidate_minus_reference"]["macro_f1"]) >= -0.010
            for report in adequate_strata
        ),
    }
    return {
        "bootstrap": bootstrap,
        "exact_directional_scenario_swap": exact_group_swap(
            labels=labels,
            reference=reference,
            candidate=candidate,
            groups=groups,
            group_order=group_order,
            assignments=swaps,
        ),
        "declared_strata": slices,
        "scenarios": scenarios,
        "promotion_guardrails": {"checks": checks, "passed": all(checks.values())},
    }


def _write_or_validate_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise RuntimeError(f"Existing aggregate schema changed: {path}")
            for name, value in arrays.items():
                if not np.array_equal(existing[name], value):
                    raise RuntimeError(f"Existing aggregate value changed: {path}:{name}")
        return
    write_npz_atomic(path, **arrays)


def aggregate_arm_seed(
    *,
    arm: str,
    seed: int,
    data: PrimaryData,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    rows: list[dict[str, np.ndarray]] = []
    for fold in EXPECTED_FOLDS:
        path = workload_paths(output_dir, arm, fold, seed)["predictions"]
        if not path.is_file():
            raise RuntimeError(f"Missing completed temporal-control workload: {arm}/{fold}/{seed}")
        with np.load(path, allow_pickle=False) as archive:
            rows.append({name: archive[name] for name in archive.files})
    merged = {name: np.concatenate([row[name] for row in rows]) for name in rows[0]}
    order = np.argsort(merged["primary_row_indices"], kind="stable")
    merged = {name: value[order] for name, value in merged.items()}
    if not np.array_equal(merged["primary_row_indices"], np.arange(EXPECTED_PRIMARY_ROWS)):
        raise RuntimeError("Arm/seed OOF predictions do not cover every primary row exactly once")
    expected = {
        "sample_ids": data.sample_ids,
        "recording_ids": data.recording_ids,
        "bundle_row_indices": data.bundle_row_indices,
        "primary_row_indices": np.arange(EXPECTED_PRIMARY_ROWS, dtype=np.int64),
        "labels": data.labels,
    }
    for name, value in expected.items():
        if not np.array_equal(merged[name], value):
            raise RuntimeError(f"Arm/seed OOF identity changed: {arm}/{seed}/{name}")
    aggregate_path = output_dir / "aggregates" / arm / f"seed-{seed}.npz"
    _write_or_validate_npz(aggregate_path, merged)
    return merged


def _load_designated_latency(output_dir: Path, arm: str) -> dict[str, Any]:
    summary = load_json(workload_paths(output_dir, arm, "fold-0", 42)["summary"])
    latency = summary.get("latency_benchmark")
    if not isinstance(latency, dict):
        raise RuntimeError(f"Designated latency benchmark is absent for {arm}")
    return latency


def aggregate_full_results(
    *,
    data: PrimaryData,
    output_dir: Path,
    resampling: dict[str, np.ndarray],
    execution_lock_sha256: str,
    joint_latency: dict[str, Any],
) -> dict[str, Any]:
    seed_outputs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    ensemble: dict[str, np.ndarray] = {}
    for arm in ARMS:
        seed_outputs[arm] = {
            seed: aggregate_arm_seed(arm=arm, seed=seed, data=data, output_dir=output_dir)
            for seed in EXPECTED_SEEDS
        }
        ensemble[arm] = np.stack(
            [seed_outputs[arm][seed]["probabilities"] for seed in EXPECTED_SEEDS]
        ).mean(axis=0, dtype=np.float64)
    teacher = data.retained_teacher_probabilities.mean(axis=1, dtype=np.float64)
    ensemble_arrays = {
        "sample_ids": data.sample_ids,
        "recording_ids": data.recording_ids,
        "bundle_row_indices": data.bundle_row_indices,
        "labels": data.labels,
        "retained_teacher_probabilities": teacher,
        **{f"{arm}_probabilities": probabilities for arm, probabilities in ensemble.items()},
    }
    ensemble_path = output_dir / "temporal_control_ensemble_predictions.npz"
    _write_or_validate_npz(ensemble_path, ensemble_arrays)

    comparisons = {
        "t1_pooling_minus_retained_teacher": comparison_statistics(
            labels=data.labels,
            groups=data.recording_ids,
            occluded=data.occluded,
            transition=data.transition,
            reference=teacher,
            candidate=ensemble["t1_arithmetic_mean"],
            resampling=resampling,
        ),
        "t2_distinct_minus_matched_repeated": comparison_statistics(
            labels=data.labels,
            groups=data.recording_ids,
            occluded=data.occluded,
            transition=data.transition,
            reference=ensemble["t2_legacy_repeated"],
            candidate=ensemble["t2_fixed_distinct"],
            resampling=resampling,
        ),
        "matched_repeated_minus_retained_teacher_diagnostic": comparison_statistics(
            labels=data.labels,
            groups=data.recording_ids,
            occluded=data.occluded,
            transition=data.transition,
            reference=teacher,
            candidate=ensemble["t2_legacy_repeated"],
            resampling=resampling,
        ),
    }
    t1_latency = _load_designated_latency(output_dir, "t1_arithmetic_mean")
    repeated_latency = _load_designated_latency(output_dir, "t2_legacy_repeated")
    if t1_latency["sample_id_sha256"] != repeated_latency["sample_id_sha256"]:
        raise RuntimeError("T1 and matched teacher latency cohorts differ")
    if joint_latency.get("status") != JOINT_LATENCY_STATUS:
        raise RuntimeError("Authoritative joint latency benchmark is incomplete")
    joint_checks = joint_latency.get("authoritative_latency_checks", {})
    if set(joint_checks) != {
        "candidate_observed_p95_below_reference",
        "paired_bootstrap_p95_delta_upper_below_zero",
    }:
        raise RuntimeError("Authoritative joint latency checks changed")
    joint_latency_path = output_dir / "joint_latency_benchmark.json"
    if not joint_latency_path.is_file():
        raise RuntimeError("Authoritative joint latency artifact is absent")
    t1_stats = comparisons["t1_pooling_minus_retained_teacher"]
    t1_bootstrap = t1_stats["bootstrap"]["aggregate"]
    t1_checks = {
        "macro_f1_one_sided_lower_above_minus_0_005": float(
            t1_bootstrap["macro_f1_delta_one_sided_95pct_lower"]
        )
        > -0.005,
        "nll_one_sided_upper_at_most_zero": float(
            t1_bootstrap["nll_delta_one_sided_95pct_upper"]
        )
        <= 0.0,
        "joint_same_process_candidate_observed_p95_below_reference": bool(
            joint_checks["candidate_observed_p95_below_reference"]
        ),
        "joint_paired_bootstrap_p95_delta_upper_below_zero": bool(
            joint_checks["paired_bootstrap_p95_delta_upper_below_zero"]
        ),
        "promotion_guardrails_pass": bool(t1_stats["promotion_guardrails"]["passed"]),
    }
    t2_stats = comparisons["t2_distinct_minus_matched_repeated"]
    t2_bootstrap = t2_stats["bootstrap"]["aggregate"]
    t2_checks = {
        "macro_f1_two_sided_lower_above_zero": float(
            t2_bootstrap["macro_f1_delta_two_sided_95pct"][0]
        )
        > 0.0,
        "nll_one_sided_upper_at_most_zero": float(
            t2_bootstrap["nll_delta_one_sided_95pct_upper"]
        )
        <= 0.0,
        "one_sided_exact_scenario_swap_p_at_most_0_05": float(
            t2_stats["exact_directional_scenario_swap"]["one_sided_pvalue"]
        )
        <= 0.05,
        "promotion_guardrails_pass": bool(t2_stats["promotion_guardrails"]["passed"]),
    }
    summary = {
        "status": FULL_STATUS,
        "rows": EXPECTED_PRIMARY_ROWS,
        "scenarios": EXPECTED_GROUPS,
        "arms": list(ARMS),
        "folds": list(EXPECTED_FOLDS),
        "seeds": list(EXPECTED_SEEDS),
        "probability_ensemble": "arithmetic_mean_of_five_seed_probability_vectors",
        "per_arm_metrics": {
            "retained_teacher": classification_summary(data.labels, teacher),
            **{
                arm: classification_summary(data.labels, probabilities)
                for arm, probabilities in ensemble.items()
            },
        },
        "comparisons": comparisons,
        "decisions": {
            "T1_noninferiority_and_efficiency": {
                "accuracy_reference": "exact_retained_five_seed_teacher_probability_ensemble",
                "latency_reference": "t2_legacy_repeated_fold-0_seed-42",
                "latency_scope": "cached feature temporal head; shared upstream extraction excluded",
                "authoritative_joint_latency": {
                    "artifact": joint_latency_path.name,
                    "sha256": sha256_file(joint_latency_path),
                    "request_sha256": joint_latency["request_sha256"],
                    "statistics": joint_latency["statistics"],
                    "checks": joint_checks,
                },
                "isolated_per_workload_latency_diagnostics": {
                    "candidate": t1_latency,
                    "reference": repeated_latency,
                    "role": "diagnostic_only_not_used_by_gate",
                },
                "checks": t1_checks,
                "cached_head_gate_passed": all(t1_checks.values()),
                "end_to_end_latency_status": "PENDING_DECODE_CROP_FEATURE_AND_HEAD_MEASUREMENT",
                "full_preference_eligible": False,
            },
            "T2_sampler_benefit": {
                "comparison": "t2_fixed_distinct_minus_newly_matched_t2_legacy_repeated",
                "checks": t2_checks,
                "passed": all(t2_checks.values()),
                "attribution": "sampling_only_never_gate_repair",
            },
        },
        "artifact_sha256": {
            ensemble_path.name: sha256_file(ensemble_path),
            joint_latency_path.name: sha256_file(joint_latency_path),
        },
        "source_sha256": {
            "execution_lock": execution_lock_sha256,
        },
        "protected_access": {
            "mixed_role_files_opened": 0,
            "calibration_rows_or_arrays_read": 0,
            "confirmation_rows_or_arrays_read": 0,
            "test_rows_or_arrays_read": 0,
        },
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def validate_representative_benchmark(
    output_dir: Path, source_sha256: dict[str, str]
) -> dict[str, Any]:
    path = output_dir / "benchmark_summary.json"
    if not path.is_file():
        raise RuntimeError("Run --mode benchmark before the full temporal-control schedule")
    summary = load_json(path)
    if (
        summary.get("status") != BENCHMARK_STATUS
        or summary.get("partial_family_workloads_complete") != 2
        or summary.get("source_sha256") != source_sha256
    ):
        raise RuntimeError("Representative temporal-control benchmark receipt changed")
    resources = summary.get("representative_resource_measurement", {})
    expected_arms = ("t1_arithmetic_mean", "t2_legacy_repeated")
    for arm in expected_arms:
        workload_summary = workload_paths(output_dir, arm, "fold-0", 42)["summary"]
        if summary.get("workload_summary_sha256", {}).get(arm) != sha256_file(
            workload_summary
        ):
            raise RuntimeError(f"Representative benchmark workload receipt changed: {arm}")
        resource = resources.get(arm, {})
        for name in (
            "runtime_seconds",
            "peak_cuda_memory_allocated_bytes",
            "peak_cuda_memory_reserved_bytes",
        ):
            if float(resource.get(name, 0)) <= 0:
                raise RuntimeError(f"Representative benchmark resource is missing: {arm}/{name}")
    joint_path = output_dir / "joint_latency_benchmark.json"
    if summary.get("joint_latency", {}).get("sha256") != sha256_file(joint_path):
        raise RuntimeError("Representative benchmark joint-latency artifact changed")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "prepare", "benchmark", "full"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/okutama_cptr_continuation_protocol.json"),
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path(
            ".runs/research_20260907/protocol_locks/okutama_materialization_lock.json"
        ),
    )
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0/eligible_feature_bundle.npz"),
    )
    parser.add_argument(
        "--bundle-summary",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0/bundle_summary.json"),
    )
    parser.add_argument(
        "--fold-map",
        type=Path,
        default=Path(".runs/research_20260907/okutama_temporal_controls_r0/fold_map.csv"),
    )
    parser.add_argument(
        "--scenario-resampling",
        type=Path,
        default=Path(
            ".runs/research_20260907/okutama_temporal_controls_r0/scenario_resampling.npz"
        ),
    )
    parser.add_argument(
        "--prepare-summary",
        type=Path,
        default=Path(".runs/research_20260907/okutama_temporal_controls_r0/prepare_summary.json"),
    )
    parser.add_argument(
        "--execution-lock",
        type=Path,
        default=Path(
            ".runs/research_20260907/protocol_locks/okutama_temporal_controls_execution_lock.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/research_20260907/okutama_temporal_controls"),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    controls = validate_temporal_contract(protocol)
    if args.workers != 0:
        raise ValueError("Deterministic temporal controls require workers=0")
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "OKUTAMA_TEMPORAL_CONTROLS_DRY_RUN_VALID",
                    "arms": list(ARMS),
                    "folds": list(EXPECTED_FOLDS),
                    "seeds": list(EXPECTED_SEEDS),
                    "workloads": len(ARMS) * len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS),
                    "model_fits": 0,
                    "feature_values_read": 0,
                    "protocol_sha256": sha256_file(protocol_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    source_lock_path = args.source_lock.resolve()
    bundle_path = args.feature_bundle.resolve()
    bundle_summary_path = args.bundle_summary.resolve()
    fold_map_path = args.fold_map.resolve()
    resampling_path = args.scenario_resampling.resolve()
    prepare_summary_path = args.prepare_summary.resolve()
    if args.mode == "prepare":
        summary = prepare_artifacts(
            protocol_path=protocol_path,
            source_lock_path=source_lock_path,
            bundle_path=bundle_path,
            bundle_summary_path=bundle_summary_path,
            fold_map_path=fold_map_path,
            resampling_path=resampling_path,
            prepare_summary_path=prepare_summary_path,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    execution_lock_path = args.execution_lock.resolve()
    execution_lock = validate_execution_lock(
        execution_lock_path=execution_lock_path,
        source_lock_path=source_lock_path,
        protocol_path=protocol_path,
        bundle_path=bundle_path,
        bundle_summary_path=bundle_summary_path,
        fold_map_path=fold_map_path,
        resampling_path=resampling_path,
        prepare_summary_path=prepare_summary_path,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Locked temporal-control fitting requires CUDA")
    if args.latency_warmup < 1 or args.latency_iterations < 20:
        raise ValueError("Latency settings must provide >=1 warmup and >=20 timed iterations")
    fold_frame = _load_fold_map(fold_map_path, protocol)
    resampling = validate_resampling(
        resampling_path,
        np.asarray(sorted(fold_frame["recording_id"].unique()), dtype=str),
    )
    data = load_primary_data(
        bundle_path=bundle_path,
        bundle_summary_path=bundle_summary_path,
        fold_map_path=fold_map_path,
        protocol=protocol,
    )
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = {
        **execution_lock["validated_source_sha256"],
        "execution_lock": sha256_file(execution_lock_path),
    }
    training = controls["training"]
    if args.mode == "benchmark":
        started = time.perf_counter()
        benchmark_summaries = {}
        for arm in ("t1_arithmetic_mean", "t2_legacy_repeated"):
            benchmark_summaries[arm] = run_workload(
                arm=arm,
                fold="fold-0",
                seed=42,
                data=data,
                training=training,
                device=device,
                output_dir=output_dir,
                source_sha256=source_sha256,
                latency_warmup=args.latency_warmup,
                latency_iterations=args.latency_iterations,
            )
        joint_latency = run_joint_latency_benchmark(
            data=data,
            training=training,
            design=controls["joint_latency_benchmark"],
            output_dir=output_dir,
            source_sha256=source_sha256,
            latency_warmup=args.latency_warmup,
            latency_iterations=args.latency_iterations,
            device=device,
        )
        benchmark = {
            "status": BENCHMARK_STATUS,
            "workloads": [
                {"arm": arm, "fold": "fold-0", "seed": 42}
                for arm in ("t1_arithmetic_mean", "t2_legacy_repeated")
            ],
            "partial_family_workloads_complete": 2,
            "full_design_workloads": len(ARMS) * len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS),
            "measured_workload_runtime_seconds": {
                arm: float(summary["runtime_seconds"])
                for arm, summary in benchmark_summaries.items()
            },
            "representative_resource_measurement": {
                arm: {
                    "runtime_seconds": float(summary["runtime_seconds"]),
                    **summary["resource_measurement"],
                }
                for arm, summary in benchmark_summaries.items()
            },
            "joint_latency": {
                "artifact": "joint_latency_benchmark.json",
                "sha256": sha256_file(output_dir / "joint_latency_benchmark.json"),
                "authoritative_latency_passed": joint_latency[
                    "authoritative_latency_passed"
                ],
                "statistics": joint_latency["statistics"],
            },
            "wall_seconds_this_invocation": float(time.perf_counter() - started),
            "resume_instruction": "rerun with --mode full and identical arguments",
            "workload_summary_sha256": {
                arm: sha256_file(workload_paths(output_dir, arm, "fold-0", 42)["summary"])
                for arm in ("t1_arithmetic_mean", "t2_legacy_repeated")
            },
            "source_sha256": source_sha256,
        }
        write_json_atomic(output_dir / "benchmark_summary.json", benchmark)
        print(json.dumps(benchmark, indent=2, sort_keys=True))
        return

    validate_representative_benchmark(output_dir, source_sha256)
    started = time.perf_counter()
    reused = 0
    completed = 0
    for arm in ARMS:
        for fold in EXPECTED_FOLDS:
            for seed in EXPECTED_SEEDS:
                summary = run_workload(
                    arm=arm,
                    fold=fold,
                    seed=seed,
                    data=data,
                    training=training,
                    device=device,
                    output_dir=output_dir,
                    source_sha256=source_sha256,
                    latency_warmup=args.latency_warmup,
                    latency_iterations=args.latency_iterations,
                )
                completed += 1
                reused += int(summary.get("resume_action") is not None)
                print(
                    json.dumps(
                        {
                            "progress": f"{completed}/{len(ARMS) * len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS)}",
                            "arm": arm,
                            "fold": fold,
                            "seed": seed,
                            "resume": summary.get("resume_action"),
                        }
                    ),
                    flush=True,
                )
    joint_latency = run_joint_latency_benchmark(
        data=data,
        training=training,
        design=controls["joint_latency_benchmark"],
        output_dir=output_dir,
        source_sha256=source_sha256,
        latency_warmup=args.latency_warmup,
        latency_iterations=args.latency_iterations,
        device=device,
    )
    final = aggregate_full_results(
        data=data,
        output_dir=output_dir,
        resampling=resampling,
        execution_lock_sha256=sha256_file(execution_lock_path),
        joint_latency=joint_latency,
    )
    final["invocation"] = {
        "wall_seconds": float(time.perf_counter() - started),
        "verified_workloads_reused": reused,
        "workloads_completed": completed,
    }
    write_json_atomic(output_dir / "summary.json", final)
    print(json.dumps(final["decisions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
