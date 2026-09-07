"""Run the locked V-COCO V0 matched-control study.

The script has two deliberately separate modes. ``prepare-fold-map`` is authorized by
the source lock and creates only the deterministic, development-only grouped fold map.
``run`` additionally requires an execution lock that binds that map and this committed
runner before any estimator is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

# This must be set before the first CUDA context/query in the fresh runner process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hac.metrics import classification_metrics
from hac.polar_analysis import per_class_metrics
from hac.vcoco_v3_models import (
    CLASS_NAMES,
    Candidate,
    CudaPrimalLinearSVM,
    candidate_rank_key,
    cuda_svm_fit_audit,
    fit_probability_head,
    geometry_features,
    grouped_splits,
    holm_adjust,
    locomotion_f1,
    normalize_probability_rows,
    predict_probability_head,
    reset_cuda_svm_fit_audit,
    restore_cuda_svm_fit_audit,
    stack_features,
)

PROTOCOL_PATH = Path("experiments/vcoco_continuation_protocol.json")
SOURCE_LOCK_PATH = Path(".runs/research_20260907/protocol_locks/vcoco_source_lock.json")
EXECUTION_LOCK_PATH = Path(
    ".runs/research_20260907/protocol_locks/vcoco_execution_lock.json"
)
FOLD_MAP_PATH = Path(".runs/research_20260907/vcoco_v0_r0/fold_map.csv")
BOOTSTRAP_INDICES_PATH = Path(
    ".runs/research_20260907/vcoco_v0_r0/bootstrap_group_indices.npy"
)
SWAP_SIGNS_PATH = Path(".runs/research_20260907/vcoco_v0_r0/swap_signs_packbits.npy")
RANDOMIZATION_RECEIPT_PATH = Path(
    ".runs/research_20260907/vcoco_v0_r0/statistics_randomization_receipt.json"
)
OUTPUT_DIR = Path(".runs/research_20260907/vcoco_v0")
RUNNER_PATH = Path("experiments/run_vcoco_continuation_v0.py")
MODEL_MODULE_PATH = Path("src/hac/vcoco_v3_models.py")

SOURCE_LOCK_STATUS = "VCOCO_CONTINUATION_SOURCE_LOCKED_BEFORE_V0_FITTING"
EXECUTION_LOCK_STATUS = "VCOCO_CONTINUATION_EXECUTION_LOCKED_BEFORE_FITTING"
PROTOCOL_STATUS = "DECLARED_BEFORE_VCOCO_CONTINUATION_FITTING"
FAMILY_ORDER = (
    "mixed_flat",
    "mixed_factorized",
    "mixed_factorized_reliability",
    "mixed_linear_svm",
)
FEATURE_ORDER = ("dino_tight", "dino_context", "siglip_tight", "siglip_context")
PROTECTED_COUNTERS = (
    "mixed_manifest_rows_read",
    "mixed_development_metadata_rows_read",
    "calibration_rows_or_arrays_read",
    "confirmation_rows_or_arrays_read",
    "test_rows_or_arrays_read",
)
FORBIDDEN_INPUTS = {
    ".runs/polar_v2/locked_protocol/vcoco_test_clean.csv",
    ".runs/vcoco_v3/temporal/development_manifest.csv",
    ".runs/vcoco_v3/okutama/features/dinov2_base/development_metadata.csv",
}
PROTECTED_PATH_COMPONENTS = {"calibration", "confirmation", "test"}
ESTIMATOR_BASE_SEED = 20260907


def configure_deterministic_execution(protocol: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if protocol.get("execution_determinism") != expected:
        raise RuntimeError("V0 deterministic execution declaration changed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected["cublas_workspace_config"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the V0 protocol")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    observed = {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if observed != expected:
        raise RuntimeError("Could not establish deterministic V0 execution")
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare-fold-map", "run"), required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK_PATH)
    parser.add_argument("--execution-lock", type=Path, default=EXECUTION_LOCK_PATH)
    parser.add_argument("--fold-map", type=Path, default=FOLD_MAP_PATH)
    parser.add_argument(
        "--bootstrap-group-indices", type=Path, default=BOOTSTRAP_INDICES_PATH
    )
    parser.add_argument("--swap-signs-packbits", type=Path, default=SWAP_SIGNS_PATH)
    parser.add_argument(
        "--statistics-randomization-receipt",
        type=Path,
        default=RANDOMIZATION_RECEIPT_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-new-family-folds",
        type=int,
        help="Stop cleanly after this many newly checkpointed family/outer-fold workloads.",
    )
    parser.add_argument(
        "--benchmark-family",
        choices=FAMILY_ORDER,
        help="With a bounded run, benchmark this family first without changing its fold or seed.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while it was hashed: {path}")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def relative_safe_path(root: Path, path: Path) -> str:
    """Return a repository-relative path after rejecting protected role paths."""

    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Input is outside the repository: {path}") from error
    normalized = str(PurePosixPath(relative)).lower()
    if normalized in FORBIDDEN_INPUTS:
        raise RuntimeError(f"Forbidden role-mixed or protected input: {relative}")
    components = {part.lower() for part in PurePosixPath(normalized).parts}
    if components.intersection(PROTECTED_PATH_COMPONENTS):
        raise RuntimeError(f"Protected-role input path: {relative}")
    return relative


def require_zero_protected_access(payload: dict[str, Any], *, name: str) -> None:
    counters = payload.get("protected_access")
    if not isinstance(counters, dict) or set(counters) != set(PROTECTED_COUNTERS):
        raise RuntimeError(f"{name} has an incomplete protected-access receipt")
    if any(type(counters[key]) is not int or counters[key] != 0 for key in PROTECTED_COUNTERS):
        raise RuntimeError(f"{name} reports protected or mixed-role access")


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
    cuda_available = bool(torch.cuda.is_available())
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": distributions,
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device": torch.cuda.get_device_name(0) if cuda_available else None,
            "compute_capability": (
                ".".join(map(str, torch.cuda.get_device_capability(0)))
                if cuda_available
                else None
            ),
        },
    }
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    environment["canonical_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return environment


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _positive_unique(values: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"{name} must be a non-empty list")
    result = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise RuntimeError(f"{name} must contain finite positive values")
    if len(set(result)) != len(result):
        raise RuntimeError(f"{name} contains duplicate settings")
    return result


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("protocol_version") != "1.0.0":
        raise RuntimeError("Unexpected V-COCO continuation protocol status or version")
    if protocol.get("execution_determinism") != {
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }:
        raise RuntimeError("V0 deterministic execution contract changed")
    data = protocol.get("data_contract", {})
    if (data.get("combined_development_people"), data.get("combined_development_images")) != (
        6640,
        4123,
    ):
        raise RuntimeError("V-COCO V0 development scope changed")
    if data.get("group_key") != "source_image_id":
        raise RuntimeError("V-COCO V0 group key changed")
    if set(data.get("protected_roles", [])) != {
        "historical_official_test",
        "calibration",
        "confirmation",
        "future_independent_replication",
    }:
        raise RuntimeError("V-COCO V0 protected roles changed")

    cross_validation = protocol.get("cross_validation", {})
    expected_cv = {
        "outer_folds": 5,
        "inner_folds": 3,
        "stack_folds": 3,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "shuffle": True,
        "random_seed": 20260827,
        "group": "source_image_id",
        "shared_fold_map_required": True,
    }
    for key, expected in expected_cv.items():
        if cross_validation.get(key) != expected:
            raise RuntimeError(f"V-COCO V0 cross-validation contract changed: {key}")
    if (
        cross_validation.get("inner_seed_by_outer_fold")
        != "20260827 + 10000 * (outer_fold + 1)"
        or cross_validation.get("stack_seed_by_outer_fold")
        != "20260827 + 100000 + outer_fold"
        or cross_validation.get("selection_stack_seed_by_outer_and_inner_fold")
        != "20260827 + 200000 + 1000 * outer_fold + inner_fold"
    ):
        raise RuntimeError("V-COCO V0 fold seed formulas changed")
    fold_declaration = cross_validation.get("fold_map", {})
    if (
        fold_declaration.get("path") != FOLD_MAP_PATH.as_posix()
        or fold_declaration.get("columns") != expected_fold_columns(5)
        or fold_declaration.get("outer_held_sentinel_for_inner_and_stack") != -1
        or fold_declaration.get("selection_stack_eligibility")
        != "outer_fold != k and inner_fold_ok != j"
        or fold_declaration.get("selection_stack_sentinel_outside_eligibility") != -1
    ):
        raise RuntimeError("V-COCO V0 fold-map declaration changed")

    v0 = protocol.get("V0", {})
    if (
        v0.get("runs")
        != "one deterministic nested run using the locked shared outer, inner, and stack folds"
        or v0.get("model_seed") != ESTIMATOR_BASE_SEED
        or v0.get("inner_candidate_seed")
        != "20260907 + 10000 * outer_fold + candidate_index"
        or v0.get("inner_candidate_seed_role")
        != "base_seed_before_the_fixed_inner_fold_offset"
        or v0.get("inner_fit_estimator_seed")
        != (
            "20260907 + 10000 * outer_fold + candidate_index + "
            "10000 * (inner_fold + 1)"
        )
        or v0.get("outer_final_fit_seed")
        != "20260907 + 100000 + 1000 * outer_fold + family_index"
    ):
        raise RuntimeError("V0 deterministic estimator seed contract changed")
    if int(v0.get("candidate_selection_budget_per_family", -1)) != 8:
        raise RuntimeError("V0 candidate-selection budget changed")
    if v0.get("candidate_selection") != {
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
        raise RuntimeError("V0 candidate-selection ranking contract changed")
    shared_geometry = v0.get("shared_geometry_features", {})
    if shared_geometry != {
        "source": "shared_pooled_feature_rows_csv",
        "source_rows_sha256": (
            "fc0772cdf1f1bce3786cfdf3c9571bcf456cfaa00cb12ea0d26212a7ff0504cb"
        ),
        "dimensions": 6,
        "names": [
            "log_bbox_area_fraction",
            "log_bbox_aspect_ratio",
            "bbox_center_x_fraction",
            "bbox_center_y_fraction",
            "log_person_pixel_height",
            "bbox_edge_distance_fraction",
        ],
        "preprocessing": "hac.vcoco_v3_models.geometry_features",
        "identical_across_all_families": True,
    }:
        raise RuntimeError("V0 shared geometry-feature contract changed")
    families = v0.get("families", {})
    if tuple(families) != FAMILY_ORDER:
        raise RuntimeError("V0 family names or order changed")
    expected_flags = {
        "mixed_flat": (False, False, "multinomial_logistic_probability_stack"),
        "mixed_factorized": (True, False, "posture_motion_logistic_probability_stack"),
        "mixed_factorized_reliability": (
            True,
            True,
            "posture_motion_logistic_probability_stack",
        ),
        "mixed_linear_svm": (
            False,
            False,
            "cuda_primal_one_vs_rest_squared_hinge_with_grouped_oof_calibration",
        ),
    }
    for family, (factorized, reliability, head) in expected_flags.items():
        declaration = families[family]
        if tuple(declaration.get("features", ())) != FEATURE_ORDER:
            raise RuntimeError(f"V0 feature set changed: {family}")
        if (
            declaration.get("factorized") is not factorized
            or declaration.get("reliability") is not reliability
            or declaration.get("head") != head
        ):
            raise RuntimeError(f"V0 head contract changed: {family}")

    probability_grid = v0.get("probability_stack_grid", {})
    component_c = _positive_unique(probability_grid.get("component_C"), name="component_C")
    meta_c = _positive_unique(probability_grid.get("meta_C"), name="meta_C")
    weights = tuple(probability_grid.get("class_weight", ()))
    if weights != ("none", "balanced") or len(component_c) * len(meta_c) * len(weights) != 8:
        raise RuntimeError("V0 probability-stack grid is not the declared eight settings")
    svm_grid = v0.get("linear_svm_grid", {})
    svm_c = _positive_unique(svm_grid.get("C"), name="linear_svm.C")
    svm_weights = tuple(svm_grid.get("class_weight", ()))
    if svm_weights != ("none", "balanced") or len(svm_c) * len(svm_weights) != 8:
        raise RuntimeError("V0 SVM grid is not the declared eight settings")
    if (
        float(svm_grid.get("calibrator_C", -1.0)) != 1.0
        or int(svm_grid.get("maximum_iterations", -1)) != 2_000
        or float(svm_grid.get("tolerance", -1.0)) != 1e-4
        or svm_grid.get("require_iteration_limit_not_reached") is not True
    ):
        raise RuntimeError("V0 calibrated SVM optimization contract changed")

    inputs = protocol.get("pooled_feature_inputs", [])
    if tuple(item.get("name") for item in inputs) != FEATURE_ORDER:
        raise RuntimeError("V0 pooled-feature declarations changed")
    if len({item.get("rows_sha256") for item in inputs}) != 1:
        raise RuntimeError("V0 pooled features do not share one row identity")
    statistics = protocol.get("statistics", {})
    expected_statistics = {
        "bootstrap_unit": "whole_source_image",
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20260906,
        "swap_test": "whole_image_probability_vector_swaps",
        "swap_monte_carlo_draws": 100_000,
        "swap_plus_one_correction": True,
        "zero_division": 0,
        "class_count": 3,
    }
    for key, expected in expected_statistics.items():
        if statistics.get(key) != expected:
            raise RuntimeError(f"V0 statistical contract changed: {key}")
    if (
        statistics.get("random_generator")
        != "numpy.random.PCG64_via_SeedSequence_spawn_2"
        or statistics.get("bootstrap_indices_artifact")
        != {
            "path": BOOTSTRAP_INDICES_PATH.as_posix(),
            "dtype": "uint16",
            "shape": [10_000, 4_123],
        }
        or statistics.get("swap_signs_artifact")
        != {
            "path": SWAP_SIGNS_PATH.as_posix(),
            "dtype": "uint8",
            "shape": [100_000, 516],
            "decode": "np.unpackbits(values, axis=1, bitorder='little')[:, :4123]",
        }
        or statistics.get("randomization_receipt")
        != RANDOMIZATION_RECEIPT_PATH.as_posix()
        or statistics.get("pre_fit_lock_required") is not True
        or statistics.get("shared_across_all_declared_comparisons") is not True
    ):
        raise RuntimeError("V0 shared statistical randomization contract changed")
    if protocol.get("resource_contract") != {
        "same_cohort": True,
        "V0_scope": "locked_cached_development_features_through_nested_heads",
        "V0_measures": [
            "bounded_outer_fold_wall_time",
            "complete_nested_run_wall_time",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "cuda_svm_iteration_audit",
        ],
        "end_to_end_pipeline_resource_claim_authorized": False,
        "deferred_until_V1_inputs_exist": [
            "decode",
            "pose_extraction",
            "spatial_feature_extraction",
            "cold_start",
            "synchronized_warm_latency",
            "throughput",
            "host_memory",
            "energy",
        ],
    }:
        raise RuntimeError("V0 resource-scope contract changed")


def _format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def enumerate_v0_candidates(protocol: dict[str, Any], family: str) -> list[Candidate]:
    validate_protocol(protocol)
    if family not in FAMILY_ORDER:
        raise ValueError(f"Unknown V0 family: {family}")
    v0 = protocol["V0"]
    candidates: list[Candidate] = []
    if family == "mixed_linear_svm":
        grid = v0["linear_svm_grid"]
        for c_value in grid["C"]:
            for class_weight in grid["class_weight"]:
                candidates.append(
                    Candidate(
                        candidate_id=(
                            f"{family}__c-{_format_float(float(c_value))}__cw-{class_weight}"
                        ),
                        family=family,
                        component_c=None,
                        meta_c=None,
                        svm_c=float(c_value),
                        class_weight=str(class_weight),
                    )
                )
    else:
        grid = v0["probability_stack_grid"]
        for component_c in grid["component_C"]:
            for meta_c in grid["meta_C"]:
                for class_weight in grid["class_weight"]:
                    candidates.append(
                        Candidate(
                            candidate_id=(
                                f"{family}__cc-{_format_float(float(component_c))}"
                                f"__mc-{_format_float(float(meta_c))}__cw-{class_weight}"
                            ),
                            family=family,
                            component_c=float(component_c),
                            meta_c=float(meta_c),
                            svm_c=None,
                            class_weight=str(class_weight),
                        )
                    )
    budget = int(v0["candidate_selection_budget_per_family"])
    if len(candidates) != budget or len({item.candidate_id for item in candidates}) != budget:
        raise RuntimeError(f"V0 candidate budget mismatch: {family}")
    return candidates


def label_indices(rows: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = rows["label_3"].map(mapping)
    if labels.isna().any():
        raise RuntimeError("An unknown source tag entered V0 development")
    return labels.to_numpy(dtype=np.int64)


def activity_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return repository metrics with the V0 zero-division rule made explicit."""

    metrics = classification_metrics(labels, probabilities)
    predictions = np.asarray(probabilities).argmax(axis=1)
    metrics["macro_f1"] = float(
        f1_score(labels, predictions, average="macro", zero_division=0)
    )
    metrics["weighted_f1"] = float(
        f1_score(labels, predictions, average="weighted", zero_division=0)
    )
    return metrics


def validate_development_rows(rows: pd.DataFrame, protocol: dict[str, Any]) -> None:
    required = {
        "person_id",
        "image_id",
        "split",
        "label_3",
        "bbox_area_fraction",
        "bbox_aspect_ratio",
        "bbox_center_x_fraction",
        "bbox_center_y_fraction",
        "person_pixel_height",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise RuntimeError(f"V0 row sidecar is missing columns: {', '.join(missing)}")
    data = protocol["data_contract"]
    if len(rows) != int(data["combined_development_people"]):
        raise RuntimeError("V0 row sidecar person count changed")
    if rows["person_id"].astype(str).duplicated().any():
        raise RuntimeError("V0 person identifiers are not unique")
    if rows["image_id"].astype(str).nunique() != int(data["combined_development_images"]):
        raise RuntimeError("V0 source-image count changed")
    expected_splits = {
        name: int(declaration["people"])
        for name, declaration in data["development_sources"].items()
    }
    observed_splits = rows["split"].astype(str).value_counts().to_dict()
    aliases = {"former_train": "train", "former_val": "val"}
    normalized_expected = {aliases[name]: count for name, count in expected_splits.items()}
    if observed_splits != normalized_expected:
        raise RuntimeError("V0 row sidecar contains an unauthorized split or changed count")
    if set(rows["label_3"].astype(str)) != set(CLASS_NAMES):
        raise RuntimeError("V0 row sidecar activity classes changed")
    values = geometry_features(rows)
    if values.shape != (len(rows), int(protocol["V0"]["shared_geometry_features"]["dimensions"])):
        raise RuntimeError("V0 geometry dimensions differ from the declared shared block")
    if not np.isfinite(values).all():
        raise RuntimeError("V0 geometry contains non-finite values")


def assignments_from_splits(
    rows: int, splits: list[tuple[np.ndarray, np.ndarray]], *, folds: int
) -> np.ndarray:
    assignment = np.full(rows, -1, dtype=np.int64)
    for fold, (_, held) in enumerate(splits):
        if fold >= folds or (assignment[held] != -1).any():
            raise RuntimeError("Invalid or overlapping grouped split")
        assignment[held] = fold
    if (assignment < 0).any() or set(np.unique(assignment)) != set(range(folds)):
        raise RuntimeError("Grouped split does not assign every row exactly once")
    return assignment


def validate_group_assignment(groups: np.ndarray, assignment: np.ndarray, *, name: str) -> None:
    frame = pd.DataFrame({"group": np.asarray(groups).astype(str), "fold": assignment})
    if (frame.groupby("group", sort=False)["fold"].nunique() != 1).any():
        raise RuntimeError(f"A source image crosses {name} fold boundaries")


def build_fold_map(rows: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    """Build the one canonical outer/inner/final-stack fold map without fitting."""

    validate_protocol(protocol)
    validate_development_rows(rows, protocol)
    labels = label_indices(rows)
    groups = rows["image_id"].astype(str).to_numpy(dtype=str)
    cv = protocol["cross_validation"]
    outer_folds = int(cv["outer_folds"])
    inner_folds = int(cv["inner_folds"])
    stack_folds = int(cv["stack_folds"])
    seed = int(cv["random_seed"])
    outer = assignments_from_splits(
        len(rows), grouped_splits(labels, groups, folds=outer_folds, seed=seed), folds=outer_folds
    )
    validate_group_assignment(groups, outer, name="outer")
    result = pd.DataFrame(
        {
            "row_index": np.arange(len(rows), dtype=np.int64),
            "person_id": rows["person_id"].astype(str).to_numpy(dtype=str),
            "image_id": groups,
            "label_index": labels,
            "outer_fold": outer,
        }
    )
    outer_train_indices: dict[int, np.ndarray] = {}
    inner_assignments: dict[int, np.ndarray] = {}
    for outer_fold in range(outer_folds):
        train = np.flatnonzero(outer != outer_fold)
        inner = assignments_from_splits(
            len(train),
            grouped_splits(
                labels[train],
                groups[train],
                folds=inner_folds,
                seed=seed + 10_000 * (outer_fold + 1),
            ),
            folds=inner_folds,
        )
        stack = assignments_from_splits(
            len(train),
            grouped_splits(
                labels[train],
                groups[train],
                folds=stack_folds,
                seed=seed + 100_000 + outer_fold,
            ),
            folds=stack_folds,
        )
        inner_full = np.full(len(rows), -1, dtype=np.int64)
        stack_full = np.full(len(rows), -1, dtype=np.int64)
        inner_full[train] = inner
        stack_full[train] = stack
        validate_group_assignment(groups[train], inner, name=f"inner-o{outer_fold}")
        validate_group_assignment(groups[train], stack, name=f"stack-o{outer_fold}")
        result[f"inner_fold_o{outer_fold}"] = inner_full
        result[f"stack_fold_o{outer_fold}"] = stack_full
        outer_train_indices[outer_fold] = train
        inner_assignments[outer_fold] = inner
    for outer_fold in range(outer_folds):
        outer_train = outer_train_indices[outer_fold]
        inner = inner_assignments[outer_fold]
        for inner_fold in range(inner_folds):
            selection_fit = outer_train[inner != inner_fold]
            selection_stack = assignments_from_splits(
                len(selection_fit),
                grouped_splits(
                    labels[selection_fit],
                    groups[selection_fit],
                    folds=stack_folds,
                    seed=seed + 200_000 + 1_000 * outer_fold + inner_fold,
                ),
                folds=stack_folds,
            )
            selection_full = np.full(len(rows), -1, dtype=np.int64)
            selection_full[selection_fit] = selection_stack
            validate_group_assignment(
                groups[selection_fit],
                selection_stack,
                name=f"selection-stack-o{outer_fold}-i{inner_fold}",
            )
            result[f"selection_stack_fold_o{outer_fold}_i{inner_fold}"] = selection_full
    return result


def expected_fold_columns(outer_folds: int) -> list[str]:
    columns = ["row_index", "person_id", "image_id", "label_index", "outer_fold"]
    for outer_fold in range(outer_folds):
        columns.extend([f"inner_fold_o{outer_fold}", f"stack_fold_o{outer_fold}"])
    for outer_fold in range(outer_folds):
        for inner_fold in range(3):
            columns.append(f"selection_stack_fold_o{outer_fold}_i{inner_fold}")
    return columns


def validate_fold_map(
    fold_map: pd.DataFrame, rows: pd.DataFrame, protocol: dict[str, Any]
) -> None:
    expected = build_fold_map(rows, protocol)
    columns = expected_fold_columns(int(protocol["cross_validation"]["outer_folds"]))
    if list(fold_map.columns) != columns:
        raise RuntimeError("V0 fold-map columns or order changed")
    normalized = fold_map.copy()
    normalized["person_id"] = normalized["person_id"].astype(str)
    normalized["image_id"] = normalized["image_id"].astype(str)
    for column in set(columns).difference({"person_id", "image_id"}):
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise RuntimeError(f"V0 fold-map column is not integral: {column}")
        normalized[column] = numeric.astype(np.int64)
    if not normalized.equals(expected):
        raise RuntimeError("V0 fold map does not match its deterministic protocol construction")


def _verify_receipt(
    root: Path,
    receipt: dict[str, Any],
    *,
    expected_path: str,
    expected_sha256: str,
) -> Path:
    if receipt.get("path") != expected_path or receipt.get("sha256") != expected_sha256:
        raise RuntimeError(f"Locked receipt changed: {expected_path}")
    path = (root / expected_path).resolve()
    relative_safe_path(root, path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(receipt.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"Locked byte size changed: {expected_path}")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Locked SHA-256 changed: {expected_path}")
    return path


def validate_source_lock(
    root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    source_lock_path: Path,
    *,
    verify_feature_arrays: bool,
) -> dict[str, Any]:
    relative_safe_path(root, source_lock_path)
    source_lock = read_json(source_lock_path)
    if (
        source_lock.get("status") != SOURCE_LOCK_STATUS
        or source_lock.get("dataset") != "vcoco"
        or source_lock.get("protocol_version") != "1.0.0"
    ):
        raise RuntimeError("Incompatible V-COCO continuation source lock")
    require_zero_protected_access(source_lock, name="V-COCO source lock")
    authorization = source_lock.get("authorization", {})
    if authorization != {
        "shared_fold_map_construction": True,
        "shared_statistics_randomization_construction": True,
        "V0_fitting": False,
        "V1_fitting": False,
    }:
        raise RuntimeError("V-COCO source-lock authorization changed")

    commit = str(source_lock.get("repository_commit", ""))
    if not commit or _git(root, "rev-parse", "HEAD") != commit:
        raise RuntimeError("Current HEAD differs from the V-COCO source-lock commit")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Working-tree changes invalidate V0 source execution")

    source_hashes = source_lock.get("source_sha256", {})
    source_blobs = source_lock.get("source_blobs", {})
    required_sources = {"protocol_spec", "nested_model_module", "v0_runner", "locker"}
    if not required_sources.issubset(source_hashes) or not required_sources.issubset(source_blobs):
        raise RuntimeError("V-COCO source lock does not bind every V0 execution source")
    expected_source_paths = {
        "protocol_spec": relative_safe_path(root, protocol_path),
        "nested_model_module": MODEL_MODULE_PATH.as_posix(),
        "v0_runner": RUNNER_PATH.as_posix(),
    }
    for name, receipt in source_blobs.items():
        if not isinstance(receipt, dict):
            raise RuntimeError(f"Malformed source-blob receipt: {name}")
        relative = str(receipt.get("path", ""))
        if name in expected_source_paths and relative != expected_source_paths[name]:
            raise RuntimeError(f"V0 source path changed: {name}")
        path = (root / relative).resolve()
        relative_safe_path(root, path)
        digest = str(receipt.get("sha256", ""))
        if source_hashes.get(name) != digest or sha256_file(path) != digest:
            raise RuntimeError(f"V0 committed source drift: {name}")
        if int(receipt.get("size_bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"V0 committed source size drift: {name}")
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        if blob != receipt.get("git_blob_oid"):
            raise RuntimeError(f"V0 committed source blob drift: {name}")
    if source_hashes["protocol_spec"] != sha256_file(protocol_path):
        raise RuntimeError("V0 protocol changed after source lock")
    if source_hashes["v0_runner"] != sha256_file(root / RUNNER_PATH):
        raise RuntimeError("V0 runner changed after source lock")

    receipt_declarations = {item["name"]: item for item in protocol["historical_input_receipts"]}
    locked_receipts = source_lock.get("inputs", {}).get("receipts", {})
    if set(locked_receipts) != set(receipt_declarations):
        raise RuntimeError("V0 historical input receipt set changed")
    expected_roles = {
        "v2_protocol_lock": "historical_split_role_receipt",
        "former_train_manifest": "reused_development",
        "former_val_manifest": "reused_development",
    }
    for name, declaration in receipt_declarations.items():
        locked = locked_receipts[name]
        if declaration.get("role") != expected_roles[name] or locked.get("role") != expected_roles[name]:
            raise RuntimeError(f"Unauthorized V0 historical input role: {name}")
        _verify_receipt(
            root,
            locked,
            expected_path=str(declaration["path"]),
            expected_sha256=str(declaration["sha256"]),
        )

    locked_features = source_lock.get("inputs", {}).get("pooled_features", [])
    by_name = {item.get("name"): item for item in locked_features if isinstance(item, dict)}
    if set(by_name) != set(FEATURE_ORDER) or len(locked_features) != len(FEATURE_ORDER):
        raise RuntimeError("V0 pooled-feature lock set changed")
    for declaration in protocol["pooled_feature_inputs"]:
        name = declaration["name"]
        locked = by_name[name]
        if locked.get("representation") != declaration["representation"]:
            raise RuntimeError(f"V0 representation receipt changed: {name}")
        directory = str(declaration["directory"])
        provenance_path = _verify_receipt(
            root,
            locked.get("provenance", {}),
            expected_path=f"{directory}/provenance.json",
            expected_sha256=str(declaration["receipt_sha256"]),
        )
        provenance = read_json(provenance_path)
        if (
            provenance.get("status") != "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE"
            or provenance.get("test_rows_read") != 0
            or provenance.get("test_predictions_run") is not False
            or int(provenance.get("rows", -1)) != 6640
        ):
            raise RuntimeError(f"V0 cache has an unsafe or incomplete receipt: {name}")
        artifacts = provenance.get("artifact_sha256", {})
        if (
            artifacts.get("features.npy") != declaration["features_sha256"]
            or artifacts.get("rows.csv") != declaration["rows_sha256"]
        ):
            raise RuntimeError(f"V0 cache provenance drift: {name}")
        _verify_receipt(
            root,
            locked.get("rows", {}),
            expected_path=f"{directory}/rows.csv",
            expected_sha256=str(declaration["rows_sha256"]),
        )
        if verify_feature_arrays:
            _verify_receipt(
                root,
                locked.get("features", {}),
                expected_path=f"{directory}/features.npy",
                expected_sha256=str(declaration["features_sha256"]),
            )
    return source_lock


def load_rows_and_features(
    root: Path, protocol: dict[str, Any], *, load_features: bool
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    reference = protocol["pooled_feature_inputs"][0]
    rows_path = root / reference["directory"] / "rows.csv"
    rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str, "split": str})
    validate_development_rows(rows, protocol)
    features: dict[str, np.ndarray] = {}
    if load_features:
        for declaration in protocol["pooled_feature_inputs"]:
            values = np.load(root / declaration["directory"] / "features.npy", mmap_mode="r")
            if values.ndim != 2 or values.shape[0] != len(rows):
                raise RuntimeError(f"V0 feature-array shape changed: {declaration['name']}")
            features[str(declaration["name"])] = values
    return rows, features


def validate_execution_lock(
    root: Path,
    source_lock_path: Path,
    source_lock: dict[str, Any],
    execution_lock_path: Path,
    fold_map_path: Path,
    bootstrap_path: Path,
    swap_path: Path,
    randomization_receipt_path: Path,
) -> dict[str, Any]:
    relative_safe_path(root, execution_lock_path)
    execution = read_json(execution_lock_path)
    if (
        execution.get("status") != EXECUTION_LOCK_STATUS
        or execution.get("dataset") != "vcoco"
        or execution.get("stage") != "v0"
        or execution.get("repository_commit") != source_lock.get("repository_commit")
    ):
        raise RuntimeError("Incompatible V-COCO V0 execution lock")
    require_zero_protected_access(execution, name="V-COCO V0 execution lock")
    if execution.get("authorization") != {"V0_fitting": True, "V1_fitting": False}:
        raise RuntimeError("V-COCO V0 fitting is not exclusively authorized")
    source_lock_digest = sha256_file(source_lock_path)
    if (
        execution.get("source_lock_sha256") != source_lock_digest
        or execution.get("source_sha256", {}).get("source_lock") != source_lock_digest
    ):
        raise RuntimeError("V0 execution lock does not bind the active source lock")
    if execution.get("environment") != collect_environment():
        raise RuntimeError("V0 execution environment differs from its execution lock")

    artifacts = execution.get("artifacts", {})
    if set(artifacts) != {
        "fold_map",
        "execution_runner",
        "bootstrap_group_indices",
        "swap_signs_packbits",
        "statistics_randomization_receipt",
    }:
        raise RuntimeError("V0 execution artifact set changed")
    expected_paths = {
        "fold_map": relative_safe_path(root, fold_map_path),
        "execution_runner": RUNNER_PATH.as_posix(),
        "bootstrap_group_indices": relative_safe_path(root, bootstrap_path),
        "swap_signs_packbits": relative_safe_path(root, swap_path),
        "statistics_randomization_receipt": relative_safe_path(
            root, randomization_receipt_path
        ),
    }
    for name, expected_path in expected_paths.items():
        receipt = artifacts[name]
        digest = str(receipt.get("sha256", ""))
        path = _verify_receipt(
            root, receipt, expected_path=expected_path, expected_sha256=digest
        )
        if execution.get("source_sha256", {}).get(name) != digest:
            raise RuntimeError(f"V0 execution source digest changed: {name}")
        if name == "execution_runner":
            if digest != source_lock["source_sha256"]["v0_runner"]:
                raise RuntimeError("V0 runner differs between source and execution locks")
            if path != (root / RUNNER_PATH).resolve():
                raise RuntimeError("V0 execution runner path changed")
    return execution


def write_fold_map(path: Path, fold_map: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fold_map.to_csv(temporary, index=False, lineterminator="\n")
    if path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise RuntimeError(f"An incompatible V0 fold map already exists: {path}")
        temporary.unlink()
    else:
        temporary.replace(path)


def canonical_group_ids(groups: np.ndarray) -> list[str]:
    return sorted(set(np.asarray(groups).astype(str).tolist()))


def group_ids_sha256(group_ids: list[str]) -> str:
    payload = ("\n".join(group_ids) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _open_npy_memmap(path: Path, *, shape: tuple[int, int], dtype: np.dtype) -> np.memmap:
    from numpy.lib.format import open_memmap

    path.parent.mkdir(parents=True, exist_ok=True)
    return open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def prepare_shared_randomization(
    groups: np.ndarray,
    protocol: dict[str, Any],
    bootstrap_path: Path,
    swap_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Create one deterministic randomization design before any V0 fitting."""

    paths = (bootstrap_path, swap_path, receipt_path)
    present = tuple(path.exists() for path in paths)
    if any(present):
        if not all(present):
            raise RuntimeError("The shared V0 randomization artifact set is incomplete")
        receipt, _, _ = load_shared_randomization(
            groups, protocol, bootstrap_path, swap_path, receipt_path
        )
        return receipt

    group_ids = canonical_group_ids(groups)
    statistics = protocol["statistics"]
    group_count = len(group_ids)
    if group_count != int(protocol["data_contract"]["combined_development_images"]):
        raise RuntimeError("V0 randomization group count changed")
    resamples = int(statistics["bootstrap_resamples"])
    draws = int(statistics["swap_monte_carlo_draws"])
    seed = int(statistics["bootstrap_seed"])
    if group_count > np.iinfo(np.uint16).max:
        raise RuntimeError("V0 group indices exceed uint16 capacity")
    seed_sequence = np.random.SeedSequence(seed)
    bootstrap_sequence, swap_sequence = seed_sequence.spawn(2)

    bootstrap_temporary = bootstrap_path.with_suffix(bootstrap_path.suffix + ".tmp")
    bootstrap = _open_npy_memmap(
        bootstrap_temporary, shape=(resamples, group_count), dtype=np.dtype(np.uint16)
    )
    bootstrap_rng = np.random.Generator(np.random.PCG64(bootstrap_sequence))
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        bootstrap[start:stop] = bootstrap_rng.integers(
            0,
            group_count,
            size=(stop - start, group_count),
            dtype=np.uint16,
        )
    bootstrap.flush()
    del bootstrap
    bootstrap_temporary.replace(bootstrap_path)

    packed_columns = (group_count + 7) // 8
    swap_temporary = swap_path.with_suffix(swap_path.suffix + ".tmp")
    swap = _open_npy_memmap(
        swap_temporary, shape=(draws, packed_columns), dtype=np.dtype(np.uint8)
    )
    swap_rng = np.random.Generator(np.random.PCG64(swap_sequence))
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        swap[start:stop] = swap_rng.integers(
            0,
            256,
            size=(stop - start, packed_columns),
            dtype=np.uint8,
        )
    swap.flush()
    del swap
    swap_temporary.replace(swap_path)

    receipt = {
        "status": "VCOCO_CONTINUATION_SHARED_RANDOMIZATION_COMPLETE",
        "group_count": group_count,
        "group_ids": group_ids,
        "group_ids_sha256": group_ids_sha256(group_ids),
        "root_seed": seed,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "stream_construction": "numpy.random.SeedSequence(root_seed).spawn(2)",
        "bootstrap_spawn_key": list(bootstrap_sequence.spawn_key),
        "swap_spawn_key": list(swap_sequence.spawn_key),
        "swap_packbits_bitorder": "little",
        "swap_padding_bits_ignored": 8 * packed_columns - group_count,
        "artifacts": {
            bootstrap_path.name: {
                "sha256": sha256_file(bootstrap_path),
                "shape": [resamples, group_count],
                "dtype": "uint16",
            },
            swap_path.name: {
                "sha256": sha256_file(swap_path),
                "shape": [draws, packed_columns],
                "dtype": "uint8",
            },
        },
    }
    write_json(receipt_path, receipt)
    return receipt


def load_shared_randomization(
    groups: np.ndarray,
    protocol: dict[str, Any],
    bootstrap_path: Path,
    swap_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    receipt = read_json(receipt_path)
    group_ids = canonical_group_ids(groups)
    statistics = protocol["statistics"]
    group_count = len(group_ids)
    resamples = int(statistics["bootstrap_resamples"])
    draws = int(statistics["swap_monte_carlo_draws"])
    packed_columns = (group_count + 7) // 8
    if (
        receipt.get("status") != "VCOCO_CONTINUATION_SHARED_RANDOMIZATION_COMPLETE"
        or receipt.get("group_count") != group_count
        or receipt.get("group_ids") != group_ids
        or receipt.get("group_ids_sha256") != group_ids_sha256(group_ids)
        or receipt.get("root_seed") != int(statistics["bootstrap_seed"])
        or receipt.get("bit_generator") != "PCG64"
        or receipt.get("numpy_version") != np.__version__
        or receipt.get("stream_construction")
        != "numpy.random.SeedSequence(root_seed).spawn(2)"
        or receipt.get("bootstrap_spawn_key") != [0]
        or receipt.get("swap_spawn_key") != [1]
        or receipt.get("swap_packbits_bitorder") != "little"
        or receipt.get("swap_padding_bits_ignored") != 8 * packed_columns - group_count
    ):
        raise RuntimeError("V0 shared randomization receipt changed")
    expected = {
        bootstrap_path.name: (
            bootstrap_path,
            [resamples, group_count],
            np.dtype(np.uint16),
            "uint16",
        ),
        swap_path.name: (
            swap_path,
            [draws, packed_columns],
            np.dtype(np.uint8),
            "uint8",
        ),
    }
    if set(receipt.get("artifacts", {})) != set(expected):
        raise RuntimeError("V0 randomization artifact receipt set changed")
    arrays = []
    for name, (path, shape, dtype, logical_dtype) in expected.items():
        artifact = receipt["artifacts"][name]
        if (
            artifact.get("shape") != shape
            or artifact.get("dtype") != logical_dtype
            or artifact.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"V0 shared randomization artifact changed: {name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != shape or values.dtype != dtype:
            raise RuntimeError(f"V0 randomization array schema changed: {name}")
        arrays.append(values)
    bootstrap, swap = arrays
    seed_sequence = np.random.SeedSequence(int(statistics["bootstrap_seed"]))
    bootstrap_sequence, swap_sequence = seed_sequence.spawn(2)
    bootstrap_rng = np.random.Generator(np.random.PCG64(bootstrap_sequence))
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        expected_values = bootstrap_rng.integers(
            0,
            group_count,
            size=(stop - start, group_count),
            dtype=np.uint16,
        )
        if not np.array_equal(bootstrap[start:stop], expected_values):
            raise RuntimeError("V0 bootstrap indices do not reproduce the locked RNG stream")
    swap_rng = np.random.Generator(np.random.PCG64(swap_sequence))
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        expected_values = swap_rng.integers(
            0,
            256,
            size=(stop - start, packed_columns),
            dtype=np.uint8,
        )
        if not np.array_equal(swap[start:stop], expected_values):
            raise RuntimeError("V0 swap signs do not reproduce the locked RNG stream")
    return receipt, bootstrap, swap


def _assignment_splits(assignment: np.ndarray, *, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(assignment, dtype=np.int64)
    if set(np.unique(values)) != set(range(folds)):
        raise RuntimeError("A locked stack assignment is incomplete")
    return [
        (np.flatnonzero(values != fold), np.flatnonzero(values == fold))
        for fold in range(folds)
    ]


def fit_probability_stack_locked(
    train_features: dict[str, np.ndarray],
    target_features: dict[str, np.ndarray],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    stack_assignment: np.ndarray,
    *,
    component_names: tuple[str, ...],
    component_c: float,
    meta_c: float,
    class_weight: str,
    factorized: bool,
    reliability: bool,
    stack_folds: int,
    estimator_seed: int,
) -> np.ndarray:
    """Existing probability-stack logic with an explicit precomputed group split."""

    splits = _assignment_splits(stack_assignment, folds=stack_folds)
    train_probabilities = []
    target_probabilities = []
    for component_index, name in enumerate(component_names):
        values = train_features[name]
        target = target_features[name]
        oof = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
        for fold, (fit_index, held_index) in enumerate(splits):
            head = fit_probability_head(
                values[fit_index],
                labels[fit_index],
                factorized=factorized,
                c_value=component_c,
                class_weight=class_weight,
                seed=estimator_seed + 1_000 * component_index + fold,
            )
            oof[held_index] = predict_probability_head(head, values[held_index])
        final_head = fit_probability_head(
            values,
            labels,
            factorized=factorized,
            c_value=component_c,
            class_weight=class_weight,
            seed=estimator_seed + 1_000 * component_index + 99,
        )
        train_probabilities.append(normalize_probability_rows(oof))
        target_probabilities.append(predict_probability_head(final_head, target))
    meta_train = stack_features(train_probabilities, train_geometry, reliability=reliability)
    meta_target = stack_features(target_probabilities, target_geometry, reliability=reliability)
    meta = fit_probability_head(
        meta_train,
        labels,
        factorized=factorized,
        c_value=meta_c,
        class_weight=class_weight,
        seed=estimator_seed + 50_000,
    )
    return predict_probability_head(meta, meta_target)


def require_converged_svm_fit(estimator: Any, *, context: str) -> None:
    optimization = getattr(estimator, "optimization_", None)
    if not isinstance(optimization, dict):
        raise RuntimeError(f"CUDA SVM omitted its optimization receipt: {context}")
    if optimization.get("iteration_limit_reached") is not False:
        raise RuntimeError(
            f"CUDA SVM reached its iteration limit; V0 cannot continue: {context}"
        )


def fit_calibrated_svm_locked(
    train_features: dict[str, np.ndarray],
    target_features: dict[str, np.ndarray],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    stack_assignment: np.ndarray,
    *,
    component_names: tuple[str, ...],
    c_value: float,
    class_weight: str,
    calibrator_c: float,
    maximum_iterations: int,
    tolerance: float,
    stack_folds: int,
    estimator_seed: int,
) -> np.ndarray:
    """Fit the existing CUDA primal SVM with explicitly locked calibration folds."""

    features = np.concatenate(
        [*(train_features[name] for name in component_names), train_geometry], axis=1
    )
    target = np.concatenate(
        [*(target_features[name] for name in component_names), target_geometry], axis=1
    )

    def estimator(seed: int):
        return make_pipeline(
            StandardScaler(),
            CudaPrimalLinearSVM(
                c_value=c_value,
                class_weight=class_weight,
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
                seed=seed,
            ),
        )

    oof_scores = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
    for fold, (fit_index, held_index) in enumerate(
        _assignment_splits(stack_assignment, folds=stack_folds)
    ):
        model = estimator(estimator_seed + fold)
        model.fit(features[fit_index], labels[fit_index])
        require_converged_svm_fit(model[-1], context=f"calibration-fold-{fold}")
        oof_scores[held_index] = model.decision_function(features[held_index])
    calibrator = LogisticRegression(
        C=calibrator_c, max_iter=3_000, random_state=estimator_seed, solver="lbfgs"
    )
    calibrator.fit(oof_scores, labels)
    final = estimator(estimator_seed + 99)
    final.fit(features, labels)
    require_converged_svm_fit(final[-1], context="final-fit")
    return normalize_probability_rows(calibrator.predict_proba(final.decision_function(target)))


FitPredict = Callable[..., np.ndarray]


def fit_v0_candidate(
    candidate: Candidate,
    declaration: dict[str, Any],
    protocol: dict[str, Any],
    train_features: dict[str, np.ndarray],
    target_features: dict[str, np.ndarray],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    stack_assignment: np.ndarray,
    *,
    estimator_seed: int,
) -> np.ndarray:
    validate_group_assignment(groups, stack_assignment, name="model-stack")
    components = tuple(map(str, declaration["features"]))
    folds = int(protocol["cross_validation"]["stack_folds"])
    if candidate.family == "mixed_linear_svm":
        grid = protocol["V0"]["linear_svm_grid"]
        return fit_calibrated_svm_locked(
            train_features,
            target_features,
            train_geometry,
            target_geometry,
            labels,
            stack_assignment,
            component_names=components,
            c_value=float(candidate.svm_c),
            class_weight=candidate.class_weight,
            calibrator_c=float(grid["calibrator_C"]),
            maximum_iterations=int(grid["maximum_iterations"]),
            tolerance=float(grid["tolerance"]),
            stack_folds=folds,
            estimator_seed=estimator_seed,
        )
    return fit_probability_stack_locked(
        train_features,
        target_features,
        train_geometry,
        target_geometry,
        labels,
        stack_assignment,
        component_names=components,
        component_c=float(candidate.component_c),
        meta_c=float(candidate.meta_c),
        class_weight=candidate.class_weight,
        factorized=bool(declaration["factorized"]),
        reliability=bool(declaration["reliability"]),
        stack_folds=folds,
        estimator_seed=estimator_seed,
    )


def evaluate_candidate_inner_locked(
    candidate: Candidate,
    candidate_index: int,
    declaration: dict[str, Any],
    protocol: dict[str, Any],
    features: dict[str, np.ndarray],
    geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    inner_assignment: np.ndarray,
    selection_stack_assignments: dict[int, np.ndarray],
    *,
    outer_fold: int,
    fit_predict: FitPredict | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    fit_predict = fit_predict or fit_v0_candidate
    inner_folds = int(protocol["cross_validation"]["inner_folds"])
    stack_folds = int(protocol["cross_validation"]["stack_folds"])
    validate_group_assignment(groups, inner_assignment, name=f"inner-o{outer_fold}")
    probabilities = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
    candidate_seed = ESTIMATOR_BASE_SEED + 10_000 * outer_fold + candidate_index
    for inner_fold, (fit_index, held_index) in enumerate(
        _assignment_splits(inner_assignment, folds=inner_folds)
    ):
        estimator_seed = candidate_seed + 10_000 * (inner_fold + 1)
        if set(selection_stack_assignments) != set(range(inner_folds)):
            raise RuntimeError("Locked V0 selection-stack assignments are incomplete")
        stack_assignment = np.asarray(
            selection_stack_assignments[inner_fold], dtype=np.int64
        )[fit_index]
        if set(np.unique(stack_assignment)) != set(range(stack_folds)):
            raise RuntimeError("Locked V0 selection-stack assignment is invalid")
        validate_group_assignment(
            groups[fit_index],
            stack_assignment,
            name=f"selection-stack-o{outer_fold}-i{inner_fold}",
        )
        probabilities[held_index] = fit_predict(
            candidate,
            declaration,
            protocol,
            {name: values[fit_index] for name, values in features.items()},
            {name: values[held_index] for name, values in features.items()},
            geometry[fit_index],
            geometry[held_index],
            labels[fit_index],
            groups[fit_index],
            stack_assignment,
            estimator_seed=estimator_seed,
        )
    metrics = activity_metrics(labels, probabilities)
    metrics["locomotion_f1"] = locomotion_f1(labels, probabilities)
    return probabilities, metrics


def fit_family_outer_fold(
    family: str,
    family_index: int,
    outer_fold: int,
    protocol: dict[str, Any],
    features: dict[str, np.ndarray],
    geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fold_map: pd.DataFrame,
    *,
    fit_predict: FitPredict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Select and fit one family while keeping outer-held labels out of every fit."""

    fit_predict = fit_predict or fit_v0_candidate
    outer_assignment = fold_map["outer_fold"].to_numpy(dtype=np.int64)
    outer_train = np.flatnonzero(outer_assignment != outer_fold)
    outer_held = np.flatnonzero(outer_assignment == outer_fold)
    if not len(outer_train) or not len(outer_held):
        raise RuntimeError(f"Empty V0 outer split: {outer_fold}")
    inner_assignment = fold_map[f"inner_fold_o{outer_fold}"].to_numpy(dtype=np.int64)[
        outer_train
    ]
    final_stack = fold_map[f"stack_fold_o{outer_fold}"].to_numpy(dtype=np.int64)[
        outer_train
    ]
    selection_stacks = {
        inner_fold: fold_map[
            f"selection_stack_fold_o{outer_fold}_i{inner_fold}"
        ].to_numpy(dtype=np.int64)[outer_train]
        for inner_fold in range(int(protocol["cross_validation"]["inner_folds"]))
    }
    declaration = protocol["V0"]["families"][family]
    candidates = enumerate_v0_candidates(protocol, family)
    train_features = {name: values[outer_train] for name, values in features.items()}
    evaluated = []
    selection_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        _, metrics = evaluate_candidate_inner_locked(
            candidate,
            candidate_index,
            declaration,
            protocol,
            train_features,
            geometry[outer_train],
            labels[outer_train],
            groups[outer_train],
            inner_assignment,
            selection_stacks,
            outer_fold=outer_fold,
            fit_predict=fit_predict,
        )
        evaluated.append((candidate, metrics))
        selection_rows.append(
            {
                "family": family,
                "outer_fold": outer_fold,
                "candidate_index": candidate_index,
                "candidate_id": candidate.candidate_id,
                **metrics,
                "selected": False,
            }
        )
    selected, selected_metrics = min(
        evaluated, key=lambda value: candidate_rank_key(value[0], value[1])
    )
    for row in selection_rows:
        row["selected"] = row["candidate_id"] == selected.candidate_id
    estimator_seed = ESTIMATOR_BASE_SEED + 100_000 + 1_000 * outer_fold + family_index
    predicted = fit_predict(
        selected,
        declaration,
        protocol,
        train_features,
        {name: values[outer_held] for name, values in features.items()},
        geometry[outer_train],
        geometry[outer_held],
        labels[outer_train],
        groups[outer_train],
        final_stack,
        estimator_seed=estimator_seed,
    )
    print(
        json.dumps(
            {
                "family": family,
                "outer_fold": outer_fold,
                "selected": selected.candidate_id,
                "inner_macro_f1": selected_metrics["macro_f1"],
            }
        ),
        flush=True,
    )
    return outer_held, predicted, selection_rows


def _checkpoint_paths(output_dir: Path, family: str) -> tuple[Path, Path]:
    directory = output_dir / "checkpoints"
    return directory / f"{family}.json", directory / f"{family}.npz"


def load_checkpoint(
    output_dir: Path, family: str, *, rows: int, sources: dict[str, str]
) -> tuple[np.ndarray, list[int], list[dict[str, Any]], list[dict[str, Any]]] | None:
    metadata_path, arrays_path = _checkpoint_paths(output_dir, family)
    if not metadata_path.exists() and not arrays_path.exists():
        return None
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise RuntimeError(f"Incomplete V0 checkpoint pair: {family}")
    metadata = read_json(metadata_path)
    if (
        metadata.get("status") != "VCOCO_CONTINUATION_V0_FAMILY_CHECKPOINT"
        or metadata.get("family") != family
        or int(metadata.get("rows", -1)) != rows
        or metadata.get("source_sha256") != sources
        or metadata.get("artifact_sha256") != sha256_file(arrays_path)
    ):
        raise RuntimeError(f"Incompatible V0 family checkpoint: {family}")
    with np.load(arrays_path) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    if probabilities.shape != (rows, len(CLASS_NAMES)):
        raise RuntimeError(f"Invalid V0 checkpoint probability shape: {family}")
    completed = list(map(int, metadata.get("completed_outer_folds", [])))
    if len(completed) != len(set(completed)) or any(fold not in range(5) for fold in completed):
        raise RuntimeError(f"Invalid V0 completed-fold receipt: {family}")
    return (
        probabilities,
        completed,
        list(metadata.get("selection_rows", [])),
        list(metadata.get("cuda_svm_fit_audit", [])),
    )


def save_checkpoint(
    output_dir: Path,
    family: str,
    probabilities: np.ndarray,
    completed: list[int],
    selection_rows: list[dict[str, Any]],
    sources: dict[str, str],
) -> None:
    metadata_path, arrays_path = _checkpoint_paths(output_dir, family)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_suffix(arrays_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, probabilities=probabilities)
    temporary.replace(arrays_path)
    write_json(
        metadata_path,
        {
            "status": "VCOCO_CONTINUATION_V0_FAMILY_CHECKPOINT",
            "family": family,
            "rows": len(probabilities),
            "completed_outer_folds": sorted(completed),
            "selection_rows": selection_rows,
            "cuda_svm_fit_audit": cuda_svm_fit_audit(),
            "source_sha256": sources,
            "artifact_sha256": sha256_file(arrays_path),
        },
    )


def paired_nll_bootstrap(
    labels: np.ndarray,
    challenger: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    """Whole-source-image paired NLL interval using the shared locked draws."""

    labels = np.asarray(labels, dtype=np.int64)
    challenger = normalize_probability_rows(challenger)
    reference = normalize_probability_rows(reference)
    groups = np.asarray(groups).astype(str)
    unique, encoded = np.unique(groups, return_inverse=True)
    indices = np.asarray(bootstrap_indices)
    if (
        len(labels) != len(groups)
        or len(unique) < 2
        or indices.ndim != 2
        or indices.shape[1] != len(unique)
        or not len(indices)
        or int(indices.max()) >= len(unique)
    ):
        raise ValueError("Paired NLL bootstrap requires aligned rows and at least two groups")
    row = np.arange(len(labels))
    difference = -np.log(np.clip(challenger[row, labels], 1e-12, 1.0))
    difference += np.log(np.clip(reference[row, labels], 1e-12, 1.0))
    group_difference = np.bincount(encoded, weights=difference, minlength=len(unique))
    group_count = np.bincount(encoded, minlength=len(unique))
    values = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), 256):
        stop = min(start + 256, len(indices))
        sampled = indices[start:stop]
        values[start:stop] = group_difference[sampled].sum(axis=1) / group_count[
            sampled
        ].sum(axis=1)
    point = float(difference.mean())
    return {
        "point_estimate": point,
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "clusters": len(unique),
        "resamples": len(values),
        "shared_locked_indices": True,
        "inferential_note": "percentile_interval_only_null_test_reported_separately",
    }


def _macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1).astype(np.float64)
    denominator = confusion.sum(axis=-2) + confusion.sum(axis=-1)
    scores = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator != 0,
    )
    return scores.mean(axis=-1)


def paired_f1_bootstrap(
    labels: np.ndarray,
    challenger: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    """Paired macro/class F1 intervals using one shared source-image draw matrix."""

    labels = np.asarray(labels, dtype=np.int64)
    challenger_predictions = np.asarray(challenger).argmax(axis=1)
    reference_predictions = np.asarray(reference).argmax(axis=1)
    groups = np.asarray(groups).astype(str)
    unique, encoded = np.unique(groups, return_inverse=True)
    indices = np.asarray(bootstrap_indices)
    if (
        len(labels) != len(groups)
        or indices.ndim != 2
        or indices.shape[1] != len(unique)
        or not len(indices)
        or int(indices.max()) >= len(unique)
    ):
        raise ValueError("Paired F1 bootstrap inputs do not match the shared group design")
    classes = len(CLASS_NAMES)
    challenger_confusion = np.zeros((len(unique), classes, classes), dtype=np.int64)
    reference_confusion = np.zeros_like(challenger_confusion)
    np.add.at(challenger_confusion, (encoded, labels, challenger_predictions), 1)
    np.add.at(reference_confusion, (encoded, labels, reference_predictions), 1)

    def class_f1(confusion: np.ndarray) -> np.ndarray:
        diagonal = np.diagonal(confusion, axis1=-2, axis2=-1).astype(np.float64)
        denominator = confusion.sum(axis=-2) + confusion.sum(axis=-1)
        return np.divide(
            2.0 * diagonal,
            denominator,
            out=np.zeros_like(diagonal),
            where=denominator != 0,
        )

    observed_challenger = class_f1(challenger_confusion.sum(axis=0)[None])[0]
    observed_reference = class_f1(reference_confusion.sum(axis=0)[None])[0]
    deltas = np.empty((len(indices), classes), dtype=np.float64)
    for start in range(0, len(indices), 256):
        stop = min(start + 256, len(indices))
        sampled = indices[start:stop]
        challenger_f1 = class_f1(challenger_confusion[sampled].sum(axis=1))
        reference_f1 = class_f1(reference_confusion[sampled].sum(axis=1))
        deltas[start:stop] = challenger_f1 - reference_f1

    def summarize(values: np.ndarray, point: float) -> dict[str, float]:
        return {
            "point_estimate": float(point),
            "ci_95_low": float(np.quantile(values, 0.025)),
            "ci_95_high": float(np.quantile(values, 0.975)),
        }

    return {
        "macro_f1": summarize(
            deltas.mean(axis=1), float((observed_challenger - observed_reference).mean())
        ),
        "per_class_f1": {
            name: summarize(
                deltas[:, class_index],
                observed_challenger[class_index] - observed_reference[class_index],
            )
            for class_index, name in enumerate(CLASS_NAMES)
        },
        "clusters": len(unique),
        "resamples": len(indices),
        "shared_locked_indices": True,
        "inferential_note": "percentile_intervals_only_null_test_reported_separately",
    }


def paired_group_swap_test(
    labels: np.ndarray,
    challenger: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    swap_signs_packbits: np.ndarray,
) -> dict[str, Any]:
    """Swap complete source-image probability vectors with plus-one correction."""

    labels = np.asarray(labels, dtype=np.int64)
    challenger = normalize_probability_rows(challenger)
    reference = normalize_probability_rows(reference)
    groups = np.asarray(groups).astype(str)
    unique, encoded = np.unique(groups, return_inverse=True)
    packed = np.asarray(swap_signs_packbits)
    expected_columns = (len(unique) + 7) // 8
    if (
        len(labels) != len(groups)
        or len(unique) < 2
        or packed.ndim != 2
        or packed.shape[1] != expected_columns
        or not len(packed)
    ):
        raise ValueError("Group swap test requires aligned rows and at least two groups")
    classes = len(CLASS_NAMES)
    challenger_confusion = np.zeros((len(unique), classes, classes), dtype=np.int64)
    reference_confusion = np.zeros_like(challenger_confusion)
    np.add.at(challenger_confusion, (encoded, labels, challenger.argmax(axis=1)), 1)
    np.add.at(reference_confusion, (encoded, labels, reference.argmax(axis=1)), 1)
    row = np.arange(len(labels))
    challenger_nll = -np.log(np.clip(challenger[row, labels], 1e-12, 1.0))
    reference_nll = -np.log(np.clip(reference[row, labels], 1e-12, 1.0))
    challenger_group_nll = np.bincount(encoded, weights=challenger_nll, minlength=len(unique))
    reference_group_nll = np.bincount(encoded, weights=reference_nll, minlength=len(unique))
    total_confusion = challenger_confusion + reference_confusion
    difference_confusion = challenger_confusion - reference_confusion
    difference_nll = challenger_group_nll - reference_group_nll
    observed_macro = float(
        _macro_f1_from_confusion(challenger_confusion.sum(axis=0)[None])[0]
        - _macro_f1_from_confusion(reference_confusion.sum(axis=0)[None])[0]
    )
    observed_nll = float(challenger_nll.mean() - reference_nll.mean())
    macro_extreme = 0
    nll_extreme = 0
    half_total_confusion = total_confusion.sum(axis=0, dtype=np.int64)
    for start in range(0, len(packed), 256):
        stop = min(start + 256, len(packed))
        bits = np.unpackbits(packed[start:stop], axis=1, bitorder="little")[:, : len(unique)]
        signs = 1 - 2 * bits.astype(np.int16)
        signed_confusion = np.einsum("bg,gij->bij", signs, difference_confusion)
        randomized_challenger = (half_total_confusion[None] + signed_confusion) / 2.0
        randomized_reference = (half_total_confusion[None] - signed_confusion) / 2.0
        macro_delta = _macro_f1_from_confusion(
            randomized_challenger
        ) - _macro_f1_from_confusion(randomized_reference)
        signed_nll = signs @ difference_nll
        nll_delta = signed_nll / len(labels)
        macro_extreme += int(np.count_nonzero(np.abs(macro_delta) >= abs(observed_macro) - 1e-15))
        nll_extreme += int(np.count_nonzero(np.abs(nll_delta) >= abs(observed_nll) - 1e-15))
    draws = len(packed)
    return {
        "unit": "whole_source_image_probability_vector",
        "clusters": len(unique),
        "draws": draws,
        "plus_one_correction": True,
        "shared_locked_signs": True,
        "packbits_bitorder": "little",
        "macro_f1": {
            "point_estimate": observed_macro,
            "two_sided_p": float((macro_extreme + 1) / (draws + 1)),
        },
        "nll": {
            "point_estimate": observed_nll,
            "two_sided_p": float((nll_extreme + 1) / (draws + 1)),
        },
    }


def build_paired_statistics(
    protocol: dict[str, Any],
    labels: np.ndarray,
    groups: np.ndarray,
    probabilities: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
    swap_signs_packbits: np.ndarray,
    randomization_receipt: dict[str, Any],
) -> dict[str, Any]:
    statistics = protocol["statistics"]
    if bootstrap_indices.shape != (
        int(statistics["bootstrap_resamples"]),
        len(randomization_receipt["group_ids"]),
    ):
        raise RuntimeError("V0 bootstrap design does not match the declared statistics")
    if swap_signs_packbits.shape[0] != int(statistics["swap_monte_carlo_draws"]):
        raise RuntimeError("V0 swap design does not match the declared statistics")
    comparisons_payload: dict[str, Any] = {}
    macro_p: dict[str, float] = {}
    nll_p: dict[str, float] = {}
    for reference_name, challenger_name in combinations(FAMILY_ORDER, 2):
        comparison_name = f"{challenger_name}_minus_{reference_name}"
        challenger = probabilities[challenger_name]
        reference = probabilities[reference_name]
        f1_bootstrap = paired_f1_bootstrap(
            labels,
            challenger,
            reference,
            groups,
            bootstrap_indices=bootstrap_indices,
        )
        nll_bootstrap = paired_nll_bootstrap(
            labels,
            challenger,
            reference,
            groups,
            bootstrap_indices=bootstrap_indices,
        )
        swap = paired_group_swap_test(
            labels,
            challenger,
            reference,
            groups,
            swap_signs_packbits=swap_signs_packbits,
        )
        comparisons_payload[comparison_name] = {
            "challenger": challenger_name,
            "reference": reference_name,
            "bootstrap": {**f1_bootstrap, "nll": nll_bootstrap},
            "swap_test": swap,
        }
        macro_p[comparison_name] = float(swap["macro_f1"]["two_sided_p"])
        nll_p[comparison_name] = float(swap["nll"]["two_sided_p"])
    adjusted_macro = holm_adjust(macro_p)
    adjusted_nll = holm_adjust(nll_p)
    for name, payload in comparisons_payload.items():
        payload["holm_adjusted_swap_p"] = {
            "macro_f1": float(adjusted_macro[name]),
            "nll": float(adjusted_nll[name]),
        }
    return {
        "status": "VCOCO_CONTINUATION_V0_PAIRED_STATISTICS_COMPLETE",
        "comparison_scope": "all_six_pairwise_family_comparisons",
        "multiplicity": "Holm_within_V0_pairwise_family",
        "shared_randomization": {
            "root_seed": randomization_receipt["root_seed"],
            "bit_generator": randomization_receipt["bit_generator"],
            "numpy_version": randomization_receipt["numpy_version"],
            "group_ids_sha256": randomization_receipt["group_ids_sha256"],
            "artifact_sha256": {
                name: declaration["sha256"]
                for name, declaration in randomization_receipt["artifacts"].items()
            },
        },
        "comparisons": comparisons_payload,
    }


def run_v0(
    root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    source_lock_path: Path,
    execution_lock_path: Path,
    execution_lock: dict[str, Any],
    fold_map_path: Path,
    bootstrap_path: Path,
    swap_path: Path,
    randomization_receipt_path: Path,
    output_dir: Path,
    *,
    max_new_family_folds: int | None = None,
    benchmark_family: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows, features = load_rows_and_features(root, protocol, load_features=True)
    fold_map = pd.read_csv(
        fold_map_path, dtype={"person_id": str, "image_id": str}
    )
    validate_fold_map(fold_map, rows, protocol)
    labels = label_indices(rows)
    groups = rows["image_id"].astype(str).to_numpy(dtype=str)
    geometry = geometry_features(rows)
    randomization_receipt, bootstrap_indices, swap_signs = load_shared_randomization(
        groups,
        protocol,
        bootstrap_path,
        swap_path,
        randomization_receipt_path,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("The matched V0 linear-SVM control requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    sources = {
        "protocol": sha256_file(protocol_path),
        "source_lock": sha256_file(source_lock_path),
        "execution_lock": sha256_file(execution_lock_path),
        "fold_map": sha256_file(fold_map_path),
        "runner": sha256_file(root / RUNNER_PATH),
        "model_module": sha256_file(root / MODEL_MODULE_PATH),
        "bootstrap_group_indices": sha256_file(bootstrap_path),
        "swap_signs_packbits": sha256_file(swap_path),
        "statistics_randomization_receipt": sha256_file(randomization_receipt_path),
        "shared_rows_sidecar": str(
            protocol["V0"]["shared_geometry_features"]["source_rows_sha256"]
        ),
    }
    family_probabilities: dict[str, np.ndarray] = {}
    selection_rows: list[dict[str, Any]] = []
    optimization_records: list[dict[str, Any]] = []
    outer_folds = int(protocol["cross_validation"]["outer_folds"])
    new_family_folds = 0
    benchmark_workloads: list[dict[str, Any]] = []
    if max_new_family_folds is not None:
        if max_new_family_folds < 1:
            raise ValueError("--max-new-family-folds must be positive")
    execution_order = (
        (benchmark_family,) if benchmark_family is not None else FAMILY_ORDER
    )
    for family in execution_order:
        family_index = FAMILY_ORDER.index(family)
        reset_cuda_svm_fit_audit()
        checkpoint = load_checkpoint(output_dir, family, rows=len(rows), sources=sources)
        if checkpoint is None:
            oof = np.full((len(rows), len(CLASS_NAMES)), np.nan, dtype=np.float64)
            completed: list[int] = []
            family_selection: list[dict[str, Any]] = []
        else:
            oof, completed, family_selection, audit = checkpoint
            restore_cuda_svm_fit_audit(audit)
        for outer_fold in range(outer_folds):
            if outer_fold in completed:
                continue
            torch.cuda.synchronize()
            workload_started = time.perf_counter()
            held, predicted, rows_for_fold = fit_family_outer_fold(
                family,
                family_index,
                outer_fold,
                protocol,
                features,
                geometry,
                labels,
                groups,
                fold_map,
            )
            oof[held] = predicted
            family_selection.extend(rows_for_fold)
            completed.append(outer_fold)
            save_checkpoint(
                output_dir,
                family,
                oof,
                completed,
                family_selection,
                sources,
            )
            torch.cuda.synchronize()
            new_family_folds += 1
            selected = next(row for row in rows_for_fold if row["selected"])
            benchmark_workloads.append(
                {
                    "family": family,
                    "family_index": family_index,
                    "outer_fold": outer_fold,
                    "candidate_settings_evaluated": len(rows_for_fold),
                    "selected_candidate": selected["candidate_id"],
                    "outer_held_people": len(held),
                    "elapsed_seconds": time.perf_counter() - workload_started,
                }
            )
            if (
                max_new_family_folds is not None
                and new_family_folds >= max_new_family_folds
            ):
                output_dir.mkdir(parents=True, exist_ok=True)
                benchmark_path = output_dir / "bounded_benchmark_receipt.json"
                benchmark = {
                    "status": "VCOCO_CONTINUATION_V0_BOUNDED_CHECKPOINT_COMPLETE",
                    "completion_status": "PARTIAL_RESUMABLE_WITH_IDENTICAL_LOCKS",
                    "max_new_family_folds": max_new_family_folds,
                    "requested_benchmark_family": benchmark_family,
                    "new_family_folds_completed": new_family_folds,
                    "workloads": benchmark_workloads,
                    "candidate_budget_per_workload": 8,
                    "candidate_selection": protocol["V0"]["candidate_selection"],
                    "cuda_device": torch.cuda.get_device_name(0),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "cuda_svm_fits": len(cuda_svm_fit_audit()),
                    "cuda_svm_iteration_limit_reached_fits": sum(
                        bool(record["iteration_limit_reached"])
                        for record in cuda_svm_fit_audit()
                    ),
                    "execution_determinism": protocol["execution_determinism"],
                    "resource_scope": protocol["resource_contract"],
                    "runtime_seconds": time.perf_counter() - started,
                    "protected_access": {key: 0 for key in PROTECTED_COUNTERS},
                    "source_sha256": sources,
                }
                write_json(benchmark_path, benchmark)
                print(json.dumps(benchmark, indent=2, sort_keys=True), flush=True)
                return benchmark
        if sorted(completed) != list(range(outer_folds)) or not np.isfinite(oof).all():
            raise RuntimeError(f"V0 family did not produce complete finite OOF predictions: {family}")
        if len(family_selection) != outer_folds * 8:
            raise RuntimeError(f"V0 family did not consume the matched selection budget: {family}")
        if sum(bool(row["selected"]) for row in family_selection) != outer_folds:
            raise RuntimeError(f"V0 family selected an invalid number of candidates: {family}")
        family_probabilities[family] = oof
        selection_rows.extend(family_selection)
        optimization_records.extend(cuda_svm_fit_audit())

    if benchmark_family is not None:
        raise RuntimeError(
            f"No new bounded workload remains for benchmark family: {benchmark_family}"
        )
    if any(bool(record.get("iteration_limit_reached")) for record in optimization_records):
        raise RuntimeError("V0 cannot complete with an iteration-limited CUDA SVM fit")

    metric_rows = []
    class_rows = []
    for family, probabilities in family_probabilities.items():
        metric_rows.append(
            {
                "family": family,
                **activity_metrics(labels, probabilities),
                "locomotion_f1": locomotion_f1(labels, probabilities),
            }
        )
        class_rows.extend(
            {"family": family, **row}
            for row in per_class_metrics(labels, probabilities, CLASS_NAMES)
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    paired = build_paired_statistics(
        protocol,
        labels,
        groups,
        family_probabilities,
        bootstrap_indices,
        swap_signs,
        randomization_receipt,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    classes_path = output_dir / "per_class_metrics.csv"
    selection_path = output_dir / "candidate_selection.csv"
    probabilities_path = output_dir / "oof_probabilities.npz"
    statistics_path = output_dir / "paired_statistics.json"
    optimization_path = output_dir / "cuda_svm_optimization.json"
    fold_receipt_path = output_dir / "fold_usage_receipt.json"
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    pd.DataFrame(class_rows).to_csv(classes_path, index=False, lineterminator="\n")
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False, lineterminator="\n")
    np.savez_compressed(
        probabilities_path,
        person_ids=rows["person_id"].astype(str).to_numpy(dtype=str),
        source_image_ids=groups,
        labels=labels,
        class_names=np.asarray(CLASS_NAMES),
        **family_probabilities,
    )
    write_json(statistics_path, paired)
    write_json(
        optimization_path,
        {
            "status": "VCOCO_CONTINUATION_V0_CUDA_SVM_OPTIMIZATION_AUDITED",
            "solver": "pytorch_cuda_lbfgs_ovr_squared_hinge",
            "device": torch.cuda.get_device_name(0),
            "fits": len(optimization_records),
            "iteration_limit_reached_fits": sum(
                bool(record["iteration_limit_reached"]) for record in optimization_records
            ),
            "records": optimization_records,
        },
    )
    write_json(
        fold_receipt_path,
        {
            "status": "VCOCO_CONTINUATION_V0_SHARED_FOLDS_CONSUMED",
            "fold_map_sha256": sources["fold_map"],
            "outer_folds": outer_folds,
            "inner_folds": int(protocol["cross_validation"]["inner_folds"]),
            "stack_folds": int(protocol["cross_validation"]["stack_folds"]),
            "outer_seed": int(protocol["cross_validation"]["random_seed"]),
            "inner_seed_formula": "random_seed + 10000 * (outer_fold + 1)",
            "final_stack_seed_formula": "random_seed + 100000 + outer_fold",
            "selection_stack_seed_formula": (
                "random_seed + 200000 + 1000 * outer_fold + inner_fold"
            ),
            "inner_candidate_estimator_seed_formula": (
                "20260907 + 10000 * outer_fold + candidate_index + 10000 * (inner_fold + 1)"
            ),
            "final_estimator_seed_formula": (
                "20260907 + 100000 + 1000 * outer_fold + family_index"
            ),
            "family_order": list(FAMILY_ORDER),
            "shared_geometry_features": protocol["V0"]["shared_geometry_features"],
            "candidate_budget_per_family_per_outer_fold": 8,
            "candidate_selection": protocol["V0"]["candidate_selection"],
        },
    )
    artifacts = [
        metrics_path,
        classes_path,
        selection_path,
        probabilities_path,
        statistics_path,
        optimization_path,
        fold_receipt_path,
    ]
    summary = {
        "status": "VCOCO_CONTINUATION_V0_MATCHED_CONTROLS_COMPLETE",
        "endpoint": "three_class_person_level_source_tag_classification",
        "best_family": str(metrics.iloc[0]["family"]),
        "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
        "development_people": len(rows),
        "development_source_images": int(pd.Series(groups).nunique()),
        "family_order": list(FAMILY_ORDER),
        "shared_geometry_features": protocol["V0"]["shared_geometry_features"],
        "candidate_budget_per_family_per_outer_fold": 8,
        "candidate_selection": protocol["V0"]["candidate_selection"],
        "outer_folds": outer_folds,
        "model_families": len(FAMILY_ORDER),
        "protected_access": {key: 0 for key in PROTECTED_COUNTERS},
        "official_test_predictions_run": False,
        "V1_fitting_performed": False,
        "execution_determinism": protocol["execution_determinism"],
        "resource_scope": protocol["resource_contract"],
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": sources,
        "execution_authorization": execution_lock["authorization"],
        "artifact_sha256": {path.name: sha256_file(path) for path in artifacts},
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.max_new_family_folds is not None and args.mode != "run":
        raise ValueError("--max-new-family-folds is valid only with --mode run")
    if args.benchmark_family is not None and args.max_new_family_folds is None:
        raise ValueError("--benchmark-family requires --max-new-family-folds")
    protocol_path = (root / args.protocol).resolve() if not args.protocol.is_absolute() else args.protocol
    source_lock_path = (
        (root / args.source_lock).resolve() if not args.source_lock.is_absolute() else args.source_lock
    )
    fold_map_path = (
        (root / args.fold_map).resolve() if not args.fold_map.is_absolute() else args.fold_map
    )
    bootstrap_path = (
        (root / args.bootstrap_group_indices).resolve()
        if not args.bootstrap_group_indices.is_absolute()
        else args.bootstrap_group_indices
    )
    swap_path = (
        (root / args.swap_signs_packbits).resolve()
        if not args.swap_signs_packbits.is_absolute()
        else args.swap_signs_packbits
    )
    randomization_receipt_path = (
        (root / args.statistics_randomization_receipt).resolve()
        if not args.statistics_randomization_receipt.is_absolute()
        else args.statistics_randomization_receipt
    )
    protocol = read_json(protocol_path)
    configure_deterministic_execution(protocol)
    validate_protocol(protocol)
    source_lock = validate_source_lock(
        root,
        protocol_path,
        protocol,
        source_lock_path,
        verify_feature_arrays=args.mode == "run",
    )
    rows, _ = load_rows_and_features(root, protocol, load_features=False)
    if args.mode == "prepare-fold-map":
        fold_map = build_fold_map(rows, protocol)
        write_fold_map(fold_map_path, fold_map)
        randomization = prepare_shared_randomization(
            rows["image_id"].astype(str).to_numpy(dtype=str),
            protocol,
            bootstrap_path,
            swap_path,
            randomization_receipt_path,
        )
        result = {
            "status": "VCOCO_CONTINUATION_V0_PREFIT_DESIGN_COMPLETE",
            "fold_map_path": relative_safe_path(root, fold_map_path),
            "rows": len(fold_map),
            "source_images": int(fold_map["image_id"].nunique()),
            "model_fits": 0,
            "protected_access": {key: 0 for key in PROTECTED_COUNTERS},
            "source_lock_sha256": sha256_file(source_lock_path),
            "artifact_sha256": {
                fold_map_path.name: sha256_file(fold_map_path),
                bootstrap_path.name: sha256_file(bootstrap_path),
                swap_path.name: sha256_file(swap_path),
                randomization_receipt_path.name: sha256_file(randomization_receipt_path),
            },
            "statistics_randomization_status": randomization["status"],
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return

    execution_lock_path = (
        (root / args.execution_lock).resolve()
        if not args.execution_lock.is_absolute()
        else args.execution_lock
    )
    execution_lock = validate_execution_lock(
        root,
        source_lock_path,
        source_lock,
        execution_lock_path,
        fold_map_path,
        bootstrap_path,
        swap_path,
        randomization_receipt_path,
    )
    fold_map = pd.read_csv(fold_map_path, dtype={"person_id": str, "image_id": str})
    validate_fold_map(fold_map, rows, protocol)
    output_dir = (
        (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    )
    run_v0(
        root,
        protocol_path,
        protocol,
        source_lock_path,
        execution_lock_path,
        execution_lock,
        fold_map_path,
        bootstrap_path,
        swap_path,
        randomization_receipt_path,
        output_dir,
        max_new_family_folds=args.max_new_family_folds,
        benchmark_family=args.benchmark_family,
    )


if __name__ == "__main__":
    main()
