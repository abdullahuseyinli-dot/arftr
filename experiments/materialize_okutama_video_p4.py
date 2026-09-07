"""Select P4 development-only context and trajectory values after source locking."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_okutama_video_probe as p1  # noqa: E402

BUNDLE_MEMBERS = {
    "sample_ids",
    "recording_ids",
    "track_ids",
    "scope",
    "fold",
    "labels",
    "transition_targets",
    "occlusion_targets",
    "source_feature_indices",
    "window_occluded",
    "static_features",
    "short_features",
    "short_valid_mask",
    "short_centre_index",
    "distinct_short_features",
    "distinct_short_valid_mask",
    "distinct_short_indices",
    "distinct_short_centre_index",
    "part_tokens",
    "part_confidence",
    "part_valid_mask",
    "part_centre_index",
    "quality_features",
    "primary_retained_teacher_sample_ids",
    "retained_teacher_seeds",
    "primary_retained_teacher_probabilities",
}
OUTPUT_MEMBERS = {
    "sample_ids",
    "legacy_static_features",
    "legacy_quality_features",
    "raw_sequence",
    "raw_summary",
    "compensated_sequence",
    "compensated_summary",
    "camera_quality",
}
SHAPES = {
    "legacy_static_features": (4977, 1542),
    "legacy_quality_features": (4977, 8),
    "raw_sequence": (4977, 17, 21),
    "raw_summary": (4977, 58),
    "compensated_sequence": (4977, 17, 21),
    "compensated_summary": (4977, 58),
    "camera_quality": (4977, 17),
}


def _load_lock(root: Path, path: Path) -> tuple[Any, dict[str, Any]]:
    locker = importlib.import_module("tools.lock_okutama_video_p4_materialization")
    return locker, locker.validate_lock(root, path.resolve())


def _primary_rows(locker: Any, path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = locker._csv(path.read_bytes())
    positions = np.asarray(
        [index for index, row in enumerate(rows) if row["scope"] == "grouped_crossfit_oof"],
        dtype=np.int64,
    )
    primary = [rows[index] for index in positions]
    if len(primary) != 4977:
        raise RuntimeError("P4 primary selection changed after locking")
    return primary, positions


def _validate_values(values: dict[str, np.ndarray], sample_ids: np.ndarray) -> None:
    if sample_ids.shape != (4977,) or sample_ids.dtype.kind != "U":
        raise RuntimeError("P4 materialized sample IDs changed")
    for name, shape in SHAPES.items():
        array = values[name]
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise RuntimeError(f"P4 materialized feature contract changed: {name}")
    if np.any((values["camera_quality"] < 0) | (values["camera_quality"] > 1)):
        raise RuntimeError("P4 camera quality left [0,1]")


def _validate_complete(output: Path, summary: dict[str, Any], request_sha: str) -> None:
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P4_PRIMARY_FEATURE_CACHE_COMPLETE"
        or summary.get("request_sha256") != request_sha
        or summary.get("rows") != 4977
    ):
        raise RuntimeError("Retained P4 materialization belongs to another request")
    expected = {"request.json", "features.npz", "summary.json"}
    if {path.name for path in output.iterdir() if path.is_file()} != expected:
        raise RuntimeError("Retained P4 materialization inventory changed")
    feature_path = output / "features.npz"
    if p1.sha256_file(feature_path) != summary.get("feature_sha256"):
        raise RuntimeError("Retained P4 feature bytes changed")
    with np.load(feature_path, allow_pickle=False) as source:
        if set(source.files) != OUTPUT_MEMBERS:
            raise RuntimeError("Retained P4 feature member inventory changed")
        sample_ids = source["sample_ids"]
        values = {name: source[name] for name in SHAPES}
    _validate_values(values, sample_ids)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P4 materialization output must be a dedicated .runs directory")
    locker, lock = _load_lock(root, args.materialization_lock)
    request = {
        "status": "P4_PRIMARY_FEATURE_REQUEST_BEFORE_VALUE_ACCESS",
        "materialization_lock_sha256": p1.sha256_file(args.materialization_lock.resolve()),
        "protocol_sha256": lock["protocol_sha256"],
        "primary_sample_ids_sha256": lock["primary_sample_ids_sha256"],
        "primary_source_indices_sha256": lock["primary_source_indices_sha256"],
        "output": str(output),
    }
    request_path, summary_path = output / "request.json", output / "summary.json"
    if request_path.exists():
        if json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise RuntimeError("P4 materialization output belongs to another request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty P4 materialization output lacks its request")
    else:
        p1.atomic_json(request_path, request)
    request_sha = p1.canonical_digest(request)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_complete(output, summary, request_sha)
        return summary
    if (output / "features.npz").exists():
        raise RuntimeError("Incomplete P4 materialization retained; refusing overwrite")

    started = time.perf_counter()
    index_path = p1._checked_path(root, lock["inputs"]["eligible_index"])
    primary, positions = _primary_rows(locker, index_path)
    expected_ids = np.asarray([row["sample_id"] for row in primary], dtype=str)
    expected_indices = np.asarray([int(row["feature_index"]) for row in primary], dtype=np.int64)

    bundle_path = p1._checked_path(root, lock["inputs"]["role_safe_bundle"])
    with np.load(bundle_path, allow_pickle=False) as bundle:
        if set(bundle.files) != BUNDLE_MEMBERS:
            raise RuntimeError("Role-safe bundle member inventory changed")
        bundle_ids = bundle["sample_ids"]
        bundle_indices = bundle["source_feature_indices"]
        if (
            bundle_ids.shape != (6360,)
            or bundle_ids.dtype.kind != "U"
            or bundle_indices.shape != (6360,)
            or bundle_indices.dtype != np.int64
            or not np.array_equal(bundle_ids[positions], expected_ids)
            or not np.array_equal(bundle_indices[positions], expected_indices)
        ):
            raise RuntimeError("Role-safe bundle identity/index alignment changed")
        static_all = bundle["static_features"]
        quality_all = bundle["quality_features"]
        if static_all.shape != (6360, 1542) or static_all.dtype != np.float32:
            raise RuntimeError("Role-safe static feature contract changed")
        if quality_all.shape != (6360, 8) or quality_all.dtype != np.float32:
            raise RuntimeError("Role-safe quality feature contract changed")
        legacy_static = np.asarray(static_all[positions], dtype=np.float32)
        legacy_quality = np.asarray(quality_all[positions], dtype=np.float32)

    source_shapes = {
        "raw_sequence": (8339, 17, 21),
        "raw_summary": (8339, 58),
        "compensated_sequence": (8339, 17, 21),
        "compensated_summary": (8339, 58),
        "camera_quality": (8339, 17),
    }
    selected: dict[str, np.ndarray] = {}
    for name, shape in source_shapes.items():
        path = p1._checked_path(root, lock["inputs"]["motion_artifacts"][name])
        source = np.load(path, allow_pickle=False, mmap_mode="r")
        if source.shape != shape or source.dtype != np.float32:
            raise RuntimeError(f"Role-mixed motion-array contract changed: {name}")
        selected[name] = np.asarray(source[expected_indices], dtype=np.float32)

    values = {
        "legacy_static_features": legacy_static,
        "legacy_quality_features": legacy_quality,
        **selected,
    }
    _validate_values(values, expected_ids)
    feature_path = output / "features.npz"
    p1.atomic_bytes(feature_path, p1.npz_bytes(sample_ids=expected_ids, **values))
    # Close the hash-to-select window against every source before publishing completion.
    _load_lock(root, args.materialization_lock)
    summary = {
        "status": "OKUTAMA_VIDEO_P4_PRIMARY_FEATURE_CACHE_COMPLETE",
        "rows": 4977,
        "sample_ids_sha256": p1.canonical_digest(expected_ids.tolist()),
        "source_indices_sha256": p1.canonical_digest(expected_indices.tolist()),
        "request_sha256": request_sha,
        "feature_path": "features.npz",
        "feature_sha256": p1.sha256_file(feature_path),
        "feature_size_bytes": feature_path.stat().st_size,
        "members": {
            "sample_ids": {"shape": [4977], "dtype": str(expected_ids.dtype)},
            **{
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in values.items()
            },
        },
        "runtime_seconds": time.perf_counter() - started,
        "access_accounting": {
            "role_safe_development_bundle_rows_decoded": 6360,
            "role_safe_auxiliary_development_rows_persisted": 0,
            "role_mixed_motion_feature_rows_read": 4977,
            "role_mixed_motion_complement_rows_read": 0,
            "labels_or_predictions_read_from_role_safe_bundle": 0,
            "protected_rows_read": 0,
            "raw_images_read": 0,
            "model_fits": 0,
        },
    }
    p1.atomic_json(summary_path, summary)
    _validate_complete(output, summary, request_sha)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
