"""Build the observation-only signed correspondence-field cache from retained P8 work."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from hac.actor_memory_base import canonical_hash, file_sha256
from hac.correspondence_field_data import (
    CONTEXT_PATHS,
    CONTEXT_SHA256,
    PAIR_COUNT,
    POINT_COLUMNS,
    index_workloads,
    load_correspondence_workload,
    load_frozen_context,
)

ROOT = Path(__file__).resolve().parents[1]
P8_RESULTS = ROOT / ".runs/research_20260907/okutama_ccac_full_v2/results"
MEMORY = ROOT / CONTEXT_PATHS["memory"]
DEFAULT_OUTPUT = ROOT / ".runs/research_20260912/correspondence_field_v1/data"


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _validate_complete(output: Path) -> dict:
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "CORRESPONDENCE_FIELD_OBSERVATION_CACHE_COMPLETE":
        raise RuntimeError("Existing field cache has no valid completion status")
    for name, expected in receipt["output_sha256"].items():
        path = output / name
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Completed field cache changed: {name}")
    return receipt


def build(output: Path = DEFAULT_OUTPUT) -> dict:
    output = output.resolve()
    output.relative_to(ROOT.resolve())
    if (output / "receipt.json").is_file():
        return _validate_complete(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Partial field cache exists; use a fresh versioned output")
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise RuntimeError(
            "Stale incomplete field cache exists; inspect it and use a fresh version"
        )
    started = time.perf_counter()
    with np.load(MEMORY, allow_pickle=False) as source:
        sample_ids = source["sample_ids"].astype(str)
    if sample_ids.shape != (4977,) or len(np.unique(sample_ids)) != 4977:
        raise RuntimeError("Canonical sample population changed")
    workloads = index_workloads(P8_RESULTS, sample_ids)
    context, context_audit = load_frozen_context(ROOT, sample_ids)
    points = np.zeros((4977, PAIR_COUNT, 32, len(POINT_COLUMNS)), dtype=np.float32)
    point_valid = np.zeros((4977, PAIR_COUNT, 32), dtype=bool)
    pair_features = np.zeros((4977, PAIR_COUNT, 5), dtype=np.float32)
    pair_valid = np.zeros((4977, PAIR_COUNT), dtype=bool)
    midpoint_seconds = np.zeros((4977, PAIR_COUNT), dtype=np.float32)
    collapsed = np.zeros((4977, 31), dtype=np.float32)
    audits = []
    for row, sample_id in enumerate(sample_ids):
        field = load_correspondence_workload(
            workloads[sample_id], repository_root=ROOT, expected_sample_id=sample_id
        )
        points[row] = field.point_features
        point_valid[row] = field.input_valid
        pair_features[row] = field.pair_features
        pair_valid[row] = field.pair_input_valid
        midpoint_seconds[row] = field.midpoint_seconds
        collapsed[row] = field.collapsed_features
        audits.append(field.audit)
        if (row + 1) % 500 == 0:
            print(json.dumps({"event": "field_cache_progress", "rows": row + 1}), flush=True)
    arrays = {
        "sample_ids": sample_ids,
        "context": context,
        "point_features": points,
        "point_input_valid": point_valid,
        "pair_features": pair_features,
        "pair_input_valid": pair_valid,
        "midpoint_seconds": midpoint_seconds,
        "collapsed_features": collapsed,
    }
    if (
        any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f")
        or not np.array_equal(pair_valid, point_valid.any(2))
        or np.any(points[~point_valid] != 0)
        or np.any(pair_features[~pair_valid] != 0)
    ):
        raise RuntimeError("Built field cache violates finite/mask sentinel contracts")
    staging.mkdir(parents=True)
    cache_path = staging / "correspondence_field.npz"
    with cache_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    audit_path = staging / "workload_audit.json"
    _write_json_exclusive(
        audit_path,
        {
            "status": "CORRESPONDENCE_FIELD_WORKLOAD_AUDIT_COMPLETE",
            "rows": len(audits),
            "sample_ids_sha256": canonical_hash(sample_ids.tolist()),
            "annotation_fields_used_for_inference": [],
            "workloads": audits,
        },
    )
    source_files = {
        "p8_summary": P8_RESULTS / "summary.json",
        "p8_features": P8_RESULTS / "ccac_features.npz",
        "memory": MEMORY,
        "dino": ROOT / CONTEXT_PATHS["dino"],
        "dino_summary": ROOT / CONTEXT_PATHS["dino_summary"],
        "builder": Path(__file__),
        "data_code": ROOT / "src/hac/correspondence_field_data.py",
    }
    receipt = {
        "status": "CORRESPONDENCE_FIELD_OBSERVATION_CACHE_COMPLETE",
        "rows": 4977,
        "pairs_per_row": PAIR_COUNT,
        "maximum_points_per_pair": 32,
        "point_channels": list(POINT_COLUMNS),
        "pair_channels": [
            "dt_seconds",
            "point_count",
            "camera_audit_median_pixels",
            "camera_audit_p90_pixels",
            "camera_usable",
        ],
        "collapsed_channels": 31,
        "context_channels": 2304,
        "valid_pairs": int(pair_valid.sum()),
        "valid_points": int(point_valid.sum()),
        "rows_with_observations": int(pair_valid.any(1).sum()),
        "sample_ids_sha256": canonical_hash(sample_ids.tolist()),
        "context_audit": context_audit,
        "annotation_fields_used_for_inference": [],
        "labels_read": 0,
        "new_pixels_read": 0,
        "source_sha256": {name: file_sha256(path) for name, path in source_files.items()},
        "pinned_context_sha256": CONTEXT_SHA256,
        "output_sha256": {
            "correspondence_field.npz": file_sha256(cache_path),
            "workload_audit.json": file_sha256(audit_path),
        },
        "seconds": time.perf_counter() - started,
    }
    _write_json_exclusive(staging / "receipt.json", receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
