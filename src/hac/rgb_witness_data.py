"""Disjoint raw-pixel RGB witness cache contracts.

The cache is label blind.  It stores two independently encoded views from one
exact native center: upper-visible and lower-visible.  Masking happens on the
native uint8 crop before any encoder resize.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hac.actor_memory_base import file_sha256

SLOT_NAMES = ("upper_visible", "lower_visible")
FEATURE_DIM = 768
GEOMETRY_DIM = 6


@dataclass(frozen=True)
class RGBWitnessCache:
    sample_ids: np.ndarray
    features: np.ndarray
    available: np.ndarray
    geometry: np.ndarray
    source_image_sha256: np.ndarray
    shard_receipts: tuple[dict, ...]


def witness_geometry(native_box: tuple[float, float, float, float], crop_box: tuple[int, int, int, int],
                     source_size: tuple[int, int]) -> np.ndarray:
    """The six prespecified inference-available geometry fields."""
    x1, y1, x2, y2 = map(float, native_box)
    left, top, right, bottom = crop_box
    width, height = map(float, source_size)
    actor_w, actor_h = x2 - x1, y2 - y1
    inside_w = max(0.0, min(width, right) - max(0.0, left))
    inside_h = max(0.0, min(height, bottom) - max(0.0, top))
    crop_area = float((right - left) * (bottom - top))
    result = np.asarray([
        (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
        actor_w / width, actor_h / height, np.log(actor_w / actor_h),
        inside_w * inside_h / crop_area,
    ], dtype=np.float32)
    if result.shape != (GEOMETRY_DIM,) or not np.isfinite(result).all():
        raise ValueError("RGB-witness geometry must be six finite fields")
    return result


def load_rgb_witness_cache(directory: Path, expected_ids: np.ndarray) -> RGBWitnessCache:
    """Load and verify a complete immutable shard collection."""
    directory = Path(directory)
    lock_path = directory / "execution_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("RGB-witness extraction lock is missing")
    import json
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    ids = np.asarray(expected_ids).astype(str)
    if lock.get("rows") != len(ids) or tuple(lock.get("slot_names", ())) != SLOT_NAMES:
        raise RuntimeError("RGB-witness lock population/schema mismatch")
    features = np.zeros((len(ids), 2, FEATURE_DIM), dtype=np.float16)
    available = np.zeros((len(ids), 2), dtype=bool)
    geometry = np.zeros((len(ids), GEOMETRY_DIM), dtype=np.float32)
    image_hash = np.full(len(ids), "", dtype="U64")
    seen = np.zeros(len(ids), dtype=np.int8)
    receipts = []
    for receipt_path in sorted(directory.glob("shard-*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        start, stop = int(receipt["start"]), int(receipt["stop"])
        npz_path = directory / f"shard-{start:04d}-{stop:04d}.npz"
        if (receipt.get("status") != "RGB_WITNESS_SHARD_COMPLETE"
                or receipt.get("npz_sha256") != file_sha256(npz_path)):
            raise RuntimeError(f"Invalid RGB-witness shard receipt: {receipt_path}")
        with np.load(npz_path, allow_pickle=False) as saved:
            shard_ids = saved["sample_ids"].astype(str)
            if (not np.array_equal(shard_ids, ids[start:stop])
                    or tuple(saved["slot_names"].astype(str).tolist()) != SLOT_NAMES
                    or saved["features"].shape != (stop-start, 2, FEATURE_DIM)
                    or saved["available"].shape != (stop-start, 2)
                    or saved["geometry"].shape != (stop-start, GEOMETRY_DIM)):
                raise RuntimeError(f"RGB-witness shard schema mismatch: {npz_path}")
            features[start:stop] = saved["features"]
            available[start:stop] = saved["available"]
            geometry[start:stop] = saved["geometry"]
            image_hash[start:stop] = saved["source_image_sha256"].astype(str)
        seen[start:stop] += 1
        receipts.append(receipt)
    if not np.all(seen == 1):
        raise RuntimeError("RGB-witness cache is incomplete or overlapping")
    if not np.isfinite(features).all() or not np.isfinite(geometry).all():
        raise RuntimeError("RGB-witness cache contains nonfinite values")
    return RGBWitnessCache(ids, features, available, geometry, image_hash, tuple(receipts))
