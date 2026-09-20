"""Strict loader for the complete label-blind source/posture descriptor cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hac.actor_memory_base import file_sha256


@dataclass(frozen=True)
class SourcePostureCache:
    sample_ids: np.ndarray
    crop_keys: tuple[str, ...]
    features: np.ndarray
    available: np.ndarray
    receipt_paths: tuple[Path, ...]

    def descriptor(self, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        indices = []
        for key in keys:
            try:
                indices.append(self.crop_keys.index(key))
            except ValueError as error:
                raise KeyError(f"Unknown source/posture crop key: {key}") from error
        values = self.features[:, indices].astype(np.float64).reshape(len(self.sample_ids), -1)
        observed = self.available[:, indices].all(1)
        if not np.isfinite(values).all():
            raise RuntimeError("Source/posture descriptor contains nonfinite values")
        return values, observed


def load_source_posture_cache(
    directory: Path,
    canonical_ids: np.ndarray,
    *,
    expected_crop_keys: tuple[str, ...],
) -> SourcePostureCache:
    root = Path(directory).resolve()
    ids = np.asarray(canonical_ids)
    lock_path = root / "execution_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "SOURCE_POSTURE_LABEL_BLIND_EXTRACTION_LOCK"
        or lock.get("rows") != len(ids)
        or tuple(lock.get("crop_keys", ())) != expected_crop_keys
        or len(np.unique(ids)) != len(ids)
    ):
        raise RuntimeError("Source/posture extraction lock differs from the required cohort")
    features = np.zeros((len(ids), len(expected_crop_keys), 768), dtype=np.float16)
    available = np.zeros((len(ids), len(expected_crop_keys)), dtype=bool)
    coverage = np.zeros(len(ids), dtype=np.uint8)
    receipts = []
    for receipt_path in sorted(root.glob("shard-*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        npz_path = receipt_path.with_suffix(".npz")
        start, stop = receipt.get("start"), receipt.get("stop")
        if (
            receipt.get("status") != "SOURCE_POSTURE_SHARD_COMPLETE"
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start < stop <= len(ids)
            or receipt.get("rows") != stop - start
            or not npz_path.is_file()
            or receipt.get("npz_sha256") != file_sha256(npz_path)
        ):
            raise RuntimeError(f"Malformed source/posture shard receipt: {receipt_path}")
        with np.load(npz_path, allow_pickle=False) as saved:
            shard_ids = saved["sample_ids"]
            shard_keys = saved["crop_keys"]
            shard_features = saved["features"]
            shard_available = saved["available"]
        if (
            not np.array_equal(shard_ids, ids[start:stop])
            or tuple(shard_keys.tolist()) != expected_crop_keys
            or shard_features.shape != (stop - start, len(expected_crop_keys), 768)
            or shard_features.dtype != np.float16
            or shard_available.shape != shard_features.shape[:2]
            or shard_available.dtype != np.bool_
            or not np.isfinite(shard_features).all()
            or np.any(~shard_available & (shard_features != 0).any(2))
            or np.any(coverage[start:stop])
        ):
            raise RuntimeError(f"Source/posture shard violates identity or schema: {npz_path}")
        norms = np.linalg.norm(shard_features.astype(np.float32), axis=-1)
        if np.any(shard_available & ((norms < 0.999) | (norms > 1.001))):
            raise RuntimeError(f"Source/posture shard violates L2 contract: {npz_path}")
        features[start:stop] = shard_features
        available[start:stop] = shard_available
        coverage[start:stop] += 1
        receipts.append(receipt_path)
    if not np.all(coverage == 1):
        raise RuntimeError("Source/posture cache does not cover every canonical row exactly once")
    return SourcePostureCache(
        sample_ids=ids.copy(),
        crop_keys=expected_crop_keys,
        features=features,
        available=available,
        receipt_paths=tuple(receipts),
    )
