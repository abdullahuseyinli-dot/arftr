from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hac.actor_memory_base import file_sha256
from hac.source_posture_cache import load_source_posture_cache


def _write_shard(
    root: Path,
    sample_ids: np.ndarray,
    crop_keys: tuple[str, ...],
    start: int,
    stop: int,
) -> None:
    features = np.zeros((stop - start, len(crop_keys), 768), dtype=np.float16)
    features[..., 0] = 1
    available = np.ones(features.shape[:2], dtype=bool)
    path = root / f"shard-{start:04d}-{stop:04d}.npz"
    np.savez_compressed(
        path,
        sample_ids=sample_ids[start:stop],
        crop_keys=np.asarray(crop_keys),
        features=features,
        available=available,
    )
    (root / f"shard-{start:04d}-{stop:04d}.json").write_text(
        json.dumps(
            {
                "status": "SOURCE_POSTURE_SHARD_COMPLETE",
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "npz_sha256": file_sha256(path),
            }
        ),
        encoding="utf-8",
    )


def _cache(tmp_path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    sample_ids = np.asarray(("a", "b", "c"))
    crop_keys = ("J_whole", "R_whole")
    (tmp_path / "execution_lock.json").write_text(
        json.dumps(
            {
                "status": "SOURCE_POSTURE_LABEL_BLIND_EXTRACTION_LOCK",
                "rows": len(sample_ids),
                "crop_keys": list(crop_keys),
            }
        ),
        encoding="utf-8",
    )
    return sample_ids, crop_keys


def test_complete_cache_loads_in_canonical_order(tmp_path: Path) -> None:
    sample_ids, crop_keys = _cache(tmp_path)
    _write_shard(tmp_path, sample_ids, crop_keys, 0, 2)
    _write_shard(tmp_path, sample_ids, crop_keys, 2, 3)

    cache = load_source_posture_cache(
        tmp_path, sample_ids, expected_crop_keys=crop_keys
    )
    descriptor, available = cache.descriptor(("R_whole",))

    assert np.array_equal(cache.sample_ids, sample_ids)
    assert descriptor.shape == (3, 768)
    assert np.all(descriptor[:, 0] == 1)
    assert available.all()
    assert len(cache.receipt_paths) == 2


def test_cache_rejects_missing_canonical_coverage(tmp_path: Path) -> None:
    sample_ids, crop_keys = _cache(tmp_path)
    _write_shard(tmp_path, sample_ids, crop_keys, 0, 2)

    with pytest.raises(RuntimeError, match="cover every canonical row"):
        load_source_posture_cache(
            tmp_path, sample_ids, expected_crop_keys=crop_keys
        )


def test_cache_rejects_shard_identity_mismatch(tmp_path: Path) -> None:
    sample_ids, crop_keys = _cache(tmp_path)
    wrong = sample_ids.copy()
    wrong[1] = "z"
    _write_shard(tmp_path, wrong, crop_keys, 0, 3)

    with pytest.raises(RuntimeError, match="identity or schema"):
        load_source_posture_cache(
            tmp_path, sample_ids, expected_crop_keys=crop_keys
        )
