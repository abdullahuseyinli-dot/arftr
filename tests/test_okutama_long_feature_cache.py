from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments import cache_okutama_long_features as cache


def _jpeg(value: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (1280, 720), (value, value, value)).save(stream, format="JPEG")
    return stream.getvalue()


def _rows(sample_id: str, members: list[str], validity: list[bool]):
    return [
        {
            "sample_id": sample_id,
            "time_index": str(index),
            "image_member": member,
            "bbox_xmin": "100",
            "bbox_ymin": "80",
            "bbox_xmax": "180",
            "bbox_ymax": "240",
            "valid_frame": str(int(valid)),
        }
        for index, (member, valid) in enumerate(zip(members, validity, strict=True))
    ]


def test_incomplete_long_clip_reads_no_jpeg_and_returns_zero(tmp_path: Path) -> None:
    archive = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("forbidden.jpg", _jpeg(3))
    sample_id = "sample"
    members = [f"missing/{index}.jpg" for index in range(16)]
    validity = [True] * 15 + [False]
    dataset = cache.LongClipDataset(
        archive,
        [{"sample_id": sample_id}],
        {sample_id: _rows(sample_id, members, validity)},
        set(),
    )
    item = dataset[0]
    assert not item["valid"] and item["decoded_members"] == 0
    assert item["encoded_bytes"] == 0
    assert item["pixels"].shape == (3, 16, 384, 384)
    assert not item["pixels"].any()


def test_complete_long_clip_preserves_frame_order(tmp_path: Path) -> None:
    archive = tmp_path / "frames.zip"
    members = [f"allowed/{index}.jpg" for index in range(16)]
    with zipfile.ZipFile(archive, "w") as stream:
        for index, member in enumerate(members):
            stream.writestr(member, _jpeg(index + 1))
    sample_id = "sample"
    dataset = cache.LongClipDataset(
        archive,
        [{"sample_id": sample_id}],
        {sample_id: _rows(sample_id, members, [True] * 16)},
        set(members),
    )
    item = dataset[0]
    assert item["valid"] and item["decoded_members"] == 16
    assert item["encoded_bytes"] > 0
    assert item["pixels"].shape == (3, 16, 384, 384)
    assert np.isfinite(item["pixels"].numpy()).all()


def test_long_cache_resume_and_completed_hash_binding(tmp_path: Path) -> None:
    request = {"encoder": "vjepa21", "ids": ["a", "b"]}
    arm, shape = "arm", (2, 3)
    features, completed, validity, request_hash = cache.prepare_cache(
        tmp_path, request, 2, arm=arm, shape=shape
    )
    features[0] = 7
    completed[0] = validity[0] = True
    cache._save_state(tmp_path, features, completed, validity)
    del features
    features, completed, validity, _ = cache.prepare_cache(
        tmp_path, request, 2, arm=arm, shape=shape
    )
    assert completed.tolist() == [True, False]
    assert validity.tolist() == [True, False]
    assert (features[0] == 7).all()
    completed[1] = True
    cache._save_state(tmp_path, features, completed, validity)
    summary = {
        "request_sha256": request_hash,
        "arms": {arm: {"sha256": cache.sha256_file(tmp_path / f"{arm}.npy")}},
        "completed": {"sha256": cache.sha256_file(tmp_path / "completed.npy")},
        "validity": {"sha256": cache.sha256_file(tmp_path / "validity.npy")},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    features[0, 0, 0] = 8
    features.flush()
    del features
    with pytest.raises(RuntimeError, match="artifact bytes changed"):
        cache.prepare_cache(tmp_path, request, 2, arm=arm, shape=shape)


def test_changed_request_and_orphan_output_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    values, _, _, _ = cache.prepare_cache(output, {"x": 1}, 1, arm="a", shape=(1,))
    del values
    with pytest.raises(RuntimeError, match="different extraction request"):
        cache.prepare_cache(output, {"x": 2}, 1, arm="a", shape=(1,))
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "user.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        cache.prepare_cache(orphan, {}, 1, arm="a", shape=(1,))
