from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.cache_okutama_video_features import (
    LockedClipDataset,
    _crop_with_mean_padding,
    _prepare_cache,
    _save_state,
)
from hac.video_encoders import sha256_file


def _jpeg(value: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1280, 720), (value, 2 * value, 3 * value)).save(output, format="JPEG")
    return output.getvalue()


def test_locked_dataset_reads_only_declared_members_and_builds_controls(tmp_path: Path) -> None:
    archive_path = tmp_path / "frames.zip"
    members = [f"allowed/{index}.jpg" for index in range(16)]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index, member in enumerate(members):
            archive.writestr(member, _jpeg(index + 1))
        archive.writestr("forbidden.jpg", _jpeg(200))
    sample_id = "sample"
    rows = [
        {
            "sample_id": sample_id,
            "time_index": str(index),
            "image_member": member,
            "bbox_xmin": "100",
            "bbox_ymin": "80",
            "bbox_xmax": "180",
            "bbox_ymax": "240",
            "valid_frame": "1",
        }
        for index, member in enumerate(members)
    ]
    dataset = LockedClipDataset(
        archive_path,
        [{"sample_id": sample_id}],
        {sample_id: rows},
        set(members),
    )
    item = dataset[0]
    assert item["actual_pixels"].shape == (3, 16, 384, 384)
    assert item["repeated_pixels"].shape == (3, 16, 384, 384)
    assert bool(item["actual_valid"])
    assert bool(item["repeated_valid"])
    assert np.isfinite(item["actual_pixels"].numpy()).all()
    center = item["repeated_pixels"][:, 8]
    assert all((item["repeated_pixels"][:, index] == center).all() for index in range(16))


def test_mean_padded_border_crop_has_declared_size() -> None:
    image = Image.new("RGB", (1280, 720), (10, 20, 30))
    crop = _crop_with_mean_padding(image, (0.0, 0.0, 30.0, 60.0))
    assert crop.size == (46, 90)
    assert crop.getpixel((0, 0)) == (124, 116, 104)
    assert crop.getpixel((8, 15)) == (10, 20, 30)


def test_video_cache_resume_binds_request_shapes_and_completed_bytes(tmp_path: Path) -> None:
    request = {"sample_ids": ["a", "b"], "precision": "bfloat16"}
    arrays, completed, validity, request_hash = _prepare_cache(tmp_path, request, 2)
    arrays["vjepa21_real_clip"][0] = 3
    arrays["vjepa21_repeated_center"][0] = 5
    completed[0] = True
    validity[0] = True
    _save_state(tmp_path, arrays, completed, validity)
    del arrays
    arrays, completed, validity, _ = _prepare_cache(tmp_path, request, 2)
    assert completed.tolist() == [True, False]
    assert validity.tolist() == [[True, True], [False, False]]
    assert (arrays["vjepa21_real_clip"][0] == 3).all()
    completed[1] = True
    _save_state(tmp_path, arrays, completed, validity)
    summary = {
        "request_sha256": request_hash,
        "arms": {name: {"sha256": sha256_file(tmp_path / f"{name}.npy")} for name in arrays},
        "completed": {"sha256": sha256_file(tmp_path / "completed.npy")},
        "validity": {"sha256": sha256_file(tmp_path / "validity.npy")},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    arrays["vjepa21_real_clip"][0, 0, 0, 0] = 4
    arrays["vjepa21_real_clip"].flush()
    del arrays
    with pytest.raises(RuntimeError, match="artifact bytes changed"):
        _prepare_cache(tmp_path, request, 2)


def test_video_cache_rejects_changed_request_and_orphan_output(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    arrays, _, _, _ = _prepare_cache(output, {"one": 1}, 1)
    del arrays
    with pytest.raises(RuntimeError, match="different extraction request"):
        _prepare_cache(output, {"one": 2}, 1)
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "retain.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        _prepare_cache(orphan, {"one": 1}, 1)
