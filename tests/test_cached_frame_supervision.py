"""Focused safeguards for cached physical-frame supervision metadata."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import hac.cached_frame_supervision as frame_data
from hac.okutama_native_video import Annotation


def _annotation(frame: int, actions: tuple[str, ...], *, lost: bool = False) -> Annotation:
    return Annotation(4, (0, 0, 10, 20), frame, lost, False, False, actions)


def test_frame_targets_masks_missing_lost_unsupported_and_multibase() -> None:
    primary = {
        "sample_ids": np.array(["sample"]),
        "labels": np.array([1], np.int8),
        "recordings": np.array(["r"]),
        "tracks": np.array(["4"]),
    }
    frames = np.arange(16, dtype=np.int32)[None, :]
    frames[0, frame_data.CENTER_INDEX] = 100
    stable = {
        (4, 0): _annotation(0, ("Walking", "Running")),
        (4, 1): _annotation(1, ("Standing", "Sitting")),
        (4, 2): _annotation(2, ("Flying",)),
        (4, 3): _annotation(3, ("Standing",), lost=True),
        (4, 100): _annotation(100, ("Standing",)),
    }
    labels, valid, counts = frame_data.frame_targets(
        primary, frames, np.ones((1, 16), bool), {"r": stable}
    )
    assert labels[0, 0] == 2
    assert labels[0, 1:8].tolist() == [-1] * 7
    assert labels[0, frame_data.CENTER_INDEX] == 1
    assert valid.sum() == 2
    assert counts["missing_lost_unsupported_or_multibase"] == 14


def test_physical_weights_split_joint_multiplicity_and_cap_tracks() -> None:
    n = 10
    recordings = np.array(["r"] * n)
    tracks = np.array(["4"] * n)
    frames = np.zeros((2, n, 16), np.int32)
    valid = np.ones_like(frames, bool)
    # 320 occurrences but only 160 physical frames. Each frame appears once per stream.
    for row in range(n):
        frames[:, row] = np.arange(row * 16, (row + 1) * 16)
    weights = frame_data.physical_multiplicity_weights(recordings, tracks, frames, valid)
    # Multiplicity gives 0.5, then the 160-frame track is scaled to the cap of 128.
    np.testing.assert_allclose(weights, 0.4, rtol=0, atol=1e-7)
    assert float(weights.sum()) == pytest.approx(128.0)
    assert float(weights[:, 0, 0].sum()) == pytest.approx(0.8)


def test_physical_weights_are_zero_for_masked_variants() -> None:
    frames = np.zeros((2, 1, 16), np.int32)
    valid = np.zeros_like(frames, bool)
    valid[0, 0, 8] = True
    valid[1, 0, 8] = True
    weights = frame_data.physical_multiplicity_weights(
        np.array(["r"]), np.array(["4"]), frames, valid
    )
    assert weights[0, 0, 8] == pytest.approx(0.5)
    assert weights[1, 0, 8] == pytest.approx(0.5)
    assert np.count_nonzero(weights) == 2


def test_manifest_reader_enforces_order_identity_and_center(tmp_path: Path) -> None:
    path = tmp_path / "frames.csv"
    fields = (
        "sample_id",
        "time_index",
        "provider_recording_id",
        "provider_track_id",
        "source_frame",
        "valid_frame",
    )
    rows = [
        {
            "sample_id": "sample",
            "time_index": slot,
            "provider_recording_id": "r",
            "provider_track_id": "4",
            "source_frame": 100 + slot - 8,
            "valid_frame": 1,
        }
        for slot in range(16)
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    primary = {
        "sample_ids": np.array(["sample"]),
        "recordings": np.array(["r"]),
        "tracks": np.array(["4"]),
        "center_frames": np.array([100], np.int32),
    }
    frames, valid = frame_data.read_ordered_frame_manifest(path, primary)
    assert frames.shape == valid.shape == (1, 16)
    assert frames[0, 8] == 100 and valid.all()
    rows[3]["time_index"] = 4
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="order"):
        frame_data.read_ordered_frame_manifest(path, primary)


def test_archive_proxy_rejects_pixel_payload() -> None:
    class Archive:
        def read(self, member: str) -> bytes:
            return member.encode()

    proxy = frame_data._AnnotationOnlyArchive(Archive(), {"Labels/allowed.txt"})  # type: ignore[arg-type]
    assert proxy.read("Labels/allowed.txt") == b"Labels/allowed.txt"
    with pytest.raises(RuntimeError, match="non-permitted"):
        proxy.read("Drone1/Morning/frame.jpg")
