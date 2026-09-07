from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from hac import okutama_native_video as native


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fixture(tmp_path, monkeypatch, *, missing_image=None, missing_annotation=None):
    monkeypatch.setattr(native, "EXPECTED_PRIMARY_ROWS", 1)
    monkeypatch.setattr(native, "EXPECTED_SAFE_ROWS", 1)
    monkeypatch.setattr(native, "EXPECTED_SCENARIOS", {"1.2": ("fold-0", 1)})
    monkeypatch.setattr(native, "EXPECTED_RECORDINGS", {"1.1.2"})
    row = {
        "sample_id": "train__1.1.2__track-5__frame-000030",
        "recording_id": "1.2",
        "provider_recording_id": "1.1.2",
        "provider_track_id": "5",
        "track_id": "1.1.2::5",
        "center_frame": "30",
        "fold": "fold-0",
        "label": "standing",
        "label_index": "1",
        "scope": "grouped_crossfit_oof",
        "development_role": "train",
        "transition_window": "False",
        "window_any_occluded": "False",
    }
    index = tmp_path / "eligible_index.csv"
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    archive_path = tmp_path / "TrainSetFrames.zip"
    frames = [30 + value for value in native.SOURCE_OFFSETS]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for kind, track in (("MultiActionLabels", 99), ("SingleActionTrackingLabels", 5)):
            lines = [
                f'{track} 30 60 330 660 {frame} 0 0 0 "Person" "Standing"'
                for frame in frames
                if frame != missing_annotation
            ]
            archive.writestr(f"Labels/{kind}/3840x2160/1.1.2.txt", "\n".join(lines))
        for frame in frames:
            if frame != missing_image:
                archive.writestr(native.frame_member("1.1.2", frame), b"not a jpeg; never decode")
        archive.writestr("Labels/MultiActionLabels/3840x2160/2.2.6.txt", b"forbidden payload")
    monkeypatch.setattr(native, "EXPECTED_INDEX_SHA256", native.sha256_file(index))
    monkeypatch.setattr(native, "EXPECTED_ARCHIVE_BYTES", archive_path.stat().st_size)
    monkeypatch.setattr(native, "EXPECTED_ARCHIVE_SHA256", native.sha256_file(archive_path))
    lock = {
        "status": native.MATERIALIZATION_STATUS,
        "authorization": {"annotation_manifest": True},
        "sampling": {
            "source_offsets": list(native.SOURCE_OFFSETS),
            "fps": 30.0,
            "center_slot": 8,
            "frame_count": 16,
        },
        "primary_rows": 1,
        "primary_scenarios": 1,
        "inputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": native.sha256_file(path),
            }
            for name, path in (("eligible_index", index), ("archive", archive_path))
        },
    }
    lock_path = tmp_path / "materialization_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return dict(
        eligible_index=index,
        archive_path=archive_path,
        output_dir=tmp_path / "manifest",
        source_lock=lock,
        source_lock_path=lock_path,
    )


def test_manifest_reads_only_permitted_annotations_and_preserves_order(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    original = zipfile.ZipFile.read
    read_members = []

    def guarded_read(self, name, *values, **kwargs):
        assert name in native.annotation_members("1.1.2")
        read_members.append(name)
        return original(self, name, *values, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    result = native.build_native_manifest(**args)
    frames = _read_csv(args["output_dir"] / "frame_manifest.csv")
    assert [int(row["source_frame"]) for row in frames] == list(range(22, 38))
    assert [int(row["time_index"]) for row in frames] == list(range(16))
    assert float(frames[8]["offset_seconds"]) == 0
    assert float(frames[8]["nominal_time_seconds"]) == 1
    assert [
        float(frames[0][name]) for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
    ] == [10, 20, 110, 220]
    assert len(read_members) == 2
    assert result["primary_rows"] == 1
    assert result["valid_frame_rows"] == 16
    assert result["access_accounting"]["image_payloads_read"] == 0
    assert result["access_accounting"]["image_payloads_decoded"] == 0
    assert len(_read_csv(args["output_dir"] / "image_allowlist.csv")) == 16


def test_missing_observations_keep_center_and_slots(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch, missing_image=24, missing_annotation=25)
    result = native.build_native_manifest(**args)
    frames = _read_csv(args["output_dir"] / "frame_manifest.csv")
    clips = _read_csv(args["output_dir"] / "clip_index.csv")
    assert len(frames) == 16 and len(clips) == 1
    assert frames[2]["missing_reason"] == "missing_image_member"
    assert frames[3]["missing_reason"] == "missing_stable_annotation"
    assert frames[3]["bbox_xmin"] == ""
    assert clips[0]["valid_frame_count"] == "14"
    assert result["primary_rows"] == 1
    assert result["unique_present_image_members"] == 15


@pytest.mark.parametrize("change", ["sampling", "authorization", "cohort", "hash", "path"])
def test_invalid_lock_fails_before_annotation_read(tmp_path, monkeypatch, change):
    args = _fixture(tmp_path, monkeypatch)
    lock = args["source_lock"]
    if change == "sampling":
        lock["sampling"]["source_offsets"][0] = -7
    elif change == "authorization":
        lock["authorization"]["annotation_manifest"] = False
    elif change == "cohort":
        lock["primary_rows"] = 2
    elif change == "hash":
        lock["inputs"]["archive"]["sha256"] = "0" * 64
    else:
        lock["inputs"]["eligible_index"]["path"] = str(tmp_path / "different.csv")

    def forbidden_read(*args, **kwargs):
        pytest.fail("Annotation payload was read before lock validation")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden_read)
    with pytest.raises(RuntimeError):
        native.build_native_manifest(**args)
    assert not args["output_dir"].exists()


def test_fold_mismatch_fails_with_current_file_hash(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    path = args["eligible_index"]
    path.write_text(path.read_text(encoding="utf-8").replace("fold-0", "fold-1"), encoding="utf-8")
    monkeypatch.setattr(native, "EXPECTED_INDEX_SHA256", native.sha256_file(path))
    with pytest.raises(RuntimeError, match="fold"):
        native.load_primary_index(path)


@pytest.mark.parametrize(
    "name",
    [
        "development_metadata.csv",
        "DEVELOPMENT_MANIFEST.CSV",
        "development_centres.csv",
        "TestSetFrames.zip",
    ],
)
def test_forbidden_mixed_paths_are_rejected_without_open(tmp_path, name):
    with pytest.raises(RuntimeError, match="forbidden"):
        native.load_primary_index(tmp_path / name)


def test_crop_roundtrip_and_camera_motion_with_crop_jitter():
    world = np.asarray([120.0, 75.0, 1.0])
    for dx, dy, crop_box in (
        (0, 0, (90, 30, 150, 130)),
        (50, -10, (125, 15, 220, 150)),
        (-110, 0, (-25, 20, 50, 160)),
    ):
        geometry = native.crop_geometry(crop_box, output_size=224)
        camera = np.asarray([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)
        observed_crop = geometry.image_to_crop @ camera @ world
        image_point = np.linalg.inv(geometry.image_to_crop) @ observed_crop
        corrected = np.linalg.inv(camera) @ image_point
        np.testing.assert_allclose(corrected, world, atol=1e-12)
        assert max(geometry.resized_size) == 224


def test_invalid_crop_and_recording_are_rejected():
    with pytest.raises(ValueError):
        native.crop_geometry((float("nan"), 0, 30, 40), output_size=224)
    with pytest.raises(ValueError):
        native.frame_member("2.2.6", 30)


def test_crop_distinguishes_pixel_edges_and_pixel_center_indices():
    crop = native.crop_geometry((0, 0, 10, 10), output_size=20, context_fraction=0)
    np.testing.assert_allclose(crop.image_to_crop @ [0, 0, 1], [0, 0, 1])
    np.testing.assert_allclose(crop.pixel_center_image_to_crop @ [0, 0, 1], [0.5, 0.5, 1])


def test_completed_manifest_does_not_overwrite_results(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    native.build_native_manifest(**args)
    before = native.sha256_file(args["output_dir"] / "summary.json")
    with pytest.raises(FileExistsError):
        native.build_native_manifest(**args)
    assert native.sha256_file(args["output_dir"] / "summary.json") == before
