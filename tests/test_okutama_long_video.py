from __future__ import annotations

import copy
import csv
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hac import okutama_long_video as long
from hac import okutama_native_video as native

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def manifest_case(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
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
    index = root / "eligible_index.csv"
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    folds = root / "fold_map.csv"
    folds.write_text(
        f"bundle_row_index,sample_id,recording_id,fold\n0,{row['sample_id']},1.2,fold-0\n",
        encoding="utf-8",
    )
    archive_path = root / "TrainSetFrames.zip"
    frames = [30 + offset for offset in long.SOURCE_OFFSETS if 30 + offset >= 0]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for kind, track in (("MultiActionLabels", 99), ("SingleActionTrackingLabels", 5)):
            # Deliberately different actions on either side, including at the center:
            # immutable input labels are not used for source-window selection.
            lines = [
                f'{track} 30 60 330 660 {frame} 0 0 0 "Person" "'
                f'{"Sitting" if frame < 30 else "Running"}"'
                for frame in frames
            ]
            archive.writestr(f"Labels/{kind}/3840x2160/1.1.2.txt", "\n".join(lines))
        for frame in frames:
            archive.writestr(native.frame_member("1.1.2", frame), b"not-a-jpeg; never decoded")
        archive.writestr("Labels/MultiActionLabels/3840x2160/2.2.6.txt", "protected")
        archive.writestr(native.frame_member("1.1.2", 200), "outside requested times")
    monkeypatch.setattr(native, "EXPECTED_INDEX_SHA256", native.sha256_file(index))
    monkeypatch.setattr(native, "EXPECTED_ARCHIVE_BYTES", archive_path.stat().st_size)
    monkeypatch.setattr(native, "EXPECTED_ARCHIVE_SHA256", native.sha256_file(archive_path))
    monkeypatch.setattr(long, "EXPECTED_FOLD_SHA256", native.sha256_file(folds))
    spec = json.loads((ROOT / long.PROTOCOL_PATH).read_text(encoding="utf-8"))
    spec["data_contract"]["primary_tracks"] = 1
    spec["historical_inputs"] = {
        name: {"path": path.name, "sha256": native.sha256_file(path)}
        for name, path in (("eligible_index", index), ("fold_map", folds))
    }
    monkeypatch.setattr(long, "_read_protocol", lambda _: copy.deepcopy(spec))
    monkeypatch.setattr(
        long,
        "_repository_state",
        lambda _: {"repository_commit": "fixture-commit", "repository_tree": "fixture-tree"},
    )
    source = root / "source.py"
    source.write_text("fixture committed source\n", encoding="utf-8")
    monkeypatch.setattr(long, "_source_receipts", lambda _: {"fixture": long._receipt(source)})
    lock_path = root / ".runs" / "p3_lock.json"
    long.create_materialization_lock(root, archive_path, lock_path)
    return SimpleNamespace(
        root=root,
        index=index,
        folds=folds,
        archive=archive_path,
        lock=lock_path,
        output=root / ".runs" / "manifest",
        source=source,
        row=row,
        spec=spec,
    )


def test_long_protocol_is_pinned_and_copies_p0_preprocessing():
    spec = long._read_protocol(ROOT)
    p0 = json.loads((ROOT / "experiments/okutama_video_protocol.json").read_text(encoding="utf-8"))
    assert spec["preprocessing"] == p0["preprocessing"]
    assert spec["matched_controls_prospective"]["arms"][2] == "dinov2_long16_native_frames"
    assert spec["materialization_lock"]["authorization"] == long.AUTHORIZATION
    assert len(long.annotation_allowlist()) == 42
    assert len(set(long.annotation_allowlist())) == 42
    assert (long.SOURCE_OFFSETS[-1] - long.SOURCE_OFFSETS[0]) / long.SOURCE_FPS == 2.0
    assert long.SOURCE_OFFSETS[long.CENTER_SLOT] == 0
    assert native.SOURCE_OFFSETS == tuple(range(-8, 8))


def test_manifest_retains_negative_slot_and_only_reads_allowed_annotations(
    manifest_case, monkeypatch
):
    case = manifest_case
    original = zipfile.ZipFile.read
    accessed = []

    def guarded_read(self, member, *args, **kwargs):
        assert member in native.annotation_members("1.1.2")
        accessed.append(member)
        return original(self, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    summary = long.build_long_manifest(
        source_lock_path=case.lock, output_dir=case.output, root=case.root
    )
    frames = read_csv(case.output / "frame_manifest.csv")
    clips = read_csv(case.output / "clip_index.csv")
    images = read_csv(case.output / "image_allowlist.csv")
    assert [int(row["source_frame"]) for row in frames] == list(range(-2, 59, 4))
    assert [int(row["time_index"]) for row in frames] == list(range(16))
    assert float(frames[-1]["offset_seconds"]) - float(frames[0]["offset_seconds"]) == 2.0
    assert frames[0]["image_member"] == ""
    assert frames[0]["source_frame_nonnegative"] == "0"
    assert frames[0]["valid_frame"] == "0"
    assert "source_frame_before_start" in frames[0]["missing_reason"]
    assert float(frames[8]["offset_seconds"]) == 0
    assert float(frames[8]["nominal_time_seconds"]) == 1
    assert [
        float(frames[8][key]) for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
    ] == [10, 20, 110, 220]
    assert clips[0]["sample_id"] == case.row["sample_id"]
    assert clips[0]["label"] == "standing"  # Source action differences never rewrite center labels.
    assert clips[0]["valid_frame_count"] == "15"
    assert len(accessed) == 2 and len(images) == 15
    assert all(int(row["source_frame"]) in range(2, 59, 4) for row in images)
    assert summary["negative_requested_frame_rows"] == 1
    assert summary["access_accounting"]["image_payloads_read"] == 0
    assert summary["access_accounting"]["model_fits"] == 0
    assert summary["access_accounting"]["protected_payloads_read"] == 0
    assert summary["access_accounting"]["annotation_action_values_used_for_sampling"] == 0


def test_missing_track_and_image_are_never_filled_from_other_tracks():
    row = {"sample_id": "train__1.1.2__track-5__frame-000060"}
    stable = {
        (5, frame): native.Annotation(
            5, (30, 60, 330, 660), frame, False, False, False, ("Standing",)
        )
        for frame in range(28, 89, 4)
        if frame != 32
    }
    stable[(6, 32)] = native.Annotation(
        6, (30, 60, 330, 660), 32, False, False, False, ("Standing",)
    )
    queried = []

    def getinfo(member):
        queried.append(member)
        if member.endswith("/36.jpg"):
            raise KeyError(member)
        return SimpleNamespace(is_dir=lambda: False, file_size=10, CRC=42)

    images = {}
    frames = long.long_frame_rows(row, stable, SimpleNamespace(getinfo=getinfo), images)
    assert len(frames) == 16
    assert frames[1]["missing_reason"] == "missing_stable_annotation"
    assert frames[1]["bbox_xmin"] == ""
    assert frames[2]["missing_reason"] == "missing_image_member"
    assert frames[8]["valid_frame"] == 1
    assert len(queried) == 16
    assert len(images) == 15


def test_lock_declaration_never_opens_zip_or_authorizes_extraction(manifest_case, monkeypatch):
    case = manifest_case
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *a, **k: pytest.fail("ZIP opened while declaring lock")
    )
    second = case.root / ".runs" / "another_lock.json"
    payload = long.create_materialization_lock(case.root, case.archive, second)
    assert payload["status"] == long.MATERIALIZATION_STATUS
    assert payload["authorization"] == long.AUTHORIZATION
    assert native.sha256_file(second) == native.sha256_file(case.lock)
    assert long.validate_materialization_lock(case.root, second) == payload


@pytest.mark.parametrize(
    "change", ["status", "authorization", "sampling", "hash", "fold", "source", "archive"]
)
def test_tamper_fails_before_annotation_access(manifest_case, monkeypatch, change):
    case = manifest_case
    payload = json.loads(case.lock.read_text(encoding="utf-8"))
    if change == "status":
        payload["status"] = native.MATERIALIZATION_STATUS
    elif change == "authorization":
        payload["authorization"]["frozen_extraction"] = True
    elif change == "sampling":
        payload["sampling"]["source_offsets"][0] = -31
    elif change == "hash":
        payload["inputs"]["archive"]["sha256"] = "0" * 64
    elif change == "fold":
        case.folds.write_text(case.folds.read_text().replace("fold-0", "fold-1"), encoding="utf-8")
    elif change == "source":
        case.source.write_text("changed source\n", encoding="utf-8")
    else:
        with case.archive.open("ab") as stream:
            stream.write(b"changed archive")
    # Re-sign mutated lock content: current-live-receipt checks must also catch it.
    payload["lock_sha256"] = long.canonical_digest(
        {k: v for k, v in payload.items() if k != "lock_sha256"}
    )
    case.lock.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *a, **k: pytest.fail("ZIP opened before lock validation")
    )
    with pytest.raises(RuntimeError):
        long.build_long_manifest(source_lock_path=case.lock, output_dir=case.output, root=case.root)
    assert not case.output.exists()


def test_missing_source_lock_fails_before_annotation_access(manifest_case, monkeypatch):
    case = manifest_case
    monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **k: pytest.fail("ZIP opened without lock"))
    with pytest.raises(FileNotFoundError):
        long.build_long_manifest(
            source_lock_path=case.root / "missing.json", output_dir=case.output, root=case.root
        )


def test_same_inputs_produce_identical_artifact_hashes_without_overwriting(manifest_case):
    case = manifest_case
    first = long.build_long_manifest(
        source_lock_path=case.lock, output_dir=case.output, root=case.root
    )
    second_dir = case.root / ".runs" / "second_manifest"
    second = long.build_long_manifest(
        source_lock_path=case.lock, output_dir=second_dir, root=case.root
    )
    assert first == second
    assert native.sha256_file(case.output / "summary.json") == native.sha256_file(
        second_dir / "summary.json"
    )
    with pytest.raises(FileExistsError):
        long.build_long_manifest(source_lock_path=case.lock, output_dir=case.output, root=case.root)
    with pytest.raises(FileExistsError):
        long.create_materialization_lock(case.root, case.archive, case.lock)


def test_pilot_order_is_independent_of_labels_and_input_order():
    rows = [{"sample_id": f"sample-{index}", "label": "standing"} for index in range(200)]
    spec = {"pilot": {"pilot_domain": "okutama-long-video-p3-20260907:", "rows": 128}}
    first = long.pilot_sample_ids(rows, spec)
    changed = [{"sample_id": row["sample_id"], "label": "sitting"} for row in reversed(rows)]
    assert first == long.pilot_sample_ids(changed, spec)
    assert len(first) == len(set(first)) == 128


@pytest.mark.parametrize(
    "name",
    [
        "development_metadata.csv",
        "TestSetFrames.zip",
        "test/payload.json",
        "confirmation/source.json",
    ],
)
def test_protected_paths_rejected_before_open(tmp_path, name):
    with pytest.raises(RuntimeError, match="forbidden"):
        long._safe_path(tmp_path / name)


def test_path_escape_and_dirty_repository_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="inside"):
        long._safe_path(tmp_path.parent / "outside.json", root=tmp_path)
    monkeypatch.setattr(long, "_git", lambda *_: " M source.py")
    with pytest.raises(RuntimeError, match="clean committed"):
        long._repository_state(tmp_path)


def test_source_blob_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(
        long,
        "_git",
        lambda _root, command, *_: "committed" if command == "rev-parse" else "changed",
    )
    with pytest.raises(RuntimeError, match="committed blob"):
        long._source_receipts(ROOT)


def test_duplicate_requested_jpeg_fails_before_annotation_read(manifest_case, monkeypatch):
    case = manifest_case
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(case.archive, "a") as archive:
            archive.writestr(native.frame_member("1.1.2", 30), b"duplicate")
    monkeypatch.setattr(native, "EXPECTED_ARCHIVE_BYTES", case.archive.stat().st_size)
    monkeypatch.setattr(native, "EXPECTED_ARCHIVE_SHA256", native.sha256_file(case.archive))
    new_lock = case.root / ".runs" / "duplicate_lock.json"
    long.create_materialization_lock(case.root, case.archive, new_lock)
    monkeypatch.setattr(
        zipfile.ZipFile,
        "read",
        lambda *a, **k: pytest.fail("Read annotations before duplicate check"),
    )
    with pytest.raises(RuntimeError, match="Duplicate requested JPEG"):
        long.build_long_manifest(source_lock_path=new_lock, output_dir=case.output, root=case.root)


def test_fold_alignment_rejects_even_rehashed_wrong_fold(manifest_case, monkeypatch):
    case = manifest_case
    case.folds.write_text(case.folds.read_text().replace("fold-0", "fold-1"), encoding="utf-8")
    monkeypatch.setattr(long, "EXPECTED_FOLD_SHA256", native.sha256_file(case.folds))
    with pytest.raises(RuntimeError, match="alignment"):
        long._read_primary(case.index, case.folds, case.spec)
