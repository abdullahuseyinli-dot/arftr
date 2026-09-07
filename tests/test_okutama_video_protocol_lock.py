from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from tools import lock_okutama_video_protocol as locks

ROOT = Path(__file__).resolve().parents[1]


def spec() -> dict:
    return locks.load_protocol(ROOT)


def write_csv(path: Path, rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue().encode()
    path.write_bytes(content)
    return locks.digest(content)


@pytest.fixture
def manifest(tmp_path: Path) -> tuple[Path, list[dict], dict]:
    directory = tmp_path / ".runs" / "native"
    directory.mkdir(parents=True)
    primary = [
        {
            "sample_id": "train__1.1.2__track-1__frame-000030",
            "recording_id": "1.2",
            "provider_recording_id": "1.1.2",
            "provider_track_id": "1",
            "fold": "fold-2",
            "center_frame": "30",
            "label_index": "1",
            "label": "standing",
        }
    ]
    clips = [{**primary[0], "valid_frame_count": "16"}]
    frames, allowlist = [], []
    for slot, offset in enumerate(range(-8, 8)):
        number = 30 + offset
        member = locks.frame_member("1.1.2", number)
        frames.append(
            {
                "sample_id": primary[0]["sample_id"],
                "time_index": slot,
                "provider_recording_id": "1.1.2",
                "provider_track_id": "1",
                "source_frame": number,
                "nominal_time_seconds": number / 30,
                "offset_seconds": offset / 30,
                "image_member": member,
                "image_width": "1280",
                "image_height": "720",
                "bbox_xmin": "100",
                "bbox_ymin": "100",
                "bbox_xmax": "200",
                "bbox_ymax": "300",
                "valid_geometry": "1",
                "valid_frame": "1",
                "image_present": "1",
            }
        )
        allowlist.append(
            {
                "image_member": member,
                "provider_recording_id": "1.1.2",
                "source_frame": number,
                "crc32": 100,
                "size_bytes": 500,
            }
        )
    summary = {
        "status": "OKUTAMA_VIDEO_P0_NATIVE_CLIP_MANIFEST_COMPLETE",
        "source_sha256": {"materialization_lock": "a" * 64},
        "access_accounting": {
            "image_payloads_read": 0,
            "protected_payloads_read": 0,
            "model_fits": 0,
        },
        "artifact_sha256": {
            name: write_csv(directory / name, rows)
            for name, rows in (
                ("clip_index.csv", clips),
                ("frame_manifest.csv", frames),
                ("image_allowlist.csv", allowlist),
            )
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory, primary, summary


def alter_csv(directory: Path, summary: dict, name: str, mutate) -> None:
    rows = locks.csv_rows((directory / name).read_bytes())
    mutate(rows)
    summary["artifact_sha256"][name] = write_csv(directory / name, rows)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_protocol_separates_manifest_extraction_and_future_fitting() -> None:
    protocol = spec()
    stages = protocol["lock_stages"]
    assert stages["materialization"]["authorization"]["annotation_manifest"]
    assert not stages["materialization"]["authorization"]["frozen_extraction"]
    assert stages["extraction"]["authorization"]["frozen_extraction"]
    assert all(not value["authorization"]["model_fitting"] for value in stages.values())
    assert protocol["upstream"]["checkpoint_size_bytes"] == 1664223428
    assert protocol["encoder_cache"]["cached_shape_per_clip"] == [8, 9, 768]
    assert protocol["image_control"]["revision"] == ("f9e44c814b77203eaa57a6bdbbd535f21ede1415")
    assert protocol["image_control_cache"]["cached_shape_per_clip"] == [16, 1, 768]


def test_image_control_snapshot_receipt_binds_exact_files(tmp_path: Path) -> None:
    files = {"config.json": b"config", "model.safetensors": b"weights"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    contract = {
        "revision": "pinned-revision",
        "snapshot_files": {
            name: {"size_bytes": len(content), "sha256": locks.digest(content)}
            for name, content in files.items()
        },
    }
    receipt = locks.snapshot_receipt(tmp_path, contract)
    assert receipt["revision"] == "pinned-revision"
    assert set(receipt["files"]) == set(files)
    assert len(receipt["sha256"]) == 64

    (tmp_path / "config.json").write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="byte count|SHA256"):
        locks.snapshot_receipt(tmp_path, contract)


@pytest.mark.parametrize(
    "relative",
    [
        ".runs/calibration/rows.csv",
        ".runs/test/rows.csv",
        ".runs/confirmation/rows.csv",
        ".runs/safe/development_metadata.csv",
        ".runs/safe/development_manifest.csv",
        ".runs/safe/test_manifest.csv",
    ],
)
def test_protected_paths_are_rejected_before_reads(
    tmp_path: Path, monkeypatch, relative: str
) -> None:
    def forbidden_read(_path):
        pytest.fail("Attempted a forbidden file read")

    monkeypatch.setattr(locks.common, "_read_bytes", forbidden_read)
    with pytest.raises(RuntimeError, match="Protected|Forbidden"):
        locks.checked_bytes(tmp_path, tmp_path / relative, "a" * 64)


def test_dirty_repository_fails_before_any_input_read(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(locks.common, "_git", lambda *_args: " M tracked.py")
    monkeypatch.setattr(
        locks, "load_protocol", lambda *_args: pytest.fail("Read before clean check")
    )
    with pytest.raises(RuntimeError, match="clean committed"):
        locks.build_materialization_payload(
            tmp_path, tmp_path / "archive", tmp_path / "upstream", tmp_path / "checkpoint"
        )


def test_manifest_exact_identity_and_allowlist_pass(tmp_path: Path, manifest) -> None:
    directory, primary, _ = manifest
    result = locks.validate_manifest_artifacts(tmp_path, directory, primary, "a" * 64, spec())
    assert result["allowlist_rows"] == 16
    assert result["present_image_members_sha256"] == result["requested_image_members_sha256"]


def test_artifact_bytes_fail_before_csv_decoding(tmp_path: Path, manifest, monkeypatch) -> None:
    directory, primary, _ = manifest
    (directory / "clip_index.csv").write_bytes(b"unexpected")
    monkeypatch.setattr(locks, "csv_rows", lambda *_args: pytest.fail("Decoded changed bytes"))
    with pytest.raises(RuntimeError, match="SHA256 mismatch before decoding"):
        locks.validate_manifest_artifacts(tmp_path, directory, primary, "a" * 64, spec())


@pytest.mark.parametrize(
    "field,value", [("label_index", "2"), ("fold", "fold-3"), ("sample_id", "different")]
)
def test_rehashed_identity_tampering_is_rejected(
    tmp_path: Path, manifest, field: str, value: str
) -> None:
    directory, primary, summary = manifest
    alter_csv(directory, summary, "clip_index.csv", lambda rows: rows[0].update({field: value}))
    with pytest.raises(RuntimeError, match="identity/order/label/fold"):
        locks.validate_manifest_artifacts(tmp_path, directory, primary, "a" * 64, spec())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("image_member", "../protected/image.jpg", "source member"),
        ("nominal_time_seconds", "nan", "timestamp"),
        ("offset_seconds", "0.5", "timestamp"),
        ("image_present", "0", "validity"),
        ("bbox_xmax", "50", "crop geometry"),
    ],
)
def test_rehashed_frame_tampering_is_rejected(
    tmp_path: Path, manifest, field: str, value: str, match: str
) -> None:
    directory, primary, summary = manifest
    alter_csv(directory, summary, "frame_manifest.csv", lambda rows: rows[0].update({field: value}))
    with pytest.raises(RuntimeError, match=match):
        locks.validate_manifest_artifacts(tmp_path, directory, primary, "a" * 64, spec())


def test_duplicate_allowlist_is_rejected(tmp_path: Path, manifest) -> None:
    directory, primary, summary = manifest
    alter_csv(directory, summary, "image_allowlist.csv", lambda rows: rows.append(rows[0]))
    with pytest.raises(RuntimeError, match="Image allowlist"):
        locks.validate_manifest_artifacts(tmp_path, directory, primary, "a" * 64, spec())


def test_manifest_cannot_change_source_lock_or_hide_access(tmp_path: Path, manifest) -> None:
    directory, primary, summary = manifest
    with pytest.raises(RuntimeError, match="not bound"):
        locks.validate_manifest_artifacts(tmp_path, directory, primary, "b" * 64, spec())
    del summary["access_accounting"]["protected_payloads_read"]
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="explicitly report zero"):
        locks.validate_manifest_artifacts(tmp_path, directory, primary, "a" * 64, spec())


def test_pilot_does_not_depend_on_labels_or_input_order() -> None:
    rows = [{"sample_id": f"sample-{index}", "label_index": index % 3} for index in range(200)]
    first = locks.pilot_sample_ids(rows, spec())
    second = locks.pilot_sample_ids([{**row, "label_index": 2} for row in reversed(rows)], spec())
    assert first == second
    assert len(first) == len(set(first)) == 128


def test_lock_check_rejects_authorization_or_source_changes() -> None:
    current = {
        "locked_at_utc": "now",
        "source_sha256": {"locker": "a" * 64},
        "authorization": {"model_fitting": False},
    }
    locks.compare_lock({**current, "locked_at_utc": "earlier"}, current)
    with pytest.raises(RuntimeError, match="authorization"):
        locks.compare_lock({**current, "authorization": {"model_fitting": True}}, current)
    with pytest.raises(RuntimeError, match="source_sha256"):
        locks.compare_lock({**current, "source_sha256": {"locker": "b" * 64}}, current)


def test_lock_creation_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / ".runs" / "lock.json"
    locks.write_lock(tmp_path, target, {"immutable": True})
    original = target.read_bytes()
    with pytest.raises(RuntimeError, match="overwrite"):
        locks.write_lock(tmp_path, target, {"immutable": False})
    assert target.read_bytes() == original


def test_upstream_discloses_only_unused_config_changes(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "configs" / "case.yaml"
    config.parent.mkdir()
    config.write_bytes(b"case collision content")

    def fake_git(_root, *args):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if args[0] == "diff":
            return "configs/case.yaml\0"
        if args[0] == "ls-files":
            return ""
        return " M configs/case.yaml"

    monkeypatch.setattr(locks.common, "_git", fake_git)
    result = locks.upstream_receipt(tmp_path, "a" * 40)
    assert result["disclosed_config_changes"][0]["sha256"] == locks.digest(config.read_bytes())
    assert result["working_status"] == " M configs/case.yaml"


@pytest.mark.parametrize(
    "relative", ["app/main.py", "src/models.py", "hubconf.py", "new_encoder.py"]
)
def test_upstream_execution_source_edits_are_rejected(
    tmp_path: Path, monkeypatch, relative: str
) -> None:
    monkeypatch.setattr(
        locks.common,
        "_git",
        lambda _root, *args: "a" * 40 if args == ("rev-parse", "HEAD") else relative + "\0",
    )
    with pytest.raises(RuntimeError, match="execution source changed"):
        locks.upstream_receipt(tmp_path, "a" * 40)
