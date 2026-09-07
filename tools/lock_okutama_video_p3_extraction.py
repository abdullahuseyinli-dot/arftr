"""Bind P3 long-video manifests and frozen encoders before any JPEG payload access."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac import okutama_long_video as long_video  # noqa: E402
from hac.image_encoders import validate_dinov2_lock_receipt  # noqa: E402
from hac.video_encoders import (  # noqa: E402
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    validate_source_checkout,
)
from tools import lock_hac_continuation_protocols as common  # noqa: E402
from tools import lock_okutama_video_protocol as p0  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_p3_extraction_protocol.json"
PROTOCOL_STATUS = "DECLARED_BEFORE_OKUTAMA_P3_IMAGE_PAYLOAD_ACCESS"
STATUS = "OKUTAMA_VIDEO_P3_EXTRACTION_LOCKED_BEFORE_IMAGE_ACCESS"
AUTHORIZATION = {
    "frozen_extraction": True,
    "image_payload_access": True,
    "annotation_payload_access": False,
    "model_fitting": False,
    "probe_fitting": False,
    "backbone_fitting": False,
    "protected_data_access": False,
}
EXPECTED_SOURCES = {
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_video_p3_extraction.py",
    "extractor": "experiments/cache_okutama_long_features.py",
    "long_video_module": "src/hac/okutama_long_video.py",
    "native_video_module": "src/hac/okutama_native_video.py",
    "shared_crop_extractor": "experiments/cache_okutama_video_features.py",
    "dino_extractor": "experiments/cache_okutama_dinov2_features.py",
    "video_encoder_module": "src/hac/video_encoders.py",
    "image_encoder_module": "src/hac/image_encoders.py",
    "requirements": "requirements-video-lock.txt",
}


def _digest(raw: bytes) -> str:
    return p0.digest(raw)


def _canonical(value: Any) -> str:
    return p0.canonical_digest(value)


def _clean_state(root: Path) -> tuple[str, str]:
    status = common._git(root, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError("A clean committed repository is required before P3 image access")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    selected = path.resolve() if path.is_absolute() else (root / path).resolve()
    common.assert_role_safe_input_path(root, selected)
    return selected


def _checked_receipt(root: Path, item: dict[str, Any]) -> tuple[Path, bytes]:
    path = _repo_path(root, item["path"])
    raw = common._read_bytes(path)
    if len(raw) != int(item["size_bytes"]) or _digest(raw) != item["sha256"]:
        raise RuntimeError(f"Bound P3 artifact changed before decoding: {path}")
    return path, raw


def _checked_external(item: dict[str, Any]) -> Path:
    path = Path(item["path"]).resolve()
    digest, size = common._sha256_file(path)
    if size != int(item["size_bytes"]) or digest != item["sha256"]:
        raise RuntimeError(f"Bound external input changed: {path}")
    return path


def _csv(raw: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise RuntimeError("Malformed P3 CSV header")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError("Malformed P3 CSV row")
    return rows


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if (spec.get("protocol_version"), spec.get("status")) != ("1.0.0", PROTOCOL_STATUS):
        raise RuntimeError("Unexpected P3 extraction protocol/version")
    if spec.get("authorization") != AUTHORIZATION:
        raise RuntimeError("P3 extraction authorization changed")
    if spec.get("execution_sources") != EXPECTED_SOURCES:
        raise RuntimeError("P3 extraction source inventory changed")
    cohort = spec.get("cohort", {})
    if (
        cohort.get("rows"),
        cohort.get("all_long_frames_valid_rows"),
        cohort.get("incomplete_long_rows"),
        cohort.get("center_valid_rows"),
        cohort.get("frame_rows"),
    ) != (4977, 4510, 467, 4977, 79632):
        raise RuntimeError("P3 extraction cohort changed")
    sampling = spec.get("sampling", {})
    if (
        sampling.get("source_offsets") != list(long_video.SOURCE_OFFSETS)
        or sampling.get("center_slot") != long_video.CENTER_SLOT
        or sampling.get("frame_count") != 16
        or sampling.get("endpoint_span_seconds") != 2.0
    ):
        raise RuntimeError("P3 long-window sampling changed")
    if spec.get("pilot", {}).get("rows") != 128 or not spec["pilot"].get("label_blind"):
        raise RuntimeError("P3 pilot contract changed")
    return spec


def _assert_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError("Historical P3 materialization commit is not an ancestor")


def _validate_historical_materialization(
    root: Path, spec: dict[str, Any], current_commit: str
) -> tuple[dict[str, Any], bytes]:
    _, raw = _checked_receipt(root, spec["materialization_lock"])
    lock = json.loads(raw)
    unsigned = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if (
        lock.get("status") != long_video.MATERIALIZATION_STATUS
        or lock.get("authorization") != long_video.AUTHORIZATION
        or lock.get("lock_sha256") != _canonical(unsigned)
    ):
        raise RuntimeError("Historical P3 materialization lock is invalid")
    _assert_ancestor(root, lock["repository_commit"], current_commit)
    for name, receipt in lock["sources"].items():
        relative = receipt["path"]
        blob = common._git(root, "rev-parse", f"{lock['repository_commit']}:{relative}")
        if blob != receipt["git_blob_oid"]:
            raise RuntimeError(f"Historical P3 source blob changed: {name}")
        content = subprocess.run(
            ["git", "-C", str(root), "show", f"{lock['repository_commit']}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if len(content) != receipt["size_bytes"] or _digest(content) != receipt["sha256"]:
            raise RuntimeError(f"Historical P3 source receipt is not reproducible: {name}")
    for name in ("eligible_index", "fold_map"):
        _checked_receipt(root, lock["inputs"][name])
    archive = _checked_external(lock["inputs"]["archive"])
    if archive.name != "TrainSetFrames.zip":
        raise RuntimeError("P3 archive filename changed")
    return lock, raw


def _validate_p0_lineage(
    root: Path, spec: dict[str, Any], source: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    _, raw = _checked_receipt(root, spec["p0_extraction_lock"])
    lock = json.loads(raw)
    if (
        lock.get("status") != p0.EXTRACTION_STATUS
        or lock.get("primary_rows") != 4977
        or lock.get("authorization", {}).get("frozen_extraction") is not True
        or lock.get("authorization", {}).get("model_fitting") is not False
    ):
        raise RuntimeError("P0 extraction lineage is not complete and fit-free")
    if lock["inputs"]["archive"] != source["inputs"]["archive"]:
        raise RuntimeError("P0 and P3 do not bind the same source archive")
    checkpoint = _checked_external(lock["inputs"]["checkpoint"])
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or lock["inputs"]["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("P3 V-JEPA checkpoint contract changed")
    upstream = Path(lock["upstream"]["path"]).resolve()
    upstream_receipt = validate_source_checkout(upstream)
    if upstream_receipt["commit"] != SOURCE_COMMIT or lock["upstream"]["commit"] != SOURCE_COMMIT:
        raise RuntimeError("P3 V-JEPA source commit changed")
    validate_dinov2_lock_receipt(
        Path(lock["inputs"]["dinov2_snapshot"]["path"]),
        lock["inputs"]["dinov2_snapshot"],
    )
    return lock, raw


def _validate_manifest(
    root: Path,
    spec: dict[str, Any],
    source: dict[str, Any],
    source_raw: bytes,
    p0_lock: dict[str, Any],
) -> dict[str, Any]:
    decoded: dict[str, list[dict[str, str]]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    summary = None
    for name, item in spec["manifest"].items():
        path, raw = _checked_receipt(root, item)
        artifacts[name] = {"path": item["path"], "sha256": _digest(raw), "size_bytes": len(raw)}
        if name == "summary":
            summary = json.loads(raw)
        else:
            decoded[name] = _csv(raw)
    assert summary is not None
    expected = spec["cohort"]
    if (
        summary.get("status") != long_video.MANIFEST_STATUS
        or summary.get("source_sha256", {}).get("materialization_lock") != _digest(source_raw)
        or summary.get("primary_rows") != expected["rows"]
        or summary.get("all_frames_valid_clips") != expected["all_long_frames_valid_rows"]
        or summary.get("center_frame_valid_clips") != expected["center_valid_rows"]
        or summary.get("frame_rows") != expected["frame_rows"]
        or summary.get("valid_frame_rows") != expected["valid_frame_rows"]
        or summary.get("negative_requested_frame_rows") != expected["negative_requested_frame_rows"]
        or summary.get("unique_present_image_members") != expected["unique_present_image_members"]
        or summary.get("sample_ids_sha256") != expected["sample_ids_sha256"]
        or summary.get("extraction_and_fitting_authorized") is not False
    ):
        raise RuntimeError("P3 manifest summary changed")
    clips, frames, allowlist = (
        decoded["clip_index"],
        decoded["frame_manifest"],
        decoded["image_allowlist"],
    )
    eligible_path = Path(source["inputs"]["eligible_index"]["path"])
    primary = long_video.native.load_primary_index(eligible_path)
    if len(clips) != 4977 or len(frames) != 4977 * 16:
        raise RuntimeError("P3 manifest dropped or duplicated rows")
    present: set[str] = set()
    for position, (original, clip) in enumerate(zip(primary, clips, strict=True)):
        if any(original[key] != clip[key] for key in ("sample_id", "recording_id", "fold", "label", "label_index", "center_frame")):
            raise RuntimeError("P3 clip identity/order differs from the primary cohort")
        valid_count = 0
        for slot, offset in enumerate(long_video.SOURCE_OFFSETS):
            frame = frames[position * 16 + slot]
            number = int(original["center_frame"]) + offset
            expected_member = long_video.native.frame_member(original["provider_recording_id"], number) if number >= 0 else ""
            if (
                frame["sample_id"] != original["sample_id"]
                or int(frame["time_index"]) != slot
                or int(frame["source_frame"]) != number
                or frame["image_member"] != expected_member
                or int(frame["source_frame_nonnegative"]) != int(number >= 0)
                or not math.isclose(float(frame["offset_seconds"]), offset / 30, abs_tol=1e-8)
            ):
                raise RuntimeError("P3 frame sampling/order changed")
            is_valid = frame["valid_frame"].lower() in {"1", "true"}
            if is_valid:
                valid_count += 1
                present.add(frame["image_member"])
            if number < 0 and (is_valid or frame["image_present"].lower() in {"1", "true"}):
                raise RuntimeError("Negative P3 frame was treated as available")
        if int(clip["valid_frame_count"]) != valid_count or int(clip["all_frames_valid"]) != int(valid_count == 16):
            raise RuntimeError("P3 clip validity disagrees with its frame rows")
    members = [row["image_member"] for row in allowlist]
    if len(members) != len(set(members)) or set(members) != present:
        raise RuntimeError("P3 image allowlist changed")
    p0_frames_item = p0_lock["manifest"]["artifacts"]["frame_manifest"]
    _, p0_frames_raw = _checked_receipt(root, p0_frames_item)
    short_frames = _csv(p0_frames_raw)
    center_fields = (
        "sample_id", "provider_recording_id", "provider_track_id", "source_frame",
        "image_member", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "valid_geometry", "image_present", "valid_frame",
    )
    for position in range(4977):
        old = short_frames[position * 16 + 8]
        new = frames[position * 16 + long_video.CENTER_SLOT]
        if any(old[field] != new[field] for field in center_fields):
            raise RuntimeError("P3 center crop differs from the locked short anchor")
    archive_path = Path(source["inputs"]["archive"]["path"])
    with zipfile.ZipFile(archive_path) as archive:
        selected = set(members)
        if sum(info.filename in selected for info in archive.infolist()) != len(selected):
            raise RuntimeError("P3 allowlisted ZIP member is missing or duplicated")
        for row in allowlist:
            info = archive.getinfo(row["image_member"])
            if info.CRC != int(row["crc32"]) or info.file_size != int(row["size_bytes"]):
                raise RuntimeError("P3 allowlisted ZIP metadata changed")
    return {
        "artifacts": artifacts,
        "sample_ids_sha256": _canonical([row["sample_id"] for row in primary]),
        "all_long_frames_valid_rows": sum(int(row["all_frames_valid"]) for row in clips),
        "center_anchor_exact_matches": len(clips),
        "allowlist_rows": len(allowlist),
        "allowed_members_sha256": _canonical(sorted(members)),
        "pilot_sample_ids": source["pilot_sample_ids"],
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean_state(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, relative, commit)
        for name, relative in EXPECTED_SOURCES.items()
    }
    source, source_raw = _validate_historical_materialization(root, spec, commit)
    p0_lock, p0_raw = _validate_p0_lineage(root, spec, source)
    manifest = _validate_manifest(root, spec, source, source_raw, p0_lock)
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "materialization_lock": {**spec["materialization_lock"], "sha256": _digest(source_raw)},
        "p0_extraction_lock": {**spec["p0_extraction_lock"], "sha256": _digest(p0_raw)},
        "manifest": manifest,
        "inputs": p0_lock["inputs"],
        "upstream": p0_lock["upstream"],
        "sampling": spec["sampling"],
        "encoders": spec["encoders"],
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "image_payloads_read": 0,
            "annotation_payloads_read": 0,
            "model_fits": 0,
            "archive_directory_metadata_only": True,
        },
    }


def _compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained P3 extraction lock changed: " + ", ".join(changed))


def validate_extraction_lock(root: Path, path: Path) -> dict[str, Any]:
    path = _repo_path(root, path)
    retained = json.loads(path.read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the fit-free P3 extraction lock")
    _compare(retained, build_payload(root))
    return retained


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _repo_path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing P3 extraction lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    payload = validate_extraction_lock(root, output) if args.mode == "check" else build_payload(root)
    if args.mode == "lock":
        write_lock(root, output, payload)
    print(json.dumps({"mode": args.mode, "status": payload["status"], "rows": payload["protocol"]["cohort"]["rows"], "valid_long_rows": payload["manifest"]["all_long_frames_valid_rows"], "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
