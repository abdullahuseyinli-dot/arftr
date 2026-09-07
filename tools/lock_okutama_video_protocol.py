"""Create/check prospective native-video P0 locks; neither stage authorizes fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import lock_hac_continuation_protocols as common  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_protocol.json"
MATERIALIZATION_STATUS = "OKUTAMA_VIDEO_P0_MANIFEST_LOCKED_BEFORE_ANNOTATION_ACCESS"
EXTRACTION_STATUS = "OKUTAMA_VIDEO_P0_EXTRACTION_LOCKED_BEFORE_IMAGE_ACCESS"
SAMPLE_RE = re.compile(r"^train__([12]\.[12]\.(?:[1-9]|1[01]))__track-(\d+)__frame-(\d{6})$")
FOLDS = {
    "fold-0": ["1.4", "2.2", "2.5"],
    "fold-1": ["1.5", "2.11"],
    "fold-2": ["1.10", "1.2"],
    "fold-3": ["2.7", "2.8"],
    "fold-4": ["1.11", "1.3"],
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_digest(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def safe_path(root: Path, path: Path) -> Path:
    relative = common.assert_role_safe_input_path(root, path)
    if Path(relative).name.lower() in {
        "development_metadata.csv",
        "development_manifest.csv",
        "calibration_manifest.csv",
        "confirmation_manifest.csv",
        "test_manifest.csv",
    }:
        raise RuntimeError("Forbidden mixed/protected manifest path")
    return (root / relative).resolve()


def checked_bytes(root: Path, path: Path, expected: str) -> bytes:
    selected = safe_path(root, path)
    raw = common._read_bytes(selected)
    if digest(raw) != expected:
        raise RuntimeError(f"Input SHA256 mismatch before decoding: {selected}")
    return raw


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads(common._read_bytes(root / PROTOCOL_PATH))
    if (spec.get("protocol_version"), spec.get("status")) != (
        "1.0.0",
        "DECLARED_BEFORE_OKUTAMA_NATIVE_VIDEO_MATERIALIZATION",
    ):
        raise RuntimeError("Unexpected native-video protocol/version")
    data = spec["data_contract"]
    if (
        data["primary_rows"],
        data["primary_scenarios"],
        data["primary_videos"],
        data["primary_tracks"],
    ) != (4977, 11, 21, 444):
        raise RuntimeError("Primary cohort scope changed")
    if spec["fold_contract"] != FOLDS:
        raise RuntimeError("Historical scenario folds changed")
    if spec["sampling"]["source_offsets"] != list(range(-8, 8)) or spec["sampling"]["fps"] != 30.0:
        raise RuntimeError("Native16 sampling contract changed")
    if spec["sampling"]["center_slot"] != 8 or spec["sampling"]["frame_count"] != 16:
        raise RuntimeError("Center/frame count changed")
    if spec["pilot"]["rows"] != 128 or spec["pilot"]["label_blind"] is not True:
        raise RuntimeError("Pilot must remain 128 label-blind clips")
    if spec["p1_prospective"]["requires_later_execution_lock"] is not True:
        raise RuntimeError("Probe fitting requires a later execution lock")
    image_control = spec.get("image_control", {})
    if (
        image_control.get("model_id") != "facebook/dinov2-base"
        or image_control.get("revision") != "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
        or image_control.get("position_interpolation") is not True
        or spec.get("image_control_cache", {}).get("cached_shape_per_clip") != [16, 1, 768]
    ):
        raise RuntimeError("Matched frozen image-control contract changed")
    for stage, status in (
        ("materialization", MATERIALIZATION_STATUS),
        ("extraction", EXTRACTION_STATUS),
    ):
        contract = spec["lock_stages"][stage]
        if (
            contract["status"] != status
            or contract["authorization"].get("model_fitting") is not False
            or contract["authorization"].get("probe_fitting") is not False
        ):
            raise RuntimeError("P0 must never authorize fitting")
    return spec


def csv_rows(raw: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise RuntimeError("Missing or duplicate CSV columns")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError("Malformed CSV row width")
    return rows


def read_primary_cohort(
    root: Path, spec: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    raw = {}
    receipts = {}
    for name, item in spec["historical_inputs"].items():
        path = safe_path(root, root / item["path"])
        raw[name] = checked_bytes(root, path, item["sha256"])
        receipts[name] = {
            "path": item["path"],
            "sha256": digest(raw[name]),
            "size_bytes": len(raw[name]),
        }
    index, folds = csv_rows(raw["eligible_index"]), csv_rows(raw["fold_map"])
    primary = [row for row in index if row["scope"] == "grouped_crossfit_oof"]
    if len(index) != 6360 or len(primary) != 4977 or len(folds) != 4977:
        raise RuntimeError("Eligible/primary cohort row count changed")
    if any(
        row["scope"] not in {"grouped_crossfit_oof", "fixed_development_validation"}
        for row in index
    ):
        raise RuntimeError("An undeclared role entered the eligible index")
    if len({row["sample_id"] for row in primary}) != len(primary):
        raise RuntimeError("Duplicate primary sample IDs")
    expected_folds = {scenario: fold for fold, scenarios in FOLDS.items() for scenario in scenarios}
    bundle_positions = [
        position for position, row in enumerate(index) if row["scope"] == "grouped_crossfit_oof"
    ]
    tracks, videos = {}, {}
    for row, fold_row, position in zip(primary, folds, bundle_positions, strict=True):
        match = SAMPLE_RE.fullmatch(row["sample_id"])
        if match is None:
            raise RuntimeError("Malformed eligible sample ID")
        video, track, frame = match.groups()
        scenario = ".".join(video.split(".")[1:])
        if (video, track, int(frame), scenario) != (
            row["provider_recording_id"],
            row["provider_track_id"],
            int(row["center_frame"]),
            row["recording_id"],
        ):
            raise RuntimeError("Sample ID/provider/scenario lineage mismatch")
        if row["development_role"] != "train" or row["fold"] != expected_folds.get(scenario):
            raise RuntimeError("Primary role or scenario fold changed")
        if (
            int(frame) < 8
            or row["label"] != spec["data_contract"]["class_order"][int(row["label_index"])]
        ):
            raise RuntimeError("Invalid center frame or label mapping")
        if row["track_id"] != f"{video}::{track}":
            raise RuntimeError("Stable track identity changed")
        if [row[key] for key in ("sample_id", "recording_id", "fold")] != [
            fold_row[key] for key in ("sample_id", "recording_id", "fold")
        ] or int(fold_row["bundle_row_index"]) != position:
            raise RuntimeError("Primary/fold-map order mismatch")
        for mapping, key in ((videos, video), (tracks, row["track_id"])):
            if key in mapping and mapping[key] != row["fold"]:
                raise RuntimeError("Video/track crosses an outer fold")
            mapping[key] = row["fold"]
    if (
        dict(Counter(row["recording_id"] for row in primary)) != spec["scenario_row_counts"]
        or len(videos) != 21
        or len(tracks) != 444
    ):
        raise RuntimeError("Scenario/video/track counts changed")
    return primary, receipts


def frame_member(video: str, frame: int) -> str:
    drone, period, _ = video.split(".")
    period_name = {"1": "Morning", "2": "Noon"}[period]
    return f"Drone{drone}/{period_name}/Extracted-Frames-1280x720/{video}/{frame}.jpg"


def pilot_sample_ids(primary: list[dict[str, str]], spec: dict[str, Any]) -> list[str]:
    domain = spec["pilot"]["pilot_domain"]
    return sorted(
        (row["sample_id"] for row in primary),
        key=lambda value: (digest((domain + value).encode()), value),
    )[: spec["pilot"]["rows"]]


def require_clean_repository(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed worktree is required before P0 access")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def opaque_receipt(path: Path, *, name: str, expected: str, size: int) -> dict[str, Any]:
    path = path.resolve()
    if path.name != name or path.stat().st_size != size:
        raise RuntimeError("Pinned external filename or byte count changed")
    observed, observed_size = common._sha256_file(path)
    if observed != expected:
        raise RuntimeError("Pinned external SHA256 changed")
    return {"path": str(path), "sha256": observed, "size_bytes": observed_size}


def snapshot_receipt(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact executable files in a local Hugging Face snapshot."""
    path = path.resolve()
    if not path.is_dir():
        raise RuntimeError("Pinned DINOv2 snapshot root is not a directory")
    files: dict[str, dict[str, Any]] = {}
    for filename, expected in contract["snapshot_files"].items():
        selected = path / filename
        if not selected.is_file() or selected.stat().st_size != expected["size_bytes"]:
            raise RuntimeError(f"Pinned DINOv2 snapshot file or byte count changed: {filename}")
        observed, observed_size = common._sha256_file(selected)
        if observed != expected["sha256"]:
            raise RuntimeError(f"Pinned DINOv2 snapshot SHA256 changed: {filename}")
        files[filename] = {"sha256": observed, "size_bytes": observed_size}
    content = {"revision": contract["revision"], "files": files}
    return {
        "path": str(path),
        "revision": contract["revision"],
        "files": files,
        "sha256": canonical_digest(content),
        "scope": "only the three declared local snapshot files are executable inputs",
    }


def upstream_receipt(path: Path, expected_commit: str) -> dict[str, Any]:
    """Permit disclosed Windows case-collision changes only in unused configs."""
    path = path.resolve()
    commit = common._git(path, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError("Upstream encoder commit differs from the protocol")
    changed = set(filter(None, common._git(path, "diff", "--name-only", "-z", "HEAD").split("\0")))
    changed.update(
        filter(
            None, common._git(path, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
        )
    )
    if any(not value.startswith("configs/") for value in changed):
        raise RuntimeError(
            "Upstream execution source changed outside the disclosed configs directory"
        )
    config_receipts = []
    for relative in sorted(changed):
        selected = path / relative
        if selected.is_file():
            sha256, size = common._sha256_file(selected)
            config_receipts.append(
                {"path": relative, "exists": True, "sha256": sha256, "size_bytes": size}
            )
        else:
            config_receipts.append({"path": relative, "exists": False})
    return {
        "path": str(path),
        "commit": commit,
        "tree": common._git(path, "rev-parse", "HEAD^{tree}"),
        "disclosed_config_changes": config_receipts,
        "working_status": common._git(path, "status", "--porcelain", "--untracked-files=normal"),
        "execution_source_policy": "Only unused configs/ differences permitted; app/, src/, hubconf.py and other execution files must match the pinned commit. Encoder loading independently verifies imported Git blobs.",
    }


def build_materialization_payload(
    root: Path,
    archive_path: Path,
    upstream_repo: Path,
    checkpoint_path: Path,
    dinov2_root: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = require_clean_repository(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, relative, commit)
        for name, relative in spec["execution_sources"].items()
    }
    primary, inputs = read_primary_cohort(root, spec)
    archive, upstream = spec["archive"], spec["upstream"]
    inputs["archive"] = opaque_receipt(
        archive_path,
        name=archive["file_name"],
        expected=archive["sha256"],
        size=archive["size_bytes"],
    )
    upstream_repo = upstream_repo.resolve()
    upstream_state = upstream_receipt(upstream_repo, upstream["commit"])
    inputs["checkpoint"] = opaque_receipt(
        checkpoint_path,
        name=upstream["checkpoint_name"],
        expected=upstream["checkpoint_sha256"],
        size=upstream["checkpoint_size_bytes"],
    )
    if dinov2_root is None:
        raise RuntimeError("A pinned local DINOv2 snapshot is required")
    inputs["dinov2_snapshot"] = snapshot_receipt(dinov2_root, spec["image_control"])
    videos = sorted({row["provider_recording_id"] for row in primary})
    members = sorted(
        {
            frame_member(row["provider_recording_id"], int(row["center_frame"]) + offset)
            for row in primary
            for offset in spec["sampling"]["source_offsets"]
        }
    )
    return {
        "status": MATERIALIZATION_STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "inputs": inputs,
        "source_sha256": {
            **{name: receipt["sha256"] for name, receipt in sources.items()},
            **{name: receipt["sha256"] for name, receipt in inputs.items()},
        },
        "upstream": {
            **upstream_state,
            "url": upstream["url"],
            "pretraining_overlap_status": upstream["pretraining_overlap_status"],
        },
        "primary_rows": len(primary),
        "primary_scenarios": len({row["recording_id"] for row in primary}),
        "primary_sample_ids_sha256": canonical_digest([row["sample_id"] for row in primary]),
        "allowed_recordings": videos,
        "allowed_annotation_members": [
            f"Labels/{kind}/3840x2160/{video}.txt"
            for video in videos
            for kind in ("MultiActionLabels", "SingleActionTrackingLabels")
        ],
        "requested_image_members_sha256": canonical_digest(members),
        "requested_image_members": len(members),
        "sampling": spec["sampling"],
        "preprocessing": spec["preprocessing"],
        "encoder_cache": spec["encoder_cache"],
        "image_control": spec["image_control"],
        "image_control_cache": spec["image_control_cache"],
        "pilot": {**spec["pilot"], "sample_ids": pilot_sample_ids(primary, spec)},
        "authorization": spec["lock_stages"]["materialization"]["authorization"],
        "environment": common.collect_environment(),
        "access_accounting": {
            "annotation_payloads_read": 0,
            "image_payloads_read": 0,
            "protected_payloads_read": 0,
            "model_fits": 0,
        },
    }


def read_lock(root: Path, path: Path) -> tuple[dict[str, Any], bytes]:
    raw = common._read_bytes(safe_path(root, path))
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("Lock must be a JSON object")
    timestamp = datetime.fromisoformat(value.get("locked_at_utc", ""))
    if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise RuntimeError("Lock timestamp must be UTC")
    return value, raw


def compare_lock(retained: dict[str, Any], current: dict[str, Any]) -> None:
    first = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    second = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if first != second:
        differing = sorted(
            key for key in set(first) | set(second) if first.get(key) != second.get(key)
        )
        raise RuntimeError(f"Retained native-video lock changed: {', '.join(differing)}")


def validate_materialization_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    retained, _ = read_lock(root, lock_path)
    if retained.get("status") != MATERIALIZATION_STATUS:
        raise RuntimeError("Expected a materialization lock before annotation access")
    current = build_materialization_payload(
        root,
        Path(retained["inputs"]["archive"]["path"]),
        Path(retained["upstream"]["path"]),
        Path(retained["inputs"]["checkpoint"]["path"]),
        Path(retained["inputs"]["dinov2_snapshot"]["path"]),
    )
    compare_lock(retained, current)
    return retained


def boolean(value: str) -> bool:
    if value.lower() not in {"true", "false", "0", "1"}:
        raise RuntimeError("Malformed Boolean manifest value")
    return value.lower() in {"true", "1"}


def validate_manifest_artifacts(
    root: Path,
    directory: Path,
    primary: list[dict[str, str]],
    source_lock_digest: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    directory = safe_path(root, directory)
    summary_raw = common._read_bytes(safe_path(root, directory / "summary.json"))
    summary = json.loads(summary_raw)
    if summary.get("status") != "OKUTAMA_VIDEO_P0_NATIVE_CLIP_MANIFEST_COMPLETE":
        raise RuntimeError("Native clip manifest did not complete")
    if summary.get("source_sha256", {}).get("materialization_lock") != source_lock_digest:
        raise RuntimeError("Manifest is not bound to this materialization lock")
    accounting = summary.get("access_accounting", {})
    for key in ("image_payloads_read", "protected_payloads_read", "model_fits"):
        if accounting.get(key) != 0:
            raise RuntimeError(f"Manifest access accounting must explicitly report zero {key}")
    artifacts = {
        "summary": {
            "path": common._relative_path(root, directory / "summary.json"),
            "sha256": digest(summary_raw),
            "size_bytes": len(summary_raw),
        }
    }
    decoded = {}
    for name in ("clip_index", "frame_manifest", "image_allowlist"):
        filename = name + ".csv"
        expected = summary.get("artifact_sha256", {}).get(filename, "")
        raw = checked_bytes(root, directory / filename, expected)
        decoded[name] = csv_rows(raw)
        artifacts[name] = {
            "path": common._relative_path(root, directory / filename),
            "sha256": digest(raw),
            "size_bytes": len(raw),
        }
    clips, frames, allowlist = (
        decoded["clip_index"],
        decoded["frame_manifest"],
        decoded["image_allowlist"],
    )
    if len(clips) != len(primary) or len(frames) != len(primary) * 16:
        raise RuntimeError("Manifest dropped or duplicated evaluation centers/frames")
    requested, present = set(), set()
    for position, (source, clip) in enumerate(zip(primary, clips, strict=True)):
        for key in ("sample_id", "recording_id", "fold", "center_frame", "label_index", "label"):
            if source[key] != clip[key]:
                raise RuntimeError(
                    "Clip identity/order/label/fold differs from immutable primary rows"
                )
        valid_frames = 0
        for slot, offset in enumerate(spec["sampling"]["source_offsets"]):
            frame = frames[position * 16 + slot]
            number = int(source["center_frame"]) + offset
            member = frame_member(source["provider_recording_id"], number)
            if (
                frame["sample_id"] != source["sample_id"]
                or frame["provider_recording_id"] != source["provider_recording_id"]
                or frame["provider_track_id"] != source["provider_track_id"]
                or int(frame["time_index"]) != slot
                or int(frame["source_frame"]) != number
                or frame["image_member"] != member
                or int(frame["image_width"]) != 1280
                or int(frame["image_height"]) != 720
            ):
                raise RuntimeError("Frame identity/order/native source member changed")
            for field, expected in (
                ("offset_seconds", offset / 30),
                ("nominal_time_seconds", number / 30),
            ):
                actual = float(frame[field])
                if not math.isfinite(actual) or abs(actual - expected) > 1e-8:
                    raise RuntimeError("Native-frame timestamp changed")
            requested.add(member)
            image_present = boolean(frame["image_present"])
            geometry_valid = boolean(frame["valid_geometry"])
            frame_valid = boolean(frame["valid_frame"])
            if frame_valid != (image_present and geometry_valid):
                raise RuntimeError("Frame validity does not equal image-and-geometry validity")
            if geometry_valid:
                box = tuple(
                    float(frame[key])
                    for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
                )
                x1, y1, x2, y2 = box
                if not (
                    all(math.isfinite(value) for value in box)
                    and x2 > x1
                    and y2 > y1
                    and min(x2, 1280) > max(x1, 0)
                    and min(y2, 720) > max(y1, 0)
                ):
                    raise RuntimeError("A valid frame has invalid locked crop geometry")
            if image_present:
                present.add(member)
            if frame_valid:
                valid_frames += 1
        if int(clip["valid_frame_count"]) != valid_frames:
            raise RuntimeError("Clip valid-frame count differs from frame manifest")
    members = [row["image_member"] for row in allowlist]
    if len(members) != len(set(members)) or set(members) != present:
        raise RuntimeError("Image allowlist differs from exact present primary requests")
    for row in allowlist:
        if (
            row["image_member"]
            != frame_member(row["provider_recording_id"], int(row["source_frame"]))
            or not 0 <= int(row["crc32"]) <= 0xFFFFFFFF
            or int(row["size_bytes"]) <= 0
        ):
            raise RuntimeError("Invalid image allowlist member receipt")
    return {
        "artifacts": artifacts,
        "requested_image_members_sha256": canonical_digest(sorted(requested)),
        "present_image_members_sha256": canonical_digest(sorted(present)),
        "allowlist_rows": len(members),
    }


def build_extraction_payload(
    root: Path, source_lock_path: Path, manifest_dir: Path
) -> dict[str, Any]:
    source = validate_materialization_lock(root, source_lock_path)
    source_raw = common._read_bytes(safe_path(root, source_lock_path))
    spec = load_protocol(root)
    primary, _ = read_primary_cohort(root, spec)
    manifest = validate_manifest_artifacts(root, manifest_dir, primary, digest(source_raw), spec)
    if manifest["requested_image_members_sha256"] != source["requested_image_members_sha256"]:
        raise RuntimeError("Materialized frame requests differ from pre-annotation lock")
    allowlist_receipt = manifest["artifacts"]["image_allowlist"]
    allowlist = csv_rows(
        checked_bytes(root, root / allowlist_receipt["path"], allowlist_receipt["sha256"])
    )
    with zipfile.ZipFile(source["inputs"]["archive"]["path"]) as archive:
        selected = {row["image_member"] for row in allowlist}
        if sum(info.filename in selected for info in archive.infolist()) != len(selected):
            raise RuntimeError("Duplicate or missing selected ZIP member name")
        for row in allowlist:
            info = archive.getinfo(row["image_member"])
            if info.CRC != int(row["crc32"]) or info.file_size != int(row["size_bytes"]):
                raise RuntimeError("Allowlisted member bytes/CRC differs from archive directory")
    return {
        "status": EXTRACTION_STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": source["repository_commit"],
        "repository_tree": source["repository_tree"],
        "protocol_sha256": source["protocol_sha256"],
        "source_lock": {
            "path": common._relative_path(root, source_lock_path),
            "sha256": digest(source_raw),
            "size_bytes": len(source_raw),
        },
        "source_sha256": {**source["source_sha256"], "materialization_lock": digest(source_raw)},
        "sources": source["sources"],
        "inputs": source["inputs"],
        "upstream": source["upstream"],
        "manifest": manifest,
        "primary_rows": source["primary_rows"],
        "primary_scenarios": source["primary_scenarios"],
        "sampling": source["sampling"],
        "preprocessing": source["preprocessing"],
        "encoder_cache": source["encoder_cache"],
        "image_control": source["image_control"],
        "image_control_cache": source["image_control_cache"],
        "pilot": source["pilot"],
        "authorization": spec["lock_stages"]["extraction"]["authorization"],
        "environment": source["environment"],
        "access_accounting": {
            "image_payloads_read": 0,
            "protected_payloads_read": 0,
            "model_fits": 0,
            "archive_directory_metadata_only": True,
        },
    }


def validate_extraction_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    retained, _ = read_lock(root, lock_path)
    if retained.get("status") != EXTRACTION_STATUS:
        raise RuntimeError("Expected an extraction lock before image access")
    source_path = safe_path(root, root / retained["source_lock"]["path"])
    checked_bytes(root, source_path, retained["source_lock"]["sha256"])
    summary_path = root / retained["manifest"]["artifacts"]["summary"]["path"]
    current = build_extraction_payload(root, source_path, summary_path.parent)
    compare_lock(retained, current)
    return retained


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = safe_path(root, path)
    if path.exists():
        raise RuntimeError("Refusing to overwrite an existing lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=("materialization", "extraction"), required=True)
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), default="prepare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--upstream-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dinov2-root", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "check":
        checker = (
            validate_materialization_lock
            if args.stage == "materialization"
            else validate_extraction_lock
        )
        payload = checker(root, args.output.resolve())
    else:
        if args.stage == "materialization":
            if None in (args.archive, args.upstream_repo, args.checkpoint, args.dinov2_root):
                parser.error(
                    "materialization requires --archive, --upstream-repo, --checkpoint and --dinov2-root"
                )
            payload = build_materialization_payload(
                root, args.archive, args.upstream_repo, args.checkpoint, args.dinov2_root
            )
        else:
            if args.source_lock is None or args.manifest_dir is None:
                parser.error("extraction requires --source-lock and --manifest-dir")
            payload = build_extraction_payload(
                root, args.source_lock.resolve(), args.manifest_dir.resolve()
            )
        if args.mode == "lock":
            write_lock(root, args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "stage": args.stage,
                "status": payload["status"],
                "rows": payload["primary_rows"],
                "authorization": payload["authorization"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
