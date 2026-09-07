"""Allowlisted Okutama native-frame manifests and explicit crop coordinates.

Manifest construction reads annotation payloads, but never JPEG payloads. The
available source pixels are 1280x720; provider boxes use 3840x2160 coordinates.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_STATUS = "OKUTAMA_VIDEO_P0_NATIVE_CLIP_MANIFEST_COMPLETE"
MATERIALIZATION_STATUS = "OKUTAMA_VIDEO_P0_MANIFEST_LOCKED_BEFORE_ANNOTATION_ACCESS"
EXPECTED_ARCHIVE_BYTES = 5_770_432_522
EXPECTED_ARCHIVE_SHA256 = "c021ce8a12c84e083f359023ffd41c145561aaedb48b118e7c5416d5ddcecb73"
EXPECTED_INDEX_SHA256 = "5bf6d0cc11a18d3e986b714f8a3de71cadd9fef3bd233eb9b5f212f701aadbbb"
SOURCE_OFFSETS = tuple(range(-8, 8))
SOURCE_FPS = 30
CENTER_SLOT = 8
IMAGE_SIZE = (1280, 720)
EXPECTED_SCENARIOS = {
    "1.4": ("fold-0", 488),
    "2.2": ("fold-0", 376),
    "2.5": ("fold-0", 332),
    "1.5": ("fold-1", 453),
    "2.11": ("fold-1", 473),
    "1.10": ("fold-2", 597),
    "1.2": ("fold-2", 464),
    "2.7": ("fold-3", 540),
    "2.8": ("fold-3", 416),
    "1.11": ("fold-4", 60),
    "1.3": ("fold-4", 778),
}
EXPECTED_PRIMARY_ROWS = 4977
EXPECTED_SAFE_ROWS = 6360
EXPECTED_RECORDINGS = {
    f"{drone}.{scenario}" for scenario in EXPECTED_SCENARIOS for drone in (1, 2)
} - {"2.1.11"}
FORBIDDEN_INPUT_NAMES = {
    "development_manifest.csv",
    "development_metadata.csv",
    "development_centres.csv",
    "calibration_manifest.csv",
    "confirmation_manifest.csv",
    "test_manifest.csv",
    "testsetframes.zip",
}
SAMPLE_PATTERN = re.compile(
    r"^train__(?P<recording>[12]\.[12]\.(?:[1-9]|1[01]))__track-"
    r"(?P<track>\d+)__frame-(?P<frame>\d{6})$"
)
TARGET_LABEL = {
    "Sitting": "sitting",
    "Standing": "standing",
    "Walking": "walking_running",
    "Running": "walking_running",
}
LABEL_INDEX = {"sitting": 0, "standing": 1, "walking_running": 2}
CLIP_FIELDS = (
    "sample_id",
    "recording_id",
    "provider_recording_id",
    "provider_track_id",
    "center_frame",
    "fold",
    "label",
    "label_index",
    "scope",
    "development_role",
    "transition_window",
    "window_any_occluded",
    "valid_frame_count",
    "all_frames_valid",
    "center_frame_valid",
    "missing_input_policy",
)
FRAME_FIELDS = (
    "sample_id",
    "time_index",
    "provider_recording_id",
    "provider_track_id",
    "source_frame",
    "nominal_time_seconds",
    "offset_seconds",
    "image_member",
    "image_width",
    "image_height",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "annotation_occluded",
    "annotation_generated",
    "valid_geometry",
    "image_present",
    "valid_frame",
    "missing_reason",
)
IMAGE_FIELDS = (
    "image_member",
    "provider_recording_id",
    "source_frame",
    "crc32",
    "size_bytes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while hashing: {path}")
    return digest.hexdigest()


def reject_forbidden_path(path: Path) -> None:
    if Path(path).name.casefold() in FORBIDDEN_INPUT_NAMES:
        raise RuntimeError("Mixed or protected source is forbidden")


def parse_sample_id(sample_id: str) -> tuple[str, str, int]:
    match = SAMPLE_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ValueError("Invalid native-video sample identity")
    return match["recording"], match["track"], int(match["frame"])


def frame_member(recording: str, frame: int) -> str:
    if recording not in EXPECTED_RECORDINGS or frame < 0:
        raise ValueError("Frame request is outside the permitted primary recordings")
    drone, period, _ = recording.split(".")
    name = {"1": "Morning", "2": "Noon"}[period]
    return f"Drone{drone}/{name}/Extracted-Frames-1280x720/{recording}/{frame}.jpg"


def load_primary_index(path: Path) -> list[dict[str, str]]:
    """Validate the frozen safe export, then retain only its original OOF centers."""
    reject_forbidden_path(path)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_INDEX_SHA256:
        raise RuntimeError("Role-safe eligible index differs from the frozen export")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) != EXPECTED_SAFE_ROWS:
        raise RuntimeError("Role-safe index cardinality changed")
    primary = [row for row in rows if row.get("scope") == "grouped_crossfit_oof"]
    if len(primary) != EXPECTED_PRIMARY_ROWS:
        raise RuntimeError("Primary cohort cardinality changed")
    if len({row["sample_id"] for row in primary}) != len(primary):
        raise RuntimeError("Duplicate primary sample identity")
    counts: dict[str, int] = {}
    recordings: set[str] = set()
    for row in primary:
        recording, track, center = parse_sample_id(row["sample_id"])
        scenario = ".".join(recording.split(".")[1:])
        if scenario not in EXPECTED_SCENARIOS:
            raise RuntimeError("Unexpected primary scenario")
        expected_fold, _ = EXPECTED_SCENARIOS[scenario]
        if (
            row.get("recording_id") != scenario
            or row.get("provider_recording_id") != recording
            or row.get("provider_track_id") != track
            or row.get("track_id") != f"{recording}::{track}"
            or row.get("fold") != expected_fold
            or int(row.get("center_frame", -1)) != center
            or row.get("development_role") != "train"
            or row.get("label") not in LABEL_INDEX
            or int(row.get("label_index", -1)) != LABEL_INDEX[row["label"]]
            or center % 30 != 0
            or center % 180 == 0
            or center < 8
        ):
            raise RuntimeError("Primary identity, role, fold, label, or center contract changed")
        counts[scenario] = counts.get(scenario, 0) + 1
        recordings.add(recording)
    if counts != {key: value[1] for key, value in EXPECTED_SCENARIOS.items()}:
        raise RuntimeError("Primary scenario counts changed")
    if recordings != EXPECTED_RECORDINGS:
        raise RuntimeError("Primary recording/view cohort changed")
    return primary


@dataclass(frozen=True)
class Annotation:
    track: int
    box: tuple[int, int, int, int]
    frame: int
    lost: bool
    occluded: bool
    generated: bool
    actions: tuple[str, ...]


def parse_annotation(line: str) -> Annotation:
    fields = shlex.split(line)
    if len(fields) < 10 or fields[9] != "Person":
        raise RuntimeError("Malformed permitted annotation")
    if any(value not in {"0", "1"} for value in fields[6:9]):
        raise RuntimeError("Malformed annotation flags")
    return Annotation(
        int(fields[0]),
        tuple(map(int, fields[1:5])),
        int(fields[5]),
        bool(int(fields[6])),
        bool(int(fields[7])),
        bool(int(fields[8])),
        tuple(fields[10:]),
    )


def annotation_members(recording: str) -> tuple[str, str]:
    if recording not in EXPECTED_RECORDINGS:
        raise RuntimeError("Annotation recording is outside the primary cohort")
    return (
        f"Labels/MultiActionLabels/3840x2160/{recording}.txt",
        f"Labels/SingleActionTrackingLabels/3840x2160/{recording}.txt",
    )


def read_stable_annotations(
    archive: zipfile.ZipFile,
    recording: str,
    accessed: list[str],
) -> dict[tuple[int, int], Annotation]:
    def read(member: str) -> list[Annotation]:
        # The only payload read in manifest construction. Never testzip()/read JPEGs.
        payload = archive.read(member)
        accessed.append(member)
        return [
            parse_annotation(line) for line in payload.decode("utf-8").splitlines() if line.strip()
        ]

    multi_member, tracking_member = annotation_members(recording)
    multi, tracking = read(multi_member), read(tracking_member)
    tracking_by_key: dict[tuple, Annotation] = {}
    for item in tracking:
        key = (item.frame, item.box, item.occluded, item.generated)
        if key in tracking_by_key:
            raise RuntimeError("Duplicate permitted tracking join key")
        tracking_by_key[key] = item
    stable: dict[tuple[int, int], Annotation] = {}
    for item in multi:
        if item.lost:
            continue
        identity = tracking_by_key.get((item.frame, item.box, item.occluded, item.generated))
        if identity is None or identity.lost:
            continue
        key = (identity.track, item.frame)
        if key in stable:
            raise RuntimeError("Duplicate stable track/frame in permitted recording")
        stable[key] = Annotation(
            identity.track, item.box, item.frame, False, item.occluded, item.generated, item.actions
        )
    return stable


def source_box(annotation: Annotation) -> tuple[float, float, float, float]:
    return tuple(value / 3.0 for value in annotation.box)


def valid_box(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return all(math.isfinite(value) for value in box) and (
        x2 > x1 and y2 > y1 and min(x2, 1280) > max(x1, 0) and min(y2, 720) > max(y1, 0)
    )


@dataclass(frozen=True)
class CropGeometry:
    crop_box: tuple[int, int, int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int]
    output_size: int
    image_to_crop: np.ndarray

    @property
    def pixel_center_image_to_crop(self) -> np.ndarray:
        """Map zero-based pixel-center indices, as used by point trackers."""
        result = self.image_to_crop.copy()
        result[0, 2] += (result[0, 0] - 1) / 2
        result[1, 2] += (result[1, 1] - 1) / 2
        return result


def crop_geometry(
    box: tuple[float, float, float, float],
    *,
    output_size: int,
    context_fraction: float = 0.25,
) -> CropGeometry:
    """Integer native crop + aspect-preserving letterbox, with pixel-edge coordinates.

    Context adds this fraction of the original extent on EACH side. Crops can extend
    outside the image; callers preserve this geometry and pad those pixels explicitly.
    """
    if not valid_box(box) or output_size < 1 or not 0 <= context_fraction <= 1:
        raise ValueError("Invalid crop geometry")
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left, top = (
        math.floor(x1 - context_fraction * width),
        math.floor(y1 - context_fraction * height),
    )
    right, bottom = (
        math.ceil(x2 + context_fraction * width),
        math.ceil(y2 + context_fraction * height),
    )
    width, height = right - left, bottom - top
    scale = output_size / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    padding = ((output_size - resized[0]) // 2, (output_size - resized[1]) // 2)
    sx, sy = resized[0] / width, resized[1] / height
    transform = np.asarray(
        [[sx, 0, padding[0] - left * sx], [0, sy, padding[1] - top * sy], [0, 0, 1]],
        dtype=np.float64,
    )
    return CropGeometry((left, top, right, bottom), resized, padding, output_size, transform)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_native_manifest(
    *,
    eligible_index: Path,
    archive_path: Path,
    output_dir: Path,
    source_lock: dict[str, Any],
    source_lock_path: Path,
) -> dict[str, Any]:
    """Called only after the independent locker validates its live source receipts."""
    reject_forbidden_path(eligible_index)
    reject_forbidden_path(archive_path)
    if source_lock.get("status") != MATERIALIZATION_STATUS:
        raise RuntimeError("Native manifest requires its pre-annotation materialization lock")
    if source_lock.get("authorization", {}).get("annotation_manifest") is not True:
        raise RuntimeError("Lock does not authorize annotation manifest materialization")
    sampling_contract = {
        "source_offsets": list(SOURCE_OFFSETS),
        "fps": SOURCE_FPS,
        "center_slot": CENTER_SLOT,
        "frame_count": len(SOURCE_OFFSETS),
    }
    if any(
        source_lock.get("sampling", {}).get(key) != value
        for key, value in sampling_contract.items()
    ):
        raise RuntimeError("Locked sampling contract changed")
    if source_lock.get("primary_rows") != EXPECTED_PRIMARY_ROWS:
        raise RuntimeError("Locked primary cohort changed")
    if source_lock.get("primary_scenarios") != len(EXPECTED_SCENARIOS):
        raise RuntimeError("Locked primary scenario cohort changed")
    lock_hash = sha256_file(source_lock_path)
    for name, path in (("eligible_index", eligible_index), ("archive", archive_path)):
        expected = source_lock.get("inputs", {}).get(name, {})
        expected_path = Path(expected.get("path", ""))
        if not expected_path.is_absolute():
            expected_path = Path(__file__).resolve().parents[2] / expected_path
        if (
            expected_path.resolve() != path.resolve()
            or expected.get("size_bytes") != path.stat().st_size
            or expected.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"Materialization input does not match its lock: {name}")
    if (
        archive_path.name != "TrainSetFrames.zip"
        or archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES
        or sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256
    ):
        raise RuntimeError("Development archive identity changed")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Manifest output directory must be new or empty")
    primary = load_primary_index(eligible_index)
    by_recording: dict[str, list[dict[str, str]]] = {}
    for row in primary:
        by_recording.setdefault(row["provider_recording_id"], []).append(row)
    frame_rows: dict[str, list[dict[str, Any]]] = {}
    image_rows: dict[str, dict[str, Any]] = {}
    accessed: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for recording in sorted(by_recording):
            stable = read_stable_annotations(archive, recording, accessed)
            for row in by_recording[recording]:
                center, track = int(row["center_frame"]), int(row["provider_track_id"])
                center_annotation = stable.get((track, center))
                if center_annotation is not None:
                    base = [
                        action for action in center_annotation.actions if action in TARGET_LABEL
                    ]
                    if len(base) != 1 or TARGET_LABEL[base[0]] != row["label"]:
                        raise RuntimeError("Original primary center label disagrees with source")
                frames = []
                for slot, offset in enumerate(SOURCE_OFFSETS):
                    frame = center + offset
                    member = frame_member(recording, frame)
                    annotation = stable.get((track, frame))
                    box = source_box(annotation) if annotation is not None else None
                    geometry_ok = box is not None and valid_box(box)
                    try:
                        info = archive.getinfo(member)
                    except KeyError:
                        info = None
                    present = info is not None and not info.is_dir() and info.file_size > 0
                    reasons = []
                    if annotation is None:
                        reasons.append("missing_stable_annotation")
                    elif not geometry_ok:
                        reasons.append("invalid_geometry")
                    if not present:
                        reasons.append("missing_image_member")
                    if present:
                        image_rows[member] = {
                            "image_member": member,
                            "provider_recording_id": recording,
                            "source_frame": frame,
                            "crc32": info.CRC,
                            "size_bytes": info.file_size,
                        }
                    frame_row = {
                        "sample_id": row["sample_id"],
                        "time_index": slot,
                        "provider_recording_id": recording,
                        "provider_track_id": track,
                        "source_frame": frame,
                        "nominal_time_seconds": frame / SOURCE_FPS,
                        "offset_seconds": offset / SOURCE_FPS,
                        "image_member": member,
                        "image_width": 1280,
                        "image_height": 720,
                        "annotation_occluded": int(annotation.occluded) if annotation else "",
                        "annotation_generated": int(annotation.generated) if annotation else "",
                        "valid_geometry": int(geometry_ok),
                        "image_present": int(present),
                        "valid_frame": int(not reasons),
                        "missing_reason": ";".join(reasons),
                    }
                    for key, value in zip(
                        ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"),
                        box if box is not None else ("", "", "", ""),
                        strict=True,
                    ):
                        frame_row[key] = value
                    frames.append(frame_row)
                frame_rows[row["sample_id"]] = frames
    if len(accessed) != 2 * len(EXPECTED_RECORDINGS) or len(set(accessed)) != len(accessed):
        raise RuntimeError("Permitted annotation access count changed")
    # Recheck opaque inputs before emitting a completed result. No model/image access.
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Development archive changed during manifest materialization")
    if sha256_file(eligible_index) != EXPECTED_INDEX_SHA256:
        raise RuntimeError("Role-safe index changed during manifest materialization")
    if sha256_file(source_lock_path) != lock_hash:
        raise RuntimeError("Materialization lock changed during annotation access")
    clips, flattened = [], []
    for row in primary:
        frames = frame_rows[row["sample_id"]]
        valid_count = sum(item["valid_frame"] for item in frames)
        clips.append(
            {
                **row,
                "valid_frame_count": valid_count,
                "all_frames_valid": int(valid_count == len(SOURCE_OFFSETS)),
                "center_frame_valid": frames[CENTER_SLOT]["valid_frame"],
                "missing_input_policy": "preserve_center_and_mask_missing_frames;anchor_fallback",
            }
        )
        flattened.extend(frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "clip_index.csv", CLIP_FIELDS, clips)
    _write_csv(output_dir / "frame_manifest.csv", FRAME_FIELDS, flattened)
    _write_csv(
        output_dir / "image_allowlist.csv",
        IMAGE_FIELDS,
        [image_rows[key] for key in sorted(image_rows)],
    )
    summary = {
        "status": MANIFEST_STATUS,
        "primary_rows": len(clips),
        "primary_scenarios": len(EXPECTED_SCENARIOS),
        "primary_recordings": len(by_recording),
        "frame_rows": len(flattened),
        "unique_present_image_members": len(image_rows),
        "valid_frame_rows": sum(item["valid_frame"] for item in flattened),
        "all_frames_valid_clips": sum(item["all_frames_valid"] for item in clips),
        "center_frame_valid_clips": sum(item["center_frame_valid"] for item in clips),
        "sampling": source_lock["sampling"],
        "source_image_size": list(IMAGE_SIZE),
        "annotation_image_size": [3840, 2160],
        "time_provenance": "provider_frame_index_divided_by_nominal_30fps;not_measured_video_pts",
        "source_sha256": {
            "archive": EXPECTED_ARCHIVE_SHA256,
            "eligible_index": EXPECTED_INDEX_SHA256,
            "native_video_module": sha256_file(Path(__file__)),
            "manifest_builder": sha256_file(
                Path(__file__).resolve().parents[2]
                / "experiments/build_okutama_native_clip_manifest.py"
            ),
            "materialization_lock": lock_hash,
        },
        "artifact_sha256": {
            name: sha256_file(output_dir / name)
            for name in ("clip_index.csv", "frame_manifest.csv", "image_allowlist.csv")
        },
        "annotation_members_read": accessed,
        "access_accounting": {
            "allowed_annotation_members_read": len(accessed),
            "disallowed_annotation_members_read": 0,
            "image_payloads_read": 0,
            "image_payloads_decoded": 0,
            "archive_wide_payload_scan": False,
            "protected_manifest_rows_read": 0,
            "protected_payloads_read": 0,
            "model_fits": 0,
            "mixed_metadata_rows_read": 0,
            "primary_rows_retained": len(clips),
            "fixed_validation_rows_materialized": 0,
            "checkpoints_loaded": 0,
        },
        "missing_policy": "no_dropped_centers;explicit_masks;no_cross_track_fill;anchor_fallback",
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary
