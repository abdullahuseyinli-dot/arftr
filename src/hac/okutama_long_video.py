"""Prospectively locked, label-blind long16 manifests; never decode JPEG payloads.

Reuse P0's role-safe index, stable-track join, source boxes and crop geometry without
altering its sources or short-window constants. Negative requested frame positions
remain in the manifest with explicit masks instead of moving the original center.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from hac import okutama_native_video as native

PROTOCOL_PATH = "experiments/okutama_video_p3_protocol.json"
PROTOCOL_STATUS = "DECLARED_BEFORE_OKUTAMA_P3_LONG_VIDEO_ANNOTATION_OR_IMAGE_ACCESS"
MATERIALIZATION_STATUS = "OKUTAMA_VIDEO_P3_MANIFEST_LOCKED_BEFORE_ANNOTATION_ACCESS"
MANIFEST_STATUS = "OKUTAMA_VIDEO_P3_LONG_CLIP_MANIFEST_COMPLETE"
SOURCE_OFFSETS = tuple(range(-32, 29, 4))
SOURCE_FPS = 30.0
CENTER_SLOT = 8
ENDPOINT_SPAN_SECONDS = 2.0
EXPECTED_FOLD_SHA256 = "5f68a87aa39dfe5f8a30eaaf34416833ee0610faaf73890093ed763b18065547"
P0_PROTOCOL_SHA256 = "bfca9f49a1dc6141e1896f849913e37e2a83ea3bcb76eb59e00f513f71982e19"
FRAME_FIELDS = (*native.FRAME_FIELDS, "source_frame_nonnegative")
AUTHORIZATION = {
    "annotation_manifest": True,
    "frozen_extraction": False,
    "image_payload_access": False,
    "model_fitting": False,
    "probe_fitting": False,
    "protected_data_access": False,
}
SOURCE_PATHS = {
    "protocol": PROTOCOL_PATH,
    "long_video_module": "src/hac/okutama_long_video.py",
    "manifest_builder": "experiments/build_okutama_long_clip_manifest.py",
    "native_video_module": "src/hac/okutama_native_video.py",
    "p0_protocol": "experiments/okutama_video_protocol.json",
}


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sampling_contract() -> dict[str, Any]:
    return {
        "source_offsets": list(SOURCE_OFFSETS),
        "fps": SOURCE_FPS,
        "frame_count": 16,
        "center_slot": CENTER_SLOT,
        "source_stride": 4,
        "endpoint_span_seconds": ENDPOINT_SPAN_SECONDS,
    }


def annotation_allowlist() -> list[str]:
    return sorted(
        member
        for recording in native.EXPECTED_RECORDINGS
        for member in native.annotation_members(recording)
    )


def _safe_path(path: Path, *, root: Path | None = None) -> Path:
    native.reject_forbidden_path(path)
    resolved = Path(path).resolve()
    native.reject_forbidden_path(resolved)
    if any(part.casefold() in {"calibration", "confirmation", "test"} for part in resolved.parts):
        raise RuntimeError("Protected path component is forbidden")
    if root is not None and not resolved.is_relative_to(root.resolve()):
        raise RuntimeError("Role-safe input must remain inside the repository")
    return resolved


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return process.stdout.strip()


def _repository_state(root: Path) -> dict[str, str]:
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before P3 annotation access")
    return {
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "repository_tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }


def _receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": native.sha256_file(path),
    }


def _source_receipts(root: Path) -> dict[str, Any]:
    result = {}
    for name, relative in SOURCE_PATHS.items():
        path = _safe_path(root / relative, root=root)
        committed = _git(root, "rev-parse", f"HEAD:{relative}")
        if _git(root, "hash-object", str(path)) != committed:
            raise RuntimeError(f"P3 execution source differs from committed blob: {name}")
        result[name] = {**_receipt(path), "path": relative, "git_blob_oid": committed}
    return result


def _read_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if (spec.get("protocol_version"), spec.get("status")) != ("1.0.0", PROTOCOL_STATUS):
        raise RuntimeError("Unexpected P3 protocol or version")
    if spec.get("execution_sources") != SOURCE_PATHS:
        raise RuntimeError("P3 execution-source scope changed")
    if spec.get("materialization_lock") != {
        "status": MATERIALIZATION_STATUS,
        "requires_clean_committed_repository": True,
        "authorization": AUTHORIZATION,
    }:
        raise RuntimeError("P3 manifest authorization changed")
    if any(
        spec.get("sampling", {}).get(key) != value for key, value in sampling_contract().items()
    ):
        raise RuntimeError("P3 long16 sampling changed")
    expected_p0 = {"path": SOURCE_PATHS["p0_protocol"], "sha256": P0_PROTOCOL_SHA256}
    if spec.get("p0_protocol") != expected_p0:
        raise RuntimeError("P3 must inherit the pinned P0 preprocessing contract")
    p0_path = root / expected_p0["path"]
    if native.sha256_file(p0_path) != P0_PROTOCOL_SHA256:
        raise RuntimeError("Pinned P0 protocol changed")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    for field in ("fold_contract", "scenario_row_counts", "historical_inputs", "preprocessing"):
        if spec.get(field) != p0[field]:
            raise RuntimeError(f"P3 changed the P0 {field} contract")
    for key in (
        "primary_scope",
        "primary_rows",
        "primary_scenarios",
        "primary_videos",
        "primary_tracks",
        "class_order",
        "all_original_centers_remain_in_denominator",
    ):
        if spec.get("data_contract", {}).get(key) != p0["data_contract"][key]:
            raise RuntimeError(f"P3 primary cohort contract changed: {key}")
    for key in ("file_name", "size_bytes", "sha256"):
        if spec.get("archive", {}).get(key) != p0["archive"][key]:
            raise RuntimeError("P3 source archive identity changed")
    if (
        spec.get("matched_controls_prospective", {}).get(
            "requires_later_extraction_and_fitting_locks"
        )
        is not True
    ):
        raise RuntimeError("P3 extraction and fitting need separate future locks")
    return spec


def _read_primary(
    eligible_index: Path, fold_map: Path, spec: dict[str, Any]
) -> list[dict[str, str]]:
    primary = native.load_primary_index(eligible_index)
    raw = fold_map.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_FOLD_SHA256:
        raise RuntimeError("Frozen primary fold map changed")
    folds = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    safe_rows = list(csv.DictReader(io.StringIO(eligible_index.read_text(encoding="utf-8-sig"))))
    if any(
        row.get("scope") not in {"grouped_crossfit_oof", "fixed_development_validation"}
        for row in safe_rows
    ):
        raise RuntimeError("Undeclared role in safe eligible index")
    positions = [i for i, row in enumerate(safe_rows) if row["scope"] == "grouped_crossfit_oof"]
    if len(folds) != len(primary):
        raise RuntimeError("Primary fold-map row count changed")
    for row, fold, position in zip(primary, folds, positions, strict=True):
        if (
            any(row[key] != fold.get(key) for key in ("sample_id", "recording_id", "fold"))
            or int(fold.get("bundle_row_index", -1)) != position
        ):
            raise RuntimeError("Primary fold-map alignment changed")
    if len({row["track_id"] for row in primary}) != spec["data_contract"]["primary_tracks"]:
        raise RuntimeError("Primary stable-track count changed")
    return primary


def pilot_sample_ids(primary: list[dict[str, str]], spec: dict[str, Any]) -> list[str]:
    domain = spec["pilot"]["pilot_domain"]
    return sorted(
        (row["sample_id"] for row in primary),
        key=lambda value: (hashlib.sha256((domain + value).encode()).hexdigest(), value),
    )[: spec["pilot"]["rows"]]


def _build_lock_payload(root: Path, archive_path: Path) -> dict[str, Any]:
    """Read only pinned source/index bytes and opaque archive bytes; never open its ZIP."""
    root = root.resolve()
    state = _repository_state(root)
    spec = _read_protocol(root)
    sources = _source_receipts(root)
    inputs = {}
    for name in ("eligible_index", "fold_map"):
        expected = spec["historical_inputs"][name]
        path = _safe_path(root / expected["path"], root=root)
        receipt = _receipt(path)
        if receipt["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Pinned P3 {name} hash changed before decoding")
        inputs[name] = receipt
    archive_path = _safe_path(archive_path)
    archive_receipt = _receipt(archive_path)
    if archive_path.name != "TrainSetFrames.zip":
        raise RuntimeError("P3 requires the pinned TrainSetFrames.zip")
    if (
        archive_receipt["size_bytes"] != native.EXPECTED_ARCHIVE_BYTES
        or archive_receipt["sha256"] != native.EXPECTED_ARCHIVE_SHA256
    ):
        raise RuntimeError("P3 archive identity changed before annotation access")
    inputs["archive"] = archive_receipt
    primary = _read_primary(
        Path(inputs["eligible_index"]["path"]), Path(inputs["fold_map"]["path"]), spec
    )
    payload = {
        "status": MATERIALIZATION_STATUS,
        "protocol_version": spec["protocol_version"],
        "study_id": spec["study_id"],
        **state,
        "authorization": AUTHORIZATION.copy(),
        "sampling": spec["sampling"],
        "preprocessing": spec["preprocessing"],
        "primary_rows": len(primary),
        "primary_scenarios": len(native.EXPECTED_SCENARIOS),
        "primary_recordings": len(native.EXPECTED_RECORDINGS),
        "primary_tracks": spec["data_contract"]["primary_tracks"],
        "sample_ids_sha256": canonical_digest([row["sample_id"] for row in primary]),
        "pilot_sample_ids": pilot_sample_ids(primary, spec),
        "annotation_members": annotation_allowlist(),
        "inputs": inputs,
        "sources": sources,
        "source_sha256": {
            **{name: receipt["sha256"] for name, receipt in inputs.items()},
            **{name: receipt["sha256"] for name, receipt in sources.items()},
        },
        "access_accounting": {
            "annotation_payloads_read": 0,
            "image_payloads_read": 0,
            "model_fits": 0,
        },
    }
    return {**payload, "lock_sha256": canonical_digest(payload)}


def create_materialization_lock(root: Path, archive_path: Path, lock_path: Path) -> dict[str, Any]:
    lock_path = _safe_path(lock_path, root=root)
    if lock_path.exists():
        raise FileExistsError("P3 source lock must not overwrite an existing declaration")
    payload = _build_lock_payload(root, archive_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def validate_materialization_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    lock_path = _safe_path(lock_path, root=root)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != MATERIALIZATION_STATUS
        or payload.get("authorization") != AUTHORIZATION
    ):
        raise RuntimeError("P3 requires its annotation-only source lock")
    unsigned = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if payload.get("lock_sha256") != canonical_digest(unsigned):
        raise RuntimeError("P3 source lock digest changed")
    archive_path = _safe_path(Path(payload.get("inputs", {}).get("archive", {}).get("path", "")))
    if payload != _build_lock_payload(root, archive_path):
        raise RuntimeError("P3 source lock does not match live sources, inputs or cohort")
    return payload


def long_frame_rows(
    row: dict[str, str],
    stable: dict[tuple[int, int], native.Annotation],
    archive: zipfile.ZipFile,
    image_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use geometry/track identity only, never annotation action labels or boundaries."""
    recording, track, center = native.parse_sample_id(row["sample_id"])
    frames = []
    for slot, offset in enumerate(SOURCE_OFFSETS):
        frame = center + offset
        member = native.frame_member(recording, frame) if frame >= 0 else ""
        annotation = stable.get((int(track), frame)) if frame >= 0 else None
        box = native.source_box(annotation) if annotation is not None else None
        geometry_ok = box is not None and native.valid_box(box)
        info = None
        if member:
            try:
                info = archive.getinfo(member)
            except KeyError:
                pass
        present = info is not None and not info.is_dir() and info.file_size > 0
        reasons = []
        if frame < 0:
            reasons.append("source_frame_before_start")
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
        item = {
            "sample_id": row["sample_id"],
            "time_index": slot,
            "provider_recording_id": recording,
            "provider_track_id": track,
            "source_frame": frame,
            "nominal_time_seconds": frame / SOURCE_FPS,
            "offset_seconds": offset / SOURCE_FPS,
            "image_member": member,
            "image_width": native.IMAGE_SIZE[0],
            "image_height": native.IMAGE_SIZE[1],
            "annotation_occluded": int(annotation.occluded) if annotation else "",
            "annotation_generated": int(annotation.generated) if annotation else "",
            "valid_geometry": int(geometry_ok),
            "image_present": int(present),
            "valid_frame": int(not reasons),
            "missing_reason": ";".join(reasons),
            "source_frame_nonnegative": int(frame >= 0),
        }
        for key, value in zip(
            ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"),
            box if box is not None else ("", "", "", ""),
            strict=True,
        ):
            item[key] = value
        frames.append(item)
    return frames


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_long_manifest(
    *,
    source_lock_path: Path,
    output_dir: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    source_lock_path = _safe_path(source_lock_path, root=root)
    output_dir = _safe_path(output_dir, root=root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("P3 manifest output must be new or empty")
    source_lock = validate_materialization_lock(root, source_lock_path)
    lock_hash = native.sha256_file(source_lock_path)
    index_path = Path(source_lock["inputs"]["eligible_index"]["path"])
    archive_path = Path(source_lock["inputs"]["archive"]["path"])
    primary = native.load_primary_index(index_path)
    by_recording: dict[str, list[dict[str, str]]] = {}
    for row in primary:
        by_recording.setdefault(row["provider_recording_id"], []).append(row)
    frames_by_id, image_rows = {}, {}
    accessed: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        allowed_annotations = set(annotation_allowlist())
        names = Counter(info.filename for info in archive.infolist())
        requested_images = {
            native.frame_member(recording, int(row["center_frame"]) + offset)
            for recording, rows in by_recording.items()
            for row in rows
            for offset in SOURCE_OFFSETS
            if int(row["center_frame"]) + offset >= 0
        }
        if any(names[name] != 1 for name in allowed_annotations):
            raise RuntimeError("Missing or duplicate exact permitted annotation member")
        if any(names[name] > 1 for name in requested_images):
            raise RuntimeError("Duplicate requested JPEG member")
        for recording in sorted(by_recording):
            stable = native.read_stable_annotations(archive, recording, accessed)
            for row in by_recording[recording]:
                frames_by_id[row["sample_id"]] = long_frame_rows(row, stable, archive, image_rows)
    if sorted(accessed) != annotation_allowlist():
        raise RuntimeError("P3 annotation access exceeded the exact 42-member contract")
    # Revalidate the same committed sources and opaque receipts after annotation access.
    if validate_materialization_lock(root, source_lock_path) != source_lock:
        raise RuntimeError("P3 sources changed during materialization")
    if native.sha256_file(source_lock_path) != lock_hash:
        raise RuntimeError("P3 source lock bytes changed during materialization")
    clips, flattened = [], []
    for row in primary:
        frames = frames_by_id[row["sample_id"]]
        valid_count = sum(item["valid_frame"] for item in frames)
        clips.append(
            {
                **row,
                "valid_frame_count": valid_count,
                "all_frames_valid": int(valid_count == 16),
                "center_frame_valid": frames[CENTER_SLOT]["valid_frame"],
                "missing_input_policy": "retain_center_and_all_slots;short_video_anchor_requires_later_lock",
            }
        )
        flattened.extend(frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "clip_index.csv", native.CLIP_FIELDS, clips)
    _write_csv(output_dir / "frame_manifest.csv", FRAME_FIELDS, flattened)
    _write_csv(
        output_dir / "image_allowlist.csv",
        native.IMAGE_FIELDS,
        [image_rows[k] for k in sorted(image_rows)],
    )
    summary = {
        "status": MANIFEST_STATUS,
        "primary_rows": len(clips),
        "primary_scenarios": len(native.EXPECTED_SCENARIOS),
        "primary_recordings": len(by_recording),
        "frame_rows": len(flattened),
        "unique_present_image_members": len(image_rows),
        "valid_frame_rows": sum(item["valid_frame"] for item in flattened),
        "all_frames_valid_clips": sum(item["all_frames_valid"] for item in clips),
        "center_frame_valid_clips": sum(item["center_frame_valid"] for item in clips),
        "negative_requested_frame_rows": sum(item["source_frame"] < 0 for item in flattened),
        "sampling": source_lock["sampling"],
        "preprocessing": source_lock["preprocessing"],
        "source_image_size": list(native.IMAGE_SIZE),
        "annotation_image_size": [3840, 2160],
        "sample_ids_sha256": source_lock["sample_ids_sha256"],
        "pilot_sample_ids": source_lock["pilot_sample_ids"],
        "time_provenance": "provider_frame_index_divided_by_nominal_30fps;not_measured_video_pts",
        "source_sha256": {**source_lock["source_sha256"], "materialization_lock": lock_hash},
        "artifact_sha256": {
            name: native.sha256_file(output_dir / name)
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
            "annotation_action_values_used_for_sampling": 0,
        },
        "missing_policy": "no_dropped_centers;explicit_masks;no_cross_track_fill;no_label_boundary_cutting",
        "prospective_fallback": "short-video anchor; not executed or authorized by this manifest lock",
        "diagnostic_strata_provenance": "copied original short-window flags; not recomputed as long-window strata",
        "extraction_and_fitting_authorized": False,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary
