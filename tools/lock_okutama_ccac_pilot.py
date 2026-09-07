"""Lock the label-blind Okutama CCAC feasibility cohort before JPEG decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hac import okutama_native_video as native  # noqa: E402
from hac.ccac_motion import SIZE_BANDS, native_size_band, select_pilot_candidates  # noqa: E402
from tools import lock_hac_continuation_protocols as common  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_ccac_pilot_protocol.json"
PROTOCOL_STATUS = "DECLARED_AFTER_P7_BEFORE_CCAC_PILOT_PIXEL_ACCESS"
STATUS = "OKUTAMA_CCAC_PILOT_LOCKED_BEFORE_PIXEL_ACCESS"
AUTHORIZATION = {
    "read_locked_label_free_frame_metadata": True,
    "opaque_hash_named_archive": True,
    "decode_selected_jpeg_members_after_lock": True,
    "fixed_geometric_estimation_after_lock": True,
    "write_local_diagnostic_overlays_after_lock": True,
    "read_action_labels": False,
    "read_model_predictions_or_errors": False,
    "fit_action_classifier": False,
    "read_protected_calibration_or_test_data": False,
    "download_external_data_or_models": False,
}
SOURCES = {
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_ccac_pilot.py",
    "runner": "experiments/run_okutama_ccac_pilot.py",
    "ccac_module": "src/hac/ccac_motion.py",
    "native_video_module": "src/hac/okutama_native_video.py",
    "common_locker": "tools/lock_hac_continuation_protocols.py",
    "decision_report": "docs/HAC_P7_RESULTS_AND_P8_DECISION_20260907.md",
    "requirements": "requirements-video-lock.txt",
}
P3_FIELDS = (
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
    "source_frame_nonnegative",
)
P0_FIELDS = P3_FIELDS[:-1]
ALLOWLIST_FIELDS = (
    "image_member",
    "provider_recording_id",
    "source_frame",
    "crc32",
    "size_bytes",
)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before the CCAC pilot lock")
    return (
        common._git(root, "rev-parse", "HEAD"),
        common._git(root, "rev-parse", "HEAD^{tree}"),
    )


def _repo_path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    common.assert_role_safe_input_path(root, path)
    return path


def _checked_artifact(root: Path, declared: dict[str, Any]) -> tuple[Path, bytes]:
    path = _repo_path(root, declared["path"])
    raw = common._read_bytes(path)
    if hashlib.sha256(raw).hexdigest() != declared["sha256"] or len(raw) != int(
        declared["size_bytes"]
    ):
        raise RuntimeError(f"CCAC input receipt changed: {declared['path']}")
    return path, raw


def _strict_bool(value: str) -> bool:
    if value not in {"0", "1"}:
        raise RuntimeError(f"Expected a canonical binary manifest value, found {value!r}")
    return value == "1"


def _optional_bool(value: str) -> bool | None:
    return None if value == "" else _strict_bool(value)


def _box(row: dict[str, str]) -> list[float] | None:
    values = [row[name] for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")]
    if all(value == "" for value in values):
        return None
    if any(value == "" for value in values):
        raise RuntimeError("CCAC manifest contains a partially missing box")
    box = [float(value) for value in values]
    if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise RuntimeError("CCAC manifest contains invalid native box geometry")
    return box


def _csv_rows(raw: bytes, fields: tuple[str, ...]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != fields:
        raise RuntimeError("CCAC input CSV schema changed")
    return list(reader)


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    metadata = spec.get("metadata_contract", {})
    selection = spec.get("selection", {})
    runtime = spec.get("runtime", {})
    background = spec.get("background_correspondence", {})
    actor = spec.get("actor_correspondence", {})
    camera = spec.get("camera_hypotheses", {})
    decomposition = spec.get("motion_decomposition", {})
    gates = spec.get("engineering_gates", {})
    decision = spec.get("decision_rule", {})
    if (
        (spec.get("protocol_version"), spec.get("status"), spec.get("declared_on"))
        != ("1.0.0", PROTOCOL_STATUS, "2026-09-07")
        or metadata.get("candidate_rows") != 4977
        or metadata.get("candidate_scenarios") != 11
        or metadata.get("candidate_recordings") != 21
        or metadata.get("candidate_tracks") != 444
        or metadata.get("long_complete_rows") != 4510
        or metadata.get("long_incomplete_rows") != 467
        or metadata.get("frame_slots") != 16
        or metadata.get("center_slot") != 8
        or metadata.get("source_offsets") != list(range(-32, 32, 4))
        or metadata.get("source_fps") != 30.0
        or metadata.get("image_size") != [1280, 720]
        or selection.get("hash_domain") != "hac-ccac-p8-pilot-v1|20260907|"
        or selection.get("selected_clips") != 128
        or selection.get("size_targets") != {"small": 48, "medium": 48, "large": 32}
        or selection.get("long_incomplete_target") != 12
        or selection.get("replacement_after_pixel_failure") is not False
        or selection.get("labels_used") is not False
        or runtime.get("opencv_version") != "5.0.0"
        or runtime.get("opencv_threads") != 1
        or runtime.get("opencv_opencl") is not False
        or runtime.get("maximum_new_clip_workloads") != 128
        or runtime.get("exact_replay_clips") != 2
        or background.get("spatial_grid") != [4, 3]
        or background.get("maximum_corners_total") != 800
        or background.get("known_actor_expansion_pixels") != 10.0
        or background.get("forward_backward_max_pixels") != 1.0
        or background.get("minimum_retained_correspondences") != 40
        or background.get("minimum_occupied_cells") != 6
        or background.get("minimum_selection_correspondences") != 8
        or background.get("minimum_audit_correspondences") != 8
        or actor.get("maximum_corners") != 32
        or actor.get("seed_region_erode_fraction") != 0.1
        or actor.get("forward_backward_max_pixels") != 1.0
        or actor.get("sensitivity_lk_window") != [15, 15]
        or actor.get("pairwise_reseeding") is not True
        or actor.get("no_grid_fallback") is not True
        or camera.get("ransac_reprojection_pixels") != 2.0
        or camera.get("upgrade_minimum_absolute_pixels") != 0.25
        or camera.get("upgrade_minimum_relative_fraction") != 0.2
        or decomposition.get("translation_minimum_points") != 4
        or decomposition.get("articulation_minimum_points") != 6
        or decomposition.get("articulation_minimum_vertical_regions") != 2
        or gates.get("minimum_selection_correspondences") != 8
        or gates.get("minimum_audit_correspondences") != 8
        or gates.get("camera_usable_pair_fraction_at_least") != 0.9
        or gates.get("full_articulation_clip_fraction_at_least") != 0.7
        or gates.get("minimum_scenarios_with_half_clips_usable") != 8
        or gates.get("minimum_clip_fraction_per_native_size_band") != 0.5
        or decision.get("precedence")
        != [
            "full_ccac_extraction",
            "narrow_translation_or_tracker_pilot",
            "abandon_initial_lk_extractor",
        ]
        or decision.get("supervised_access") is not False
        or spec.get("authorization", {}).get("read_action_labels") is not False
        or spec.get("authorization", {}).get("read_model_predictions_or_errors") is not False
        or spec.get("authorization", {}).get("fit_action_classifier") is not False
        or spec.get("authorization", {}).get("read_protected_calibration_or_test_data") is not False
    ):
        raise RuntimeError("CCAC feasibility protocol changed")
    return spec


def _validate_upstream_locks(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("p0_r1_extraction_lock", "p3_extraction_lock"):
        path, raw = _checked_artifact(root, spec["inputs"][name])
        payload = json.loads(raw)
        expected = spec["inputs"][name]["status"]
        if payload.get("status") != expected:
            raise RuntimeError(f"Unexpected status for {name}")
        output[name] = {
            "path": common._relative_path(root, path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "status": expected,
        }
    return output


def _validate_manifests(
    root: Path, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    _, p3_raw = _checked_artifact(root, spec["inputs"]["p3_frame_manifest"])
    _, p0_raw = _checked_artifact(root, spec["inputs"]["p0_r1_frame_manifest"])
    p3 = _csv_rows(p3_raw, P3_FIELDS)
    p0 = _csv_rows(p0_raw, P0_FIELDS)
    expected_rows = int(spec["metadata_contract"]["candidate_rows"])
    slots = int(spec["metadata_contract"]["frame_slots"])
    if len(p3) != expected_rows * slots or len(p0) != expected_rows * slots:
        raise RuntimeError("CCAC frame-manifest cardinality changed")
    p0_centers = {row["sample_id"]: row for row in p0 if int(row["time_index"]) == 8}
    if len(p0_centers) != expected_rows:
        raise RuntimeError("CCAC P0 center identities changed")
    by_sample: dict[str, list[dict[str, str]]] = {}
    known_manifest_members: set[str] = set()
    for row in p3:
        by_sample.setdefault(row["sample_id"], []).append(row)
        if row["image_member"]:
            known_manifest_members.add(row["image_member"])
    if len(by_sample) != expected_rows:
        raise RuntimeError("CCAC P3 sample identities changed")
    candidates: list[dict[str, Any]] = []
    offsets = spec["metadata_contract"]["source_offsets"]
    center_slot = int(spec["metadata_contract"]["center_slot"])
    center_compare = (
        "sample_id",
        "provider_recording_id",
        "provider_track_id",
        "source_frame",
        "image_member",
        "image_width",
        "image_height",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "valid_geometry",
        "image_present",
        "valid_frame",
    )
    for sample_id in sorted(by_sample):
        rows = sorted(by_sample[sample_id], key=lambda row: int(row["time_index"]))
        recording, track, center = native.parse_sample_id(sample_id)
        scenario = ".".join(recording.split(".")[1:])
        if len(rows) != slots or [int(row["time_index"]) for row in rows] != list(range(slots)):
            raise RuntimeError("CCAC P3 frame slots changed")
        frames: list[dict[str, Any]] = []
        for slot, (row, offset) in enumerate(zip(rows, offsets, strict=True)):
            number = center + int(offset)
            valid = _strict_bool(row["valid_frame"])
            geometry = _strict_bool(row["valid_geometry"])
            present = _strict_bool(row["image_present"])
            box = _box(row)
            expected_member = native.frame_member(recording, number) if number >= 0 else ""
            if (
                row["provider_recording_id"] != recording
                or row["provider_track_id"] != track
                or int(row["source_frame"]) != number
                or not math.isclose(float(row["nominal_time_seconds"]), number / 30.0)
                or not math.isclose(float(row["offset_seconds"]), int(offset) / 30.0)
                or row["image_member"] != expected_member
                or int(row["image_width"]) != 1280
                or int(row["image_height"]) != 720
                or _strict_bool(row["source_frame_nonnegative"]) != (number >= 0)
                or valid != (geometry and present)
                or geometry != (box is not None)
            ):
                raise RuntimeError("CCAC P3 identity, timing, or validity contract changed")
            frames.append(
                {
                    "time_index": slot,
                    "source_frame": number,
                    "nominal_time_seconds": float(row["nominal_time_seconds"]),
                    "offset_seconds": float(row["offset_seconds"]),
                    "image_member": row["image_member"] or None,
                    "image_width": 1280,
                    "image_height": 720,
                    "bbox": box,
                    "annotation_occluded": _optional_bool(row["annotation_occluded"]),
                    "annotation_generated": _optional_bool(row["annotation_generated"]),
                    "valid_geometry": geometry,
                    "image_present": present,
                    "valid_frame": valid,
                    "missing_reason": row["missing_reason"] or None,
                }
            )
        center_row = rows[center_slot]
        old = p0_centers.get(sample_id)
        if old is None or any(old[field] != center_row[field] for field in center_compare):
            raise RuntimeError("CCAC P0/P3 center geometry identity changed")
        center_box = frames[center_slot]["bbox"]
        if not frames[center_slot]["valid_frame"] or center_box is None:
            raise RuntimeError("CCAC candidate lacks its native center anchor")
        height = float(center_box[3] - center_box[1])
        candidates.append(
            {
                "sample_id": sample_id,
                "scenario": scenario,
                "provider_recording_id": recording,
                "provider_track_id": track,
                "track_key": f"{recording}::{track}",
                "center_frame": center,
                "native_center_height": height,
                "size_band": native_size_band(height),
                "long_valid": all(frame["valid_frame"] for frame in frames),
                "any_known_occlusion": any(
                    frame["annotation_occluded"] is True for frame in frames
                ),
                "frames": frames,
            }
        )
    metadata = spec["metadata_contract"]
    if (
        len({row["scenario"] for row in candidates}) != metadata["candidate_scenarios"]
        or len({row["provider_recording_id"] for row in candidates})
        != metadata["candidate_recordings"]
        or len({row["track_key"] for row in candidates}) != metadata["candidate_tracks"]
        or sum(row["long_valid"] for row in candidates) != metadata["long_complete_rows"]
        or sum(not row["long_valid"] for row in candidates) != metadata["long_incomplete_rows"]
    ):
        raise RuntimeError("CCAC label-free metadata population changed")
    return (
        candidates,
        p3,
        {
            "p0_p3_center_exact_matches": len(candidates),
            "manifest_member_count": len(known_manifest_members),
        },
    )


def _archive_and_boxes(
    root: Path,
    spec: dict[str, Any],
    selected: list[dict[str, Any]],
    p3_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[list[float]]]]:
    declared_archive = spec["inputs"]["archive"]
    archive_path = Path(declared_archive["path"]).resolve()
    if archive_path.name != "TrainSetFrames.zip" or not archive_path.is_file():
        raise RuntimeError("CCAC archive is not the named Okutama training-frame archive")
    digest, size = common._sha256_file(archive_path)
    if digest != declared_archive["sha256"] or size != int(declared_archive["size_bytes"]):
        raise RuntimeError("CCAC opaque archive receipt changed")
    _, allowlist_raw = _checked_artifact(root, spec["inputs"]["p3_image_allowlist"])
    allowlist_rows = _csv_rows(allowlist_raw, ALLOWLIST_FIELDS)
    allowlist: dict[str, dict[str, Any]] = {}
    for row in allowlist_rows:
        member = row["image_member"]
        if member in allowlist:
            raise RuntimeError("CCAC allowlist contains duplicate members")
        allowlist[member] = {
            "image_member": member,
            "provider_recording_id": row["provider_recording_id"],
            "source_frame": int(row["source_frame"]),
            "crc32": int(row["crc32"]),
            "size_bytes": int(row["size_bytes"]),
        }
    selected_members = sorted(
        {
            frame["image_member"]
            for clip in selected
            for frame in clip["frames"]
            if frame["valid_frame"] and frame["image_member"] is not None
        }
    )
    if any(member not in allowlist for member in selected_members):
        raise RuntimeError("CCAC selected a member outside the P3 allowlist")
    with zipfile.ZipFile(archive_path) as archive:
        counts = Counter(
            info.filename for info in archive.infolist() if info.filename in selected_members
        )
        if any(counts[member] != 1 for member in selected_members):
            raise RuntimeError("CCAC selected archive member is absent or duplicated")
        for member in selected_members:
            info = archive.getinfo(member)
            declared = allowlist[member]
            if info.CRC != declared["crc32"] or info.file_size != declared["size_bytes"]:
                raise RuntimeError("CCAC selected archive-member metadata changed")
    selected_set = set(selected_members)
    box_sets: dict[str, set[tuple[float, float, float, float]]] = {
        member: set() for member in selected_members
    }
    for row in p3_rows:
        member = row["image_member"]
        if member in selected_set and _strict_bool(row["valid_geometry"]):
            box = _box(row)
            if box is None:
                raise RuntimeError("CCAC valid geometry lacks a box")
            box_sets[member].add(tuple(box))
    known_boxes = {
        member: [list(box) for box in sorted(box_sets[member])] for member in selected_members
    }
    if any(not boxes for boxes in known_boxes.values()):
        raise RuntimeError("CCAC selected image has no known actor box")
    return (
        {"path": archive_path.as_posix(), "sha256": digest, "size_bytes": size},
        [allowlist[member] for member in selected_members],
        known_boxes,
    )


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, relative, commit)
        for name, relative in SOURCES.items()
    }
    upstream_locks = _validate_upstream_locks(root, spec)
    _, p7_raw = _checked_artifact(root, spec["design_inputs"]["p7_result"])
    if json.loads(p7_raw).get("status") != spec["design_inputs"]["p7_result"]["status"]:
        raise RuntimeError("CCAC P7 decision input status changed")
    candidates, p3_rows, manifest_audit = _validate_manifests(root, spec)
    selected = select_pilot_candidates(candidates, spec["selection"])
    archive, selected_members, known_boxes = _archive_and_boxes(root, spec, selected, p3_rows)
    selected_ids = [row["sample_id"] for row in selected]
    scenario_counts = Counter(row["scenario"] for row in selected)
    size_counts = Counter(row["size_band"] for row in selected)
    overlay_ids = sorted(
        selected_ids,
        key=lambda value: (
            hashlib.sha256(
                (spec["selection"]["hash_domain"] + "overlay:" + value).encode()
            ).hexdigest(),
            value,
        ),
    )[:12]
    replay_ids = sorted(
        selected_ids,
        key=lambda value: (
            hashlib.sha256(
                (spec["selection"]["hash_domain"] + "replay:" + value).encode()
            ).hexdigest(),
            value,
        ),
    )[:2]
    opencv_build = cv2.getBuildInformation()
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: receipt["sha256"] for name, receipt in sources.items()},
        "upstream_locks": upstream_locks,
        "archive": archive,
        "selected_clips": selected,
        "selected_sample_ids_sha256": canonical_digest(selected_ids),
        "selected_clip_records_sha256": canonical_digest(selected),
        "selection_summary": {
            "clips": len(selected),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "size_counts": {band: size_counts[band] for band in SIZE_BANDS},
            "long_complete": sum(row["long_valid"] for row in selected),
            "long_incomplete": sum(not row["long_valid"] for row in selected),
            "distinct_tracks": len({row["track_key"] for row in selected}),
            "maximum_track_repeats": max(Counter(row["track_key"] for row in selected).values()),
        },
        "selected_image_members": selected_members,
        "selected_image_members_sha256": canonical_digest(selected_members),
        "known_boxes_by_member": known_boxes,
        "known_boxes_sha256": canonical_digest(known_boxes),
        "overlay_sample_ids": overlay_ids,
        "replay_sample_ids": replay_ids,
        "manifest_audit": manifest_audit,
        "opencv": {
            "version": cv2.__version__,
            "build_information_sha256": hashlib.sha256(opencv_build.encode()).hexdigest(),
        },
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "label_free_manifest_rows_read": len(p3_rows) + 4977 * 16,
            "archive_bytes_hashed_opaquely": archive["size_bytes"],
            "archive_directory_entries_inspected": len(selected_members),
            "jpeg_payloads_decoded": 0,
            "action_labels_read": 0,
            "prediction_rows_read": 0,
            "classifier_fits": 0,
            "protected_rows_read": 0,
        },
    }


def compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained CCAC pilot lock changed: " + ", ".join(changed))


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _repo_path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing CCAC pilot lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_repo_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the CCAC pre-pixel execution lock")
    compare(retained, build_payload(root))
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    payload = validate_lock(root, output) if args.mode == "check" else build_payload(root)
    if args.mode == "lock":
        write_lock(root, output, payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload["status"],
                "selected_clips": len(payload["selected_clips"]),
                "selection_summary": payload["selection_summary"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
