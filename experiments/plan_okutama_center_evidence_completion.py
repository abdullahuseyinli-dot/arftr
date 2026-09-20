"""Label-blind dependency and geometry plan for center evidence completion."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_center_evidence_completion_protocol.json"
PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
FRAME_MANIFEST = (
    ROOT
    / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/frame_manifest.csv"
)
IMAGE_ALLOWLIST = (
    ROOT
    / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/image_allowlist.csv"
)
DEFAULT_RUN = ROOT / ".runs/research_20260913/center_evidence_completion_v1"
EXPECTED_PROTOCOL_SHA256 = "f2e95656cfd179e72522d02103d2542e96b4124ccb76c5f4c0c1706783505244"
SOURCE_TIME_INDICES = (0, 4, 8, 12, 15)
NEIGHBOR_TIME_INDICES = (0, 4, 12, 15)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _intersection_area(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def plan(run: Path) -> dict[str, Any]:
    protocol_hash = sha256_file(PROTOCOL)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Center-completion protocol changed after the planner was frozen")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (
        protocol["schema_version"] != 8
        or protocol["status"] != "LABEL_BLIND_128_CENTER_EXTRACTION_AUTHORIZED_NO_TASK_FIT"
        or protocol["pilot"]["task_labels_read"] != 0
        or protocol["pilot"]["pose_outputs_read"] != 0
    ):
        raise RuntimeError("Center-completion planning authorization differs")
    selection = json.loads((PILOT / "pilot_selection.json").read_text(encoding="utf-8"))
    crops = json.loads((PILOT / "crop_manifest.json").read_text(encoding="utf-8"))
    selected_ids = [row["sample_id"] for row in selection]
    if (
        len(selected_ids) != 128
        or len(set(selected_ids)) != 128
        or selected_ids != [row["sample_id"] for row in crops]
    ):
        raise RuntimeError("Frozen label-blind pilot population differs")

    selected = set(selected_ids)
    first_per_scenario: dict[str, int] = {}
    for index, row in enumerate(selection):
        first_per_scenario.setdefault(row["scenario"], index)
    preflight_indices = set(first_per_scenario.values())
    remaining = sorted(
        (index for index in range(len(selection)) if index not in preflight_indices),
        key=lambda index: hashlib.sha256(
            f"center-completion-preflight-v1|{selected_ids[index]}".encode()
        ).hexdigest(),
    )
    preflight_indices.update(remaining[: 16 - len(preflight_indices)])
    preflight_indices = set(sorted(preflight_indices))
    if len(preflight_indices) != 16:
        raise RuntimeError("Label-blind preflight selection did not produce 16 centers")
    preflight_rows = [selection[index] for index in sorted(preflight_indices)]
    if len({row["scenario"] for row in preflight_rows}) != 11:
        raise RuntimeError("Label-blind preflight does not cover every scenario")
    source_rows: dict[tuple[str, int], dict[str, str]] = {}
    donors: dict[tuple[str, int], dict[str, set[tuple[float, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    with FRAME_MANIFEST.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        forbidden_columns = {"label", "label_index", "action", "support_category"}
        if forbidden_columns & set(reader.fieldnames or ()):
            raise RuntimeError("Frame geometry manifest unexpectedly contains action fields")
        all_rows = list(reader)
    for row in all_rows:
        box = tuple(
            float(row[field])
            for field in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
        )
        donors[(row["provider_recording_id"], int(row["source_frame"]))][
            row["provider_track_id"]
        ].add(box)
        if row["sample_id"] in selected and int(row["time_index"]) in SOURCE_TIME_INDICES:
            key = (row["sample_id"], int(row["time_index"]))
            if key in source_rows:
                raise RuntimeError("Duplicate selected source geometry row")
            source_rows[key] = row
    if len(source_rows) != 128 * 5:
        raise RuntimeError("Selected pilot is missing fixed center/neighbor source rows")

    with IMAGE_ALLOWLIST.open(encoding="utf-8", newline="") as stream:
        allowlist_rows = list(csv.DictReader(stream))
    allowed_members = {row["image_member"] for row in allowlist_rows}
    if len(allowed_members) != len(allowlist_rows):
        raise RuntimeError("Pinned image allowlist contains duplicate members")
    selected_members = {
        row["image_member"] for row in source_rows.values() if row["image_member"]
    }
    if not selected_members <= allowed_members:
        raise RuntimeError("Selected source member falls outside the pinned allowlist")

    archive_path = Path(protocol["pinned_evidence"]["provider_frame_archive"]["path"])
    archive_size = archive_path.stat().st_size
    expected_archive_size = protocol["pinned_evidence"]["provider_frame_archive"][
        "size_bytes"
    ]
    if archive_size != expected_archive_size:
        raise RuntimeError("Provider frame archive byte count differs from the source lock")

    geometry_valid = 0
    image_present = 0
    neighbor_valid = 0
    wrong_track_slots = 0
    centers_with_all_neighbors = 0
    centers_with_wrong_track = 0
    box_heights: list[float] = []
    box_widths: list[float] = []
    scenario_counts = Counter(row["scenario"] for row in selection)
    center_valid = 0
    wrong_track_map: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        center_row = source_rows[(sample_id, 8)]
        center_valid += int(center_row["valid_frame"])
        rows = [source_rows[(sample_id, index)] for index in NEIGHBOR_TIME_INDICES]
        center_all_valid = True
        center_has_donor = False
        for row in rows:
            geometry_valid += int(row["valid_geometry"])
            image_present += int(row["image_present"])
            neighbor_valid += int(row["valid_frame"])
            center_all_valid &= row["valid_frame"] == "1"
            width = float(row["bbox_xmax"]) - float(row["bbox_xmin"])
            height = float(row["bbox_ymax"]) - float(row["bbox_ymin"])
            if width <= 0 or height <= 0:
                raise RuntimeError("Nonpositive selected neighbor box")
            box_widths.append(width)
            box_heights.append(height)
            current_box = (
                float(row["bbox_xmin"]),
                float(row["bbox_ymin"]),
                float(row["bbox_xmax"]),
                float(row["bbox_ymax"]),
            )
            candidates = donors[
                (row["provider_recording_id"], int(row["source_frame"]))
            ]
            donor_candidates = []
            for track, boxes in candidates.items():
                if track == row["provider_track_id"]:
                    continue
                for candidate in boxes:
                    if _intersection_area(current_box, candidate) != 0:
                        continue
                    canonical = "|".join(
                        [
                            sample_id,
                            row["time_index"],
                            track,
                            *(format(value, ".12g") for value in candidate),
                        ]
                    )
                    donor_candidates.append(
                        (
                            hashlib.sha256(
                                f"center-completion-wrong-track-v1|{canonical}".encode()
                            ).hexdigest(),
                            canonical,
                            track,
                            candidate,
                        )
                    )
            chosen = min(donor_candidates) if donor_candidates else None
            has_nonoverlapping_donor = chosen is not None
            wrong_track_map.append(
                {
                    "sample_id": sample_id,
                    "time_index": int(row["time_index"]),
                    "source_frame": int(row["source_frame"]),
                    "image_member": row["image_member"],
                    "source_track_id": row["provider_track_id"],
                    "donor_available": has_nonoverlapping_donor,
                    "donor_track_id": chosen[2] if chosen else None,
                    "donor_box": list(chosen[3]) if chosen else None,
                    "selection_sha256": chosen[0] if chosen else None,
                }
            )
            wrong_track_slots += has_nonoverlapping_donor
            center_has_donor |= has_nonoverlapping_donor
        centers_with_all_neighbors += center_all_valid
        centers_with_wrong_track += center_has_donor

    wrong_track_artifact = run / "wrong_track_donor_map_v4.json"
    _write_json(
        wrong_track_artifact,
        {
            "status": "FIXED_LABEL_BLIND_WRONG_TRACK_MAP_COMPLETE",
            "salt": "center-completion-wrong-track-v1",
            "protocol_sha256": protocol_hash,
            "rows": wrong_track_map,
        },
    )
    preflight_ids = {row["sample_id"] for row in preflight_rows}
    preflight_donor_slots = sum(
        row["donor_available"] for row in wrong_track_map if row["sample_id"] in preflight_ids
    )

    result = {
        "status": "CENTER_EVIDENCE_COMPLETION_LABEL_BLIND_PLAN_V8_COMPLETE",
        "scope": "dependency/geometry inventory only; no pixels decoded, models built, features extracted, or labels fit",
        "protocol_sha256": protocol_hash,
        "population": {
            "centers": len(selected_ids),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "selection_sha256": sha256_file(PILOT / "pilot_selection.json"),
            "crop_manifest_sha256": sha256_file(PILOT / "crop_manifest.json"),
        },
        "preflight_selection": {
            "rule": protocol["compute_boundary"]["preflight_selection"],
            "centers": len(preflight_rows),
            "scenario_counts": dict(
                sorted(Counter(row["scenario"] for row in preflight_rows).items())
            ),
            "sample_ids": [row["sample_id"] for row in preflight_rows],
        },
        "neighbor_geometry": {
            "requested_rows": 128 * 4,
            "geometry_valid_rows": geometry_valid,
            "image_present_rows": image_present,
            "valid_frame_rows": neighbor_valid,
            "centers_with_all_four_valid_neighbors": centers_with_all_neighbors,
            "nonoverlapping_wrong_track_control_available_slots": wrong_track_slots,
            "centers_with_at_least_one_nonoverlapping_wrong_track_control": centers_with_wrong_track,
            "fixed_wrong_track_map": {
                "path": str(wrong_track_artifact.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(wrong_track_artifact),
                "rows": len(wrong_track_map),
                "available_rows": wrong_track_slots,
                "preflight_available_rows": preflight_donor_slots,
            },
            "box_width_720_median": float(sorted(box_widths)[len(box_widths) // 2]),
            "box_height_720_median": float(sorted(box_heights)[len(box_heights) // 2]),
            "time_indices": list(NEIGHBOR_TIME_INDICES),
            "offset_frames": protocol["pilot"]["neighbor_offsets_frames_at_30fps"],
        },
        "source_contract": {
            "condition": "provider-supplied 1280x720 JPEG frames, not original video decode",
            "center_rows_valid": center_valid,
            "center_and_neighbor_member_slots": 128 * 5,
            "unique_selected_members": len(selected_members),
            "selected_members_all_allowlisted": True,
            "image_allowlist_sha256": sha256_file(IMAGE_ALLOWLIST),
            "archive_path": str(archive_path),
            "archive_size_bytes": archive_size,
            "archive_size_matches": True,
            "archive_declared_sha256": protocol["pinned_evidence"][
                "provider_frame_archive"
            ]["sha256"],
            "archive_bytes_hashed_during_plan": 0,
            "preflight_requirement": "hash the archive and validate each selected member against pinned CRC32/size before decoding",
        },
        "planned_cost": {
            "center_unmasked_forwards": 128,
            "center_masked_forwards": 256,
            "neighbor_forwards": 512,
            "wrong_track_control_forwards": wrong_track_slots,
            "total_frozen_dino_forwards": 896 + wrong_track_slots,
            "preflight_centers": 16,
            "preflight_primary_frozen_dino_forwards": 112,
            "preflight_wrong_track_forwards": preflight_donor_slots,
            "preflight_frozen_dino_forwards": 112 + preflight_donor_slots,
            "self_supervised_small_fits_if_extraction_passes": 20,
            "task_fits": 0,
        },
        "source_accounting": {
            "frame_manifest_sha256": sha256_file(FRAME_MANIFEST),
            "frame_manifest_has_action_columns": False,
            "pixel_payloads_read": 0,
            "pose_outputs_read": 0,
            "task_label_fields_read": 0,
            "arftr_probabilities_interpreted": 0,
            "arftr_archive_bytes_hashed_for_ancestry": True,
            "arftr_sha256": sha256_file(
                ROOT / protocol["pinned_evidence"]["retained_arftr"]["path"]
            ),
        },
        "authorization_after_plan": (
            "Run one immutable 128-center label-blind extraction, independently replay its "
            "cache and dependency receipts, then run only the locked reconstruction screen."
        ),
        "full_extraction_authorized": True,
        "task_training_authorized": False,
    }
    _write_json(run / "plan_v8.json", result)
    return result


def main() -> None:
    result = plan(DEFAULT_RUN)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
