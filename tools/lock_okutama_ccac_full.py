"""Lock full label-blind Okutama CCAC extraction before nonpilot JPEG access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hac.ccac_features import (  # noqa: E402
    COMPENSATED_TRANSLATION_COLUMNS,
    QUALITY_COLUMNS,
    RAW_TRANSLATION_COLUMNS,
    WITHIN_ACTOR_COLUMNS,
)
from tools import lock_hac_continuation_protocols as common  # noqa: E402
from tools import lock_okutama_ccac_pilot as pilot  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_ccac_full_protocol.json"
PROTOCOL_STATUS = "DECLARED_AFTER_CCAC_PILOT_BEFORE_NONPILOT_PIXEL_ACCESS"
STATUS = "OKUTAMA_CCAC_FULL_LOCKED_BEFORE_NONPILOT_PIXEL_ACCESS"
FULL_ORDER_DOMAIN = "hac-ccac-p8-full-v1|20260907|order:"
AUTHORIZATION = {
    "read_pilot_measurements": True,
    "read_locked_label_free_frame_metadata": True,
    "decode_prelocked_nonpilot_jpeg_members_after_full_lock": True,
    "fixed_geometric_estimation": True,
    "derive_fixed_label_free_feature_blocks": True,
    "read_action_labels": False,
    "read_model_predictions_or_errors": False,
    "fit_action_classifier": False,
    "read_protected_calibration_or_test_data": False,
    "download_external_data_or_models": False,
}
SOURCES = {
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_ccac_full.py",
    "runner": "experiments/run_okutama_ccac_full.py",
    "feature_module": "src/hac/ccac_features.py",
    "measurement_module": "src/hac/ccac_motion.py",
    "pilot_locker": "tools/lock_okutama_ccac_pilot.py",
    "pilot_runner": "experiments/run_okutama_ccac_pilot.py",
    "native_video_module": "src/hac/okutama_native_video.py",
    "common_locker": "tools/lock_hac_continuation_protocols.py",
    "pilot_report": "docs/HAC_P8_CCAC_PILOT_RESULTS_20260907.md",
    "requirements": "requirements-video-lock.txt",
}


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before full CCAC locking")
    return common._git(root, "rev-parse", "HEAD"), common._git(root, "rev-parse", "HEAD^{tree}")


def _repo_path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    common.assert_role_safe_input_path(root, path)
    return path


def _checked(root: Path, declared: dict[str, Any]) -> tuple[Path, bytes]:
    path = _repo_path(root, declared["path"])
    raw = common._read_bytes(path)
    if hashlib.sha256(raw).hexdigest() != declared["sha256"] or len(raw) != int(
        declared["size_bytes"]
    ):
        raise RuntimeError(f"Full CCAC source receipt changed: {declared['path']}")
    return path, raw


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    cohort = spec.get("cohort", {})
    reuse = spec.get("reuse", {})
    execution = spec.get("execution", {})
    feature = spec.get("feature_recipe", {})
    arrays = spec.get("feature_artifact", {}).get("arrays", {})
    arms = spec.get("future_supervised_arms", {})
    if (
        (spec.get("protocol_version"), spec.get("status"), spec.get("declared_on"))
        != ("1.0.0", PROTOCOL_STATUS, "2026-09-07")
        or cohort
        != {
            "clips": 4977,
            "scenarios": 11,
            "recordings": 21,
            "tracks": 444,
            "requested_frame_slots": 79632,
            "valid_frame_slots": 78714,
            "unique_valid_jpeg_members": 13093,
            "requested_adjacent_pairs": 74655,
            "available_adjacent_pairs": 73737,
            "unique_available_jpeg_pairs": 12854,
            "long_complete_clips": 4510,
            "long_incomplete_clips": 467,
            "ordering": (
                "retain the 128 pilot clips in their locked ranks 0..127, then order "
                "remaining rows by SHA256('hac-ccac-p8-full-v1|20260907|order:'"
                "+sample_id), sample_id"
            ),
            "row_dropping": False,
        }
        or spec.get("measurement_contract", {}).get("threshold_changes_after_pilot") is not False
        or spec.get("measurement_contract", {}).get("pair_seed_domain")
        != "hac-ccac-p8-pilot-v1|20260907|"
        or reuse.get("pilot_clip_workloads") != 128
        or reuse.get("new_clip_workloads") != 4849
        or reuse.get("selection_rank_is_provenance_only") is not True
        or execution.get("opencv_threads_per_process") != 1
        or execution.get("opencv_opencl") is not False
        or execution.get("fixed_shard_count") != 8
        or execution.get("per_process_decompressed_jpeg_byte_lru_members") != 128
        or execution.get("new_nonpilot_exact_replay_clips") != 2
        or feature.get("quality_columns") != list(QUALITY_COLUMNS)
        or feature.get("quality_width") != 16
        or feature.get("translation_channels")
        != ["x_height_per_second", "y_height_per_second", "norm_height_per_second"]
        or feature.get("raw_translation_width") != 9
        or feature.get("compensated_translation_width") != 9
        or feature.get("within_actor_channels")
        != ["median_height_per_second", "p90_height_per_second"]
        or feature.get("within_actor_width") != 6
        or feature.get("temporal_quantiles") != [0.1, 0.5, 0.9]
        or feature.get("quantile_method") != "linear"
        or arrays.get("sample_ids") != [4977]
        or arrays.get("quality") != [4977, 16]
        or arrays.get("raw_translation") != [4977, 9]
        or arrays.get("compensated_translation") != [4977, 9]
        or arrays.get("within_actor") != [4977, 6]
        or arms.get("A0_spatial_refit") != {"posture_width": 9984, "motion_width": 4608}
        or arms.get("Q_reliability") != {"posture_width": 9984, "motion_width": 4624}
        or arms.get("R_raw_translation") != {"posture_width": 9984, "motion_width": 4633}
        or arms.get("C_compensated_translation") != {"posture_width": 9984, "motion_width": 4633}
        or arms.get("F_primary") != {"posture_width": 9984, "motion_width": 4639}
        or arms.get("authorization_in_this_stage") is not False
        or spec.get("authorization") != AUTHORIZATION
    ):
        raise RuntimeError("Full CCAC protocol changed")
    if (
        len(RAW_TRANSLATION_COLUMNS) != 9
        or len(COMPENSATED_TRANSLATION_COLUMNS) != 9
        or len(WITHIN_ACTOR_COLUMNS) != 6
    ):
        raise RuntimeError("Full CCAC feature source widths changed")
    return spec


def _pilot_evidence(root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name in ("execution_lock", "request", "summary", "completion"):
        path, raw = _checked(root, spec["pilot_authorization"][name])
        payload = json.loads(raw)
        payloads[name] = payload
        evidence[name] = {
            "path": common._relative_path(root, path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    lock, summary, completion = (
        payloads["execution_lock"],
        payloads["summary"],
        payloads["completion"],
    )
    if (
        lock.get("status") != pilot.STATUS
        or summary.get("status") != "OKUTAMA_CCAC_PILOT_COMPLETE"
        or summary.get("decision") != "FULL_CCAC_EXTRACTION_AUTHORIZED"
        or summary.get("gate_results")
        != {name: True for name in spec["pilot_authorization"]["passed_gates"]}
        or completion.get("status") != "OKUTAMA_CCAC_PILOT_PUBLICATION_COMPLETE"
        or completion.get("summary_sha256") != evidence["summary"]["sha256"]
    ):
        raise RuntimeError("Full CCAC pilot authorization evidence changed")
    pilot_output = _repo_path(root, spec["pilot_authorization"]["summary"]["path"]).parent
    for name, digest in summary.get("artifacts", {}).items():
        path = _repo_path(root, pilot_output / name)
        if common._sha256_file(path)[0] != digest:
            raise RuntimeError(f"Pilot artifact changed before full reuse: {name}")
    return lock, evidence


def _full_selection(
    candidates: list[dict[str, Any]], pilot_lock: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates_by_id = {row["sample_id"]: row for row in candidates}
    selected: list[dict[str, Any]] = []
    pilot_ids: set[str] = set()
    for retained in pilot_lock["selected_clips"]:
        sample_id = retained["sample_id"]
        candidate = dict(candidates_by_id[sample_id])
        candidate["selection_rank"] = retained["selection_rank"]
        if candidate != retained:
            raise RuntimeError("Pilot clip metadata changed before full extraction")
        selected.append(candidate)
        pilot_ids.add(sample_id)
    remaining = [row for row in candidates if row["sample_id"] not in pilot_ids]
    remaining.sort(
        key=lambda row: (
            hashlib.sha256((FULL_ORDER_DOMAIN + row["sample_id"]).encode()).hexdigest(),
            row["sample_id"],
        )
    )
    for row in remaining:
        retained = dict(row)
        retained["selection_rank"] = len(selected)
        selected.append(retained)
    if len(selected) != 4977 or len({row["sample_id"] for row in selected}) != 4977:
        raise RuntimeError("Full CCAC selected population changed")
    return selected


def _cohort_audit(selected: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    members: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    valid_slots = available_pairs = 0
    for clip in selected:
        frames = clip["frames"]
        for frame in frames:
            if frame["valid_frame"]:
                valid_slots += 1
                members.add(frame["image_member"])
        for previous, current in zip(frames[:-1], frames[1:], strict=True):
            if previous["valid_frame"] and current["valid_frame"]:
                available_pairs += 1
                pairs.add((previous["image_member"], current["image_member"]))
    expected = spec["cohort"]
    observed = {
        "clips": len(selected),
        "scenarios": len({row["scenario"] for row in selected}),
        "recordings": len({row["provider_recording_id"] for row in selected}),
        "tracks": len({row["track_key"] for row in selected}),
        "requested_frame_slots": sum(len(row["frames"]) for row in selected),
        "valid_frame_slots": valid_slots,
        "unique_valid_jpeg_members": len(members),
        "requested_adjacent_pairs": len(selected) * 15,
        "available_adjacent_pairs": available_pairs,
        "unique_available_jpeg_pairs": len(pairs),
        "long_complete_clips": sum(row["long_valid"] for row in selected),
        "long_incomplete_clips": sum(not row["long_valid"] for row in selected),
    }
    if any(observed[key] != expected[key] for key in observed):
        raise RuntimeError("Full CCAC cohort audit changed")
    return observed


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, relative, commit)
        for name, relative in SOURCES.items()
    }
    pilot_lock, pilot_evidence = _pilot_evidence(root, spec)
    inherited_source_names = {
        "measurement_module": "ccac_module",
        "pilot_locker": "locker",
        "pilot_runner": "runner",
        "native_video_module": "native_video_module",
        "common_locker": "common_locker",
        "requirements": "requirements",
    }
    for current_name, pilot_name in inherited_source_names.items():
        if sources[current_name]["sha256"] != pilot_lock["source_sha256"][pilot_name]:
            raise RuntimeError(
                "Full CCAC measurement implementation drifted from the authorized pilot: "
                f"{current_name}"
            )
    measurement = pilot_lock["protocol"]
    unchanged = spec["measurement_contract"]["unchanged_sections"]
    if any(section not in measurement for section in unchanged):
        raise RuntimeError("Pilot measurement protocol is incomplete")
    candidates, p3_rows, manifest_audit = pilot._validate_manifests(root, measurement)
    selected = _full_selection(candidates, pilot_lock)
    cohort_audit = _cohort_audit(selected, spec)
    archive, selected_members, known_boxes = pilot._archive_and_boxes(
        root, measurement, selected, p3_rows
    )
    if len(selected_members) != spec["cohort"]["unique_valid_jpeg_members"]:
        raise RuntimeError("Full CCAC selected image-member count changed")
    pilot_ids = {row["sample_id"] for row in pilot_lock["selected_clips"]}
    nonpilot = [row["sample_id"] for row in selected if row["sample_id"] not in pilot_ids]
    replay_ids = sorted(
        nonpilot,
        key=lambda value: (
            hashlib.sha256((FULL_ORDER_DOMAIN + "replay:" + value).encode()).hexdigest(),
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
        "source_sha256": {name: value["sha256"] for name, value in sources.items()},
        "pilot_evidence": pilot_evidence,
        "pilot_artifact_sha256": pilot_lock["selected_clip_records_sha256"],
        "measurement_protocol": measurement,
        "measurement_source_sha256": pilot_lock["source_sha256"],
        "archive": archive,
        "selected_clips": selected,
        "selected_sample_ids_sha256": pilot.canonical_digest(
            [row["sample_id"] for row in selected]
        ),
        "selected_clip_records_sha256": pilot.canonical_digest(selected),
        "selected_image_members": selected_members,
        "selected_image_members_sha256": pilot.canonical_digest(selected_members),
        "known_boxes_by_member": known_boxes,
        "known_boxes_sha256": pilot.canonical_digest(known_boxes),
        "pilot_sample_ids": [row["sample_id"] for row in pilot_lock["selected_clips"]],
        "pilot_overlay_sample_ids": pilot_lock["overlay_sample_ids"],
        "nonpilot_replay_sample_ids": replay_ids,
        "cohort_audit": cohort_audit,
        "manifest_audit": manifest_audit,
        "opencv": {
            "version": cv2.__version__,
            "build_information_sha256": hashlib.sha256(opencv_build.encode()).hexdigest(),
        },
        "feature_columns": {
            "quality": list(QUALITY_COLUMNS),
            "raw_translation": list(RAW_TRANSLATION_COLUMNS),
            "compensated_translation": list(COMPENSATED_TRANSLATION_COLUMNS),
            "within_actor": list(WITHIN_ACTOR_COLUMNS),
        },
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "pilot_measurement_artifacts_validated": len(
                json.loads(
                    _repo_path(root, spec["pilot_authorization"]["summary"]["path"]).read_text()
                )["artifacts"]
            ),
            "label_free_manifest_rows_read": 2 * 79632,
            "archive_bytes_hashed_opaquely": archive["size_bytes"],
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
        raise RuntimeError("Retained full CCAC lock changed: " + ", ".join(changed))


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _repo_path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing full CCAC lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_repo_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the full CCAC pre-nonpilot-pixel lock")
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
                "clips": len(payload["selected_clips"]),
                "pilot_reuse": len(payload["pilot_sample_ids"]),
                "nonpilot_replay": payload["nonpilot_replay_sample_ids"],
                "cohort": payload["cohort_audit"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
