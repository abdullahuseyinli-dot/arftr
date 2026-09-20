"""Run the label-blind GIPA persistent-articulation measurement smoke.

This script intentionally stops at geometry.  It reads only the fixed,
label-blind 128-center selection and the pinned 4K source videos.  No labels,
ARFTR predictions, support annotations, human-review fields, or task model are
loaded.  The default backend is explicitly named ``opencv_lk_persistent``: it
is a lower-bound feasibility measurement while the learned CoTracker/TAPIR
dependency is absent, not a silent replacement for the future frozen tracker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hac.persistent_articulation import (  # noqa: E402
    actor_seed_points,
    background_seed_points,
    decompose_tracks,
    pair_local_held_error,
    stable_seed,
)
from hac.persistent_tracking import median_cycle_error, resample_tracks, track_persistent  # noqa: E402

SELECTION_PATH = ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"
CROP_MANIFEST_PATH = ROOT / ".runs/research_20260913/body_witness_pilot_v1/crop_manifest.json"
SOURCE_ROOT = Path(r"C:\Users\DELL\hac_external_data\OkutamaAction\source4k\allowed_videos")
P3_FRAME_MANIFEST = ROOT / ".runs/research_20260907/okutama_native_video_p3/manifest/frame_manifest.csv"
DEFAULT_RUN = ROOT / ".runs/research_20260916/gipa_persistent_articulation_v1"
OFFSETS = (-30, -22, -15, -8, 0, 8, 15, 22, 30)
CENTER_INDEX = 4
FPS = 30000.0 / 1001.0
BACKEND = "opencv_lk_persistent"
LK_SPEC = {
    "window": [21, 21],
    "maximum_pyramid_level": 3,
    "iterations": 30,
    "epsilon": 0.01,
    "minimum_eigenvalue": 0.0001,
}
MAX_FB_ERROR_4K = 3.0
MAX_FB_ERROR_720 = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_selection(max_clips: int) -> list[dict[str, Any]]:
    selected = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    crop_rows = json.loads(CROP_MANIFEST_PATH.read_text(encoding="utf-8"))
    crops = {row["sample_id"]: row for row in crop_rows}
    if len(selected) != 128 or len(crops) != 128:
        raise RuntimeError("Pinned GIPA selection/crop population is not exactly 128")
    if [row["selection_index"] for row in selected] != list(range(128)):
        raise RuntimeError("Pinned selection indices are not contiguous")
    known_by_member: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    member_by_sample: dict[str, str] = {}
    with P3_FRAME_MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("valid_frame", "0") != "1" or not row.get("image_member"):
                continue
            if row.get("time_index") == str(CENTER_INDEX + 4):
                member_by_sample[row["sample_id"]] = row["image_member"]
            try:
                known_by_member[row["image_member"]].append(
                    tuple(float(row[name]) for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
                )
            except (KeyError, TypeError, ValueError):
                continue
    result = []
    for row in selected[:max_clips]:
        crop = crops.get(row["sample_id"])
        if crop is None or not crop.get("decode_valid"):
            raise RuntimeError(f"Pinned native crop is unavailable: {row['sample_id']}")
        request = crop.get("source_request", {})
        box = request.get("native_box")
        frame = request.get("native_index")
        image_member = member_by_sample.get(row["sample_id"])
        if not isinstance(box, list) or len(box) != 4 or frame is None:
            raise RuntimeError(f"Pinned native box/frame missing: {row['sample_id']}")
        if int(frame) != int(row["center_frame"]):
            raise RuntimeError(f"Selection/crop frame mismatch: {row['sample_id']}")
        result.append(
            {
                "sample_id": row["sample_id"],
                "selection_index": int(row["selection_index"]),
                "recording": row["recording"],
                "track": row["track"],
                "scenario": row["scenario"],
                "center_frame": int(frame),
                "native_box": [float(value) for value in box],
                "known_boxes_720": [list(values) for values in known_by_member.get(image_member, [])],
            }
        )
    return result


def _video_path(recording: str) -> Path:
    matches = sorted(SOURCE_ROOT.glob(recording + ".*"))
    matches = [path for path in matches if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one authorized source video for {recording}: {matches}")
    return matches[0]


def _required_indices(clips: list[dict[str, Any]]) -> dict[str, list[int]]:
    required: dict[str, set[int]] = defaultdict(set)
    for clip in clips:
        for offset in OFFSETS:
            required[clip["recording"]].add(int(clip["center_frame"]) + int(offset))
    return {key: sorted(value) for key, value in required.items()}


def preflight(run: Path, clips: list[dict[str, Any]]) -> dict[str, Any]:
    modules = {
        name: bool(importlib.util.find_spec(name))
        for name in ("cotracker", "co_tracker", "tapir")
    }
    videos = []
    required = _required_indices(clips)
    for recording in sorted(required):
        path = _video_path(recording)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open authorized source video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        indices = required[recording]
        if min(indices) < 0 or max(indices) >= frames:
            raise RuntimeError(f"Requested frame outside source bounds: {recording}")
        if (width, height) != (3840, 2160):
            raise RuntimeError(f"Source is not locked 3840x2160: {recording}")
        videos.append(
            {
                "recording": recording,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "fps": fps,
                "frame_count": frames,
                "size": [width, height],
                "requested_min_frame": min(indices),
                "requested_max_frame": max(indices),
                "source_hash": sha256_file(path),
            }
        )
    gpu: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        gpu = {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - environment-only diagnostic
        gpu = {"cuda_available": False, "error": str(exc)}
    result = {
        "status": "GIPA_PREFLIGHT_COMPLETE",
        "selection": {
            "path": str(SELECTION_PATH),
            "sha256": sha256_file(SELECTION_PATH),
            "crop_manifest_sha256": sha256_file(CROP_MANIFEST_PATH),
            "clips": len(clips),
            "selection_indices": [clip["selection_index"] for clip in clips],
            "labels_used": False,
        },
        "backend": {
            "requested_learned_trackers": ["CoTracker3", "TAPIR"],
            "installed_modules": modules,
            "selected_backend": BACKEND,
            "explicit_lower_bound": True,
            "task_fit_authorized": False,
        },
        "sources": videos,
        "offsets": list(OFFSETS),
        "fps_expected": FPS,
        "gpu": gpu,
        "protocol_note": "OpenCV LK is an explicit frozen feasibility lower bound because CoTracker/TAPIR is absent; it is not a silent tracker substitution.",
    }
    write_json(run / "preflight.json", result)
    return result


def read_dense_clip_frames(clip: dict[str, Any]) -> dict[int, np.ndarray]:
    """Decode the physical frame bridge so identity is propagated one frame at a time."""
    recording = clip["recording"]
    path = _video_path(recording)
    start = int(clip["center_frame"]) + min(OFFSETS)
    stop = int(clip["center_frame"]) + max(OFFSETS)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {path}")
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        cap.release()
        raise RuntimeError(f"Cannot seek source video: {recording}")
    frames: dict[int, np.ndarray] = {}
    for frame_index in range(start, stop + 1):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed decoding frame {recording}:{frame_index}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape != (2160, 3840) or gray.dtype != np.uint8:
            cap.release()
            raise RuntimeError("Decoded source dimensions/dtype changed")
        frames[frame_index] = gray
    cap.release()
    if len(frames) != stop - start + 1:
        raise RuntimeError(f"Dense frame bridge incomplete for {recording}")
    return frames


def _source_frames(
    clip: dict[str, Any], frame_store: dict[int, np.ndarray], source: str
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[float, float, float, float], list[int]]:
    native_box = tuple(float(value) for value in clip["native_box"])
    start = int(clip["center_frame"]) + min(OFFSETS)
    center = int(clip["center_frame"]) - start
    sampled_indices = [center + offset for offset in OFFSETS]
    dense = [frame_store[index] for index in range(start, start + len(frame_store))]
    sampled = [dense[index] for index in sampled_indices]
    if source == "native4k":
        return sampled, dense, native_box, sampled_indices
    if source == "downsample720":
        low_dense = [cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA) for frame in dense]
        return [low_dense[index] for index in sampled_indices], low_dense, tuple(value / 3.0 for value in native_box), sampled_indices
    raise ValueError(f"Unknown GIPA source: {source}")


def _aggregate(values: list[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)])
    return float(np.median(finite)) if len(finite) else None


def _measure_clip(
    clip: dict[str, Any],
    frames: list[np.ndarray],
    dense_frames: list[np.ndarray],
    box: tuple[float, float, float, float],
    *,
    source: str,
    sampled_indices: list[int],
    save_arrays: dict[str, np.ndarray],
    overlay_path: Path | None,
) -> dict[str, Any]:
    image_shape = frames[CENTER_INDEX].shape
    actor_seeds = actor_seed_points(frames[CENTER_INDEX], box, maximum=64)
    scale = 1.0 if source == "native4k" else 1.0 / 3.0
    known_boxes = [tuple(float(value) * scale for value in values) for values in clip["known_boxes_720"]]
    background_seeds = background_seed_points(
        frames[CENTER_INDEX], box, maximum=64, exclude_boxes=known_boxes
    )
    maximum_fb = MAX_FB_ERROR_4K if source == "native4k" else MAX_FB_ERROR_720
    base = {
        "source": source,
        "sample_id": clip["sample_id"],
        "selection_index": clip["selection_index"],
        "recording": clip["recording"],
        "scenario": clip["scenario"],
        "center_frame": clip["center_frame"],
        "box": list(box),
        "image_shape": list(image_shape),
        "actor_seed_points": int(len(actor_seeds.points)),
        "background_seed_points": int(len(background_seeds.points)),
        "known_background_exclusion_boxes": int(len(known_boxes)),
        "tracker_backend": BACKEND,
    }
    if len(actor_seeds.points) < 8 or len(background_seeds.points) < 16:
        return {**base, "status": "INSUFFICIENT_SEEDS", "phase_a_eligible": False}
    dense_actor_tracks = track_persistent(
        dense_frames,
        actor_seeds.points,
        center_index=sampled_indices[CENTER_INDEX],
        spec=LK_SPEC,
        maximum_error=maximum_fb,
    )
    dense_background_tracks = track_persistent(
        dense_frames,
        background_seeds.points,
        center_index=sampled_indices[CENTER_INDEX],
        spec=LK_SPEC,
        maximum_error=maximum_fb,
    )
    actor_tracks = resample_tracks(dense_actor_tracks, sampled_indices)
    background_tracks = resample_tracks(dense_background_tracks, sampled_indices)
    seed = stable_seed("gipa-v1", clip["sample_id"], source)
    decomposition = decompose_tracks(
        actor_tracks,
        background_tracks,
        actor_box=box,
        center_index=CENTER_INDEX,
        image_shape=image_shape,
        seed=seed,
    )
    pair_errors: list[float] = []
    actor_source = actor_tracks.positions[CENTER_INDEX]
    for time_index in range(len(frames)):
        if time_index == CENTER_INDEX:
            continue
        valid = actor_tracks.valid[CENTER_INDEX] & actor_tracks.valid[time_index]
        if valid.sum() < 4:
            continue
        camera = np.asarray(decomposition["per_time"][time_index]["camera_transform"], dtype=np.float64)
        error = pair_local_held_error(
            frames[CENTER_INDEX],
            frames[time_index],
            box,
            actor_source[valid],
            actor_tracks.positions[time_index, valid],
            lk_spec=LK_SPEC,
            maximum_error=maximum_fb,
            seed=seed + time_index,
            camera_transform=camera,
        )
        if error is not None:
            pair_errors.append(error)
    persistent_error = decomposition["held_error_median"]
    pair_local_error = _aggregate(pair_errors)
    held_reduction = (
        float(1.0 - persistent_error / pair_local_error)
        if persistent_error is not None and pair_local_error is not None and pair_local_error > 1e-8
        else None
    )
    # A deliberately mismatched control: shift centre actor seeds outside the actor box and
    # track them with the same frozen backend.  It tests whether any easy-to-track texture
    # earns a comparable articulation result without actor identity.
    box_width = box[2] - box[0]
    wrong_points = actor_seeds.points + np.asarray((2.5 * box_width, 0.0), dtype=np.float32)
    wrong_tracks = track_persistent(
        frames,
        wrong_points,
        center_index=CENTER_INDEX,
        spec=LK_SPEC,
        maximum_error=maximum_fb,
    )
    wrong_survival = float(np.mean(np.all(wrong_tracks.valid, axis=0))) if len(wrong_points) else 0.0
    repeated = [frames[CENTER_INDEX]] * len(frames)
    repeated_actor = track_persistent(
        repeated,
        actor_seeds.points,
        center_index=CENTER_INDEX,
        spec=LK_SPEC,
        maximum_error=maximum_fb,
    )
    repeated_motion = float(np.nanmedian(np.linalg.norm(np.diff(repeated_actor.positions, axis=0), axis=2))) if len(actor_seeds.points) else None
    save_key = f"clip_{clip['selection_index']:03d}_{source}"
    save_arrays[f"{save_key}_actor_positions"] = actor_tracks.positions
    save_arrays[f"{save_key}_actor_valid"] = actor_tracks.valid
    save_arrays[f"{save_key}_background_positions"] = background_tracks.positions
    save_arrays[f"{save_key}_background_valid"] = background_tracks.valid
    metrics = {
        **base,
        "status": "MEASURED",
        "phase_a_eligible": True,
        "actor_survival_all_frames": decomposition["actor_point_survival_all_frames"],
        "actor_survival_mean_frames": decomposition["actor_point_survival_mean_frames"],
        "cycle_error_median_pixels": decomposition["cycle_error_median"],
        "cycle_error_relative_box": (
            decomposition["cycle_error_median"] / decomposition["actor_box_diagonal"]
            if decomposition["cycle_error_median"] is not None
            else None
        ),
        "camera_removal_median": decomposition["camera_removal_median"],
        "actor_residual_retention_median": decomposition["actor_residual_retention_median"],
        "identity_penalty_median": decomposition["identity_penalty_median"],
        "held_error_persistent_median_pixels": persistent_error,
        "held_error_pair_local_median_pixels": pair_local_error,
        "held_error_reduction_vs_pair_local": held_reduction,
        "wrong_track_survival_all_frames": wrong_survival,
        "wrong_track_comparable": bool(wrong_survival >= max(0.75, decomposition["actor_point_survival_all_frames"] - 0.05)),
        "repeated_center_median_motion_pixels": repeated_motion,
        "median_cycle_error_function": median_cycle_error(dense_actor_tracks),
        "decomposition": decomposition,
    }
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = cv2.cvtColor(frames[CENTER_INDEX], cv2.COLOR_GRAY2BGR)
        for point in background_seeds.points:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 4, (255, 160, 0), -1)
        for point in actor_seeds.points:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 5, (0, 255, 0), -1)
        for point_index in range(min(32, len(actor_seeds.points))):
            trajectory = actor_tracks.positions[:, point_index]
            good = actor_tracks.valid[:, point_index]
            for left, right in zip(np.flatnonzero(good)[:-1], np.flatnonzero(good)[1:], strict=False):
                cv2.line(
                    canvas,
                    tuple(np.rint(trajectory[left]).astype(int)),
                    tuple(np.rint(trajectory[right]).astype(int)),
                    (0, 220, 0),
                    2,
                )
        cv2.rectangle(
            canvas,
            tuple(np.rint(box[:2]).astype(int)),
            tuple(np.rint(box[2:]).astype(int)),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            canvas,
            f"GIPA {source} clip={clip['selection_index']} actor={len(actor_seeds.points)}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        if not cv2.imwrite(str(overlay_path), canvas):
            raise RuntimeError(f"Could not write overlay: {overlay_path}")
    return metrics


def _source_summary(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    source_rows = [row for row in rows if row["source"] == source]
    measured = [row for row in source_rows if row.get("phase_a_eligible")]
    survival = [float(row["actor_survival_all_frames"]) for row in measured]
    cycle = [row.get("cycle_error_relative_box") for row in measured]
    camera = [row.get("camera_removal_median") for row in measured]
    retention = [row.get("actor_residual_retention_median") for row in measured]
    identity = [row.get("identity_penalty_median") for row in measured]
    reductions = [row.get("held_error_reduction_vs_pair_local") for row in measured]
    smoke_gates = {
        "survival": bool(measured) and float(np.mean(np.asarray(survival) >= 0.75)) >= 0.8,
        "cycle": bool(_aggregate(cycle) is not None and float(_aggregate(cycle)) <= 0.05),
        "camera_separation": bool(
            _aggregate(camera) is not None
            and _aggregate(retention) is not None
            and float(_aggregate(camera)) >= 0.5
            and float(_aggregate(retention)) >= 0.5
        ),
        "identity_specificity": bool(_aggregate(identity) is not None and float(_aggregate(identity)) >= 0.25),
        "held_reduction_projection": bool(
            _aggregate(reductions) is not None and float(_aggregate(reductions)) >= 0.20
        ),
        "wrong_track": not any(bool(row.get("wrong_track_comparable")) for row in measured),
    }
    return {
        "source": source,
        "clips_requested": len(source_rows),
        "clips_measured": len(measured),
        "survival_center_pass_fraction": float(np.mean(np.asarray(survival) >= 0.75)) if survival else None,
        "cycle_relative_box_median": _aggregate(cycle),
        "camera_removal_median": _aggregate(camera),
        "actor_residual_retention_median": _aggregate(retention),
        "identity_penalty_median": _aggregate(identity),
        "held_reduction_median": _aggregate(reductions),
        "smoke_gates": smoke_gates,
        "smoke_passed": bool(smoke_gates) and all(smoke_gates.values()),
        "foldwise_held_gate": "NOT_EVALUATED; expand to fixed 128 centers before any fold-wise claim",
    }


def run_smoke(run: Path, clips: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    sources = ("native4k", "downsample720")
    for clip in clips:
        dense_store = read_dense_clip_frames(clip)
        for source in sources:
            frames, dense_frames, box, sampled_indices = _source_frames(clip, dense_store, source)
            overlay = run / "overlays" / f"{clip['selection_index']:03d}_{source}.png" if clip["selection_index"] < 4 else None
            rows.append(
                _measure_clip(
                    clip,
                    frames,
                    dense_frames,
                    box,
                    source=source,
                    sampled_indices=sampled_indices,
                    save_arrays=arrays,
                    overlay_path=overlay,
                )
            )
    np.savez_compressed(run / "trajectories.npz", **arrays)
    by_source = {source: _source_summary(rows, source) for source in sources}
    gate_keys = ("survival", "cycle", "camera_separation", "identity_specificity", "wrong_track")
    source_direction = {
        key: by_source["native4k"]["smoke_gates"].get(key)
        == by_source["downsample720"]["smoke_gates"].get(key)
        for key in gate_keys
    }
    summary = {
        "status": "GIPA_PHASE_A_SMOKE_COMPLETE",
        "backend": BACKEND,
        "labels_used": False,
        "arftr_outputs_used": False,
        "support_annotations_used": False,
        "human_review_used": False,
        "task_fit_authorized": False,
        "clips": len(clips),
        "offsets": list(OFFSETS),
        "center_index": CENTER_INDEX,
        "clip_metrics": rows,
        "source_summaries": by_source,
        "source_gate_direction_agreement": source_direction,
        "source_consistency_gate": all(source_direction.values()),
        "smoke_passed": bool(all(item["smoke_passed"] for item in by_source.values()) and all(source_direction.values())),
        "expanded_128_required_for_fold_gate": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(run / "clip_metrics.json", rows)
    write_json(run / "summary.json", summary)
    artifact_names = ["preflight.json", "clip_metrics.json", "summary.json", "trajectories.npz"]
    artifact_names.extend(str(path.relative_to(run)) for path in sorted((run / "overlays").glob("*.png")) if path.is_file())
    receipt = {
        "status": "GIPA_PHASE_A_SMOKE_RECEIPT",
        "run": str(run),
        "selection_sha256": sha256_file(SELECTION_PATH),
        "crop_manifest_sha256": sha256_file(CROP_MANIFEST_PATH),
        "artifacts": {name: sha256_file(run / name) for name in artifact_names},
        "artifact_inventory": sorted(artifact_names),
    }
    write_json(run / "receipt.json", receipt)
    return summary


def audit(run: Path) -> dict[str, Any]:
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    failures = []
    for name, digest in receipt["artifacts"].items():
        path = run / name
        if not path.exists() or sha256_file(path) != digest:
            failures.append(name)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((run / "clip_metrics.json").read_text(encoding="utf-8"))
    checks = {
        "receipt_artifacts_exact": not failures,
        "row_count_matches": len(rows) == 2 * int(summary["clips"]),
        "label_blind": not any(
            summary.get(key, True) for key in ("labels_used", "arftr_outputs_used", "support_annotations_used", "human_review_used")
        ),
        "task_fit_not_authorized": summary.get("task_fit_authorized") is False,
        "source_summaries_present": set(summary.get("source_summaries", {})) == {"native4k", "downsample720"},
    }
    result = {
        "status": "GIPA_PHASE_A_INDEPENDENT_REPLAY_AUDIT_COMPLETE",
        "passed": bool(all(checks.values()) and not failures),
        "checks": checks,
        "hash_failures": failures,
        "smoke_passed_recorded": bool(summary.get("smoke_passed")),
    }
    write_json(run / "independent_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "smoke", "audit"), default="preflight")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--max-clips", type=int, default=16)
    args = parser.parse_args()
    if args.max_clips < 1 or args.max_clips > 128:
        raise SystemExit("--max-clips must be between 1 and 128")
    run = args.run.resolve()
    if args.stage == "audit":
        print(json.dumps(audit(run), indent=2, sort_keys=True))
        return
    if run.exists() and args.stage == "preflight" and (run / "preflight.json").exists():
        raise SystemExit(f"Refusing to overwrite existing preflight: {run}")
    run.mkdir(parents=True, exist_ok=True)
    clips = load_selection(args.max_clips)
    preflight_result = preflight(run, clips)
    if args.stage == "preflight":
        print(json.dumps(json_safe(preflight_result), indent=2, sort_keys=True, allow_nan=False))
        return
    summary = run_smoke(run, clips)
    audit_result = audit(run)
    print(json.dumps(json_safe({"summary": summary, "audit": audit_result}), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
