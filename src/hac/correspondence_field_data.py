"""Auditable observation-only P8 fields for the preregistered C0/C1/C2 screen.

No label is accepted by the observation loader. ``input_valid`` describes measured
points; ``motion_targets`` creates the independent training-only ``loss_valid``.
The endpoint remains conditional on the supplied physical actor boxes/tracks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hac.ccac_features import aggregate_clip_features

PAIR_COUNT = 15
MAX_POINTS = 32
POINT_COLUMNS = (
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "local_x",
    "local_y",
    "translation_x",
    "translation_y",
    "fb_error_pixels",
    "fb_error_actor_heights",
)
PAIR_COLUMNS = (
    "dt_seconds",
    "point_count",
    "camera_audit_median_pixels",
    "camera_audit_p90_pixels",
    "camera_usable",
)
P8_RUN_REQUEST_SHA256 = "36c49248a28c07680c89f2b228398b7310eb61ad3c64c2c05dfe3e1074a07e15"
CONTEXT_PATHS = {
    "memory": ".runs/research_20260908/source_swap_v1/data/memory_data.npz",
    "dino": ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/dinov2_native_frames.npy",
    "dino_summary": ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json",
}
CONTEXT_SHA256 = {
    "memory": "0bfadd5f080b162ab4ccfce76affbde4779d9fb61757124029b159adec4079e9",
    "dino": "b40cfea1659795bf7554b0e77f7f6d95226c1745038baca884460065546ec2f2",
    "dino_summary": "17086c64f9627b2a20865976c698f5541f954967ab96fb4961bd2831519b6267",
}
_COMPLETE = "OKUTAMA_CCAC_CLIP_WORKLOAD_COMPLETE"
_REFERENCE = "OKUTAMA_CCAC_PILOT_WORKLOAD_REUSED_BY_REFERENCE"
_FRAME_INPUT_KEYS = (
    "source_frame",
    "time_index",
    "nominal_time_seconds",
    "offset_seconds",
    "bbox",
    "image_member",
    "image_present",
    "image_width",
    "image_height",
)


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise ValueError("Object arrays are not auditable field inputs")
    digest = hashlib.sha256(canonical_digest([value.dtype.str, list(value.shape)]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _within_root(root: Path, path: Path) -> Path:
    path = (root / path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Workload reference escaped the repository")
    return path


def _frame_inputs(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Explicit allowlist: annotation flags, valid_frame and missing_reason are absent.
    return [{key: frame.get(key) for key in _FRAME_INPUT_KEYS} for frame in frames]


@dataclass(frozen=True)
class CorrespondenceField:
    sample_id: str
    point_features: np.ndarray  # [15, max_points, 10]
    input_valid: np.ndarray  # [15, max_points], never a supervision mask
    pair_features: np.ndarray  # [15, 5]; contains no slot or midpoint timestamp
    pair_input_valid: np.ndarray  # [15]
    midpoint_seconds: np.ndarray  # [15], passed ONLY to C2
    collapsed_features: np.ndarray  # [31]: P8 quality16 + compensated9 + residual6
    audit: dict[str, Any]


def deterministic_point_indices(
    positions: np.ndarray,
    point_records: np.ndarray,
    *,
    sample_id: str,
    pair_index: int,
    maximum: int = MAX_POINTS,
) -> np.ndarray:
    """Spatial farthest-point coverage, with content-hash ties independent of row order.

    Points are pair-local measurements, not persistent identities. The hash order
    uses sample ID and measured values, never a class or annotation-presence bit.
    """
    positions = np.asarray(positions, dtype=np.float64)
    records = np.asarray(point_records, dtype="<f8")
    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or records.ndim != 2
        or len(records) != len(positions)
        or maximum < 1
        or not np.isfinite(positions).all()
        or not np.isfinite(records).all()
    ):
        raise ValueError("Malformed deterministic point-capping input")
    keys = [
        hashlib.sha256(
            f"hac-field-v1|{sample_id}|pair:{pair_index}|".encode() + row.tobytes()
        ).digest()
        for row in records
    ]
    order = np.asarray(sorted(range(len(records)), key=lambda i: keys[i]), dtype=np.int64)
    if len(order) <= maximum:
        return order
    selected = [int(order[0])]
    nearest = np.sum((positions - positions[selected[0]]) ** 2, axis=1)
    available = np.ones(len(order), dtype=bool)
    available[selected[0]] = False
    for _ in range(1, maximum):
        candidates = order[available[order]]
        index = int(candidates[np.argmax(nearest[candidates])])
        selected.append(index)
        available[index] = False
        nearest = np.minimum(nearest, np.sum((positions - positions[index]) ** 2, axis=1))
    return np.asarray(selected, dtype=np.int64)


def resolve_workload(
    directory: Path,
    *,
    repository_root: Path,
    expected_sample_id: str,
    expected_run_request_sha256: str | None = P8_RUN_REQUEST_SHA256,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve and hash-check one reference level before opening correspondence arrays."""
    root = Path(repository_root).resolve()
    directory = _within_root(root, Path(directory))
    request = _read_json(directory / "request.json")
    receipt = _read_json(directory / "receipt.json")
    if (
        request.get("sample_id") != expected_sample_id
        or receipt.get("request") != request
        or receipt.get("request_sha256") != canonical_digest(request)
    ):
        raise ValueError("Workload sample/request identity or digest mismatch")
    if (
        expected_run_request_sha256 is not None
        and request.get("run_request_sha256") != expected_run_request_sha256
    ):
        raise ValueError("Workload source-processing request differs from the locked recipe")
    source = directory
    source_request, source_receipt = request, receipt
    if receipt.get("status") == _REFERENCE:
        source = _within_root(root, Path(receipt["source_directory"]))
        expected_source = root / ".runs/research_20260907/okutama_ccac_pilot/results/workloads"
        expected_source /= f"{int(request['selection_rank']):03d}"
        if source != expected_source.resolve():
            raise ValueError("Pilot reference does not match the expected workload rank")
        source_request = _read_json(source / "request.json")
        source_receipt = _read_json(source / "receipt.json")

        def comparable(r: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in r.items() if k != "run_request_sha256"}

        if (
            comparable(source_request) != comparable(request)
            or receipt.get("source_request_sha256") != canonical_digest(source_request)
            or receipt.get("source_artifacts") != source_receipt.get("artifacts")
        ):
            raise ValueError("Pilot reference source/request mismatch")
    if (
        source_receipt.get("status") != _COMPLETE
        or source_receipt.get("request") != source_request
        or source_receipt.get("request_sha256") != canonical_digest(source_request)
    ):
        raise ValueError("Source workload is incomplete or its request changed")
    artifacts = source_receipt.get("artifacts", {})
    if not {"correspondences.npz", "pair_metrics.json", "clip_metrics.json"} <= set(artifacts):
        raise ValueError("Required workload artifact receipt is missing")
    verified = {}
    for name, expected in sorted(artifacts.items()):
        if Path(name).name != name:
            raise ValueError("Artifact receipt names must be local filenames")
        path = source / name
        if file_sha256(path) != expected:
            raise ValueError(f"Workload artifact SHA256 mismatch: {name}")
        verified[name] = expected
    audit = {
        "workload_directory": directory.relative_to(root).as_posix(),
        "source_directory": source.relative_to(root).as_posix(),
        "request_sha256": canonical_digest(request),
        "source_request_sha256": canonical_digest(source_request),
        "receipt_sha256": file_sha256(directory / "receipt.json"),
        "source_receipt_sha256": file_sha256(source / "receipt.json"),
        "source_artifacts": verified,
    }
    return source, request, source_receipt, audit


def _positive_box(frame: Mapping[str, Any]) -> np.ndarray | None:
    box = frame.get("bbox")
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return None
    return box


def _safe_transform(transform: np.ndarray, frame: Mapping[str, Any]) -> bool:
    if transform.shape != (3, 3):
        raise ValueError("Camera transform must have shape (3, 3)")
    if not np.isfinite(transform).all() or np.linalg.cond(transform) > 1e6:
        return False
    corners = np.asarray(
        [
            [0, 0, 1],
            [frame["image_width"], 0, 1],
            [0, frame["image_height"], 1],
            [frame["image_width"], frame["image_height"], 1],
        ],
        dtype=np.float64,
    )
    denom = corners @ transform[2]
    return bool(
        np.isfinite(denom).all()
        and (np.all(denom > 1e-9) or np.all(denom < -1e-9))
        and np.all(np.linalg.det(transform) / denom**3 > 0)
    )


def field_from_observations(
    sample_id: str,
    frames: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    *,
    max_points: int = MAX_POINTS,
    fps: float = 30.0,
    center_frame: int | None = None,
    camera_mode: str = "saved",
    motion_mode: str = "full",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pure physical conversion; annotation fields may be absent or arbitrarily changed."""
    if len(frames) != PAIR_COUNT + 1 or len(pairs) != PAIR_COUNT:
        raise ValueError("A field requires the locked 16 frames and 15 pairs")
    if (
        not 1 <= max_points <= MAX_POINTS
        or not np.isfinite(fps)
        or fps <= 0
        or camera_mode not in {"saved", "identity"}
        or motion_mode not in {"full", "translation_free", "residual_free"}
    ):
        raise ValueError("Unsupported field conversion contract")
    center = int(frames[8]["source_frame"]) if center_frame is None else int(center_frame)
    frame_numbers = np.asarray([int(frame["source_frame"]) for frame in frames])
    if center != frame_numbers[8] or not np.array_equal(
        frame_numbers - center, np.arange(-32, 29, 4)
    ):
        raise ValueError("Frame identities do not match the locked center and source offsets")
    points = np.zeros((PAIR_COUNT, max_points, len(POINT_COLUMNS)), dtype=np.float32)
    valid = np.zeros((PAIR_COUNT, max_points), dtype=bool)
    pair_features = np.zeros((PAIR_COUNT, len(PAIR_COLUMNS)), dtype=np.float32)
    midpoints = np.zeros(PAIR_COUNT, dtype=np.float32)
    selected_indices, invalid_reasons = [], []
    for t, pair in enumerate(pairs):
        left, right = frames[t : t + 2]
        dt = (frame_numbers[t + 1] - frame_numbers[t]) / fps
        nominal_dt = float(right["nominal_time_seconds"]) - float(left["nominal_time_seconds"])
        if (
            pair.get("sample_id") != sample_id
            or int(pair.get("pair_index", -1)) != t
            or int(pair.get("source_frame", -1)) != frame_numbers[t]
            or int(pair.get("target_frame", -1)) != frame_numbers[t + 1]
            or int(left["time_index"]) != t
            or int(right["time_index"]) != t + 1
            or not np.isclose(nominal_dt, dt, rtol=0, atol=1e-8)
            or not np.isclose(float(pair["elapsed_seconds"]), dt, rtol=0, atol=1e-8)
        ):
            raise ValueError("Pair/frame/sample identity or elapsed-time mismatch")
        if not np.isclose(
            float(left["nominal_time_seconds"]), frame_numbers[t] / fps, atol=1e-8, rtol=0
        ) or not np.isclose(
            float(right["nominal_time_seconds"]), frame_numbers[t + 1] / fps, atol=1e-8, rtol=0
        ):
            raise ValueError("Nominal timestamps disagree with physical frame IDs")
        midpoints[t] = ((frame_numbers[t] + frame_numbers[t + 1]) / 2 - center) / fps
        prefix = f"pair_{t:02d}"
        source = np.asarray(arrays[f"{prefix}_actor_source"], dtype=np.float64)
        target = np.asarray(arrays[f"{prefix}_actor_target"], dtype=np.float64)
        fb = np.asarray(arrays[f"{prefix}_actor_fb_error"], dtype=np.float64)
        transform = np.asarray(arrays[f"{prefix}_camera_transform"], dtype=np.float64)
        if (
            source.ndim != 2
            or source.shape[1] != 2
            or target.shape != source.shape
            or fb.shape != (len(source),)
        ):
            raise ValueError("Correspondence coordinates and FB errors are not aligned")
        left_box, right_box = _positive_box(left), _positive_box(right)
        safe_transform = _safe_transform(transform, left)
        reason = ""
        if (
            not bool(left.get("image_present"))
            or not bool(right.get("image_present"))
            or not left.get("image_member")
            or not right.get("image_member")
            or left_box is None
            or right_box is None
        ):
            reason = "missing_pixels_or_physical_box"
        elif not safe_transform or not bool(pair.get("camera_usable")):
            reason = "unusable_camera_measurement"
        eligible = (
            np.isfinite(source).all(1) & np.isfinite(target).all(1) & np.isfinite(fb) & (fb >= 0)
        )
        if not reason and not eligible.any():
            reason = "no_finite_correspondences"
        if reason:
            selected_indices.append([])
            invalid_reasons.append(reason)
            continue
        assert left_box is not None and right_box is not None
        raw_indices = np.flatnonzero(eligible)
        source, target, fb = source[eligible], target[eligible], fb[eligible]
        used_transform = np.eye(3) if camera_mode == "identity" else transform
        projected_h = np.c_[source, np.ones(len(source))] @ used_transform.T
        safe_points = np.isfinite(projected_h).all(1) & (np.abs(projected_h[:, 2]) > 1e-9)
        source, target, fb, raw_indices = (
            source[safe_points],
            target[safe_points],
            fb[safe_points],
            raw_indices[safe_points],
        )
        projected_h = projected_h[safe_points]
        if not len(source):
            selected_indices.append([])
            invalid_reasons.append("no_finite_projected_points")
            continue
        scale = np.sqrt((left_box[3] - left_box[1]) * (right_box[3] - right_box[1]))
        projected = projected_h[:, :2] / projected_h[:, 2:]
        velocity = (target - projected) / (scale * dt)
        translation = np.median(velocity, axis=0)
        local = velocity - translation
        if motion_mode == "translation_free":
            velocity, translation = local.copy(), np.zeros(2)
        elif motion_mode == "residual_free":
            velocity, local = (
                np.broadcast_to(translation, velocity.shape).copy(),
                np.zeros_like(local),
            )
        position = (source - left_box[:2]) / (left_box[2:] - left_box[:2])
        records = np.c_[source, target, fb]
        chosen = deterministic_point_indices(
            position, records, sample_id=sample_id, pair_index=t, maximum=max_points
        )
        tokens = np.c_[
            position, velocity, local, np.broadcast_to(translation, velocity.shape), fb, fb / scale
        ]
        tokens = tokens[chosen].astype(np.float32)
        if not np.isfinite(tokens).all():
            raise ValueError("Physical field normalization produced nonfinite tokens")
        points[t, : len(chosen)] = tokens
        valid[t, : len(chosen)] = True
        quality = [pair.get("camera_audit_median_pixels"), pair.get("camera_audit_p90_pixels")]
        if any(
            value is None or not np.isfinite(float(value)) or float(value) < 0 for value in quality
        ):
            raise ValueError("Usable camera pair lacks finite observed audit quality")
        pair_features[t] = [dt, len(chosen), *quality, 1.0]
        selected_indices.append(raw_indices[chosen].tolist())
        invalid_reasons.append("")
    values = {
        "point_features": points,
        "input_valid": valid,
        "pair_features": pair_features,
        "pair_input_valid": valid.any(1),
        "midpoint_seconds": midpoints,
    }
    audit = {
        "sample_id": sample_id,
        "observation_frames_sha256": canonical_digest(_frame_inputs(frames)),
        "center_frame": center,
        "fps": fps,
        "max_points": max_points,
        "camera_mode": camera_mode,
        "motion_mode": motion_mode,
        "selected_raw_point_indices": selected_indices,
        "invalid_pair_reasons": invalid_reasons,
        "array_sha256": {key: array_digest(value) for key, value in values.items()},
        "annotation_fields_used_for_inference": [],
        "point_identity_persists_across_pairs": False,
    }
    return values, audit


def load_correspondence_workload(
    directory: Path,
    *,
    repository_root: Path,
    expected_sample_id: str,
    expected_frames: Sequence[Mapping[str, Any]] | None = None,
    expected_run_request_sha256: str | None = P8_RUN_REQUEST_SHA256,
    max_points: int = MAX_POINTS,
    fps: float = 30.0,
    camera_mode: str = "saved",
    motion_mode: str = "full",
) -> CorrespondenceField:
    source, request, _, provenance = resolve_workload(
        directory,
        repository_root=repository_root,
        expected_sample_id=expected_sample_id,
        expected_run_request_sha256=expected_run_request_sha256,
    )
    frames = request["frames"]
    if expected_frames is not None and _frame_inputs(frames) != _frame_inputs(expected_frames):
        raise ValueError("Physical boxes/frame source contract differs from expected inputs")
    payload = _read_json(source / "pair_metrics.json")
    clip = _read_json(source / "clip_metrics.json")
    if (
        payload.get("sample_id") != expected_sample_id
        or clip.get("sample_id") != expected_sample_id
    ):
        raise ValueError("Workload pair/clip sample mismatch")
    with np.load(source / "correspondences.npz", allow_pickle=False) as arrays:
        values, audit = field_from_observations(
            expected_sample_id,
            frames,
            payload["pairs"],
            arrays,
            max_points=max_points,
            fps=fps,
            center_frame=clip["center_frame"],
            camera_mode=camera_mode,
            motion_mode=motion_mode,
        )
    quality, _, compensated, within, _, _ = aggregate_clip_features(payload["pairs"], clip)
    collapsed = np.concatenate([quality, compensated, within]).astype(np.float32)
    audit.update(provenance)
    audit["array_sha256"]["collapsed_features"] = array_digest(collapsed)
    return CorrespondenceField(
        expected_sample_id, **values, collapsed_features=collapsed, audit=audit
    )


def index_workloads(results_directory: Path, expected_sample_ids: Sequence[str]) -> dict[str, Path]:
    """Metadata-only lookup. Exact population equality prevents dropping missing rows."""
    expected = list(map(str, expected_sample_ids))
    if len(set(expected)) != len(expected):
        raise ValueError("Expected sample IDs contain duplicates")
    found = {}
    for path in sorted((Path(results_directory) / "workloads").glob("*/request.json")):
        sample_id = str(_read_json(path)["sample_id"])
        if sample_id in found:
            raise ValueError("Duplicate correspondence workload sample ID")
        found[sample_id] = path.parent
    if set(found) != set(expected):
        raise ValueError("Correspondence workload population differs from expected cohort")
    return {sample_id: found[sample_id] for sample_id in expected}


def motion_targets(labels: np.ndarray, training_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Make center-only upright supervision; never feed this mask to an encoder."""
    labels, training_rows = np.asarray(labels), np.asarray(training_rows)
    if (
        labels.ndim != 1
        or training_rows.shape != labels.shape
        or training_rows.dtype != np.bool_
        or not np.issubdtype(labels.dtype, np.integer)
        or not np.isin(labels, [-1, 0, 1, 2]).all()
    ):
        raise ValueError("Invalid center labels or training-only row mask")
    loss_valid = training_rows & np.isin(labels, [1, 2])
    targets = np.where(loss_valid, labels == 2, 0).astype(np.float32)
    return {"targets": targets, "loss_valid": loss_valid}


def load_frozen_context(
    repository_root: Path,
    sample_ids: Sequence[str],
    *,
    paths: Mapping[str, str] = CONTEXT_PATHS,
    expected_hashes: Mapping[str, str] = CONTEXT_SHA256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only frozen features/IDs, never memory labels/support or task predictions."""
    root = Path(repository_root).resolve()
    resolved = {name: _within_root(root, Path(paths[name])) for name in CONTEXT_PATHS}
    hashes = {name: file_sha256(path) for name, path in resolved.items()}
    if hashes != dict(expected_hashes):
        raise ValueError("Frozen context differs from the pinned source hashes")
    expected = np.asarray(sample_ids, dtype=str)
    if expected.ndim != 1 or len(np.unique(expected)) != len(expected):
        raise ValueError("Context sample IDs must be unique")
    with np.load(resolved["memory"], allow_pickle=False) as saved:
        ids = saved["sample_ids"].astype(str)
        features = saved["features"]
        if not np.array_equal(ids, expected) or features.shape != (len(expected), 3072):
            raise ValueError("Memory feature population or width mismatch")
        video = np.asarray(features[:, :1536], dtype=np.float32)
    summary = _read_json(resolved["dino_summary"])
    if not np.array_equal(np.asarray(summary["sample_ids"], dtype=str), expected):
        raise ValueError("DINO center row order differs from the frozen context")
    dino = np.load(resolved["dino"], mmap_mode="r", allow_pickle=False)
    if dino.shape != (len(expected), 16, 1, 768):
        raise ValueError("DINO frozen frame shape differs from the declared center view")
    context = np.concatenate([video, dino[:, 8, 0, :].astype(np.float32)], axis=1)
    if not np.isfinite(context).all():
        raise ValueError("Frozen absolute context contains nonfinite values")
    return context, {
        "source_sha256": hashes,
        "sample_ids_sha256": array_digest(expected),
        "context_sha256": array_digest(context),
        "columns": ["short_video768", "long_video768", "center_dino768"],
        "memory_npz_members_read": ["sample_ids", "features"],
    }
