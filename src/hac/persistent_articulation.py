"""Gauge-invariant persistent articulation measurement for the GIPA smoke."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from hac.persistent_tracking import PersistentTracks, StepTracks, track_step


@dataclass(frozen=True)
class SeedPoints:
    points: np.ndarray
    cells: np.ndarray


@dataclass(frozen=True)
class TransformFit:
    transform: np.ndarray
    usable: bool
    inlier_count: int
    median_error: float | None
    p90_error: float | None


def stable_seed(domain: str, sample_id: str, source: str) -> int:
    digest = hashlib.sha256(f"{domain}:{source}:{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _clip_box(box: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[int, ...]:
    height, width = shape
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, int(math.floor(x0)))),
        max(0, min(height, int(math.floor(y0)))),
        max(0, min(width, int(math.ceil(x1)))),
        max(0, min(height, int(math.ceil(y1)))),
    )


def _cell_ids(points: np.ndarray, box: tuple[float, float, float, float], grid: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    columns, rows = grid
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    column = np.clip(((points[:, 0] - x0) / width * columns).astype(int), 0, columns - 1)
    row = np.clip(((points[:, 1] - y0) / height * rows).astype(int), 0, rows - 1)
    return (row * columns + column).astype(np.int16)


def _round_robin(points: np.ndarray, cells: np.ndarray, maximum: int) -> SeedPoints:
    if not len(points):
        return SeedPoints(np.empty((0, 2), np.float32), np.empty(0, np.int16))
    groups: dict[int, list[int]] = {}
    for index, cell in enumerate(cells.tolist()):
        groups.setdefault(int(cell), []).append(index)
    selected: list[int] = []
    while len(selected) < min(maximum, len(points)):
        progressed = False
        for cell in sorted(groups):
            if groups[cell]:
                selected.append(groups[cell].pop(0))
                progressed = True
                if len(selected) >= maximum:
                    break
        if not progressed:
            break
    selected_array = np.asarray(selected, dtype=np.int64)
    return SeedPoints(points[selected_array].astype(np.float32), cells[selected_array].astype(np.int16))


def actor_seed_points(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    maximum: int = 64,
    grid: tuple[int, int] = (3, 2),
) -> SeedPoints:
    """Choose observable textured points in a centre actor box, without labels."""
    x0, y0, x1, y1 = box
    dx, dy = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    left, top, right, bottom = _clip_box((x0 + dx, y0 + dy, x1 - dx, y1 - dy), image.shape)
    if right - left < 5 or bottom - top < 5:
        return SeedPoints(np.empty((0, 2), np.float32), np.empty(0, np.int16))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    points = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max(maximum * 4, maximum),
        qualityLevel=0.001,
        minDistance=2.0,
        blockSize=5,
        mask=mask,
    )
    if points is None:
        return SeedPoints(np.empty((0, 2), np.float32), np.empty(0, np.int16))
    values = points.reshape(-1, 2).astype(np.float32)
    cells = _cell_ids(values, box, grid)
    order = np.lexsort((values[:, 0], values[:, 1], cells))
    return _round_robin(values[order], cells[order], maximum)


def background_seed_points(
    image: np.ndarray,
    actor_box: tuple[float, float, float, float],
    *,
    maximum: int = 64,
    grid: tuple[int, int] = (4, 3),
    exclude_boxes: list[tuple[float, float, float, float]] | None = None,
) -> SeedPoints:
    """Choose spatially distributed background points outside the actor region."""
    height, width = image.shape
    mask = np.full(image.shape, 255, dtype=np.uint8)
    boxes = [actor_box] + list(exclude_boxes or [])
    for x0, y0, x1, y1 in boxes:
        margin = max(x1 - x0, y1 - y0) * 0.75 + 12.0
        left, top, right, bottom = _clip_box(
            (x0 - margin, y0 - margin, x1 + margin, y1 + margin), image.shape
        )
        mask[top:bottom, left:right] = 0
    columns, rows = grid
    per_cell = max(2, int(math.ceil(maximum / (columns * rows))))
    all_points: list[np.ndarray] = []
    all_cells: list[np.ndarray] = []
    for row in range(rows):
        for column in range(columns):
            cell_mask = np.zeros_like(mask)
            cell_left, cell_right = column * width // columns, (column + 1) * width // columns
            cell_top, cell_bottom = row * height // rows, (row + 1) * height // rows
            cell_mask[cell_top:cell_bottom, cell_left:cell_right] = mask[
                cell_top:cell_bottom, cell_left:cell_right
            ]
            points = cv2.goodFeaturesToTrack(
                image,
                maxCorners=per_cell,
                qualityLevel=0.005,
                minDistance=8.0,
                blockSize=7,
                mask=cell_mask,
            )
            if points is None:
                continue
            values = points.reshape(-1, 2).astype(np.float32)
            all_points.append(values)
            all_cells.append(np.full(len(values), row * columns + column, dtype=np.int16))
    if not all_points:
        return SeedPoints(np.empty((0, 2), np.float32), np.empty(0, np.int16))
    points = np.concatenate(all_points)
    cells = np.concatenate(all_cells)
    order = np.lexsort((points[:, 0], points[:, 1], cells))
    return _round_robin(points[order], cells[order], maximum)


def project_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((values, np.ones(len(values)))) @ np.asarray(transform).T
    denominator = homogeneous[:, 2]
    result = np.full((len(values), 2), np.nan, dtype=np.float64)
    usable = np.isfinite(homogeneous).all(axis=1) & (np.abs(denominator) > 1e-12)
    result[usable] = homogeneous[usable, :2] / denominator[usable, None]
    return result


def _error_summary(predicted: np.ndarray, observed: np.ndarray) -> tuple[float | None, float | None]:
    if len(predicted) == 0 or len(predicted) != len(observed):
        return None, None
    error = np.linalg.norm(predicted - observed, axis=1)
    error = error[np.isfinite(error)]
    if not len(error):
        return None, None
    return float(np.median(error)), float(np.quantile(error, 0.9))


def fit_similarity(source: np.ndarray, target: np.ndarray, *, seed: int) -> TransformFit:
    source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    if len(source) != len(target) or len(source) < 3:
        return TransformFit(np.eye(3), False, 0, None, None)
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    affine, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )
    if affine is None or not np.isfinite(affine).all():
        return TransformFit(np.eye(3), False, 0, None, None)
    transform = np.eye(3, dtype=np.float64)
    transform[:2] = np.asarray(affine, dtype=np.float64)
    predicted = project_points(transform, source)
    median, p90 = _error_summary(predicted, target)
    inlier_count = int(np.asarray(inliers).reshape(-1).astype(bool).sum()) if inliers is not None else 0
    return TransformFit(transform, median is not None, inlier_count, median, p90)


def _safe_median(values: list[float]) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def _held_error(
    source: np.ndarray,
    target: np.ndarray,
    camera: np.ndarray,
    *,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    if len(source) < 6:
        return None, None, None
    projected = project_points(camera, source)
    split = (np.arange(len(source)) % 2) == 0
    fit = split & np.isfinite(projected).all(axis=1) & np.isfinite(target).all(axis=1)
    held = ~split & np.isfinite(projected).all(axis=1) & np.isfinite(target).all(axis=1)
    if fit.sum() < 3 or held.sum() < 2:
        return None, None, None
    root = fit_similarity(projected[fit], target[fit], seed=seed)
    if not root.usable:
        return None, None, None
    prediction = project_points(root.transform, projected[held])
    correct, _ = _error_summary(prediction, target[held])
    permutation = target[fit][np.roll(np.arange(int(fit.sum())), 1)]
    shuffled_root = fit_similarity(projected[fit], permutation, seed=seed + 17)
    if not shuffled_root.usable:
        return correct, None, None
    shuffled_prediction = project_points(shuffled_root.transform, projected[held])
    shuffled, _ = _error_summary(shuffled_prediction, target[held])
    penalty = (
        float(shuffled / correct - 1.0)
        if correct is not None and correct > 1e-8 and shuffled is not None
        else None
    )
    return correct, shuffled, penalty


def decompose_tracks(
    actor_tracks: PersistentTracks,
    background_tracks: PersistentTracks,
    *,
    actor_box: tuple[float, float, float, float],
    center_index: int,
    image_shape: tuple[int, int],
    seed: int,
) -> dict[str, Any]:
    """Estimate camera, actor-root, and articulation channels from persistent tracks."""
    actor_source = actor_tracks.positions[center_index]
    background_source = background_tracks.positions[center_index]
    diagonal = math.hypot(actor_box[2] - actor_box[0], actor_box[3] - actor_box[1])
    per_time: list[dict[str, Any]] = []
    camera_removal: list[float] = []
    actor_retention: list[float] = []
    held_correct: list[float] = []
    held_shuffled: list[float] = []
    identity_penalties: list[float] = []
    articulation_vectors: list[np.ndarray] = []
    for time_index in range(len(actor_tracks.positions)):
        actor_valid = actor_tracks.valid[center_index] & actor_tracks.valid[time_index]
        bg_valid = background_tracks.valid[center_index] & background_tracks.valid[time_index]
        actor_src = actor_source[actor_valid]
        actor_tgt = actor_tracks.positions[time_index, actor_valid]
        bg_src = background_source[bg_valid]
        bg_tgt = background_tracks.positions[time_index, bg_valid]
        if time_index == center_index:
            camera = TransformFit(np.eye(3), True, len(bg_src), 0.0, 0.0)
        else:
            camera = fit_similarity(bg_src, bg_tgt, seed=seed + time_index)
        camera_pred_bg = project_points(camera.transform, bg_src)
        raw_bg, _ = _error_summary(bg_src, bg_tgt)
        camera_bg, _ = _error_summary(camera_pred_bg, bg_tgt)
        if raw_bg is not None and raw_bg > 1e-6 and camera_bg is not None:
            camera_removal.append(float(1.0 - camera_bg / raw_bg))
        projected_actor = project_points(camera.transform, actor_src)
        root = fit_similarity(projected_actor, actor_tgt, seed=seed + 100 + time_index)
        root_prediction = project_points(root.transform, projected_actor)
        raw_actor, _ = _error_summary(actor_src, actor_tgt)
        compensated_actor, _ = _error_summary(projected_actor, actor_tgt)
        if raw_actor is not None and compensated_actor is not None and raw_actor > 1e-6:
            actor_retention.append(float(compensated_actor / raw_actor))
        articulation = actor_tgt - root_prediction if root.usable else np.empty((0, 2))
        if len(articulation):
            articulation_vectors.append(articulation)
        correct, shuffled, penalty = _held_error(
            actor_src, actor_tgt, camera.transform, seed=seed + 200 + time_index
        )
        if correct is not None:
            held_correct.append(correct)
        if shuffled is not None:
            held_shuffled.append(shuffled)
        if penalty is not None:
            identity_penalties.append(penalty)
        per_time.append(
            {
                "time_index": time_index,
                "actor_points": int(len(actor_src)),
                "background_points": int(len(bg_src)),
                "camera_usable": bool(camera.usable),
                "camera_median_error": camera.median_error,
                "camera_p90_error": camera.p90_error,
                "camera_inliers": camera.inlier_count,
                "camera_transform": camera.transform.tolist(),
                "root_usable": bool(root.usable),
                "root_median_error": root.median_error,
                "raw_actor_displacement": raw_actor,
                "camera_compensated_actor_displacement": compensated_actor,
                "articulation_median": (
                    float(np.median(np.linalg.norm(articulation, axis=1)))
                    if len(articulation)
                    else None
                ),
                "held_error": correct,
                "identity_permuted_held_error": shuffled,
                "identity_penalty": penalty,
            }
        )
    all_articulation = np.concatenate(articulation_vectors) if articulation_vectors else np.empty((0, 2))
    velocities = np.diff(all_articulation, axis=0) if len(all_articulation) > 1 else np.empty((0, 2))
    return {
        "per_time": per_time,
        "camera_removal_median": _safe_median(camera_removal),
        "actor_residual_retention_median": _safe_median(actor_retention),
        "held_error_median": _safe_median(held_correct),
        "identity_permuted_held_error_median": _safe_median(held_shuffled),
        "identity_penalty_median": _safe_median(identity_penalties),
        "actor_point_survival_all_frames": float(
            np.mean(np.all(actor_tracks.valid, axis=0)) if actor_tracks.valid.shape[1] else 0.0
        ),
        "actor_point_survival_mean_frames": float(actor_tracks.valid.mean())
        if actor_tracks.valid.size
        else 0.0,
        "cycle_error_median": (
            float(
                np.median(
                    actor_tracks.link_forward_backward_error[
                        np.isfinite(actor_tracks.link_forward_backward_error)
                    ]
                )
            )
            if np.isfinite(actor_tracks.link_forward_backward_error).any()
            else None
        ),
        "articulation_shape_energy": float(np.mean(np.sum(all_articulation**2, axis=1)))
        if len(all_articulation)
        else None,
        "articulation_time_odd_energy": float(np.mean(np.sum(velocities**2, axis=1)))
        if len(velocities)
        else None,
        "actor_box_diagonal": float(diagonal),
        "actor_points_seeded": int(actor_tracks.valid.shape[1]),
        "background_points_seeded": int(background_tracks.valid.shape[1]),
    }


def pair_local_held_error(
    previous: np.ndarray,
    current: np.ndarray,
    actor_box: tuple[float, float, float, float],
    held_source: np.ndarray,
    held_target: np.ndarray,
    *,
    lk_spec: dict[str, Any],
    maximum_error: float,
    seed: int,
    camera_transform: np.ndarray | None = None,
) -> float | None:
    """One-link re-detection baseline evaluated on persistent held identities."""
    seeds = actor_seed_points(previous, actor_box, maximum=64).points
    if len(seeds) < 4:
        return None
    tracked: StepTracks = track_step(
        previous,
        current,
        seeds,
        lk_spec,
        maximum_error=maximum_error,
    )
    if tracked.valid.sum() < 4:
        return None
    camera = np.eye(3, dtype=np.float64) if camera_transform is None else np.asarray(camera_transform)
    tracked_source = project_points(camera, tracked.source[tracked.valid])
    transform = fit_similarity(tracked_source, tracked.target[tracked.valid], seed=seed)
    if not transform.usable:
        return None
    held_projected = project_points(camera, held_source)
    prediction = project_points(transform.transform, held_projected)
    error, _ = _error_summary(prediction, held_target)
    return error
