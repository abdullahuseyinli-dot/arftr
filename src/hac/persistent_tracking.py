"""Deterministic label-blind persistent point tracking primitives.

The GIPA smoke deliberately uses one identity set seeded at the centre frame and
propagates those identities through all requested frames.  This is different
from the historical CCAC pair-local extractor, which re-detected points for
each pair.  The implementation is frozen OpenCV LK for the feasibility smoke;
learned trackers can be plugged in later behind the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class StepTracks:
    source: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    forward_backward_error: np.ndarray


@dataclass(frozen=True)
class PersistentTracks:
    positions: np.ndarray  # [time, point, xy], NaN after identity loss
    valid: np.ndarray  # [time, point]
    link_forward_backward_error: np.ndarray  # [time - 1, point]


def resample_tracks(tracks: PersistentTracks, indices: list[int]) -> PersistentTracks:
    """Keep requested timestamps while preserving dense-link identity diagnostics."""
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) < 1 or np.any(np.diff(selected) <= 0):
        raise ValueError("Track resampling indices must be strictly increasing")
    if selected[0] < 0 or selected[-1] >= len(tracks.positions):
        raise ValueError("Track resampling index outside dense trajectory")
    links = np.full((max(len(selected) - 1, 0), tracks.positions.shape[1]), np.inf, dtype=np.float32)
    for output_index, (left, right) in enumerate(zip(selected[:-1], selected[1:], strict=True)):
        values = tracks.link_forward_backward_error[left:right]
        finite = np.isfinite(values)
        for point_index in range(values.shape[1]):
            point_values = values[:, point_index][finite[:, point_index]]
            if len(point_values):
                links[output_index, point_index] = np.float32(np.median(point_values))
    return PersistentTracks(
        tracks.positions[selected].copy(),
        tracks.valid[selected].copy(),
        links,
    )


def _lk_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "winSize": tuple(int(value) for value in spec["window"]),
        "maxLevel": int(spec["maximum_pyramid_level"]),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(spec["iterations"]),
            float(spec["epsilon"]),
        ),
        "minEigThreshold": float(spec["minimum_eigenvalue"]),
    }


def _inside(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )


def track_step(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    spec: dict[str, Any],
    *,
    maximum_error: float,
) -> StepTracks:
    """Track fixed point identities one link with forward/backward validation."""
    previous = np.asarray(previous)
    current = np.asarray(current)
    source = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    count = len(source)
    target = np.full((count, 2), np.nan, dtype=np.float32)
    fb_error = np.full(count, np.inf, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    if count == 0:
        return StepTracks(source, target, valid, fb_error)
    if (
        previous.dtype != np.uint8
        or current.dtype != np.uint8
        or previous.ndim != 2
        or current.ndim != 2
        or previous.shape != current.shape
    ):
        raise ValueError("Persistent LK inputs must be aligned uint8 grayscale frames")
    finite = np.isfinite(source).all(axis=1) & _inside(source, previous.shape)
    indices = np.flatnonzero(finite)
    if not len(indices):
        return StepTracks(source, target, valid, fb_error)
    query = source[indices].reshape(-1, 1, 2)
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, query, None, **_lk_kwargs(spec)
    )
    if forward is None or forward_status is None:
        return StepTracks(source, target, valid, fb_error)
    forward = np.asarray(forward, dtype=np.float32).reshape(-1, 2)
    forward_status = np.asarray(forward_status).reshape(-1).astype(bool)
    forward_ok = forward_status & _inside(forward, current.shape)
    target[indices[forward_ok]] = forward[forward_ok]
    reverse_indices = indices[forward_ok]
    if not len(reverse_indices):
        return StepTracks(source, target, valid, fb_error)
    reverse, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        forward[forward_ok].reshape(-1, 1, 2),
        None,
        **_lk_kwargs(spec),
    )
    if reverse is None or reverse_status is None:
        return StepTracks(source, target, valid, fb_error)
    reverse = np.asarray(reverse, dtype=np.float32).reshape(-1, 2)
    reverse_status = np.asarray(reverse_status).reshape(-1).astype(bool)
    error = np.linalg.norm(reverse - source[reverse_indices], axis=1)
    fb_error[reverse_indices] = error.astype(np.float32)
    accepted = reverse_status & np.isfinite(error) & (error <= float(maximum_error))
    valid[reverse_indices[accepted]] = True
    return StepTracks(source, target, valid, fb_error)


def track_persistent(
    frames: list[np.ndarray],
    seed_points: np.ndarray,
    *,
    center_index: int,
    spec: dict[str, Any],
    maximum_error: float,
) -> PersistentTracks:
    """Propagate one fixed identity set forward and backward from the centre."""
    if not frames:
        raise ValueError("Persistent tracking requires at least one frame")
    if not 0 <= center_index < len(frames):
        raise ValueError("Persistent tracking centre is outside the frame list")
    seeds = np.asarray(seed_points, dtype=np.float32).reshape(-1, 2)
    time_count, point_count = len(frames), len(seeds)
    positions = np.full((time_count, point_count, 2), np.nan, dtype=np.float32)
    valid = np.zeros((time_count, point_count), dtype=bool)
    links = np.full((max(time_count - 1, 0), point_count), np.inf, dtype=np.float32)
    positions[center_index] = seeds
    valid[center_index] = _inside(seeds, frames[center_index].shape)

    for direction in (1, -1):
        current_index = center_index
        alive = valid[center_index].copy()
        while 0 <= current_index + direction < time_count:
            next_index = current_index + direction
            if frames[current_index] is None or frames[next_index] is None:
                break
            active = np.flatnonzero(alive)
            if not len(active):
                break
            step = track_step(
                frames[current_index],
                frames[next_index],
                positions[current_index, active],
                spec,
                maximum_error=maximum_error,
            )
            keep = step.valid
            positions[next_index, active[keep]] = step.target[keep]
            valid[next_index, active[keep]] = True
            links[min(current_index, next_index), active] = step.forward_backward_error
            alive[active[~keep]] = False
            current_index = next_index
    positions[~valid] = np.nan
    return PersistentTracks(positions, valid, links)


def median_cycle_error(tracks: PersistentTracks) -> float | None:
    values = tracks.link_forward_backward_error[np.isfinite(tracks.link_forward_backward_error)]
    return float(np.median(values)) if len(values) else None
