"""Identity-safe geometry for the conditional learned-tracker measurement.

This module has no tracker, model, image decoder, labels, or promotion rule.
Coordinates stay [time, point_identity, xy]. Camera and actor-root similarities
are fitted to fixed, disjoint fitting IDs; held IDs are never reassigned after
missing observations. Real residual/raw-motion ratios are not quality gates.

Similarities use deterministic least squares, not the old RANSAC implementation.
Nonrigid displacement and root motion are identifiable in synthetic recovery
tests only when the displacement has zero similarity component on fitting IDs.
Real root/articulation channels are a decomposition, not motion ground truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointSplit:
    fit_ids: tuple[str, ...]
    held_ids: tuple[str, ...]

    def indices(self, point_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("Point IDs must be unique")
        if not self.fit_ids or not self.held_ids:
            raise ValueError("Both fixed point groups must be nonempty")
        if len(set(self.fit_ids + self.held_ids)) != len(self.fit_ids + self.held_ids):
            raise ValueError("Fit and held IDs must be unique and disjoint")
        if set(self.fit_ids + self.held_ids) != set(point_ids):
            raise ValueError("Fixed split must cover exactly the supplied point IDs")
        lookup = {value: index for index, value in enumerate(point_ids)}
        return tuple(
            np.asarray([lookup[value] for value in group], dtype=np.int64)
            for group in (self.fit_ids, self.held_ids)
        )


def fixed_point_split(point_ids: tuple[str, ...]) -> PointSplit:
    """Choose roles before observing visibility; sorting makes reordering safe."""
    if len(point_ids) < 6 or len(set(point_ids)) != len(point_ids):
        raise ValueError("At least six unique point IDs are required")
    if not all(isinstance(value, str) for value in point_ids):
        raise ValueError("Point IDs must be strings")
    ordered = tuple(sorted(point_ids))
    return PointSplit(ordered[::2], ordered[1::2])


def _trajectory_arrays(positions, valid, times):
    positions = np.asarray(positions, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    times = np.asarray(times, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("Positions must have shape [time, point_identity, 2]")
    if valid.shape != positions.shape[:2] or times.shape != (len(positions),):
        raise ValueError("Position, visibility and time shapes must agree")
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("At least two finite, strictly increasing physical times are required")
    if not np.isfinite(positions[valid]).all():
        raise ValueError("A visible identity cannot have nonfinite coordinates")
    return positions, valid, times


@dataclass(frozen=True)
class TrackSequence:
    positions: np.ndarray
    valid: np.ndarray
    times_seconds: np.ndarray
    point_ids: tuple[str, ...]

    def __post_init__(self):
        positions, valid, times = _trajectory_arrays(self.positions, self.valid, self.times_seconds)
        if len(self.point_ids) != positions.shape[1] or len(set(self.point_ids)) != len(
            self.point_ids
        ):
            raise ValueError("Unique point IDs must match the identity axis")
        if not all(isinstance(value, str) for value in self.point_ids):
            raise ValueError("Point IDs must be strings")
        positions = positions.copy()
        positions[~valid] = np.nan
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "valid", valid.copy())
        object.__setattr__(self, "times_seconds", times.copy())


@dataclass(frozen=True)
class TimeDerivatives:
    velocity: np.ndarray
    velocity_valid: np.ndarray
    velocity_times: np.ndarray
    acceleration: np.ndarray
    acceleration_valid: np.ndarray
    acceleration_times: np.ndarray


def identity_time_derivatives(positions, valid, times_seconds) -> TimeDerivatives:
    """Difference adjacent times of the same ID only; never bridge missingness."""
    positions, valid, times = _trajectory_arrays(positions, valid, times_seconds)
    velocity_valid = valid[:-1] & valid[1:]
    velocity = np.diff(positions, axis=0) / np.diff(times)[:, None, None]
    velocity[~velocity_valid] = np.nan
    velocity_times = (times[:-1] + times[1:]) / 2
    acceleration_valid = velocity_valid[:-1] & velocity_valid[1:]
    acceleration = np.diff(velocity, axis=0) / np.diff(velocity_times)[:, None, None]
    acceleration[~acceleration_valid] = np.nan
    return TimeDerivatives(
        velocity,
        velocity_valid,
        velocity_times,
        acceleration,
        acceleration_valid,
        (velocity_times[:-1] + velocity_times[1:]) / 2,
    )


def similarity_design(points: np.ndarray) -> np.ndarray:
    """Linear design for a*x-b*y+tx, b*x+a*y+ty; no reflections."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Similarity points must be finite [identity, 2]")
    design = np.zeros((2 * len(points), 4), dtype=np.float64)
    x, y = points.T
    design[::2] = np.column_stack((x, -y, np.ones(len(x)), np.zeros(len(x))))
    design[1::2] = np.column_stack((y, x, np.zeros(len(x)), np.ones(len(x))))
    return design


def fit_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Return a 3x3 similarity, or None if fixed fit support is insufficient."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape:
        raise ValueError("Similarity source and target must be aligned [identity, 2]")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        return None
    coefficients, _, rank, _ = np.linalg.lstsq(
        similarity_design(source), target.reshape(-1), rcond=None
    )
    a, b, tx, ty = coefficients
    if rank != 4 or not np.isfinite(coefficients).all() or a * a + b * b < 1e-12:
        return None
    return np.asarray([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def apply_similarity(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform, points = (
        np.asarray(transform, dtype=np.float64),
        np.asarray(points, dtype=np.float64),
    )
    if transform.shape != (3, 3) or points.shape[-1] != 2 or not np.isfinite(transform).all():
        raise ValueError("Expected finite 3x3 similarity and coordinates ending in xy")
    if not np.allclose(transform[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0):
        raise ValueError("Projective transformations are not supported")
    return points @ transform[:2, :2].T + transform[:2, 2]


@dataclass(frozen=True)
class ArticulationDecomposition:
    camera_transforms: np.ndarray
    camera_valid: np.ndarray
    root_transforms: np.ndarray
    root_valid: np.ndarray
    camera_background_prediction: np.ndarray
    camera_background_valid: np.ndarray
    root_actor_prediction: np.ndarray
    root_actor_prediction_valid: np.ndarray
    camera_compensated_actor: np.ndarray
    camera_compensated_valid: np.ndarray
    root_displacement: np.ndarray
    articulation: np.ndarray
    articulation_valid: np.ndarray


def decompose_tracks(
    actor: TrackSequence,
    background: TrackSequence,
    *,
    center_index: int,
    actor_split: PointSplit,
    background_split: PointSplit,
) -> ArticulationDecomposition:
    """Remove the fitted camera first, then retain separate root/articulation.

    Only fixed fitting IDs enter either fit. Missing fitting points remain
    missing; held identities are never borrowed to make a transform usable.
    Root and articulation are expressed in the reference camera coordinates.
    Raw positions remain available on the original TrackSequence.
    """
    if not np.array_equal(actor.times_seconds, background.times_seconds):
        raise ValueError("Actor and background must use exactly the same timestamps")
    if not 0 <= center_index < len(actor.times_seconds):
        raise ValueError("Center index is outside the sequence")
    afit, _ = actor_split.indices(actor.point_ids)
    bfit, _ = background_split.indices(background.point_ids)
    count = len(actor.times_seconds)
    camera = np.full((count, 3, 3), np.nan)
    root = np.full_like(camera, np.nan)
    camera_ok, root_ok = np.zeros(count, bool), np.zeros(count, bool)
    bg_prediction = np.full_like(background.positions, np.nan)
    bg_ok = np.zeros_like(background.valid)
    actor_prediction = np.full_like(actor.positions, np.nan)
    actor_prediction_ok = np.zeros_like(actor.valid)
    compensated, displacement, articulation = (
        np.full_like(actor.positions, np.nan) for _ in range(3)
    )
    compensated_ok, articulation_ok = np.zeros_like(actor.valid), np.zeros_like(actor.valid)
    actor_center, bg_center = actor.positions[center_index], background.positions[center_index]
    for t in range(count):
        fitting = bfit[background.valid[center_index, bfit] & background.valid[t, bfit]]
        transform = fit_similarity(bg_center[fitting], background.positions[t, fitting])
        if transform is None:
            continue
        camera[t], camera_ok[t] = transform, True
        bg_prediction[t] = apply_similarity(transform, bg_center)
        bg_ok[t] = background.valid[center_index]
        valid = actor.valid[center_index] & actor.valid[t]
        compensated[t, valid] = apply_similarity(
            np.linalg.inv(transform), actor.positions[t, valid]
        )
        compensated_ok[t] = valid
        fitting = afit[valid[afit]]
        transform_root = fit_similarity(actor_center[fitting], compensated[t, fitting])
        if transform_root is None:
            continue
        root[t], root_ok[t] = transform_root, True
        root_positions = apply_similarity(transform_root, actor_center)
        displacement[t, valid] = root_positions[valid] - actor_center[valid]
        articulation[t, valid] = compensated[t, valid] - root_positions[valid]
        articulation_ok[t] = valid
        actor_prediction[t] = apply_similarity(transform, root_positions)
        actor_prediction_ok[t] = actor.valid[center_index]
    return ArticulationDecomposition(
        camera,
        camera_ok,
        root,
        root_ok,
        bg_prediction,
        bg_ok,
        actor_prediction,
        actor_prediction_ok,
        compensated,
        compensated_ok,
        displacement,
        articulation,
        articulation_ok,
    )


def common_held_endpoint_errors(
    reference: TrackSequence,
    predictions: Mapping[str, np.ndarray],
    prediction_valid: Mapping[str, np.ndarray],
    *,
    split: PointSplit,
    center_index: int,
    actor_box_diagonal: float,
) -> dict:
    """Score every carrier on identical held IDs/times, excluding the center.

    Predictions must already be aligned to reference.point_ids/times_seconds.
    The reference can be an inferred track, so this is a consistency endpoint
    unless the caller explicitly supplies synthetic physical ground truth.
    Report missing support separately; never credit dropped points as accuracy.
    """
    if not predictions or set(predictions) != set(prediction_valid):
        raise ValueError("Every named carrier needs positions and validity")
    if not 0 <= center_index < len(reference.times_seconds):
        raise ValueError("Center index is outside the sequence")
    if not np.isfinite(actor_box_diagonal) or actor_box_diagonal <= 0:
        raise ValueError("Normalization diagonal must be positive and finite")
    _, held = split.indices(reference.point_ids)
    eligible = np.zeros_like(reference.valid)
    eligible[:, held] = True
    eligible[center_index] = False
    common = eligible & reference.valid
    per_carrier_available = {}
    checked = {}
    for name, values in predictions.items():
        values, valid, _ = _trajectory_arrays(
            values, prediction_valid[name], reference.times_seconds
        )
        if values.shape != reference.positions.shape:
            raise ValueError("Carrier identity/time axes must match the reference")
        checked[name] = values
        per_carrier_available[name] = int(np.sum(eligible & reference.valid & valid))
        common &= valid
    errors = {
        name: np.linalg.norm(values[common] - reference.positions[common], axis=1)
        / actor_box_diagonal
        for name, values in checked.items()
    }
    expected = int(eligible.sum())
    return {
        "expected_held_time_pairs": expected,
        "reference_available_pairs": int((eligible & reference.valid).sum()),
        "common_pairs": int(common.sum()),
        "missing_common_pairs": expected - int(common.sum()),
        "per_carrier_available_pairs": per_carrier_available,
        "common_mask": common,
        "errors_normalized": errors,
        "median_error_normalized": {
            name: float(np.median(value)) if len(value) else None for name, value in errors.items()
        },
        "includes_center": False,
        "endpoint": "common_held_point_consistency_not_ground_truth",
    }
