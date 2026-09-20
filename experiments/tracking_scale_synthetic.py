"""Locked native4K synthetic correspondences for the tracking scale probe.

These textured point sprites are a controlled measurement fixture, not natural
video validation. The 64 synthetic identities exist by construction and NEVER
waive the original real-video 64-point availability gate. Camera ground truth
is exposed only for this fixture's warp and physical-correspondence scoring.
No tracker inference, activity labels, downloads, or training are performed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from hac.persistent_articulation_v2 import (
    TrackSequence,
    apply_similarity,
    fit_similarity,
    fixed_point_split,
)

ACTOR_SIZES = {"typical": (80, 124), "tiny": (32, 64)}  # width, height in native pixels
CASES = ("camera_only", "actor_only", "same_direction", "opposite_direction", "nonrigid")
POINT_IDS = tuple(f"actor-{index:03d}" for index in range(64))
IMAGE_SHAPE = (2160, 3840)
TIMES = np.linspace(-1.0, 1.0, 9)
SEED = 20260920


def _translation(displacement):
    transform = np.eye(3, dtype=np.float64)
    transform[:2, 2] = displacement
    return transform


def _paint_sprite(frame, sprite, point):
    """Translate a 3x3 texture around its known center, with alpha compositing.

    Bilinear interpolation has <=4 native-pixel support per axis. Even the tiny
    fixture has >4 pixels between rows. Black bounding rectangles are not pasted
    over adjacent sprites or their point centers.
    """
    x, y = point
    left, top = math.floor(x) - 2, math.floor(y) - 2
    matrix = np.asarray([[1.0, 0.0, x - left - 1.0], [0.0, 1.0, y - top - 1.0]])
    color = cv2.warpAffine(sprite.astype(np.float32), matrix, (5, 5), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(np.ones((3, 3), np.float32), matrix, (5, 5), flags=cv2.INTER_LINEAR)
    background = frame[top : top + 5, left : left + 5]
    if background.shape != (5, 5, 3):
        raise ValueError("Synthetic sprite left the native frame")
    combined = color + background.astype(np.float32) * (1.0 - alpha[..., None])
    background[:] = np.clip(np.rint(combined), 0, 255).astype(np.uint8)


def synthetic_scale_case(size: str, case: str, *, render: bool = True) -> dict:
    """Return one of exactly 10 fixed physical-scale fixtures.

    Default output includes nine native 3840x2160 RGB uint8 frames. ``render=False``
    is only a low-memory arithmetic/unit-test option; it preserves identical
    ground truth and returns an empty frame list. Coordinates remain native
    pixel centers; the single bbox is from t=0 and is not moved using future data.
    """
    if size not in ACTOR_SIZES or case not in CASES:
        raise ValueError("Use fixed sizes typical/tiny and one of the five locked cases")
    actor_width, actor_height = ACTOR_SIZES[size]
    height, width = IMAGE_SHAPE
    x0, y0 = (width - actor_width) / 2, (height - actor_height) / 2
    box = (x0, y0, x0 + actor_width, y0 + actor_height)
    xs = np.linspace(box[0] + 1.5, box[2] - 1.5, 4)
    ys = np.linspace(box[1] + 1.5, box[3] - 1.5, 16)
    actor0 = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(64, 2)
    diagonal = math.hypot(actor_width, actor_height)
    times = TIMES.copy()
    camera_speed = np.asarray([0.0, 0.0] if case == "actor_only" else [320.0, 80.0])
    direction = np.asarray([320.0, 80.0]) / math.hypot(320.0, 80.0)
    root_speed = (
        direction
        * (0.0 if case == "camera_only" else -0.12 if case == "opposite_direction" else 0.12)
        * diagonal
    )
    camera = np.stack([_translation(t * camera_speed) for t in times])
    root_displacement = np.broadcast_to(times[:, None, None] * root_speed, (9, 64, 2)).copy()

    # Pure horizontal row bending keeps all same-row column distances and every
    # vertical row spacing unchanged. Removing constant and linear-y components
    # makes it orthogonal to all four similarity DOFs on the fixed even IDs.
    row_coordinate = np.linspace(-1.0, 1.0, 16)
    basis = np.column_stack((np.ones(16), row_coordinate))
    bend = np.cos(np.pi * row_coordinate)
    bend -= basis @ np.linalg.lstsq(basis, bend, rcond=None)[0]
    bend /= np.max(np.abs(bend))
    bend_vector = np.column_stack((np.repeat(bend, 4), np.zeros(64)))
    articulation = np.stack(
        [
            0.04 * diagonal * np.sin(t * np.pi / 2) * bend_vector
            if case == "nonrigid"
            else np.zeros_like(actor0)
            for t in times
        ]
    )
    compensated_truth = actor0[None] + root_displacement + articulation
    actor_truth = np.stack(
        [apply_similarity(c, points) for c, points in zip(camera, compensated_truth, strict=True)]
    )

    frames = []
    if render:
        rng = np.random.default_rng(SEED)
        # The larger world texture supplies real previously unseen background
        # under camera translation. No reflected/zero-padded image borders.
        margin_x, margin_y = 328, 88
        world_width, world_height = width + 2 * margin_x, height + 2 * margin_y
        coarse = rng.integers(
            15, 100, (math.ceil(world_height / 16), math.ceil(world_width / 16), 3), dtype=np.uint8
        )
        world = cv2.resize(coarse, (world_width, world_height), interpolation=cv2.INTER_LINEAR)
        sprites = rng.integers(155, 256, (64, 3, 3, 3), dtype=np.uint8)
        for transform, points in zip(camera, actor_truth, strict=True):
            dx, dy = transform[:2, 2]
            if dx != round(dx) or dy != round(dy):
                raise ValueError("Locked camera translations must have integer native-pixel shifts")
            left, top = margin_x - int(dx), margin_y - int(dy)
            frame = np.ascontiguousarray(world[top : top + height, left : left + width])
            for sprite, point in zip(sprites, points, strict=True):
                _paint_sprite(frame, sprite, point)
            frames.append(frame)
    return {
        "fixture_id": f"native4k_scale_v1::{size}::{case}",
        "size": size,
        "case": case,
        "seed": SEED,
        "frames": frames,
        "actor_truth": actor_truth,
        "articulation_truth": articulation,
        "camera_truth": camera,
        "root_displacement_truth": root_displacement,
        "camera_compensated_truth": compensated_truth,
        "times_seconds": times,
        "center_index": 4,
        "box_native": box,
        "diagonal": diagonal,
        "point_ids": POINT_IDS,
        "image_shape": IMAGE_SHAPE,
        "actor_size": (actor_width, actor_height),
        "synthetic_point_availability_not_real_gate_waiver": True,
        "camera_truth_use": "synthetic warp/scoring only; never a real-video input",
    }


def score_synthetic_scale_tracks(
    sequence: TrackSequence, fixture: dict, cycle_error, cycle_valid
) -> dict:
    """Score native-coordinate predictions against known physical correspondences.

    Camera inversion uses fixture truth ONLY, to isolate actor information from
    camera-estimation failures. Camera estimation is a separate measurement.
    Root similarity uses fixed fitting IDs, articulation/relative-motion errors
    use disjoint held IDs. Center is excluded from every temporal score.
    High/native is the measurement gate; full-frame/low-detail results may fail
    as diagnostic arms and MUST NOT be silently substituted as a new primary.
    """
    ids = tuple(fixture["point_ids"])
    times = np.asarray(fixture["times_seconds"], dtype=np.float64)
    center = int(fixture["center_index"])
    if sequence.point_ids != ids or not np.array_equal(sequence.times_seconds, times):
        raise ValueError("Native prediction identities/timestamps differ from the fixed fixture")
    if sequence.positions.shape != (9, 64, 2) or center != 4:
        raise ValueError("This locked synthetic screen requires nine times and 64 fixed identities")
    cycles, return_mask = (
        np.asarray(cycle_error, dtype=np.float64),
        np.asarray(cycle_valid, dtype=bool),
    )
    if cycles.shape != (9, 64) or return_mask.shape != (9, 64):
        raise ValueError("Independent cycle arrays must preserve all nine times and 64 identities")
    if return_mask[center].any() or np.isfinite(cycles[center]).any():
        raise ValueError("Center cannot count as an independent return cycle")
    if np.any(cycles[return_mask] < 0) or not np.isfinite(cycles[return_mask]).all():
        raise ValueError("Measured return cycles must be finite nonnegative pixel errors")
    noncenter = np.arange(9) != center
    diagonal = float(fixture["diagonal"])
    center_error = np.linalg.norm(
        sequence.positions[center] - fixture["actor_truth"][center], axis=-1
    )
    center_ok = sequence.valid[center].all() and bool(np.max(center_error) <= 0.01)
    compensated = np.stack(
        [
            apply_similarity(np.linalg.inv(camera), points)
            for camera, points in zip(fixture["camera_truth"], sequence.positions, strict=True)
        ]
    )
    truth_compensated = np.asarray(fixture["camera_compensated_truth"])
    temporal_valid = sequence.valid.copy()
    temporal_valid[center] = False
    actor_error = np.linalg.norm(compensated - truth_compensated, axis=-1) / diagonal
    actor_median = float(np.median(actor_error[temporal_valid])) if temporal_valid.any() else None

    fit, held = fixed_point_split(ids).indices(ids)
    actor0 = fixture["actor_truth"][center]
    root_displacement = np.full_like(compensated, np.nan)
    articulation = np.full_like(compensated, np.nan)
    geometry_mask = np.zeros((9, 64), bool)
    for index in np.flatnonzero(noncenter):
        fit_visible = fit[sequence.valid[index, fit] & sequence.valid[center, fit]]
        root = fit_similarity(actor0[fit_visible], compensated[index, fit_visible])
        if root is None:
            continue
        visible = sequence.valid[index] & sequence.valid[center]
        prediction = apply_similarity(root, actor0)
        root_displacement[index, visible] = prediction[visible] - actor0[visible]
        articulation[index, visible] = compensated[index, visible] - prediction[visible]
        geometry_mask[index] = visible
    held_mask = geometry_mask[:, held]
    true_root = fixture["root_displacement_truth"][:, held]
    root_error = np.linalg.norm(root_displacement[:, held] - true_root, axis=-1) / diagonal
    root_median = float(np.median(root_error[held_mask])) if held_mask.any() else None
    amplitude = np.linalg.norm(true_root, axis=-1) / diagonal
    high_motion = (amplitude > 0.05) & noncenter[:, None]
    high_valid = held_mask & high_motion
    high_coverage = float(high_valid.sum() / high_motion.sum()) if high_motion.any() else None
    relative_root = (
        float(np.median(root_error[high_valid] / amplitude[high_valid]))
        if high_valid.any()
        else None
    )
    true_articulation = fixture["articulation_truth"][:, held]
    residual_error = articulation[:, held] - true_articulation
    true_articulation_energy = float(np.sum(true_articulation[held_mask] ** 2))
    articulation_relative = (
        float(np.sqrt(np.sum(residual_error[held_mask] ** 2) / true_articulation_energy))
        if true_articulation_energy > 1e-12
        else None
    )
    forward_all = np.all(sequence.valid, axis=0)
    returned_all = np.all(return_mask[noncenter] & np.isfinite(cycles[noncenter]), axis=0)
    forward = float(np.mean(forward_all))
    joint = float(np.mean(forward_all & returned_all))
    measured_cycles = cycles[noncenter][return_mask[noncenter]]
    cycle_median = float(np.median(measured_cycles) / diagonal) if measured_cycles.size else None
    held_coverage = float(held_mask.sum() / (8 * 32))
    checks = {
        "center_query_identity": bool(center_ok),
        "forward_survival": forward >= 0.75,
        "joint_independent_return_survival": joint >= 0.75,
        "cycle_error": cycle_median is not None and cycle_median <= 0.05,
        "actor_known_truth_epe": actor_median is not None and actor_median <= 0.01,
        "held_geometry_support": held_coverage >= 0.75,
        "root_known_truth_epe": root_median is not None and root_median <= 0.01,
        "relative_root_error": not high_motion.any()
        or (high_coverage >= 0.75 and relative_root is not None and relative_root <= 0.2),
        "nonrigid_known_truth_recovery": fixture["case"] != "nonrigid"
        or (articulation_relative is not None and articulation_relative <= 0.5),
    }
    return {
        "fixture_id": fixture["fixture_id"],
        "pass": all(bool(value) for value in checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "actor_endpoint_median_diagonal": actor_median,
        "root_endpoint_median_diagonal": root_median,
        "relative_root_error_above_0_05_diagonal": relative_root,
        "nonrigid_known_truth_relative_rms": articulation_relative,
        "forward_all_time_survival": forward,
        "joint_forward_return_survival": joint,
        "cycle_median_diagonal": cycle_median,
        "cycle_pairs_expected": 512,
        "cycle_pairs_observed": int(return_mask[noncenter].sum()),
        "actor_pairs_expected": 512,
        "actor_pairs_observed": int(temporal_valid.sum()),
        "held_pairs_expected": 256,
        "held_pairs_observed": int(held_mask.sum()),
        "high_motion_pairs_expected": int(high_motion.sum()),
        "high_motion_pairs_observed": int(high_valid.sum()),
        "center_excluded": True,
        "truth_scope": "synthetic physical correspondence only",
        "camera_estimator_evaluated": False,
        "real_point_gate_waived": False,
    }
