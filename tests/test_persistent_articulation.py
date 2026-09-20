from __future__ import annotations

import numpy as np
import cv2

from hac.persistent_articulation import actor_seed_points, fit_similarity, project_points
from hac.persistent_tracking import track_persistent, track_step


def test_persistent_tracker_keeps_identity_through_translation() -> None:
    rng = np.random.default_rng(11)
    first = cv2.GaussianBlur(rng.integers(0, 256, (160, 200), dtype=np.uint8), (3, 3), 0)
    frames = [
        cv2.warpAffine(first, np.asarray(((1.0, 0.0, 2.0 * i), (0.0, 1.0, -1.0 * i))), (200, 160))
        for i in range(5)
    ]
    seeds = cv2.goodFeaturesToTrack(first, maxCorners=24, qualityLevel=0.01, minDistance=5, blockSize=3)
    assert seeds is not None
    points = seeds.reshape(-1, 2)
    tracks = track_persistent(
        frames,
        points,
        center_index=2,
        spec={
            "window": [15, 15],
            "maximum_pyramid_level": 2,
            "iterations": 30,
            "epsilon": 0.01,
            "minimum_eigenvalue": 0.00001,
        },
        maximum_error=1.0,
    )
    assert tracks.valid[2].all()
    assert tracks.valid[1].sum() >= len(points) * 0.8
    assert tracks.valid[3].sum() >= len(points) * 0.8
    assert np.nanmedian(tracks.link_forward_backward_error) < 0.5


def test_similarity_fit_and_projection_are_deterministic() -> None:
    source = np.asarray([(x, y) for y in range(10, 90, 10) for x in range(10, 100, 10)], dtype=np.float32)
    target = source @ np.asarray(((1.0, -0.03), (0.03, 1.0)), dtype=np.float32).T + (4.0, -2.0)
    fit = fit_similarity(source, target, seed=7)
    assert fit.usable
    assert fit.median_error is not None and fit.median_error < 0.1
    np.testing.assert_allclose(project_points(fit.transform, source), target, atol=0.1)


def test_actor_seed_points_are_inside_box_and_stratified() -> None:
    rng = np.random.default_rng(4)
    image = cv2.GaussianBlur(rng.integers(0, 256, (120, 160), dtype=np.uint8), (3, 3), 0)
    result = actor_seed_points(image, (30.0, 20.0, 130.0, 110.0), maximum=32)
    assert 0 < len(result.points) <= 32
    assert ((result.points[:, 0] >= 30) & (result.points[:, 0] <= 130)).all()
    assert ((result.points[:, 1] >= 20) & (result.points[:, 1] <= 110)).all()

