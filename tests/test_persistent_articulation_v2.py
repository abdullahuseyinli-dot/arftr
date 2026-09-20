from __future__ import annotations

import numpy as np
import pytest

from hac.persistent_articulation_v2 import (
    PointSplit,
    TrackSequence,
    apply_similarity,
    common_held_endpoint_errors,
    decompose_tracks,
    fixed_point_split,
    identity_time_derivatives,
    similarity_design,
)


def _transform(a=1.0, b=0.0, tx=0.0, ty=0.0):
    return np.asarray([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def _scene(camera_speed, actor_speed, *, articulation=False, rotate=False):
    rng = np.random.default_rng(7301)
    actor0 = rng.uniform(-0.35, 0.35, (16, 2))
    bg0 = rng.uniform(-2.0, 2.0, (16, 2))
    aid, bid = tuple(f"actor-{i:02d}" for i in range(16)), tuple(f"bg-{i:02d}" for i in range(16))
    asplit, bsplit = fixed_point_split(aid), fixed_point_split(bid)
    fit, _ = asplit.indices(aid)
    times = np.asarray([-1.0, -0.61, -0.2, 0.0, 0.35, 0.8, 1.0])
    camera, root, observed, background, expected_art = [], [], [], [], []
    direction = rng.normal(size=actor0.shape)
    # Remove every similarity DOF on the exact fitting IDs, not on pooled rows.
    beta = np.linalg.lstsq(similarity_design(actor0[fit]), direction[fit].reshape(-1), rcond=None)[
        0
    ]
    direction -= (similarity_design(actor0) @ beta).reshape(-1, 2)
    for t in times:
        c = _transform(
            1.0 + 0.03 * t if rotate else 1.0,
            0.04 * t if rotate else 0.0,
            camera_speed[0] * t,
            camera_speed[1] * t,
        )
        r = _transform(
            1.0 + 0.02 * t if rotate else 1.0,
            -0.03 * t if rotate else 0.0,
            actor_speed[0] * t,
            actor_speed[1] * t,
        )
        art = 0.03 * np.sin(t) * direction if articulation else np.zeros_like(actor0)
        camera.append(c)
        root.append(r)
        expected_art.append(art)
        observed.append(apply_similarity(c, apply_similarity(r, actor0) + art))
        background.append(apply_similarity(c, bg0))
    valid = np.ones((len(times), 16), bool)
    actor = TrackSequence(np.asarray(observed), valid, times, aid)
    bg = TrackSequence(np.asarray(background), valid, times, bid)
    result = decompose_tracks(
        actor, bg, center_index=3, actor_split=asplit, background_split=bsplit
    )
    return (
        actor,
        bg,
        result,
        np.asarray(camera),
        np.asarray(root),
        np.asarray(expected_art),
        asplit,
        bsplit,
    )


@pytest.mark.parametrize(
    "camera,actor",
    [
        ((0.2, -0.1), (0.0, 0.0)),
        ((0.0, 0.0), (0.08, 0.02)),
        ((0.2, 0.0), (0.04, 0.0)),
        ((0.2, 0.0), (-0.04, 0.0)),
    ],
)
def test_known_camera_and_root_motion_are_recovered_including_stationary_actor(camera, actor):
    seq, bg, result, expected_camera, expected_root, art, asplit, bsplit = _scene(camera, actor)
    assert result.camera_valid.all() and result.root_valid.all()
    np.testing.assert_allclose(result.camera_transforms, expected_camera, atol=1e-12)
    np.testing.assert_allclose(result.root_transforms, expected_root, atol=1e-12)
    np.testing.assert_allclose(result.articulation, art, atol=1e-12)
    np.testing.assert_allclose(
        result.root_displacement,
        seq.times_seconds[:, None, None] * np.asarray(actor)[None, None, :] * np.ones((1, 16, 1)),
        atol=1e-12,
    )
    # Perfect stationary-actor compensation is zero, which invalidated the old >=.5 retention gate.
    if actor == (0.0, 0.0):
        assert np.max(np.abs(result.root_displacement)) < 1e-12


def test_nonrigid_articulation_orthogonal_to_root_is_recovered():
    seq, _, result, cameras, roots, art, _, _ = _scene(
        (0.2, -0.1), (0.04, 0.08), articulation=True, rotate=True
    )
    np.testing.assert_allclose(result.camera_transforms, cameras, atol=1e-12)
    np.testing.assert_allclose(result.root_transforms, roots, atol=1e-12)
    np.testing.assert_allclose(result.articulation, art, atol=1e-12)
    np.testing.assert_allclose(
        result.camera_compensated_actor,
        np.stack([apply_similarity(r, seq.positions[3]) for r in roots]) + art,
        atol=1e-12,
    )


def test_center_identity_and_forward_backward_transform_composition():
    seq, _, result, cameras, roots, _, _, _ = _scene((0.2, -0.1), (0.04, 0.08), rotate=True)
    np.testing.assert_allclose(result.camera_transforms[3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(result.root_transforms[3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(result.articulation[3], 0.0, atol=1e-12)
    for camera, root, observed in zip(cameras, roots, seq.positions, strict=True):
        combined = camera @ root
        np.testing.assert_allclose(
            apply_similarity(np.linalg.inv(combined), observed), seq.positions[3], atol=1e-12
        )
        np.testing.assert_allclose(combined @ np.linalg.inv(combined), np.eye(3), atol=1e-12)


def test_derivatives_preserve_identity_physical_dt_and_time_reversal():
    times = np.asarray([0.0, 0.25, 1.0, 2.0])
    identity_offset = np.asarray([[0.0, 0.0], [10.0, -20.0], [100.0, 300.0]])
    positions = identity_offset[None] + times[:, None, None] * np.asarray([2.0, -3.0])
    valid = np.ones((4, 3), bool)
    result = identity_time_derivatives(positions, valid, times)
    np.testing.assert_allclose(result.velocity, np.broadcast_to([2.0, -3.0], (3, 3, 2)))
    np.testing.assert_allclose(result.acceleration, 0.0, atol=1e-12)
    reverse = identity_time_derivatives(positions[::-1], valid[::-1], -times[::-1])
    np.testing.assert_allclose(reverse.velocity, -result.velocity[::-1])
    np.testing.assert_allclose(reverse.acceleration, result.acceleration[::-1])


def test_nonzero_acceleration_reverses_with_even_parity():
    times = np.asarray([0.0, 0.25, 1.0, 2.0])
    positions = times[:, None, None] ** 2 * np.asarray([2.0, -3.0])[None, None, :]
    valid = np.ones((4, 1), bool)
    result = identity_time_derivatives(positions, valid, times)
    np.testing.assert_allclose(result.acceleration, np.broadcast_to([4.0, -6.0], (2, 1, 2)))
    reverse = identity_time_derivatives(positions[::-1], valid[::-1], -times[::-1])
    np.testing.assert_allclose(reverse.velocity, -result.velocity[::-1])
    np.testing.assert_allclose(reverse.acceleration, result.acceleration[::-1])


def test_missing_identity_is_not_repacked_or_bridged():
    times = np.asarray([0.0, 0.5, 1.0, 2.0])
    positions = np.broadcast_to(times[:, None, None], (4, 3, 2)).copy()
    valid = np.ones((4, 3), bool)
    valid[1, 1] = False
    positions[1, 1] = np.nan
    result = identity_time_derivatives(positions, valid, times)
    assert result.velocity_valid[:, 1].tolist() == [False, False, True]
    assert np.isnan(result.velocity[:2, 1]).all()
    np.testing.assert_allclose(result.velocity[:, 0], 1.0)


def test_fixed_roles_do_not_leak_held_points_or_change_after_visibility_loss():
    actor, bg, result, _, _, _, asplit, bsplit = _scene((0.2, 0.0), (0.04, 0.03))
    bfit, bheld = bsplit.indices(bg.point_ids)
    corrupted = bg.positions.copy()
    corrupted[:, bheld] += [0.4, -0.5]
    corrupted[3] = bg.positions[3]
    altered = TrackSequence(corrupted, bg.valid, bg.times_seconds, bg.point_ids)
    after = decompose_tracks(
        actor, altered, center_index=3, actor_split=asplit, background_split=bsplit
    )
    np.testing.assert_allclose(after.camera_transforms, result.camera_transforms)
    visibility = bg.valid.copy()
    visibility[0, bfit] = False
    lost = TrackSequence(bg.positions, visibility, bg.times_seconds, bg.point_ids)
    after = decompose_tracks(
        actor, lost, center_index=3, actor_split=asplit, background_split=bsplit
    )
    assert not after.camera_valid[0]  # held observations cannot substitute for fit IDs
    with pytest.raises(ValueError, match="disjoint"):
        PointSplit(("a", "b"), ("b", "c")).indices(("a", "b", "c"))


def test_reordered_identity_axis_preserves_roles_and_geometry():
    actor, bg, result, _, _, _, asplit, bsplit = _scene((0.2, 0.0), (0.04, 0.03), articulation=True)
    order = np.random.default_rng(14).permutation(len(actor.point_ids))
    ids = tuple(actor.point_ids[i] for i in order)
    assert fixed_point_split(ids) == asplit
    reordered = TrackSequence(
        actor.positions[:, order], actor.valid[:, order], actor.times_seconds, ids
    )
    after = decompose_tracks(
        reordered, bg, center_index=3, actor_split=asplit, background_split=bsplit
    )
    np.testing.assert_allclose(after.root_transforms, result.root_transforms, atol=1e-12)
    np.testing.assert_allclose(after.articulation, result.articulation[:, order], atol=1e-12)


def test_common_held_error_excludes_center_and_reports_missing_support():
    actor, _, _, _, _, _, split, _ = _scene((0.0, 0.0), (0.1, 0.0))
    _, held = split.indices(actor.point_ids)
    perfect = actor.positions.copy()
    bad = actor.positions.copy()
    bad += [0.1, 0.0]
    perfect[3] += 100.0  # center must have no effect
    masks = {"good": actor.valid.copy(), "bad": actor.valid.copy()}
    masks["good"][0, held[0]] = False
    report = common_held_endpoint_errors(
        actor,
        {"good": perfect, "bad": bad},
        masks,
        split=split,
        center_index=3,
        actor_box_diagonal=1.0,
    )
    assert report["expected_held_time_pairs"] == 6 * 8
    assert report["common_pairs"] == 47 and report["missing_common_pairs"] == 1
    assert report["per_carrier_available_pairs"] == {"good": 47, "bad": 48}
    assert not report["common_mask"][3].any()
    assert report["median_error_normalized"]["good"] == 0.0
    assert report["median_error_normalized"]["bad"] == pytest.approx(0.1)


def test_wrong_actor_transform_is_scored_on_same_held_target_identities():
    target, _, result, _, _, _, split, _ = _scene((0.1, 0.0), (0.08, 0.0))
    other, _, other_result, _, _, _, _, _ = _scene((0.1, 0.0), (-0.08, 0.0))
    wrong = np.stack(
        [
            apply_similarity(c, apply_similarity(r, target.positions[3]))
            for c, r in zip(
                other_result.camera_transforms, other_result.root_transforms, strict=True
            )
        ]
    )
    report = common_held_endpoint_errors(
        target,
        {"right": result.root_actor_prediction, "wrong": wrong},
        {
            "right": result.root_actor_prediction_valid,
            "wrong": other_result.root_actor_prediction_valid,
        },
        split=split,
        center_index=3,
        actor_box_diagonal=1.0,
    )
    right, wrong_error = (report["median_error_normalized"][key] for key in ("right", "wrong"))
    assert right < 1e-12 and wrong_error > 0.05
    assert (wrong_error - right) / max(right, 0.005) > 0.25


def test_identity_permutation_fails_held_target_prediction():
    actor, bg, correct, _, _, _, split, bsplit = _scene((0.1, 0.0), (0.08, -0.02))
    fit, _ = split.indices(actor.point_ids)
    positions = actor.positions.copy()
    for time in range(len(positions)):
        if time != 3:
            positions[time, fit] = positions[time, np.roll(fit, 1)]
    corrupted = TrackSequence(positions, actor.valid, actor.times_seconds, actor.point_ids)
    wrong = decompose_tracks(
        corrupted, bg, center_index=3, actor_split=split, background_split=bsplit
    )
    report = common_held_endpoint_errors(
        actor,
        {"right": correct.root_actor_prediction, "wrong": wrong.root_actor_prediction},
        {"right": correct.root_actor_prediction_valid, "wrong": wrong.root_actor_prediction_valid},
        split=split,
        center_index=3,
        actor_box_diagonal=1.0,
    )
    right, bad = (report["median_error_normalized"][key] for key in ("right", "wrong"))
    assert report["common_pairs"] == report["expected_held_time_pairs"]
    assert (bad - right) / max(right, 0.005) > 0.25


def test_invalid_clock_or_visible_nan_is_rejected():
    positions = np.zeros((3, 6, 2))
    valid = np.ones((3, 6), bool)
    with pytest.raises(ValueError, match="increasing"):
        identity_time_derivatives(positions, valid, [0.0, 0.0, 1.0])
    positions[1, 1] = np.nan
    with pytest.raises(ValueError, match="visible"):
        identity_time_derivatives(positions, valid, [0.0, 0.5, 1.0])
