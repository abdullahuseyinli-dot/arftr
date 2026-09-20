"""Coordinate/tensor controls only: no CoTracker inference or scientific fits."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hac.camera_following_scale_probe import (
    BACKGROUND_FIT_IDS,
    BACKGROUND_HELD_IDS,
    NATIVE_TO_720,
    camera_from_background,
    map_camera_tracks_to_native,
    prepare_camera_crop_pair,
)
from hac.learned_persistent_tracking import LocalCoTracker3, PreparedVideo
from hac.persistent_articulation_v2 import apply_similarity
from hac.tracking_scale_probe import FixedCrop, prepare_matched_crop_pair


def _transform(a=1.0, b=0.0, tx=0.0, ty=0.0):
    return np.asarray([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def _backend(model_shape=(9, 13)):
    backend = LocalCoTracker3.__new__(LocalCoTracker3)
    backend.torch, backend.device, backend.model_shape, backend.calls = torch, "cpu", model_shape, 0
    return backend


def _ramp_frames(height=61, width=91, count=3):
    yy, xx = np.indices((height, width))
    image = np.stack((xx, yy, xx + yy), -1).astype(np.uint8)
    return [image + np.uint8(3 * time) for time in range(count)]


def _background_fixture():
    rng = np.random.default_rng(20260920)
    reference = rng.uniform([200.0, 200.0], [2800.0, 1500.0], (64, 2))
    matrices = np.stack(
        [
            _transform(1.0 + 0.01 * time, 0.015 * time, 700.0 * time, -120.0 * time)
            for time in (-2, -1, 0, 1, 2)
        ]
    )
    native = np.stack([apply_similarity(matrix, reference) for matrix in matrices])
    source720 = (native + 0.5) / 3.0 - 0.5
    return source720, np.ones(source720.shape[:2], bool), matrices


def test_background_camera_recovers_known_native_motion_and_exact_center_identity():
    positions, valid, expected = _background_fixture()
    before = positions.copy()
    actual, receipt = camera_from_background(positions, valid, 2, 150.0)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=0)
    np.testing.assert_array_equal(actual[2], np.eye(3))
    np.testing.assert_array_equal(positions, before)
    assert receipt["camera_fit_ids"] == list(range(0, 64, 2))
    assert receipt["camera_held_ids"] == list(range(1, 64, 2))
    assert receipt["fit_count_by_time"] == [32] * 5
    assert receipt["held_count_by_time"] == [32, 32, 0, 32, 32]
    assert receipt["held_expected_noncenter_pairs"] == 128
    assert receipt["held_observed_noncenter_pairs"] == 128
    assert receipt["held_noncenter_support_fraction"] == 1.0
    assert receipt["held_error_normalized_median"] < 1e-11
    assert receipt["center_exact_identity"] and not receipt["center_in_error_aggregate"]
    assert receipt["interpolated_frames"] == 0


def test_native_camera_conjugation_includes_area_pixel_center_offset():
    positions, valid, _ = _background_fixture()
    matrix720 = _transform(1.15, 0.12, 8.0, -6.0)
    reference720 = positions[2]
    frames = np.stack((reference720, apply_similarity(matrix720, reference720)))
    camera, _ = camera_from_background(frames, valid[:2], 0, 150.0)
    expected = np.linalg.inv(NATIVE_TO_720) @ matrix720 @ NATIVE_TO_720
    np.testing.assert_allclose(camera[1], expected, atol=2e-10, rtol=0)
    wrong_origin = matrix720.copy()
    wrong_origin[:2, 2] *= 3.0
    assert not np.allclose(camera[1], wrong_origin, atol=0.01)


def test_held_background_corruption_changes_diagnostic_not_fitted_camera():
    positions, valid, _ = _background_fixture()
    original, before = camera_from_background(positions, valid, 2, 150.0)
    altered = positions.copy()
    altered[:, BACKGROUND_HELD_IDS] += [10.0, -20.0]
    altered[2] = positions[2]
    after, receipt = camera_from_background(altered, valid, 2, 150.0)
    np.testing.assert_array_equal(after, original)
    assert before["held_error_normalized_median"] < 1e-11
    assert receipt["held_error_normalized_median"] == pytest.approx(3 * np.hypot(10, 20) / 150)


def test_missing_held_points_do_not_reduce_expected_support_or_reassign_roles():
    positions, valid, _ = _background_fixture()
    valid[0, BACKGROUND_HELD_IDS] = False
    positions[0, BACKGROUND_HELD_IDS] = np.nan
    _, receipt = camera_from_background(positions, valid, 2, 150.0)
    assert receipt["held_expected_noncenter_pairs"] == 128
    assert receipt["held_observed_noncenter_pairs"] == 96
    assert receipt["held_noncenter_support_fraction"] == 0.75
    assert receipt["held_count_by_time"] == [0, 32, 0, 32, 32]
    assert receipt["fit_count_by_time"] == [32] * 5


@pytest.mark.parametrize("defect", ["missing_fit", "rank"])
def test_unusable_fixed_fit_support_aborts_without_interpolation_or_held_borrowing(defect):
    positions, valid, _ = _background_fixture()
    if defect == "missing_fit":
        valid[1, BACKGROUND_FIT_IDS[2:]] = False
        positions[1, BACKGROUND_FIT_IDS[2:]] = np.nan
    else:
        positions[2, BACKGROUND_FIT_IDS] = [12.0, 7.0]
    assert valid[:, BACKGROUND_HELD_IDS].all()
    with pytest.raises(ValueError, match="no held substitution or interpolation"):
        camera_from_background(positions, valid, 2, 150.0)


def test_camera_estimation_is_time_reversal_equivariant_without_identity_repacking():
    positions, valid, _ = _background_fixture()
    forward, _ = camera_from_background(positions, valid, 2, 150.0)
    reverse, _ = camera_from_background(positions[::-1], valid[::-1], 2, 150.0)
    np.testing.assert_array_equal(reverse, forward[::-1])


def test_identity_camera_matches_existing_static_crop_preprocessing():
    backend, frames = _backend(), _ramp_frames()
    full = backend.prepare_video(frames)
    roi = FixedCrop(20, 15, 25, 19)
    identity = np.repeat(np.eye(3)[None], len(frames), axis=0)
    actual = prepare_camera_crop_pair(backend, full, frames, roi, identity)
    expected = prepare_matched_crop_pair(backend, full, frames, roi)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.tensor, reference.tensor, atol=2e-5, rtol=0)
        assert (value.source_height, value.source_width) == (19, 25)
    assert backend.calls == 0


def test_affine_ramp_parity_uses_same_native_camera_grid_without_half_pixel_shift():
    backend, frames = _backend(), _ramp_frames()
    full = backend.prepare_video(frames)
    roi = FixedCrop(20, 15, 25, 19)
    transforms = np.stack(
        (_transform(0.95, -0.1, 3.25, 1.5), np.eye(3), _transform(1.05, 0.07, -1.2, 2.4))
    )
    high, low = prepare_camera_crop_pair(backend, full, frames, roi, transforms)
    torch.testing.assert_close(high.tensor, low.tensor, atol=4e-5, rtol=0)
    xx, yy = np.meshgrid(
        np.linspace(roi.left, roi.right - 1, 13), np.linspace(roi.top, roi.bottom - 1, 9)
    )
    grid = np.stack((xx, yy), -1)
    for time, camera in enumerate(transforms):
        physical = apply_similarity(camera, grid)
        expected = np.stack((physical[..., 0], physical[..., 1], physical.sum(-1)), 0) + 3 * time
        np.testing.assert_allclose(high.tensor[0, time].numpy(), expected, atol=3e-5, rtol=0)
    assert high.source_width == low.source_width and high.source_height == low.source_height
    assert backend.calls == 0


def test_low_arm_reads_only_exact_prepared_tensor_and_both_arms_share_every_grid(monkeypatch):
    backend, frames = _backend(), _ramp_frames()
    full = backend.prepare_video(frames)
    before = full.tensor.clone()
    roi = FixedCrop(20, 15, 25, 19)
    transforms = np.stack((_transform(tx=-3), np.eye(3), _transform(tx=7)))
    actual_grid_sample = torch.nn.functional.grid_sample
    calls = []

    def record(source, grid, **options):
        calls.append((tuple(source.shape), source.data_ptr(), grid.clone(), options))
        return actual_grid_sample(source, grid, **options)

    def forbidden(_):
        raise AssertionError("No crop re-preparation or native-resolution upsample is allowed")

    monkeypatch.setattr(torch.nn.functional, "grid_sample", record)
    backend.prepare_video = forbidden
    prepare_camera_crop_pair(backend, full, frames, roi, transforms)
    assert len(calls) == 2 * len(frames)
    for time in range(len(frames)):
        high, low = calls[2 * time : 2 * time + 2]
        assert high[0] == (1, 3, 61, 91)
        assert low[0] == (1, 3, 9, 13)
        assert low[1] == full.tensor[:, time].data_ptr()
        torch.testing.assert_close(high[2], low[2], atol=0, rtol=0)
        assert (
            high[3]
            == low[3]
            == {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True}
        )
    torch.testing.assert_close(full.tensor, before, atol=0, rtol=0)


def test_native_detail_is_not_recoverable_from_low_tensor_by_coordinate_magnification():
    backend = _backend()
    yy, xx = np.indices((33, 49))
    checker = np.repeat((((xx + yy) % 2) * 255).astype(np.uint8)[..., None], 3, axis=-1)
    frames = [checker.copy(), checker.copy()]
    full = backend.prepare_video(frames)
    transforms = np.repeat(np.eye(3)[None], 2, axis=0)
    high, low = prepare_camera_crop_pair(backend, full, frames, FixedCrop(7, 6, 13, 9), transforms)
    assert float(high.tensor.max()) > 254.99
    assert float(high.tensor.min()) < 0.01
    torch.testing.assert_close(low.tensor, torch.zeros_like(low.tensor), atol=0, rtol=0)
    assert backend.calls == 0


def test_large_camera_motion_remains_in_following_crop_but_leaves_static_crop():
    height, width = 61, 241
    roi = FixedCrop(80, 20, 25, 19)
    backend = _backend((19, 25))
    pattern = np.random.default_rng(29).integers(1, 255, (19, 25, 3), dtype=np.uint8)
    transforms = np.stack((_transform(tx=-60), np.eye(3), _transform(tx=100)))
    frames = []
    for offset in (-60, 0, 100):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[20:39, 80 + offset : 105 + offset] = pattern
        frames.append(frame)
    full = backend.prepare_video(frames)
    # Use double sampling to isolate exact large-motion geometry from float32
    # grid quantization at high-contrast pixels; production float32 is covered
    # by the ramp and shared-grid tests above.
    full = PreparedVideo(full.tensor.to(torch.float64), height, width)
    high, _ = prepare_camera_crop_pair(backend, full, frames, roi, transforms)
    expected = torch.from_numpy(pattern).permute(2, 0, 1).double()
    for time in range(3):
        torch.testing.assert_close(high.tensor[0, time], expected, atol=1e-9, rtol=0)
    assert not frames[0][20:39, 80:105].any() and not frames[2][20:39, 80:105].any()
    local = np.repeat(np.array([[[4.0, 5.0], [20.0, 15.0]]]), 3, axis=0)
    physical, visible = map_camera_tracks_to_native(
        local, np.ones((3, 2), bool), transforms, roi, (height, width)
    )
    assert visible.all()
    np.testing.assert_array_equal(physical[:, 0, 0], [24.0, 84.0, 184.0])


def test_outside_native_samples_are_exactly_zero_in_both_resolution_arms():
    backend = _backend()
    frames = [np.full((37, 49, 3), 149, np.uint8) for _ in range(2)]
    roi = FixedCrop(7, 6, 24, 18)
    transforms = np.stack((np.eye(3), _transform(tx=-7.5, ty=-6.5)))
    high, low = prepare_camera_crop_pair(
        backend, backend.prepare_video(frames), frames, roi, transforms
    )
    torch.testing.assert_close(high.tensor, low.tensor, atol=3e-5, rtol=0)
    assert torch.count_nonzero(high.tensor[0, 1, :, 0]) == 0
    assert torch.count_nonzero(high.tensor[0, 1, :, :, 0]) == 0
    torch.testing.assert_close(
        high.tensor[0, 1, :, 1:, 1:], torch.full((3, 8, 12), 149.0), atol=3e-5, rtol=0
    )
    local = np.repeat(np.array([[[0.0, 0.0], [1.0, 1.0], [23.2, 5.0]]]), 2, axis=0)
    mapped, ok = map_camera_tracks_to_native(
        local, np.ones((2, 3), bool), transforms, roi, (37, 49)
    )
    np.testing.assert_array_equal(ok, [[True, True, False], [False, True, False]])
    assert np.isnan(mapped[1, 0]).all()  # Padding cannot manufacture visible stationary points.
    np.testing.assert_allclose(mapped[1, 1], [0.5, 0.5])


def test_padding_only_tracker_success_is_rejected_not_counted_as_survival():
    backend, frames = _backend(), _ramp_frames(count=2)
    roi = FixedCrop(20, 15, 25, 19)
    transforms = np.stack((np.eye(3), _transform(tx=1000, ty=1000)))
    high, low = prepare_camera_crop_pair(
        backend, backend.prepare_video(frames), frames, roi, transforms
    )
    assert (
        torch.count_nonzero(high.tensor[:, 1]) == 0 and torch.count_nonzero(low.tensor[:, 1]) == 0
    )
    hallucinated = np.full((2, 4, 2), 4.0)
    mapped, visible = map_camera_tracks_to_native(
        hallucinated, np.ones((2, 4), bool), transforms, roi, (61, 91)
    )
    assert visible[0].all() and not visible[1].any()
    assert np.isnan(mapped[1]).all() and not np.all(visible, axis=0).any()


def test_mapping_keeps_missing_id_axis_and_does_not_mutate_inputs():
    roi = FixedCrop(20, 15, 25, 19)
    positions = np.array(
        [[[2, 3], [np.nan, np.nan], [9, 10]], [[4, 5], [999, 999], [11, 12]]], dtype=float
    )
    valid = np.array([[True, False, True], [True, False, True]])
    transforms = np.stack((np.eye(3), _transform(tx=5, ty=-2)))
    copies = [value.copy() for value in (positions, valid, transforms)]
    mapped, actual_valid = map_camera_tracks_to_native(positions, valid, transforms, roi, (61, 91))
    assert mapped.shape == (2, 3, 2)
    np.testing.assert_array_equal(actual_valid, valid)
    assert np.isnan(mapped[:, 1]).all()
    np.testing.assert_allclose(mapped[1, [0, 2]], [[29, 18], [36, 25]])
    for before, after in zip(copies, (positions, valid, transforms), strict=True):
        np.testing.assert_array_equal(before, after)


def test_reverse_query_time_uses_corresponding_reversed_camera_not_forward_order():
    roi = FixedCrop(200, 120, 80, 60)
    transforms = np.stack((_transform(tx=-80), np.eye(3), _transform(tx=120)))
    local = np.asarray(
        [[[10, 12], [30, 20]], [[11, 13], [31, 21]], [[12, 14], [32, 22]]], dtype=float
    )
    valid = np.ones((3, 2), bool)
    native, _ = map_camera_tracks_to_native(local, valid, transforms, roi, (768, 1024))
    reverse_native, _ = map_camera_tracks_to_native(
        local[::-1], valid[::-1], transforms[::-1], roi, (768, 1024)
    )
    np.testing.assert_array_equal(reverse_native, native[::-1])
    wrong, _ = map_camera_tracks_to_native(local[::-1], valid[::-1], transforms, roi, (768, 1024))
    assert not np.allclose(wrong[[0, 2]], native[::-1][[0, 2]])
    original_query_time = 0
    reverse_query_time = 2 - original_query_time
    recovered_local = apply_similarity(
        np.linalg.inv(transforms[::-1][reverse_query_time]), native[original_query_time]
    )
    recovered_local -= [roi.left, roi.top]
    np.testing.assert_allclose(recovered_local, local[original_query_time])


def test_exact_center_identity_preserves_native_cycle_distance():
    roi = FixedCrop(200, 120, 80, 60)
    seed = np.array([[[10.0, 12.0], [30.0, 20.0]]])
    returned = seed + [[[0.1, -0.2], [-0.3, 0.4]]]
    identity = np.eye(3)[None]
    source, _ = map_camera_tracks_to_native(seed, np.ones((1, 2), bool), identity, roi, (768, 1024))
    target, _ = map_camera_tracks_to_native(
        returned, np.ones((1, 2), bool), identity, roi, (768, 1024)
    )
    np.testing.assert_allclose(
        np.linalg.norm(target - source, axis=-1),
        np.linalg.norm(returned - seed, axis=-1),
        atol=3e-14,
        rtol=0,
    )


@pytest.mark.parametrize("defect", ["missing", "projective", "shear", "reflection", "singular"])
def test_invalid_camera_is_rejected_before_sampling_or_point_mapping(defect):
    backend, frames = _backend(), _ramp_frames(count=2)
    full = backend.prepare_video(frames)
    transforms = np.repeat(np.eye(3)[None], 2, axis=0)
    if defect == "missing":
        transforms[1] = np.nan
    elif defect == "projective":
        transforms[1, 2, 0] = 0.1
    elif defect == "shear":
        transforms[1, 0, 1] = 0.1
    elif defect == "reflection":
        transforms[1, 0, 0] = -1
    else:
        transforms[1, :2, :2] = 0
    roi = FixedCrop(20, 15, 25, 19)
    with pytest.raises(ValueError):
        prepare_camera_crop_pair(backend, full, frames, roi, transforms)
    with pytest.raises(ValueError):
        map_camera_tracks_to_native(
            np.zeros((2, 4, 2)), np.ones((2, 4), bool), transforms, roi, (61, 91)
        )


@pytest.mark.parametrize(
    "defect", ["source_dimensions", "time_count", "native_dtype", "missing_frame", "tensor_nan"]
)
def test_mismatched_frame_provenance_shapes_and_pixels_fail_closed(defect):
    backend, frames = _backend(), _ramp_frames(count=2)
    full = backend.prepare_video(frames)
    if defect == "source_dimensions":
        full = PreparedVideo(full.tensor, 60, 91)
    elif defect == "time_count":
        full = PreparedVideo(full.tensor[:, :1], 61, 91)
    elif defect == "native_dtype":
        frames[1] = frames[1].astype(np.float32)
    elif defect == "missing_frame":
        frames[1] = None
    else:
        corrupted = full.tensor.clone()
        corrupted[0, 0, 0, 0, 0] = np.nan
        full = PreparedVideo(corrupted, 61, 91)
    with pytest.raises(ValueError):
        prepare_camera_crop_pair(
            backend, full, frames, FixedCrop(20, 15, 25, 19), np.repeat(np.eye(3)[None], 2, axis=0)
        )
    assert backend.calls == 0
