from __future__ import annotations

import numpy as np
import pytest
import torch

from hac.learned_persistent_tracking import LocalCoTracker3, PreparedVideo
from hac.tracking_scale_probe import (
    FixedCrop,
    fixed_actor_crop,
    points_from_crop,
    points_to_crop,
    prepare_matched_crop_pair,
)


def _backend(model_shape=(9, 13)):
    backend = LocalCoTracker3.__new__(LocalCoTracker3)
    backend.torch, backend.device, backend.model_shape, backend.calls = torch, "cpu", model_shape, 0
    return backend


def _ramp_frames(height=37, width=49):
    yy, xx = np.indices((height, width))
    frame = np.stack((xx, yy, xx + yy), -1).astype(np.uint8)
    return [frame, frame + np.uint8(3)]


def test_fixed_crop_uses_center_box_model_aspect_and_fourfold_linear_context():
    roi = fixed_actor_crop((1000, 700, 1040, 780), (2160, 3840), (384, 512))
    assert roi == FixedCrop(806, 579, 428, 321)
    assert roi.width * 384 == roi.height * 512
    assert roi.width >= 4 * 40 and roi.height >= 4 * 80
    assert (roi.left, roi.top, roi.right, roi.bottom) == (806, 579, 1234, 900)


@pytest.mark.parametrize(
    "box,expected",
    [
        ((0, 0, 20, 40), FixedCrop(0, 0, 216, 162)),
        ((300, 200, 320, 240), FixedCrop(104, 78, 216, 162)),
    ],
)
def test_border_crops_shift_without_padding_or_aspect_change(box, expected):
    crop = fixed_actor_crop(box, (240, 320), (384, 512))
    assert crop == expected
    assert crop.left <= box[0] and crop.top <= box[1]
    assert crop.right >= box[2] and crop.bottom >= box[3]
    assert crop.right <= 320 and crop.bottom <= 240


def test_image_limit_reduces_context_but_never_cuts_actor():
    crop = fixed_actor_crop((150, 60, 230, 200), (256, 384), (384, 512))
    assert crop == FixedCrop(20, 1, 340, 255)
    assert crop.width * 3 == crop.height * 4
    with pytest.raises(ValueError, match="without padding"):
        fixed_actor_crop((0, 0, 384, 100), (256, 384), (384, 512))


@pytest.mark.parametrize(
    "box,context",
    [
        ((0, 0, 0, 20), 4),
        ((-1, 0, 10, 20), 4),
        ((0, 0, 1000, 20), 4),
        ((0, 0, np.nan, 20), 4),
        ((0, 0, 10, 20), 0.5),
        ((0, 0, 10, 20), np.inf),
    ],
)
def test_invalid_center_geometry_is_not_silently_clamped(box, context):
    with pytest.raises(ValueError):
        fixed_actor_crop(box, (100, 200), (3, 4), context)


def test_future_box_sequence_and_noninteger_roi_are_rejected():
    with pytest.raises(ValueError, match="One finite center box"):
        fixed_actor_crop(np.ones((9, 4)), (100, 200), (3, 4))
    with pytest.raises(ValueError, match="integers"):
        FixedCrop(0.5, 0, 20, 20)
    with pytest.raises(ValueError, match="integers"):
        FixedCrop(True, 0, 20, 20)


def test_native_roi_point_roundtrip_preserves_ids_missingness_and_outside_coordinates():
    crop = FixedCrop(7, 6, 24, 18)
    points = np.asarray(
        [
            [[7.0, 6.0], [30.0, 23.0], [np.nan, np.nan]],
            [[3.0, 8.0], [10.25, 12.75], [np.nan, np.nan]],
        ]
    )
    original = points.copy()
    local = points_to_crop(points, crop)
    assert local.shape == points.shape
    np.testing.assert_allclose(local[0, :2], [[0.0, 0.0], [23.0, 17.0]])
    assert local[1, 0, 0] == -4.0  # never clamp or drop an outside identity
    np.testing.assert_allclose(points_from_crop(local, crop), points, equal_nan=True)
    np.testing.assert_equal(points, original)


def test_visibility_mask_never_repacks_identity_axis():
    crop = FixedCrop(7, 6, 24, 18)
    points = np.asarray([[[7.0, 6.0], [999.0, 999.0], [30.0, 23.0]]])
    valid = np.asarray([[True, False, True]])
    local = points_to_crop(points, crop, valid)
    assert local.shape == (1, 3, 2) and np.isnan(local[0, 1]).all()
    np.testing.assert_allclose(points_from_crop(local, crop, valid)[valid], points[valid])
    with pytest.raises(ValueError, match="Visibility"):
        points_to_crop(points, crop, valid[:, :2])
    with pytest.raises(ValueError, match="Visible"):
        points_from_crop(local, crop, np.ones((1, 3), bool))


def test_ramp_high_low_parity_has_no_half_pixel_offset_and_preserves_time():
    backend, frames = _backend(), _ramp_frames()
    full = backend.prepare_video(frames)
    roi = FixedCrop(7, 6, 24, 18)
    high, low = prepare_matched_crop_pair(backend, full, frames, roi)
    torch.testing.assert_close(high.tensor, low.tensor, atol=2e-5, rtol=0)
    assert (
        (high.source_height, high.source_width) == (low.source_height, low.source_width) == (18, 24)
    )
    assert tuple(high.tensor.shape) == tuple(low.tensor.shape) == (1, 2, 3, 9, 13)
    np.testing.assert_allclose(low.tensor[0, 0, :2, 0, 0].numpy(), [7.0, 6.0], atol=1e-5)
    np.testing.assert_allclose(low.tensor[0, 0, :2, -1, -1].numpy(), [30.0, 23.0], atol=1e-5)
    torch.testing.assert_close(
        low.tensor[0, 1] - low.tensor[0, 0], torch.full((3, 9, 13), 3.0), atol=2e-5, rtol=0
    )
    assert backend.calls == 0  # preprocessing only, no tracker/model inference


def test_low_detail_samples_model_resolution_without_native_upsample(monkeypatch):
    backend, frames = _backend(), _ramp_frames()
    full = backend.prepare_video(frames)
    actual = torch.nn.functional.grid_sample
    calls = []

    def recorded(tensor, grid, **kwargs):
        calls.append((tuple(tensor.shape), tuple(grid.shape), grid.detach().clone(), kwargs))
        return actual(tensor, grid, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grid_sample", recorded)
    prepare_matched_crop_pair(backend, full, frames, FixedCrop(7, 6, 24, 18))
    assert len(calls) == 1
    tensor_shape, grid_shape, grid, options = calls[0]
    assert tensor_shape == (2, 3, 9, 13) and grid_shape == (2, 9, 13, 2)
    assert grid.min() >= -1.0 and grid.max() <= 1.0
    assert options == {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True}


def test_native_detail_contains_information_lost_by_full_frame_preprocessing():
    backend = _backend()
    yy, xx = np.indices((33, 49))
    checker = ((xx + yy) % 2 * 255).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=-1)
    frames = [image.copy(), image.copy()]
    full = backend.prepare_video(frames)  # all samples land at native multiples of four
    roi = FixedCrop(7, 6, 13, 9)
    high, low = prepare_matched_crop_pair(backend, full, frames, roi)
    assert high.tensor.max() == 255.0 and high.tensor.min() == 0.0
    torch.testing.assert_close(low.tensor, torch.zeros_like(low.tensor), atol=0, rtol=0)
    assert high.source_width == low.source_width and high.source_height == low.source_height


@pytest.mark.parametrize("roi", [FixedCrop(0, 0, 49, 37), FixedCrop(25, 19, 24, 18)])
def test_border_grid_never_inserts_padding_pixels(roi):
    backend = _backend()
    frames = [np.full((37, 49, 3), 149, np.uint8) for _ in range(2)]
    high, low = prepare_matched_crop_pair(backend, backend.prepare_video(frames), frames, roi)
    torch.testing.assert_close(high.tensor, low.tensor, atol=3e-5, rtol=0)
    torch.testing.assert_close(low.tensor, torch.full_like(low.tensor, 149.0), atol=3e-5, rtol=0)


@pytest.mark.parametrize(
    "defect", ["source_size", "time_count", "resolution", "outside_roi", "missing_frame", "dtype"]
)
def test_mismatched_source_or_missing_frame_is_rejected_before_crop_prepare(defect):
    backend, frames = _backend(), _ramp_frames()
    full = backend.prepare_video(frames)
    roi = FixedCrop(7, 6, 24, 18)
    if defect == "source_size":
        full = PreparedVideo(full.tensor, 36, 49)
    elif defect == "time_count":
        full = PreparedVideo(full.tensor[:, :1], 37, 49)
    elif defect == "resolution":
        full = PreparedVideo(full.tensor[..., :-1], 37, 49)
    elif defect == "outside_roi":
        roi = FixedCrop(40, 6, 24, 18)
    elif defect == "missing_frame":
        frames[1] = None
    elif defect == "dtype":
        frames[1] = frames[1].astype(np.float32)

    def forbidden(_):
        raise AssertionError("Invalid input reached crop preprocessing")

    backend.prepare_video = forbidden
    with pytest.raises(ValueError):
        prepare_matched_crop_pair(backend, full, frames, roi)
