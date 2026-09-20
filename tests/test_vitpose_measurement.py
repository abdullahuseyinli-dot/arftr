from __future__ import annotations

import numpy as np

from hac.body_witness_data import BodyCropGeometry
from hac.vitpose_measurement import DESCRIPTOR_BINS, measure_pose_heatmaps


def geometry() -> BodyCropGeometry:
    return BodyCropGeometry(
        native_box=(0.0, 0.0, 192.0, 256.0),
        crop_box=(0, 0, 192, 256),
        source_size=(192, 256),
        extent=1.0,
    )


def test_full_descriptor_is_normalized_and_peak_uses_cell_center():
    heatmaps = np.zeros((8, 64, 48), dtype=np.float32)
    heatmaps[:, 10, 20] = 2.0
    result = measure_pose_heatmaps(heatmaps, np.eye(3), geometry())
    assert result.descriptor.shape == (8, DESCRIPTOR_BINS)
    assert np.allclose(result.descriptor.sum(1), 1)
    assert np.allclose(result.peak_magnitude, 2)
    assert np.allclose(result.peak_raw_xy, [82.0, 42.0])
    expected_bin = (10 // 4) * 12 + (20 // 4)
    assert np.all(result.descriptor[:, expected_bin] > 0.999)


def test_non_target_mass_is_retained_in_outside_bin():
    heatmaps = np.zeros((8, 64, 48), dtype=np.float32)
    heatmaps[:, 10, 20] = 1.0  # upper half
    upper_visible = measure_pose_heatmaps(
        heatmaps, np.eye(3), geometry(), visible_mask_id="upper_visible"
    )
    lower_visible = measure_pose_heatmaps(
        heatmaps, np.eye(3), geometry(), visible_mask_id="lower_visible"
    )
    assert np.all(upper_visible.descriptor[:, -1] > 0.999)
    assert np.all(lower_visible.descriptor[:, -1] < 1e-4)


def test_negative_heatmaps_fall_back_to_outside_without_nan():
    heatmaps = -np.ones((8, 64, 48), dtype=np.float32)
    result = measure_pose_heatmaps(heatmaps, np.eye(3), geometry())
    assert np.isfinite(result.descriptor).all()
    assert np.all(result.descriptor[:, -1] == 1)
