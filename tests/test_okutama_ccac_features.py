from __future__ import annotations

import numpy as np

from hac.ccac_features import (
    ARMS,
    EXPECTED_WIDTHS,
    CCACFeatures,
    aggregate_clip_features,
    arm_inputs,
    validate_feature_blocks,
)
from hac.video_token_moments import TokenMomentFeatures


def _pair(index: int, *, usable: bool = True) -> dict:
    value = float(index + 1)
    return {
        "sample_id": "sample",
        "pair_index": index,
        "pair_available": True,
        "camera_usable": usable,
        "translation_usable": usable,
        "articulation_usable": usable,
        "actor_seed_points": 16,
        "actor_primary_retained_points": 12,
        "actor_vertical_regions": 3,
        "actor_primary_fb_median_pixels": 0.1,
        "camera_audit_median_pixels": 0.2,
        "raw_translation_x_height_per_second": value,
        "raw_translation_y_height_per_second": -value,
        "compensated_translation_x_height_per_second": value / 2,
        "compensated_translation_y_height_per_second": -value / 4,
        "articulation_median_height_per_second": value / 10,
        "articulation_p90_height_per_second": value / 5,
    }


def _clip() -> dict:
    return {
        "sample_id": "sample",
        "center_track_survival_fraction_75pct": 0.5,
        "center_seed_points": 24,
        "native_center_height": 72.0,
    }


def test_fixed_clip_blocks_use_shared_mask_and_linear_quantiles() -> None:
    pairs = [_pair(index) for index in range(15)]
    quality, raw, compensated, within, translation_valid, articulation_valid = (
        aggregate_clip_features(pairs, _clip())
    )
    assert quality.shape == (16,)
    np.testing.assert_allclose(quality[:6], 1)
    np.testing.assert_allclose(quality[6:9], (0.5, 0.375, 1.0))
    np.testing.assert_allclose(quality[9:16], (0.1, 1.0, 0.2, 1.0, 0.5, 0.75, 0.1))
    expected = np.quantile(np.arange(1, 16, dtype=float), (0.1, 0.5, 0.9))
    np.testing.assert_allclose(raw[:3], expected)
    np.testing.assert_allclose(raw[3:6], -expected[::-1])
    np.testing.assert_allclose(compensated[:3], expected / 2)
    np.testing.assert_allclose(within[:3], expected / 10)
    assert translation_valid and articulation_valid


def test_insufficient_pairs_zero_only_aggregate_motion_blocks() -> None:
    pairs = [_pair(index, usable=index < 9) for index in range(15)]
    quality, raw, compensated, within, translation_valid, articulation_valid = (
        aggregate_clip_features(pairs, _clip())
    )
    assert not translation_valid and not articulation_valid
    assert quality[4] == 0 and quality[5] == 0
    assert np.all(raw == 0) and np.all(compensated == 0) and np.all(within == 0)
    assert quality[0] == 1 and quality[2] == np.float32(9 / 15)


def _base(rows: int) -> TokenMomentFeatures:
    return TokenMomentFeatures(
        np.zeros((rows, 3072), np.float32),
        np.zeros((rows, 4608), np.float32),
        np.zeros((rows, 4608), np.float32),
        np.zeros((rows, 6912), np.float32),
        np.zeros((rows, 6144), np.float32),
        np.ones(rows, bool),
    )


def test_arm_inputs_keep_posture_fixed_and_match_declared_widths() -> None:
    rows = 2
    ccac = CCACFeatures(
        np.zeros((rows, 16), np.float32),
        np.zeros((rows, 9), np.float32),
        np.zeros((rows, 9), np.float32),
        np.zeros((rows, 6), np.float32),
        np.zeros(rows, bool),
        np.zeros(rows, bool),
    )
    validate_feature_blocks(ccac, rows)
    base = _base(rows)
    posture_arrays = []
    for arm in ARMS:
        inputs = arm_inputs(base, ccac, arm)
        expected_posture, expected_motion = EXPECTED_WIDTHS[arm]
        assert inputs.posture_or_direct.shape == (rows, expected_posture)
        assert inputs.motion_references[0].shape == (rows, expected_motion)
        posture_arrays.append(inputs.posture_or_direct)
    assert all(np.array_equal(posture_arrays[0], values) for values in posture_arrays[1:])
