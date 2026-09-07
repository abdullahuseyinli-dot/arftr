from __future__ import annotations

import numpy as np
import pytest

from hac.video_fusion import (
    ARMS,
    SOURCE_ARMS,
    arm_features,
    decode_factorized_probabilities,
    derive_features,
)


def source_arrays(rows=3):
    real = np.zeros((rows, 8, 9, 768), dtype=np.float16)
    repeated = np.zeros_like(real)
    dino = np.zeros((rows, 16, 1, 768), dtype=np.float16)
    validity = {arm: np.ones(rows, dtype=bool) for arm in SOURCE_ARMS}
    return real, repeated, dino, validity


def test_derivation_exact_formulas_float32_and_block_invariance():
    real, repeated, dino, validity = source_arrays()
    real[:] = np.arange(8, dtype=np.float16)[None, :, None, None]
    real[1] *= -2
    repeated[:] = 1
    dino[:] = 4
    before = real.copy()
    result = derive_features(real, repeated, dino, validity, block_size=1)
    other = derive_features(real, repeated, dino, validity, block_size=64)
    np.testing.assert_array_equal(real, before)
    for name in ("video_mean", "dino_mean", "motion"):
        actual = getattr(result, name)
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, getattr(other, name))
    np.testing.assert_array_equal(result.video_mean[:, 0], [3.5, -7, 3.5])
    np.testing.assert_array_equal(result.dino_mean[:, 0], [4, 4, 4])
    np.testing.assert_array_equal(result.motion[:, 0], [2.75, 8, 2.75])
    np.testing.assert_array_equal(result.motion[:, 768], [1, 2, 1])


def test_half_precision_subtraction_never_overflows_before_float32_cast():
    real, repeated, dino, validity = source_arrays(1)
    real[:] = np.finfo(np.float16).max
    repeated[:] = -np.finfo(np.float16).max
    result = derive_features(real, repeated, dino, validity)
    assert result.motion[0, 0] == 131008
    assert np.isfinite(result.motion).all()


def test_concatenation_order_dimensions_and_separate_specialist_inputs():
    real, repeated, dino, validity = source_arrays(1)
    real[:] = 3
    repeated[:] = 1
    dino[:] = 7
    features = derive_features(real, repeated, dino, validity)
    for arm, width in zip(ARMS[:3], (768, 1536, 3072), strict=True):
        values, second = arm_features(features, arm)
        assert values.shape == (1, width) and second is None
        np.testing.assert_array_equal(values[:, :768], features.video_mean)
    posture, motion = arm_features(features, ARMS[3])
    assert posture.shape == (1, 1536) and motion.shape == (1, 2304)
    np.testing.assert_array_equal(posture[:, 768:], features.dino_mean)
    np.testing.assert_array_equal(motion[:, 768:], features.motion)
    with pytest.raises(ValueError, match="Unknown"):
        arm_features(features, "unlocked")


def test_validity_intersection_preserves_rows_and_does_not_mutate_inputs():
    real, repeated, dino, validity = source_arrays()
    validity[SOURCE_ARMS[1]][1] = False
    validity[SOURCE_ARMS[2]][2] = False
    result = derive_features(real, repeated, dino, validity)
    assert len(result.video_mean) == 3
    np.testing.assert_array_equal(result.validity[ARMS[0]], [True, True, True])
    np.testing.assert_array_equal(result.validity[ARMS[1]], [True, True, False])
    np.testing.assert_array_equal(result.validity[ARMS[2]], [True, False, False])
    result.validity[ARMS[0]][0] = False
    assert validity[SOURCE_ARMS[0]][0]


@pytest.mark.parametrize("failure", ["shape", "dtype", "validity", "nonfinite", "block"])
def test_bad_derivation_inputs_fail_closed(failure):
    real, repeated, dino, validity = source_arrays(1)
    block = 64
    if failure == "shape":
        repeated = repeated[:, :7]
    elif failure == "dtype":
        dino = dino.astype(np.float32)
    elif failure == "validity":
        validity[SOURCE_ARMS[0]] = np.ones(1, dtype=int)
    elif failure == "nonfinite":
        real[0, 0, 0, 0] = np.nan
    else:
        block = 0
    with pytest.raises(ValueError):
        derive_features(real, repeated, dino, validity, block_size=block)


def test_factorized_decode_preserves_class_order_and_extreme_probabilities():
    result = decode_factorized_probabilities(
        np.asarray([1, 0, 0, 0.2]), np.asarray([0.5, 0, 1, 0.75])
    )
    np.testing.assert_allclose(result, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 0.6]])
    np.testing.assert_allclose(result.sum(1), 1)


@pytest.mark.parametrize("bad", [np.asarray([np.nan]), np.asarray([-0.1]), np.asarray([1.1])])
def test_invalid_binary_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        decode_factorized_probabilities(bad, np.asarray([0.5]))
