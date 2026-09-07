from __future__ import annotations

import numpy as np
import pytest

from hac.video_multiscale import ARMS, arm_features, derive_multiscale_features


def sources() -> tuple[np.ndarray, ...]:
    short_video = np.zeros((2, 8, 9, 768), dtype=np.float16)
    long_video = np.zeros_like(short_video)
    short_dino = np.zeros((2, 16, 1, 768), dtype=np.float16)
    long_dino = np.zeros_like(short_dino)
    short_video[0] = np.arange(8, dtype=np.float16)[:, None, None]
    long_video[0] = (2 * np.arange(8, dtype=np.float16))[:, None, None]
    short_dino[0] = np.arange(16, dtype=np.float16)[:, None, None]
    long_dino[0] = (3 * np.arange(16, dtype=np.float16))[:, None, None]
    short_video[1] = 3
    short_dino[1] = 5
    validity = np.asarray([True, False])
    return short_video, long_video, short_dino, long_dino, validity, validity.copy()


def test_fixed_multiscale_statistics_and_exact_short_fallback() -> None:
    features = derive_multiscale_features(*sources(), block_size=1)
    for value in features.__dict__.values():
        if isinstance(value, np.ndarray) and value.dtype != np.bool_:
            assert value.dtype == np.float32 and np.isfinite(value).all()
    np.testing.assert_allclose(features.short_v[0], 3.5)
    np.testing.assert_allclose(features.long_v[0], 7)
    np.testing.assert_allclose(features.v_scale[0], 3.5)
    np.testing.assert_allclose(features.short_motion[0], 1)
    np.testing.assert_allclose(features.long_motion[0], 2)
    np.testing.assert_allclose(features.motion_scale[0], 1)
    np.testing.assert_allclose(features.short_d[0], 7.5)
    np.testing.assert_allclose(features.long_d[0], 22.5)
    np.testing.assert_allclose(features.d_scale[0], 15)
    np.testing.assert_array_equal(features.long_v[1], features.short_v[1])
    np.testing.assert_array_equal(features.long_motion[1], features.short_motion[1])
    np.testing.assert_array_equal(features.long_d[1], features.short_d[1])
    assert not features.v_scale[1].any()
    assert not features.motion_scale[1].any()
    assert not features.d_scale[1].any()


def test_six_declared_arm_dimensions() -> None:
    features = derive_multiscale_features(*sources())
    expected = {
        "long_vjepa_mean": (768, None),
        "dual_scale_vjepa": (2304, None),
        "dual_scale_vjepa_motion": (4608, None),
        "long_dino_mean": (768, None),
        "dual_scale_vjepa_dino": (4608, None),
        "dual_scale_factorized": (3072, 4608),
    }
    assert set(expected) == set(ARMS)
    for arm, (posture_width, motion_width) in expected.items():
        posture, motion = arm_features(features, arm)
        assert posture.shape == (2, posture_width)
        assert (None if motion is None else motion.shape) == (
            None if motion_width is None else (2, motion_width)
        )


def test_mismatched_validity_and_nonzero_invalid_sentinel_fail_closed() -> None:
    values = list(sources())
    values[-1] = ~values[-1]
    with pytest.raises(ValueError, match="validity masks differ"):
        derive_multiscale_features(*values)
    values = list(sources())
    values[1][1, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="exact zero sentinel"):
        derive_multiscale_features(*values)


def test_shape_dtype_and_unknown_arm_fail_closed() -> None:
    values = list(sources())
    values[0] = values[0].astype(np.float32)
    with pytest.raises(ValueError, match="shape/dtype"):
        derive_multiscale_features(*values)
    features = derive_multiscale_features(*sources())
    with pytest.raises(ValueError, match="Unknown locked"):
        arm_features(features, "posthoc")
