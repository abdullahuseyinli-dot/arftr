from __future__ import annotations

import numpy as np
import pytest

from hac.video_multiscale import derive_multiscale_features
from hac.video_token_moments import ARMS, arm_inputs, dct_basis, derive_token_moments


def sources() -> tuple[np.ndarray, ...]:
    short_v = np.zeros((2, 8, 9, 768), dtype=np.float16)
    long_v = np.zeros_like(short_v)
    short_d = np.zeros((2, 16, 1, 768), dtype=np.float16)
    long_d = np.zeros_like(short_d)
    for time in range(8):
        short_v[:, time] = time + np.arange(9, dtype=np.float16)[None, :, None]
        long_v[0, time] = 2 * time + np.arange(9, dtype=np.float16)[:, None]
    for time in range(16):
        short_d[:, time] = time
        long_d[0, time] = 2 * time
    valid = np.asarray([True, False])
    return short_v, long_v, short_d, long_d, valid, valid.copy()


def derived():
    short_v, long_v, short_d, long_d, v_valid, d_valid = sources()
    multiscale = derive_multiscale_features(
        short_v, long_v, short_d, long_d, v_valid, d_valid
    )
    return derive_token_moments(multiscale, short_v, long_v, short_d, long_d)


def test_non_dc_dct_basis_is_orthonormal() -> None:
    for times in (8, 16):
        basis = dct_basis(times)
        np.testing.assert_allclose(basis @ basis.T, np.eye(4), atol=2e-6, rtol=0)
        np.testing.assert_allclose(basis.sum(axis=1), 0, atol=2e-6, rtol=0)


def test_moments_preserve_spatial_contrast_and_exact_invalid_fallback() -> None:
    result = derived()
    spatial = result.spatial_contrast.reshape(2, 9, 768)
    np.testing.assert_allclose(spatial.sum(axis=1), 0, atol=1e-5, rtol=0)
    assert np.all(result.temporal_spectrum >= 0)
    short_v, _, short_d, _, _, _ = sources()
    replacement = derive_multiscale_features(
        short_v,
        short_v.copy(),
        short_d,
        short_d.copy(),
        np.ones(2, dtype=bool),
        np.ones(2, dtype=bool),
    )
    comparison = derive_token_moments(
        replacement, short_v, short_v.copy(), short_d, short_d.copy()
    )
    np.testing.assert_array_equal(result.spatial_contrast[1], comparison.spatial_contrast[1])
    np.testing.assert_array_equal(result.temporal_spectrum[1], comparison.temporal_spectrum[1])


def test_five_arm_dimensions() -> None:
    result = derived()
    expected = {
        "mean_factorized_refit": ("factorized", 3072, (4608,)),
        "spatial_contrast_factorized": ("factorized", 9984, (4608,)),
        "temporal_spectrum_factorized": ("factorized", 3072, (10752,)),
        "orthogonal_moments_factorized": ("factorized", 9984, (10752,)),
        "orthogonal_moments_multinomial": ("multinomial", 17664, ()),
    }
    assert set(expected) == set(ARMS)
    for arm, (kind, posture_width, motion_widths) in expected.items():
        inputs = arm_inputs(result, arm)
        assert inputs.kind == kind
        assert inputs.posture_or_direct.shape == (2, posture_width)
        assert tuple(value.shape[1] for value in inputs.motion_references) == motion_widths


def test_nonzero_invalid_sentinel_and_unknown_arm_fail_closed() -> None:
    short_v, long_v, short_d, long_d, v_valid, d_valid = sources()
    multiscale = derive_multiscale_features(
        short_v, long_v, short_d, long_d, v_valid, d_valid
    )
    long_v[1, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="exact zero sentinel"):
        derive_token_moments(multiscale, short_v, long_v, short_d, long_d)
    with pytest.raises(ValueError, match="Unknown locked"):
        arm_inputs(derived(), "posthoc")
