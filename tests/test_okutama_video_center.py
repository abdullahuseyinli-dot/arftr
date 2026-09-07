from __future__ import annotations

import numpy as np
import pytest

from hac.video_center import ARMS, EXPECTED_WIDTHS, arm_inputs, derive_center_aware_features
from hac.video_token_moments import TokenMomentFeatures
from hac.video_token_moments import arm_inputs as moment_arm_inputs


def moments(valid: np.ndarray) -> TokenMomentFeatures:
    rows = len(valid)
    return TokenMomentFeatures(
        np.zeros((rows, 3072), np.float32),
        np.zeros((rows, 4608), np.float32),
        np.zeros((rows, 4608), np.float32),
        np.zeros((rows, 6912), np.float32),
        np.zeros((rows, 6144), np.float32),
        valid.copy(),
    )


def sources(rows: int = 2) -> tuple[np.ndarray, ...]:
    return (
        np.zeros((rows, 8, 9, 768), np.float16),
        np.zeros((rows, 8, 9, 768), np.float16),
        np.zeros((rows, 16, 1, 768), np.float16),
        np.zeros((rows, 16, 1, 768), np.float16),
    )


def test_physical_time_ramps_and_fallback_use_declared_stride() -> None:
    valid = np.asarray([True, False])
    short_v, long_v, short_d, long_d = sources()
    for slot in range(16):
        long_d[0, slot, 0] = np.float16(4 * (slot - 8) / 30)
        short_d[1, slot, 0] = np.float16((slot - 8) / 30)
    derived = derive_center_aware_features(
        moments(valid), short_v, long_v, short_d, long_d, block_size=1
    )
    slope = derived.center_signed_change[:, :768]
    curvature = derived.center_signed_change[:, 768:]
    np.testing.assert_allclose(slope, 1, rtol=0, atol=0.002)
    np.testing.assert_allclose(curvature, 0, rtol=0, atol=0.02)


def test_quadratic_ramp_uses_declared_curvature_convention() -> None:
    valid = np.asarray([True, False])
    short_v, long_v, short_d, long_d = sources()
    for slot in range(16):
        long_d[0, slot, 0] = np.float16((4 * (slot - 8) / 30) ** 2)
        short_d[1, slot, 0] = np.float16(((slot - 8) / 30) ** 2)
    derived = derive_center_aware_features(
        moments(valid), short_v, long_v, short_d, long_d
    )
    np.testing.assert_allclose(derived.center_signed_change[:, :768], 0, atol=0.002)
    np.testing.assert_allclose(
        derived.center_signed_change[:, 768:], -1, rtol=0, atol=0.003
    )


def test_center_offcenter_and_unsigned_controls_are_distinct_and_exact() -> None:
    valid = np.asarray([True])
    short_v, long_v, short_d, long_d = sources(1)
    long_d[0, 8, 0, 0] = 3
    long_d[0, 7, 0, 0] = -1
    long_d[0, 9, 0, 0] = 2
    long_d[0, 4, 0, 0] = 7
    long_v[0, 4, :, 0] = 5
    long_v[0, 2, :, 0] = -2
    derived = derive_center_aware_features(moments(valid), short_v, long_v, short_d, long_d)
    assert not np.array_equal(derived.center_anchor, derived.offcenter_anchor)
    assert not np.array_equal(
        derived.center_signed_change, derived.offcenter_signed_change
    )
    np.testing.assert_array_equal(
        derived.center_unsigned_change, np.abs(derived.center_signed_change)
    )


def test_constant_offsets_leave_contrasts_and_changes_unchanged() -> None:
    valid = np.asarray([True])
    original = list(sources(1))
    rng = np.random.default_rng(7)
    original[1][...] = (rng.integers(-8, 9, original[1].shape) / 16).astype(np.float16)
    original[3][...] = (rng.integers(-8, 9, original[3].shape) / 16).astype(np.float16)
    shifted = [value.copy() for value in original]
    shifted[1] += np.float16(2)
    shifted[3] += np.float16(2)
    left = derive_center_aware_features(moments(valid), *original)
    right = derive_center_aware_features(moments(valid), *shifted)
    for name in (
        "center_anchor",
        "center_signed_change",
        "offcenter_anchor",
        "offcenter_signed_change",
    ):
        np.testing.assert_allclose(getattr(left, name), getattr(right, name), rtol=0, atol=1e-6)


def test_blocking_is_invariant_and_sources_are_not_mutated() -> None:
    valid = np.asarray([True, False, True])
    values = list(sources(3))
    rng = np.random.default_rng(17)
    values[0][...] = rng.normal(0, 0.1, values[0].shape).astype(np.float16)
    values[2][...] = rng.normal(0, 0.1, values[2].shape).astype(np.float16)
    values[1][valid] = rng.normal(0, 0.1, values[1][valid].shape).astype(np.float16)
    values[3][valid] = rng.normal(0, 0.1, values[3][valid].shape).astype(np.float16)
    retained = [value.copy() for value in values]
    one = derive_center_aware_features(moments(valid), *values, block_size=1)
    all_rows = derive_center_aware_features(moments(valid), *values, block_size=3)
    for name in (
        "center_anchor",
        "center_signed_change",
        "offcenter_anchor",
        "offcenter_signed_change",
        "center_unsigned_change",
    ):
        np.testing.assert_array_equal(getattr(one, name), getattr(all_rows, name))
    for actual, expected in zip(values, retained, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_arm_dimensions_a0_identity_unsigned_scope_and_direct_union_order() -> None:
    valid = np.asarray([True, True])
    base = moments(valid)
    for index, value in enumerate(
        (
            base.visual_posture,
            base.visual_motion,
            base.spatial_contrast,
        ),
        start=1,
    ):
        value[:] = index
    values = list(sources(2))
    derived = derive_center_aware_features(base, *values)
    derived.center_anchor[:] = 4
    derived.center_signed_change[:] = np.tile(
        np.concatenate((np.full(768, -5), np.full(768, 6))), (2, 1)
    )
    derived.center_unsigned_change[:] = np.abs(derived.center_signed_change)
    refit = arm_inputs(derived, ARMS[0])
    expected = moment_arm_inputs(base, "spatial_contrast_factorized")
    np.testing.assert_array_equal(refit.posture_or_direct, expected.posture_or_direct)
    np.testing.assert_array_equal(refit.motion_references[0], expected.motion_references[0])
    signed, unsigned = arm_inputs(derived, ARMS[2]), arm_inputs(derived, ARMS[4])
    np.testing.assert_array_equal(signed.posture_or_direct, unsigned.posture_or_direct)
    np.testing.assert_array_equal(
        unsigned.motion_references[0][:, :-1536], signed.motion_references[0][:, :-1536]
    )
    np.testing.assert_array_equal(
        unsigned.motion_references[0][:, -1536:], np.abs(signed.motion_references[0][:, -1536:])
    )
    direct = arm_inputs(derived, ARMS[5]).posture_or_direct
    blocks = (3072, 3072, 6912, 1536, 1536)
    starts = np.cumsum((0, *blocks))
    for start, end, expected_value in zip(starts[:-1], starts[1:], (1, 2, 3, 4, None), strict=True):
        if expected_value is not None:
            np.testing.assert_array_equal(direct[:, start:end], expected_value)
    np.testing.assert_array_equal(direct[:, starts[-2] :], derived.center_signed_change)
    for arm, (posture_width, motion_width) in EXPECTED_WIDTHS.items():
        inputs = arm_inputs(derived, arm)
        assert inputs.posture_or_direct.shape == (2, posture_width)
        assert (inputs.motion_references[0].shape[1] if motion_width else None) == motion_width


def test_invalid_long_sentinel_fails_closed() -> None:
    valid = np.asarray([False])
    values = list(sources(1))
    values[1][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="zero sentinel"):
        derive_center_aware_features(moments(valid), *values)
