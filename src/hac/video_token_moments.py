"""Orthogonal fixed space--time moments for frozen Okutama video tokens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hac.video_kinematic import ArmInputs
from hac.video_multiscale import MultiScaleFeatures
from hac.video_multiscale import arm_features as multiscale_arm_features

ARMS = (
    "mean_factorized_refit",
    "spatial_contrast_factorized",
    "temporal_spectrum_factorized",
    "orthogonal_moments_factorized",
    "orthogonal_moments_multinomial",
)


@dataclass(frozen=True)
class TokenMomentFeatures:
    visual_posture: np.ndarray
    visual_motion: np.ndarray
    visual_direct: np.ndarray
    spatial_contrast: np.ndarray
    temporal_spectrum: np.ndarray
    long_valid: np.ndarray


def dct_basis(times: int, coefficients: int = 4) -> np.ndarray:
    if times < 2 or not 1 <= coefficients < times:
        raise ValueError("DCT moment dimensions are invalid")
    t = np.arange(times, dtype=np.float32)
    k = np.arange(1, coefficients + 1, dtype=np.float32)[:, None]
    return (
        np.sqrt(np.float32(2.0 / times))
        * np.cos(np.pi * (t[None, :] + np.float32(0.5)) * k / np.float32(times))
    ).astype(np.float32)


def _check(
    values: np.ndarray, rows: int, tail: tuple[int, ...], name: str
) -> None:
    if values.shape != (rows, *tail) or values.dtype != np.float16:
        raise ValueError(f"Locked P5 source shape/dtype changed: {name}")


def derive_token_moments(
    multiscale: MultiScaleFeatures,
    short_video: np.ndarray,
    long_video: np.ndarray,
    short_dino: np.ndarray,
    long_dino: np.ndarray,
    *,
    block_size: int = 32,
) -> TokenMomentFeatures:
    """Build fixed spatial contrasts and non-DC DCT magnitudes without labels."""
    rows = len(multiscale.long_valid)
    if rows < 1 or block_size < 1:
        raise ValueError("P5 moment derivation requires rows and a positive block size")
    for values, tail, name in (
        (short_video, (8, 9, 768), "short_video"),
        (long_video, (8, 9, 768), "long_video"),
        (short_dino, (16, 1, 768), "short_dino"),
        (long_dino, (16, 1, 768), "long_dino"),
    ):
        _check(values, rows, tail, name)
    if multiscale.long_valid.shape != (rows,) or multiscale.long_valid.dtype != np.bool_:
        raise ValueError("P5 long-validity contract changed")

    visual_posture, visual_motion = multiscale_arm_features(
        multiscale, "dual_scale_factorized"
    )
    visual_direct, unexpected = multiscale_arm_features(
        multiscale, "dual_scale_vjepa_dino"
    )
    if visual_motion is None or unexpected is not None:
        raise RuntimeError("P3 feature API no longer matches P5")
    spatial = np.empty((rows, 9 * 768), dtype=np.float32)
    spectrum = np.empty((rows, 8 * 768), dtype=np.float32)
    v_basis, d_basis = dct_basis(8), dct_basis(16)
    for start in range(0, rows, block_size):
        end = min(rows, start + block_size)
        valid = multiscale.long_valid[start:end]
        short_v = np.asarray(short_video[start:end], dtype=np.float32)
        short_d = np.asarray(short_dino[start:end], dtype=np.float32)
        long_v_raw = np.asarray(long_video[start:end], dtype=np.float32)
        long_d_raw = np.asarray(long_dino[start:end], dtype=np.float32)
        if not np.isfinite(short_v).all() or not np.isfinite(short_d).all():
            raise ValueError("P5 short anchor contains nonfinite values")
        if valid.any() and (
            not np.isfinite(long_v_raw[valid]).all()
            or not np.isfinite(long_d_raw[valid]).all()
        ):
            raise ValueError("P5 valid long token contains nonfinite values")
        if (~valid).any() and (
            np.any(long_v_raw[~valid] != 0) or np.any(long_d_raw[~valid] != 0)
        ):
            raise ValueError("P5 invalid long token is not its exact zero sentinel")
        long_v = np.where(valid[:, None, None, None], long_v_raw, short_v)
        long_d = np.where(valid[:, None, None, None], long_d_raw, short_d)
        region = long_v.mean(axis=1, dtype=np.float32)
        contrast = region - region.mean(axis=1, keepdims=True, dtype=np.float32)
        spatial[start:end] = contrast.reshape(end - start, -1)
        v_time = long_v.mean(axis=2, dtype=np.float32)
        d_time = long_d[:, :, 0]
        v_coefficients = np.abs(np.einsum("kt,btd->bkd", v_basis, v_time))
        d_coefficients = np.abs(np.einsum("kt,btd->bkd", d_basis, d_time))
        spectrum[start:end] = np.concatenate(
            (v_coefficients, d_coefficients), axis=1
        ).reshape(end - start, -1)
    values = (visual_posture, visual_motion, visual_direct, spatial, spectrum)
    expected = (3072, 4608, 4608, 6912, 6144)
    for value, width in zip(values, expected, strict=True):
        if value.shape != (rows, width) or value.dtype != np.float32:
            raise RuntimeError("Derived P5 feature shape/dtype changed")
        if not np.isfinite(value).all():
            raise ValueError("Derived P5 feature contains nonfinite values")
    return TokenMomentFeatures(*values, multiscale.long_valid.copy())


def arm_inputs(features: TokenMomentFeatures, arm: str) -> ArmInputs:
    vp, vm, direct = (
        features.visual_posture,
        features.visual_motion,
        features.visual_direct,
    )
    spatial, spectrum = features.spatial_contrast, features.temporal_spectrum
    if arm == "mean_factorized_refit":
        return ArmInputs("factorized", vp, (vm,))
    if arm == "spatial_contrast_factorized":
        return ArmInputs(
            "factorized", np.concatenate((vp, spatial), axis=1), (vm,)
        )
    if arm == "temporal_spectrum_factorized":
        return ArmInputs(
            "factorized", vp, (np.concatenate((vm, spectrum), axis=1),)
        )
    if arm == "orthogonal_moments_factorized":
        return ArmInputs(
            "factorized",
            np.concatenate((vp, spatial), axis=1),
            (np.concatenate((vm, spectrum), axis=1),),
        )
    if arm == "orthogonal_moments_multinomial":
        return ArmInputs(
            "multinomial", np.concatenate((direct, spatial, spectrum), axis=1), ()
        )
    raise ValueError(f"Unknown locked P5 arm: {arm}")
