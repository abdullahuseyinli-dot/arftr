"""Label-blind fixed summaries for the locked short/long Okutama representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARMS = (
    "long_vjepa_mean",
    "dual_scale_vjepa",
    "dual_scale_vjepa_motion",
    "long_dino_mean",
    "dual_scale_vjepa_dino",
    "dual_scale_factorized",
)


@dataclass(frozen=True)
class MultiScaleFeatures:
    short_v: np.ndarray
    long_v: np.ndarray
    v_scale: np.ndarray
    short_motion: np.ndarray
    long_motion: np.ndarray
    motion_scale: np.ndarray
    short_d: np.ndarray
    long_d: np.ndarray
    d_scale: np.ndarray
    long_valid: np.ndarray


def _check_source(value: np.ndarray, rows: int, tail: tuple[int, ...], name: str) -> None:
    if value.shape != (rows, *tail) or value.dtype != np.float16:
        raise ValueError(f"Locked shape/dtype changed for {name}")


def derive_multiscale_features(
    short_video: np.ndarray,
    long_video: np.ndarray,
    short_dino: np.ndarray,
    long_dino: np.ndarray,
    long_video_valid: np.ndarray,
    long_dino_valid: np.ndarray,
    *,
    block_size: int = 64,
) -> MultiScaleFeatures:
    """Build fixed dual-scale statistics with exact short fallback, never labels."""
    rows = len(short_video)
    if rows < 1 or block_size < 1:
        raise ValueError("Multiscale derivation needs rows and a positive block size")
    for values, tail, name in (
        (short_video, (8, 9, 768), "short_video"),
        (long_video, (8, 9, 768), "long_video"),
        (short_dino, (16, 1, 768), "short_dino"),
        (long_dino, (16, 1, 768), "long_dino"),
    ):
        _check_source(values, rows, tail, name)
    for name, valid in (
        ("long_video_valid", long_video_valid),
        ("long_dino_valid", long_dino_valid),
    ):
        if valid.shape != (rows,) or valid.dtype != np.bool_:
            raise ValueError(f"Invalid P3 row-validity vector: {name}")
    if not np.array_equal(long_video_valid, long_dino_valid):
        raise ValueError("P3 encoder validity masks differ")
    long_valid = long_video_valid.copy()
    outputs = [np.empty((rows, 768), np.float32) for _ in range(9)]
    (
        short_v,
        long_v,
        v_scale,
        short_motion,
        long_motion,
        motion_scale,
        short_d,
        long_d,
        d_scale,
    ) = outputs
    for start in range(0, rows, block_size):
        end = min(rows, start + block_size)
        mask = long_valid[start:end]
        sv_raw = np.asarray(short_video[start:end], dtype=np.float32)
        sd_raw = np.asarray(short_dino[start:end], dtype=np.float32)
        lv_raw = np.asarray(long_video[start:end], dtype=np.float32)
        ld_raw = np.asarray(long_dino[start:end], dtype=np.float32)
        if not np.isfinite(sv_raw).all() or not np.isfinite(sd_raw).all():
            raise ValueError("Short anchor contains nonfinite features")
        if mask.any() and (
            not np.isfinite(lv_raw[mask]).all() or not np.isfinite(ld_raw[mask]).all()
        ):
            raise ValueError("Valid long representation contains nonfinite features")
        if (~mask).any() and (np.any(lv_raw[~mask] != 0) or np.any(ld_raw[~mask] != 0)):
            raise ValueError("Invalid long representation is not its exact zero sentinel")
        sv = sv_raw.mean(axis=(1, 2), dtype=np.float32)
        sd = sd_raw.mean(axis=(1, 2), dtype=np.float32)
        sm = np.abs(np.diff(sv_raw.mean(axis=2, dtype=np.float32), axis=1)).mean(
            axis=1, dtype=np.float32
        )
        lv_actual = lv_raw.mean(axis=(1, 2), dtype=np.float32)
        ld_actual = ld_raw.mean(axis=(1, 2), dtype=np.float32)
        lm_actual = np.abs(np.diff(lv_raw.mean(axis=2, dtype=np.float32), axis=1)).mean(
            axis=1, dtype=np.float32
        )
        lv = np.where(mask[:, None], lv_actual, sv)
        ld = np.where(mask[:, None], ld_actual, sd)
        lm = np.where(mask[:, None], lm_actual, sm)
        short_v[start:end], long_v[start:end] = sv, lv
        short_d[start:end], long_d[start:end] = sd, ld
        short_motion[start:end], long_motion[start:end] = sm, lm
        v_scale[start:end] = np.where(mask[:, None], np.abs(lv_actual - sv), 0)
        d_scale[start:end] = np.where(mask[:, None], np.abs(ld_actual - sd), 0)
        motion_scale[start:end] = np.where(mask[:, None], np.abs(lm_actual - sm), 0)
    if not all(np.isfinite(value).all() for value in outputs):
        raise ValueError("Derived P3 multiscale feature contains nonfinite values")
    return MultiScaleFeatures(*outputs, long_valid)


def arm_features(
    features: MultiScaleFeatures, arm: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return a direct feature matrix or independent posture/motion matrices."""
    if arm == "long_vjepa_mean":
        return features.long_v, None
    if arm == "dual_scale_vjepa":
        return np.concatenate((features.short_v, features.long_v, features.v_scale), axis=1), None
    if arm == "dual_scale_vjepa_motion":
        return np.concatenate(
            (
                features.short_v,
                features.long_v,
                features.v_scale,
                features.short_motion,
                features.long_motion,
                features.motion_scale,
            ),
            axis=1,
        ), None
    if arm == "long_dino_mean":
        return features.long_d, None
    if arm == "dual_scale_vjepa_dino":
        return np.concatenate(
            (
                features.short_v,
                features.long_v,
                features.v_scale,
                features.short_d,
                features.long_d,
                features.d_scale,
            ),
            axis=1,
        ), None
    if arm == "dual_scale_factorized":
        posture = np.concatenate(
            (features.short_v, features.long_v, features.short_d, features.long_d), axis=1
        )
        motion = np.concatenate(
            (
                features.short_v,
                features.long_v,
                features.v_scale,
                features.short_motion,
                features.long_motion,
                features.d_scale,
            ),
            axis=1,
        )
        return posture, motion
    raise ValueError(f"Unknown locked P3 arm: {arm}")
