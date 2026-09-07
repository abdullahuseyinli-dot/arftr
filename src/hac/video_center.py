"""Fixed center-aware frozen-token features for the Okutama P7 trial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hac.video_kinematic import ArmInputs
from hac.video_token_moments import TokenMomentFeatures
from hac.video_token_moments import arm_inputs as moment_arm_inputs

ARMS = (
    "spatial_refit",
    "center_posture_factorized",
    "center_signed_factorized",
    "offcenter_signed_factorized",
    "center_unsigned_factorized",
    "center_signed_multinomial",
)

EXPECTED_WIDTHS = {
    ARMS[0]: (9984, 4608),
    ARMS[1]: (11520, 4608),
    ARMS[2]: (11520, 6144),
    ARMS[3]: (11520, 6144),
    ARMS[4]: (11520, 6144),
    ARMS[5]: (16128, None),
}


@dataclass(frozen=True)
class CenterAwareFeatures:
    moments: TokenMomentFeatures
    center_anchor: np.ndarray
    center_signed_change: np.ndarray
    offcenter_anchor: np.ndarray
    offcenter_signed_change: np.ndarray
    center_unsigned_change: np.ndarray
    long_valid: np.ndarray


def _check_source(values: np.ndarray, rows: int, tail: tuple[int, ...], name: str) -> None:
    if values.shape != (rows, *tail) or values.dtype != np.float16:
        raise ValueError(f"Locked P7 source shape/dtype changed: {name}")


def _anchor(
    dino: np.ndarray,
    video: np.ndarray,
    dino_mean: np.ndarray,
    video_mean: np.ndarray,
    *,
    frame: int,
    tubelet: int,
) -> np.ndarray:
    return np.concatenate(
        (
            dino[:, frame] - dino_mean,
            video[:, tubelet].mean(axis=1, dtype=np.float32) - video_mean,
        ),
        axis=1,
    )


def _signed_change(
    dino: np.ndarray,
    h: np.ndarray,
    *,
    frame: int,
) -> np.ndarray:
    before, center, after = dino[:, frame - 1], dino[:, frame], dino[:, frame + 1]
    slope = (after - before) / (np.float32(2.0) * h[:, None])
    curvature = (
        center - np.float32(0.5) * (before + after)
    ) / (h[:, None] * h[:, None])
    return np.concatenate((slope, curvature), axis=1)


def derive_center_aware_features(
    moments: TokenMomentFeatures,
    short_video: np.ndarray,
    long_video: np.ndarray,
    short_dino: np.ndarray,
    long_dino: np.ndarray,
    *,
    block_size: int = 32,
) -> CenterAwareFeatures:
    """Derive fixed target-time anchors and local finite differences without labels."""
    rows = len(moments.long_valid)
    if rows < 1 or block_size < 1:
        raise ValueError("P7 center feature derivation requires rows and a positive block size")
    for values, tail, name in (
        (short_video, (8, 9, 768), "short_video"),
        (long_video, (8, 9, 768), "long_video"),
        (short_dino, (16, 1, 768), "short_dino"),
        (long_dino, (16, 1, 768), "long_dino"),
    ):
        _check_source(values, rows, tail, name)
    valid = moments.long_valid
    if valid.shape != (rows,) or valid.dtype != np.bool_:
        raise ValueError("P7 long-validity contract changed")
    outputs = [np.empty((rows, 1536), dtype=np.float32) for _ in range(4)]
    center_anchor, center_signed, offcenter_anchor, offcenter_signed = outputs
    for start in range(0, rows, block_size):
        end = min(rows, start + block_size)
        mask = valid[start:end]
        sv = np.asarray(short_video[start:end], dtype=np.float32)
        sd = np.asarray(short_dino[start:end], dtype=np.float32)
        lv_raw = np.asarray(long_video[start:end], dtype=np.float32)
        ld_raw = np.asarray(long_dino[start:end], dtype=np.float32)
        if not np.isfinite(sv).all() or not np.isfinite(sd).all():
            raise ValueError("P7 short anchor contains nonfinite values")
        if mask.any() and (
            not np.isfinite(lv_raw[mask]).all() or not np.isfinite(ld_raw[mask]).all()
        ):
            raise ValueError("P7 valid long token contains nonfinite values")
        if (~mask).any() and (
            np.any(lv_raw[~mask] != 0) or np.any(ld_raw[~mask] != 0)
        ):
            raise ValueError("P7 invalid long token is not its exact zero sentinel")
        video = np.where(mask[:, None, None, None], lv_raw, sv)
        dino = np.where(mask[:, None, None, None], ld_raw, sd)[:, :, 0]
        dino_mean = dino.mean(axis=1, dtype=np.float32)
        video_mean = video.mean(axis=(1, 2), dtype=np.float32)
        h = np.where(mask, np.float32(4.0 / 30.0), np.float32(1.0 / 30.0)).astype(
            np.float32
        )
        center_anchor[start:end] = _anchor(
            dino, video, dino_mean, video_mean, frame=8, tubelet=4
        )
        center_signed[start:end] = _signed_change(dino, h, frame=8)
        offcenter_anchor[start:end] = _anchor(
            dino, video, dino_mean, video_mean, frame=4, tubelet=2
        )
        offcenter_signed[start:end] = _signed_change(dino, h, frame=4)
    for value in outputs:
        if value.shape != (rows, 1536) or value.dtype != np.float32:
            raise RuntimeError("Derived P7 feature shape/dtype changed")
        if not np.isfinite(value).all():
            raise ValueError("Derived P7 feature contains nonfinite values")
    return CenterAwareFeatures(
        moments=moments,
        center_anchor=center_anchor,
        center_signed_change=center_signed,
        offcenter_anchor=offcenter_anchor,
        offcenter_signed_change=offcenter_signed,
        center_unsigned_change=np.abs(center_signed),
        long_valid=valid.copy(),
    )


def arm_inputs(features: CenterAwareFeatures, arm: str) -> ArmInputs:
    """Assemble one locked P7 expert without retaining copies for every arm."""
    if arm == ARMS[5]:
        moments = features.moments
        direct = np.concatenate(
            (
                moments.visual_posture,
                moments.visual_motion[:, 1536:],
                moments.spatial_contrast,
                features.center_anchor,
                features.center_signed_change,
            ),
            axis=1,
        )
        result = ArmInputs("multinomial", direct, ())
    else:
        base = moment_arm_inputs(features.moments, "spatial_contrast_factorized")
        posture, motion = base.posture_or_direct, base.motion_references[0]
        if arm == ARMS[0]:
            result = base
        elif arm == ARMS[1]:
            result = ArmInputs(
                "factorized",
                np.concatenate((posture, features.center_anchor), axis=1),
                (motion,),
            )
        elif arm == ARMS[2]:
            result = ArmInputs(
                "factorized",
                np.concatenate((posture, features.center_anchor), axis=1),
                (np.concatenate((motion, features.center_signed_change), axis=1),),
            )
        elif arm == ARMS[3]:
            result = ArmInputs(
                "factorized",
                np.concatenate((posture, features.offcenter_anchor), axis=1),
                (np.concatenate((motion, features.offcenter_signed_change), axis=1),),
            )
        elif arm == ARMS[4]:
            result = ArmInputs(
                "factorized",
                np.concatenate((posture, features.center_anchor), axis=1),
                (np.concatenate((motion, features.center_unsigned_change), axis=1),),
            )
        else:
            raise ValueError(f"Unknown locked P7 arm: {arm}")
    expected_posture, expected_motion = EXPECTED_WIDTHS[arm]
    if result.posture_or_direct.shape != (len(features.long_valid), expected_posture):
        raise RuntimeError("P7 posture/direct feature width changed")
    if expected_motion is None:
        if result.motion_references:
            raise RuntimeError("P7 direct arm unexpectedly has a motion head")
    elif (
        len(result.motion_references) != 1
        or result.motion_references[0].shape != (len(features.long_valid), expected_motion)
    ):
        raise RuntimeError("P7 motion feature width changed")
    return result
