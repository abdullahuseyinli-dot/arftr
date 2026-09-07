"""Label-blind P2a summaries of locked frozen video and image representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARMS = (
    "video_mean_multinomial",
    "video_dino_mean_multinomial",
    "video_dino_motion_multinomial",
    "factorized_specialist",
)
SOURCE_ARMS = ("vjepa21_real_clip", "vjepa21_repeated_center", "dinov2_native_frames")


@dataclass(frozen=True)
class DerivedFeatures:
    video_mean: np.ndarray
    dino_mean: np.ndarray
    motion: np.ndarray
    validity: dict[str, np.ndarray]


def derive_features(
    real: np.ndarray,
    repeated: np.ndarray,
    dino: np.ndarray,
    validity: dict[str, np.ndarray],
    *,
    block_size: int = 64,
) -> DerivedFeatures:
    """Keep every row; compute reductions/subtractions in bounded float32 blocks.

    The residual is a representation contrast, not a calibrated physical-motion
    measurement. No labels, scenario statistics or fitted transformations enter it.
    """
    rows = len(real)
    if rows < 1 or block_size < 1:
        raise ValueError("Feature derivation requires rows and a positive block size")
    for name, value, tail in (
        (SOURCE_ARMS[0], real, (8, 9, 768)),
        (SOURCE_ARMS[1], repeated, (8, 9, 768)),
        (SOURCE_ARMS[2], dino, (16, 1, 768)),
    ):
        if value.shape != (rows, *tail) or value.dtype != np.float16:
            raise ValueError(f"Locked feature shape/dtype changed for {name}")
        mask = validity.get(name)
        if mask is None or mask.shape != (rows,) or mask.dtype != np.bool_:
            raise ValueError(f"Invalid row-validity vector for {name}")
    video_mean = np.empty((rows, 768), dtype=np.float32)
    dino_mean = np.empty_like(video_mean)
    motion = np.empty((rows, 1536), dtype=np.float32)
    for start in range(0, rows, block_size):
        end = min(rows, start + block_size)
        video_block = np.asarray(real[start:end], dtype=np.float32)
        repeat_block = np.asarray(repeated[start:end], dtype=np.float32)
        dino_block = np.asarray(dino[start:end], dtype=np.float32)
        if not all(np.isfinite(value).all() for value in (video_block, repeat_block, dino_block)):
            raise ValueError("Frozen feature block contains nonfinite values")
        video_mean[start:end] = video_block.mean(axis=(1, 2), dtype=np.float32)
        dino_mean[start:end] = dino_block.mean(axis=(1, 2), dtype=np.float32)
        motion[start:end, :768] = np.abs(video_block - repeat_block).mean(
            axis=(1, 2), dtype=np.float32
        )
        temporal = video_block.mean(axis=2, dtype=np.float32)
        motion[start:end, 768:] = np.abs(np.diff(temporal, axis=1)).mean(axis=1, dtype=np.float32)
    if not all(np.isfinite(value).all() for value in (video_mean, dino_mean, motion)):
        raise ValueError("Derived feature contains nonfinite values")
    real_valid, repeat_valid, dino_valid = (validity[name] for name in SOURCE_ARMS)
    both = real_valid & dino_valid
    all_valid = both & repeat_valid
    return DerivedFeatures(
        video_mean,
        dino_mean,
        motion,
        {
            ARMS[0]: real_valid.copy(),
            ARMS[1]: both.copy(),
            ARMS[2]: all_valid.copy(),
            ARMS[3]: all_valid.copy(),
        },
    )


def arm_features(features: DerivedFeatures, arm: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return multinomial X or separate posture/motion X, never fitted globally."""
    if arm == ARMS[0]:
        return features.video_mean, None
    if arm == ARMS[1]:
        return np.concatenate((features.video_mean, features.dino_mean), axis=1), None
    if arm == ARMS[2]:
        return np.concatenate(
            (features.video_mean, features.dino_mean, features.motion), axis=1
        ), None
    if arm == ARMS[3]:
        return (
            np.concatenate((features.video_mean, features.dino_mean), axis=1),
            np.concatenate((features.video_mean, features.motion), axis=1),
        )
    raise ValueError(f"Unknown locked P2a arm: {arm}")


def decode_factorized_probabilities(sitting: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Decode P(sitting) and P(locomotion | upright) in the original class order."""
    sitting, motion = np.asarray(sitting, dtype=np.float64), np.asarray(motion, dtype=np.float64)
    if sitting.ndim != 1 or motion.shape != sitting.shape:
        raise ValueError("Factorized probabilities must be aligned vectors")
    if any(
        not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)
        for value in (sitting, motion)
    ):
        raise ValueError("Factorized probabilities must be finite and in [0,1]")
    upright = 1 - sitting
    return np.column_stack((sitting, upright * (1 - motion), upright * motion))
