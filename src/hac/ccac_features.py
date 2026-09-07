"""Fixed clip-level feature blocks for the full Okutama CCAC extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hac.video_kinematic import ArmInputs
from hac.video_token_moments import TokenMomentFeatures
from hac.video_token_moments import arm_inputs as moment_arm_inputs

QUANTILES = (0.1, 0.5, 0.9)
QUALITY_COLUMNS = (
    "available_pair_fraction",
    "camera_usable_pair_fraction",
    "translation_usable_pair_fraction",
    "articulation_usable_pair_fraction",
    "translation_feature_valid",
    "articulation_feature_valid",
    "median_actor_seed_fraction_of_32",
    "median_actor_retained_fraction_of_32",
    "median_actor_vertical_region_fraction_of_3",
    "median_actor_forward_backward_error_pixels",
    "actor_forward_backward_observed",
    "median_camera_audit_error_pixels",
    "camera_audit_observed",
    "center_seeded_survival_fraction_75pct",
    "center_seed_fraction_of_32",
    "native_center_height_fraction_of_720",
)
TRANSLATION_CHANNELS = ("x_height_per_second", "y_height_per_second", "norm_height_per_second")
RAW_TRANSLATION_COLUMNS = tuple(
    f"raw_{channel}_q{int(100 * quantile):02d}"
    for channel in TRANSLATION_CHANNELS
    for quantile in QUANTILES
)
COMPENSATED_TRANSLATION_COLUMNS = tuple(
    f"compensated_{channel}_q{int(100 * quantile):02d}"
    for channel in TRANSLATION_CHANNELS
    for quantile in QUANTILES
)
WITHIN_ACTOR_CHANNELS = ("median_height_per_second", "p90_height_per_second")
WITHIN_ACTOR_COLUMNS = tuple(
    f"within_actor_{channel}_q{int(100 * quantile):02d}"
    for channel in WITHIN_ACTOR_CHANNELS
    for quantile in QUANTILES
)
ARMS = (
    "spatial_refit",
    "ccac_reliability",
    "ccac_raw_translation",
    "ccac_compensated_translation",
    "ccac_full",
)
EXPECTED_WIDTHS = {
    "spatial_refit": (9984, 4608),
    "ccac_reliability": (9984, 4624),
    "ccac_raw_translation": (9984, 4633),
    "ccac_compensated_translation": (9984, 4633),
    "ccac_full": (9984, 4639),
}


@dataclass(frozen=True)
class CCACFeatures:
    quality: np.ndarray
    raw_translation: np.ndarray
    compensated_translation: np.ndarray
    within_actor: np.ndarray
    translation_valid: np.ndarray
    articulation_valid: np.ndarray


def _finite_median(values: list[Any]) -> tuple[float, bool]:
    retained = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )
    return (float(np.median(retained)), True) if len(retained) else (0.0, False)


def _quantile_channels(channels: list[np.ndarray], *, valid: bool) -> np.ndarray:
    if not valid:
        return np.zeros(sum(len(QUANTILES) for _ in channels), dtype=np.float32)
    values = np.concatenate(
        [np.quantile(channel, QUANTILES, method="linear") for channel in channels]
    )
    if not np.isfinite(values).all():
        raise ValueError("CCAC temporal feature reduction produced a nonfinite value")
    return values.astype(np.float32)


def aggregate_clip_features(
    pairs: list[dict[str, Any]], clip: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Reduce all 15 requested pair rows without treating failures as observations."""
    if len(pairs) != 15 or [int(row["pair_index"]) for row in pairs] != list(range(15)):
        raise ValueError("CCAC feature aggregation requires all 15 ordered pair rows")
    sample_id = str(clip["sample_id"])
    if any(str(row["sample_id"]) != sample_id for row in pairs):
        raise ValueError("CCAC pair and clip identities are not aligned")
    available = [row for row in pairs if bool(row["pair_available"])]
    translation = [row for row in pairs if bool(row["translation_usable"])]
    articulation = [row for row in pairs if bool(row["articulation_usable"])]
    translation_valid = len(translation) >= 10
    articulation_valid = len(articulation) >= 10
    actor_fb, actor_fb_observed = _finite_median(
        [row.get("actor_primary_fb_median_pixels") for row in available]
    )
    camera_audit, camera_audit_observed = _finite_median(
        [row.get("camera_audit_median_pixels") for row in available]
    )
    median_seed, _ = _finite_median([row.get("actor_seed_points") for row in available])
    median_retained, _ = _finite_median(
        [row.get("actor_primary_retained_points") for row in available]
    )
    median_regions, _ = _finite_median([row.get("actor_vertical_regions") for row in available])
    quality = np.asarray(
        (
            len(available) / 15,
            sum(bool(row["camera_usable"]) for row in pairs) / 15,
            len(translation) / 15,
            len(articulation) / 15,
            float(translation_valid),
            float(articulation_valid),
            median_seed / 32,
            median_retained / 32,
            median_regions / 3,
            actor_fb,
            float(actor_fb_observed),
            camera_audit,
            float(camera_audit_observed),
            float(clip["center_track_survival_fraction_75pct"]),
            float(clip["center_seed_points"]) / 32,
            float(clip["native_center_height"]) / 720,
        ),
        dtype=np.float32,
    )
    if quality.shape != (16,) or not np.isfinite(quality).all():
        raise RuntimeError("CCAC quality block contract changed")

    raw_x = np.asarray([float(row["raw_translation_x_height_per_second"]) for row in translation])
    raw_y = np.asarray([float(row["raw_translation_y_height_per_second"]) for row in translation])
    compensated_x = np.asarray(
        [float(row["compensated_translation_x_height_per_second"]) for row in translation]
    )
    compensated_y = np.asarray(
        [float(row["compensated_translation_y_height_per_second"]) for row in translation]
    )
    raw = _quantile_channels([raw_x, raw_y, np.hypot(raw_x, raw_y)], valid=translation_valid)
    compensated = _quantile_channels(
        [
            compensated_x,
            compensated_y,
            np.hypot(compensated_x, compensated_y),
        ],
        valid=translation_valid,
    )
    within_median = np.asarray(
        [float(row["articulation_median_height_per_second"]) for row in articulation]
    )
    within_p90 = np.asarray(
        [float(row["articulation_p90_height_per_second"]) for row in articulation]
    )
    within = _quantile_channels([within_median, within_p90], valid=articulation_valid)
    if raw.shape != (9,) or compensated.shape != (9,) or within.shape != (6,):
        raise RuntimeError("CCAC motion-block width changed")
    return quality, raw, compensated, within, translation_valid, articulation_valid


def validate_feature_blocks(features: CCACFeatures, rows: int) -> None:
    expected = (
        (features.quality, (rows, 16), np.float32),
        (features.raw_translation, (rows, 9), np.float32),
        (features.compensated_translation, (rows, 9), np.float32),
        (features.within_actor, (rows, 6), np.float32),
        (features.translation_valid, (rows,), np.bool_),
        (features.articulation_valid, (rows,), np.bool_),
    )
    for values, shape, dtype in expected:
        if values.shape != shape or values.dtype != dtype:
            raise RuntimeError("CCAC feature array shape or dtype changed")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(values).all():
            raise ValueError("CCAC feature array contains a nonfinite value")
    if np.any(features.raw_translation[~features.translation_valid] != 0):
        raise RuntimeError("Invalid CCAC raw-translation rows must use the zero sentinel")
    if np.any(features.compensated_translation[~features.translation_valid] != 0):
        raise RuntimeError("Invalid CCAC compensated rows must use the zero sentinel")
    if np.any(features.within_actor[~features.articulation_valid] != 0):
        raise RuntimeError("Invalid CCAC within-actor rows must use the zero sentinel")


def arm_inputs(base: TokenMomentFeatures, ccac: CCACFeatures, arm: str) -> ArmInputs:
    """Append only the declared CCAC block to the established P5 motion head."""
    validate_feature_blocks(ccac, len(base.long_valid))
    spatial = moment_arm_inputs(base, "spatial_contrast_factorized")
    posture, motion = spatial.posture_or_direct, spatial.motion_references[0]
    additions = {
        "ccac_reliability": (ccac.quality,),
        "ccac_raw_translation": (ccac.quality, ccac.raw_translation),
        "ccac_compensated_translation": (
            ccac.quality,
            ccac.compensated_translation,
        ),
        "ccac_full": (
            ccac.quality,
            ccac.compensated_translation,
            ccac.within_actor,
        ),
    }
    if arm == "spatial_refit":
        result = spatial
    elif arm in additions:
        result = ArmInputs(
            "factorized", posture, (np.concatenate((motion, *additions[arm]), axis=1),)
        )
    else:
        raise ValueError(f"Unknown CCAC arm: {arm}")
    expected_posture, expected_motion = EXPECTED_WIDTHS[arm]
    if result.posture_or_direct.shape != (len(base.long_valid), expected_posture):
        raise RuntimeError("CCAC posture width changed")
    if len(result.motion_references) != 1 or result.motion_references[0].shape != (
        len(base.long_valid),
        expected_motion,
    ):
        raise RuntimeError("CCAC motion width changed")
    return result
