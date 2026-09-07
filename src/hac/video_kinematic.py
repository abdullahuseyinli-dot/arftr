"""Fixed visual--kinematic feature recipes for the declared Okutama P4 trial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hac.video_multiscale import MultiScaleFeatures
from hac.video_multiscale import arm_features as multiscale_arm_features

ARMS = (
    "visual_factorized_refit",
    "visual_context_factorized",
    "visual_raw_track_factorized",
    "visual_comp_track_factorized",
    "visual_dual_track_factorized",
    "camera_reference_marginalized",
    "visual_dual_track_multinomial",
)


@dataclass(frozen=True)
class KinematicFeatures:
    visual_posture: np.ndarray
    visual_motion: np.ndarray
    visual_direct: np.ndarray
    context_geometry: np.ndarray
    raw_track: np.ndarray
    compensated_track: np.ndarray
    reference_disagreement: np.ndarray
    dual_track: np.ndarray
    camera_quality_mean: np.ndarray


@dataclass(frozen=True)
class ArmInputs:
    kind: str
    posture_or_direct: np.ndarray
    motion_references: tuple[np.ndarray, ...]


def _source(
    values: np.ndarray, rows: int, tail: tuple[int, ...], name: str
) -> np.ndarray:
    if values.shape != (rows, *tail) or values.dtype != np.float32:
        raise ValueError(f"Locked P4 shape/dtype changed for {name}")
    if not np.isfinite(values).all():
        raise ValueError(f"Locked P4 source contains nonfinite values: {name}")
    return np.asarray(values, dtype=np.float32)


def derive_kinematic_features(
    visual: MultiScaleFeatures,
    legacy_static_features: np.ndarray,
    legacy_quality_features: np.ndarray,
    raw_sequence: np.ndarray,
    raw_summary: np.ndarray,
    compensated_sequence: np.ndarray,
    compensated_summary: np.ndarray,
    camera_quality: np.ndarray,
) -> KinematicFeatures:
    """Apply the label-blind, parameter-free P4 concatenation recipe."""
    rows = len(visual.long_valid)
    static = _source(legacy_static_features, rows, (1542,), "legacy_static_features")
    quality = _source(legacy_quality_features, rows, (8,), "legacy_quality_features")
    raw_seq = _source(raw_sequence, rows, (17, 21), "raw_sequence")
    raw_sum = _source(raw_summary, rows, (58,), "raw_summary")
    comp_seq = _source(
        compensated_sequence, rows, (17, 21), "compensated_sequence"
    )
    comp_sum = _source(compensated_summary, rows, (58,), "compensated_summary")
    camera = _source(camera_quality, rows, (17,), "camera_quality")
    if np.any((camera < 0) | (camera > 1)):
        raise ValueError("P4 camera quality left its declared [0,1] range")

    visual_posture, visual_motion = multiscale_arm_features(
        visual, "dual_scale_factorized"
    )
    visual_direct, unexpected = multiscale_arm_features(
        visual, "dual_scale_vjepa_dino"
    )
    if visual_motion is None or unexpected is not None:
        raise RuntimeError("P3 feature API no longer matches the P4 declaration")
    context_geometry = np.concatenate((static[:, 768:1542], quality), axis=1)
    raw_track = np.concatenate((raw_sum, raw_seq.reshape(rows, -1), camera), axis=1)
    compensated_track = np.concatenate(
        (comp_sum, comp_seq.reshape(rows, -1), camera), axis=1
    )
    disagreement = np.abs(raw_sum - comp_sum)
    dual_track = np.concatenate((raw_track, compensated_track, disagreement), axis=1)
    values = (
        visual_posture,
        visual_motion,
        visual_direct,
        context_geometry,
        raw_track,
        compensated_track,
        disagreement,
        dual_track,
    )
    expected = (3072, 4608, 4608, 782, 432, 432, 58, 922)
    for value, width in zip(values, expected, strict=True):
        if value.shape != (rows, width) or value.dtype != np.float32:
            raise RuntimeError("Derived P4 feature width or dtype changed")
        if not np.isfinite(value).all():
            raise ValueError("Derived P4 feature contains nonfinite values")
    return KinematicFeatures(
        *values,
        camera.mean(axis=1, dtype=np.float32),
    )


def arm_inputs(features: KinematicFeatures, arm: str) -> ArmInputs:
    """Return the exact direct, one-reference, or two-reference P4 matrices."""
    vp, vm = features.visual_posture, features.visual_motion
    context = features.context_geometry
    raw, compensated, dual = (
        features.raw_track,
        features.compensated_track,
        features.dual_track,
    )
    if arm == "visual_factorized_refit":
        return ArmInputs("factorized", vp, (vm,))
    if arm == "visual_context_factorized":
        return ArmInputs("factorized", np.concatenate((vp, context), axis=1), (vm,))
    if arm == "visual_raw_track_factorized":
        return ArmInputs("factorized", vp, (np.concatenate((vm, raw), axis=1),))
    if arm == "visual_comp_track_factorized":
        return ArmInputs(
            "factorized", vp, (np.concatenate((vm, compensated), axis=1),)
        )
    if arm == "visual_dual_track_factorized":
        return ArmInputs(
            "factorized",
            np.concatenate((vp, context), axis=1),
            (np.concatenate((vm, dual), axis=1),),
        )
    if arm == "camera_reference_marginalized":
        return ArmInputs(
            "marginalized",
            np.concatenate((vp, context), axis=1),
            (
                np.concatenate((vm, raw), axis=1),
                np.concatenate((vm, compensated), axis=1),
            ),
        )
    if arm == "visual_dual_track_multinomial":
        return ArmInputs(
            "multinomial",
            np.concatenate((features.visual_direct, context, dual), axis=1),
            (),
        )
    raise ValueError(f"Unknown locked P4 arm: {arm}")
