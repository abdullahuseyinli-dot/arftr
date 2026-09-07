from __future__ import annotations

import numpy as np
import pytest

from hac.video_kinematic import ARMS, arm_inputs, derive_kinematic_features
from hac.video_multiscale import MultiScaleFeatures


def visual(rows: int = 2) -> MultiScaleFeatures:
    values = [np.full((rows, 768), index, dtype=np.float32) for index in range(9)]
    return MultiScaleFeatures(*values, np.ones(rows, dtype=bool))


def sources(rows: int = 2) -> tuple[np.ndarray, ...]:
    static = np.zeros((rows, 1542), dtype=np.float32)
    static[:, :768] = -100
    static[:, 768:] = 3
    quality = np.full((rows, 8), 4, dtype=np.float32)
    raw_sequence = np.full((rows, 17, 21), 5, dtype=np.float32)
    raw_summary = np.full((rows, 58), 6, dtype=np.float32)
    comp_sequence = np.full((rows, 17, 21), 7, dtype=np.float32)
    comp_summary = np.full((rows, 58), 8, dtype=np.float32)
    camera = np.tile(np.linspace(0, 1, 17, dtype=np.float32), (rows, 1))
    return static, quality, raw_sequence, raw_summary, comp_sequence, comp_summary, camera


def test_declared_feature_recipe_widths_and_context_excludes_tight_block() -> None:
    features = derive_kinematic_features(visual(), *sources())
    assert features.visual_posture.shape == (2, 3072)
    assert features.visual_motion.shape == (2, 4608)
    assert features.visual_direct.shape == (2, 4608)
    assert features.context_geometry.shape == (2, 782)
    assert features.raw_track.shape == features.compensated_track.shape == (2, 432)
    assert features.reference_disagreement.shape == (2, 58)
    assert features.dual_track.shape == (2, 922)
    assert not np.any(features.context_geometry == -100)
    np.testing.assert_allclose(features.reference_disagreement, 2)
    np.testing.assert_allclose(features.camera_quality_mean, 0.5)


def test_seven_arm_kinds_reference_counts_and_dimensions() -> None:
    features = derive_kinematic_features(visual(), *sources())
    expected = {
        "visual_factorized_refit": ("factorized", 3072, (4608,)),
        "visual_context_factorized": ("factorized", 3854, (4608,)),
        "visual_raw_track_factorized": ("factorized", 3072, (5040,)),
        "visual_comp_track_factorized": ("factorized", 3072, (5040,)),
        "visual_dual_track_factorized": ("factorized", 3854, (5530,)),
        "camera_reference_marginalized": ("marginalized", 3854, (5040, 5040)),
        "visual_dual_track_multinomial": ("multinomial", 6312, ()),
    }
    assert set(expected) == set(ARMS)
    for arm, (kind, width, motion_widths) in expected.items():
        inputs = arm_inputs(features, arm)
        assert inputs.kind == kind
        assert inputs.posture_or_direct.shape == (2, width)
        assert tuple(value.shape[1] for value in inputs.motion_references) == motion_widths


def test_nonfinite_bad_dtype_camera_range_and_unknown_arm_fail_closed() -> None:
    values = list(sources())
    values[0] = values[0].astype(np.float16)
    with pytest.raises(ValueError, match="shape/dtype"):
        derive_kinematic_features(visual(), *values)
    values = list(sources())
    values[-1][0, 0] = 2
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        derive_kinematic_features(visual(), *values)
    features = derive_kinematic_features(visual(), *sources())
    with pytest.raises(ValueError, match="Unknown locked"):
        arm_inputs(features, "posthoc")
