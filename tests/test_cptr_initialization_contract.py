"""Characterize current CPTR behavior; these tests do not assert model improvement."""

import numpy as np
import pytest
import torch

from experiments.audit_cptr_initialization_contract import (
    initialization_report,
    synthetic_kwargs,
    synthetic_model,
)
from hac.cptr_features import sample_indices_with_centre
from hac.vcoco_v3_neural import decode_factorized_logits
from hac.vcoco_v3_temporal import uniform_clip_indices


@pytest.mark.parametrize("valid_frames", [8, 4, 0])
def test_zero_initialized_new_heads_leave_an_attenuated_legacy_path(valid_frames):
    model = synthetic_model()
    kwargs = synthetic_kwargs(valid_frames)
    with torch.inference_mode():
        output = model(**kwargs)
        static = model.static_fallback(kwargs["static_features"])
        teacher = model.legacy_temporal
        assert torch.equal(teacher.last_mask, torch.ones(1, 8, dtype=torch.bool))
        legacy_logits = teacher.logits.expand(1, -1)
        coefficient = torch.sigmoid(torch.tensor(5.0)) * (valid_frames / 8)
        expected = static.posture_logits + coefficient * (legacy_logits - static.posture_logits)
        assert torch.equal(output.posture_logits, expected)
        assert torch.equal(output.motion_logits, expected)
        assert torch.count_nonzero(output.learned_temporal_residual) == 0
        assert torch.equal(output.probabilities, decode_factorized_logits(expected, expected))
        assert not torch.equal(output.probabilities, output.legacy_probabilities)
        assert torch.isfinite(output.probabilities).all()
        if valid_frames == 0:
            assert torch.equal(output.probabilities, output.static_probabilities)


def test_current_and_center_preserving_samplers_match_declared_short_clip():
    original = uniform_clip_indices(17, center_index=8, samples=8, span_frames=8)
    cptr, centre = sample_indices_with_centre(17, centre_index=8, samples=8, span_frames=8)
    assert np.array_equal(original, cptr)
    assert cptr[centre] == 8
    assert original.tolist() == [4, 6, 6, 8, 8, 10, 10, 12]
    assert len(np.unique(original)) == 5


def test_initialization_report_is_repeatable_and_preserves_rng_state():
    before = torch.random.get_rng_state().clone()
    first = initialization_report()
    assert torch.equal(before, torch.random.get_rng_state())
    assert first == initialization_report()
    assert first["dataset_rows_read"] == first["checkpoints_loaded"] == first["fit_steps"] == 0
    assert all(row["max_logit_equation_error"] == 0 for row in first["rows"])
    assert all(row["new_residual_max_abs"] == 0 for row in first["rows"])
    assert all(not row["baseline_exactly_preserved"] for row in first["rows"])
