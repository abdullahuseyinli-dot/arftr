from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from audit_okutama_memory_observation_density import transform_observations  # noqa: E402

from hac.actor_evidence_memory import ActorEvidenceMemory  # noqa: E402


def inputs():
    generator = torch.Generator().manual_seed(3)
    return (
        torch.randn(2, 5, 8, generator=generator),
        torch.randn(2, 5, 3, generator=generator, dtype=torch.float64).softmax(-1),
        torch.arange(-2, 3, dtype=torch.float32).repeat(2, 1),
        torch.tensor([[True] * 5, [False, True, True, False, True]]),
    )


def test_drop_preserves_pixels_timestamps_and_center_without_mutating_mask():
    values = inputs()
    original = values[-1].clone()
    changed = transform_observations(*values, "drop_inner_neighbors")
    assert all(changed[index] is values[index] for index in range(3))
    assert changed[4] == 2
    assert not changed[3][:, (1, 3)].any()
    assert torch.equal(changed[3][:, (0, 2, 4)], original[:, (0, 2, 4)])
    assert torch.equal(values[-1], original)


def test_duplicate_repeats_exact_observations_and_physical_times():
    values = inputs()
    changed = transform_observations(*values, "duplicate_all_observations")
    assert changed[4] == 4
    for source, repeated in zip(values, changed[:4], strict=True):
        assert torch.equal(source, repeated[:, ::2])
        assert torch.equal(source, repeated[:, 1::2])
    assert torch.equal(changed[0][:, changed[4]], values[0][:, 2])


@pytest.mark.parametrize("arm", ("query_attention", "survival_memory"))
def test_uniform_duplication_adds_no_hazard_exposure_and_preserves_attention_model(arm):
    torch.manual_seed(9)
    model = ActorEvidenceMemory(8, arm, width=16, layers=2, heads=4, dropout=0).eval()
    values = inputs()
    with torch.no_grad():
        original = model(*values)
        duplicated = model(*transform_observations(*values, "duplicate_all_observations"))
    assert torch.allclose(original["survival"], duplicated["survival"][:, ::2], atol=2e-7)
    assert not duplicated["boundary_valid"][:, 1::2].any()
    assert torch.allclose(original["probabilities"], duplicated["probabilities"], atol=1e-7, rtol=0)


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match="Unknown fixed"):
        transform_observations(*inputs(), "tune_on_labels")
