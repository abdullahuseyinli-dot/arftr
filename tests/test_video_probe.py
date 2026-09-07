from __future__ import annotations

import pytest
import torch

from hac.video_probe import SpatiotemporalFactorizedProbe


def test_spatiotemporal_probe_shapes_and_probabilities() -> None:
    torch.manual_seed(7)
    model = SpatiotemporalFactorizedProbe(
        12, model_dim=16, layers=1, attention_heads=4, feedforward_dim=32, dropout=0.0
    ).eval()
    features = torch.randn(3, 8, 9, 12)
    mask = torch.ones(3, 8, 9, dtype=torch.bool)
    mask[1, -1] = False
    output = model(features, mask)
    assert output.probabilities.shape == (3, 3)
    assert output.attention_weights.shape == (3, 8, 9)
    assert torch.allclose(output.probabilities.sum(1), torch.ones(3), atol=1e-6)
    assert torch.all(output.attention_weights[1, -1] == 0)


def test_spatiotemporal_probe_rejects_empty_or_oversized_inputs() -> None:
    model = SpatiotemporalFactorizedProbe(
        4,
        model_dim=8,
        attention_heads=2,
        feedforward_dim=16,
        maximum_times=2,
        maximum_regions=2,
    )
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.zeros(1, 3, 1, 4))
    with pytest.raises(ValueError, match="at least one"):
        model(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, dtype=torch.bool))
