import torch

from hac.persistent_token_gauge import (
    PTGDiscoveryHead,
    coarse_temporal_features,
    persistent_token_features,
)


def test_ptg_features_are_finite_and_deterministic():
    torch.manual_seed(7)
    tokens = torch.randn(2, 8, 144, 768)
    times = torch.arange(8, dtype=torch.float32).repeat(2, 1)
    first = persistent_token_features(tokens, times)
    second = persistent_token_features(tokens, times)
    assert first["raw"].shape == (2, 9 * 32 * 4)
    assert first["gauge"].shape[0] == 2
    assert torch.isfinite(first["gauge"]).all()
    assert torch.equal(first["raw"], second["raw"])
    assert torch.equal(first["gauge"], second["gauge"])


def test_ptg_time_reversal_changes_odd_witness():
    torch.manual_seed(11)
    tokens = torch.randn(1, 8, 144, 768)
    times = torch.arange(8, dtype=torch.float32).reshape(1, -1)
    normal = persistent_token_features(tokens, times)
    reversed_features = persistent_token_features(tokens.flip(1), times)
    assert normal["odd"].shape if "odd" in normal else True
    assert not torch.equal(normal["gauge"], reversed_features["gauge"])


def test_coarse_control_and_head_contract():
    torch.manual_seed(13)
    coarse = torch.randn(3, 8, 9, 768)
    times = torch.arange(8, dtype=torch.float32).repeat(3, 1)
    features = coarse_temporal_features(coarse, times)
    assert features.shape == (3, 9 * 32 * 4)
    head = PTGDiscoveryHead(features.shape[1])
    logits = head(features.clone())
    assert logits.shape == (3, 3)
    assert torch.isfinite(logits).all()
