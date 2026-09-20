import numpy as np
import pytest
import torch

from hac.motion_null_contrast import MotionNullContrast, exact_retaining_candidate
from hac.native_motion_innovation import NativeMotionInnovation, motion_loss


def test_initialization_matches_plain_except_frozen_bias():
    torch.manual_seed(42)
    plain = NativeMotionInnovation()
    torch.manual_seed(42)
    paired = MotionNullContrast()
    assert all(torch.equal(v, paired.state_dict()[k]) for k, v in plain.state_dict().items())
    assert plain.trainable_parameters == 120129
    assert paired.trainable_parameters == 120128
    assert not paired.head[-1].bias.requires_grad


def test_nonzero_weights_nullness_and_independent_formula():
    torch.manual_seed(17)
    model = MotionNullContrast()
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.1)
        model.head[-1].bias.fill_(0.3)
    x = torch.randn(5, 2, 6, 64, 64)
    g = torch.randn(5, 12)
    reference = x.clone()
    reference[:, :, 3:] = 0
    assert torch.equal(model(reference, g), torch.zeros(5))

    def independently_raw(value):
        encoded = model.encoder(value.reshape(-1, 6, 64, 64)).reshape(5, -1)
        return model.head(torch.cat([encoded, g], dim=1)).reshape(-1)

    expected = 0.5 * torch.tanh(independently_raw(x) - independently_raw(reference))
    assert torch.equal(model(x, g), expected)
    assert torch.count_nonzero(expected) > 0


def test_training_moves_weights_and_retains_trained_null_invariant():
    torch.manual_seed(8)
    model = MotionNullContrast()
    initial = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    opt = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=0.001)
    x = torch.randn(16, 2, 6, 64, 64)
    g = torch.randn(16, 12)
    anchor = torch.tensor([[0.05, 0.5, 0.45]]).repeat(16, 1)
    labels = torch.arange(16) % 3
    available = torch.ones(16, 2, dtype=torch.bool)
    for step in range(4):
        opt.zero_grad()
        delta = model(x, g)
        loss, _ = motion_loss(delta, anchor, labels, available)
        loss.backward()
        if step == 0:
            assert torch.count_nonzero(model.head[-1].weight.grad) > 0
        opt.step()
    assert all(not torch.equal(initial[k], v) for k, v in model.named_parameters() if v.requires_grad)
    null = x.clone()
    null[:, :, 3:] = 0
    assert torch.equal(model(null, g), torch.zeros(16))
    assert model.head[-1].bias.grad is None
    assert torch.max(torch.abs(model(x, g))) <= 0.5


def test_inference_retain_is_exact_and_training_cannot_call_wrapper():
    anchor = np.array([[0.07, 0.47, 0.46], [0.6, 0.3, 0.1], [0.08, 0.51, 0.41], [0.02, 0.61, 0.37]])
    available = np.array([[1, 1], [1, 1], [0, 1], [1, 1]], dtype=bool)
    delta = np.array([0.0, 0.2, -0.1, 0.4])
    candidate, eligible = exact_retaining_candidate(anchor, delta, available)
    assert eligible.tolist() == [True, False, False, True]
    assert np.array_equal(candidate[:3], anchor[:3])
    assert np.array_equal(candidate[:, 0], anchor[:, 0])
    assert np.allclose(candidate[:, 1:].sum(1), anchor[:, 1:].sum(1), atol=1e-15, rtol=0)
    with pytest.raises(TypeError):
        exact_retaining_candidate(torch.tensor(anchor), torch.tensor(delta), available)
    with pytest.raises(ValueError):
        exact_retaining_candidate(anchor, np.full(4, 0.6), available)
