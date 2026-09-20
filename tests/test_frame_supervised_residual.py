from __future__ import annotations

import numpy as np
import pytest
import torch

from hac.frame_supervised_residual import (
    FrameSupervisedAnchorResidual,
    anchored_probabilities,
    class_weights,
    frame_residual_loss,
)


def test_zero_initialized_residual_is_exact_anchor_probability():
    model = FrameSupervisedAnchorResidual(input_dim=8, width=4, dropout=0.0)
    model.eval()
    features = torch.randn(5, 8)
    anchor = torch.softmax(torch.randn(5, 3), dim=1)
    output = torch.softmax(model(features, anchor)["logits"], dim=1)
    assert torch.allclose(output, anchor, atol=1e-7)
    assert torch.count_nonzero(model.local_classifier.weight) == 0
    assert model.trainable_parameters < 110_000
    assert np.array_equal(
        anchored_probabilities(anchor.numpy(), np.zeros((5, 3)), residual_scale=0.25),
        anchor.numpy().astype(np.float64),
    )


def test_loss_trains_shared_local_head_and_anchor_residual():
    model = FrameSupervisedAnchorResidual(input_dim=8, width=4, dropout=0.0)
    center = torch.randn(5, 8)
    auxiliary = torch.randn(9, 8)
    anchor = torch.softmax(torch.randn(5, 3), dim=1)
    loss, parts = frame_residual_loss(
        model,
        center_features=center,
        anchor_probabilities=anchor,
        center_labels=torch.tensor([0, 1, 2, 1, 2]),
        auxiliary_features=auxiliary,
        auxiliary_labels=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        auxiliary_weights=torch.ones(9),
        class_weight=torch.ones(3),
        center_loss_weight=1.0,
        auxiliary_loss_weight=0.5,
        anchor_kl_weight=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"center", "auxiliary", "anchor_kl"}
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_class_weights_use_only_supplied_rows():
    labels = np.array([0, 0, 1, 2, 2, 2])
    weights = class_weights(labels, np.array([0, 2, 3, 4]))
    assert np.allclose(weights, [4 / 3, 4 / 3, 2 / 3])


def test_invalid_anchor_is_rejected():
    model = FrameSupervisedAnchorResidual(input_dim=8, width=4)
    with pytest.raises(ValueError, match="simplex"):
        model(torch.randn(2, 8), torch.ones(2, 3))


def test_auxiliary_weighted_mean_matches_center_ce_semantics():
    model = FrameSupervisedAnchorResidual(input_dim=3, width=3, dropout=0.0)
    with torch.no_grad():
        model.encoder[1].weight.copy_(torch.eye(3))
        model.encoder[1].bias.zero_()
        model.local_classifier.weight.copy_(torch.eye(3))
        model.local_classifier.bias.zero_()
    auxiliary = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]])
    labels = torch.tensor([0, 1, 2])
    physical = torch.tensor([0.5, 1.0, 2.0])
    class_weight = torch.tensor([3.0, 2.0, 1.0])
    loss, parts = frame_residual_loss(
        model,
        center_features=auxiliary,
        anchor_probabilities=torch.full((3, 3), 1 / 3),
        center_labels=labels,
        auxiliary_features=auxiliary,
        auxiliary_labels=labels,
        auxiliary_weights=physical,
        class_weight=class_weight,
        center_loss_weight=0.0,
        auxiliary_loss_weight=1.0,
        anchor_kl_weight=0.0,
    )
    logits = model.local_logits(auxiliary)
    per_row = torch.nn.functional.cross_entropy(
        logits, labels, weight=class_weight, reduction="none"
    )
    expected = (per_row * physical).sum() / (physical * class_weight[labels]).sum()
    assert torch.allclose(parts["auxiliary"], expected)
    assert torch.allclose(loss, expected)
