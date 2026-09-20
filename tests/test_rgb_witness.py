import numpy as np
import pytest
import torch

from hac.rgb_witness import (ARMS, EXPECTED_PARAMETERS, RGBWitness, bounded_pair_candidate,
                             fit_scope_transform, rgb_witness_loss)


@pytest.mark.parametrize("arm", ARMS)
def test_matched_heads_parameter_counts_and_anchor_initialization(arm):
    torch.manual_seed(3)
    model = RGBWitness(arm)
    assert model.trainable_parameters == EXPECTED_PARAMETERS[arm]
    features = torch.randn(8, 2, 96)
    geometry = torch.randn(8, 6)
    available = torch.ones(8, 2, dtype=torch.bool)
    anchor = torch.tensor([[.45, .50, .05]]).repeat(8, 1)
    output = model(features, geometry, available, anchor)
    assert torch.equal(output["candidate"], anchor)
    labels = torch.arange(8) % 3
    loss, pieces = rgb_witness_loss(output, features, anchor, labels, arm)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert set(pieces) == {"task", "kl", "residual", "witness", "energy_ce"}


def test_pair_action_retains_and_never_changes_walking_mass():
    anchor = np.asarray([[.45,.50,.05],[.1,.2,.7],[.2,.2,.6]], np.float64)
    available = np.asarray([[1,1],[1,1],[1,0]], bool)
    output, eligible = bounded_pair_candidate(anchor, np.asarray([.5,.5,-.5]), available)
    assert eligible.tolist() == [True, False, False]
    assert np.array_equal(output[~eligible], anchor[~eligible])
    assert np.array_equal(output[:,2], anchor[:,2])
    assert np.allclose(output.sum(1), 1)
    assert output[0].argmax() == 0


def test_ineligible_rows_have_no_reader_gradient():
    model = RGBWitness("direct")
    features = torch.randn(5,2,96)
    geometry = torch.randn(5,6)
    available = torch.zeros(5,2,dtype=torch.bool)
    anchor = torch.tensor([[.1,.2,.7]]).repeat(5,1)
    labels = torch.full((5,),2,dtype=torch.long)
    output = model(features,geometry,available,anchor)
    loss,_ = rgb_witness_loss(output,features,anchor,labels,"direct")
    loss.backward()
    assert torch.count_nonzero(model.reader.final.weight.grad) == 0


def test_scope_transform_is_finite_and_training_only_shape():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(160,2,768)).astype(np.float32)
    geometry = rng.normal(size=(160,6)).astype(np.float32)
    transform = fit_scope_transform(features[:128], geometry[:128])
    projected, normalized = transform.apply(features[128:], geometry[128:])
    assert projected.shape == (32,2,96)
    assert normalized.shape == (32,6)
    assert np.isfinite(projected).all() and np.isfinite(normalized).all()
