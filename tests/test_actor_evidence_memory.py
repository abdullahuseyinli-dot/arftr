from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from hac.actor_evidence_memory import VARIANTS, ActorEvidenceMemory


def _model(variant: str = "survival_memory") -> ActorEvidenceMemory:
    torch.manual_seed(19)
    return ActorEvidenceMemory(8, variant, width=16, layers=2, heads=4, dropout=0.0)


def _inputs(batch: int = 3) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(31)
    features = torch.randn(batch, 5, 8, generator=generator)
    probabilities = torch.randn(batch, 5, 3, generator=generator).softmax(-1)
    times = torch.arange(-2, 3, dtype=torch.float32).expand(batch, -1).clone()
    valid = torch.ones(batch, 5, dtype=torch.bool)
    return features, probabilities, times, valid


@pytest.mark.parametrize("variant", VARIANTS)
def test_outputs_are_distributions_and_masked_nan_padding_is_inert(variant: str) -> None:
    model = _model(variant).eval()
    features, probabilities, times, valid = _inputs()
    valid[1, 1] = False
    valid[2] = torch.tensor([False, False, True, False, False])
    finite = model(features, probabilities, times, valid)
    for value in (features, probabilities, times):
        value[~valid] = torch.nan
    output = model(features, probabilities, times, valid)
    assert torch.equal(output["probabilities"], finite["probabilities"])
    assert torch.isfinite(output["probabilities"]).all()
    assert torch.allclose(output["probabilities"].sum(-1), torch.ones(3), atol=1e-6)
    assert torch.all(output["attention"][~valid] == 0)
    assert torch.equal(output["probabilities"][2], probabilities[2, 2])
    assert output["gate"][2] == 0
    assert output["attention"][2, 2] == 1


def test_no_neighbor_fallback_preserves_float64_center_exactly() -> None:
    features, probabilities, times, valid = _inputs(1)
    probabilities = probabilities.double()
    probabilities[0, 2] = torch.tensor([0.1234567890123, 0.3210987654321, 0.5554444455556])
    probabilities[0, 2] /= probabilities[0, 2].sum()
    valid[:] = False
    valid[:, 2] = True
    output = _model()(features, probabilities, times, valid)
    assert output["probabilities"].dtype == torch.float64
    assert torch.equal(output["probabilities"][0], probabilities[0, 2])


def test_neighbor_mixture_uses_original_float64_probabilities_and_zero_gate_is_exact() -> None:
    features, probabilities, times, valid = _inputs(2)
    probabilities = probabilities.double()
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    net = _model().eval()
    output = net(features, probabilities, times, valid)
    expected_memory = torch.einsum("bn,bnc->bc", output["attention"], probabilities)
    expected_gate = output["gate"].double()[:, None]
    expected = (1 - expected_gate) * probabilities[:, 2] + expected_gate * expected_memory
    assert torch.equal(output["memory_probabilities"], expected_memory)
    assert torch.equal(output["probabilities"], expected)
    with torch.no_grad():
        net.fusion_gate[-1].weight.zero_()
        net.fusion_gate[-1].bias.fill_(-1000)
    zero_gate = net(features, probabilities, times, valid)
    assert torch.equal(zero_gate["probabilities"], probabilities[:, 2])


def test_edges_bridge_missing_slots_and_boundary_logits_match_probabilities() -> None:
    features, probabilities, times, valid = _inputs(1)
    valid[0] = torch.tensor([True, False, True, False, True])
    output = _model()(features, probabilities, times, valid)
    assert output["boundary_left_index"].tolist() == [[-1, -1, 0, -1, 2]]
    assert output["boundary_dt"].tolist() == [[0.0, 0.0, 2.0, 0.0, 2.0]]
    edge_mask = output["boundary_valid"]
    assert torch.allclose(
        output["boundary_logits"][edge_mask].sigmoid(),
        output["boundary_probabilities"][edge_mask],
        atol=1e-6,
    )
    assert output["survival"][0, 2] == 1


def test_constant_hazard_respects_elapsed_time_and_missing_observation_density() -> None:
    model = _model().eval()
    with torch.no_grad():
        for parameter in model.boundary_head.parameters():
            parameter.zero_()
        model.boundary_head[-1].bias.fill_(math.log(math.expm1(0.25)))
    features, probabilities, times, valid = _inputs(1)
    dense = model(features, probabilities, times, valid)
    valid[:, 1] = False
    valid[:, 3] = False
    sparse = model(features, probabilities, times, valid)
    expected = torch.tensor([[math.exp(-0.5), math.exp(-0.25), 1, math.exp(-0.25), math.exp(-0.5)]])
    assert torch.allclose(dense["survival"], expected, atol=1e-6)
    assert torch.allclose(sparse["survival"][valid], expected[valid], atol=1e-6)
    stretched = model(features, probabilities, times * 2, valid)
    assert torch.allclose(stretched["survival"][valid], expected[valid].square(), atol=1e-6)


def test_survival_changes_retrieval_and_boundary_auxiliary_has_gradient() -> None:
    attention_model = _model("query_attention")
    survival_model = _model("survival_memory")
    survival_model.load_state_dict(attention_model.state_dict(), strict=True)
    features, probabilities, times, valid = _inputs()
    ordinary = attention_model(features, probabilities, times, valid)
    survival = survival_model(features, probabilities, times, valid)
    assert torch.equal(ordinary["boundary_logits"], survival["boundary_logits"])
    assert not torch.allclose(ordinary["attention"], survival["attention"])
    assert torch.all(survival["attention"][:, 2] > ordinary["attention"][:, 2])
    mask = ordinary["boundary_valid"]
    boundary_loss = F.binary_cross_entropy_with_logits(
        ordinary["boundary_logits"][mask], torch.ones_like(ordinary["boundary_logits"][mask])
    )
    boundary_loss.backward()
    assert attention_model.boundary_head[-1].weight.grad.abs().sum() > 0


def test_boundary_head_affects_classification_only_when_survival_is_enabled() -> None:
    inputs = _inputs()
    ordinary = _model("query_attention")
    survival = _model("survival_memory")
    for model in (ordinary, survival):
        prediction = model(*inputs)["probabilities"]
        (-prediction[:, 0].log().mean()).backward()
    assert ordinary.boundary_head[-1].weight.grad is None
    assert survival.boundary_head[-1].weight.grad.abs().sum() > 0


def test_large_hazards_keep_boundary_loss_and_gradients_finite() -> None:
    model = _model()
    with torch.no_grad():
        model.boundary_head[-1].weight.zero_()
        model.boundary_head[-1].bias.fill_(1000.0)
    output = model(*_inputs())
    mask = output["boundary_valid"]
    logits = output["boundary_logits"][mask]
    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.boundary_head[-1].bias.grad).all()
    assert torch.isfinite(output["probabilities"]).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_nonzero_gate_allows_classification_gradient_and_invalid_rows_stay_finite(
    variant: str,
) -> None:
    model = _model(variant)
    features, probabilities, times, valid = _inputs()
    valid[1, 0] = False
    features[~valid] = torch.nan
    features.requires_grad_()
    output = model(features, probabilities, times, valid)
    assert torch.all(output["gate"] > 0)
    assert torch.all(output["gate"] < 0.1)
    loss = -output["probabilities"][:, 0].log().mean() + 0.001 * output["gate"].square().mean()
    loss.backward()
    assert model.input_projection[1].weight.grad.abs().sum() > 0
    assert model.fusion_gate[-1].bias.grad.abs().sum() > 0
    assert torch.isfinite(features.grad).all()
    assert torch.all(features.grad[~valid] == 0)


def test_one_useful_witness_is_available_without_pair_support() -> None:
    features, probabilities, times, valid = _inputs(1)
    valid[:] = False
    valid[:, 2:4] = True
    output = _model("corroborated_memory")(features, probabilities, times, valid)
    assert output["attention"][0, 3] > 0
    assert output["gate"][0] > 0
    assert torch.equal(output["support_features"][0, :3], probabilities[0, 3])
    assert torch.all(output["support_features"][0, 3:6] == 0)


def test_pair_support_does_not_count_duplicate_bin_as_an_extra_witness() -> None:
    probabilities = torch.tensor(
        [[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]]
    )
    times = torch.tensor([[-1.8, -1.6, 0.0, 1.2]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    centers = torch.tensor([2])
    duplicate = ActorEvidenceMemory._support(probabilities, times, valid, centers)
    valid[:, 1] = False
    single = ActorEvidenceMemory._support(probabilities, times, valid, centers)
    assert torch.allclose(duplicate[:, 3:6], single[:, 3:6])
    assert torch.allclose(duplicate[:, 7], single[:, 7])


def test_center_can_be_per_row_and_forward_does_not_accept_labels() -> None:
    features, probabilities, times, valid = _inputs(2)
    valid[:] = False
    valid[0, 1] = True
    valid[1, 3] = True
    centers = torch.tensor([1, 3])
    model = _model()
    output = model(features, probabilities, times, valid, centers)
    assert torch.equal(output["probabilities"], probabilities[torch.arange(2), centers])
    with pytest.raises(TypeError, match="labels"):
        model(features, probabilities, times, valid, centers, labels=torch.zeros(2))


def test_invalid_center_bad_probabilities_and_backward_time_are_rejected() -> None:
    features, probabilities, times, valid = _inputs(1)
    model = _model()
    valid[:, 2] = False
    with pytest.raises(ValueError, match="center observation"):
        model(features, probabilities, times, valid)
    valid[:, 2] = True
    probabilities[:, 0] = 1
    with pytest.raises(ValueError, match="sum to one"):
        model(features, probabilities, times, valid)
    probabilities[:, 0] /= 3
    times[:, 1] = -3
    with pytest.raises(ValueError, match="ordered"):
        model(features, probabilities, times, valid)


@pytest.mark.parametrize("variant", VARIANTS)
def test_declared_feature_width_fits_parameter_cap(variant: str) -> None:
    model = ActorEvidenceMemory(3100, variant)
    assert model.trainable_parameters < 1_000_000
    with pytest.raises(ValueError, match="exceeding"):
        ActorEvidenceMemory(10000, variant)
