from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from hac.center_episode_memory import (
    VARIANTS,
    CenterEpisodeMemory,
    expected_episode_attention,
    unweighted_boundary_loss,
)


def _model(variant: str = "expected_episode") -> CenterEpisodeMemory:
    torch.manual_seed(71)
    return CenterEpisodeMemory(8, variant, width=16, hazard_width=8, layers=2, heads=4, dropout=0)


def _inputs(batch: int = 3) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(38)
    return (
        torch.randn(batch, 5, 8, generator=generator),
        torch.randn(batch, 5, 3, generator=generator).softmax(-1),
        torch.arange(-2, 3, dtype=torch.float32).expand(batch, -1).clone(),
        torch.ones(batch, 5, dtype=torch.bool),
    )


def _constant_rate(model: CenterEpisodeMemory, rate: float) -> None:
    with torch.no_grad():
        for parameter in model.hazard_head.parameters():
            parameter.zero_()
        model.hazard_head[-1].bias.fill_(math.log(math.expm1(rate)))


@pytest.mark.parametrize("variant", VARIANTS)
def test_distributions_masked_nan_and_singleton_exactness(variant: str) -> None:
    model = _model(variant).eval()
    features, probabilities, times, valid = _inputs()
    probabilities = probabilities.double()
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    valid[1, 1] = False
    valid[2] = torch.tensor([False, False, True, False, False])
    finite = model(features, probabilities, times, valid)
    for item in (features, probabilities, times):
        item[~valid] = torch.nan
    output = model(features, probabilities, times, valid)
    assert torch.equal(output["probabilities"], finite["probabilities"])
    assert torch.equal(output["probabilities"][2], probabilities[2, 2])
    assert output["gate"][2] == 0 and output["attention"][2, 2] == 1
    assert torch.all(output["attention"][~valid] == 0)
    for key in ("probabilities", "attention", "segment_probabilities"):
        assert torch.isfinite(output[key]).all()
        assert torch.allclose(
            output[key].sum(-1), torch.ones(3, dtype=output[key].dtype), atol=1e-6
        )
    assert output["segment_valid"].sum(1).tolist() == [9, 6, 1]


def test_enumerated_segment_membership_marginals_equal_path_survival() -> None:
    generator = torch.Generator().manual_seed(11)
    # Include every possible nonempty mask and each possible valid center.
    masks, centers = [], []
    for bits in range(1, 32):
        mask = [(bits >> slot) & 1 == 1 for slot in range(5)]
        for center, present in enumerate(mask):
            if present:
                masks.append(mask)
                centers.append(center)
    valid = torch.tensor(masks)
    centers = torch.tensor(centers)
    mass = torch.rand(len(masks), 5, generator=generator, dtype=torch.float64) * 4
    mass[~valid] = 0
    # First available edge has no preceding observation and zero mass.
    first = valid.long().argmax(1)
    mass[torch.arange(len(masks)), first] = 0
    utility = torch.randn(len(masks), 5, generator=generator, dtype=torch.float64) * 20
    output = expected_episode_attention(utility, mass, valid, centers)
    assert torch.allclose(
        output["segment_probabilities"].sum(1),
        torch.ones(len(masks), dtype=torch.float64),
        atol=1e-12,
    )
    slots = torch.arange(5)[None, None, :]
    membership = (
        valid[:, None, :]
        & output["segment_valid"][..., None]
        & (slots >= output["segment_left_index"][..., None])
        & (slots <= output["segment_right_index"][..., None])
    )
    marginal = (membership * output["segment_probabilities"][..., None]).sum(1)
    assert torch.allclose(marginal, output["survival"], atol=1e-12)
    assert torch.all(output["attention"] <= output["survival"] + 1e-12)
    assert torch.allclose(
        output["attention"].sum(1), torch.ones(len(masks), dtype=torch.float64), atol=1e-12
    )


def test_extreme_usefulness_is_bounded_but_log_survival_control_is_not() -> None:
    valid = torch.ones(1, 5, dtype=torch.bool)
    mass = torch.tensor([[0.0, 2.0, 3.0, 2.0, 3.0]], dtype=torch.float64)
    utility = torch.tensor([[10000.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    output = expected_episode_attention(utility, mass, valid, torch.tensor([2]))
    assert torch.allclose(
        output["attention"][0, 0], torch.tensor(math.exp(-5), dtype=torch.float64), atol=1e-12
    )
    assert torch.all(output["attention"] <= output["survival"] + 1e-12)
    control = (utility + output["log_survival"]).softmax(1)
    assert control[0, 0] > 0.999 and control[0, 0] > output["survival"][0, 0] * 100


def test_missing_slots_are_bridged_not_treated_as_extra_boundaries() -> None:
    model = _model().eval()
    _constant_rate(model, 0.25)
    features, probabilities, times, valid = _inputs(1)
    valid[0] = torch.tensor([True, False, True, False, True])
    output = model(features, probabilities, times, valid)
    assert output["boundary_left_index"].tolist() == [[-1, -1, 0, -1, 2]]
    assert output["boundary_dt"].tolist() == [[0.0, 0.0, 2.0, 0.0, 2.0]]
    assert output["segment_valid"].sum() == 4
    endpoints = list(
        zip(
            output["segment_left_index"][0, :4].tolist(),
            output["segment_right_index"][0, :4].tolist(),
            strict=True,
        )
    )
    assert endpoints == [(0, 2), (0, 4), (2, 2), (2, 4)]
    mask = output["boundary_valid"]
    assert torch.allclose(output["boundary_logits"][mask].sigmoid(), output["boundary_probs"][mask])


def test_constant_hazard_physical_time_and_observation_density() -> None:
    model = _model().eval()
    _constant_rate(model, 0.25)
    features, probabilities, times, valid = _inputs(1)
    dense = model(features, probabilities, times, valid)
    expected = torch.tensor([[math.exp(-0.5), math.exp(-0.25), 1, math.exp(-0.25), math.exp(-0.5)]])
    assert torch.allclose(dense["survival"], expected, atol=1e-6)
    valid[:, 1] = False
    valid[:, 3] = False
    sparse = model(features, probabilities, times, valid)
    doubled = model(features, probabilities, times * 2, valid)
    assert torch.allclose(sparse["survival"][valid], expected[valid], atol=1e-6)
    assert torch.allclose(doubled["survival"][valid], expected[valid].square(), atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_classification_and_boundary_gradients_are_separate(variant: str) -> None:
    model = _model(variant)
    features, probabilities, times, valid = _inputs()
    features.requires_grad_()
    output = model(features, probabilities, times, valid)
    classification = (
        -output["probabilities"][:, 0].log().mean() + 0.001 * output["gate"].square().mean()
    )
    classification.backward()
    assert model.input_projection[1].weight.grad.abs().sum() > 0
    assert model.fusion_gate[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is None for name, p in model.named_parameters() if name.startswith("hazard_"))
    assert features.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    features.grad = None
    output = model(features, probabilities, times, valid)
    loss = unweighted_boundary_loss(output, torch.zeros(3, 5), valid)
    loss.backward()
    assert model.hazard_projection[1].weight.grad.abs().sum() > 0
    assert model.hazard_head[-1].weight.grad.abs().sum() > 0
    assert all(
        p.grad is None for name, p in model.named_parameters() if not name.startswith("hazard_")
    )
    assert features.grad is None


def test_hazard_is_identical_when_same_physical_edge_is_recentered_and_shifted() -> None:
    model = _model().eval()
    features, probabilities, times, valid = _inputs(1)
    original = model(features, probabilities, times, valid, center_index=2)
    recentered = model(features, probabilities.roll(1, -1), times - 123.0, valid, center_index=1)
    assert torch.equal(original["boundary_logits"], recentered["boundary_logits"])
    # Place one unchanged physical pair in different slots and shift the clock.
    shifted_features = features.clone()
    shifted_features[:, 2:4] = features[:, :2]
    shifted_times = torch.tensor([[-100.0, -100.0, 51.0, 52.0, -100.0]])
    shifted_valid = torch.tensor([[False, False, True, True, False]])
    relocated = model(shifted_features, probabilities, shifted_times, shifted_valid)
    assert torch.equal(original["boundary_logits"][:, 1], relocated["boundary_logits"][:, 3])


def test_matched_arms_have_identical_modules_hazards_and_utility() -> None:
    episode = _model("expected_episode").eval()
    control = _model("log_survival_control").eval()
    control.load_state_dict(episode.state_dict(), strict=True)
    left, right = episode(*_inputs()), control(*_inputs())
    assert episode.trainable_parameters == control.trainable_parameters
    assert torch.equal(left["boundary_logits"], right["boundary_logits"])
    assert torch.equal(left["utility"], right["utility"])
    assert not torch.allclose(left["attention"], right["attention"])


def test_parameter_iterators_are_disjoint_exhaustive_and_allow_separate_clipping() -> None:
    model = _model()
    hazard = list(model.hazard_parameters())
    utility = list(model.utility_parameters())
    hazard_ids = {id(parameter) for parameter in hazard}
    utility_ids = {id(parameter) for parameter in utility}
    assert not hazard_ids & utility_ids
    assert hazard_ids | utility_ids == {id(parameter) for parameter in model.parameters()}
    for parameter in hazard:
        parameter.grad = torch.ones_like(parameter) * 0.00001
    for parameter in utility:
        parameter.grad = torch.ones_like(parameter) * 100
    before = [parameter.grad.clone() for parameter in hazard]
    torch.nn.utils.clip_grad_norm_(model.utility_parameters(), 1.0)
    assert all(
        torch.equal(parameter.grad, initial)
        for parameter, initial in zip(hazard, before, strict=True)
    )


def test_boundary_loss_is_unweighted_masks_unknown_and_has_empty_zero_gradient() -> None:
    model = _model()
    output = model(*_inputs())
    targets = torch.tensor([[float("nan"), 0, 0, 1, 0]]).expand(3, -1)
    annotation = torch.ones(3, 5, dtype=torch.bool)
    annotation[:, 3] = False
    loss = unweighted_boundary_loss(output, targets, annotation)
    mask = output["boundary_valid"] & annotation
    expected = F.binary_cross_entropy_with_logits(output["boundary_logits"][mask], targets[mask])
    assert torch.equal(loss, expected)
    empty = unweighted_boundary_loss(
        output, torch.full((3, 5), torch.nan), torch.zeros(3, 5, dtype=torch.bool)
    )
    assert empty == 0 and empty.requires_grad
    empty.backward()
    assert model.hazard_head[-1].weight.grad.abs().sum() == 0


def test_zero_gate_uses_original_probability_precision_and_labels_are_absent() -> None:
    model = _model().eval()
    features, probabilities, times, valid = _inputs()
    probabilities = probabilities.double()
    probabilities /= probabilities.sum(-1, keepdim=True)
    with torch.no_grad():
        model.fusion_gate[-1].weight.zero_()
        model.fusion_gate[-1].bias.fill_(-1000)
    output = model(features, probabilities, times, valid)
    assert torch.equal(output["probabilities"], probabilities[:, 2])
    with pytest.raises(TypeError, match="labels"):
        model(features, probabilities, times, valid, labels=torch.zeros(3))


@pytest.mark.parametrize("rate_bias", [-1000.0, 1000.0])
def test_extreme_hazard_losses_and_gradients_are_finite(rate_bias: float) -> None:
    model = _model()
    with torch.no_grad():
        model.hazard_head[-1].weight.zero_()
        model.hazard_head[-1].bias.fill_(rate_bias)
    output = model(*_inputs())
    loss = unweighted_boundary_loss(output, torch.zeros(3, 5), torch.ones(3, 5, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.hazard_head[-1].bias.grad).all()
    assert torch.isfinite(output["probabilities"]).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_cap_and_invalid_inputs(variant: str) -> None:
    assert CenterEpisodeMemory(3072, variant).trainable_parameters == 849059
    assert CenterEpisodeMemory(3100, variant).trainable_parameters < 1_000_000
    with pytest.raises(ValueError, match="exceeding"):
        CenterEpisodeMemory(10000, variant)
    model = _model(variant)
    features, probabilities, times, valid = _inputs(1)
    times[:, 1] = times[:, 0]
    with pytest.raises(ValueError, match="strictly increasing"):
        model(features, probabilities, times, valid)
