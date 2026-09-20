from __future__ import annotations

import math

import pytest
import torch

from hac.center_episode_memory import (
    CenterEpisodeMemory,
    expected_episode_attention,
    unweighted_boundary_loss,
)
from hac.stable_center_episode_memory import (
    StableCenterEpisodeMemory,
    stable_expected_episode_attention,
)


def _inputs(batch=3):
    generator = torch.Generator().manual_seed(821)
    return (
        torch.randn(batch, 5, 8, generator=generator),
        torch.randn(batch, 5, 3, dtype=torch.float64, generator=generator).softmax(-1),
        torch.arange(-2, 3, dtype=torch.float32).expand(batch, -1).clone(),
        torch.ones(batch, 5, dtype=torch.bool),
    )


def _model(variant="expected_episode"):
    torch.manual_seed(39)
    return StableCenterEpisodeMemory(8, variant, width=16, hazard_width=8, dropout=0)


def _normalized_attention(output):
    values = output["attention"].double()
    return values / values.sum(-1, keepdim=True)


@pytest.mark.parametrize(
    "masses,utilities",
    [
        (
            [0, 19.77034568786621, 16.217815399169922, 0.00019455834990367293, 1.3005503416061401],
            [
                -45.75699996948242,
                -47.198036193847656,
                4.813899040222168,
                58.39836502075195,
                -45.898799896240234,
            ],
        ),
        (
            [0, 11.395994186401367, 9030.2138671875, 1.1855940101668239e-5, 52.908973693847656],
            [
                -22.59271240234375,
                -1.5198841094970703,
                46.70502471923828,
                82.53208923339844,
                4.692370891571045,
            ],
        ),
    ],
)
def test_known_prefix_cancellation_failures_keep_original_strict_bound(masses, utilities):
    masses, utilities = torch.tensor([masses]), torch.tensor([utilities])
    valid, centers = torch.ones(1, 5, dtype=torch.bool), torch.tensor([2])
    original = expected_episode_attention(utilities, masses, valid, centers)
    assert (_normalized_attention(original) - original["survival"]).max() > 1e-6
    stable = stable_expected_episode_attention(utilities, masses, valid, centers)
    attention = _normalized_attention(stable)
    assert (attention - stable["survival"]).max() < 1e-12
    torch.testing.assert_close(
        stable["survival"][0, 3], torch.exp(-masses[0, 3].double()), atol=0, rtol=0
    )
    assert (stable["segment_probabilities"].sum(-1) - 1).abs().max() < 1e-12


@pytest.mark.parametrize("maximum_mass", [30.0, 1e8])
def test_all_eighty_masks_centers_and_adversarial_masses_are_normalized_and_bounded(maximum_mass):
    masks, centers = [], []
    for bits in range(1, 32):
        mask = [(bits >> slot) & 1 == 1 for slot in range(5)]
        for center, present in enumerate(mask):
            if present:
                masks.append(mask)
                centers.append(center)
    assert len(masks) == 80
    valid = torch.tensor(masks).repeat(100, 1)
    centers = torch.tensor(centers).repeat(100)
    generator = torch.Generator().manual_seed(718)
    mass = 10 ** torch.empty(len(valid), 5).uniform_(
        -8, math.log10(maximum_mass), generator=generator
    )
    mass[~valid] = 0
    mass[torch.arange(len(valid)), valid.long().argmax(1)] = 0
    utility = torch.randn(len(valid), 5, generator=generator) * 100
    output = stable_expected_episode_attention(utility, mass, valid, centers)
    assert output["attention"].dtype == torch.float64
    assert (output["segment_probabilities"].sum(1) - 1).abs().max() < 1e-12
    attention = _normalized_attention(output)
    assert torch.isfinite(attention).all()
    assert (attention.sum(1) - 1).abs().max() < 1e-12
    assert (attention - output["survival"]).max() < 1e-12
    slots = torch.arange(5)[None, None, :]
    membership = (
        valid[:, None, :]
        & output["segment_valid"][..., None]
        & (slots >= output["segment_left_index"][..., None])
        & (slots <= output["segment_right_index"][..., None])
    )
    marginal = (membership * output["segment_probabilities"][..., None]).sum(1)
    torch.testing.assert_close(marginal, output["survival"], atol=1e-12, rtol=0)


def test_unrelated_huge_prefix_cannot_change_local_path_mass():
    utility, valid, center = (
        torch.zeros(1, 5),
        torch.ones(1, 5, dtype=torch.bool),
        torch.tensor([2]),
    )
    initial = torch.tensor([[0, 1.0, 1.0, 0.0001, 0.5]])
    huge = initial.clone()
    huge[:, :3] = torch.tensor([0, 1e30, 1e30])
    before = stable_expected_episode_attention(utility, initial, valid, center)
    after = stable_expected_episode_attention(utility, huge, valid, center)
    torch.testing.assert_close(
        before["log_survival"][:, 2:], after["log_survival"][:, 2:], atol=0, rtol=0
    )


@pytest.mark.parametrize("variant", ["expected_episode", "log_survival_control"])
def test_identical_parameter_schema_and_small_mass_mathematical_equivalence(variant):
    stable = _model(variant).eval()
    original = CenterEpisodeMemory(8, variant, width=16, hazard_width=8, dropout=0).eval()
    original.load_state_dict(stable.state_dict(), strict=True)
    inputs = _inputs()
    old, new = original(*inputs), stable(*inputs)
    assert original.trainable_parameters == stable.trainable_parameters
    assert torch.equal(old["boundary_logits"], new["boundary_logits"])
    assert torch.equal(old["utility"], new["utility"])
    torch.testing.assert_close(old["probabilities"], new["probabilities"], atol=2e-7, rtol=0)
    assert StableCenterEpisodeMemory(3072, variant).trainable_parameters == 849059


@pytest.mark.parametrize("variant", ["expected_episode", "log_survival_control"])
def test_neural_float32_gradient_separation_and_exact_singleton(variant):
    model = _model(variant)
    features, probabilities, times, valid = _inputs()
    valid[1, 1] = False
    valid[2] = torch.tensor([False, False, True, False, False])
    for item in (features, probabilities, times):
        item[~valid] = torch.nan
    observed_gate_dtypes = []
    hook = model.fusion_gate[0].register_forward_pre_hook(
        lambda _module, args: observed_gate_dtypes.append(args[0].dtype)
    )
    output = model(features, probabilities, times, valid)
    hook.remove()
    assert observed_gate_dtypes == [torch.float32]
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert output["attention"].dtype == torch.float64
    assert torch.equal(output["probabilities"][2], probabilities[2, 2])
    assert output["gate"][2] == 0
    assert torch.isfinite(output["probabilities"]).all()
    (-output["probabilities"][:, 0].log().mean()).backward()
    assert all(parameter.grad is None for parameter in model.hazard_parameters())
    assert model.usefulness[-1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    output = model(features, probabilities, times, valid)
    unweighted_boundary_loss(output, torch.zeros(3, 5), valid).backward()
    assert model.hazard_head[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.utility_parameters())


def test_constant_rate_duration_and_missing_slot_bridges():
    model = _model().eval()
    with torch.no_grad():
        for parameter in model.hazard_head.parameters():
            parameter.zero_()
        model.hazard_head[-1].bias.fill_(math.log(math.expm1(0.25)))
    features, p, times, valid = _inputs(1)
    dense = model(features, p, times, valid)
    valid[:, 1] = False
    valid[:, 3] = False
    sparse = model(features, p, times, valid)
    doubled = model(features, p, times * 2, valid)
    assert sparse["boundary_left_index"].tolist() == [[-1, -1, 0, -1, 2]]
    torch.testing.assert_close(
        dense["survival"][valid], sparse["survival"][valid], atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        doubled["survival"][valid], sparse["survival"][valid].square(), atol=1e-12, rtol=0
    )


def test_float32_original_probability_mixture_remains_normalized_and_bounded():
    model = _model().eval()
    features, p, times, valid = _inputs()
    output = model(features, p.float(), times, valid)
    assert output["probabilities"].dtype == torch.float32
    assert (output["attention"].sum(1) - 1).abs().max() < 2e-7
    assert (output["attention"].double() - output["survival"]).max() < 1e-6
    assert (output["probabilities"].sum(1) - 1).abs().max() < 2e-7
