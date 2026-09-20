from __future__ import annotations

import pytest
import torch

from hac.fine_local_motion import (
    FineLocalMotion,
    centered_second_moment,
    expand_coarse_tokens,
    group_fine_cells,
    ungroup_fine_cells,
)


def small_model(match="fixed", detail="both"):
    torch.manual_seed(5)
    return FineLocalMotion(
        6, rank=4, width=16, layers=1, heads=4, dropout=0.0, match_mode=match, detail_mode=detail
    ).eval()


def sample():
    generator = torch.Generator().manual_seed(11)
    coarse = torch.randn(2, 8, 9, 6, generator=generator)
    times = torch.arange(8).float()[None].expand(2, -1) * 8 / 30
    return coarse, expand_coarse_tokens(coarse), times


def test_region_layout_roundtrip_and_repetition_match_exactly():
    coarse, control, _ = sample()
    grouped = group_fine_cells(control)
    assert torch.equal(grouped, coarse[:, :, :, None, :].expand(-1, -1, -1, 16, -1))
    assert torch.equal(ungroup_fine_cells(grouped), control)


def test_zero_mean_fluctuations_survive_before_pooling_squared_operator():
    values = torch.zeros(1, 1, 9, 16, 2)
    values[..., 0, 0] = 2
    values[..., 1, 0] = -2
    _, moments = centered_second_moment(values, torch.ones(values.shape[:-1]))
    assert torch.equal(values.mean(-2), torch.zeros(1, 1, 9, 2))
    assert torch.all(moments[..., 0] == 0.5)
    assert torch.all(moments[..., 1] == 0)


@pytest.mark.parametrize("match", ["fixed", "local_soft"])
def test_constant_within_cell_inputs_equal_repeated_coarse_control(match):
    coarse, repeated, times = sample()
    explicit_fine = ungroup_fine_cells(coarse[:, :, :, None, :].expand(-1, -1, -1, 16, -1).clone())
    net = small_model(match)
    first = net(repeated, times, coarse_reference=coarse)
    second = net(explicit_fine, times, coarse_reference=coarse)
    assert torch.equal(first["logits"], second["logits"])
    assert torch.equal(first["detail_moments"], second["detail_moments"])
    assert torch.allclose(
        first["detail_moments"], torch.zeros_like(first["detail_moments"]), atol=1e-10
    )


def test_identical_macro_means_can_produce_different_final_output():
    coarse, repeated, times = sample()
    grouped = group_fine_cells(repeated).clone()
    drift = torch.arange(8).float()[None, :, None] * 0.3
    grouped[..., 0, 0] += drift
    grouped[..., 1, 0] -= drift
    fine = ungroup_fine_cells(grouped)
    assert torch.allclose(grouped.mean(-2), coarse, atol=1e-6)
    net = small_model()
    control = net(repeated, times, coarse_reference=coarse)
    candidate = net(fine, times, coarse_reference=coarse)
    assert torch.equal(candidate["macro_tokens"], control["macro_tokens"])
    assert candidate["detail_moments"][:, 1].abs().sum() > 0
    assert not torch.allclose(candidate["logits"], control["logits"], atol=1e-8)


@pytest.mark.parametrize("match", ["fixed", "local_soft"])
def test_static_fine_detail_does_not_create_temporal_motion(match):
    coarse, _, times = sample()
    fine = torch.randn(2, 1, 144, 6).expand(-1, 8, -1, -1).clone()
    coarse = group_fine_cells(fine).mean(-2)
    output = small_model(match)(fine, times, coarse_reference=coarse)
    assert output["detail_moments"][:, 0].sum() > 0
    assert torch.allclose(
        output["detail_moments"][:, 1:],
        torch.zeros_like(output["detail_moments"][:, 1:]),
        atol=1e-10,
    )


def test_static_detail_ablation_keeps_parameter_shapes_and_masks_only_motion():
    coarse, fine, times = sample()
    fine = fine + torch.randn_like(fine) * 0.2
    spatial = small_model(detail="spatial_only")
    motion = small_model(detail="motion_only")
    assert spatial.trainable_parameters == motion.trainable_parameters
    left = spatial(fine, times, coarse_reference=coarse)
    right = motion(fine, times, coarse_reference=coarse)
    assert torch.all(left["used_detail_moments"][:, 1:] == 0)
    assert torch.all(right["used_detail_moments"][:, 0] == 0)


@pytest.mark.parametrize("match", ["fixed", "local_soft"])
def test_gradients_finite_and_masked_nans_inert(match):
    coarse, fine, times = sample()
    fine = fine + torch.randn_like(fine) * 0.1
    valid = torch.ones(fine.shape[:3], dtype=torch.bool)
    valid[:, 1, 3] = False
    net = small_model(match)
    before = net(fine, times, coarse_reference=coarse, valid=valid)
    fine[~valid] = torch.nan
    fine.requires_grad_()
    after = net(fine, times, coarse_reference=coarse, valid=valid)
    assert torch.equal(before["logits"], after["logits"])
    after["logits"].square().mean().backward()
    assert net.projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(fine.grad).all()
    assert torch.all(fine.grad[~valid] == 0)


@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("match", ["fixed", "local_soft"])
def test_declared_ranks_remain_below_one_million(rank, match):
    assert FineLocalMotion(rank=rank, match_mode=match).trainable_parameters < 1_000_000


def test_forward_cannot_receive_action_labels():
    coarse, fine, times = sample()
    with pytest.raises(TypeError, match="labels"):
        small_model()(fine, times, coarse_reference=coarse, labels=torch.zeros(2))
