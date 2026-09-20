"""Synthetic invariants for shared evidence; no HAC fitting or dataset access."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from hac.sear import (
    SEAR,
    CommonVideoHead,
    SharedSlotExtractor,
    shared_template_energy,
    square_coordinates,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(703)


def small_model(*, geometry=True):
    return SEAR(
        12,
        15,
        rank=8,
        slots=3,
        templates_per_class=2,
        iterations=2,
        width=16,
        dropout=0.0,
        geometry=geometry,
    )


def inputs(batch=4):
    return torch.randn(batch, 9, 12), torch.randn(batch, 15)


def energy_inputs():
    return (
        torch.randn(2, 3, 5),
        torch.randn(2, 3, 2),
        torch.tensor([[0.2, 0.3, 0.1], [0.0, 0.4, 0.2]]),
        torch.randn(3, 4, 3, 5),
        torch.randn(3, 4, 3, 2),
    )


@pytest.mark.parametrize("geometry", [False, True])
def test_forward_shapes_shared_axes_and_normalization(geometry):
    model = small_model(geometry=geometry).eval()
    result = model(*inputs())
    assert result["logits"].shape == (4, 3)
    assert result["template_energies"].shape == (4, 3, 2)
    assert result["slot_descriptors"].shape == (4, 3, 8)
    assert result["unary_masses"].shape == (4, 3)
    assert result["pairwise_masses"].shape == (4, 3, 3)
    assert result["nuisance_transform"].shape == (4, 2, 2)
    assert result["cost_weights"].shape == (2,)
    assert result["temperature"].ndim == 0
    for tensor in result.values():
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(result["probabilities"].sum(-1), torch.ones(4))
    torch.testing.assert_close(result["coverage"] + result["unmatched_mass"], torch.ones(4))
    assert torch.all(result["coverage"] <= 1)
    torch.testing.assert_close(
        result["pairwise_masses"],
        result["slot_masses"][:, :, None]
        * result["slot_masses"][:, None, :]
        * (1 - torch.eye(3))[None],
    )


@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("mode", ["invalid", "zero_matchability"])
def test_missing_local_evidence_exact_video_fallback_with_nan_padding(geometry, mode):
    model = small_model(geometry=geometry).eval()
    patches, video = inputs()
    valid = torch.ones(4, 9, dtype=torch.bool)
    matchability = torch.ones(4, 9)
    if mode == "invalid":
        valid[:2] = False
        matchability[:2] = torch.nan
    else:
        matchability[:2] = 0
    patches[:2] = torch.nan
    coordinates = square_coordinates(9)[None].repeat(4, 1, 1)
    coordinates[:2] = torch.nan
    result = model(patches, video, valid, coordinates, matchability)
    video_only = model.video(video)
    assert torch.equal(result["logits"][:2], video_only[:2])
    assert torch.equal(result["probabilities"][:2], video_only.softmax(-1)[:2])
    assert torch.equal(result["local_logits"][:2], torch.zeros(2, 3))
    assert torch.equal(result["template_energies"][:2], torch.zeros(2, 3, 2))
    assert torch.equal(result["slot_masses"][:2], torch.zeros(2, 3))
    assert torch.equal(result["nuisance_transform"][:2], torch.eye(2)[None].repeat(2, 1, 1))


@pytest.mark.parametrize("geometry", [False, True])
def test_finite_nonzero_gradients_in_all_active_modules(geometry):
    model = small_model(geometry=geometry)
    result = model(*inputs())
    loss = F.cross_entropy(result["logits"], torch.tensor([0, 1, 2, 0]))
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for group in (model.extractor.projection, model.extractor.update, model.video):
        assert sum(p.grad.abs().sum() for p in group.parameters()) > 0
    assert model.template_descriptors.grad.abs().sum() > 0
    assert model.extractor.unmatched.weight.grad.abs().sum() > 0
    if geometry:
        assert model.template_positions.grad.abs().sum() > 0
        assert sum(p.grad.abs().sum() for p in model.extractor.nuisance.parameters()) > 0


def test_missing_only_backward_never_creates_nan_or_local_gradient():
    model = small_model()
    patches, video = inputs()
    patches[:] = torch.nan
    result = model(patches, video, valid=torch.zeros(4, 9, dtype=torch.bool))
    F.cross_entropy(result["logits"], torch.tensor([0, 1, 2, 0])).backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            if not name.startswith("video."):
                assert torch.count_nonzero(p.grad) == 0, name


def test_templates_cannot_change_shared_evidence_or_other_class_energies():
    model = small_model().eval()
    patches, video = inputs()
    initial = model(patches, video)
    with torch.no_grad():
        model.template_descriptors[1].normal_(mean=15, std=3)
        model.template_positions[1].fill_(-2)
    changed = model(patches, video)
    for key in (
        "slot_descriptors",
        "slot_positions",
        "slot_masses",
        "slot_spread",
        "patch_allocation",
        "pairwise_masses",
        "nuisance_transform",
        "cost_weights",
        "video_logits",
    ):
        assert torch.equal(initial[key], changed[key]), key
    assert torch.equal(
        initial["template_energies"][:, [0, 2]], changed["template_energies"][:, [0, 2]]
    )
    assert not torch.equal(initial["template_energies"][:, 1], changed["template_energies"][:, 1])
    assert not any(
        "visibility" in name or "salience" in name for name, _ in model.named_parameters()
    )


def test_unobserved_slots_have_zero_unary_and_pairwise_evidence():
    descriptor, position, mass, template, layout = energy_inputs()
    mass[:, 1] = 0
    original = shared_template_energy(descriptor, position, mass, template, layout)
    descriptor[:, 1] = torch.nan
    position[:, 1] = torch.nan
    template[:, :, 1] = 1000
    layout[:, :, 1] = -1000
    changed = shared_template_energy(descriptor, position, mass, template, layout)
    assert torch.equal(original["template_energies"], changed["template_energies"])
    assert torch.count_nonzero(changed["pairwise_masses"][:, 1]) == 0
    assert torch.count_nonzero(changed["pairwise_masses"][:, :, 1]) == 0


def test_energy_gradient_to_invisible_templates_is_exactly_zero():
    descriptor, position, mass, template, layout = energy_inputs()
    mass[:, 1] = 0
    template.requires_grad_(True)
    layout.requires_grad_(True)
    result = shared_template_energy(descriptor, position, mass, template, layout)
    result["template_energies"].sum().backward()
    assert torch.count_nonzero(template.grad[:, :, 1]) == 0
    assert torch.count_nonzero(layout.grad[:, :, 1]) == 0


def test_joint_slot_permutation_preserves_energy():
    descriptor, position, mass, template, layout = energy_inputs()
    permutation = torch.tensor([2, 0, 1])
    initial = shared_template_energy(descriptor, position, mass, template, layout)
    reordered = shared_template_energy(
        descriptor[:, permutation],
        position[:, permutation],
        mass[:, permutation],
        template[:, :, permutation],
        layout[:, :, permutation],
    )
    torch.testing.assert_close(initial["template_energies"], reordered["template_energies"])


def test_patch_permutation_with_coordinates_preserves_predictions():
    model = small_model().eval()
    patches, video = inputs()
    coordinates = square_coordinates(9)
    permutation = torch.randperm(9)
    before = model(patches, video, coordinates=coordinates)
    after = model(patches[:, permutation], video, coordinates=coordinates[permutation])
    torch.testing.assert_close(before["logits"], after["logits"], atol=2e-6, rtol=2e-6)


def test_uniform_duplicate_observations_do_not_add_evidence():
    model = small_model().eval()
    patches, video = inputs()
    coordinates = square_coordinates(9)
    initial = model(patches, video, coordinates=coordinates)
    repeated = model(
        patches.repeat_interleave(2, dim=1),
        video,
        coordinates=coordinates.repeat_interleave(2, dim=0),
    )
    for key in ("coverage", "slot_masses", "template_energies", "logits"):
        torch.testing.assert_close(initial[key], repeated[key], atol=3e-6, rtol=3e-6)


def test_invalid_padding_has_no_influence():
    model = small_model().eval()
    patches, video = inputs()
    coordinates = square_coordinates(9)
    initial = model(patches, video, coordinates=coordinates)
    padded = model(
        torch.cat((patches, torch.full((4, 3, 12), torch.nan)), dim=1),
        video,
        valid=torch.tensor([[True] * 9 + [False] * 3]).expand(4, -1),
        coordinates=torch.cat((coordinates, torch.full((3, 2), torch.nan)), dim=0),
    )
    torch.testing.assert_close(initial["logits"], padded["logits"], atol=3e-6, rtol=3e-6)


def test_low_coverage_weakens_evidence_instead_of_renormalizing_it():
    descriptor, position, mass, template, layout = energy_inputs()
    high = shared_template_energy(descriptor, position, mass, template, layout)
    low = shared_template_energy(descriptor, position, mass * 0.01, template, layout)
    torch.testing.assert_close(low["unary_energies"], high["unary_energies"] * 0.01)
    torch.testing.assert_close(low["geometry_energies"], high["geometry_energies"] * 0.0001)
    assert torch.all(low["template_energies"] < high["template_energies"])


def test_physical_patch_occlusion_and_padding_have_different_support_budgets():
    model = small_model().eval()
    patches, video = inputs()
    missing = torch.zeros(4, 9)
    missing[:, 0] = 1
    occluded = model(patches, video, matchability=missing)
    padding = model(patches, video, valid=missing.bool())
    torch.testing.assert_close(occluded["slot_masses"], padding["slot_masses"] / 9)
    torch.testing.assert_close(occluded["unary_energies"], padding["unary_energies"] / 9)
    torch.testing.assert_close(occluded["geometry_energies"], padding["geometry_energies"] / 81)


@pytest.mark.parametrize("prototype_norm", [0.0, 1e-12, 1e-8])
def test_near_zero_prototype_has_no_systematic_reduced_norm_cost_advantage(prototype_norm):
    descriptors = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    positions = torch.zeros(2, 1, 2)
    masses = torch.ones(2, 1)
    prototypes = torch.tensor([[[[1.0, 0.0]]], [[[prototype_norm, 0.0]]]], requires_grad=True)
    result = shared_template_energy(
        descriptors, positions, masses, prototypes, torch.zeros(2, 1, 1, 2)
    )
    torch.testing.assert_close(result["unary_energies"].mean(0), torch.full((2, 1), 2.0))
    result["template_energies"].sum().backward()
    assert torch.isfinite(prototypes.grad).all()


def test_relative_geometry_distinguishes_equal_appearance_layouts():
    descriptor = torch.ones(1, 2, 3)
    positions = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
    mass = torch.full((1, 2), 0.5)
    template = torch.ones(3, 1, 2, 3)
    layout = positions.repeat(3, 1, 1)[:, None]
    layout[1] = layout[1].flip(-2)
    original = shared_template_energy(descriptor, positions, mass, template, layout)
    assert original["template_energies"][0, 0, 0] < original["template_energies"][0, 1, 0]
    off = shared_template_energy(descriptor, positions, mass, template, layout, geometry_weight=0)
    torch.testing.assert_close(off["template_energies"], torch.zeros(1, 3, 1), atol=1e-6, rtol=0)
    translated = shared_template_energy(descriptor, positions + 0.31, mass, template, layout)
    torch.testing.assert_close(original["template_energies"], translated["template_energies"])


def test_geometry_off_never_reads_layout_values_into_logits():
    model = small_model(geometry=False).eval()
    patches, video = inputs()
    initial = model(patches, video)
    with torch.no_grad():
        model.template_positions.fill_(500)
        model.extractor.nuisance[-1].bias.fill_(3)
    changed = model(patches, video)
    assert torch.equal(initial["logits"], changed["logits"])


def test_class_template_permutation_only_permutes_local_logits():
    model = small_model().eval()
    patches, video = inputs()
    first = model(patches, video)
    changed = copy.deepcopy(model)
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        changed.template_descriptors.copy_(model.template_descriptors[permutation])
        changed.template_positions.copy_(model.template_positions[permutation])
    second = changed(patches, video)
    assert torch.equal(first["local_logits"][:, permutation], second["local_logits"])
    assert torch.equal(first["video_logits"], second["video_logits"])


def test_nuisance_budget_is_bounded_and_common():
    model = small_model().eval()
    with torch.no_grad():
        model.extractor.nuisance[-1].bias.copy_(torch.tensor([100.0, -100.0]))
    result = model(*inputs())
    assert torch.all(result["nuisance_angle"].abs() <= torch.pi / 6 + 1e-7)
    assert torch.all(result["nuisance_scale"] >= 1 / 1.25 - 1e-7)
    assert torch.all(result["nuisance_scale"] <= 1.25 + 1e-7)
    assert result["nuisance_transform"].shape == (4, 2, 2)


def test_729_patch_default_contract_and_parameter_counts():
    model = SEAR().eval()
    off = SEAR(geometry=False)
    for candidate in (model, off):
        counts = candidate.parameter_counts()
        assert counts["total"] == sum(value for key, value in counts.items() if key != "total")
        assert 500000 <= counts["total"] < 1_000_000
    assert abs(model.trainable_parameters / off.trainable_parameters - 1) < 0.05
    result = model(torch.randn(1, 729, 768), torch.randn(1, 1536))
    assert result["patch_allocation"].shape == (1, 729, 7)
    assert torch.isfinite(result["logits"]).all()


def test_rejects_class_specific_matchability_and_invalid_active_evidence():
    model = small_model()
    patches, video = inputs()
    with pytest.raises(ValueError, match="class-independent"):
        model(patches, video, matchability=torch.ones(4, 3, 9))
    patches[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="Available patches"):
        model(patches, video)
    with pytest.raises(ValueError, match="Masses must have only"):
        values = list(energy_inputs())
        values[2] = torch.ones(2, 3, 3)
        shared_template_energy(*values)


def test_rejects_class_specific_cost_weights():
    with pytest.raises(ValueError, match="shared finite"):
        shared_template_energy(*energy_inputs(), appearance_weight=torch.ones(3))


@pytest.mark.parametrize(
    "constructor,kwargs",
    [
        (SEAR, {"templates_per_class": 0}),
        (SEAR, {"parameter_limit": 10}),
        (SEAR, {"geometry": "class_specific"}),
        (SharedSlotExtractor, {"iterations": 0}),
        (CommonVideoHead, {"input_dim": 0}),
    ],
)
def test_invalid_configuration_is_rejected(constructor, kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)
