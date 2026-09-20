"""Synthetic, label-free tests for the center-completion reconstruction screen."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from experiments.screen_okutama_center_evidence_completion import (
    build_partial_transport,
    repeated_center_transport,
)
from hac.center_completion_screen import (
    ARMS,
    FEATURE_DIM,
    BaseCompletionModel,
    PackedScreenData,
    build_gate_features_21,
    fit_outer_fold,
    matched_parameter_counts,
    new_screen_model,
    outer_scenario_splits,
    pack_dense_arm,
    reconstruction_loss,
    repeated_masked_center_control,
    spatial_reassignment_control,
    time_reversal_control,
    wrong_track_control,
)
from hac.okutama_native_video import EXPECTED_SCENARIOS


def packed_data(centers: int = 3, targets: int = 4, *, p3: bool = True) -> PackedScreenData:
    generator = torch.Generator().manual_seed(91)
    shape = (centers, targets)
    masked = torch.randn((*shape, FEATURE_DIM), generator=generator)
    teacher = torch.randn((*shape, FEATURE_DIM), generator=generator)
    evidence = torch.randn((*shape, FEATURE_DIM), generator=generator)
    positions = torch.zeros((*shape, 2), dtype=torch.int64)
    positions[..., 0] = torch.arange(targets) % 27
    positions[..., 1] = (torch.arange(targets) * 3) % 27
    masks = (torch.arange(targets)[None] % 2).expand(centers, -1).clone()
    valid = torch.ones(shape, dtype=torch.bool)
    available = valid.clone()
    temporal = (
        torch.tensor([-1.0, -0.5, 0.5, 0.875, 1, 1, 1, 1]).expand(centers, targets, -1).clone()
    )
    gate = torch.randn((*shape, 21), generator=generator) if p3 else None
    return PackedScreenData(
        masked,
        masked.mean(dim=1, keepdim=True).expand_as(masked).clone(),
        teacher,
        positions,
        masks,
        valid,
        evidence,
        available,
        temporal,
        gate,
    )


def test_fixed_projection_and_active_capacity_are_locked() -> None:
    left = new_screen_model("P0_center_only")
    right = new_screen_model("P0_center_only")
    assert isinstance(left, BaseCompletionModel)
    torch.testing.assert_close(left.fixed.projection, right.fixed.projection, atol=0, rtol=0)
    identity = left.fixed.projection.T @ left.fixed.projection
    torch.testing.assert_close(identity, torch.eye(128), atol=2e-6, rtol=2e-6)
    counts = matched_parameter_counts()
    assert counts == {
        "P0_center_only": 395776,
        "P1_unordered_pool": 395776,
        "P2_same_grid_pool": 395776,
        "P3_partial_transport": 396555,
        "P1_P3_relative_range": pytest.approx(0.0019682850905562745),
    }
    assert counts["P1_P3_relative_range"] < 0.10


def test_gate_features_have_exact_inference_only_contract() -> None:
    tokens = torch.zeros((2, 4, FEATURE_DIM))
    tokens[:, :, :4] = torch.eye(4)[None]
    observed = torch.tensor([[True, True, False, False], [True, True, True, True]])
    residual = torch.tensor([[1.0, 3.0, float("inf"), float("inf")], [0.0, 1.0, 2.0, 3.0]])
    cosine = torch.tensor([[0.8, 0.6, float("nan"), float("nan")], [1.2, 0.5, 0.0, -2.0]])
    distance = torch.tensor([[2.0, 8.0, float("inf"), float("inf")], [0.0, 1.0, 2.0, 8.0]])
    features = build_gate_features_21(tokens, observed, residual, cosine, distance)
    assert features.shape == (2, 21)
    assert not features.requires_grad
    assert features[0, 0] == 0.5
    torch.testing.assert_close(features[0, 1:9], torch.tensor([1, 0.5, 0.8, 0.5, 1, 1, 0.6, 1]))
    assert features[0, 17] == pytest.approx(1.0)
    assert features[0, 18] == pytest.approx(1.0)
    assert features[0, 19] == 1.0 and features[0, 20] == 0.0
    assert torch.isfinite(features).all()


def test_p3_no_donor_is_bit_exact_p0_retain() -> None:
    data = packed_data()
    unavailable = copy.copy(data)
    object.__setattr__(unavailable, "evidence_available", torch.zeros_like(data.target_valid))
    base = new_screen_model("P0_center_only")
    model = new_screen_model("P3_partial_transport", base=base)
    output = model(unavailable)
    assert torch.equal(output.prediction, output.base_prediction)
    assert torch.count_nonzero(output.gate) == 0


def test_teacher_is_detached_and_gate_sees_only_twenty_one_features() -> None:
    data = packed_data()
    teacher = data.teacher_tokens.clone().requires_grad_(True)
    data = copy.copy(data)
    object.__setattr__(data, "teacher_tokens", teacher)
    base = new_screen_model("P0_center_only")
    model = new_screen_model("P3_partial_transport", base=base)
    assert model.gate[1].in_features == 21
    output = model(data)
    reconstruction_loss(output, teacher, data.target_valid).objective.backward()
    assert teacher.grad is None
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_outer_splits_are_the_five_locked_scenario_folds() -> None:
    scenarios = np.array(sorted(EXPECTED_SCENARIOS))
    splits = outer_scenario_splits(scenarios)
    assert len(splits) == 5
    assert sorted(np.concatenate([held for _, held in splits]).tolist()) == list(
        range(len(scenarios))
    )
    for fold, (train, held) in enumerate(splits):
        assert {
            int(EXPECTED_SCENARIOS[scenario][0].split("-")[1]) for scenario in scenarios[held]
        } == {fold}
        assert not set(scenarios[train]) & set(scenarios[held])
    with pytest.raises(ValueError, match="11 locked scenarios"):
        outer_scenario_splits(scenarios[:-1])


def test_fit_never_reads_held_teacher_and_enforces_update_budget() -> None:
    scenarios = np.array(sorted(EXPECTED_SCENARIOS))
    clean = packed_data(len(scenarios), targets=1, p3=False)
    held = outer_scenario_splits(scenarios)[0][1]
    poisoned = copy.copy(clean)
    poison_teacher = clean.teacher_tokens.clone()
    poison_teacher[held] = torch.nan
    object.__setattr__(poisoned, "teacher_tokens", poison_teacher)
    first = fit_outer_fold(clean, scenarios, 0, arm="P0_center_only", updates=1)
    second = fit_outer_fold(poisoned, scenarios, 0, arm="P0_center_only", updates=1)
    assert first.training_history == second.training_history
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="at most 400"):
        fit_outer_fold(clean, scenarios, 0, arm="P0_center_only", updates=401)


def test_controls_change_only_the_declared_p3_view_without_refitting() -> None:
    data = packed_data(2, 6)
    repeated_gate = torch.zeros_like(data.gate_features)
    repeated_gate[..., 0] = 1
    repeated_gate[..., 1] = 1
    repeated_gate[..., 5] = 1
    repeated_gate[..., 9] = 1
    repeated_gate[..., 13] = 1
    repeated = repeated_masked_center_control(
        data,
        repeated_gate_features=repeated_gate,
        repeated_available=torch.ones_like(data.evidence_available),
    )
    assert torch.equal(repeated.evidence_tokens, data.masked_tokens)
    assert torch.count_nonzero(repeated.gate_features[..., 17:19]) == 0
    assert torch.equal(repeated.gate_features, repeated_gate)
    reversed_data = time_reversal_control(data)
    assert torch.equal(reversed_data.evidence_tokens, data.evidence_tokens)
    assert torch.equal(
        reversed_data.temporal_metadata[..., :4], torch.flip(data.temporal_metadata[..., :4], (-1,))
    )
    reassigned_a = spatial_reassignment_control(data, ["sample-a", "sample-b"])
    reassigned_b = spatial_reassignment_control(data, ["sample-a", "sample-b"])
    assert torch.equal(reassigned_a.evidence_tokens, reassigned_b.evidence_tokens)
    assert not torch.equal(reassigned_a.evidence_tokens, data.evidence_tokens)
    assert torch.equal(reassigned_a.teacher_tokens, data.teacher_tokens)
    wrong = packed_data(2, 6)
    substituted = wrong_track_control(data, wrong)
    assert torch.equal(substituted.evidence_tokens, wrong.evidence_tokens)
    assert torch.equal(substituted.teacher_tokens, data.teacher_tokens)
    changed = copy.copy(wrong)
    object.__setattr__(changed, "teacher_tokens", wrong.teacher_tokens + 1)
    with pytest.raises(ValueError, match="target population"):
        wrong_track_control(data, changed)


@pytest.mark.parametrize("arm", ARMS)
def test_dense_packing_preserves_row_major_targets_and_arm_evidence(arm) -> None:
    masked = torch.zeros((1, 2, 27, 27, FEATURE_DIM), dtype=torch.float16)
    teacher = torch.zeros((1, 27, 27, FEATURE_DIM), dtype=torch.float16)
    neighbors = torch.zeros((1, 4, 27, 27, FEATURE_DIM), dtype=torch.float16)
    neighbor_valid = torch.zeros((1, 4, 27, 27), dtype=torch.bool)
    targets = torch.zeros((1, 2, 27, 27), dtype=torch.bool)
    visible = torch.zeros_like(targets)
    targets[0, 0, 2, 3] = True
    targets[0, 1, 7, 5] = True
    visible[0, :, 0, 0] = True
    masked[0, :, 0, 0] = 2
    teacher[0, 2, 3] = 3
    teacher[0, 7, 5] = 4
    neighbors[0, :, 2, 3] = torch.arange(1, 5)[:, None]
    neighbors[0, :, 7, 5] = torch.arange(5, 9)[:, None]
    neighbor_valid[0, :, 2, 3] = True
    neighbor_valid[0, :, 7, 5] = True
    kwargs = {}
    if arm == "P3_partial_transport":
        transport = torch.full((1, 2, 27, 27, FEATURE_DIM), 9.0, dtype=torch.float16)
        available = targets.clone()
        gates = torch.zeros((1, 2, 27, 27, 21))
        gates[..., 0] = 1
        gates[..., 1] = 1
        gates[..., 5] = 1
        gates[..., 9] = 1
        gates[..., 13] = 1
        kwargs = {
            "transport_tokens": transport,
            "transport_available": available,
            "gate_features": gates,
        }
    packed = pack_dense_arm(
        arm=arm,
        masked_tokens=masked,
        teacher_tokens=teacher,
        neighbor_tokens=neighbors,
        neighbor_valid=neighbor_valid,
        target_masks=targets,
        visible_masks=visible,
        **kwargs,
    )
    assert packed.positions_yx.tolist() == [[[2, 3], [7, 5]]]
    assert packed.mask_ids.tolist() == [[0, 1]]
    assert packed.teacher_tokens[0, :, 0].tolist() == [3, 4]
    if arm == "P0_center_only":
        assert torch.count_nonzero(packed.evidence_tokens) == 0
    elif arm == "P1_unordered_pool":
        assert packed.evidence_tokens[0, :, 0].tolist() == [4.5, 4.5]
    elif arm == "P2_same_grid_pool":
        assert packed.evidence_tokens[0, :, 0].tolist() == [2.5, 6.5]
    else:
        assert packed.evidence_tokens[0, :, 0].tolist() == [9, 9]
    base = new_screen_model("P0_center_only")
    model = base if arm == "P0_center_only" else new_screen_model(arm, base=base)
    output = model(packed)
    assert output.prediction.shape == (1, 2, FEATURE_DIM)
    assert output.prediction.dtype == torch.float32
    assert torch.isfinite(output.prediction).all()


def test_data_refuses_wrong_gate_width_and_empty_center() -> None:
    data = packed_data()
    bad = copy.copy(data)
    object.__setattr__(bad, "gate_features", torch.zeros((3, 4, 20)))
    with pytest.raises(ValueError, match="exactly 21"):
        bad.validate(require_gate=True)
    empty = copy.copy(data)
    object.__setattr__(empty, "target_valid", torch.zeros_like(data.target_valid))
    with pytest.raises(ValueError, match="at least one"):
        empty.validate()


def test_device_transfer_promotes_float16_cache_fields_for_float32_decoder() -> None:
    data = packed_data()
    half = PackedScreenData(
        **{
            name: value.half() if value is not None and value.is_floating_point() else value
            for name, value in data.__dict__.items()
        }
    )
    promoted = half.to("cpu")
    for name in (
        "masked_tokens",
        "visible_context",
        "teacher_tokens",
        "evidence_tokens",
        "temporal_metadata",
        "gate_features",
    ):
        assert getattr(promoted, name).dtype == torch.float32
    output = new_screen_model("P3_partial_transport", base=new_screen_model("P0_center_only"))(
        promoted
    )
    assert output.prediction.dtype == torch.float32


def test_real_transport_builder_contract_uses_only_visible_anchors() -> None:
    generator = np.random.default_rng(109)
    masked = np.zeros((1, 2, 27, 27, FEATURE_DIM), dtype=np.float32)
    targets = np.zeros((1, 2, 27, 27), dtype=bool)
    visible = np.zeros_like(targets)
    anchors = [(row, column) for row in range(2, 6) for column in range(2, 6)]
    base = np.zeros((27, 27, FEATURE_DIM), dtype=np.float32)
    for row, column in anchors:
        base[row, column] = generator.normal(size=FEATURE_DIM)
    for mask_id in range(2):
        masked[0, mask_id] = base
        targets[0, mask_id, 8 + mask_id, 8] = True
        masked[0, mask_id, 8 + mask_id, 8, mask_id] = 7
        visible[0, mask_id, 2:6, 2:6] = True
    donors = np.repeat(base[None, None], 4, axis=1)
    donors[0, :, 8, 8, 0] = 7
    donors[0, :, 9, 8, 1] = 7
    donor_valid = np.ones((1, 4, 27, 27), dtype=bool)
    built = build_partial_transport(masked, donors, donor_valid, targets, visible)
    assert built.available[targets].all()
    assert built.gate_features[targets][:, 0].tolist() == [1.0, 1.0]
    assert np.isfinite(built.gate_features).all()
    assert built.diagnostics["affine_valid"] == 8
    repeated = repeated_center_transport(masked, targets, visible)
    assert repeated.available[targets].all()
    np.testing.assert_array_equal(repeated.tokens[targets], masked[targets].astype(np.float16))
    assert np.count_nonzero(repeated.gate_features[targets][:, 17:19]) == 0
