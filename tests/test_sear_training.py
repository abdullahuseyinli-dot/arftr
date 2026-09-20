import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from hac.sear import square_coordinates
from hac.sear_training import (
    ARMS,
    TEMPLATE_ARMS,
    forward_arm,
    make_model,
    parameter_counts,
    seed_training,
    slot_regularization,
    training_loss,
    video_module,
)

PROTOCOL = json.loads(
    (
        Path(__file__).resolve().parents[1] / "experiments/okutama_sear_matrix_protocol.json"
    ).read_text()
)


@pytest.mark.parametrize("arm", ARMS)
def test_factory_forward_loss_and_gradients(arm):
    torch.manual_seed(9)
    model = make_model(PROTOCOL, arm)
    patches = torch.randn(3, 9, 768)
    center = torch.randn(3, 768)
    video = torch.randn(3, 1536)
    output = forward_arm(model, arm, patches, center, video, torch.ones(3, dtype=torch.bool))
    loss, pieces = training_loss(output, torch.tensor([0, 1, 2]), torch.ones(3), arm, PROTOCOL)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(pieces) == (
        {"classification", "compactness", "batch_slot_balance", "weighted"}
        if arm in TEMPLATE_ARMS
        else {"classification"}
    )
    for name, value in model.named_parameters():
        if value.requires_grad:
            assert value.grad is not None, name
            assert torch.isfinite(value.grad).all(), name


def test_locked_parameter_counts_and_common_video_schema():
    counts = parameter_counts(PROTOCOL)
    assert counts == {
        "a0_center_cls": 563462,
        "a1_dense_cnn": 551266,
        "a2_dense_transformer": 548066,
        "a3_unrestricted_templates": 552508,
        "a4_shared_no_geometry": 549171,
        "a5_sear": 551558,
    }
    references = None
    for arm in ARMS:
        state = video_module(make_model(PROTOCOL, arm), arm).state_dict()
        schema = {name: tuple(value.shape) for name, value in state.items()}
        references = schema if references is None else references
        assert schema == references


@pytest.mark.parametrize("arm", ARMS)
def test_all_local_missing_is_exact_video_fallback(arm):
    model = make_model(PROTOCOL, arm).eval()
    patches = torch.full((2, 9, 768), float("nan"))
    center = torch.full((2, 768), float("nan"))
    video = torch.randn(2, 1536)
    output = forward_arm(model, arm, patches, center, video, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(output["logits"], output["video_logits"])
    assert torch.equal(output["local_logits"], torch.zeros_like(output["local_logits"]))


def test_arm_inventory_change_is_refused():
    changed = {**PROTOCOL, "arms": dict(reversed(PROTOCOL["arms"].items()))}
    with pytest.raises(ValueError, match="inventory"):
        make_model(changed, "a5_sear")


@pytest.mark.parametrize("arm", ARMS[1:])
def test_dense_arms_receive_identical_explicit_patch_center_coordinates(arm):
    model = make_model(PROTOCOL, arm).eval()
    seen = []

    def capture(_module, _args, kwargs):
        seen.append(kwargs["coordinates"].detach().clone())

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        forward_arm(
            model,
            arm,
            torch.randn(2, 9, 768, dtype=torch.float16),
            torch.randn(2, 768, dtype=torch.float16),
            torch.randn(2, 1536),
            torch.ones(2, dtype=torch.bool),
        )
    finally:
        handle.remove()
    assert len(seen) == 1
    assert seen[0].dtype == torch.float32
    assert torch.equal(seen[0], square_coordinates(9))
    assert torch.allclose(seen[0][0], torch.tensor([-2 / 3, -2 / 3]))
    assert seen[0][1, 0] > seen[0][0, 0]  # Row-major x changes first.
    assert seen[0][1, 1] == seen[0][0, 1]


def test_transformer_adapter_equals_explicit_center_grid_evaluation():
    model = make_model(PROTOCOL, "a2_dense_transformer").eval()
    patches, center, video = torch.randn(2, 9, 768), torch.randn(2, 768), torch.randn(2, 1536)
    local_valid = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        actual = forward_arm(model, "a2_dense_transformer", patches, center, video, local_valid)
        explicit = model(
            patches,
            video,
            valid=local_valid[:, None].expand(-1, 9),
            coordinates=square_coordinates(9),
        )
    assert torch.equal(actual["logits"], explicit["logits"])


def test_seed_training_repeats_rngs_and_disables_tf32_without_changing_cublas_environment():
    cublas_environment = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    try:
        seed_training(17)
        first = random.random(), np.random.random(4), torch.rand(4)
        seed_training(17)
        second = random.random(), np.random.random(4), torch.rand(4)
        assert first[0] == second[0]
        assert np.array_equal(first[1], second[1])
        assert torch.equal(first[2], second[2])
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
        # Torch may lazily initialize its own cache-directory environment, but
        # the SEAR helper must leave CUBLAS setup entirely to the entry point.
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == cublas_environment
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]
        torch.set_float32_matmul_precision(previous[5])


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("patch_allocation", (2, 9)),
        ("patch_allocation", (2, 9, 6, 2)),
        ("patch_allocation", (0, 9, 7)),
        ("patch_allocation", (2, 0, 7)),
        ("patch_allocation", (2, 9, 1)),
        ("slot_positions", (2, 6, 3)),
        ("slot_positions", (2, 3, 6, 2)),
        ("slot_masses", (2, 5)),
        ("slot_masses", (2, 3, 6)),
    ],
)
def test_slot_regularization_rejects_malformed_shapes(field, shape):
    outputs = {
        "patch_allocation": torch.zeros(2, 9, 7),
        "slot_positions": torch.zeros(2, 6, 2),
        "slot_masses": torch.zeros(2, 6),
    }
    outputs[field] = torch.zeros(shape)
    with pytest.raises(ValueError, match="Patch allocation|Slot allocation"):
        slot_regularization(outputs, PROTOCOL)


def test_regularization_uses_patch_centers_and_is_slot_permutation_invariant():
    allocation = torch.cat((torch.eye(4), torch.zeros(4, 1)), dim=-1)[None] / 4
    outputs = {
        "patch_allocation": allocation,
        "slot_positions": square_coordinates(4)[None],
        "slot_masses": allocation[..., :-1].sum(1),
    }
    expected = slot_regularization(outputs, PROTOCOL)
    assert expected["compactness"].item() == 0
    assert expected["batch_slot_balance"].item() == 0
    outputs["slot_positions"] = outputs["slot_positions"] + 0.25
    nonzero = slot_regularization(outputs, PROTOCOL)
    assert nonzero["compactness"].item() == pytest.approx(0.125)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = {
        "patch_allocation": torch.cat((allocation[..., permutation], allocation[..., -1:]), dim=-1),
        "slot_positions": outputs["slot_positions"][:, permutation],
        "slot_masses": outputs["slot_masses"][:, permutation],
    }
    for key, value in slot_regularization(permuted, PROTOCOL).items():
        assert torch.equal(value, nonzero[key])


def test_all_missing_regularization_is_zero_with_finite_gradients():
    outputs = {
        "patch_allocation": torch.zeros(2, 9, 7, requires_grad=True),
        "slot_positions": torch.zeros(2, 6, 2, requires_grad=True),
        "slot_masses": torch.zeros(2, 6, requires_grad=True),
    }
    losses = slot_regularization(outputs, PROTOCOL)
    assert all(value.item() == 0 for value in losses.values())
    losses["weighted"].backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all() for value in outputs.values()
    )
