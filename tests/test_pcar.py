"""Contract tests for the PCAR-v5 bounded residual primitives."""

import numpy as np
import pytest
import torch

from hac.pcar import (
    PCARPolicy,
    action_costs,
    center_balanced_expected_cost,
    exact_convex_actions,
    exact_convex_actions_numpy,
    hard_actions,
)


def test_endpoints_are_arithmetic_free_and_unavailable_retains_p2() -> None:
    p2 = torch.tensor([[1.0, -0.0, 3.25] + [0.0] * 765], dtype=torch.float32)
    affine = torch.tensor([[9.0, 2.0, -7.5] + [1.0] * 765], dtype=torch.float32)
    available = torch.tensor([True])
    actions = exact_convex_actions(p2, affine, available=available)
    assert actions.shape == (1, 5, 768)
    assert actions[:, 0].numpy().tobytes() == p2.numpy().tobytes()
    assert actions[:, -1].numpy().tobytes() == affine.numpy().tobytes()
    unavailable = exact_convex_actions(p2, affine, available=torch.tensor([False]))
    for action in unavailable[0]:
        assert action.numpy().tobytes() == p2.numpy().tobytes()


def test_numpy_and_torch_actions_agree_and_interiors_are_bounded() -> None:
    rng = np.random.default_rng(42)
    p2 = rng.normal(size=(3, 768)).astype(np.float32)
    affine = rng.normal(size=(3, 768)).astype(np.float32)
    available = np.array([True, False, True])
    numpy_actions = exact_convex_actions_numpy(p2, affine, available=available)
    torch_actions = exact_convex_actions(
        torch.from_numpy(p2), torch.from_numpy(affine), available=torch.from_numpy(available)
    ).numpy()
    np.testing.assert_array_equal(numpy_actions, torch_actions)
    np.testing.assert_allclose(numpy_actions[0, 2], p2[0] + 0.5 * (affine[0] - p2[0]))


def test_policy_objective_is_center_balanced() -> None:
    logits = torch.zeros((2, 3, 5), dtype=torch.float32)
    costs = torch.zeros_like(logits)
    costs[0] = 1.0
    costs[1] = 3.0
    valid = torch.tensor([[True, True, False], [True, False, False]])
    assert center_balanced_expected_cost(logits, costs, valid).item() == pytest.approx(2.0)


def test_hard_actions_ties_and_missingness_choose_retain() -> None:
    logits = torch.zeros((1, 4, 5), dtype=torch.float32)
    logits[0, 1, 3] = 2.0
    logits[0, 2, 4] = 2.0
    valid = torch.tensor([[True, True, False, True]])
    available = torch.tensor([[True, True, True, False]])
    assert hard_actions(logits, valid, available=available).tolist() == [[0, 3, 0, 0]]


def test_policy_and_action_cost_shapes_are_locked() -> None:
    policy = PCARPolicy()
    features = torch.zeros((2, 4, 32), dtype=torch.float32)
    assert policy(features).shape == (2, 4, 5)
    candidates = torch.ones((2, 4, 5, 768), dtype=torch.float32)
    teacher = torch.ones((2, 4, 768), dtype=torch.float32)
    assert action_costs(candidates, teacher).shape == (2, 4, 5)
    with pytest.raises(ValueError):
        policy(torch.zeros((2, 4, 31)))

