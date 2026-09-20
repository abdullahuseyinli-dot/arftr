from __future__ import annotations

import numpy as np

from hac.posture_witness import (
    DEAD_ZONE,
    PostureHead,
    apply_posture_policy,
    apply_projection,
    fit_posture_head,
    fit_projection,
)


def _anchor(rows: int, dtype=np.float64):
    base = np.asarray([[0.2, 0.5, 0.3], [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=dtype)
    return np.resize(base, (rows, 3)).astype(dtype)


def test_exact_retain_and_motion_conditional_share():
    anchor = _anchor(4, np.float32)
    features = np.asarray([[0.0], [1.0], [-1.0], [0.0]])
    head = PostureHead(np.asarray([1.0]), 0.0, 2.0, DEAD_ZONE, 0.01, {})
    result, intervene, _ = apply_posture_policy(anchor, features, head, available=np.asarray([True, True, True, False]))
    assert np.array_equal(result[~intervene], anchor[~intervene])
    old_share = anchor[:, 2] / (anchor[:, 1] + anchor[:, 2])
    new_share = result[:, 2] / (result[:, 1] + result[:, 2])
    assert np.allclose(old_share[intervene], new_share[intervene], atol=1e-7, rtol=0)


def test_zero_upright_mass_is_an_exact_retain_observation_failure():
    anchor = np.asarray([[1.0, 0.0, 0.0], [0.2, 0.5, 0.3]], dtype=np.float32)
    features = np.ones((2, 1))
    head = PostureHead(np.asarray([1.0]), 1.0, 2.0, DEAD_ZONE, 0.01, {})
    result, intervene, _ = apply_posture_policy(anchor, features, head)
    assert not intervene[0]
    assert np.array_equal(result[0], anchor[0])
    assert intervene[1]


def test_projection_is_train_only_deterministic_and_fixed_width():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(70, 60))
    held = rng.normal(size=(9, 60))
    first = fit_projection(train)
    second = fit_projection(train)
    assert np.array_equal(first.components, second.components)
    assert apply_projection(held, first).shape == (9, 48)
    assert first.receipt()["output_dim"] == 48


def test_continuous_training_can_leave_zero_initialization():
    rng = np.random.default_rng(12)
    rows = 90
    features = rng.normal(size=(rows, 2))
    labels = np.resize(np.arange(3), rows)
    labels[(features[:, 0] > 0.4)] = 0
    head = fit_posture_head(features, labels, _anchor(rows))
    assert head.optimizer_receipt["success"]
    assert head.optimizer_receipt["dead_zone_applied_during_training"] is False
    assert np.linalg.norm(head.coefficients) + abs(head.intercept) > 0
    assert len(head.coefficients) + 1 <= 49
