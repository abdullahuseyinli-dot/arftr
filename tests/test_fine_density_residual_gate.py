from __future__ import annotations

import numpy as np

from hac.fine_density_residual_gate import (
    ACTION_NAMES,
    FineDensityResidualGate,
    build_actions,
    build_features,
)


def _probs(rows: int = 18, seed: int = 4):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.1, 1.0, size=(rows, 3))
    return values / values.sum(1, keepdims=True)


def test_actions_have_retain_anchor_and_bounded_simplex():
    anchor, fine, coarse = _probs(), _probs(18, 5), _probs(18, 6)
    actions = build_actions(anchor, fine, coarse)
    assert actions.shape == (18, 4, 3)
    assert np.array_equal(actions[:, 0], anchor)
    assert np.allclose(actions.sum(2), 1.0)
    assert ACTION_NAMES[0] == "retain_ARFTR"


def test_features_are_finite_and_gate_can_retain():
    anchor, fine, coarse = _probs(), _probs(18, 5), _probs(18, 6)
    quality = np.zeros((18, 6), dtype=np.float64)
    times = np.arange(8, dtype=np.float64)[None, :] + np.zeros((18, 1))
    labels = np.arange(18) % 3
    features = build_features(anchor, fine, coarse, quality=quality, times=times)
    actions = build_actions(anchor, fine, coarse)
    gate = FineDensityResidualGate.fit(features, anchor, actions, labels)
    output, choices = gate.apply(features, actions)
    assert features.shape[0] == 18
    assert np.isfinite(features).all()
    assert output.shape == (18, 3)
    assert np.allclose(output.sum(1), 1.0)
    assert np.all((choices >= 0) & (choices < 4))

