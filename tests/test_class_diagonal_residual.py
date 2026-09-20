from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hac.actor_memory_base import probability_metrics
from hac.class_diagonal_residual import (
    apply_class_diagonal_residual,
    fit_class_diagonal_residual,
    sitting_disagreement_protection,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sitting_disagreement_protection_is_bidirectional():
    m4 = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
    )
    a3 = np.array(
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.1, 0.8, 0.1]]
    )
    assert sitting_disagreement_protection(m4, a3).tolist() == [True, True, False, False]


def test_protected_residual_preserves_exact_anchor_bytes():
    m4 = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
    a3 = np.array([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.2, 0.7, 0.1]])
    protected = np.array([True, False, True])
    result = apply_class_diagonal_residual(
        m4, a3, np.array([0.25, 0.1, 0.2]), protected=protected, epsilon=1e-12
    )
    assert np.array_equal(result[protected], m4[protected])
    assert not np.array_equal(result[~protected], m4[~protected])
    assert np.allclose(result.sum(1), 1.0)


def test_zero_coefficients_replay_m4_everywhere():
    rng = np.random.default_rng(7)
    m4 = rng.dirichlet(np.ones(3), 20)
    a3 = rng.dirichlet(np.ones(3), 20)
    result = apply_class_diagonal_residual(
        m4, a3, np.zeros(3), protected=np.zeros(20, bool), epsilon=1e-12
    )
    assert np.array_equal(result, m4)


def test_fit_uses_exactly_three_bounded_nonnegative_coefficients_and_nll_objective():
    rng = np.random.default_rng(11)
    labels = np.tile(np.arange(3), 100)
    target = np.eye(3)[labels]
    m4 = 0.55 * target + 0.45 / 3
    a3 = 0.85 * target + 0.15 / 3
    m4 = 0.99 * m4 + 0.01 * rng.dirichlet(np.ones(3), len(labels))
    a3 = 0.99 * a3 + 0.01 * rng.dirichlet(np.ones(3), len(labels))
    m4 /= m4.sum(1, keepdims=True)
    a3 /= a3.sum(1, keepdims=True)
    fit = fit_class_diagonal_residual(
        m4,
        a3,
        labels,
        protected=np.zeros(len(labels), bool),
        coefficient_cap=0.25,
        l2=0.1,
        epsilon=1e-12,
        maximum_iterations=1000,
        tolerance=1e-12,
    )
    assert fit.coefficients.shape == (3,)
    assert np.all((fit.coefficients >= 0) & (fit.coefficients <= 0.25))
    assert fit.training_nll < fit.anchor_nll
    assert fit.success


def test_locked_protocol_forbids_boundary_scenario_and_correctness_features():
    protocol = json.loads(
        (ROOT / "experiments/okutama_class_diagonal_residual_protocol.json").read_text()
    )
    assert protocol["optimization"]["coefficients"] == 3
    assert protocol["optimization"]["coefficient_cap"] == 0.25
    assert protocol["optimization"]["l2"] == 0.1
    assert protocol["sitting_protection"]["applies_during_training_and_inference"] is True
    forbidden = " ".join(protocol["inference_contract"]["forbidden"])
    assert "scenario" in forbidden
    assert "boundary" in forbidden
    assert "correctness" in forbidden
    assert protocol["statistics"]["exact_scenario_swap_test"]["random_sampling"] is False
    assert protocol["statistics"]["exact_scenario_swap_test"]["significance_alpha"] == 0.05


def test_probability_metrics_anchor_sanity():
    labels = np.array([0, 1, 2])
    p = np.eye(3)
    assert probability_metrics(labels, p)["macro_f1"] == 1.0
