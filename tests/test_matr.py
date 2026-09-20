from __future__ import annotations

import numpy as np

from hac.matr import (
    boundary_context_features,
    fit_correction_gate,
    geometric_residual,
    select_boundary_threshold,
    smooth_gate,
    uncertainty_features,
)


def test_geometric_residual_preserves_exact_anchor_on_zero_gate():
    m4 = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], dtype=np.float64)
    a3 = np.array([[0.4, 0.3, 0.3], [0.1, 0.2, 0.7]], dtype=np.float64)
    output = geometric_residual(m4, a3, np.zeros(2), epsilon=1e-12)
    assert np.array_equal(output, m4)


def test_geometric_residual_full_gate_reaches_residual_expert():
    m4 = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], dtype=np.float64)
    a3 = np.array([[0.4, 0.3, 0.3], [0.1, 0.2, 0.7]], dtype=np.float64)
    output = geometric_residual(m4, a3, np.ones(2), epsilon=1e-12)
    assert np.allclose(output, a3, atol=1e-15)


def test_smooth_gate_requires_disagreement_and_honors_suppression():
    probability = np.array([0.4, 0.6, 0.8, 0.9])
    disagreement = np.array([True, True, False, True])
    suppress = np.array([False, False, False, True])
    gate = smooth_gate(
        probability,
        disagreement,
        activation_probability=0.5,
        full_gate_probability=0.7,
        suppress=suppress,
    )
    assert np.allclose(gate, [0.0, 0.5, 0.0, 0.0])


def test_boundary_threshold_is_selected_on_valid_rows_only():
    score = np.array([0.9, 0.8, 0.4, 0.3, 0.95])
    target = np.array([True, True, False, False, False])
    valid = np.array([True, True, True, True, False])
    selected, candidates = select_boundary_threshold(score, target, valid, [0.3, 0.5, 0.8])
    assert selected == 0.8
    assert len(candidates) == 3
    assert max(value["boundary_f1"] for value in candidates) == 1.0


def test_context_features_are_finite_and_scenario_free():
    rng = np.random.default_rng(4)
    rows = 7
    m4 = rng.dirichlet(np.ones(3), rows)
    a3 = rng.dirichlet(np.ones(3), rows)
    attention = rng.dirichlet(np.ones(5), rows)
    values, names = boundary_context_features(
        m4,
        a3,
        m4_gate=rng.random(rows),
        attention=attention,
        survival=rng.random((rows, 5)),
        boundary_probabilities=rng.random((rows, 5)),
        slot_masses=rng.random((rows, 6)),
        quality=rng.random((rows, 6)),
        memory_features=rng.normal(size=(rows, 3072)),
    )
    assert values.shape == (rows, len(names))
    assert np.isfinite(values).all()
    assert not any("scenario" in name or "label" in name for name in names)


def test_correction_gate_trains_only_on_disagreements():
    m4 = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
    )
    a3 = np.array(
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.1, 0.8, 0.1]]
    )
    labels = np.array([1, 1, 2, 2])
    features, names = uncertainty_features(m4, a3)
    gate = fit_correction_gate(
        features,
        names,
        m4,
        a3,
        labels,
        c=0.1,
        harm_weight=1.5,
        maximum_iterations=2000,
        tolerance=1e-8,
        random_state=42,
    )
    assert gate.training_summary["rows"] == 4
    assert gate.training_summary["benefits"] == 2
    assert gate.training_summary["harms"] == 2
    assert gate.training_summary["converged"] is True
    assert gate.predict(features).shape == (4,)
