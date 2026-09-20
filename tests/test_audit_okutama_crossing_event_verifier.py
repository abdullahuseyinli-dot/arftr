"""Deliberate corruptions and semantic fixtures for the independent audit."""

import numpy as np
import pytest

from experiments.audit_okutama_crossing_event_verifier import (
    FEATURE_NAMES,
    VARIANTS,
    reconstruct_decision,
    reconstruct_features,
    reconstruct_scaler,
    reconstruct_scores,
    reconstruct_targets,
    solution_checks,
)


def _examples():
    anchor = np.tile([0.7, 0.2, 0.1], (3, 1))
    actions = np.repeat(anchor[:, None, :], 4, axis=1)
    actions[:, 1] = [0.1, 0.8, 0.1]
    actions[:, 2] = [0.1, 0.1, 0.8]
    # Third action is noncrossing, so it must not enter conditional fitting.
    actions[:, 3] = [0.6, 0.3, 0.1]
    return anchor, actions


def test_independent_event_labels_weights_and_feature_direction():
    anchor, actions = _examples()
    targets = reconstruct_targets(anchor, actions, np.asarray([1, 0, 2]))
    np.testing.assert_array_equal(targets["events"], [[0, 2, -1], [1, 1, -1], [2, 0, -1]])
    np.testing.assert_array_equal(targets["utility"], [[1, 0, 0], [-2, -2, 0], [0, 1, 0]])
    np.testing.assert_allclose(targets["crossing_weights"].sum(1), 1)
    np.testing.assert_array_equal(targets["crossing_weights"][:, 2], 0)
    features = reconstruct_features(anchor, actions, np.zeros((3, 2)), np.zeros((3, 4)), np.zeros((3, 6)))
    assert features.shape == (3, 3, 35) and len(FEATURE_NAMES) == 35
    assert features[0, 0, 23] == 1  # ordered pair 0 -> 1
    assert features[0, 1, 24] == 1  # ordered pair 0 -> 2
    assert np.sum(features[0, 2, 23:29]) == 0  # no class change
    np.testing.assert_array_equal(features[0, :, 29:32], np.eye(3))
    assert features[0, 0, 32] == pytest.approx(np.log(0.2) - np.log(0.7))
    assert features[0, 0, 33] == pytest.approx(np.log(0.8) - np.log(0.1))
    np.testing.assert_array_equal(features[:, :, 34], 2)


def test_decision_rejects_non_crossing_invalid_and_zero_utility_and_resolves_tie():
    anchor, actions = _examples()
    features = np.zeros((3, 3, 35))
    features[:, :, 15] = 1
    features[1, 0, 0] = np.nan
    utility = np.asarray([[1.0, 1.0, 100.0], [1.0, 1.0, 1.0], [0.0, -1.0, 100.0]])
    nll = np.ones((3, 3))
    probabilities, choices, invalid = reconstruct_decision(anchor, actions, features, utility, nll)
    np.testing.assert_array_equal(choices, [1, 0, 0])
    np.testing.assert_array_equal(invalid, [False, True, False])
    assert probabilities[1:].tobytes() == anchor[1:].tobytes()
    np.testing.assert_array_equal(probabilities[0], actions[0, 1])
    nll[0, :2] = 0.0
    _, choices, _ = reconstruct_decision(anchor, actions, features, utility, nll)
    np.testing.assert_array_equal(choices, 0)


def _constant_model(variant):
    rows = 3 if variant == VARIANTS[2] else 1
    return {"variant": np.asarray(variant), "scaler_mean": np.zeros(35),
            "scaler_scale": np.ones(35), "coefficients": np.zeros((rows, 35)),
            "intercepts": np.zeros(rows)}


def test_ridge_stationarity_rejects_parameter_and_scaler_corruption():
    features = np.zeros((3, 3, 35))
    anchor = np.tile([0.7, 0.2, 0.1], (3, 1))
    actions = np.repeat(anchor[:, None], 4, axis=1)
    targets = reconstruct_targets(anchor, actions, np.zeros(3, dtype=int))
    model = _constant_model(VARIANTS[0])
    result = solution_checks(model, features, targets)
    assert result["normal_equation_relative_residual"] == 0
    model["coefficients"][0, 0] = 0.01
    with pytest.raises(RuntimeError, match="normal equation"):
        solution_checks(model, features, targets)
    model["coefficients"][0, 0] = 0
    model["scaler_mean"][0] = 0.01
    with pytest.raises(RuntimeError, match="scaler mean"):
        solution_checks(model, features, targets)


def test_softmax_independent_objective_stationarity_rejects_false_convergence():
    features = np.zeros((3, 3, 35))
    targets = {"crossing": np.ones((3, 3), dtype=bool),
               "events": np.repeat(np.arange(3)[:, None], 3, axis=1),
               "crossing_weights": np.full((3, 3), 1 / 3)}
    model = _constant_model(VARIANTS[2])
    checked = solution_checks(model, features, targets)
    assert checked["objective"] == pytest.approx(np.log(3))
    assert checked["gradient_linf"] < 1e-14
    utility, probabilities = reconstruct_scores(model, features)
    np.testing.assert_allclose(probabilities, 1 / 3)
    np.testing.assert_allclose(utility, -1 / 3)
    model["intercepts"][0] = 0.01
    with pytest.raises(RuntimeError, match="did not converge"):
        solution_checks(model, features, targets)


def test_common_scaler_is_all_event_population_not_crossing_subset():
    features = np.zeros((2, 3, 35))
    features[:, :, 0] = [[0, 2, 4], [6, 8, 10]]
    mean, scale = reconstruct_scaler(features)
    assert mean[0] == 5
    assert scale[0] == pytest.approx(np.std([0, 2, 4, 6, 8, 10]))
    np.testing.assert_array_equal(scale[1:], 1)
