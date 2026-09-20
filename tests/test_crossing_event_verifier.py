from __future__ import annotations

import io

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from hac.crossing_event_verifier import (
    FEATURE_NAMES,
    ORIGINAL_FEATURE_NAMES,
    VARIANTS,
    CrossingEventVerifier,
    build_event_features,
    event_targets,
    event_weights,
    fit_common_scaler,
    fit_verifier,
    multinomial_objective,
    route_actions,
)
from hac.intervention_utility import verifier_features


def fixture(rows=72):
    rng = np.random.default_rng(910)
    anchor = rng.dirichlet(np.ones(3), size=rows)
    actions = np.repeat(anchor[:, None], 4, axis=1)
    for action in range(1, 4):
        actions[:, action] = rng.dirichlet(np.ones(3), size=rows)
    original, names = verifier_features(
        anchor, actions, rng.uniform(-0.5, 0.5, (rows, 2)),
        rng.uniform(0.1, 2, (rows, 4)), rng.normal(size=(rows, 6)),
    )
    features, feature_names = build_event_features(anchor, actions, original, names)
    labels = np.arange(rows, dtype=np.int64) % 3
    return anchor, actions, features, labels, feature_names


def test_features_are_label_free_and_lock_rival_direction_and_action_identity():
    anchor = np.array([[0.6, 0.3, 0.1]])
    actions = np.array([[[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.6, 0.1, 0.3], [0.1, 0.2, 0.7]]])
    original = np.zeros((1, 3, 23))
    original[:, :, 15] = 1
    before = original.copy()
    values, names = build_event_features(anchor, actions, original, list(ORIGINAL_FEATURE_NAMES))
    assert values.shape == (1, 3, 35)
    assert names == list(FEATURE_NAMES)
    np.testing.assert_array_equal(values[0, :, 23:29], [[1, 0, 0, 0, 0, 0], [0] * 6, [0, 1, 0, 0, 0, 0]])
    np.testing.assert_array_equal(values[0, :, 29:32], np.eye(3))
    np.testing.assert_allclose(values[0, :, 32], [np.log(0.3 / 0.6), 0, np.log(0.1 / 0.6)])
    np.testing.assert_allclose(values[0, :, 33], [np.log(0.7 / 0.2), 0, np.log(0.7 / 0.1)])
    np.testing.assert_array_equal(values[0, :, 34], [2, 2, 2])
    np.testing.assert_array_equal(before, original)
    with pytest.raises(ValueError, match="names/order"):
        build_event_features(anchor, actions, original, list(reversed(ORIGINAL_FEATURE_NAMES)))


def test_targets_distinguish_rescue_harm_both_wrong_and_structural_zero():
    anchor = np.tile([0.7, 0.2, 0.1], (3, 1))
    actions = np.repeat(anchor[:, None], 4, axis=1)
    actions[:, 1] = [0.2, 0.7, 0.1]
    crossing, outcomes, utility = event_targets(anchor, actions, np.array([1, 0, 2]))
    np.testing.assert_array_equal(crossing[:, 0], True)
    np.testing.assert_array_equal(outcomes[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(utility[:, 0], [1, -2, 0])
    np.testing.assert_array_equal(outcomes[:, 1:], -1)
    np.testing.assert_array_equal(utility[:, 1:], 0)


def test_weights_give_each_center_one_vote_and_scaler_uses_all_events():
    crossing = np.array([[True, False, False], [True, True, False], [True, True, True], [False] * 3])
    np.testing.assert_allclose(event_weights(crossing, VARIANTS[0]), 1 / 3)
    expected = [[1, 0, 0], [0.5, 0.5, 0], [1 / 3] * 3, [0] * 3]
    for variant in VARIANTS[1:]:
        np.testing.assert_allclose(event_weights(crossing, variant), expected)
        np.testing.assert_allclose(event_weights(crossing, variant).sum(1), [1, 1, 1, 0])
    _, _, x, _, _ = fixture()
    mean, scale = fit_common_scaler(x)
    np.testing.assert_allclose(mean, x.reshape(-1, 35).mean(0), atol=1e-14)
    expected_scale = x.reshape(-1, 35).std(0)
    expected_scale[expected_scale == 0] = 1
    np.testing.assert_allclose(scale, expected_scale, atol=1e-14)


@pytest.mark.parametrize("variant", VARIANTS[:2])
def test_ridge_exact_weighted_sum_objective_matches_sklearn(variant):
    anchor, actions, x, labels, _ = fixture()
    scaler = fit_common_scaler(x)
    model, receipt = fit_verifier(x, anchor, actions, labels, variant, scaler=scaler)
    crossing, _, targets = event_targets(anchor, actions, labels)
    weights = event_weights(crossing, variant)
    included = weights > 0
    standardized = ((x - scaler[0]) / scaler[1])[included]
    reference = Ridge(alpha=10, fit_intercept=True, solver="cholesky").fit(
        standardized, targets[included], sample_weight=weights[included],
    )
    np.testing.assert_allclose(model.coefficients[0], reference.coef_, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(model.intercepts[0], reference.intercept_, atol=2e-13, rtol=2e-13)
    assert receipt["gradient_linf"] < 1e-9
    assert receipt["event_weight_sum"] == pytest.approx(float(weights.sum()))
    assert receipt["converged"]
    assert model.scores(x)[1].shape == (0,)


def test_multinomial_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(9, 4))
    outcomes = np.arange(9, dtype=np.int64) % 3
    weights = np.array([1, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 1, 0.5, 0.5])
    parameters = rng.normal(scale=0.3, size=15)
    before = parameters.copy()
    _, analytic = multinomial_objective(parameters, x, outcomes, weights)
    epsilon = 1e-6
    numeric = np.empty_like(parameters)
    for index in range(len(parameters)):
        direction = np.zeros_like(parameters)
        direction[index] = epsilon
        positive = multinomial_objective(parameters + direction, x, outcomes, weights)[0]
        negative = multinomial_objective(parameters - direction, x, outcomes, weights)[0]
        numeric[index] = (positive - negative) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, atol=3e-9, rtol=3e-8)
    np.testing.assert_array_equal(before, parameters)


def test_multinomial_penalties_and_probability_cost_sign():
    x = np.zeros((6, 2))
    outcomes = np.arange(6, dtype=np.int64) % 3
    p = np.array([1, 2, 3, 4, 5, 6, 0.5, -0.2, 0.7], dtype=np.float64)
    objective, _ = multinomial_objective(p, x, outcomes, np.ones(6))
    z = p[-3:]
    ce = np.log(np.exp(z).sum()) - z.mean()
    assert objective == pytest.approx(ce + 0.025 * np.sum(p[:-3] ** 2) + 0.00005 * np.sum(z ** 2))
    model = CrossingEventVerifier(VARIANTS[2], np.zeros(35), np.ones(35), np.zeros((3, 35)), np.log([0.6, 0.35, 0.05]))
    utility, probabilities = model.scores(np.zeros((2, 3, 35)))
    np.testing.assert_allclose(probabilities[0, 0], [0.6, 0.35, 0.05])
    np.testing.assert_allclose(utility, -0.1)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fit_save_load_deterministic_and_inputs_immutable(variant):
    anchor, actions, x, labels, _ = fixture()
    original = [v.copy() for v in (anchor, actions, x, labels)]
    model, receipt = fit_verifier(x, anchor, actions, labels, variant)
    repeated, _ = fit_verifier(x, anchor, actions, labels, variant)
    assert receipt["converged"]
    if variant == VARIANTS[2]:
        assert receipt["gradient_linf"] <= 1e-7
    np.testing.assert_array_equal(model.coefficients, repeated.coefficients)
    stream = io.BytesIO()
    np.savez(stream, **model.to_arrays())
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as arrays:
        restored = CrossingEventVerifier.from_arrays(arrays)
    for left, right in zip(model.scores(x), restored.scores(x), strict=True):
        np.testing.assert_array_equal(left, right)
    for before, after in zip(original, (anchor, actions, x, labels), strict=True):
        np.testing.assert_array_equal(before, after)
    detached = model.to_arrays()
    detached["coefficients"][:] = 99
    assert not np.any(model.coefficients == 99)
    with pytest.raises(ValueError):
        model.coefficients[0, 0] = 0


def test_empty_crossing_population_is_valid_for_all_events_but_conditional_stops():
    anchor, _, x, labels, _ = fixture()
    actions = np.repeat(anchor[:, None], 4, axis=1)
    all_event, _ = fit_verifier(x, anchor, actions, labels, VARIANTS[0])
    np.testing.assert_array_equal(all_event.scores(x)[0], 0)
    for variant in VARIANTS[1:]:
        with pytest.raises(ValueError, match="no class-changing"):
            fit_verifier(x, anchor, actions, labels, variant)


def test_route_enforces_crossing_nll_sign_ties_and_exact_float32_retention():
    anchor = np.tile(np.array([0.7, 0.2, 0.1], dtype=np.float32), (6, 1))
    actions = np.repeat(anchor[:, None], 4, axis=1).astype(np.float64)
    actions[:, 1] = [0.2, 0.7, 0.1]
    actions[:, 2] = [0.1, 0.2, 0.7]
    x = np.zeros((6, 3, 35))
    x[:, :, 15] = 1
    utility = np.array([[1, 1, 3], [0, -1, 3], [1, 0.5, 3], [1, 0.5, 3], [1, 0.5, 3], [1, 0.5, 3]], dtype=float)
    nll = np.ones((6, 3))
    nll[2, :2] = 0
    x[3, 0, 0] = np.nan
    x[4, :, 15] = 0
    utility[5, 2] = np.inf
    before = anchor.tobytes()
    routed, choices, invalid = route_actions(anchor, actions, x, utility, nll)
    np.testing.assert_array_equal(choices, [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(invalid, [False, False, False, True, True, True])
    assert routed.dtype == anchor.dtype
    assert routed[1:].tobytes() == anchor[1:].tobytes()
    assert anchor.tobytes() == before


def test_invalid_candidates_propagate_for_retention_but_cannot_train():
    anchor, actions, x, labels, _ = fixture()
    actions[0, 1] = np.nan
    built, _ = build_event_features(anchor, actions, x[:, :, :23], list(ORIGINAL_FEATURE_NAMES))
    assert np.isnan(built[0]).all()
    utility = np.ones((len(anchor), 3))
    routed, choices, invalid = route_actions(anchor, actions, built, utility, utility)
    assert choices[0] == 0 and invalid[0]
    assert routed[0].tobytes() == anchor[0].tobytes()
    with pytest.raises(ValueError, match="finite training features"):
        fit_verifier(built, anchor, actions, labels, VARIANTS[2])
    with pytest.raises(ValueError, match="invalid probabilities"):
        event_targets(anchor, actions, labels)


def test_serialized_schema_and_label_validation_reject_corruption():
    anchor, actions, x, labels, _ = fixture()
    model, _ = fit_verifier(x, anchor, actions, labels, VARIANTS[0])
    arrays = model.to_arrays()
    arrays["feature_names"] = arrays["feature_names"][::-1]
    with pytest.raises(ValueError, match="schema"):
        CrossingEventVerifier.from_arrays(arrays)
    with pytest.raises(ValueError, match="class IDs"):
        event_targets(anchor, actions, labels.astype(float))
    with pytest.raises(ValueError, match="anchor must"):
        route_actions(anchor * 2, actions, x, np.ones((len(anchor), 3)), np.ones((len(anchor), 3)))
