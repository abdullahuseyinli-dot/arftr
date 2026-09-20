"""Synthetic-only checks for the dimension-derived solver-memory amendment.

No scientific caches, labels, predictions, or fit receipts are loaded here.
The conditioning fixture was fixed before testing the amended scientific run.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from hac import crossing_event_verifier as original
from hac import crossing_event_verifier_numerical_v2 as amended
from hac.intervention_utility import verifier_features


def _event_fixture(rows=72):
    rng = np.random.default_rng(910)
    anchor = rng.dirichlet(np.ones(3), size=rows)
    actions = np.repeat(anchor[:, None], 4, axis=1)
    for action in range(1, 4):
        actions[:, action] = rng.dirichlet(np.ones(3), size=rows)
    base, names = verifier_features(
        anchor, actions, rng.uniform(-0.5, 0.5, (rows, 2)),
        rng.uniform(0.1, 2, (rows, 4)), rng.normal(size=(rows, 6)),
    )
    features, _ = original.build_event_features(anchor, actions, base, names)
    return anchor, actions, features, np.arange(rows, dtype=np.int64) % 3


def _fixed_conditioning_fixture():
    """Correlated conditional evidence with large nonzero means, not real data."""
    rng = np.random.default_rng(20260919)
    rows, dimensions = 480, 35
    latent = rng.normal(size=(rows, 6))
    loadings = rng.normal(size=(6, dimensions))
    features = (
        4.0 * (latent @ loadings) / np.sqrt(6)
        + rng.normal(size=(rows, dimensions)) * 0.03
        + 8.0 * rng.choice([-1.0, 1.0], size=dimensions)
    )
    outcomes = np.tile(np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 1], dtype=np.int64), rows // 10)
    weights = rng.choice(np.array([1.0, 0.5, 1 / 3]), size=rows)
    return features, outcomes, weights


def test_scientific_schema_and_policy_are_the_original_objects():
    assert amended.original is original
    for name in (
        "VARIANTS", "build_event_features", "event_targets", "fit_common_scaler",
        "route_actions", "CrossingEventVerifier",
    ):
        assert getattr(amended, name) is getattr(original, name)
    assert amended.VARIANTS == (
        "V1_all_event_ridge", "V2_crossing_ridge", "V3_crossing_multinomial",
    )
    assert len(original.FEATURE_NAMES) == 35
    assert original.RIDGE_ALPHA == 10.0
    assert original.WEIGHT_PENALTY == 0.025
    assert original.BIAS_PENALTY == 0.00005
    assert original.MAX_ITERATIONS == 1000
    assert original.GRADIENT_TOLERANCE == 1e-7


def test_real_optimizer_calls_differ_only_in_dimension_derived_memory(monkeypatch):
    anchor, actions, features, labels = _event_fixture()
    inputs_before = [value.copy() for value in (anchor, actions, features, labels)]
    scaler = original.fit_common_scaler(features)
    calls = []

    def recorded_minimize(objective, initial, **kwargs):
        calls.append((objective, initial.copy(), kwargs))
        return minimize(objective, initial, **kwargs)

    monkeypatch.setattr(original, "minimize", recorded_minimize)
    monkeypatch.setattr(amended, "minimize", recorded_minimize)
    with threadpool_limits(limits=1):
        old_model, old_receipt = original.fit_verifier(
            features, anchor, actions, labels, original.VARIANTS[2], scaler=scaler,
        )
        new_model, new_receipt = amended.fit_verifier(
            features, anchor, actions, labels, original.VARIANTS[2], scaler=scaler,
        )

    assert len(calls) == 2
    for objective, initial, kwargs in calls:
        assert objective is original.multinomial_objective
        assert initial.shape == (108,) and initial.dtype == np.float64
        np.testing.assert_array_equal(initial, 0)
        assert kwargs["jac"] is True and kwargs["method"] == "L-BFGS-B"
    old_options, new_options = (call[2]["options"] for call in calls)
    assert old_options == {"maxiter": 1000, "gtol": 1e-7, "ftol": 0.0, "maxls": 50}
    assert new_options == {**old_options, "maxcor": 108}
    for old, new in zip(calls[0][2]["args"], calls[1][2]["args"], strict=True):
        np.testing.assert_array_equal(old, new)

    for model, receipt, call in zip(
        (old_model, new_model), (old_receipt, new_receipt), calls, strict=True,
    ):
        parameters = np.r_[model.coefficients.ravel(), model.intercepts]
        objective, gradient = original.multinomial_objective(parameters, *call[2]["args"])
        assert receipt["objective"] == objective
        assert receipt["gradient_linf"] == np.max(np.abs(gradient))
        assert receipt["converged"] and receipt["gradient_linf"] <= 1e-7
    assert new_receipt["objective"] == pytest.approx(old_receipt["objective"], abs=1e-11, rel=0)
    assert new_receipt["curvature_history"] == 108
    assert new_receipt["curvature_history_rule"] == "number_of_free_softmax_parameters"
    assert new_receipt["numerical_amendment"] == "v2_dimension_derived_memory; no scientific settings changed"

    numerical_fields = {
        "elapsed_seconds", "optimizer_iterations", "optimizer_message", "objective", "gradient_linf",
    }
    for name, value in old_receipt.items():
        if name not in numerical_fields:
            assert new_receipt[name] == value
    for before, after in zip(inputs_before, (anchor, actions, features, labels), strict=True):
        np.testing.assert_array_equal(before, after)


def test_fixed_synthetic_conditioning_objective_and_original_gradient_agree():
    features, outcomes, weights = _fixed_conditioning_fixture()
    parameters = np.zeros(3 * features.shape[1] + 3, dtype=np.float64)
    before = [value.copy() for value in (features, outcomes, weights)]
    results = []
    with threadpool_limits(limits=1):
        for history in (10, len(parameters), len(parameters)):
            result = minimize(
                original.multinomial_objective, parameters.copy(),
                args=(features, outcomes, weights), jac=True, method="L-BFGS-B",
                options={"maxiter": 1000, "gtol": 1e-7, "ftol": 0.0, "maxls": 50, "maxcor": history},
            )
            objective, gradient = original.multinomial_objective(result.x, features, outcomes, weights)
            assert result.success and result.nit <= 1000
            assert np.max(np.abs(gradient)) <= 1e-7  # Original coordinates, not a rescaled gradient.
            results.append((result, objective))
    old, new, repeated = results
    assert new[1] == pytest.approx(old[1], abs=1e-11, rel=0)
    assert new[0].nit < old[0].nit  # Fixture evidence only; no real-fold speed guarantee.
    np.testing.assert_array_equal(new[0].x, repeated[0].x)
    assert new[0].nit == repeated[0].nit
    for first, after in zip(before, (features, outcomes, weights), strict=True):
        np.testing.assert_array_equal(first, after)


@pytest.mark.parametrize("variant", original.VARIANTS[:2])
def test_both_ridge_controls_are_bit_identical_to_original(variant):
    anchor, actions, features, labels = _event_fixture()
    with threadpool_limits(limits=1):
        old, old_receipt = original.fit_verifier(features, anchor, actions, labels, variant)
        new, new_receipt = amended.fit_verifier(features, anchor, actions, labels, variant)
    for name, expected in old.to_arrays().items():
        np.testing.assert_array_equal(new.to_arrays()[name], expected)
    old_receipt.pop("elapsed_seconds")
    new_receipt.pop("elapsed_seconds")
    assert new_receipt == old_receipt


@pytest.mark.parametrize("variant", original.VARIANTS)
def test_all_variants_keep_original_serialization_and_deterministic_scores(variant):
    anchor, actions, features, labels = _event_fixture()
    with threadpool_limits(limits=1):
        model, _ = amended.fit_verifier(features, anchor, actions, labels, variant)
        repeated, _ = amended.fit_verifier(features, anchor, actions, labels, variant)
    arrays = model.to_arrays()
    assert set(arrays) == {
        "format_version", "variant", "feature_names", "scaler_mean", "scaler_scale",
        "coefficients", "intercepts",
    }
    assert arrays["format_version"].shape == () and int(arrays["format_version"]) == 1
    assert arrays["variant"].shape == () and str(arrays["variant"]) == variant
    assert tuple(arrays["feature_names"]) == original.FEATURE_NAMES
    for name, values in arrays.items():
        np.testing.assert_array_equal(values, repeated.to_arrays()[name])
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as saved:
        restored = original.CrossingEventVerifier.from_arrays(saved)
    for expected, actual in zip(model.scores(features), restored.scores(features), strict=True):
        np.testing.assert_array_equal(actual, expected)
    arrays["coefficients"][:] = 999
    assert not np.any(model.coefficients == 999)
    with pytest.raises(ValueError):
        model.coefficients[0, 0] = 0


def test_original_gradient_failure_cannot_be_hidden_by_optimizer_success(monkeypatch):
    anchor, actions, features, labels = _event_fixture()

    def premature_success(objective, initial, **kwargs):
        _, gradient = objective(initial, *kwargs["args"])
        assert np.max(np.abs(gradient)) > 1e-7
        return SimpleNamespace(x=initial, success=True, status=0, nit=1, message="synthetic early stop")

    monkeypatch.setattr(amended, "minimize", premature_success)
    with pytest.raises(RuntimeError, match="did not converge"):
        amended.fit_verifier(features, anchor, actions, labels, original.VARIANTS[2])


def test_optimizer_failure_cannot_be_hidden_by_zero_original_gradient(monkeypatch):
    anchor = np.tile([0.7, 0.2, 0.1], (3, 1))
    actions = np.repeat(anchor[:, None], 4, axis=1)
    actions[:, 1:] = [0.2, 0.7, 0.1]
    features = np.zeros((3, 3, 35))
    features[:, :, 15] = 1
    labels = np.array([1, 0, 2], dtype=np.int64)

    def numerical_failure(objective, initial, **kwargs):
        _, gradient = objective(initial, *kwargs["args"])
        assert np.max(np.abs(gradient)) <= 1e-7
        return SimpleNamespace(x=initial, success=False, status=1, nit=1000, message="synthetic failure")

    monkeypatch.setattr(amended, "minimize", numerical_failure)
    with pytest.raises(RuntimeError, match="did not converge"):
        amended.fit_verifier(features, anchor, actions, labels, original.VARIANTS[2])


def test_conditional_support_and_invalid_evidence_controls_are_not_relaxed():
    anchor, actions, features, labels = _event_fixture()
    no_crossing = np.repeat(anchor[:, None], 4, axis=1)
    with pytest.raises(ValueError, match="no class-changing"):
        amended.fit_verifier(features, anchor, no_crossing, labels, original.VARIANTS[2])
    invalid_features = features.copy()
    invalid_features[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite training features"):
        amended.fit_verifier(invalid_features, anchor, actions, labels, original.VARIANTS[2])
    invalid_actions = actions.copy()
    invalid_actions[0, 1] = np.nan
    with pytest.raises(ValueError, match="invalid probabilities"):
        amended.fit_verifier(features, anchor, invalid_actions, labels, original.VARIANTS[2])
    with pytest.raises(ValueError, match="unknown verifier variant"):
        amended.fit_verifier(features, anchor, actions, labels, "undeclared_variant")


def test_original_routing_cost_nll_and_exact_retain_controls_survive_amendment():
    anchor = np.tile(np.array([0.7, 0.2, 0.1], dtype=np.float32), (5, 1))
    actions = np.repeat(anchor[:, None], 4, axis=1).astype(np.float64)
    actions[:, 1] = [0.2, 0.7, 0.1]
    actions[:, 2] = [0.1, 0.2, 0.7]
    features = np.zeros((5, 3, 35))
    features[:, :, 15] = 1
    utility = np.array([[1, 1, 9], [0, -1, 9], [1, 1, 9], [1, 1, 9], [1, 1, 9]], dtype=float)
    nll = np.ones((5, 3))
    nll[2, :2] = 0
    features[3, 0, 0] = np.nan
    features[4, :, 15] = 0
    original_route = original.route_actions(anchor, actions, features, utility, nll)
    new_route = amended.route_actions(anchor, actions, features, utility, nll)
    for expected, actual in zip(original_route, new_route, strict=True):
        np.testing.assert_array_equal(actual, expected)
    routed, choices, invalid = new_route
    np.testing.assert_array_equal(choices, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(invalid, [False, False, False, True, True])
    assert routed[1:].tobytes() == anchor[1:].tobytes()
