"""One numerical amendment: dimension-derived L-BFGS curvature memory.

The original scientific loss, scaler, feature set, zero initialization, iteration
budget, gradient tolerance, and policy are unchanged. The original solver/module
remains immutable. No fit-specific retries or regularization search are allowed.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from hac import crossing_event_verifier as original

VARIANTS = original.VARIANTS
build_event_features = original.build_event_features
event_targets = original.event_targets
fit_common_scaler = original.fit_common_scaler
route_actions = original.route_actions
CrossingEventVerifier = original.CrossingEventVerifier


def fit_verifier(features, anchor, actions, labels, variant, scaler=None):
    if variant != VARIANTS[2]:
        return original.fit_verifier(features, anchor, actions, labels, variant, scaler=scaler)
    anchor = original._anchor(anchor)
    features = original._features(features, len(anchor))
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("gate fitting requires nonempty finite training features")
    crossing, outcomes, _ = event_targets(anchor, actions, labels)
    mean, scale = original._scaler(fit_common_scaler(features) if scaler is None else scaler)
    weights = original.event_weights(crossing, variant)
    included = weights > 0
    if not included.any():
        raise ValueError("conditional gate has no class-changing training events")
    x = ((features - mean) / scale)[included]
    weight, target = weights[included], outcomes[included]
    if not np.isfinite(x).all():
        raise ValueError("standardized training features are nonfinite")
    initial = np.zeros(3 * len(original.FEATURE_NAMES) + 3, dtype=np.float64)
    started = perf_counter()
    result = minimize(
        original.multinomial_objective, initial, args=(x, target, weight), jac=True,
        method="L-BFGS-B",
        options={"maxiter": original.MAX_ITERATIONS, "gtol": original.GRADIENT_TOLERANCE,
                 "ftol": 0.0, "maxls": 50, "maxcor": len(initial)},
    )
    objective, gradient = original.multinomial_objective(result.x, x, target, weight)
    gradient_linf = float(np.max(np.abs(gradient)))
    converged = bool(result.success and np.isfinite(objective) and gradient_linf <= original.GRADIENT_TOLERANCE)
    if not converged:
        raise RuntimeError(f"numerical-v2 locked multinomial did not converge: status={result.status}, "
                           f"iterations={result.nit}, gradient_linf={gradient_linf}, message={result.message}")
    model = CrossingEventVerifier(variant, mean, scale, result.x[:-3].reshape(3, -1), result.x[-3:])
    receipt = {
        "status": "CROSSING_EVENT_VERIFIER_FIT_COMPLETE", "variant": variant,
        "rows": len(anchor), "feature_count": len(original.FEATURE_NAMES),
        "event_rows": int(included.sum()), "crossing_centers": int(crossing.any(1).sum()),
        "event_weight_sum": float(weight.sum()), "row_weight_max": float(weights.sum(1).max()),
        "outcome_event_counts": {name: int((outcomes == index).sum()) for index, name in enumerate(original.OUTCOMES)},
        "noncrossing_event_count": int((~crossing).sum()),
        "scaler": "all_training_row_actions_weight_1_over_3_population_std",
        "outcome_class_rebalancing": False, "elapsed_seconds": perf_counter() - started,
        "optimizer": "L-BFGS-B", "converged": converged, "optimizer_status": int(result.status),
        "optimizer_iterations": int(result.nit), "optimizer_message": str(result.message),
        "gradient_tolerance": original.GRADIENT_TOLERANCE, "max_iterations": original.MAX_ITERATIONS,
        "weight_penalty": original.WEIGHT_PENALTY, "bias_penalty": original.BIAS_PENALTY,
        "objective": objective, "gradient_linf": gradient_linf,
        "curvature_history": len(initial), "curvature_history_rule": "number_of_free_softmax_parameters",
        "numerical_amendment": "v2_dimension_derived_memory; no scientific settings changed",
    }
    return model, receipt
