"""Locked, pooled class-event verifiers over frozen PDI action predictions.

No producer, NLL model, threshold or activity classifier is fitted here. The
three variants differ only in the declared event sampling/objective. Training
validation is strict; inference with missing evidence returns the original
anchor rather than trying to impute evidence or use a different action bank.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

VARIANTS = ("V1_all_event_ridge", "V2_crossing_ridge", "V3_crossing_multinomial")
OUTCOMES = ("rescue", "harm", "both_wrong")
ORDERED_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
ORIGINAL_FEATURE_NAMES = (
    *(f"log_anchor_{i}" for i in range(3)),
    "anchor_margin", "anchor_entropy",
    *(f"coarse_scaffold_norm_{i}" for i in range(4)),
    *(f"quality_{i}" for i in range(6)),
    "available_0", "posture_delta", "motion_delta",
    *(f"logit_delta_{i}" for i in range(3)),
    "candidate_margin", "crosses_class_boundary",
)
FEATURE_NAMES = (
    *ORIGINAL_FEATURE_NAMES,
    *(f"predicted_pair_{left}_to_{right}" for left, right in ORDERED_PAIRS),
    "action_posture", "action_motion", "action_both",
    "rival_log_margin_before", "rival_log_margin_after", "crossing_action_count",
)
RIDGE_ALPHA = 10.0
WEIGHT_PENALTY = 0.025
BIAS_PENALTY = 0.00005
GRADIENT_TOLERANCE = 1e-7
MAX_ITERATIONS = 1000


def _anchor(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all()
        or np.any(result < 0)
        or not np.allclose(result.sum(1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("anchor must be finite [rows,3] probability simplexes")
    return result


def _actions(values: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(anchor), 4, 3):
        raise ValueError("actions must have shape [rows,4,3]")
    valid = (
        np.isfinite(result).all((1, 2)) & (result >= 0).all((1, 2))
        & np.isclose(result.sum(2), 1.0, atol=1e-6, rtol=0).all(1)
        # PDI action 0 passed through float32; retain uses the original anchor.
        & np.isclose(result[:, 0], anchor, atol=1e-6, rtol=0).all(1)
    )
    return result, valid


def _features(values: np.ndarray, rows: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (3, len(FEATURE_NAMES)):
        raise ValueError("features must have shape [rows,3,35]")
    if rows is not None and len(result) != rows:
        raise ValueError("feature and probability rows differ")
    return result


def build_event_features(
    anchor: np.ndarray,
    actions: np.ndarray,
    original_features: np.ndarray,
    original_feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Append the locked twelve inference-only event features.

    Invalid candidate rows propagate NaNs for the inference retention path.
    Training functions reject them. No input arrays are modified.
    """

    anchor = _anchor(anchor)
    actions, valid = _actions(actions, anchor)
    original = np.asarray(original_features, dtype=np.float64)
    if original.shape != (len(anchor), 3, 23):
        raise ValueError("original features must have shape [rows,3,23]")
    if tuple(original_feature_names) != ORIGINAL_FEATURE_NAMES:
        raise ValueError("original feature names/order differ from the locked schema")
    safe_actions = np.where(valid[:, None, None], actions, anchor[:, None, :])
    old = anchor.argmax(1)
    new = safe_actions[:, 1:].argmax(2)
    crossing = new != old[:, None]
    log_anchor = np.log(np.clip(anchor, 1e-12, 1.0))
    log_actions = np.log(np.clip(safe_actions[:, 1:], 1e-12, 1.0))
    count = crossing.sum(1)
    rows = np.arange(len(anchor))
    extras = np.zeros((len(anchor), 3, 12), dtype=np.float64)
    for action in range(3):
        for pair_index, (left, right) in enumerate(ORDERED_PAIRS):
            extras[:, action, pair_index] = (old == left) & (new[:, action] == right)
        extras[:, action, 6 + action] = 1.0
        extras[:, action, 9] = log_anchor[rows, new[:, action]] - log_anchor[rows, old]
        extras[:, action, 10] = (
            log_actions[rows, action, new[:, action]] - log_actions[rows, action, old]
        )
        extras[:, action, 11] = count
    result = np.concatenate((original, extras), axis=2)
    result[~valid] = np.nan
    return result, list(FEATURE_NAMES)


def event_targets(
    anchor: np.ndarray, actions: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return crossing flags, R/H/W outcomes and exact +1/-2/0 utilities."""

    anchor = _anchor(anchor)
    actions, valid = _actions(actions, anchor)
    labels = np.asarray(labels)
    if not valid.all():
        raise ValueError("training actions contain invalid probabilities")
    if (
        labels.shape != (len(anchor),) or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels > 2))
    ):
        raise ValueError("labels must be integer class IDs in [0,2]")
    old = anchor.argmax(1)
    new = actions[:, 1:].argmax(2)
    crossing = new != old[:, None]
    rescue = crossing & (old[:, None] != labels[:, None]) & (new == labels[:, None])
    harm = crossing & (old[:, None] == labels[:, None]) & (new != labels[:, None])
    outcomes = np.full(crossing.shape, -1, dtype=np.int64)
    outcomes[crossing] = 2
    outcomes[rescue] = 0
    outcomes[harm] = 1
    utility = np.zeros(crossing.shape, dtype=np.float64)
    utility[rescue], utility[harm] = 1.0, -2.0
    return crossing, outcomes, utility


def event_weights(crossing: np.ndarray, variant: str) -> np.ndarray:
    """Each included center has total weight one, regardless of action count."""

    crossing = np.asarray(crossing)
    if crossing.ndim != 2 or crossing.shape[1] != 3 or crossing.dtype != np.bool_:
        raise ValueError("crossing must be boolean [rows,3]")
    if variant not in VARIANTS:
        raise ValueError("unknown verifier variant")
    if variant == VARIANTS[0]:
        return np.full(crossing.shape, 1.0 / 3.0, dtype=np.float64)
    return crossing / np.maximum(crossing.sum(1), 1)[:, None]


def fit_common_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All-row/action weighted population scaler; constant columns use scale 1.

    Every row has exactly three events of weight 1/3, so the weighted moments
    equal the unweighted moments of the flattened event matrix.
    """

    features = _features(features)
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("scaler needs nonempty finite training features")
    flat = features.reshape(-1, len(FEATURE_NAMES))
    mean = flat.mean(0)
    scale = np.sqrt(np.mean((flat - mean) ** 2, axis=0))
    scale = np.where(scale == 0, 1.0, scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("training scaler overflowed")
    return mean, scale


def _scaler(scaler: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(scaler) != 2:
        raise ValueError("scaler must be (mean, scale)")
    mean, scale = (np.asarray(value, dtype=np.float64) for value in scaler)
    if (
        mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape
        or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
    ):
        raise ValueError("scaler parameters must be finite 35-vectors with positive scale")
    return mean, scale


def multinomial_objective(
    parameters: np.ndarray, features: np.ndarray, outcomes: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Weighted-mean CE and its analytic gradient, with the locked L2 penalties.

    Parameters flatten class-major W[3,D], followed by b[3]. This function is
    exposed for finite-difference verification and does not mutate its inputs.
    """

    features = np.asarray(features, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    outcomes = np.asarray(outcomes)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        features.ndim != 2 or not len(features) or outcomes.shape != (len(features),)
        or weights.shape != (len(features),)
        or not np.issubdtype(outcomes.dtype, np.integer)
        or np.any((outcomes < 0) | (outcomes > 2)) or np.any(weights < 0)
        or not np.isfinite(features).all() or not np.isfinite(weights).all()
        or not np.isfinite(parameters).all()
        or parameters.shape != (3 * features.shape[1] + 3,) or weights.sum() <= 0
    ):
        raise ValueError("malformed multinomial objective inputs")
    coefficient = parameters[:-3].reshape(3, features.shape[1])
    intercept = parameters[-3:]
    logits = features @ coefficient.T + intercept
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    normalized_weight = weights / weights.sum()
    loss = -np.sum(normalized_weight * log_probabilities[np.arange(len(features)), outcomes])
    loss += WEIGHT_PENALTY * np.sum(coefficient ** 2) + BIAS_PENALTY * np.sum(intercept ** 2)
    error = np.exp(log_probabilities)
    error[np.arange(len(features)), outcomes] -= 1.0
    error *= normalized_weight[:, None]
    coefficient_gradient = error.T @ features + 2 * WEIGHT_PENALTY * coefficient
    intercept_gradient = error.sum(0) + 2 * BIAS_PENALTY * intercept
    gradient = np.concatenate((coefficient_gradient.ravel(), intercept_gradient))
    return float(loss), gradient


@dataclass(frozen=True)
class CrossingEventVerifier:
    variant: str
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError("unknown verifier variant")
        mean, scale = _scaler((self.scaler_mean, self.scaler_scale))
        outputs = 3 if self.variant == VARIANTS[2] else 1
        coefficients, intercepts = (
            np.asarray(value, dtype=np.float64) for value in (self.coefficients, self.intercepts)
        )
        if (
            coefficients.shape != (outputs, len(FEATURE_NAMES))
            or intercepts.shape != (outputs,) or not np.isfinite(coefficients).all()
            or not np.isfinite(intercepts).all()
        ):
            raise ValueError("model parameter shape/finiteness differs from its variant")
        for name, value in (
            ("scaler_mean", mean), ("scaler_scale", scale),
            ("coefficients", coefficients), ("intercepts", intercepts),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def scores(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = _features(features)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = (values - self.scaler_mean) / self.scaler_scale
            logits = transformed @ self.coefficients.T + self.intercepts
            if self.variant != VARIANTS[2]:
                return logits[..., 0], np.empty((0,), dtype=np.float64)
            probabilities = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
            return probabilities[..., 0] - 2 * probabilities[..., 1], probabilities

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "format_version": np.asarray(1, dtype=np.int64),
            "variant": np.asarray(self.variant), "feature_names": np.asarray(FEATURE_NAMES),
            "scaler_mean": self.scaler_mean.copy(), "scaler_scale": self.scaler_scale.copy(),
            "coefficients": self.coefficients.copy(), "intercepts": self.intercepts.copy(),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> CrossingEventVerifier:
        required = {"format_version", "variant", "feature_names", "scaler_mean", "scaler_scale", "coefficients", "intercepts"}
        if not required.issubset(arrays):
            raise ValueError("serialized model is missing required fields")
        if np.asarray(arrays["format_version"]).shape != () or int(arrays["format_version"]) != 1:
            raise ValueError("unsupported serialized model format")
        if np.asarray(arrays["variant"]).shape != () or tuple(np.asarray(arrays["feature_names"]).tolist()) != FEATURE_NAMES:
            raise ValueError("serialized model feature schema differs")
        return cls(
            str(np.asarray(arrays["variant"]).item()), arrays["scaler_mean"], arrays["scaler_scale"],
            arrays["coefficients"], arrays["intercepts"],
        )


def fit_verifier(
    features: np.ndarray,
    anchor: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    variant: str,
    scaler: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[CrossingEventVerifier, dict]:
    """Fit exactly one declared gate; ancestry/support checks belong to the runner."""

    if variant not in VARIANTS:
        raise ValueError("unknown verifier variant")
    anchor = _anchor(anchor)
    features = _features(features, len(anchor))
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("gate fitting requires nonempty finite training features")
    crossing, outcomes, utility = event_targets(anchor, actions, labels)
    mean, scale = _scaler(fit_common_scaler(features) if scaler is None else scaler)
    weights = event_weights(crossing, variant)
    included = weights > 0
    if not included.any():
        raise ValueError("conditional gate has no class-changing training events")
    x = ((features - mean) / scale)[included]
    weight = weights[included]
    if not np.isfinite(x).all():
        raise ValueError("standardized training features are nonfinite")
    started = perf_counter()
    if variant == VARIANTS[2]:
        target = outcomes[included]
        initial = np.zeros(3 * len(FEATURE_NAMES) + 3, dtype=np.float64)
        result = minimize(
            multinomial_objective, initial, args=(x, target, weight), jac=True,
            method="L-BFGS-B",
            options={"maxiter": MAX_ITERATIONS, "gtol": GRADIENT_TOLERANCE, "ftol": 0.0, "maxls": 50},
        )
        objective, gradient = multinomial_objective(result.x, x, target, weight)
        gradient_linf = float(np.max(np.abs(gradient)))
        converged = bool(result.success and np.isfinite(objective) and gradient_linf <= GRADIENT_TOLERANCE)
        if not converged:
            raise RuntimeError(
                f"locked multinomial did not converge: status={result.status}, "
                f"iterations={result.nit}, gradient_linf={gradient_linf}, message={result.message}"
            )
        coefficients, intercepts = result.x[:-3].reshape(3, -1), result.x[-3:]
        optimizer = {
            "optimizer": "L-BFGS-B", "converged": converged,
            "optimizer_status": int(result.status), "optimizer_iterations": int(result.nit),
            "optimizer_message": str(result.message), "gradient_tolerance": GRADIENT_TOLERANCE,
            "max_iterations": MAX_ITERATIONS, "weight_penalty": WEIGHT_PENALTY,
            "bias_penalty": BIAS_PENALTY, "objective": objective, "gradient_linf": gradient_linf,
        }
    else:
        target = utility[included]
        total_weight = weight.sum()
        x_mean = (weight[:, None] * x).sum(0) / total_weight
        y_mean = float(np.sum(weight * target) / total_weight)
        centered = x - x_mean
        lhs = centered.T @ (weight[:, None] * centered) + RIDGE_ALPHA * np.eye(x.shape[1])
        rhs = centered.T @ (weight * (target - y_mean))
        coefficient = np.linalg.solve(lhs, rhs)
        intercept = y_mean - x_mean @ coefficient
        residual = x @ coefficient + intercept - target
        objective = float(np.sum(weight * residual ** 2) + RIDGE_ALPHA * np.sum(coefficient ** 2))
        gradient = np.r_[2 * x.T @ (weight * residual) + 2 * RIDGE_ALPHA * coefficient, 2 * np.sum(weight * residual)]
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            raise RuntimeError("weighted ridge produced nonfinite parameters/objective")
        coefficients, intercepts = coefficient[None], np.asarray([intercept])
        optimizer = {
            "optimizer": "closed_form_weighted_ridge", "converged": True,
            "optimizer_status": 0, "optimizer_iterations": 0,
            "optimizer_message": "weighted centered normal equations solved",
            "ridge_alpha": RIDGE_ALPHA, "intercept_penalized": False,
            "objective": objective, "gradient_linf": float(np.max(np.abs(gradient))),
        }
    model = CrossingEventVerifier(variant, mean, scale, coefficients, intercepts)
    receipt = {
        "status": "CROSSING_EVENT_VERIFIER_FIT_COMPLETE", "variant": variant,
        "rows": len(anchor), "feature_count": len(FEATURE_NAMES),
        "event_rows": int(included.sum()), "crossing_centers": int(crossing.any(1).sum()),
        "event_weight_sum": float(weight.sum()), "row_weight_max": float(weights.sum(1).max()),
        "outcome_event_counts": {name: int((outcomes == index).sum()) for index, name in enumerate(OUTCOMES)},
        "noncrossing_event_count": int((~crossing).sum()),
        "scaler": "all_training_row_actions_weight_1_over_3_population_std",
        "outcome_class_rebalancing": False, "elapsed_seconds": perf_counter() - started,
        **optimizer,
    }
    return model, receipt


def route_actions(
    anchor: np.ndarray,
    actions: np.ndarray,
    features: np.ndarray,
    utility: np.ndarray,
    nll_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply only valid positive-utility crossings; copy anchor bytes otherwise.

    ``invalid_mask`` is per center. Any invalid candidate/feature/score or
    nonpositive availability retains the entire center conservatively. Equal
    positive utilities select the lowest action; equality with zero retains.
    """

    original = np.asarray(anchor)
    anchor64 = _anchor(original)
    candidates, valid_actions = _actions(actions, anchor64)
    features = _features(features, len(anchor64))
    utility, nll_scores = (np.asarray(value, dtype=np.float64) for value in (utility, nll_scores))
    if utility.shape != (len(anchor64), 3) or nll_scores.shape != utility.shape:
        raise ValueError("utility and NLL scores must have shape [rows,3]")
    invalid = (
        ~valid_actions | ~np.isfinite(features).all((1, 2))
        | ~np.isfinite(utility).all(1) | ~np.isfinite(nll_scores).all(1)
        | (features[:, :, ORIGINAL_FEATURE_NAMES.index("available_0")] <= 0).any(1)
    )
    crossing = candidates[:, 1:].argmax(2) != anchor64.argmax(1)[:, None]
    eligible = crossing & (utility > 0) & (nll_scores > 0) & ~invalid[:, None]
    selected = np.argmax(np.where(eligible, utility, -np.inf), axis=1)
    active = eligible.any(1)
    choices = np.zeros(len(anchor64), dtype=np.int64)
    choices[active] = selected[active] + 1
    routed = original.copy()
    routed[active] = candidates[np.flatnonzero(active), choices[active]]
    return routed, choices, invalid
