"""Compact ARFTR-preserving posture likelihood correction.

The head never relearns locomotion.  It modifies sitting-versus-upright mass,
preserves the anchor's standing/walking conditional share, and includes an
exact byte-copy retain action.  All preprocessing is fit on the supplied
training population; outer orchestration is deliberately separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

PROJECTION_DIM = 48
SUPERVISED_COEFFICIENTS = 49
DELTA_CAP = 2.0
DEAD_ZONE = 0.25
L2_WEIGHT = 0.01
LOG_EPSILON = 1e-9


def _probabilities(value: np.ndarray) -> np.ndarray:
    p = np.asarray(value)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not np.issubdtype(p.dtype, np.floating)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Anchor must be finite three-class probabilities")
    return p


@dataclass(frozen=True)
class Projection:
    feature_mean: np.ndarray
    components: np.ndarray
    projected_mean: np.ndarray
    projected_scale: np.ndarray

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def receipt(self) -> dict[str, Any]:
        return {
            "input_dim": int(self.components.shape[1]),
            "output_dim": self.output_dim,
            "fitted_unsupervised_parameters": int(
                self.feature_mean.size
                + self.components.size
                + self.projected_mean.size
                + self.projected_scale.size
            ),
            "method": "deterministic_full_svd_pca_then_population_standardizer",
            "std_floor": 1e-6,
        }


@dataclass(frozen=True)
class PostureHead:
    coefficients: np.ndarray
    intercept: float
    cap: float
    dead_zone: float
    l2_weight: float
    optimizer_receipt: dict[str, Any]

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients)
        if (
            coefficients.ndim != 1
            or coefficients.size > PROJECTION_DIM
            or not np.isfinite(coefficients).all()
            or not np.isfinite([self.intercept, self.cap, self.dead_zone, self.l2_weight]).all()
            or self.cap <= 0
            or not 0 <= self.dead_zone < self.cap
            or self.l2_weight < 0
        ):
            raise ValueError("Malformed compact posture head")


def fit_projection(features: np.ndarray, *, output_dim: int = PROJECTION_DIM) -> Projection:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("PCA requires a nonempty finite training matrix")
    if output_dim != PROJECTION_DIM or values.shape[1] < output_dim:
        raise ValueError("The source posture projection is fixed at48 dimensions")
    mean = values.mean(0)
    centered = values - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:output_dim].copy()
    # Remove the sign ambiguity without consulting labels or held rows.
    pivot = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(output_dim), pivot])
    signs[signs == 0] = 1
    components *= signs[:, None]
    projected = centered @ components.T
    projected_mean = projected.mean(0)
    projected_scale = np.maximum(projected.std(0, ddof=0), 1e-6)
    return Projection(mean, components, projected_mean, projected_scale)


def apply_projection(features: np.ndarray, projection: Projection) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != projection.components.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("Features do not match the fitted projection")
    result = (values - projection.feature_mean) @ projection.components.T
    result = (result - projection.projected_mean) / projection.projected_scale
    if not np.isfinite(result).all():
        raise RuntimeError("Projected posture features are nonfinite")
    return result


def posture_factors(anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = _probabilities(anchor).astype(np.float64, copy=False)
    upright = p[:, 1] + p[:, 2]
    if np.any(upright <= 0):
        raise ValueError("Zero upright mass requires exact-retain handling")
    q_posture = np.log(np.clip(p[:, 0], LOG_EPSILON, None)) - np.log(
        np.clip(upright, LOG_EPSILON, None)
    )
    motion_share = p[:, 2] / upright
    return q_posture, motion_share


def continuous_delta(features: np.ndarray, head: PostureHead) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != head.coefficients.size or not np.isfinite(values).all():
        raise ValueError("Projected features do not match the posture head")
    score = values @ head.coefficients + head.intercept
    return head.cap * np.tanh(score / head.cap)


def _decode(anchor: np.ndarray, delta: np.ndarray) -> np.ndarray:
    p = _probabilities(anchor)
    q_posture, motion_share = posture_factors(p)
    correction = np.asarray(delta, dtype=np.float64)
    if correction.shape != (len(p),) or not np.isfinite(correction).all():
        raise ValueError("Posture correction must be one finite value per row")
    q_new = q_posture + correction
    sitting = np.empty_like(q_new)
    positive = q_new >= 0
    sitting[positive] = 1 / (1 + np.exp(-q_new[positive]))
    exp_q = np.exp(q_new[~positive])
    sitting[~positive] = exp_q / (1 + exp_q)
    upright = 1 - sitting
    result = np.stack((sitting, upright * (1 - motion_share), upright * motion_share), axis=1)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-12, rtol=0):
        raise RuntimeError("Posture decode left the probability simplex")
    return result


def apply_posture_policy(
    anchor: np.ndarray,
    features: np.ndarray,
    head: PostureHead,
    *,
    available: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply fixed dead-zone policy and copy retained rows exactly.

    Returns ``(probabilities, intervention_mask, continuous_delta)``.
    """
    p = _probabilities(anchor)
    delta = continuous_delta(features, head)
    observed = np.ones(len(p), dtype=bool) if available is None else np.asarray(available)
    if observed.shape != (len(p),) or observed.dtype != np.bool_:
        raise ValueError("Availability must be a boolean vector")
    valid_upright = (p[:, 1] + p[:, 2]) > 0
    intervene = observed & valid_upright & (np.abs(delta) > head.dead_zone)
    result = p.copy()
    if intervene.any():
        result[intervene] = _decode(p[intervene], delta[intervene]).astype(p.dtype, copy=False)
    if not np.array_equal(result[~intervene], p[~intervene]):
        raise RuntimeError("Retain action did not preserve exact anchor bytes")
    return result, intervene, delta


def class_weights(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels)
    if y.ndim != 1 or y.dtype.kind not in "iu" or not np.array_equal(np.unique(y), np.arange(3)):
        raise ValueError("Training labels must contain all three integer classes")
    counts = np.bincount(y, minlength=3)
    return len(y) / (3 * counts.astype(np.float64))


def fit_posture_head(
    features: np.ndarray,
    labels: np.ndarray,
    anchor: np.ndarray,
    *,
    available: np.ndarray | None = None,
    cap: float = DELTA_CAP,
    dead_zone: float = DEAD_ZONE,
    l2_weight: float = L2_WEIGHT,
) -> PostureHead:
    """Fit the locked continuous correction; dead-zone is inference-only."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    p = _probabilities(anchor)
    observed = np.ones(len(y), dtype=bool) if available is None else np.asarray(available)
    if (
        x.ndim != 2
        or x.shape != (len(y), x.shape[1])
        or x.shape[1] > PROJECTION_DIM
        or y.shape != (len(p),)
        or y.dtype.kind not in "iu"
        or np.any((y < 0) | (y > 2))
        or observed.shape != (len(y),)
        or observed.dtype != np.bool_
        or not observed.any()
        or not np.isfinite(x).all()
    ):
        raise ValueError("Malformed posture-head training inputs")
    if (
        not np.isfinite([cap, dead_zone, l2_weight]).all()
        or cap <= 0
        or not 0 <= dead_zone < cap
        or l2_weight < 0
    ):
        raise ValueError("Malformed fixed posture-head settings")
    weights = class_weights(y)
    upright = p[:, 1] + p[:, 2]
    valid_upright = upright > 0
    effective = observed & valid_upright
    if not effective.any():
        raise ValueError("No row has valid available posture evidence")
    q_posture = np.log(np.clip(p[:, 0], LOG_EPSILON, None)) - np.log(
        np.clip(upright, LOG_EPSILON, None)
    )
    motion_share = np.divide(
        p[:, 2], upright, out=np.full(len(p), 0.5, dtype=np.float64), where=valid_upright
    )
    row_weight = weights[y]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients, intercept = parameters[:-1], float(parameters[-1])
        score = x @ coefficients + intercept
        tanh = np.tanh(score / cap)
        delta = cap * tanh
        q_new = q_posture + np.where(effective, delta, 0.0)
        posture_nll = np.where(
            y == 0, np.logaddexp(0, -q_new), np.logaddexp(0, q_new)
        )
        motion_nll = np.zeros(len(y), dtype=np.float64)
        motion_nll[y == 1] = -np.log(
            np.clip(1 - motion_share[y == 1], LOG_EPSILON, 1)
        )
        motion_nll[y == 2] = -np.log(
            np.clip(motion_share[y == 2], LOG_EPSILON, 1)
        )
        anchor_nll = -np.log(
            np.clip(p[np.arange(len(y)), y], LOG_EPSILON, None)
        )
        nll = np.where(effective, posture_nll + motion_nll, anchor_nll)
        value = float(np.mean(row_weight * nll))
        value += l2_weight * float(coefficients @ coefficients)
        sitting = 1 / (1 + np.exp(-np.clip(q_new, -700, 700)))
        derivative_q = sitting - (y == 0)
        derivative_score = derivative_q * (1 - tanh**2) * effective
        derivative_score *= row_weight / len(y)
        gradient = np.concatenate(
            (x.T @ derivative_score + 2 * l2_weight * coefficients, [derivative_score.sum()])
        )
        return value, gradient

    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    result = minimize(
        lambda value: objective(value), initial, method="L-BFGS-B", jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 20}
    )
    value, gradient = objective(result.x)
    receipt = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": value,
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "continuous_training": True,
        "dead_zone_applied_during_training": False,
        "rows": int(len(y)),
        "available_rows": int(observed.sum()),
        "effective_rows": int(effective.sum()),
        "invalid_upright_rows": int((~valid_upright).sum()),
        "supervised_coefficients": int(len(result.x)),
    }
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(value):
        raise RuntimeError(f"Fixed posture-head fit did not converge: {receipt}")
    return PostureHead(result.x[:-1].copy(), float(result.x[-1]), cap, dead_zone, l2_weight, receipt)
