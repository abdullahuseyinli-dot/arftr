"""Three-parameter, M4-anchored class-diagonal residual fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be finite three-class probability vectors")
    return values


def sitting_disagreement_protection(
    m4_probabilities: np.ndarray, a3_probabilities: np.ndarray, *, sitting_class: int = 0
) -> np.ndarray:
    """Protect any M4/A3 class disagreement involving predicted sitting."""

    m4 = _probabilities(m4_probabilities, "m4_probabilities")
    a3 = _probabilities(a3_probabilities, "a3_probabilities")
    if m4.shape != a3.shape or sitting_class not in range(3):
        raise ValueError("Protection inputs or sitting class are invalid")
    m4_class, a3_class = m4.argmax(1), a3.argmax(1)
    return (m4_class != a3_class) & (
        (m4_class == sitting_class) | (a3_class == sitting_class)
    )


def apply_class_diagonal_residual(
    m4_probabilities: np.ndarray,
    a3_probabilities: np.ndarray,
    coefficients: np.ndarray,
    *,
    protected: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Apply classwise log-residuals, retaining exact M4 bytes on protected rows."""

    m4 = _probabilities(m4_probabilities, "m4_probabilities")
    a3 = _probabilities(a3_probabilities, "a3_probabilities")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    protected = np.asarray(protected, dtype=bool)
    if (
        m4.shape != a3.shape
        or coefficients.shape != (3,)
        or protected.shape != (len(m4),)
        or not np.isfinite(coefficients).all()
        or (coefficients < 0).any()
        or (coefficients > 0.25).any()
        or not 0 < epsilon < 1
    ):
        raise ValueError("Class-diagonal residual inputs violate the locked contract")
    output = m4.copy()
    active = ~protected
    if active.any() and np.any(coefficients > 0):
        anchor = np.log(np.clip(m4[active], epsilon, 1.0))
        expert = np.log(np.clip(a3[active], epsilon, 1.0))
        logits = anchor + coefficients[None, :] * (expert - anchor)
        logits -= logits.max(1, keepdims=True)
        exponential = np.exp(logits)
        output[active] = exponential / exponential.sum(1, keepdims=True)
    if not np.array_equal(output[protected], m4[protected]):
        raise RuntimeError("Protected sitting-disagreement fallback is not bit-exact")
    return output


@dataclass(frozen=True)
class ClassDiagonalFit:
    coefficients: np.ndarray
    objective: float
    training_nll: float
    anchor_nll: float
    iterations: int
    function_evaluations: int
    gradient_norm: float
    success: bool
    message: str


def fit_class_diagonal_residual(
    m4_probabilities: np.ndarray,
    a3_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    protected: np.ndarray,
    coefficient_cap: float,
    l2: float,
    epsilon: float,
    maximum_iterations: int,
    tolerance: float,
) -> ClassDiagonalFit:
    """Minimize mean training NLL plus fixed L2 over exactly three coefficients."""

    m4 = _probabilities(m4_probabilities, "m4_probabilities")
    a3 = _probabilities(a3_probabilities, "a3_probabilities")
    labels = np.asarray(labels, dtype=np.int64)
    protected = np.asarray(protected, dtype=bool)
    if (
        m4.shape != a3.shape
        or labels.shape != (len(m4),)
        or protected.shape != (len(m4),)
        or not np.array_equal(np.unique(labels), np.arange(3))
        or coefficient_cap != 0.25
        or l2 != 0.1
    ):
        raise ValueError("Class-diagonal fitting inputs violate the locked contract")
    log_m4 = np.log(np.clip(m4, epsilon, 1.0))
    residual = np.log(np.clip(a3, epsilon, 1.0)) - log_m4
    active = ~protected
    target = np.eye(3, dtype=np.float64)[labels]

    def objective(alpha: np.ndarray) -> tuple[float, np.ndarray]:
        logits = log_m4.copy()
        logits[active] += alpha[None, :] * residual[active]
        logits -= logits.max(1, keepdims=True)
        exponential = np.exp(logits)
        probabilities = exponential / exponential.sum(1, keepdims=True)
        nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], epsilon, 1)).mean()
        gradient = ((probabilities - target) * residual * active[:, None]).mean(0)
        return float(nll + l2 * np.dot(alpha, alpha)), gradient + 2 * l2 * alpha

    result = minimize(
        objective,
        np.zeros(3, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, coefficient_cap),) * 3,
        options={"maxiter": maximum_iterations, "ftol": tolerance, "gtol": tolerance},
    )
    coefficients = np.asarray(result.x, dtype=np.float64)
    if (
        not result.success
        or coefficients.shape != (3,)
        or (coefficients < -1e-12).any()
        or (coefficients > coefficient_cap + 1e-12).any()
    ):
        raise RuntimeError(f"Class-diagonal optimization failed: {result.message}")
    coefficients = np.clip(coefficients, 0.0, coefficient_cap)
    output = apply_class_diagonal_residual(
        m4, a3, coefficients, protected=protected, epsilon=epsilon
    )
    training_nll = float(
        -np.log(np.clip(output[np.arange(len(labels)), labels], epsilon, 1.0)).mean()
    )
    anchor_nll = float(-np.log(np.clip(m4[np.arange(len(labels)), labels], epsilon, 1)).mean())
    _, gradient = objective(coefficients)
    return ClassDiagonalFit(
        coefficients=coefficients,
        objective=float(training_nll + l2 * np.dot(coefficients, coefficients)),
        training_nll=training_nll,
        anchor_nll=anchor_nll,
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        gradient_norm=float(np.linalg.norm(gradient)),
        success=True,
        message=str(result.message),
    )
