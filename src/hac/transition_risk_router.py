"""Leakage-safe two-action router for an ARFTR anchor and a residual candidate.

The router is intentionally conservative: it can retain the anchor exactly or
apply the candidate, and it never has an action that forces a broad overwrite.
Labels are accepted only by ``fit_transition_risk_router``; the caller is
responsible for supplying inner-cross-fitted outer-training rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape [rows, 3]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return values


def _entropy(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _margin(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def router_features(
    anchor_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    availability: np.ndarray | None = None,
    quality: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build label-blind action-risk features from aligned probabilities.

    ``availability`` and ``quality`` are optional inference-time summaries. No
    scenario, support annotation, or label-derived column is accepted here.
    """

    anchor = _probabilities(anchor_probabilities, "anchor_probabilities")
    candidate = _probabilities(candidate_probabilities, "candidate_probabilities")
    if anchor.shape != candidate.shape:
        raise ValueError("anchor and candidate probabilities must align")
    rows = len(anchor)
    anchor_class = anchor.argmax(axis=1)
    candidate_class = candidate.argmax(axis=1)
    midpoint = 0.5 * (anchor + candidate)
    js = 0.5 * (
        np.sum(anchor * (np.log(np.clip(anchor, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0))), axis=1)
        + np.sum(candidate * (np.log(np.clip(candidate, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0))), axis=1)
    )
    values = [
        np.log(np.clip(anchor, 1e-12, 1.0)),
        np.log(np.clip(candidate, 1e-12, 1.0)),
        np.log(np.clip(candidate, 1e-12, 1.0)) - np.log(np.clip(anchor, 1e-12, 1.0)),
        np.column_stack(
            (
                _entropy(anchor),
                _entropy(candidate),
                _margin(anchor),
                _margin(candidate),
                anchor.max(axis=1),
                candidate.max(axis=1),
                candidate.max(axis=1) - anchor.max(axis=1),
                np.abs(candidate - anchor).sum(axis=1),
                js,
                (anchor_class != candidate_class).astype(np.float64),
            )
        ),
    ]
    names = (
        [f"log_anchor_{index}" for index in range(3)]
        + [f"log_candidate_{index}" for index in range(3)]
        + [f"log_delta_{index}" for index in range(3)]
        + [
            "anchor_entropy",
            "candidate_entropy",
            "anchor_margin",
            "candidate_margin",
            "anchor_confidence",
            "candidate_confidence",
            "confidence_advantage",
            "l1_disagreement",
            "jensen_shannon",
            "class_disagreement",
        ]
    )
    if availability is not None:
        available = np.asarray(availability, dtype=np.float64)
        if available.ndim == 1:
            available = available[:, None]
        if available.shape[0] != rows or not np.isfinite(available).all():
            raise ValueError("availability must be finite and row-aligned")
        values.append(available)
        names.extend(f"availability_{index}" for index in range(available.shape[1]))
    if quality is not None:
        quality_values = np.asarray(quality, dtype=np.float64)
        if quality_values.ndim == 1:
            quality_values = quality_values[:, None]
        if quality_values.shape[0] != rows or not np.isfinite(quality_values).all():
            raise ValueError("quality must be finite and row-aligned")
        values.append(quality_values)
        names.extend(f"quality_{index}" for index in range(quality_values.shape[1]))
    matrix = np.column_stack(values)
    if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise RuntimeError("transition-risk feature assembly failed")
    return matrix, names


@dataclass
class TransitionRiskRouter:
    scaler: StandardScaler | None
    model: LogisticRegression | None
    feature_names: list[str]
    threshold: float
    harm_cost: float
    training_summary: dict[str, float | int | bool]

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("router feature schema changed")
        if self.model is None or self.scaler is None:
            return np.zeros(len(values), dtype=np.float64)
        return self.model.predict_proba(self.scaler.transform(values))[:, 1]

    def apply(
        self,
        anchor_probabilities: np.ndarray,
        candidate_probabilities: np.ndarray,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return probabilities and an explicit apply-candidate action mask."""

        anchor = _probabilities(anchor_probabilities, "anchor_probabilities")
        candidate = _probabilities(candidate_probabilities, "candidate_probabilities")
        if anchor.shape != candidate.shape or len(features) != len(anchor):
            raise ValueError("router inputs are not aligned")
        action = (self.score(features) >= self.threshold) & (anchor.argmax(1) != candidate.argmax(1))
        output = anchor.copy()
        output[action] = candidate[action]
        return output, action


def fit_transition_risk_router(
    features: np.ndarray,
    feature_names: list[str],
    anchor_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    harm_cost: float = 2.0,
    c: float = 0.1,
    maximum_iterations: int = 2000,
    tolerance: float = 1e-8,
    random_state: int = 42,
    threshold_grid: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> TransitionRiskRouter:
    """Fit on inner cross-fitted rows and choose a utility threshold therein.

    This function must never receive an outer-test row. The caller's audit
    should record that fact in the experiment receipt.
    """

    if harm_cost < 1.0:
        raise ValueError("harm_cost must be at least one")
    anchor = _probabilities(anchor_probabilities, "anchor_probabilities")
    candidate = _probabilities(candidate_probabilities, "candidate_probabilities")
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(features, dtype=np.float64)
    if labels.shape != (len(anchor),) or values.shape != (len(anchor), len(feature_names)):
        raise ValueError("router training inputs do not align")
    disagreement = anchor.argmax(1) != candidate.argmax(1)
    benefit = (candidate.argmax(1) == labels) & (anchor.argmax(1) != labels)
    harm = (anchor.argmax(1) == labels) & (candidate.argmax(1) != labels)
    usable = disagreement
    summary: dict[str, float | int | bool] = {
        "rows": int(usable.sum()),
        "benefits": int(benefit.sum()),
        "harms": int(harm.sum()),
        "neither_correct": int((usable & ~benefit & ~harm).sum()),
        "harm_cost": float(harm_cost),
        "constant_fallback": False,
    }
    if not usable.any() or len(np.unique(benefit[usable].astype(np.int64))) != 2:
        summary["constant_fallback"] = True
        return TransitionRiskRouter(None, None, feature_names, 1.0, harm_cost, summary)
    scaler = StandardScaler().fit(values[usable])
    target = benefit[usable].astype(np.int64)
    weights = np.ones(len(target), dtype=np.float64)
    weights[harm[usable]] = harm_cost
    model = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=maximum_iterations,
        tol=tolerance,
        random_state=random_state,
    ).fit(scaler.transform(values[usable]), target, sample_weight=weights)
    iterations = int(model.n_iter_.max())
    if iterations >= maximum_iterations:
        raise RuntimeError("transition-risk router did not converge")
    summary["converged"] = True
    summary["iterations"] = iterations
    scores = model.predict_proba(scaler.transform(values[usable]))[:, 1]
    best = (0.0, 1.0)
    for threshold in threshold_grid:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold grid must lie in (0, 1]")
        apply = scores >= threshold
        utility = float((benefit[usable][apply]).sum() - harm_cost * (harm[usable][apply]).sum())
        candidate_key = (utility, threshold)
        if candidate_key > best:
            best = candidate_key
    summary["inner_selected_utility"] = float(best[0])
    summary["inner_selected_threshold"] = float(best[1])
    summary["weighted_positive_prior"] = float(np.average(target, weights=weights))
    return TransitionRiskRouter(scaler, model, feature_names, float(best[1]), harm_cost, summary)
