"""M4-anchored transition-gated residual utilities.

The module deliberately operates on cross-fitted cached outputs.  It contains no
scenario feature and preserves M4 bytes on every row whose gate is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _validated_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape [rows, 3]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(values.sum(1), 1.0, atol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return values


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def uncertainty_features(
    m4_probabilities: np.ndarray, a3_probabilities: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Return label-blind probability and disagreement features."""

    m4 = _validated_probabilities(m4_probabilities, "m4_probabilities")
    a3 = _validated_probabilities(a3_probabilities, "a3_probabilities")
    if m4.shape != a3.shape:
        raise ValueError("M4 and A3 probabilities must align")
    m4_class, a3_class = m4.argmax(1), a3.argmax(1)
    pair = np.eye(9, dtype=np.float64)[3 * m4_class + a3_class]
    midpoint = 0.5 * (m4 + a3)
    js = 0.5 * (
        np.sum(m4 * (np.log(np.clip(m4, 1e-12, 1)) - np.log(midpoint)), axis=1)
        + np.sum(a3 * (np.log(np.clip(a3, 1e-12, 1)) - np.log(midpoint)), axis=1)
    )
    scalar = np.column_stack(
        (
            _entropy(m4),
            _entropy(a3),
            _margin(m4),
            _margin(a3),
            m4.max(1),
            a3.max(1),
            a3.max(1) - m4.max(1),
            np.abs(a3 - m4).sum(1),
            js,
            m4_class != a3_class,
        )
    )
    values = np.column_stack(
        (
            np.log(np.clip(m4, 1e-12, 1)),
            np.log(np.clip(a3, 1e-12, 1)),
            np.log(np.clip(a3, 1e-12, 1)) - np.log(np.clip(m4, 1e-12, 1)),
            scalar,
            pair,
        )
    )
    names = (
        [f"log_m4_{index}" for index in range(3)]
        + [f"log_a3_{index}" for index in range(3)]
        + [f"log_residual_{index}" for index in range(3)]
        + [
            "m4_entropy",
            "a3_entropy",
            "m4_margin",
            "a3_margin",
            "m4_confidence",
            "a3_confidence",
            "confidence_advantage",
            "l1_disagreement",
            "jensen_shannon",
            "class_disagreement",
        ]
        + [f"predicted_pair_{left}_{right}" for left in range(3) for right in range(3)]
    )
    if not np.isfinite(values).all():
        raise ValueError("Uncertainty features are non-finite")
    return values, names


def _slot_summaries(slot_masses: np.ndarray) -> np.ndarray:
    masses = np.asarray(slot_masses, dtype=np.float64)
    if masses.ndim != 2 or len(masses) == 0 or not np.isfinite(masses).all():
        raise ValueError("slot_masses must be finite [rows, slots]")
    masses = np.maximum(masses, 0)
    total = masses.sum(1, keepdims=True)
    normalized = np.divide(
        masses,
        total,
        out=np.full_like(masses, 1.0 / masses.shape[1]),
        where=total > 0,
    )
    entropy = _entropy(normalized)
    return np.column_stack(
        (
            entropy,
            np.exp(entropy),
            normalized.max(1),
            (normalized >= 0.05).sum(1),
        )
    )


def _temporal_summaries(memory_features: np.ndarray) -> np.ndarray:
    values = np.asarray(memory_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3072 or not np.isfinite(values).all():
        raise ValueError("memory_features must be finite [rows, 3072]")
    streams = values.reshape(len(values), 4, 768)
    norms = np.linalg.norm(streams, axis=2) / np.sqrt(768.0)
    derived = []
    for short_index, long_index in ((0, 1), (2, 3)):
        short, long = streams[:, short_index], streams[:, long_index]
        raw_norms = np.linalg.norm(short, axis=1) * np.linalg.norm(long, axis=1)
        cosine = np.divide(
            np.sum(short * long, axis=1),
            raw_norms,
            out=np.zeros(len(values)),
            where=raw_norms > 0,
        )
        delta = np.linalg.norm(long - short, axis=1) / np.sqrt(768.0)
        relative = delta / np.maximum(norms[:, short_index] + norms[:, long_index], 1e-12)
        derived.extend((cosine, delta, relative))
    return np.column_stack((norms, *derived))


def boundary_context_features(
    m4_probabilities: np.ndarray,
    a3_probabilities: np.ndarray,
    *,
    m4_gate: np.ndarray,
    attention: np.ndarray,
    survival: np.ndarray,
    boundary_probabilities: np.ndarray,
    slot_masses: np.ndarray,
    quality: np.ndarray,
    memory_features: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return uncertainty plus predicted-transition and compact source features."""

    uncertainty, names = uncertainty_features(m4_probabilities, a3_probabilities)
    rows = len(uncertainty)
    gate = np.asarray(m4_gate, dtype=np.float64).reshape(-1, 1)
    attention = np.asarray(attention, dtype=np.float64)
    survival = np.asarray(survival, dtype=np.float64)
    boundary = np.asarray(boundary_probabilities, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if (
        gate.shape != (rows, 1)
        or attention.shape != (rows, 5)
        or survival.shape != (rows, 5)
        or boundary.shape != (rows, 5)
        or quality.shape != (rows, 6)
    ):
        raise ValueError("MATR context arrays do not follow the locked shape contract")
    if any(not np.isfinite(value).all() for value in (gate, attention, survival, boundary, quality)):
        raise ValueError("MATR context arrays must be finite")
    expected_boundary = np.sum(attention * boundary, axis=1, keepdims=True)
    boundary_summary = np.column_stack(
        (boundary.max(1), boundary.mean(1), expected_boundary[:, 0])
    )
    slots = _slot_summaries(slot_masses)
    temporal = _temporal_summaries(memory_features)
    values = np.column_stack(
        (
            uncertainty,
            gate,
            attention,
            survival,
            boundary,
            boundary_summary,
            slots,
            quality,
            temporal,
        )
    )
    feature_names = names + (
        ["memory_gate"]
        + [f"attention_{index}" for index in range(5)]
        + [f"survival_{index}" for index in range(5)]
        + [f"boundary_probability_{index}" for index in range(5)]
        + ["boundary_max", "boundary_mean", "boundary_attention_mean"]
        + ["slot_entropy", "effective_slots", "slot_max_share", "active_slots_005"]
        + [f"quality_{index}" for index in range(6)]
        + [f"stream_norm_{index}" for index in range(4)]
        + [
            "vjepa_short_long_cosine",
            "vjepa_short_long_delta",
            "vjepa_short_long_relative_delta",
            "dino_short_long_cosine",
            "dino_short_long_delta",
            "dino_short_long_relative_delta",
        ]
    )
    if values.shape[1] != len(feature_names) or not np.isfinite(values).all():
        raise RuntimeError("MATR feature assembly failed")
    return values, feature_names


def predicted_boundary_score(attention: np.ndarray, boundary_probabilities: np.ndarray) -> np.ndarray:
    attention = np.asarray(attention, dtype=np.float64)
    boundary = np.asarray(boundary_probabilities, dtype=np.float64)
    if attention.ndim != 2 or attention.shape != boundary.shape or attention.shape[1] != 5:
        raise ValueError("attention and boundary probabilities must align as [rows, 5]")
    if not np.isfinite(attention).all() or not np.isfinite(boundary).all():
        raise ValueError("Predicted boundary inputs must be finite")
    return np.maximum(boundary.max(1), np.sum(attention * boundary, axis=1))


def select_boundary_threshold(
    score: np.ndarray, target: np.ndarray, valid: np.ndarray, thresholds: list[float]
) -> tuple[float, list[dict[str, float | int]]]:
    """Select on outer-training boundary labels only; prefer higher threshold on ties."""

    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if score.shape != target.shape or score.shape != valid.shape or not valid.any():
        raise ValueError("Boundary threshold inputs must be aligned with at least one valid row")
    candidates = []
    for threshold in thresholds:
        predicted = score[valid] >= threshold
        truth = target[valid]
        tp = int(np.sum(predicted & truth))
        fp = int(np.sum(predicted & ~truth))
        fn = int(np.sum(~predicted & truth))
        denominator = 2 * tp + fp + fn
        f1 = float(2 * tp / denominator) if denominator else 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "boundary_f1": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "selected_rows": int(predicted.sum()),
            }
        )
    selected = max(candidates, key=lambda value: (value["boundary_f1"], value["threshold"]))
    return float(selected["threshold"]), candidates


@dataclass
class CorrectionGate:
    scaler: StandardScaler | None
    model: LogisticRegression | None
    feature_names: list[str]
    training_summary: dict[str, float | int | bool]

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("Correction-gate feature shape changed")
        if self.model is None or self.scaler is None:
            return np.zeros(len(features), dtype=np.float64)
        return self.model.predict_proba(self.scaler.transform(features))[:, 1]

    def checkpoint(self, prefix: str) -> dict[str, np.ndarray]:
        if self.model is None or self.scaler is None:
            return {
                f"{prefix}_constant_probability": np.array([0.0], dtype=np.float64),
                f"{prefix}_feature_names": np.asarray(self.feature_names),
            }
        return {
            f"{prefix}_coefficients": self.model.coef_,
            f"{prefix}_intercept": self.model.intercept_,
            f"{prefix}_classes": self.model.classes_,
            f"{prefix}_iterations": self.model.n_iter_,
            f"{prefix}_scaler_mean": self.scaler.mean_,
            f"{prefix}_scaler_scale": self.scaler.scale_,
            f"{prefix}_feature_names": np.asarray(self.feature_names),
        }


def fit_correction_gate(
    features: np.ndarray,
    feature_names: list[str],
    m4_probabilities: np.ndarray,
    a3_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    c: float,
    harm_weight: float,
    maximum_iterations: int,
    tolerance: float,
    random_state: int,
) -> CorrectionGate:
    """Fit repair utility on disagreements from an outer-training OOF population."""

    m4 = _validated_probabilities(m4_probabilities, "m4_probabilities")
    a3 = _validated_probabilities(a3_probabilities, "a3_probabilities")
    labels = np.asarray(labels, dtype=np.int64)
    features = np.asarray(features, dtype=np.float64)
    if (
        m4.shape != a3.shape
        or labels.shape != (len(m4),)
        or features.shape != (len(m4), len(feature_names))
        or not np.isfinite(features).all()
    ):
        raise ValueError("Correction-gate training inputs do not align")
    disagreement = m4.argmax(1) != a3.argmax(1)
    benefit = (a3.argmax(1) == labels) & (m4.argmax(1) != labels)
    harm = (m4.argmax(1) == labels) & (a3.argmax(1) != labels)
    target = benefit[disagreement].astype(np.int64)
    summary: dict[str, float | int | bool] = {
        "rows": int(disagreement.sum()),
        "benefits": int(benefit.sum()),
        "harms": int(harm.sum()),
        "neither_correct": int((disagreement & ~benefit & ~harm).sum()),
        "harm_weight": float(harm_weight),
        "constant_fallback": False,
    }
    if len(target) == 0 or len(np.unique(target)) != 2:
        summary["constant_fallback"] = True
        return CorrectionGate(None, None, feature_names, summary)
    scaler = StandardScaler().fit(features[disagreement])
    standardized = scaler.transform(features[disagreement])
    weights = np.ones(len(target), dtype=np.float64)
    weights[harm[disagreement]] = harm_weight
    model = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=maximum_iterations,
        tol=tolerance,
        random_state=random_state,
    ).fit(standardized, target, sample_weight=weights)
    converged = bool(int(model.n_iter_.max()) < maximum_iterations)
    summary["converged"] = converged
    summary["iterations"] = int(model.n_iter_.max())
    summary["weighted_positive_prior"] = float(np.average(target, weights=weights))
    if not converged:
        raise RuntimeError("Correction gate did not converge within the locked budget")
    return CorrectionGate(scaler, model, feature_names, summary)


def smooth_gate(
    correction_probability: np.ndarray,
    disagreement: np.ndarray,
    *,
    activation_probability: float,
    full_gate_probability: float,
    suppress: np.ndarray | None = None,
) -> np.ndarray:
    probability = np.asarray(correction_probability, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=bool)
    if probability.shape != disagreement.shape or not np.isfinite(probability).all():
        raise ValueError("Gate probability and disagreement must align")
    if not 0 <= activation_probability < full_gate_probability <= 1:
        raise ValueError("Invalid gate activation interval")
    gate = np.clip(
        (probability - activation_probability)
        / (full_gate_probability - activation_probability),
        0.0,
        1.0,
    )
    gate[~disagreement] = 0.0
    if suppress is not None:
        suppress = np.asarray(suppress, dtype=bool)
        if suppress.shape != gate.shape:
            raise ValueError("Suppression mask must align")
        gate[suppress] = 0.0
    return gate


def geometric_residual(
    m4_probabilities: np.ndarray,
    a3_probabilities: np.ndarray,
    gate: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Blend in log space while retaining exact M4 bytes wherever gate is zero."""

    m4 = _validated_probabilities(m4_probabilities, "m4_probabilities")
    a3 = _validated_probabilities(a3_probabilities, "a3_probabilities")
    gate = np.asarray(gate, dtype=np.float64)
    if m4.shape != a3.shape or gate.shape != (len(m4),):
        raise ValueError("Residual inputs must align")
    if not np.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
        raise ValueError("Gate must be finite in [0, 1]")
    output = m4.copy()
    active = gate > 0
    if active.any():
        log_m4 = np.log(np.clip(m4[active], epsilon, 1.0))
        residual = np.log(np.clip(a3[active], epsilon, 1.0)) - log_m4
        logits = log_m4 + gate[active, None] * residual
        logits -= logits.max(1, keepdims=True)
        exp = np.exp(logits)
        output[active] = exp / exp.sum(1, keepdims=True)
    return output
