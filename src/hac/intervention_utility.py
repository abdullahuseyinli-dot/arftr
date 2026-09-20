"""Fixed two-criterion verifier for bounded ARFTR residual actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _probabilities(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if (
        values.shape != shape or not np.isfinite(values).all() or np.any(values < 0)
        or not np.allclose(values.sum(-1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"malformed {name}")
    return values


def verifier_features(
    anchor: np.ndarray,
    actions: np.ndarray,
    correction: np.ndarray,
    scaffold_norms: np.ndarray,
    quality: np.ndarray,
    availability: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    rows = len(anchor)
    anchor = _probabilities(anchor, (rows, 3), "anchor")
    actions = _probabilities(actions, (rows, 4, 3), "actions")
    correction = np.asarray(correction, dtype=np.float64)
    scaffold_norms = np.asarray(scaffold_norms, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if correction.shape != (rows, 2) or scaffold_norms.shape != (rows, 4):
        raise ValueError("correction/scaffold schema changed")
    if quality.ndim != 2 or len(quality) != rows:
        raise ValueError("quality schema changed")
    if availability is None:
        availability = np.ones((rows, 1), dtype=np.float64)
    availability = np.asarray(availability, dtype=np.float64).reshape(rows, -1)
    if not all(np.isfinite(x).all() for x in (correction, scaffold_norms, quality, availability)):
        raise ValueError("nonfinite verifier input")
    log_anchor = np.log(np.clip(anchor, 1e-12, 1.0))
    ordered = np.sort(anchor, axis=1)
    entropy = -(anchor * log_anchor).sum(1, keepdims=True)
    base = np.column_stack((log_anchor, ordered[:, -1] - ordered[:, -2], entropy,
                            scaffold_norms, quality, availability))
    base_names = ([f"log_anchor_{i}" for i in range(3)] + ["anchor_margin", "anchor_entropy"]
                  + [f"coarse_scaffold_norm_{i}" for i in range(4)]
                  + [f"quality_{i}" for i in range(quality.shape[1])]
                  + [f"available_{i}" for i in range(availability.shape[1])])
    matrices = []
    names = None
    for action in range(1, 4):
        candidate = actions[:, action]
        log_delta = np.log(np.clip(candidate, 1e-12, 1.0)) - log_anchor
        margins = np.sort(candidate, axis=1)
        action_values = np.column_stack((
            base, correction * np.asarray(
                ([1, 0] if action == 1 else [0, 1] if action == 2 else [1, 1]),
                dtype=np.float64,
            ),
            log_delta,
            margins[:, -1] - margins[:, -2],
            (candidate.argmax(1) != anchor.argmax(1)).astype(np.float64),
        ))
        matrices.append(action_values)
        current = base_names + ["posture_delta", "motion_delta"] + [f"logit_delta_{i}" for i in range(3)] + ["candidate_margin", "crosses_class_boundary"]
        names = current if names is None else names
        if current != names:
            raise RuntimeError("action feature schema differs")
    result = np.stack(matrices, axis=1)
    if result.shape != (rows, 3, len(names)):
        raise RuntimeError("verifier feature assembly failed")
    return result, names


@dataclass
class InterventionVerifier:
    scaler: StandardScaler
    class_models: tuple[Ridge, Ridge, Ridge]
    nll_models: tuple[Ridge, Ridge, Ridge]
    feature_names: list[str]

    def scores(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (3, len(self.feature_names)):
            raise ValueError("verifier feature schema changed")
        transformed = self.scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape)
        class_scores = np.column_stack([m.predict(transformed[:, i]) for i, m in enumerate(self.class_models)])
        nll_scores = np.column_stack([m.predict(transformed[:, i]) for i, m in enumerate(self.nll_models)])
        return class_scores, nll_scores

    def apply(self, anchor: np.ndarray, actions: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = len(anchor)
        anchor = _probabilities(anchor, (rows, 3), "anchor")
        actions = _probabilities(actions, (rows, 4, 3), "actions")
        class_scores, nll_scores = self.scores(features)
        eligible = (class_scores > 0) & (nll_scores > 0)
        masked = np.where(eligible, class_scores, -np.inf)
        best = masked.argmax(1)
        active = eligible.any(1)
        choices = np.zeros(rows, dtype=np.int64)
        choices[active] = best[active] + 1
        output = anchor.copy()
        output[active] = actions[np.flatnonzero(active), choices[active]]
        return output, choices


def fit_intervention_verifier(
    features: np.ndarray,
    anchor: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 10.0,
    rescue_utility: float = 1.0,
    harm_utility: float = -2.0,
) -> tuple[InterventionVerifier, dict]:
    rows = len(anchor)
    anchor = _probabilities(anchor, (rows, 3), "anchor")
    actions = _probabilities(actions, (rows, 4, 3), "actions")
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(features, dtype=np.float64)
    if labels.shape != (rows,) or values.ndim != 3 or values.shape[:2] != (rows, 3):
        raise ValueError("verifier training inputs do not align")
    scaler = StandardScaler().fit(values.reshape(-1, values.shape[-1]))
    transformed = scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape)
    anchor_correct = anchor.argmax(1) == labels
    anchor_nll = -np.log(np.clip(anchor[np.arange(rows), labels], 1e-12, 1.0))
    class_models, nll_models, summaries = [], [], []
    for i, action in enumerate(range(1, 4)):
        candidate = actions[:, action]
        correct = candidate.argmax(1) == labels
        class_utility = np.zeros(rows, dtype=np.float64)
        class_utility[correct & ~anchor_correct] = rescue_utility
        class_utility[~correct & anchor_correct] = harm_utility
        nll_utility = anchor_nll + np.log(np.clip(candidate[np.arange(rows), labels], 1e-12, 1.0))
        cm = Ridge(alpha=alpha).fit(transformed[:, i], class_utility)
        nm = Ridge(alpha=alpha).fit(transformed[:, i], nll_utility)
        class_models.append(cm)
        nll_models.append(nm)
        summaries.append({"action": action, "rescues": int((correct & ~anchor_correct).sum()),
                          "harms": int((~correct & anchor_correct).sum()),
                          "positive_nll_utility": int((nll_utility > 0).sum())})
    names = [f"feature_{i}" for i in range(values.shape[-1])]
    return InterventionVerifier(scaler, tuple(class_models), tuple(nll_models), names), {
        "alpha": alpha, "rows": rows, "actions": summaries,
        "rescue_utility": rescue_utility, "harm_utility": harm_utility,
    }
