"""Bounded, ARFTR-preserving residual gate for fine-density evidence.

The module deliberately separates three concerns:

* ``build_features`` creates inference-only witnesses from anchor/fine/coarse
  probabilities and observable quality/clock metadata;
* ``build_actions`` creates the four fixed candidate distributions, with a
  0.25 factor-logit residual cap;
* ``FineDensityResidualGate`` predicts action utility and retains the anchor
  unless an intervention clears a threshold learned inside the training fold.

No function accepts labels as an inference input. Labels are used only by
``fit`` on an already cross-fitted training population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from hac.arftr import decode_factor_scores, factor_scores


ACTION_NAMES = ("retain_ARFTR", "apply_bounded_posture", "apply_bounded_motion", "apply_both")


def _probabilities(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite [rows,3] probabilities")
    if (value < 0).any() or not np.allclose(value.sum(1), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"{name} is not a probability simplex")
    return value


def _log_prob(value: np.ndarray) -> np.ndarray:
    return np.log(np.clip(value, 1e-8, 1.0))


def _entropy(value: np.ndarray) -> np.ndarray:
    return -np.sum(value * _log_prob(value), axis=1)


def _margin(value: np.ndarray) -> np.ndarray:
    ordered = np.sort(value, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    mean = 0.5 * (left + right)
    return 0.5 * np.sum(left * (_log_prob(left) - _log_prob(mean)), axis=1) + 0.5 * np.sum(
        right * (_log_prob(right) - _log_prob(mean)), axis=1
    )


def build_features(
    anchor: np.ndarray,
    fine: np.ndarray,
    coarse: np.ndarray,
    *,
    quality: np.ndarray | None = None,
    times: np.ndarray | None = None,
    moments: np.ndarray | None = None,
) -> np.ndarray:
    """Build finite inference-time witnesses for the residual gate.

    The first blocks are the three probability vectors and their logit
    contrasts.  Entropy/margin/JS blocks describe confidence and density
    disagreement.  Optional quality, clock, and moment blocks are observable
    metadata only; no annotation-derived support signal is accepted.
    """

    anchor, fine, coarse = (_probabilities(value, name) for value, name in ((anchor, "anchor"), (fine, "fine"), (coarse, "coarse")))
    if not (anchor.shape == fine.shape == coarse.shape):
        raise ValueError("anchor, fine, and coarse rows must align")
    rows = len(anchor)
    blocks = [anchor, fine, coarse, fine - coarse]
    blocks.extend(
        [
            _log_prob(anchor),
            _log_prob(fine),
            _log_prob(coarse),
            _log_prob(fine) - _log_prob(coarse),
        ]
    )
    blocks.extend(
        [
            _entropy(anchor)[:, None],
            _entropy(fine)[:, None],
            _entropy(coarse)[:, None],
            _js_divergence(fine, coarse)[:, None],
            _margin(anchor)[:, None],
            _margin(fine)[:, None],
            _margin(coarse)[:, None],
        ]
    )
    anchor_post, anchor_motion = factor_scores(anchor)
    fine_post, fine_motion = factor_scores(fine)
    coarse_post, coarse_motion = factor_scores(coarse)
    blocks.extend(
        [
            (fine_post - coarse_post)[:, None],
            (fine_motion - coarse_motion)[:, None],
            (fine_post - anchor_post)[:, None],
            (fine_motion - anchor_motion)[:, None],
        ]
    )
    if quality is not None:
        quality = np.asarray(quality, dtype=np.float64)
        if quality.ndim != 2 or len(quality) != rows or not np.isfinite(quality).all():
            raise ValueError("quality must be finite [rows,features]")
        blocks.append(quality)
    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 2 or len(times) != rows or times.shape[1] < 2 or not np.isfinite(times).all():
            raise ValueError("times must be finite [rows,time>=2]")
        dt = np.diff(times, axis=1)
        blocks.append(
            np.column_stack(
                [
                    times[:, -1] - times[:, 0],
                    dt.mean(1),
                    dt.std(1),
                    np.abs(times).mean(1),
                ]
            )
        )
    if moments is not None:
        moments = np.asarray(moments, dtype=np.float64)
        if moments.ndim != 2 or len(moments) != rows or not np.isfinite(moments).all():
            raise ValueError("moments must be finite [rows,features]")
        blocks.append(moments)
    features = np.concatenate(blocks, axis=1).astype(np.float64, copy=False)
    if features.shape[0] != rows or not np.isfinite(features).all():
        raise ValueError("Feature construction produced nonfinite values")
    return features


def build_actions(
    anchor: np.ndarray,
    fine: np.ndarray,
    coarse: np.ndarray,
    *,
    residual_cap: float = 0.25,
) -> np.ndarray:
    """Return [rows,4,3] fixed candidate distributions.

    Candidate 0 is an exact copy of ARFTR.  The other candidates move only the
    corresponding factor toward the fine-minus-coarse contrast, bounded by
    ``residual_cap`` factor-logit units.
    """

    if not np.isfinite(residual_cap) or residual_cap <= 0 or residual_cap > 0.25:
        raise ValueError("residual_cap must be in (0, 0.25]")
    anchor, fine, coarse = (_probabilities(value, name) for value, name in ((anchor, "anchor"), (fine, "fine"), (coarse, "coarse")))
    if not (anchor.shape == fine.shape == coarse.shape):
        raise ValueError("anchor, fine, and coarse rows must align")
    posture, motion = factor_scores(anchor)
    fine_posture, fine_motion = factor_scores(fine)
    coarse_posture, coarse_motion = factor_scores(coarse)
    post_delta = residual_cap * np.tanh(fine_posture - coarse_posture)
    motion_delta = residual_cap * np.tanh(fine_motion - coarse_motion)
    actions = np.empty((len(anchor), 4, 3), dtype=np.float64)
    actions[:, 0] = anchor
    actions[:, 1] = decode_factor_scores(posture + post_delta, motion)
    actions[:, 2] = decode_factor_scores(posture, motion + motion_delta)
    actions[:, 3] = decode_factor_scores(posture + post_delta, motion + motion_delta)
    if not np.isfinite(actions).all() or not np.allclose(actions.sum(2), 1.0, atol=1e-10, rtol=0):
        raise RuntimeError("Residual actions are not finite probability simplexes")
    return actions


@dataclass
class FineDensityResidualGate:
    """Utility-regression gate fitted only on cross-fitted training rows."""

    scaler: StandardScaler
    regressors: tuple[Ridge | None, Ridge, Ridge, Ridge]
    threshold: float
    residual_cap: float
    intervention_budget: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        anchor: np.ndarray,
        actions: np.ndarray,
        labels: np.ndarray,
        *,
        residual_cap: float = 0.25,
        intervention_budget: float = 0.15,
        ridge_alpha: float = 2.0,
    ) -> "FineDensityResidualGate":
        features = np.asarray(features, dtype=np.float64)
        anchor = _probabilities(anchor, "anchor")
        actions = np.asarray(actions, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if features.ndim != 2 or len(features) != len(anchor) or len(labels) != len(anchor):
            raise ValueError("Gate training arrays are misaligned")
        if actions.shape != (len(anchor), 4, 3) or not np.isfinite(actions).all():
            raise ValueError("actions must be finite [rows,4,3]")
        if not 0 < intervention_budget <= 0.30:
            raise ValueError("intervention_budget must be in (0, 0.30]")
        if not np.array_equal(np.unique(labels), np.arange(3)):
            raise ValueError("Gate training rows must contain all three classes")
        row_index = np.arange(len(labels))
        anchor_loss = -np.log(np.clip(anchor[row_index, labels], 1e-8, 1.0))
        # Explicit gather keeps the action/class axes unambiguous.
        action_loss = -np.log(np.clip(actions[np.arange(len(labels))[:, None], np.arange(4)[None, :], labels[:, None]], 1e-8, 1.0))
        utility = anchor_loss[:, None] - action_loss
        utility[:, 0] = 0.0
        scaler = StandardScaler().fit(features)
        transformed = scaler.transform(features)
        regressors: list[Ridge | None] = [None]
        for action in range(1, 4):
            model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
            model.fit(transformed, utility[:, action])
            regressors.append(model)
        predicted = np.column_stack(
            [np.zeros(len(features), dtype=np.float64)]
            + [regressors[action].predict(transformed) for action in range(1, 4)]
        )
        best = predicted[:, 1:].max(1)
        positive = best[best > 0]
        if len(positive):
            threshold = float(max(0.0, np.quantile(best, 1.0 - intervention_budget)))
        else:
            threshold = 0.0
        return cls(scaler, tuple(regressors), threshold, residual_cap, intervention_budget)

    def scores(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        transformed = self.scaler.transform(features)
        return np.column_stack(
            [np.zeros(len(features), dtype=np.float64)]
            + [self.regressors[action].predict(transformed) for action in range(1, 4)]
        )

    def choose(self, features: np.ndarray) -> np.ndarray:
        scores = self.scores(features)
        best_action = 1 + np.argmax(scores[:, 1:], axis=1)
        best_score = scores[np.arange(len(scores)), best_action]
        return np.where((best_score > self.threshold) & (best_score > 0), best_action, 0).astype(np.int64)

    def apply(self, features: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float64)
        choices = self.choose(features)
        if actions.ndim != 3 or actions.shape[1:] != (4, 3) or len(actions) != len(choices):
            raise ValueError("actions must be [rows,4,3] aligned to features")
        probabilities = actions[np.arange(len(actions)), choices]
        return probabilities, choices
