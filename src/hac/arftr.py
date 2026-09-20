"""Anchor-restored factorized temporal residuals for three-class HAC.

The transform keeps an M4 probability as its anchor.  It restores a bounded
fraction of the P6 posture/motion logits, adds only A3's upright-motion
residual, and optionally averages factor logits with exact one-second track
neighbours.  No labels are accepted by the inference functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ARFTRParameters:
    posture_restoration: float
    motion_restoration: float
    a3_motion_residual: float
    temporal_strength: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.posture_restoration,
                self.motion_restoration,
                self.a3_motion_residual,
                self.temporal_strength,
            ),
            dtype=np.float64,
        )


def _probabilities(values: np.ndarray, description: str) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"Malformed three-class probabilities: {description}")
    return values


def factor_scores(probabilities: np.ndarray, *, epsilon: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Return sitting-vs-upright and walking-vs-standing log odds."""

    probabilities = _probabilities(probabilities, "factor_scores")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    clipped = np.clip(probabilities.astype(np.float64, copy=False), epsilon, 1.0)
    posture = np.log(clipped[:, 0] / (clipped[:, 1] + clipped[:, 2]))
    motion = np.log(clipped[:, 2] / clipped[:, 1])
    return posture, motion


def decode_factor_scores(posture: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Decode the two factor logits to sitting/standing/walking probabilities."""

    posture, motion = np.asarray(posture, dtype=np.float64), np.asarray(motion, dtype=np.float64)
    if posture.ndim != 1 or motion.shape != posture.shape or not (
        np.isfinite(posture).all() and np.isfinite(motion).all()
    ):
        raise ValueError("Factor scores must be aligned finite vectors")
    sitting = 1.0 / (1.0 + np.exp(-np.clip(posture, -30.0, 30.0)))
    walking_share = 1.0 / (1.0 + np.exp(-np.clip(motion, -30.0, 30.0)))
    upright = 1.0 - sitting
    result = np.column_stack((sitting, upright * (1.0 - walking_share), upright * walking_share))
    result /= result.sum(1, keepdims=True)
    return result


def exact_track_neighbors(data: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Map exact -1/+1 second neighbours to local row positions.

    Valid temporal links must remain inside the supplied population, scenario,
    outer fold, recording and track.  A declared valid +/-1 slot with a
    non-one-second frame displacement is rejected instead of silently used.
    """

    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or not np.array_equal(rows, np.unique(rows)):
        raise ValueError("rows must be a non-empty sorted unique vector")
    required = {
        "neighbor_indices",
        "valid",
        "times",
        "frames",
        "recordings",
        "tracks",
        "scenarios",
        "folds",
    }
    if not required.issubset(data):
        raise ValueError("Temporal metadata is incomplete")
    total = len(data["frames"])
    if rows.min() < 0 or rows.max() >= total:
        raise ValueError("rows are outside temporal metadata")
    if data["neighbor_indices"].shape != (total, 5) or data["valid"].shape != (total, 5):
        raise ValueError("Unexpected five-slot temporal metadata shape")
    if not np.array_equal(np.asarray(data["times"])[0], np.asarray([-2, -1, 0, 1, 2])):
        raise ValueError("Temporal slot ordering changed")
    local = np.full(total, -1, dtype=np.int64)
    local[rows] = np.arange(len(rows))
    result = np.full((len(rows), 2), -1, dtype=np.int64)
    for target_local, target in enumerate(rows):
        for output_slot, source_slot in enumerate((1, 3)):
            if not bool(data["valid"][target, source_slot]):
                continue
            source = int(data["neighbor_indices"][target, source_slot])
            expected_delta = -30 if source_slot == 1 else 30
            if source < 0 or int(data["frames"][source]) - int(data["frames"][target]) != expected_delta:
                raise RuntimeError("A valid +/-1 second edge is not exactly 30 signed frames")
            for name in ("recordings", "tracks", "scenarios", "folds"):
                if data[name][source] != data[name][target]:
                    raise RuntimeError(f"Temporal edge crosses {name}")
            if local[source] < 0:
                raise RuntimeError("Exact temporal edge escapes its prediction population")
            result[target_local, output_slot] = local[source]
    return result


def shuffled_neighbors(
    actual: np.ndarray,
    scenarios: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Destroy temporal identity while preserving per-row neighbour counts.

    Pseudo-neighbours are sampled deterministically inside the same scenario.
    Self and true exact neighbours are excluded.  This is a negative control,
    not an inference strategy.
    """

    actual = np.asarray(actual, dtype=np.int64)
    scenarios = np.asarray(scenarios)
    if actual.ndim != 2 or actual.shape[1] != 2 or scenarios.shape != (len(actual),):
        raise ValueError("Shuffled-neighbour inputs are misaligned")
    rng = np.random.default_rng(int(seed))
    result = np.full_like(actual, -1)
    for scenario in np.unique(scenarios):
        members = np.flatnonzero(scenarios == scenario)
        if len(members) < 4:
            raise RuntimeError("Scenario is too small for a temporal shuffle control")
        for target in members:
            count = int(np.sum(actual[target] >= 0))
            if count == 0:
                continue
            forbidden = {int(target), *(int(value) for value in actual[target] if value >= 0)}
            candidates = np.asarray([value for value in members if int(value) not in forbidden])
            if len(candidates) < count:
                raise RuntimeError("No valid within-scenario shuffled neighbours")
            result[target, :count] = rng.choice(candidates, size=count, replace=False)
    if not np.array_equal(np.sum(result >= 0, axis=1), np.sum(actual >= 0, axis=1)):
        raise AssertionError("Shuffle did not preserve neighbour counts")
    return result


def apply_arftr(
    m4: np.ndarray,
    p6: np.ndarray,
    a3: np.ndarray,
    neighbors: np.ndarray,
    parameters: ARFTRParameters,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Apply one bounded factorized residual and temporal message pass."""

    m4 = _probabilities(m4, "M4")
    p6 = _probabilities(p6, "P6")
    a3 = _probabilities(a3, "A3")
    if p6.shape != m4.shape or a3.shape != m4.shape:
        raise ValueError("M4, P6 and A3 probabilities must align")
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if neighbors.shape != (len(m4), 2) or ((neighbors < -1) | (neighbors >= len(m4))).any():
        raise ValueError("Neighbor map is malformed")
    coefficient = parameters.as_array()
    if not np.isfinite(coefficient).all() or (coefficient < 0).any():
        raise ValueError("ARFTR coefficients must be finite and nonnegative")
    # The control/fallback path must preserve the exact M4 bytes, including its
    # original dtype.  It must not decode through log odds.
    if np.count_nonzero(coefficient) == 0:
        return m4.copy()
    m4_posture, m4_motion = factor_scores(m4, epsilon=epsilon)
    p6_posture, p6_motion = factor_scores(p6, epsilon=epsilon)
    _, a3_motion = factor_scores(a3, epsilon=epsilon)
    posture = m4_posture + coefficient[0] * (p6_posture - m4_posture)
    motion = (
        m4_motion
        + coefficient[1] * (p6_motion - m4_motion)
        + coefficient[2] * (a3_motion - m4_motion)
    )
    beta = coefficient[3]
    if beta:
        temporal_posture, temporal_motion = posture.copy(), motion.copy()
        for row in range(len(m4)):
            sources = neighbors[row, neighbors[row] >= 0]
            if len(sources):
                temporal_posture[row] = posture[row] + beta * (posture[sources].mean() - posture[row])
                temporal_motion[row] = motion[row] + beta * (motion[sources].mean() - motion[row])
        posture, motion = temporal_posture, temporal_motion
    return decode_factor_scores(posture, motion)
