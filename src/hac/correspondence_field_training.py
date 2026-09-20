"""Leakage-resistant training and evaluation helpers for correspondence fields."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_field_training(seed: int) -> None:
    """Configure the deterministic training contract used by the screen."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def upright_class_weights(labels: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return mean-one inverse-frequency weights for standing and locomotion."""

    labels = np.asarray(labels, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or np.any(labels[rows] == 0):
        raise ValueError("Class weights require a nonempty upright-only population")
    binary = labels[rows] - 1
    if np.any((binary < 0) | (binary > 1)):
        raise ValueError("Upright labels must be standing or walking/running")
    counts = np.bincount(binary, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Both upright classes are required in the training population")
    weights = len(rows) / (2.0 * counts)
    if not np.isclose(weights[binary].mean(), 1.0):
        raise RuntimeError("Upright class weights are not normalized to mean one")
    return weights.astype(np.float32)


class ActorUniformSampler:
    """Uniformly sample actors, then uniformly sample one center per actor."""

    def __init__(
        self,
        rows: np.ndarray,
        recordings: np.ndarray,
        tracks: np.ndarray,
        *,
        seed: int,
    ) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        recordings = np.asarray(recordings)
        tracks = np.asarray(tracks)
        if (
            rows.ndim != 1
            or len(rows) == 0
            or recordings.ndim != 1
            or tracks.shape != recordings.shape
            or rows.min() < 0
            or rows.max() >= len(recordings)
        ):
            raise ValueError("Malformed actor-uniform sampling inputs")
        keys = np.char.add(
            np.char.add(recordings[rows].astype(str), "::"), tracks[rows].astype(str)
        )
        actors = np.unique(keys)
        if len(actors) == 0:
            raise ValueError("No actors are available for sampling")
        self._actors = actors
        self._rows = {actor: rows[keys == actor] for actor in actors}
        self._rng = np.random.default_rng(seed)

    @property
    def actor_count(self) -> int:
        return len(self._actors)

    def batch(self, batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        actor_indices = self._rng.integers(0, len(self._actors), size=batch_size)
        result = np.empty(batch_size, dtype=np.int64)
        for index, actor_index in enumerate(actor_indices):
            candidates = self._rows[self._actors[actor_index]]
            result[index] = candidates[self._rng.integers(0, len(candidates))]
        return result


def conditional_motion_candidate(
    anchor_probabilities: np.ndarray,
    motion_probability: np.ndarray,
    observation_available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the fixed candidate while byte-preserving every retained row."""

    anchor = np.asarray(anchor_probabilities)
    motion = np.asarray(motion_probability, dtype=np.float64)
    observed = np.asarray(observation_available, dtype=bool)
    if (
        anchor.ndim != 2
        or anchor.shape[1] != 3
        or motion.shape != (len(anchor),)
        or observed.shape != (len(anchor),)
        or not np.isfinite(anchor).all()
        or not np.isfinite(motion).all()
        or np.any(anchor < 0)
        or np.any((motion < 0) | (motion > 1))
        or not np.allclose(anchor.sum(1), 1.0, atol=1e-8)
    ):
        raise ValueError("Malformed conditional-motion candidate inputs")
    raw = np.column_stack(
        (
            anchor[:, 0],
            (1.0 - anchor[:, 0]) * (1.0 - motion),
            (1.0 - anchor[:, 0]) * motion,
        )
    ).astype(anchor.dtype, copy=False)
    anchor_class, raw_class = anchor.argmax(1), raw.argmax(1)
    eligible = observed & (anchor_class != raw_class) & (anchor_class != 0) & (raw_class != 0)
    candidate = anchor.copy()
    candidate[eligible] = raw[eligible]
    if not np.array_equal(candidate[~eligible], anchor[~eligible]):
        raise RuntimeError("Retain action failed to preserve exact anchor bytes")
    return candidate, eligible


def upright_balanced_accuracy(labels: np.ndarray, motion_probability: np.ndarray) -> float:
    """Balanced accuracy on true upright centers, with both classes required."""

    labels = np.asarray(labels, dtype=np.int64)
    motion = np.asarray(motion_probability, dtype=np.float64)
    if labels.shape != motion.shape or not np.isfinite(motion).all():
        raise ValueError("Malformed upright metric inputs")
    retained = labels > 0
    binary = labels[retained] - 1
    if not retained.any() or not np.array_equal(np.unique(binary), np.asarray([0, 1])):
        raise ValueError("Upright balanced accuracy requires both classes")
    # The candidate uses np.argmax([P(standing), P(walking)]), whose exact tie
    # resolves to standing. Keep the screen endpoint identical to that action.
    predicted = (motion[retained] > 0.5).astype(np.int64)
    recalls = [(predicted[binary == value] == value).mean() for value in (0, 1)]
    return float(np.mean(recalls))
