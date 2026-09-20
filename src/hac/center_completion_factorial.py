"""Pure, label-blind primitives for the matched center-completion factorial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorialAggregation:
    features: np.ndarray
    available: np.ndarray
    donor_count: np.ndarray


def _validate_inputs(
    donor_features: np.ndarray,
    donor_observed: np.ndarray,
    donor_logits: np.ndarray,
) -> None:
    if (
        donor_features.ndim != 3
        or donor_features.shape[0] != 4
        or donor_observed.shape != donor_features.shape[:2]
        or donor_logits.shape != donor_features.shape[:2]
        or donor_observed.dtype != np.bool_
        or not np.issubdtype(donor_features.dtype, np.floating)
        or not np.issubdtype(donor_logits.dtype, np.floating)
        or not np.isfinite(donor_features).all()
        or not np.isfinite(donor_logits[donor_observed]).all()
    ):
        raise ValueError("Factorial donor inputs differ from the locked 4,T,D contract")


def uniform_consensus(
    donor_features: np.ndarray,
    donor_observed: np.ndarray,
    donor_logits: np.ndarray,
) -> FactorialAggregation:
    """Arithmetic mean over the common donor support; logits are integrity-only."""

    _validate_inputs(donor_features, donor_observed, donor_logits)
    counts = donor_observed.sum(axis=0).astype(np.int64)
    available = counts > 0
    features = np.zeros(donor_features.shape[1:], dtype=np.float32)
    weighted = donor_features.astype(np.float32) * donor_observed[..., None]
    features[available] = (
        weighted.sum(axis=0)[available] / counts[available, None]
    )
    return FactorialAggregation(features, available, counts)


def _weighted_median_one(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorized feature-wise median matching the locked stable-tie semantics."""

    if values.ndim != 2 or weights.shape != (len(values),) or not len(values):
        raise ValueError("Weighted-median inputs are invalid")
    weights = weights / weights.sum()
    order = np.argsort(values, axis=0, kind="stable")
    ordered_weights = np.take_along_axis(
        np.broadcast_to(weights[:, None], values.shape), order, axis=0
    )
    cumulative = np.cumsum(ordered_weights, axis=0)
    position = np.argmax(cumulative >= 0.5, axis=0)
    columns = np.arange(values.shape[1])
    chosen = order[position, columns]

    cumulative_at = cumulative[position, columns]
    tied = np.isclose(cumulative_at, 0.5, rtol=0, atol=1e-12) & (
        position + 1 < len(values)
    )
    if tied.any():
        next_position = np.minimum(position + 1, len(values) - 1)
        next_donor = order[next_position, columns]
        chosen[tied] = np.minimum(chosen[tied], next_donor[tied])
    return values[chosen, columns].astype(np.float32)


def robust_consensus(
    donor_features: np.ndarray,
    donor_observed: np.ndarray,
    donor_logits: np.ndarray,
    *,
    temperature: float = 0.25,
) -> FactorialAggregation:
    """P3 componentwise confidence-weighted median on fixed common support."""

    _validate_inputs(donor_features, donor_observed, donor_logits)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    counts = donor_observed.sum(axis=0).astype(np.int64)
    available = counts > 0
    features = np.zeros(donor_features.shape[1:], dtype=np.float32)
    for target in np.flatnonzero(available):
        valid = np.flatnonzero(donor_observed[:, target])
        scaled = donor_logits[valid, target].astype(np.float64) / temperature
        weights = np.exp(scaled - scaled.max())
        weights /= weights.sum()
        features[target] = _weighted_median_one(
            donor_features[valid, target].astype(np.float32), weights
        )
    return FactorialAggregation(features, available, counts)


def cosine_errors(candidate: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    """Full-dimensional raw cosine error, one value per target token."""

    if (
        candidate.shape != teacher.shape
        or candidate.ndim != 2
        or not np.isfinite(candidate).all()
        or not np.isfinite(teacher).all()
    ):
        raise ValueError("Candidate and teacher tokens must be finite T,D arrays")
    candidate64 = candidate.astype(np.float64)
    teacher64 = teacher.astype(np.float64)
    denominator = np.linalg.norm(candidate64, axis=1) * np.linalg.norm(
        teacher64, axis=1
    )
    if (denominator <= 1e-12).any():
        raise ValueError("Raw cosine is undefined for a zero-norm token")
    similarity = np.sum(candidate64 * teacher64, axis=1) / denominator
    return (1.0 - np.clip(similarity, -1.0, 1.0)).astype(np.float64)


def relative_reduction(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(candidate) or reference <= 0:
        raise ValueError("Cosine errors must be finite and the reference positive")
    return float((reference - candidate) / reference)


def locked_factorial_decision(
    means: dict[str, float],
    fold_means: dict[str, list[float]],
) -> dict[str, object]:
    """Apply prospectively declared attribution gates to A/B/C/D raw errors."""

    if set(means) != {"A", "B", "C", "D"} or set(fold_means) != set(means):
        raise ValueError("The factorial requires exactly A, B, C, and D")
    if any(len(values) != 5 for values in fold_means.values()):
        raise ValueError("The factorial decision requires exactly five folds")
    if not all(np.isfinite(value) and value > 0 for value in means.values()):
        raise ValueError("Factorial means must be finite positive errors")

    transport_uniform = relative_reduction(means["A"], means["C"])
    transport_robust = relative_reduction(means["B"], means["D"])
    aggregation_same = relative_reduction(means["A"], means["B"])
    aggregation_affine = relative_reduction(means["C"], means["D"])
    total_gain = means["A"] - means["D"]
    uniform_transport_gain = means["A"] - means["C"]
    transport_share = uniform_transport_gain / total_gain if total_gain > 0 else None
    c_better_folds = sum(c < a for a, c in zip(fold_means["A"], fold_means["C"], strict=True))
    b_better_folds = sum(b < a for a, b in zip(fold_means["A"], fold_means["B"], strict=True))
    d_better_c_folds = sum(d < c for c, d in zip(fold_means["C"], fold_means["D"], strict=True))

    any_transport_five = max(transport_uniform, transport_robust) >= 0.05
    transport_survives = bool(
        any_transport_five
        and transport_share is not None
        and transport_share >= 0.50
        and c_better_folds >= 4
    )
    aggregation_survives = bool(
        aggregation_same >= 0.02
        and aggregation_affine >= 0.02
        and b_better_folds >= 4
        and d_better_c_folds >= 4
    )
    interaction_survives = bool(
        means["D"] == min(means.values())
        and transport_robust >= 0.05
        and aggregation_affine >= 0.02
        and not transport_survives
    )
    stop_transport_branch = bool(not any_transport_five)
    aggregation_dominates_uniform_transport = bool(
        means["A"] - means["B"] > uniform_transport_gain
    )
    return {
        "checks": {
            "at_least_one_matched_transport_contrast_reduction_min_0.05": any_transport_five,
            "C_minus_A_explains_half_of_D_minus_A": transport_share is not None
            and transport_share >= 0.50,
            "C_better_than_A_folds_min_4": c_better_folds >= 4,
            "aggregation_reduction_both_strata_min_0.02": aggregation_same >= 0.02
            and aggregation_affine >= 0.02,
            "aggregation_better_folds_both_strata_min_4": b_better_folds >= 4
            and d_better_c_folds >= 4,
        },
        "measurements": {
            "transport_uniform_relative_reduction_C_vs_A": transport_uniform,
            "transport_robust_relative_reduction_D_vs_B": transport_robust,
            "aggregation_same_grid_relative_reduction_B_vs_A": aggregation_same,
            "aggregation_affine_relative_reduction_D_vs_C": aggregation_affine,
            "uniform_transport_share_of_total_D_vs_A_gain": transport_share,
            "C_better_than_A_folds": c_better_folds,
            "B_better_than_A_folds": b_better_folds,
            "D_better_than_C_folds": d_better_c_folds,
        },
        "verdict": {
            "transport_survives": transport_survives,
            "aggregation_survives": aggregation_survives,
            "interaction_survives": interaction_survives,
            "aggregation_dominates_uniform_transport": aggregation_dominates_uniform_transport,
            "stop_transport_branch": stop_transport_branch,
        },
    }
