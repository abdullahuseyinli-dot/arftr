"""Visible-anchor partial transport primitives for center evidence completion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PATCH_GRID = 27


@dataclass(frozen=True)
class AnchorMatches:
    center_xy: np.ndarray
    neighbor_xy: np.ndarray
    cosine: np.ndarray


@dataclass(frozen=True)
class AffineTransport:
    matrix: np.ndarray
    match_count: int
    weighted_residual: float
    valid: bool
    reason: str


@dataclass(frozen=True)
class AggregatedTransport:
    features: np.ndarray
    used_transport: np.ndarray
    dustbin_posterior: np.ndarray
    entropy: np.ndarray
    valid_donor_count: np.ndarray


@dataclass(frozen=True)
class TransportConsensus:
    features: np.ndarray
    available: np.ndarray
    donor_weight_entropy: np.ndarray
    valid_donor_count: np.ndarray


def _validate_tokens(tokens: np.ndarray, name: str) -> None:
    if tokens.ndim != 3 or tokens.shape[:2] != (PATCH_GRID, PATCH_GRID):
        raise ValueError(f"{name} tokens must be 27,27,D")
    if not np.issubdtype(tokens.dtype, np.floating) or not np.isfinite(tokens).all():
        raise ValueError(f"{name} tokens must be finite floating point")


def _validate_mask(mask: np.ndarray, name: str) -> None:
    if mask.shape != (PATCH_GRID, PATCH_GRID) or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a 27,27 boolean mask")


def visible_anchor_matches(
    center_tokens: np.ndarray,
    neighbor_tokens: np.ndarray,
    target_mask: np.ndarray,
    *,
    center_valid: np.ndarray | None = None,
    neighbor_valid: np.ndarray | None = None,
    radius: int = 4,
) -> AnchorMatches:
    """Return deterministic mutual-cosine matches from visible center cells only."""

    _validate_tokens(center_tokens, "center")
    _validate_tokens(neighbor_tokens, "neighbor")
    if center_tokens.shape != neighbor_tokens.shape or radius < 0:
        raise ValueError("Anchor token shapes or search radius changed")
    _validate_mask(target_mask, "target_mask")
    if center_valid is None:
        center_valid = np.ones((PATCH_GRID, PATCH_GRID), dtype=bool)
    if neighbor_valid is None:
        neighbor_valid = np.ones((PATCH_GRID, PATCH_GRID), dtype=bool)
    _validate_mask(center_valid, "center_valid")
    _validate_mask(neighbor_valid, "neighbor_valid")

    visible_flat = np.flatnonzero((center_valid & ~target_mask).reshape(-1))
    neighbor_flat = np.flatnonzero(neighbor_valid.reshape(-1))
    if not len(visible_flat) or not len(neighbor_flat):
        empty = np.empty((0, 2), dtype=np.float64)
        return AnchorMatches(empty, empty.copy(), np.empty(0, dtype=np.float64))

    center = center_tokens.reshape(-1, center_tokens.shape[-1])[visible_flat].astype(np.float32)
    neighbor = neighbor_tokens.reshape(-1, neighbor_tokens.shape[-1])[neighbor_flat].astype(
        np.float32
    )
    center /= np.maximum(np.linalg.norm(center, axis=1, keepdims=True), 1e-12)
    neighbor /= np.maximum(np.linalg.norm(neighbor, axis=1, keepdims=True), 1e-12)
    similarity = center @ neighbor.T

    center_yx = np.column_stack(np.unravel_index(visible_flat, (PATCH_GRID, PATCH_GRID)))
    neighbor_yx = np.column_stack(
        np.unravel_index(neighbor_flat, (PATCH_GRID, PATCH_GRID))
    )
    within = (
        np.max(np.abs(center_yx[:, None, :] - neighbor_yx[None, :, :]), axis=2)
        <= radius
    )
    masked = np.where(within, similarity, -np.inf)
    forward = np.argmax(masked, axis=1)
    forward_valid = np.isfinite(masked[np.arange(len(center)), forward])
    reverse = np.argmax(masked, axis=0)
    mutual = forward_valid & (reverse[forward] == np.arange(len(center)))
    selected_center = np.flatnonzero(mutual)
    selected_neighbor = forward[selected_center]
    center_xy = center_yx[selected_center][:, ::-1].astype(np.float64)
    neighbor_xy = neighbor_yx[selected_neighbor][:, ::-1].astype(np.float64)
    cosine = similarity[selected_center, selected_neighbor].astype(np.float64)
    return AnchorMatches(center_xy, neighbor_xy, cosine)


def robust_affine_fit(
    center_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    *,
    iterations: int = 5,
    huber_delta: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Fit center-grid to neighbor-grid affine coordinates with fixed Huber IRLS."""

    if (
        center_xy.ndim != 2
        or center_xy.shape[1:] != (2,)
        or neighbor_xy.shape != center_xy.shape
        or len(center_xy) < 3
        or iterations < 1
        or huber_delta <= 0
        or not np.isfinite(center_xy).all()
        or not np.isfinite(neighbor_xy).all()
    ):
        raise ValueError("Affine correspondences or robust-fit settings are invalid")
    design = np.column_stack([center_xy, np.ones(len(center_xy))])
    median_displacement = np.median(neighbor_xy - center_xy, axis=0)
    matrix = np.array(
        [[1.0, 0.0, median_displacement[0]], [0.0, 1.0, median_displacement[1]]]
    )
    initial_residuals = np.linalg.norm(design @ matrix.T - neighbor_xy, axis=1)
    weights = np.minimum(1.0, huber_delta / np.maximum(initial_residuals, 1e-12))
    for _ in range(iterations):
        root = np.sqrt(weights)[:, None]
        weighted_design = design * root
        solution, _, rank, _ = np.linalg.lstsq(
            weighted_design, neighbor_xy * root, rcond=None
        )
        if rank < 3 or np.linalg.cond(weighted_design) > 10_000:
            raise RuntimeError("Visible anchors do not identify a full affine transform")
        matrix = solution.T
        residuals = np.linalg.norm(design @ solution - neighbor_xy, axis=1)
        weights = np.minimum(1.0, huber_delta / np.maximum(residuals, 1e-12))
    residuals = np.linalg.norm(design @ matrix.T - neighbor_xy, axis=1)
    weighted = float(np.sum(weights * residuals) / np.sum(weights))
    return matrix, weighted


def fit_visible_anchor_transport(
    center_tokens: np.ndarray,
    neighbor_tokens: np.ndarray,
    target_mask: np.ndarray,
    *,
    center_valid: np.ndarray | None = None,
    neighbor_valid: np.ndarray | None = None,
    radius: int = 4,
    minimum_matches: int = 12,
    maximum_residual: float = 2.0,
) -> tuple[AffineTransport, AnchorMatches]:
    matches = visible_anchor_matches(
        center_tokens,
        neighbor_tokens,
        target_mask,
        center_valid=center_valid,
        neighbor_valid=neighbor_valid,
        radius=radius,
    )
    if len(matches.center_xy) < minimum_matches:
        return (
            AffineTransport(np.full((2, 3), np.nan), len(matches.center_xy), np.inf, False, "too_few_matches"),
            matches,
        )
    try:
        matrix, residual = robust_affine_fit(matches.center_xy, matches.neighbor_xy)
    except RuntimeError:
        return (
            AffineTransport(np.full((2, 3), np.nan), len(matches.center_xy), np.inf, False, "rank_deficient"),
            matches,
        )
    valid = residual <= maximum_residual
    return (
        AffineTransport(
            matrix,
            len(matches.center_xy),
            residual,
            valid,
            "ok" if valid else "residual_above_limit",
        ),
        matches,
    )


def bilinear_transport(
    neighbor_tokens: np.ndarray,
    target_mask: np.ndarray,
    transform: AffineTransport,
    *,
    neighbor_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample donor tokens at mapped target centers; invalid targets select dustbin."""

    _validate_tokens(neighbor_tokens, "neighbor")
    _validate_mask(target_mask, "target_mask")
    if neighbor_valid is None:
        neighbor_valid = np.ones((PATCH_GRID, PATCH_GRID), dtype=bool)
    _validate_mask(neighbor_valid, "neighbor_valid")
    target_yx = np.argwhere(target_mask)
    values = np.zeros((len(target_yx), neighbor_tokens.shape[-1]), dtype=np.float32)
    observed = np.zeros(len(target_yx), dtype=bool)
    if not transform.valid:
        return values, observed
    xy1 = np.column_stack([target_yx[:, ::-1], np.ones(len(target_yx))])
    mapped = xy1 @ transform.matrix.T
    for index, (x, y) in enumerate(mapped):
        if not (0 <= x <= PATCH_GRID - 1 and 0 <= y <= PATCH_GRID - 1):
            continue
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, PATCH_GRID - 1), min(y0 + 1, PATCH_GRID - 1)
        support = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
        if not all(neighbor_valid[row, column] for row, column in support):
            continue
        wx, wy = x - x0, y - y0
        values[index] = (
            (1 - wx) * (1 - wy) * neighbor_tokens[y0, x0]
            + wx * (1 - wy) * neighbor_tokens[y0, x1]
            + (1 - wx) * wy * neighbor_tokens[y1, x0]
            + wx * wy * neighbor_tokens[y1, x1]
        )
        observed[index] = True
    return values, observed


def donor_confidence_logits(
    matches: AnchorMatches,
    transform: AffineTransport,
    target_mask: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Compute the preregistered image-derived confidence logit per target cell."""

    _validate_mask(target_mask, "target_mask")
    target_xy = np.argwhere(target_mask)[:, ::-1].astype(np.float64)
    if observed.shape != (len(target_xy),) or observed.dtype != np.bool_:
        raise ValueError("Observed-transport mask differs from the target population")
    logits = np.full(len(target_xy), -np.inf, dtype=np.float64)
    if not transform.valid or not len(matches.center_xy):
        return logits
    for index in np.flatnonzero(observed):
        distances = np.linalg.norm(matches.center_xy - target_xy[index], axis=1)
        nearest = np.argsort(distances, kind="stable")[:4]
        logits[index] = (
            float(np.median(matches.cosine[nearest]))
            - 0.25 * transform.weighted_residual
            - 0.10 * float(distances[nearest[0]])
        )
    return logits


def _componentwise_weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if (
        values.ndim != 2
        or weights.shape != (len(values),)
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Weighted-median values or weights are invalid")
    weights = weights / weights.sum()
    result = np.empty(values.shape[1], dtype=np.float32)
    for feature in range(values.shape[1]):
        order = np.argsort(values[:, feature], kind="stable")
        cumulative = np.cumsum(weights[order])
        position = int(np.searchsorted(cumulative, 0.5, side="left"))
        if (
            position + 1 < len(order)
            and np.isclose(cumulative[position], 0.5, rtol=0, atol=1e-12)
        ):
            lower_offset_position = min(order[position], order[position + 1])
            result[feature] = values[lower_offset_position, feature]
        else:
            result[feature] = values[order[position], feature]
    return result


def aggregate_transported_donors(
    donor_features: np.ndarray,
    donor_observed: np.ndarray,
    donor_logits: np.ndarray,
    center_only: np.ndarray,
    *,
    temperature: float = 0.25,
    entropy_fraction: float = 0.75,
) -> AggregatedTransport:
    """Aggregate up to four donors or copy center-only through the dustbin action."""

    if (
        donor_features.ndim != 3
        or donor_observed.shape != donor_features.shape[:2]
        or donor_logits.shape != donor_features.shape[:2]
        or center_only.shape != donor_features.shape[1:]
        or donor_observed.dtype != np.bool_
        or donor_features.shape[0] not in {1, 2, 3, 4}
        or temperature <= 0
        or not 0 < entropy_fraction < 1
        or not np.isfinite(donor_features).all()
        or not np.isfinite(center_only).all()
    ):
        raise ValueError("Transport aggregation inputs differ from the locked contract")
    donors, targets, _ = donor_features.shape
    features = center_only.astype(np.float32, copy=True)
    used = np.zeros(targets, dtype=bool)
    dustbin = np.ones(targets, dtype=np.float64)
    entropy = np.zeros(targets, dtype=np.float64)
    counts = donor_observed.sum(axis=0).astype(np.int64)
    for target in range(targets):
        valid = np.flatnonzero(donor_observed[:, target])
        if not len(valid):
            continue
        logits = np.concatenate([donor_logits[valid, target], np.array([0.0])])
        if not np.isfinite(logits[:-1]).all():
            raise ValueError("An observed donor lacks a finite confidence logit")
        scaled = logits / temperature
        probabilities = np.exp(scaled - np.max(scaled))
        probabilities /= probabilities.sum()
        dustbin[target] = probabilities[-1]
        entropy[target] = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
        )
        entropy_limit = entropy_fraction * np.log(len(probabilities))
        if dustbin[target] >= 0.5 or entropy[target] > entropy_limit:
            continue
        donor_weights = probabilities[:-1]
        features[target] = _componentwise_weighted_median(
            donor_features[valid, target], donor_weights
        )
        used[target] = True
    if donors != donor_observed.shape[0]:
        raise RuntimeError("Internal donor count changed")
    return AggregatedTransport(features, used, dustbin, entropy, counts)


def transport_consensus(
    donor_features: np.ndarray,
    donor_observed: np.ndarray,
    donor_logits: np.ndarray,
    center_only: np.ndarray,
    *,
    temperature: float = 0.25,
) -> TransportConsensus:
    """Build donor consensus without misusing donor-identity entropy as rejection."""

    if (
        donor_features.ndim != 3
        or donor_observed.shape != donor_features.shape[:2]
        or donor_logits.shape != donor_features.shape[:2]
        or center_only.shape != donor_features.shape[1:]
        or donor_observed.dtype != np.bool_
        or donor_features.shape[0] not in {1, 2, 3, 4}
        or temperature <= 0
        or not np.isfinite(donor_features).all()
        or not np.isfinite(center_only).all()
    ):
        raise ValueError("Consensus inputs differ from the locked contract")
    _, targets, _ = donor_features.shape
    features = center_only.astype(np.float32, copy=True)
    available = np.zeros(targets, dtype=bool)
    entropy = np.zeros(targets, dtype=np.float64)
    counts = donor_observed.sum(axis=0).astype(np.int64)
    for target in range(targets):
        valid = np.flatnonzero(donor_observed[:, target])
        if not len(valid):
            continue
        logits = donor_logits[valid, target]
        if not np.isfinite(logits).all():
            raise ValueError("An observed donor lacks a finite confidence logit")
        scaled = logits / temperature
        probabilities = np.exp(scaled - np.max(scaled))
        probabilities /= probabilities.sum()
        entropy[target] = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
        )
        features[target] = _componentwise_weighted_median(
            donor_features[valid, target], probabilities
        )
        available[target] = True
    return TransportConsensus(features, available, entropy, counts)
