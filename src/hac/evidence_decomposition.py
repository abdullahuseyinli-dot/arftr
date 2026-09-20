"""Observation-only aligned evidence for the HAC action-utility bridge.

This module contains no labels, folds, incumbent predictions, support categories,
or routing decisions.  It converts one observed center and four observed neighbours
into matched four-arm summaries.  Task fitting belongs in the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hac.center_evidence_completion import (
    AffineTransport,
    fit_visible_anchor_transport,
)

PATCH_GRID = 27
FEATURE_DIM = 768
CONTEXT_DIM = 2304
ROI_BINS = 6
ARM_NAMES = ("B0_center_repeat", "B1_unaligned", "B2_phase_control", "B3_aligned_primary")
READER_INPUT_DIM = CONTEXT_DIM + 2 * ROI_BINS * FEATURE_DIM + ROI_BINS


@dataclass(frozen=True)
class EvidenceDecomposition:
    """Pooled observations and audit measurements for one center."""

    center_bins: np.ndarray  # [6, D]
    contrasts: np.ndarray  # [4, 6, D], B0 is exactly zero
    coverage: np.ndarray  # [6], shared by all arms
    roi_bin_map: np.ndarray  # [27, 27], -1 outside the valid actor ROI
    common_target_count: np.ndarray  # [6]
    roi_target_count: np.ndarray  # [6]
    donor_common_support: np.ndarray  # [4, 27, 27]
    transform_matrix: np.ndarray  # [4, 2, 3], NaN for invalid fits
    transform_valid: np.ndarray  # [4]
    match_count: np.ndarray  # [4]
    weighted_residual: np.ndarray  # [4]


def _validate_grid(tokens: np.ndarray, valid: np.ndarray, name: str) -> None:
    if tokens.ndim != 3 or tokens.shape[:2] != (PATCH_GRID, PATCH_GRID):
        raise ValueError(f"{name} tokens must have shape 27,27,D")
    if valid.shape != (PATCH_GRID, PATCH_GRID) or valid.dtype != np.bool_:
        raise ValueError(f"{name} validity must be a 27,27 boolean array")
    if not np.issubdtype(tokens.dtype, np.floating) or not np.isfinite(tokens).all():
        raise ValueError(f"{name} tokens must be finite floating point")


def actor_roi_bin_map(
    image_to_crop: np.ndarray,
    actor_box_source: tuple[float, float, float, float],
    source_valid: np.ndarray,
) -> np.ndarray:
    """Assign DINO patch centers to a 3x2 actor-relative grid.

    ``image_to_crop`` maps source pixel-edge coordinates to the 384px letterbox.
    DINO consumes only the leading 378px (27 patches of size 14), so patch centers
    are exactly ``14 * (index + .5)`` in that coordinate system.
    """

    transform = np.asarray(image_to_crop, dtype=np.float64)
    box = np.asarray(actor_box_source, dtype=np.float64)
    if (
        transform.shape != (3, 3)
        or not np.isfinite(transform).all()
        or abs(np.linalg.det(transform)) <= 1e-12
        or box.shape != (4,)
        or not np.isfinite(box).all()
        or np.any(box[2:] <= box[:2])
        or source_valid.shape != (PATCH_GRID, PATCH_GRID)
        or source_valid.dtype != np.bool_
    ):
        raise ValueError("Malformed actor-ROI geometry")
    y, x = np.mgrid[:PATCH_GRID, :PATCH_GRID]
    crop_xy1 = np.stack((14.0 * (x + 0.5), 14.0 * (y + 0.5), np.ones_like(x)), axis=-1)
    source_h = crop_xy1 @ np.linalg.inv(transform).T
    safe = np.isfinite(source_h).all(2) & (np.abs(source_h[..., 2]) > 1e-12)
    source_xy = np.zeros((*source_h.shape[:2], 2), dtype=np.float64)
    source_xy[safe] = source_h[safe, :2] / source_h[safe, 2, None]
    normalized = (source_xy - box[:2]) / (box[2:] - box[:2])
    inside = (
        safe
        & source_valid
        & (normalized[..., 0] >= 0)
        & (normalized[..., 0] <= 1)
        & (normalized[..., 1] >= 0)
        & (normalized[..., 1] <= 1)
    )
    result = np.full((PATCH_GRID, PATCH_GRID), -1, dtype=np.int8)
    horizontal = np.minimum(np.floor(2 * normalized[..., 0]).astype(np.int64), 1)
    vertical = np.minimum(np.floor(3 * normalized[..., 1]).astype(np.int64), 2)
    result[inside] = (2 * vertical + horizontal)[inside].astype(np.int8)
    return result


def _grid_coordinates() -> np.ndarray:
    y, x = np.mgrid[:PATCH_GRID, :PATCH_GRID]
    return np.stack((x, y), axis=-1).astype(np.float64)


def bilinear_sample_grid(
    tokens: np.ndarray, valid: np.ndarray, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a complete donor grid, requiring valid four-corner support."""

    _validate_grid(tokens, valid, "donor")
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.shape != (PATCH_GRID, PATCH_GRID, 2):
        raise ValueError("Sampling coordinates must have shape 27,27,2")
    finite = np.isfinite(xy).all(2)
    inside = (
        finite & (xy[..., 0] >= 0) & (xy[..., 0] <= 26) & (xy[..., 1] >= 0) & (xy[..., 1] <= 26)
    )
    safe = np.where(finite[..., None], xy, 0.0)
    x0 = np.floor(np.clip(safe[..., 0], 0, 26)).astype(np.int64)
    y0 = np.floor(np.clip(safe[..., 1], 0, 26)).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, 26), np.minimum(y0 + 1, 26)
    observed = inside & valid[y0, x0] & valid[y0, x1] & valid[y1, x0] & valid[y1, x1]
    wx = (safe[..., 0] - x0)[..., None].astype(np.float32)
    wy = (safe[..., 1] - y0)[..., None].astype(np.float32)
    source = tokens.astype(np.float32, copy=False)
    values = (
        (1 - wx) * (1 - wy) * source[y0, x0]
        + wx * (1 - wy) * source[y0, x1]
        + (1 - wx) * wy * source[y1, x0]
        + wx * wy * source[y1, x1]
    )
    values[~observed] = 0
    return values.astype(np.float32, copy=False), observed


def _mapped_coordinates(transform: AffineTransport) -> np.ndarray:
    grid = _grid_coordinates()
    xy1 = np.concatenate((grid, np.ones((*grid.shape[:2], 1))), axis=2)
    if not transform.valid:
        return np.full_like(grid, np.nan)
    return xy1 @ np.asarray(transform.matrix, dtype=np.float64).T


def phase_only_coordinates(mapped: np.ndarray) -> np.ndarray:
    """Keep affine fractional phase while removing integer displacement."""

    mapped = np.asarray(mapped, dtype=np.float64)
    if mapped.shape != (PATCH_GRID, PATCH_GRID, 2):
        raise ValueError("Mapped coordinates must have shape 27,27,2")
    return _grid_coordinates() + mapped - np.floor(mapped + 0.5)


def decompose_evidence(
    center_tokens: np.ndarray,
    neighbor_tokens: np.ndarray,
    center_valid: np.ndarray,
    neighbor_valid: np.ndarray,
    roi_bin_map: np.ndarray,
    *,
    radius: int = 4,
    minimum_matches: int = 12,
    maximum_residual: float = 2.0,
) -> EvidenceDecomposition:
    """Compute center, unaligned, phase-only and aligned matched summaries."""

    _validate_grid(center_tokens, center_valid, "center")
    if (
        neighbor_tokens.ndim != 4
        or neighbor_tokens.shape[:3] != (4, PATCH_GRID, PATCH_GRID)
        or neighbor_tokens.shape[-1] != center_tokens.shape[-1]
        or neighbor_valid.shape != (4, PATCH_GRID, PATCH_GRID)
        or neighbor_valid.dtype != np.bool_
        or not np.isfinite(neighbor_tokens).all()
        or roi_bin_map.shape != (PATCH_GRID, PATCH_GRID)
        or not np.issubdtype(roi_bin_map.dtype, np.integer)
        or ((roi_bin_map < -1) | (roi_bin_map >= ROI_BINS)).any()
    ):
        raise ValueError("Malformed four-donor evidence inputs")
    roi = (roi_bin_map >= 0) & center_valid
    center = center_tokens.astype(np.float32, copy=False)
    center_bins = np.zeros((ROI_BINS, center.shape[-1]), dtype=np.float32)
    roi_counts = np.zeros(ROI_BINS, dtype=np.int64)
    for bin_index in range(ROI_BINS):
        selected = roi & (roi_bin_map == bin_index)
        roi_counts[bin_index] = int(selected.sum())
        if selected.any():
            center_bins[bin_index] = center[selected].mean(0, dtype=np.float64).astype(np.float32)

    arm_samples = np.zeros((3, 4, PATCH_GRID, PATCH_GRID, center.shape[-1]), dtype=np.float32)
    shared = np.zeros((4, PATCH_GRID, PATCH_GRID), dtype=bool)
    matrices = np.full((4, 2, 3), np.nan, dtype=np.float64)
    transform_valid = np.zeros(4, dtype=bool)
    match_count = np.zeros(4, dtype=np.int64)
    residual = np.full(4, np.inf, dtype=np.float64)
    no_exclusion = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)
    grid = _grid_coordinates()
    for donor in range(4):
        tokens = neighbor_tokens[donor]
        valid = neighbor_valid[donor]
        transform, matches = fit_visible_anchor_transport(
            center_tokens,
            tokens,
            no_exclusion,
            center_valid=center_valid,
            neighbor_valid=valid,
            radius=radius,
            minimum_matches=minimum_matches,
            maximum_residual=maximum_residual,
        )
        match_count[donor] = len(matches.center_xy)
        residual[donor] = transform.weighted_residual
        transform_valid[donor] = transform.valid
        if transform.valid:
            matrices[donor] = transform.matrix
        unaligned, unaligned_valid = bilinear_sample_grid(tokens, valid, grid)
        mapped = _mapped_coordinates(transform)
        phase, phase_valid = bilinear_sample_grid(tokens, valid, phase_only_coordinates(mapped))
        aligned, aligned_valid = bilinear_sample_grid(tokens, valid, mapped)
        arm_samples[:, donor] = (unaligned, phase, aligned)
        shared[donor] = roi & unaligned_valid & phase_valid & aligned_valid

    donor_count = shared.sum(0)
    available = donor_count > 0
    evidence = np.zeros((3, PATCH_GRID, PATCH_GRID, center.shape[-1]), dtype=np.float32)
    for arm in range(3):
        numerator = (arm_samples[arm] * shared[..., None]).sum(0, dtype=np.float64)
        evidence[arm, available] = (numerator[available] / donor_count[available, None]).astype(
            np.float32
        )
    contrasts = np.zeros((len(ARM_NAMES), ROI_BINS, center.shape[-1]), dtype=np.float32)
    common_counts = np.zeros(ROI_BINS, dtype=np.int64)
    for bin_index in range(ROI_BINS):
        selected = available & (roi_bin_map == bin_index)
        common_counts[bin_index] = int(selected.sum())
        if selected.any():
            for arm in range(3):
                contrasts[arm + 1, bin_index] = (
                    (evidence[arm, selected] - center[selected])
                    .mean(0, dtype=np.float64)
                    .astype(np.float32)
                )
    coverage = np.divide(
        common_counts,
        roi_counts,
        out=np.zeros(ROI_BINS, dtype=np.float64),
        where=roi_counts > 0,
    ).astype(np.float32)
    return EvidenceDecomposition(
        center_bins=center_bins,
        contrasts=contrasts,
        coverage=coverage,
        roi_bin_map=roi_bin_map.astype(np.int8, copy=True),
        common_target_count=common_counts,
        roi_target_count=roi_counts,
        donor_common_support=shared,
        transform_matrix=matrices,
        transform_valid=transform_valid,
        match_count=match_count,
        weighted_residual=residual,
    )


def compose_arm_features(
    context: np.ndarray,
    center_bins: np.ndarray,
    contrasts: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """Create the exact 11,526-dimensional reader inputs for all four arms."""

    context = np.asarray(context, dtype=np.float32)
    center_bins = np.asarray(center_bins, dtype=np.float32)
    contrasts = np.asarray(contrasts, dtype=np.float32)
    coverage = np.asarray(coverage, dtype=np.float32)
    rows = len(context)
    if (
        context.shape != (rows, CONTEXT_DIM)
        or center_bins.shape != (rows, ROI_BINS, FEATURE_DIM)
        or contrasts.shape != (rows, len(ARM_NAMES), ROI_BINS, FEATURE_DIM)
        or coverage.shape != (rows, ROI_BINS)
        or not all(np.isfinite(x).all() for x in (context, center_bins, contrasts, coverage))
        or np.any(contrasts[:, 0] != 0)
        or np.any((coverage < 0) | (coverage > 1))
    ):
        raise ValueError("Malformed pooled evidence arrays")
    shared = np.concatenate((context, center_bins.reshape(rows, -1)), axis=1)
    result = np.empty((rows, len(ARM_NAMES), READER_INPUT_DIM), dtype=np.float32)
    for arm in range(len(ARM_NAMES)):
        result[:, arm] = np.concatenate(
            (shared, contrasts[:, arm].reshape(rows, -1), coverage), axis=1
        )
    return result


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the locked float64 population scaler on training observations only."""

    values = np.asarray(features)
    if values.ndim != 2 or not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("Scaler needs a nonempty finite matrix")
    values = values.astype(np.float64)
    mean = values.mean(0)
    scale = np.maximum(values.std(0, ddof=0), 1e-6)
    return mean, scale


def apply_standardizer(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(features)
    mean, scale = np.asarray(mean), np.asarray(scale)
    if (
        values.ndim != 2
        or mean.shape != (values.shape[1],)
        or scale.shape != mean.shape
        or not np.isfinite(values).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale < 1e-6)
    ):
        raise ValueError("Malformed standardizer input")
    return ((values.astype(np.float64) - mean) / scale).astype(np.float32)


class EvidenceUtilityReader(nn.Module):
    """The locked small factor reader; it is not an ARFTR intervention router."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(READER_INPUT_DIM, 64), nn.GELU(), nn.Linear(64, 2))

    @property
    def trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != READER_INPUT_DIM:
            raise ValueError("Evidence reader expects B,11526 features")
        return self.layers(features)


def decode_factor_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("Factor logits must be finite B,2 values")
    sitting = torch.sigmoid(logits[:, 0])
    walking_share = torch.sigmoid(logits[:, 1])
    upright = 1 - sitting
    return torch.stack((sitting, upright * (1 - walking_share), upright * walking_share), 1)


def weighted_factor_nll(
    logits: torch.Tensor, labels: torch.Tensor, class_weight: torch.Tensor
) -> torch.Tensor:
    """Return mean(weight[row] * NLL[row]) for uniform-center minibatches."""

    if (
        logits.ndim != 2
        or logits.shape[1] != 2
        or labels.shape != (len(logits),)
        or class_weight.shape != (3,)
        or labels.dtype != torch.long
        or not torch.isfinite(logits).all()
        or not torch.isfinite(class_weight).all()
        or (class_weight <= 0).any()
        or ((labels < 0) | (labels > 2)).any()
    ):
        raise ValueError("Malformed factor-loss inputs")
    a, b = logits[:, 0], logits[:, 1]
    losses = torch.stack(
        (F.softplus(-a), F.softplus(a) + F.softplus(b), F.softplus(a) + F.softplus(-b)), 1
    )
    per_row = losses.gather(1, labels[:, None]).squeeze(1)
    return (class_weight[labels] * per_row).mean()
