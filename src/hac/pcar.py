"""Small, fail-closed primitives for the PCAR-v5 reconstruction policy.

The module intentionally contains no activity labels, ARFTR outputs, support
categories, or task-routing logic.  It only implements the five bounded
full-768 reconstruction actions and the centre-balanced policy objective used
by the prospective PCAR-v5 runner.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ACTION_ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
FEATURE_DIM = 768
FEATURE_COUNT = 32
ACTION_COUNT = len(ACTION_ALPHAS)


class PCARPolicy(nn.Module):
    """The locked 32-to-5 hard competitive gate."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(FEATURE_COUNT),
            nn.Linear(FEATURE_COUNT, 32),
            nn.GELU(),
            nn.Linear(32, ACTION_COUNT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != FEATURE_COUNT or not features.is_floating_point():
            raise ValueError("PCAR policy features must end in 32 floating values")
        if not torch.isfinite(features).all():
            raise ValueError("PCAR policy features must be finite")
        return self.network(features)


def exact_convex_actions(
    p2: torch.Tensor,
    affine: torch.Tensor,
    *,
    available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize ``[...,5,768]`` actions with arithmetic-free endpoints.

    ``alpha=0`` and ``alpha=1`` are cloned directly, so the retain and direct
    affine actions are byte-identical to their inputs.  Unavailable transport
    is represented by five copies of exact P2; the caller's hard selector will
    therefore retain P2 without ever consulting a fabricated candidate.
    """

    if p2.shape != affine.shape or p2.shape[-1] != FEATURE_DIM:
        raise ValueError("P2 and affine tensors must share [...,768] geometry")
    if not p2.is_floating_point() or not affine.is_floating_point():
        raise ValueError("P2 and affine tensors must be floating point")
    if not torch.isfinite(p2).all() or not torch.isfinite(affine).all():
        raise ValueError("P2 and affine tensors must be finite")
    if available is not None and available.shape != p2.shape[:-1]:
        raise ValueError("transport availability must match the token prefix")
    actions: list[torch.Tensor] = [p2.clone()]
    direction = affine - p2
    for alpha in ACTION_ALPHAS[1:-1]:
        actions.append(p2 + float(alpha) * direction)
    actions.append(affine.clone())
    result = torch.stack(actions, dim=-2)
    if available is not None:
        result = torch.where(available[..., None, None], result, p2[..., None, :])
    return result


def exact_convex_actions_numpy(
    p2: np.ndarray,
    affine: np.ndarray,
    *,
    available: np.ndarray | None = None,
) -> np.ndarray:
    """NumPy counterpart used by the label-blind cache/forensic runner."""

    p2_array = np.asarray(p2)
    affine_array = np.asarray(affine)
    if p2_array.shape != affine_array.shape or p2_array.shape[-1] != FEATURE_DIM:
        raise ValueError("P2 and affine arrays must share [...,768] geometry")
    if not np.issubdtype(p2_array.dtype, np.floating) or not np.issubdtype(
        affine_array.dtype, np.floating
    ):
        raise ValueError("P2 and affine arrays must be floating point")
    if not np.isfinite(p2_array).all() or not np.isfinite(affine_array).all():
        raise ValueError("P2 and affine arrays must be finite")
    if available is not None and np.asarray(available).shape != p2_array.shape[:-1]:
        raise ValueError("transport availability must match the token prefix")
    actions = [p2_array.copy()]
    direction = affine_array - p2_array
    actions.extend(p2_array + alpha * direction for alpha in ACTION_ALPHAS[1:-1])
    actions.append(affine_array.copy())
    result = np.stack(actions, axis=-2)
    if available is not None:
        result = np.where(
            np.asarray(available)[..., None, None], result, p2_array[..., None, :]
        )
    return result


def center_balanced_expected_cost(
    logits: torch.Tensor,
    action_costs: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Expected raw-cosine cost with equal total weight per centre."""

    if logits.shape[:-1] != action_costs.shape[:-1] or logits.shape[-1] != ACTION_COUNT:
        raise ValueError("policy logits and five action costs differ")
    if valid.shape != logits.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("policy validity mask differs")
    if not logits.is_floating_point() or not action_costs.is_floating_point():
        raise ValueError("policy objective requires floating logits and costs")
    if not torch.isfinite(logits).all() or not torch.isfinite(action_costs).all():
        raise ValueError("policy objective inputs must be finite")
    if not valid.any(dim=-1).all():
        raise ValueError("each centre must contain a valid target")
    probabilities = torch.softmax(logits, dim=-1)
    expected = (probabilities * action_costs).sum(dim=-1)
    weights = valid.to(expected.dtype)
    per_center = (expected * weights).sum(dim=-1) / weights.sum(dim=-1)
    return per_center.mean()


def hard_actions(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose actions; ties are resolved by argmax's first action (alpha=0)."""

    if logits.shape[-1] != ACTION_COUNT or valid.shape != logits.shape[:-1]:
        raise ValueError("policy logits and validity geometry differ")
    if valid.dtype != torch.bool:
        raise ValueError("validity must be boolean")
    if available is not None and available.shape != valid.shape:
        raise ValueError("availability geometry differs")
    actions = torch.argmax(logits, dim=-1)
    actions = torch.where(valid, actions, torch.zeros_like(actions))
    if available is not None:
        actions = torch.where(available, actions, torch.zeros_like(actions))
    return actions


def action_costs(
    candidates: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Raw full-dimensional cosine error for every action."""

    if candidates.shape[:-1] != teacher.shape[:-1] + (ACTION_COUNT,):
        # This branch gives a useful error without relying on broadcasting
        # surprises when callers accidentally swap action and feature axes.
        if candidates.shape[:-2] != teacher.shape[:-1] or candidates.shape[-2:] != (
            ACTION_COUNT,
            FEATURE_DIM,
        ):
            raise ValueError("candidate/teacher action geometry differs")
    if candidates.shape[-2:] != (ACTION_COUNT, FEATURE_DIM):
        raise ValueError("candidates must end in 5,768")
    if teacher.shape != candidates.shape[:-2] + (FEATURE_DIM,):
        raise ValueError("teacher geometry differs from candidates")
    if not torch.isfinite(candidates).all() or not torch.isfinite(teacher).all():
        raise ValueError("candidate and teacher tokens must be finite")
    normalized_candidate = F.normalize(candidates.float(), dim=-1, eps=1e-12)
    normalized_teacher = F.normalize(teacher.float(), dim=-1, eps=1e-12)
    return 1.0 - (normalized_candidate * normalized_teacher[..., None, :]).sum(dim=-1)


def finite_feature_matrix(rows: Iterable[np.ndarray]) -> np.ndarray:
    """Stack feature rows while preserving an explicit finite-value contract."""

    array = np.asarray(list(rows), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != FEATURE_COUNT or not np.isfinite(array).all():
        raise ValueError("PCAR feature rows must form a finite N,32 matrix")
    return array

