"""The isolated, train-from-initialization motion-null hypothesis.

Historical modules are intentionally unchanged. Ordered means and geometry still
contain temporal information; this is an explicit-carrier-null invariant only.
"""
from __future__ import annotations

import numpy as np
import torch

from hac.native_motion_innovation import NativeMotionInnovation, bounded_motion_candidate


class MotionNullContrast(NativeMotionInnovation):
    """Shared reader difference, bounded after subtracting the raw scalar heads."""

    def __init__(self) -> None:
        super().__init__()
        # This scalar cancels algebraically. Do not count it as an active parameter.
        self.head[-1].bias.requires_grad_(False)

    def raw_score(self, phases: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        if phases.ndim != 5 or phases.shape[1:3] != (2, 6):
            raise ValueError("Expected N x 2 x 6 x H x W phases")
        if geometry.shape != (len(phases), 12):
            raise ValueError("Expected twelve geometry fields")
        encoded = self.encoder(phases.flatten(0, 1)).flatten(1).reshape(len(phases), -1)
        return self.head(torch.cat((encoded, geometry), 1)).squeeze(1)

    def forward(self, phases: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        reference = phases.clone()
        reference[:, :, 3:] = 0
        return 0.5 * torch.tanh(self.raw_score(phases, geometry) - self.raw_score(reference, geometry))


def exact_retaining_candidate(anchor, delta, available):
    """Inference-only byte-exact retain; training must use the historical loss.

    A training-time where(delta==0, anchor, candidate) would kill gradients at the
    zero-initialized last layer. This wrapper is deliberately NumPy-only.
    """
    if isinstance(anchor, torch.Tensor) or isinstance(delta, torch.Tensor):
        raise TypeError("Inference-only mapping; use motion_loss for differentiable training")
    p = np.asarray(anchor, np.float64)
    d = np.asarray(delta, np.float64)
    if not np.isfinite(d).all() or np.any(np.abs(d) > 0.5):
        raise ValueError("Nonfinite or out-of-bound correction")
    candidate, eligible = bounded_motion_candidate(p, d, available)
    retain = (~eligible) | (d == 0)
    candidate[retain] = p[retain]
    return candidate, eligible
