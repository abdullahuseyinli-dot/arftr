"""Frame-supervised corrections that preserve a probability anchor."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def probability_factor_scores(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sitting-vs-upright and walking-vs-standing log odds."""

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Expected [batch, 3] probabilities")
    values = probabilities.clamp_min(1e-12)
    posture = torch.log(values[:, 0]) - torch.logsumexp(torch.log(values[:, 1:]), dim=1)
    motion = torch.log(values[:, 2]) - torch.log(values[:, 1])
    return posture, motion


def factor_logits(posture: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """Map the two factor scores to equivalent three-class logits."""

    if posture.ndim != 1 or motion.shape != posture.shape:
        raise ValueError("Factor scores must be aligned vectors")
    upright_normalizer = torch.logaddexp(torch.zeros_like(motion), motion)
    return torch.stack((posture + upright_normalizer, torch.zeros_like(motion), motion), dim=1)


class BoundedFactorCorrection(nn.Module):
    """Shared frame encoder with auxiliary and anchor-correction outputs."""

    def __init__(
        self,
        *,
        input_dim: int = 768,
        width: int = 128,
        dropout: float = 0.1,
        posture_bound: float = 0.5,
        motion_bound: float = 0.5,
        shared_output: bool = False,
        bounded: bool = True,
        parameter_limit: int = 111_000,
    ) -> None:
        super().__init__()
        if min(input_dim, width) < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid correction dimensions")
        if posture_bound <= 0 or motion_bound <= 0:
            raise ValueError("Correction limits must be positive")
        self.input_dim = input_dim
        self.shared_output = bool(shared_output)
        self.bounded = bool(bounded)
        self.register_buffer("limits", torch.tensor((posture_bound, motion_bound)))
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.auxiliary_classifier = nn.Linear(width, 3)
        self.correction_head = None if shared_output else nn.Linear(width, 2)
        nn.init.zeros_(self.auxiliary_classifier.weight)
        nn.init.zeros_(self.auxiliary_classifier.bias)
        if self.correction_head is not None:
            nn.init.zeros_(self.correction_head.weight)
            nn.init.zeros_(self.correction_head.bias)
        if self.trainable_parameters > parameter_limit:
            raise ValueError("Correction model exceeds the parameter limit")

    @property
    def trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def encoded(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Frame features must have shape [batch, input_dim]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Frame features must be finite floating values")
        return self.encoder(features)

    def heads(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoded(features)
        auxiliary = self.auxiliary_classifier(encoded)
        if self.correction_head is None:
            # At zero logits, sitting-vs-upright is -log(2).  Centering it
            # makes the initialized shared-output correction exactly zero.
            raw = torch.stack(
                (
                    auxiliary[:, 0] - torch.logsumexp(auxiliary[:, 1:], dim=1) + np.log(2.0),
                    auxiliary[:, 2] - auxiliary[:, 1],
                ),
                dim=1,
            )
        else:
            raw = self.correction_head(encoded)
        correction = self.limits * (torch.tanh(raw) if self.bounded else raw)
        return auxiliary, correction

    def forward(
        self, features: torch.Tensor, anchor_probabilities: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if anchor_probabilities.shape != (len(features), 3):
            raise ValueError("Anchor probabilities must align with features")
        if (
            not anchor_probabilities.is_floating_point()
            or not torch.isfinite(anchor_probabilities).all()
            or torch.any(anchor_probabilities < 0)
            or not torch.allclose(
                anchor_probabilities.sum(1),
                torch.ones(len(features), device=anchor_probabilities.device),
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("Anchor probabilities must be a finite simplex")
        auxiliary, correction = self.heads(features)
        posture, motion = probability_factor_scores(anchor_probabilities)
        logits = factor_logits(posture + correction[:, 0], motion + correction[:, 1])
        return {"logits": logits, "auxiliary_logits": auxiliary, "correction": correction}


def correction_loss(
    model: BoundedFactorCorrection,
    *,
    center_features: torch.Tensor,
    anchor_probabilities: torch.Tensor,
    center_labels: torch.Tensor,
    auxiliary_features: torch.Tensor,
    auxiliary_labels: torch.Tensor,
    auxiliary_weights: torch.Tensor,
    class_weight: torch.Tensor,
    center_loss_weight: float,
    auxiliary_loss_weight: float,
    anchor_kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(center_features, anchor_probabilities)
    center = F.cross_entropy(output["logits"], center_labels, weight=class_weight)
    auxiliary_logits, _ = model.heads(auxiliary_features)
    per_auxiliary = F.cross_entropy(
        auxiliary_logits, auxiliary_labels, weight=class_weight, reduction="none"
    )
    effective = auxiliary_weights * class_weight[auxiliary_labels]
    auxiliary = (per_auxiliary * auxiliary_weights).sum() / effective.sum().clamp_min(1e-12)
    anchor_kl = F.kl_div(
        F.log_softmax(output["logits"], dim=1), anchor_probabilities, reduction="batchmean"
    )
    total = (
        center_loss_weight * center
        + auxiliary_loss_weight * auxiliary
        + anchor_kl_weight * anchor_kl
    )
    return total, {"center": center, "auxiliary": auxiliary, "anchor_kl": anchor_kl}


def apply_numpy_correction(anchor: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Apply already bounded factor-score corrections, preserving exact zeros."""

    anchor = np.asarray(anchor, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if anchor.ndim != 2 or anchor.shape[1] != 3 or correction.shape != (len(anchor), 2):
        raise ValueError("Correction arrays are misaligned")
    if not np.isfinite(anchor).all() or not np.isfinite(correction).all():
        raise ValueError("Correction arrays must be finite")
    output = anchor.copy()
    active = np.any(correction != 0, axis=1)
    if active.any():
        values = np.clip(anchor[active], 1e-12, 1.0)
        posture = np.log(values[:, 0] / (values[:, 1] + values[:, 2])) + correction[active, 0]
        motion = np.log(values[:, 2] / values[:, 1]) + correction[active, 1]
        upright_norm = np.logaddexp(0.0, motion)
        logits = np.column_stack((posture + upright_norm, np.zeros(len(posture)), motion))
        logits -= logits.max(1, keepdims=True)
        probabilities = np.exp(logits)
        output[active] = probabilities / probabilities.sum(1, keepdims=True)
    return output
