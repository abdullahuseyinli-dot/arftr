"""Frame-supervised local evidence with an exact probability anchor."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_frame_training(seed: int) -> None:
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


class FrameSupervisedAnchorResidual(nn.Module):
    """A small shared frame head whose logits form a bounded anchor residual."""

    def __init__(
        self,
        *,
        input_dim: int = 768,
        width: int = 128,
        classes: int = 3,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        parameter_limit: int = 110_000,
    ) -> None:
        super().__init__()
        if min(input_dim, width, classes) < 1 or classes != 3:
            raise ValueError("Invalid frame-residual dimensions")
        if not 0 <= dropout < 1 or not 0 < residual_scale <= 1:
            raise ValueError("Invalid dropout or residual scale")
        self.input_dim = input_dim
        self.classes = classes
        self.residual_scale = residual_scale
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_classifier = nn.Linear(width, classes)
        # The untrained architecture is an exact no-op on its supplied anchor.
        nn.init.zeros_(self.local_classifier.weight)
        nn.init.zeros_(self.local_classifier.bias)
        if self.trainable_parameters > parameter_limit:
            raise ValueError("Frame residual exceeds the declared parameter limit")

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def local_logits(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != self.input_dim
            or not features.is_floating_point()
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Frame features must be finite floating [batch, input_dim]")
        return self.local_classifier(self.encoder(features))

    def forward(
        self, features: torch.Tensor, anchor_probabilities: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_logits = self.local_logits(features)
        if (
            anchor_probabilities.shape != (len(features), self.classes)
            or not anchor_probabilities.is_floating_point()
            or not torch.isfinite(anchor_probabilities).all()
            or torch.any(anchor_probabilities < 0)
            or not torch.allclose(
                anchor_probabilities.sum(1),
                torch.ones(len(features), device=features.device),
                atol=1e-6,
            )
        ):
            raise ValueError("Anchor probabilities must be a finite simplex")
        anchor_logits = torch.log(anchor_probabilities.clamp_min(1e-12))
        logits = anchor_logits + self.residual_scale * local_logits
        return {"logits": logits, "local_logits": local_logits}


def class_weights(labels: np.ndarray, rows: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    counts = np.bincount(labels[rows], minlength=3).astype(np.float64)
    if len(rows) == 0 or np.any(counts == 0):
        raise ValueError("Frame residual training population needs all classes")
    return (len(rows) / (3 * counts)).astype(np.float32)


def anchored_probabilities(
    anchor_probabilities: np.ndarray, local_logits: np.ndarray, *, residual_scale: float
) -> np.ndarray:
    """Apply a local logit residual and exactly copy rows with a zero residual."""

    anchor = np.asarray(anchor_probabilities, dtype=np.float64)
    residual = np.asarray(local_logits, dtype=np.float64)
    if (
        anchor.ndim != 2
        or anchor.shape[1] != 3
        or residual.shape != anchor.shape
        or not np.isfinite(anchor).all()
        or not np.isfinite(residual).all()
        or (anchor < 0).any()
        or not np.allclose(anchor.sum(1), 1.0, atol=1e-6)
        or not 0 < residual_scale <= 1
    ):
        raise ValueError("Invalid anchored residual arrays")
    output = anchor.copy()
    # A common additive constant is also a probability no-op.
    centered = residual - residual.mean(1, keepdims=True)
    active = np.any(centered != 0, axis=1)
    if active.any():
        logits = np.log(np.clip(anchor[active], 1e-12, 1.0)) + residual_scale * centered[active]
        logits -= logits.max(1, keepdims=True)
        values = np.exp(logits)
        output[active] = values / values.sum(1, keepdims=True)
    return output


def frame_residual_loss(
    model: FrameSupervisedAnchorResidual,
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
    if (
        center_labels.shape != (len(center_features),)
        or auxiliary_labels.shape != (len(auxiliary_features),)
        or auxiliary_weights.shape != (len(auxiliary_features),)
        or class_weight.shape != (3,)
        or not torch.isfinite(auxiliary_weights).all()
        or not torch.isfinite(class_weight).all()
        or torch.any(auxiliary_weights < 0)
        or torch.any(class_weight <= 0)
        or min(center_loss_weight, auxiliary_loss_weight, anchor_kl_weight) < 0
    ):
        raise ValueError("Frame-residual loss inputs do not align")
    output = model(center_features, anchor_probabilities)
    center = F.cross_entropy(output["logits"], center_labels, weight=class_weight)
    auxiliary_logits = model.local_logits(auxiliary_features)
    per_auxiliary = F.cross_entropy(
        auxiliary_logits, auxiliary_labels, weight=class_weight, reduction="none"
    )
    # Match PyTorch's weighted-mean CE semantics while layering physical-frame
    # multiplicity/cap weights on top of the locked class weights.
    effective_auxiliary_weight = auxiliary_weights * class_weight[auxiliary_labels]
    auxiliary = (per_auxiliary * auxiliary_weights).sum() / effective_auxiliary_weight.sum().clamp_min(1e-12)
    anchor_kl = F.kl_div(
        F.log_softmax(output["logits"], dim=1),
        anchor_probabilities,
        reduction="batchmean",
    )
    total = (
        center_loss_weight * center
        + auxiliary_loss_weight * auxiliary
        + anchor_kl_weight * anchor_kl
    )
    return total, {"center": center, "auxiliary": auxiliary, "anchor_kl": anchor_kl}
