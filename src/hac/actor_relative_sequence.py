"""Timestamped actor-frame sequence heads for the HAC evidence trial."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = (
    "s0_independent_all_frames",
    "s1_unordered_pooled_all_frames",
    "s2_timestamp_tcn_all_frames",
    "s3_actor_relative_tcn_all_frames",
    "s4_actor_relative_relation_all_frames",
)


def seed_sequence_training(seed: int) -> None:
    import random

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


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class ActorRelativeStateSequence(nn.Module):
    """Matched frame, unordered, temporal, and actor-relative sequence heads."""

    def __init__(
        self,
        arm: str,
        *,
        input_dim: int = 768,
        width: int = 64,
        dropout: float = 0.1,
        temporal_blocks: int = 2,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if arm not in ARMS or min(input_dim, width, temporal_blocks, parameter_limit) < 1:
            raise ValueError("Invalid actor-relative sequence architecture")
        if not 0 <= dropout < 1 or temporal_blocks > 2:
            raise ValueError("Invalid sequence dropout or temporal block count")
        self.arm, self.input_dim, self.width = arm, input_dim, width
        self.relative = arm in ARMS[3:]
        self.temporal = arm in ARMS[2:]
        self.unordered = arm == ARMS[1]
        self.relation = arm == ARMS[4]
        self.token = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU())
        self.time = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.dropout = nn.Dropout(dropout)
        if self.temporal:
            self.temporal_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width),
                        nn.GELU(),
                        nn.Conv1d(width, width, kernel_size=1),
                        nn.GELU(),
                    )
                    for _ in range(temporal_blocks)
                ]
            )
        else:
            self.temporal_layers = nn.ModuleList()
        if self.unordered:
            self.context = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout))
        else:
            self.context = nn.Identity()
        self.classifier = nn.Linear(width, 3)
        self.relation_scale = nn.Parameter(torch.tensor(1.0)) if self.relation else None
        self.relation_bias = nn.Parameter(torch.tensor(0.0)) if self.relation else None
        if self.trainable_parameters > parameter_limit:
            raise ValueError("Actor-relative sequence exceeds the declared parameter limit")

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def states(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        streams: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if (
            features.ndim != 3
            or features.shape[2] != self.input_dim
            or timestamps.shape != features.shape[:2]
            or streams.shape != features.shape[:2]
            or valid.shape != features.shape[:2]
            or not features.is_floating_point()
            or not timestamps.is_floating_point()
            or not torch.isfinite(features).all()
            or not torch.isfinite(timestamps).all()
            or valid.dtype != torch.bool
            or not valid.any(1).all()
        ):
            raise ValueError("Malformed actor-frame sequence inputs")
        token = self.token(features)
        if self.temporal:
            time_input = torch.stack(
                (timestamps, timestamps.square(), streams.to(timestamps.dtype)), dim=-1
            )
            token = token + self.time(time_input)
        if self.relative:
            token = token + (token - _masked_mean(token, valid).unsqueeze(1))
        if self.unordered:
            context = _masked_mean(token, valid).unsqueeze(1).expand_as(token)
            token = self.context(torch.cat((token, context), dim=-1))
        if self.temporal:
            masked = token * valid.unsqueeze(-1).to(token.dtype)
            for layer in self.temporal_layers:
                update = layer(masked.transpose(1, 2)).transpose(1, 2)
                masked = (masked + self.dropout(update)) * valid.unsqueeze(-1).to(token.dtype)
            token = masked
        return token * valid.unsqueeze(-1).to(token.dtype)

    def forward(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        streams: torch.Tensor,
        valid: torch.Tensor,
        center_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if center_indices.shape != (len(features),) or torch.any(center_indices < 0) or torch.any(
            center_indices >= features.shape[1]
        ):
            raise ValueError("Center indices are malformed")
        states = self.states(features, timestamps, streams, valid)
        if not valid.gather(1, center_indices[:, None]).all():
            raise ValueError("Every center index must be an observed frame")
        logits = self.classifier(states)
        centers = states.gather(1, center_indices[:, None, None].expand(-1, 1, self.width)).squeeze(1)
        result = {"frame_logits": logits, "center_logits": self.classifier(centers), "states": states}
        if self.relation:
            normalized = F.normalize(states, dim=-1, eps=1e-8)
            relation = self.relation_scale * normalized @ normalized.transpose(1, 2) + self.relation_bias
            result["relation_logits"] = relation
        return result


def sequence_loss(
    model: ActorRelativeStateSequence,
    *,
    features: torch.Tensor,
    timestamps: torch.Tensor,
    streams: torch.Tensor,
    valid: torch.Tensor,
    center_indices: torch.Tensor,
    labels: torch.Tensor,
    physical_weights: torch.Tensor,
    class_weight: torch.Tensor,
    relation_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if (
        labels.shape != valid.shape
        or physical_weights.shape != valid.shape
        or class_weight.shape != (3,)
        or labels.dtype != torch.long
        or torch.any(labels[valid] < 0)
        or torch.any(labels[valid] > 2)
        or not torch.isfinite(physical_weights).all()
        or torch.any(physical_weights[valid] <= 0)
        or torch.any(physical_weights[~valid] != 0)
        or not torch.isfinite(class_weight).all()
        or torch.any(class_weight <= 0)
        or relation_loss_weight < 0
    ):
        raise ValueError("Malformed actor-relative sequence loss inputs")
    output = model(features, timestamps, streams, valid, center_indices)
    frame_loss = F.cross_entropy(
        output["frame_logits"].transpose(1, 2), labels.clamp_min(0), weight=class_weight, reduction="none"
    )
    effective = physical_weights * class_weight[labels.clamp_min(0)] * valid.to(physical_weights.dtype)
    frame = (frame_loss * physical_weights * valid).sum() / effective.sum().clamp_min(1e-12)
    center_labels = labels.gather(1, center_indices[:, None]).squeeze(1)
    center = F.cross_entropy(output["center_logits"], center_labels, weight=class_weight)
    relation = torch.zeros((), device=features.device)
    if model.relation:
        pair_valid = valid[:, :, None] & valid[:, None, :]
        pair_valid &= torch.triu(torch.ones_like(pair_valid), diagonal=1).bool()
        same = (labels[:, :, None] == labels[:, None, :]).to(features.dtype)
        pair_loss = F.binary_cross_entropy_with_logits(output["relation_logits"], same, reduction="none")
        pair_weight = (physical_weights[:, :, None] * physical_weights[:, None, :]) * pair_valid
        relation = (pair_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-12)
    total = frame + center + relation_loss_weight * relation
    return total, {"frame": frame, "center": center, "relation": relation}
