"""Small probes for frozen spatiotemporal features.

The probe deliberately keeps the video backbone frozen.  Time and spatial-region
positions are represented separately so flattening a pooled feature grid does not
silently turn spatial cells into later time steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from hac.vcoco_v3_neural import FactorizedClassifier, decode_factorized_logits


@dataclass(frozen=True)
class VideoProbeOutput:
    probabilities: torch.Tensor
    posture_logits: torch.Tensor
    motion_logits: torch.Tensor
    pooled_features: torch.Tensor
    attention_weights: torch.Tensor


class SpatiotemporalFactorizedProbe(nn.Module):
    """A bounded-capacity attention probe over frozen ``[time, region]`` tokens."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 192,
        layers: int = 1,
        attention_heads: int = 4,
        feedforward_dim: int = 384,
        dropout: float = 0.1,
        maximum_times: int = 16,
        maximum_regions: int = 9,
    ) -> None:
        super().__init__()
        if (
            min(
                input_dim,
                model_dim,
                layers,
                attention_heads,
                feedforward_dim,
                maximum_times,
                maximum_regions,
            )
            < 1
        ):
            raise ValueError("Probe dimensions must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.maximum_times = int(maximum_times)
        self.maximum_regions = int(maximum_regions)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_embedding = nn.Parameter(torch.empty(maximum_times, model_dim))
        self.region_embedding = nn.Parameter(torch.empty(maximum_regions, model_dim))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.region_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.pool_query = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.pool_query, std=model_dim**-0.5)
        self.classifier = FactorizedClassifier(model_dim, dropout)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> VideoProbeOutput:
        if features.ndim != 4:
            raise ValueError("features must have shape [batch, time, region, dimension]")
        batch, times, regions, _ = features.shape
        if times > self.maximum_times or regions > self.maximum_regions:
            raise ValueError("Feature grid exceeds configured positional embeddings")
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, times, regions), dtype=torch.bool, device=features.device
            )
        if valid_mask.shape != (batch, times, regions) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [batch, time, region]")
        flat_mask = valid_mask.reshape(batch, times * regions)
        if not torch.all(flat_mask.any(dim=1)):
            raise ValueError("Every sample must contain at least one valid token")

        values = self.input_projection(features)
        values = (
            values
            + self.time_embedding[:times][None, :, None, :]
            + self.region_embedding[:regions][None, None, :, :]
        )
        values = values.reshape(batch, times * regions, -1)
        values = self.encoder(values, src_key_padding_mask=~flat_mask)
        scores = torch.einsum("bnd,d->bn", values, self.pool_query) / math.sqrt(values.shape[-1])
        scores = scores.masked_fill(~flat_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bn,bnd->bd", weights, values)
        posture_logits, motion_logits = self.classifier(pooled)
        return VideoProbeOutput(
            probabilities=decode_factorized_logits(posture_logits, motion_logits),
            posture_logits=posture_logits,
            motion_logits=motion_logits,
            pooled_features=pooled,
            attention_weights=weights.reshape(batch, times, regions),
        )
