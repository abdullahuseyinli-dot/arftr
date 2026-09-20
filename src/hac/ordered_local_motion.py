"""Bounded frozen-token heads for ordered local representation changes.

Regions are visual grid cells, not anatomical landmarks. Learned correspondence
and displacement are expressed in that crop grid. They are not physical velocity,
and the frozen V-JEPA observations already incorporate the full clip context.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("pooled_mlp", "temporal_adapter", "ordered_relational", "correspondence_relational")


class OrderedLocalMotion(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        variant: str = "ordered_relational",
        *,
        width: int = 128,
        rank: int = 32,
        regions: int = 9,
        quality_dim: int = 6,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        lags: tuple[int, ...] = (0, 1, 2),
        spatial_penalty: float = 2.0,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        grid = math.isqrt(regions)
        if variant not in ARMS:
            raise ValueError(f"Unknown ordered-motion arm: {variant}")
        if min(input_dim, width, rank, regions, layers, heads, parameter_limit) < 1:
            raise ValueError("Model dimensions must be positive")
        if grid * grid != regions or quality_dim < 0 or width % heads:
            raise ValueError("Regions must be square and width divisible by heads")
        if not 0 <= dropout < 1 or spatial_penalty < 0:
            raise ValueError("Invalid dropout or spatial penalty")
        if not lags or tuple(sorted(set(lags))) != lags or min(lags) < 0:
            raise ValueError("Lags must be unique nonnegative increasing integers")
        self.variant, self.input_dim = variant, input_dim
        self.width, self.rank, self.regions, self.quality_dim = width, rank, regions, quality_dim
        self.lags, self.spatial_penalty = lags, float(spatial_penalty)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.first_order = nn.Sequential(
            nn.Linear(2 * regions * width + quality_dim, width), nn.GELU(), nn.Dropout(dropout)
        )
        if variant == "pooled_mlp":
            self.pooled = nn.Sequential(
                nn.Linear(regions * width, 2 * width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * width, width),
                nn.GELU(),
            )
        else:
            self.region_embedding = nn.Parameter(torch.empty(regions, width))
            nn.init.normal_(self.region_embedding, std=0.02)
            self.time_embedding = nn.Sequential(
                nn.Linear(3, width), nn.GELU(), nn.Linear(width, width)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                width,
                heads,
                2 * width,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
            )
        if variant in ("ordered_relational", "correspondence_relational"):
            self.left_projection = nn.Linear(width, rank, bias=False)
            self.right_projection = nn.Linear(width, rank, bias=False)
            self.time_kernel = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
            self.relation_projection = nn.Sequential(
                nn.Linear(len(lags) * regions * regions, width), nn.GELU()
            )
        if variant == "correspondence_relational":
            self.match_projection = nn.Linear(width, rank, bias=False)
            self.unmatched_logit = nn.Parameter(torch.tensor(-1.0))
            self.displacement_projection = nn.Linear(2, width, bias=False)
            y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
            coordinates = torch.stack((x.flatten(), y.flatten()), dim=-1).float()
            offsets = coordinates[:, None, :] - coordinates[None, :, :]
            self.register_buffer("coordinates", coordinates)
            self.register_buffer("spatial_distance", offsets.square().sum(-1))
            self.register_buffer("local_match", offsets.abs().amax(-1) <= 1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 3)
        )
        if self.trainable_parameters > parameter_limit:
            raise ValueError(
                f"{self.trainable_parameters} parameters exceed limit {parameter_limit}"
            )

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _matching(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_valid: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = F.normalize(self.match_projection(source), dim=-1, eps=1e-6)
        key = F.normalize(self.match_projection(target), dim=-1, eps=1e-6)
        scores = torch.einsum("btpd,btqd->btpq", query, key) * math.sqrt(self.rank)
        scores = scores - self.spatial_penalty * self.spatial_distance
        allowed = self.local_match[None, None, :, :] & target_valid[:, :, None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        unmatched = self.unmatched_logit.expand(*scores.shape[:-1], 1)
        assignment = torch.cat((scores, unmatched), dim=-1).softmax(-1)
        assignment = torch.where(source_valid[..., None], assignment, 0.0)
        matched = assignment[..., :-1]
        support = matched.sum(-1)
        conditional = matched / support.clamp_min(1e-6)[..., None]
        transported = torch.einsum("btpq,btqd->btpd", conditional, target)
        position = torch.einsum("btpq,qd->btpd", conditional, self.coordinates)
        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(-1)
        return transported, position, support, entropy

    def _changes(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target = values[:, :-1], values[:, 1:]
        if self.variant != "correspondence_relational":
            support = (valid[:, :-1] & valid[:, 1:]).to(values.dtype)
            return (
                target - source,
                support,
                torch.zeros_like(support),
                torch.zeros((*support.shape, 2), device=values.device, dtype=values.dtype),
            )
        cross, cross_position, support, entropy = self._matching(
            source, target, valid[:, :-1], valid[:, 1:]
        )
        static, static_position, _, _ = self._matching(source, source, valid[:, :-1], valid[:, :-1])
        displacement = cross_position - static_position
        displacement = torch.where((support > 1e-6)[..., None], displacement, 0.0)
        # Subtract the same soft matching operation on an unchanged source grid.
        # This removes artificial change caused only by soft spatial averaging.
        change = cross - static + self.displacement_projection(displacement)
        change = torch.where((support > 1e-6)[..., None], change, 0.0)
        return change, support, entropy, displacement

    def _interactions(
        self, changes: torch.Tensor, support: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        dt = times[:, 1:] - times[:, :-1]
        midpoints = (times[:, 1:] + times[:, :-1]) / 2
        left, right = self.left_projection(changes), self.right_projection(changes)
        outputs = []
        for lag in self.lags:
            length = changes.shape[1] - lag
            a, b = left[:, :length], right[:, lag:]
            products = torch.einsum("btpd,btqd->btpq", a, b) / math.sqrt(self.rank)
            physical = torch.stack(
                (dt[:, :length], dt[:, lag:], midpoints[:, lag:] - midpoints[:, :length]), dim=-1
            )
            time_gain = F.softplus(self.time_kernel(physical).squeeze(-1)) + 1e-6
            weights = support[:, :length, :, None] * support[:, lag:, None, :]
            # Normalize only observation support. Normalizing by the learned
            # time gain would cancel it completely on uniformly sampled clips.
            numerator = (weights * time_gain[:, :, None, None] * products).sum(dim=1)
            denominator = weights.sum(dim=1).clamp_min(1e-6)
            outputs.append(numerator / denominator)
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        times: torch.Tensor,
        quality: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        center_index: int = 4,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 4 or tokens.shape[2:] != (self.regions, self.input_dim):
            raise ValueError("tokens must be [batch, time, regions, input_dim]")
        batch, temporal, regions, _ = tokens.shape
        if temporal <= max(self.lags) + 1 or not 0 <= center_index < temporal or batch < 1:
            raise ValueError("Not enough times for lags or invalid center index")
        if times.shape != (batch, temporal) or not torch.isfinite(times).all():
            raise ValueError("Physical times must be finite [batch, time]")
        if torch.any(times[:, 1:] <= times[:, :-1]):
            raise ValueError("Physical timestamps must increase strictly")
        if valid is None:
            valid = torch.ones(tokens.shape[:3], dtype=torch.bool, device=tokens.device)
        if valid.shape != tokens.shape[:3] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch, time, regions]")
        if not valid.flatten(1).any(1).all() or not valid[:, center_index].any(1).all():
            raise ValueError("Each clip needs a valid center and at least one observation")
        if not torch.isfinite(tokens[valid]).all():
            raise ValueError("Valid tokens must be finite")
        dtype = self.projection[1].weight.dtype
        safe = torch.where(valid[..., None], tokens, 0.0).to(dtype)
        times = times.to(dtype)
        if quality is None:
            quality = torch.zeros(batch, self.quality_dim, dtype=dtype, device=tokens.device)
        if quality.shape != (batch, self.quality_dim) or not torch.isfinite(quality).all():
            raise ValueError("Quality must be finite [batch, quality_dim]")
        projected = torch.where(valid[..., None], self.projection(safe), 0.0)
        region_mean = projected.sum(1) / valid.sum(1).clamp_min(1)[..., None]
        first = self.first_order(
            torch.cat(
                (region_mean.flatten(1), projected[:, center_index].flatten(1), quality.to(dtype)),
                dim=-1,
            )
        )
        if self.variant == "pooled_mlp":
            contextual = self.pooled(region_mean.flatten(1))
        else:
            offset = times - times[:, center_index, None]
            position = torch.stack((offset, offset.abs(), offset.square()), dim=-1)
            values = (
                projected
                + self.region_embedding[None, None]
                + self.time_embedding(position)[:, :, None]
            )
            values = self.temporal(
                values.flatten(1, 2), src_key_padding_mask=~valid.flatten(1)
            ).reshape(batch, temporal, regions, self.width)
            values = torch.where(valid[..., None], values, 0.0)
            contextual = values.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)[:, None]
        interactions = torch.zeros(
            (batch, len(self.lags), regions, regions), dtype=dtype, device=tokens.device
        )
        entropy = torch.zeros((batch, temporal - 1, regions), dtype=dtype, device=tokens.device)
        displacement = torch.zeros((*entropy.shape, 2), dtype=dtype, device=tokens.device)
        if self.variant in ("ordered_relational", "correspondence_relational"):
            changes, support, entropy, displacement = self._changes(projected, valid)
            interactions = self._interactions(changes, support, times)
            contextual = contextual + self.relation_projection(interactions.flatten(1))
        logits = self.classifier(torch.cat((first, contextual), dim=-1))
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "interactions": interactions,
            "correspondence_entropy": entropy,
            "correspondence_displacement": displacement,
        }
