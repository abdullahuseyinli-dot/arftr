"""Exploratory center-episode marginalization; no active memory code is changed.

An independently supervised hazard defines a distribution over the contiguous
observed segments containing the center. Utility is normalized *inside* each
segment and then marginalized. Therefore a node's probability-mixture attention
cannot exceed its predicted probability of belonging to the center episode.
This is a coefficient bound, not guaranteed state correctness or information
isolation: contextual utility/gating can still see all allowed observations.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("expected_episode", "log_survival_control")


def expected_episode_attention(
    utility: torch.Tensor,
    edge_mass: torch.Tensor,
    valid: torch.Tensor,
    centers: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Marginalize at most nine center-containing segments in a five-slot window.

    ``edge_mass[:, j]`` is rate times physical duration for the edge from the
    preceding available node into slot j. The first/invalid nodes have mass zero.
    Hazards are detached here as a defense against accidental classification
    gradients; use the model's original boundary logits for auxiliary training.
    Missing slots are not potential boundaries or empty episode endpoints.
    """
    batch, observations = valid.shape
    if observations != 5 or utility.shape != valid.shape or edge_mass.shape != valid.shape:
        raise ValueError("Episode attention requires aligned [batch, 5] tensors")
    if centers.shape != (batch,):
        raise ValueError("centers must have shape [batch]")
    slots = torch.arange(observations, device=valid.device).expand(batch, -1)
    mass = torch.where(valid, edge_mass.detach(), 0.0)
    cumulative = mass.cumsum(1)
    log_survival = -(cumulative - cumulative.gather(1, centers[:, None])).abs()
    survival = torch.where(valid, log_survival.exp(), 0.0)
    boundary = -torch.expm1(-mass)

    preceding = F.pad(
        torch.cummax(torch.where(valid, slots, -1), 1).values[:, :-1], (1, 0), value=-1
    )
    reversed_next = torch.cummin(torch.where(valid, slots, observations).flip(1), 1).values.flip(1)
    following = F.pad(reversed_next[:, 1:], (0, 1), value=observations)
    left_stop = survival * torch.where(preceding >= 0, boundary, 1.0)
    next_boundary = boundary.gather(1, following.clamp_max(observations - 1))
    right_stop = survival * torch.where(following < observations, next_boundary, 1.0)

    left_candidates = (
        torch.where(valid & (slots <= centers[:, None]), slots, observations).sort(1).values
    )
    right_candidates = (
        torch.where(valid & (slots >= centers[:, None]), slots, observations).sort(1).values
    )
    left_count = (left_candidates < observations).sum(1)
    right_count = (right_candidates < observations).sum(1)
    # For five slots, (available_left+1)*(available_right+1) <= 9.
    ordinal = torch.arange(9, device=valid.device).expand(batch, -1)
    segment_valid = ordinal < (left_count * right_count)[:, None]
    left_ordinal = torch.div(ordinal, right_count[:, None], rounding_mode="floor")
    right_ordinal = ordinal % right_count[:, None]
    left = left_candidates.gather(1, left_ordinal.clamp_max(observations - 1))
    right = right_candidates.gather(1, right_ordinal)
    left = torch.where(segment_valid, left, centers[:, None])
    right = torch.where(segment_valid, right, centers[:, None])
    segment_probability = left_stop.gather(1, left) * right_stop.gather(1, right)
    segment_probability = torch.where(segment_valid, segment_probability, 0.0)
    membership = (
        valid[:, None, :]
        & (slots[:, None, :] >= left[..., None])
        & (slots[:, None, :] <= right[..., None])
    )
    within = utility[:, None, :].expand(-1, 9, -1).masked_fill(~membership, -torch.inf).softmax(-1)
    # Invalid padded segments contain the center for finite softmax, but have zero mass.
    attention = (segment_probability[..., None] * within).sum(1)
    return {
        "attention": attention,
        "survival": survival,
        "log_survival": torch.where(valid, log_survival, -torch.inf),
        "segment_probabilities": segment_probability,
        "segment_valid": segment_valid,
        "segment_left_index": torch.where(segment_valid, left, -1),
        "segment_right_index": torch.where(segment_valid, right, -1),
        "segment_attention": torch.where(segment_valid[..., None], within, 0.0),
    }


def unweighted_boundary_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    annotation_valid: torch.Tensor,
) -> torch.Tensor:
    """Proper unweighted edge BCE; unknown annotation intervals are never negatives.

    No positive weights, focal terms, class balancing, or correctness filtering.
    The caller can multiply this entire mean by a fixed auxiliary coefficient.
    Edge occurrence weighting is uniform; repeated windows can repeat an edge and
    must be disclosed in calibration diagnostics, not called independent trials.
    """
    logits = output["boundary_logits"]
    if targets.shape != logits.shape or annotation_valid.shape != logits.shape:
        raise ValueError("Boundary targets and annotation mask must match boundary logits")
    if annotation_valid.dtype != torch.bool:
        raise ValueError("Boundary annotation mask must be boolean")
    mask = output["boundary_valid"] & annotation_valid
    observed = targets[mask]
    if not torch.isfinite(observed).all() or torch.any((observed < 0) | (observed > 1)):
        raise ValueError("Known boundary targets must be finite and between zero and one")
    safe_targets = torch.where(mask, targets, 0.0).to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    return torch.where(mask, losses, 0.0).sum() / mask.sum().clamp_min(1)


class CenterEpisodeMemory(nn.Module):
    """Matched episode-expectation and recalibrated log-survival attention heads.

    Hazards receive independently projected frozen node features, no probabilities,
    center-relative coordinates, or contextual utility activations. Edge rates use
    ordered endpoint features; physical dt enters their exposure mass. Classification
    cannot train this branch. Inference requires no labels or known state boundaries.
    Feature/quality scaling and all nested base probabilities remain caller-owned.
    """

    def __init__(
        self,
        input_dim: int,
        variant: str = "expected_episode",
        *,
        width: int = 128,
        hazard_width: int = 32,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        time_decay_seconds: float = 2.0,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown episode variant: {variant}")
        if min(input_dim, width, hazard_width, layers, heads, parameter_limit) < 1 or width % heads:
            raise ValueError("Dimensions must be positive and width divisible by heads")
        if not 0 <= dropout < 1 or time_decay_seconds <= 0:
            raise ValueError("Invalid dropout or physical time scale")
        self.input_dim = int(input_dim)
        self.variant = variant
        self.width = int(width)
        self.hazard_width = int(hazard_width)
        self.time_decay_seconds = float(time_decay_seconds)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.side_projection = nn.Linear(5, width)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.usefulness = nn.Sequential(nn.Linear(3 * width + 7, 64), nn.GELU(), nn.Linear(64, 1))
        self.hazard_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hazard_width), nn.GELU()
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(4 * hazard_width, 32), nn.GELU(), nn.Linear(32, 1)
        )
        nn.init.constant_(self.hazard_head[-1].bias, math.log(math.expm1(0.1)))
        self.fusion_gate = nn.Sequential(nn.Linear(2 * width + 8, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.normal_(self.fusion_gate[-1].weight, std=0.01)
        nn.init.constant_(self.fusion_gate[-1].bias, -4.0)
        if self.trainable_parameters > parameter_limit:
            raise ValueError(
                f"Episode memory has {self.trainable_parameters} parameters, exceeding {parameter_limit}"
            )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def hazard_parameters(self) -> Iterator[nn.Parameter]:
        """Independent proper-BCE branch; clip separately from classification."""
        yield from self.hazard_projection.parameters()
        yield from self.hazard_head.parameters()

    def utility_parameters(self) -> Iterator[nn.Parameter]:
        """Utility, contextual representation and fusion gate, excluding hazards."""
        for name, parameter in self.named_parameters():
            if not name.startswith("hazard_"):
                yield parameter

    def _inputs(
        self,
        features: torch.Tensor,
        probabilities: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        center_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (5, self.input_dim) or features.shape[0] < 1:
            raise ValueError("features must have shape [nonempty batch, 5, input_dim]")
        batch = features.shape[0]
        if (
            probabilities.shape != (batch, 5, 3)
            or times.shape != (batch, 5)
            or valid.shape != (batch, 5)
        ):
            raise ValueError("Probabilities, times and validity must align with five feature slots")
        if valid.dtype != torch.bool or not all(
            x.is_floating_point() for x in (features, probabilities, times)
        ):
            raise ValueError("Validity must be boolean and observations floating point")
        if any(x.device != features.device for x in (probabilities, times, valid)):
            raise ValueError("All observation inputs must share a device")
        if isinstance(center_index, int):
            centers = torch.full((batch,), center_index, device=features.device, dtype=torch.long)
        else:
            if center_index.shape != (batch,) or center_index.dtype != torch.long:
                raise ValueError("center_index must be an integer or a long tensor [batch]")
            centers = center_index.to(features.device)
        if (
            torch.any((centers < 0) | (centers >= 5))
            or not valid.gather(1, centers.clamp(0, 4)[:, None]).all()
        ):
            raise ValueError("Every center index must select a valid observation")
        if not all(torch.isfinite(x[valid]).all() for x in (features, probabilities, times)):
            raise ValueError("Valid observations must contain finite inputs")
        observed = probabilities[valid]
        if torch.any(observed < 0) or not torch.allclose(
            observed.sum(-1), torch.ones_like(observed[:, 0]), atol=1e-5, rtol=0
        ):
            raise ValueError("Valid probabilities must be nonnegative and sum to one")
        dtype = self.input_projection[1].weight.dtype
        return (
            torch.where(valid[..., None], features, 0.0).to(dtype),
            torch.where(valid[..., None], probabilities, 0.0).to(dtype),
            torch.where(valid, times, 0.0).to(dtype),
            centers,
        )

    def _boundaries(
        self, features: torch.Tensor, times: torch.Tensor, valid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        slots = torch.arange(5, device=valid.device).expand_as(valid)
        previous = F.pad(
            torch.cummax(torch.where(valid, slots, -1), 1).values[:, :-1], (1, 0), value=-1
        )
        edge_valid = valid & (previous >= 0)
        left = previous.clamp_min(0)
        dt = times - times.gather(1, left)
        if torch.any(edge_valid & (dt <= 0)):
            raise ValueError(
                "Available observations require strictly increasing physical timestamps"
            )
        dt = torch.where(edge_valid, dt, 0.0)
        values = self.hazard_projection(features.detach())
        left_values = values.gather(1, left[..., None].expand(-1, -1, self.hazard_width))
        delta = values - left_values
        edge_features = torch.cat((left_values, values, delta, delta.abs()), -1)
        rate = F.softplus(self.hazard_head(edge_features).squeeze(-1))
        rate = torch.where(edge_valid, rate, 0.0)
        mass = rate * dt
        safe_mass = mass.clamp_min(torch.finfo(mass.dtype).tiny)
        logits = safe_mass + torch.log(-torch.expm1(-safe_mass))
        boundary = torch.where(edge_valid, -torch.expm1(-mass), 0.0)
        return {
            "boundary_logits": torch.where(edge_valid, logits, 0.0),
            "boundary_probabilities": boundary,
            "boundary_probs": boundary,
            "boundary_rates": rate,
            "boundary_mass": mass,
            "boundary_dt": dt,
            "boundary_valid": edge_valid,
            "boundary_left_index": torch.where(edge_valid, previous, -1),
        }

    def forward(
        self,
        features: torch.Tensor,
        probabilities: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        center_index: int | torch.Tensor = 2,
    ) -> dict[str, torch.Tensor]:
        features, base, times, centers = self._inputs(
            features, probabilities, times, valid, center_index
        )
        result = self._boundaries(features, times, valid)
        batch = features.shape[0]
        row = torch.arange(batch, device=features.device)
        relative = torch.where(valid, times - times.gather(1, centers[:, None]), 0.0)
        scaled = relative / self.time_decay_seconds
        side = torch.cat((base, scaled[..., None], scaled.abs()[..., None]), -1)
        projected = self.input_projection(features) + self.side_projection(side)
        projected = torch.where(valid[..., None], projected, 0.0)
        contextual = self.temporal(projected, src_key_padding_mask=~valid)
        contextual = torch.where(valid[..., None], contextual, 0.0)
        center_values = contextual[row, centers]
        center_expanded = center_values[:, None, :].expand(-1, 5, -1)
        center_base = base[row, centers]
        pair = torch.cat(
            (
                center_expanded,
                contextual,
                (contextual - center_expanded).abs(),
                center_base[:, None, :].expand(-1, 5, -1),
                base,
                scaled.abs()[..., None],
            ),
            -1,
        )
        utility = torch.einsum(
            "bd,bnd->bn", self.query(center_values), self.key(contextual)
        ) / math.sqrt(self.width)
        utility = utility + self.usefulness(pair).squeeze(-1) - scaled.abs()
        episode = expected_episode_attention(utility, result["boundary_mass"], valid, centers)
        result.update(episode)
        if self.variant == "log_survival_control":
            result["attention"] = (
                (utility + episode["log_survival"]).masked_fill(~valid, -torch.inf).softmax(1)
            )
        attention = result["attention"]
        memory = torch.einsum("bn,bnc->bc", attention, base)
        retrieved = torch.einsum("bn,bnd->bd", attention, contextual)
        gate_inputs = torch.cat(
            (
                center_values,
                retrieved,
                center_base,
                memory,
                valid.sum(-1, keepdim=True) / 5,
                (attention * scaled.abs()).sum(-1, keepdim=True),
            ),
            -1,
        )
        gate = self.fusion_gate(gate_inputs).squeeze(-1).sigmoid()
        has_neighbor = valid.sum(1) > 1
        gate = torch.where(has_neighbor, gate, 0.0)
        original_base = torch.where(valid[..., None], probabilities, 0.0)
        original_center = probabilities[row, centers]
        mixture_attention = attention.to(probabilities.dtype)
        mixture_attention = mixture_attention / mixture_attention.sum(-1, keepdim=True)
        original_memory = torch.einsum("bn,bnc->bc", mixture_attention, original_base)
        mixture_gate = gate.to(probabilities.dtype)[:, None]
        prediction = (1 - mixture_gate) * original_center + mixture_gate * original_memory
        result.update(
            {
                "probabilities": torch.where(has_neighbor[:, None], prediction, original_center),
                "memory_probabilities": original_memory,
                "attention": mixture_attention,
                "gate": gate,
                "utility": utility,
            }
        )
        return result
