"""Numerical-only v2 of center-episode expectation, preserving the v1 module.

Float32 subtraction of two large cumulative hazard totals can erase or invent a
small local path exposure. Independently computed stopping probabilities then no
longer form the same distribution as the reported path survival. Final attention
normalization can consequently break the intended bound.

This version sums only the edges on each actual path and computes probability
arithmetic in float64. Neural layers, parameter schema, losses and hypotheses are
unchanged; no attention clipping or relaxed bound is used. Both matched arms must
be refit under a new protocol rather than mixing v1 and v2 trained checkpoints.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from hac.center_episode_memory import CenterEpisodeMemory

NUMERICAL_VERSION = "direct-path-float64-v2"


def stable_expected_episode_attention(
    utility: torch.Tensor,
    edge_mass: torch.Tensor,
    valid: torch.Tensor,
    centers: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Same latent-segment distribution as v1, with stable path/probability math.

    Destination edge convention and missing-slot bridging are unchanged. Float64
    arithmetic is used only for the five-node/nine-segment probability operations;
    gradients through utility cast back to its original neural dtype.
    """
    batch, observations = valid.shape
    if observations != 5 or utility.shape != valid.shape or edge_mass.shape != valid.shape:
        raise ValueError("Episode attention requires aligned [batch, 5] tensors")
    if centers.shape != (batch,):
        raise ValueError("centers must have shape [batch]")
    slots = torch.arange(observations, device=valid.device).expand(batch, -1)
    mass = torch.where(valid, edge_mass.detach(), 0.0).to(torch.float64)
    lower = torch.minimum(slots, centers[:, None])
    upper = torch.maximum(slots, centers[:, None])
    edge_slots = slots[:, None, :]
    on_path = (edge_slots > lower[..., None]) & (edge_slots <= upper[..., None])
    # No cumulative-prefix subtraction: an unrelated large mass cannot corrupt
    # the small elapsed exposure between the center and an adjacent observation.
    log_survival = -torch.where(on_path, mass[:, None, :], 0.0).sum(-1)
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
    within = (
        utility.to(torch.float64)[:, None, :]
        .expand(-1, 9, -1)
        .masked_fill(~membership, -torch.inf)
        .softmax(-1)
    )
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


class StableCenterEpisodeMemory(CenterEpisodeMemory):
    """V1 parameter/state schema with stable arithmetic in both matched arms.

    Inherits all unchanged input, hazard, parameter-group and initialization code.
    The frozen parent source is therefore an explicit dependency in v2 provenance.
    Neural memory/gate inputs use the layer dtype; final probability mixtures use
    the original caller dtype exactly, including singleton and zero-gate fallback.
    """

    numerical_version = NUMERICAL_VERSION

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
        episode = stable_expected_episode_attention(
            utility, result["boundary_mass"], valid, centers
        )
        result.update(episode)
        if self.variant == "log_survival_control":
            result["attention"] = (
                (utility.to(torch.float64) + episode["log_survival"])
                .masked_fill(~valid, -torch.inf)
                .softmax(1)
            )
        attention = result["attention"]
        neural_attention = attention.to(base.dtype)
        memory = torch.einsum("bn,bnc->bc", neural_attention, base)
        retrieved = torch.einsum("bn,bnd->bd", neural_attention, contextual)
        gate_inputs = torch.cat(
            (
                center_values,
                retrieved,
                center_base,
                memory,
                valid.sum(-1, keepdim=True) / 5,
                (neural_attention * scaled.abs()).sum(-1, keepdim=True),
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
