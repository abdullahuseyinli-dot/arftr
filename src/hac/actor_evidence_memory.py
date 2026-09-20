"""Small observation-only heads for center-connected actor evidence retrieval.

Inputs are temporally ordered observations from one actor track. The caller owns
scenario isolation, base-model cross-fitting, feature scaling, and boundary labels.
No labels, IDs, or known correctness enter ``forward``. An edge is stored at its
destination slot and connects the preceding *available* observation, including
across missing slots. Boundary targets must follow that same edge contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = (
    "temporal_conv",
    "query_attention",
    "survival_memory",
    "corroborated_memory",
)


class _TemporalBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        update = self.conv(self.norm(values).transpose(1, 2)).transpose(1, 2)
        result = values + self.dropout(F.gelu(update))
        return torch.where(valid[..., None], result, 0.0)


class ActorEvidenceMemory(nn.Module):
    """M2--M5 with a fixed center fallback and a probability-mixture output.

    ``boundary_logits`` describe whether any target-state change occurred over
    the destination edge, suitable for BCE with logits. Invalid/first edges use
    logit zero and probability zero; always apply ``boundary_valid`` together
    with the caller's boundary-annotation mask. M2 has no boundary head and its
    boundary mask is entirely false.

    The time-based hazard parameterization is density invariant for a constant
    rate, not a claim that learned rates are invariant to changing observations.
    M3 and M4 have identical trainable modules; only M4 uses survival in attention.
    """

    def __init__(
        self,
        input_dim: int,
        variant: str = "survival_memory",
        *,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        time_decay_seconds: float = 2.0,
        survival_epsilon: float = 1e-6,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown memory variant: {variant}")
        if min(input_dim, width, layers, heads, parameter_limit) < 1 or width % heads:
            raise ValueError("Memory dimensions must be positive and width divisible by heads")
        if not 0 <= dropout < 1 or time_decay_seconds <= 0 or not 0 < survival_epsilon < 1:
            raise ValueError("Invalid memory dropout, time scale, or survival epsilon")
        self.input_dim = int(input_dim)
        self.variant = variant
        self.width = int(width)
        self.time_decay_seconds = float(time_decay_seconds)
        self.survival_epsilon = float(survival_epsilon)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        # Probabilities and signed/absolute physical time are separate from the
        # training-standardized frozen visual features.
        self.side_projection = nn.Linear(5, width)
        if variant == "temporal_conv":
            self.temporal = nn.ModuleList([_TemporalBlock(width, dropout) for _ in range(layers)])
            self.temporal_score = nn.Linear(width, 1)
            self.boundary_head = None
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=2 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                layer, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
            )
            self.query = nn.Linear(width, width, bias=False)
            self.key = nn.Linear(width, width, bias=False)
            self.usefulness = nn.Sequential(
                nn.Linear(3 * width + 7, 64), nn.GELU(), nn.Linear(64, 1)
            )
            self.boundary_head = nn.Sequential(
                nn.Linear(4 * width + 4, 64), nn.GELU(), nn.Linear(64, 1)
            )
            # Initially 0.1 state changes per second; classification has a soft
            # continuation prior while boundary supervision is first learned.
            nn.init.constant_(self.boundary_head[-1].bias, math.log(math.expm1(0.1)))
        self.fusion_gate = nn.Sequential(nn.Linear(2 * width + 8, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.normal_(self.fusion_gate[-1].weight, std=0.01)
        nn.init.constant_(self.fusion_gate[-1].bias, -4.0)
        if variant == "corroborated_memory":
            self.support_score = nn.Sequential(
                nn.Linear(width + 12, 64), nn.GELU(), nn.Linear(64, 1)
            )
            self.support_gate = nn.Linear(9, 1, bias=False)
        if self.trainable_parameters > parameter_limit:
            raise ValueError(
                f"Memory has {self.trainable_parameters} parameters, exceeding {parameter_limit}"
            )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _inputs(
        self,
        features: torch.Tensor,
        probabilities: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        center_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("features must have shape [batch, observations, input_dim]")
        batch, observations, _ = features.shape
        if batch < 1 or observations < 1:
            raise ValueError("Memory requires a nonempty batch and observation axis")
        if probabilities.shape != (batch, observations, 3):
            raise ValueError("probabilities must have shape [batch, observations, 3]")
        if times.shape != (batch, observations) or valid.shape != times.shape:
            raise ValueError("times and valid must match the first two feature axes")
        if valid.dtype != torch.bool:
            raise ValueError("valid must be boolean")
        if any(value.device != features.device for value in (probabilities, times, valid)):
            raise ValueError("All memory inputs must be on the same device")
        if not all(value.is_floating_point() for value in (features, probabilities, times)):
            raise ValueError("Features, probabilities, and physical times must be floating point")
        if isinstance(center_index, int):
            centers = torch.full((batch,), center_index, dtype=torch.long, device=features.device)
        else:
            if center_index.shape != (batch,) or center_index.dtype != torch.long:
                raise ValueError("center_index must be an integer or a long tensor [batch]")
            centers = center_index.to(features.device)
        if torch.any((centers < 0) | (centers >= observations)):
            raise ValueError("center_index is outside the observation axis")
        if not torch.all(valid.gather(1, centers[:, None])):
            raise ValueError("Every center observation must be valid")
        if not (
            torch.isfinite(features[valid]).all()
            and torch.isfinite(probabilities[valid]).all()
            and torch.isfinite(times[valid]).all()
        ):
            raise ValueError("Valid observations must contain finite inputs")
        observed_probabilities = probabilities[valid]
        if torch.any(observed_probabilities < 0) or not torch.allclose(
            observed_probabilities.sum(-1),
            torch.ones_like(observed_probabilities[:, 0]),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Valid base probabilities must be nonnegative and sum to one")
        dtype = self.input_projection[1].weight.dtype
        safe_features = torch.where(valid[..., None], features, 0.0).to(dtype)
        safe_probabilities = torch.where(valid[..., None], probabilities, 0.0).to(dtype)
        safe_times = torch.where(valid, times, 0.0).to(dtype)
        return safe_features, safe_probabilities, safe_times, centers

    def _boundaries(
        self,
        values: torch.Tensor,
        probabilities: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        centers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, observations = valid.shape
        slots = torch.arange(observations, device=valid.device).expand(batch, -1)
        last_available = torch.cummax(torch.where(valid, slots, -1), dim=1).values
        previous = F.pad(last_available[:, :-1], (1, 0), value=-1)
        edge_valid = valid & (previous >= 0)
        left = previous.clamp_min(0)
        dt = times - times.gather(1, left)
        if torch.any(edge_valid & (dt < 0)):
            raise ValueError("Available observations must be ordered by physical timestamp")
        # Duplicated timestamps provide no elapsed exposure and no boundary target.
        edge_valid = edge_valid & (dt > 0)
        dt = torch.where(edge_valid, dt, 0.0)
        if self.boundary_head is None:
            rates = torch.zeros_like(times)
            edge_valid = torch.zeros_like(edge_valid)
        else:
            left_values = values.gather(1, left[..., None].expand(-1, -1, self.width))
            delta = values - left_values
            delta_p = probabilities - probabilities.gather(1, left[..., None].expand(-1, -1, 3))
            edge_features = torch.cat(
                (left_values, values, delta, delta.abs(), delta_p, dt[..., None]), dim=-1
            )
            rates = F.softplus(self.boundary_head(edge_features).squeeze(-1))
            rates = torch.where(edge_valid, rates, 0.0)
        mass = rates * dt
        cumulative = mass.cumsum(dim=1)
        distance = (cumulative - cumulative.gather(1, centers[:, None])).abs()
        survival = torch.where(valid, torch.exp(-distance), 0.0)
        interval_probability = -torch.expm1(-mass)
        # logit(1-exp(-x)) = x + log(1-exp(-x)), stable for large x.
        positive_mass = mass.clamp_min(1e-7)
        logits = positive_mass + torch.log(-torch.expm1(-positive_mass))
        return {
            "boundary_logits": torch.where(edge_valid, logits, 0.0),
            "boundary_probabilities": torch.where(edge_valid, interval_probability, 0.0),
            "boundary_probs": torch.where(edge_valid, interval_probability, 0.0),
            "boundary_rates": rates,
            "boundary_dt": dt,
            "boundary_valid": edge_valid,
            "boundary_left_index": torch.where(edge_valid, previous, -1),
            "survival": survival,
        }

    @staticmethod
    def _support(
        probabilities: torch.Tensor,
        relative_times: torch.Tensor,
        valid: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Nine optional features; each one-second bin has total pair weight one.

        This aggregates all observations within a bin before cross-bin products.
        It never drops candidates from retrieval or treats products as calibrated
        probabilities. The center itself is excluded as a complementary witness.
        """
        slots = torch.arange(valid.shape[1], device=valid.device)[None, :]
        witnesses = valid & (slots != centers[:, None])
        bins = torch.floor(relative_times).long()
        same_bin = bins[:, :, None] == bins[:, None, :]
        present_pairs = witnesses[:, :, None] & witnesses[:, None, :]
        counts = (same_bin & present_pairs).sum(-1).clamp_min(1)
        bin_weights = witnesses.to(probabilities.dtype) / counts
        different_pairs = present_pairs & ~same_bin
        pair_weights = bin_weights[:, :, None] * bin_weights[:, None, :]
        pair_weights = torch.where(different_pairs, pair_weights, 0.0)
        denominator = pair_weights.sum((1, 2)).clamp_min(1.0)
        products = probabilities[:, :, None, :] * probabilities[:, None, :, :]
        pair_support = (products * pair_weights[..., None]).sum((1, 2)) / denominator[:, None]
        single_support = torch.where(witnesses[..., None], probabilities, 0.0).amax(dim=1)
        separation = (relative_times[:, :, None] - relative_times[:, None, :]).abs()
        pair_separation = (separation * pair_weights).sum((1, 2)) / denominator
        maximum = max(valid.shape[1] - 1, 1)
        coverage = torch.stack(
            (
                witnesses.sum(-1) / maximum,
                bin_weights.sum(-1) / maximum,
                pair_separation / 4.0,
            ),
            dim=-1,
        )
        return torch.cat((single_support, pair_support, coverage), dim=-1)

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
        batch, observations = valid.shape
        row = torch.arange(batch, device=features.device)
        relative = times - times.gather(1, centers[:, None])
        relative = torch.where(valid, relative, 0.0)
        scaled_time = relative / self.time_decay_seconds
        side = torch.cat((base, scaled_time[..., None], scaled_time.abs()[..., None]), dim=-1)
        projected = self.input_projection(features) + self.side_projection(side)
        projected = torch.where(valid[..., None], projected, 0.0)
        result = self._boundaries(projected, base, times, valid, centers)
        if self.variant == "temporal_conv":
            contextual = projected
            for block in self.temporal:
                contextual = block(contextual, valid)
            scores = self.temporal_score(contextual).squeeze(-1)
        else:
            contextual = self.temporal(projected, src_key_padding_mask=~valid)
            contextual = torch.where(valid[..., None], contextual, 0.0)
            center_values = contextual[row, centers]
            center_expanded = center_values[:, None, :].expand(-1, observations, -1)
            center_base = base[row, centers][:, None, :].expand(-1, observations, -1)
            pair_features = torch.cat(
                (
                    center_expanded,
                    contextual,
                    (contextual - center_expanded).abs(),
                    center_base,
                    base,
                    scaled_time.abs()[..., None],
                ),
                dim=-1,
            )
            scores = torch.einsum("bd,bnd->bn", self.query(center_values), self.key(contextual))
            scores = scores / math.sqrt(self.width) + self.usefulness(pair_features).squeeze(-1)
        scores = scores - scaled_time.abs()
        if self.variant in ("survival_memory", "corroborated_memory"):
            scores = scores + torch.log(result["survival"] + self.survival_epsilon)
        support = torch.zeros((batch, 9), dtype=features.dtype, device=features.device)
        if self.variant == "corroborated_memory":
            support = self._support(base, relative, valid, centers)
            support_inputs = torch.cat(
                (contextual, base, support[:, None, :].expand(-1, observations, -1)), dim=-1
            )
            scores = scores + self.support_score(support_inputs).squeeze(-1)
        attention = scores.masked_fill(~valid, -torch.inf).softmax(dim=1)
        memory = torch.einsum("bn,bnc->bc", attention, base)
        retrieved = torch.einsum("bn,bnd->bd", attention, contextual)
        center_base = base[row, centers]
        gate_inputs = torch.cat(
            (
                contextual[row, centers],
                retrieved,
                center_base,
                memory,
                valid.sum(-1, keepdim=True) / observations,
                (attention * scaled_time.abs()).sum(-1, keepdim=True),
            ),
            dim=-1,
        )
        gate_logits = self.fusion_gate(gate_inputs).squeeze(-1)
        if self.variant == "corroborated_memory":
            gate_logits = gate_logits + self.support_gate(support).squeeze(-1)
        gate = gate_logits.sigmoid()
        has_neighbor = valid.sum(dim=1) > 1
        gate = torch.where(has_neighbor, gate, 0.0)
        # Learn in the module dtype, but combine the caller's original probability
        # vectors in their dtype. A zero gate must preserve the center even when
        # neighbors exist; rounding base probabilities before mixing would not.
        original_center = probabilities[row, centers]
        original_base = torch.where(valid[..., None], probabilities, 0.0)
        mixture_attention = attention.to(probabilities.dtype)
        mixture_attention = mixture_attention / mixture_attention.sum(-1, keepdim=True)
        original_memory = torch.einsum("bn,bnc->bc", mixture_attention, original_base)
        mixture_gate = gate.to(probabilities.dtype)[:, None]
        prediction = (1.0 - mixture_gate) * original_center + mixture_gate * original_memory
        prediction = torch.where(has_neighbor[:, None], prediction, original_center)
        result.update(
            {
                "probabilities": prediction,
                "memory_probabilities": original_memory,
                "gate": gate,
                "attention": mixture_attention,
                "support_features": support,
            }
        )
        return result
