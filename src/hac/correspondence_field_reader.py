"""Label-independent C0/C1/C2 conditional-motion readers.

These modules consume already measured fields. They neither construct fields nor
inspect annotations, ARFTR outputs, folds, or intervention targets. The caller
must retain its anchor when ``observation_available`` is false. Otherwise the
output is only P(walking/running | upright), not an intervention recommendation.
"""

from __future__ import annotations

import math

import torch
from torch import nn

ARMS = ("c0", "c1", "c2")
WIDTH = 64
PAIR_COUNT = 15
MAX_POINTS = 32
POINT_DIM = 10
PAIR_FEATURE_DIM = 5
CONTEXT_STREAM_DIM = 768
CONTEXT_DIM = 3 * CONTEXT_STREAM_DIM
SUMMARY_DIM = 31
# P8 ccac_full: quality16 + compensated translation9 + within-actor6.
PAIR_FEATURE_ORDER = (
    "elapsed_seconds",
    "point_count",
    "camera_audit_median_pixels",
    "camera_audit_p90_pixels",
    "camera_usable",
)
POINT_FEATURE_ORDER = (
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "local_residual_x",
    "local_residual_y",
    "translation_x",
    "translation_y",
    "forward_backward_error_pixels",
    "forward_backward_error_actor_heights",
)


def _linear_count(input_dim: int, output_dim: int) -> int:
    return (input_dim + 1) * output_dim


def c2_parameter_target() -> int:
    """Count the declared C2 architecture without constructing unused modules."""
    context = 3 * 2 * CONTEXT_STREAM_DIM + _linear_count(CONTEXT_DIM, WIDTH)
    point = _linear_count(POINT_DIM, WIDTH) + _linear_count(WIDTH, WIDTH)
    pair = _linear_count(2 * WIDTH + PAIR_FEATURE_DIM, WIDTH) + _linear_count(WIDTH, WIDTH)
    # Q/K/V and output projections; two FF layers; two affine LayerNorms.
    attention_block = (
        4 * WIDTH * WIDTH
        + 4 * WIDTH
        + _linear_count(WIDTH, 2 * WIDTH)
        + _linear_count(2 * WIDTH, WIDTH)
        + 4 * WIDTH
    )
    time_and_query = _linear_count(9, WIDTH) + WIDTH
    fusion_and_head = _linear_count(2 * WIDTH, WIDTH) + _linear_count(WIDTH, 1)
    return context + point + pair + 2 * attention_block + time_and_query + fusion_and_head


def _masked_mean_max(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pool the penultimate dimension, mapping an empty set to exact zeros."""
    masked = torch.where(valid[..., None], values, torch.zeros_like(values))
    count = valid.sum(-1, keepdim=True).to(values.dtype)
    mean = masked.sum(-2) / count.clamp_min(1)
    maximum = values.masked_fill(~valid[..., None], -torch.inf).amax(-2)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class _ResidualMLP(nn.Module):
    """Task-active capacity matching on the representation sent to the head."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(WIDTH),
            nn.Linear(WIDTH, 2 * WIDTH),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * WIDTH, WIDTH),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(features)


class CorrespondenceFieldReader(nn.Module):
    """Fixed-width matched-capacity readers for the preregistered screen.

    ``context`` concatenates short V-JEPA, long V-JEPA and center DINO streams.
    ``points`` contains ten channels in ``POINT_FEATURE_ORDER``. Padding uses a
    separate boolean ``point_valid`` mask; no annotation-validity bit belongs in
    that mask. Pair features follow ``PAIR_FEATURE_ORDER``. Only C2 consumes
    actual midpoint seconds relative to the target center; C1 has no slot or
    timestamp input to its learned computation. C0 consumes the original 31 P8
    summary channels, and does not consume raw point/pair values.

    All arms require all 15 pair slots, with between 1 and 32 padded point slots.
    Invalid observations are masked before any learned projection. Nonfinite
    values on available observations fail closed. An empty field gives a finite
    context-only output, accompanied by ``observation_available=False``.
    """

    def __init__(self, arm: str, *, dropout: float = 0.1) -> None:
        super().__init__()
        if arm not in ARMS or not 0 <= dropout < 1:
            raise ValueError("Invalid correspondence reader arm or dropout")
        self.arm = arm
        self.width = WIDTH
        self.context_norms = nn.ModuleList([nn.LayerNorm(CONTEXT_STREAM_DIM) for _ in range(3)])
        self.context_projection = nn.Sequential(nn.Linear(CONTEXT_DIM, WIDTH), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(WIDTH, 1)
        if arm == "c0":
            self.summary_encoder = nn.Sequential(
                nn.Linear(SUMMARY_DIM, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH), nn.GELU()
            )
        else:
            self.point_encoder = nn.Sequential(
                nn.Linear(POINT_DIM, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH), nn.GELU()
            )
            self.pair_encoder = nn.Sequential(
                nn.Linear(2 * WIDTH + PAIR_FEATURE_DIM, WIDTH),
                nn.GELU(),
                nn.Linear(WIDTH, WIDTH),
                nn.GELU(),
            )
            if arm == "c1":
                self.unordered_projection = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU())
            else:
                self.register_buffer("time_frequencies", math.pi * torch.tensor([0.5, 1, 2, 4]))
                self.time_projection = nn.Linear(9, WIDTH)
                self.center_query = nn.Parameter(torch.zeros(WIDTH))
                self.attention_blocks = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=WIDTH,
                            nhead=4,
                            dim_feedforward=2 * WIDTH,
                            dropout=dropout,
                            activation="gelu",
                            batch_first=True,
                            norm_first=True,
                        )
                        for _ in range(2)
                    ]
                )
        target = c2_parameter_target()
        initial_count = self.trainable_parameters
        residual_count = (
            2 * WIDTH + _linear_count(WIDTH, 2 * WIDTH) + _linear_count(2 * WIDTH, WIDTH)
        )
        depth = (
            min(
                range(16),
                key=lambda count: (abs(initial_count + count * residual_count - target), count),
            )
            if arm != "c2"
            else 0
        )
        self.capacity_blocks = nn.Sequential(*[_ResidualMLP(dropout) for _ in range(depth)])
        if arm == "c2" and self.trainable_parameters != target:
            raise RuntimeError("C2 parameter accounting differs from the declared architecture")
        if abs(self.trainable_parameters / target - 1) > 0.1:
            raise RuntimeError("Task-active reader capacity is not within 10% of C2")

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def capacity_report(self) -> dict[str, int | float | str]:
        return {
            "arm": self.arm,
            "width": WIDTH,
            "trainable_parameters": self.trainable_parameters,
            "c2_reference_parameters": c2_parameter_target(),
            "relative_difference_from_c2": self.trainable_parameters / c2_parameter_target() - 1,
            "task_active_residual_blocks": len(self.capacity_blocks),
        }

    @staticmethod
    def _floating(
        values: torch.Tensor | None,
        *,
        shape: tuple[int, ...],
        context: torch.Tensor,
        name: str,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            values is None
            or values.shape != shape
            or not values.is_floating_point()
            or values.device != context.device
            or values.dtype != context.dtype
        ):
            raise ValueError(f"Malformed {name}: wrong shape, floating dtype, or device")
        if valid is not None:
            mask = valid
            while mask.ndim < values.ndim:
                mask = mask.unsqueeze(-1)
            values = torch.where(mask, values, torch.zeros_like(values))
        if not torch.isfinite(values).all():
            raise ValueError(f"Malformed {name}: nonfinite available observation")
        return values

    def forward(
        self,
        *,
        context: torch.Tensor,
        point_valid: torch.Tensor,
        pair_valid: torch.Tensor,
        points: torch.Tensor | None = None,
        pair_features: torch.Tensor | None = None,
        midpoint_times: torch.Tensor | None = None,
        summaries: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            context.ndim != 2
            or context.shape[1] != CONTEXT_DIM
            or not context.is_floating_point()
            or not torch.isfinite(context).all()
            or len(context) == 0
        ):
            raise ValueError("Malformed absolute context: expected finite [B,2304] streams")
        batch = len(context)
        if (
            point_valid.ndim != 3
            or point_valid.shape[:2] != (batch, PAIR_COUNT)
            or not 1 <= point_valid.shape[2] <= MAX_POINTS
            or pair_valid.shape != (batch, PAIR_COUNT)
            or point_valid.dtype != torch.bool
            or pair_valid.dtype != torch.bool
            or point_valid.device != context.device
            or pair_valid.device != context.device
        ):
            raise ValueError("Malformed observation masks: expected 15 pairs and up to 32 points")
        effective_points = point_valid & pair_valid[..., None]
        effective_pairs = effective_points.any(-1)
        available = effective_pairs.any(-1)
        context_state = self.context_projection(
            torch.cat(
                [
                    norm(stream)
                    for norm, stream in zip(
                        self.context_norms, context.split(CONTEXT_STREAM_DIM, -1), strict=True
                    )
                ],
                dim=-1,
            )
        )
        if self.arm == "c0":
            summary = self._floating(
                summaries,
                shape=(batch, SUMMARY_DIM),
                context=context,
                name="P8 summaries",
                valid=available,
            )
            motion = self.summary_encoder(summary)
        else:
            points = self._floating(
                points,
                shape=(*point_valid.shape, POINT_DIM),
                context=context,
                name="point fields",
                valid=effective_points,
            )
            pairs = self._floating(
                pair_features,
                shape=(batch, PAIR_COUNT, PAIR_FEATURE_DIM),
                context=context,
                name="pair features",
                valid=effective_pairs,
            )
            if torch.any(pairs[..., 0][effective_pairs] <= 0):
                raise ValueError("Available pairs must have positive elapsed seconds")
            point_states = self.point_encoder(points)
            pooled = _masked_mean_max(point_states, effective_points)
            pair_states = self.pair_encoder(torch.cat((pooled, pairs), dim=-1))
            pair_states = torch.where(effective_pairs[..., None], pair_states, 0.0)
            if self.arm == "c1":
                motion = self.unordered_projection(_masked_mean_max(pair_states, effective_pairs))
            else:
                times = self._floating(
                    midpoint_times,
                    shape=(batch, PAIR_COUNT),
                    context=context,
                    name="midpoint times",
                    valid=effective_pairs,
                )
                angles = times[..., None] * self.time_frequencies
                time_input = torch.cat((times[..., None], angles.sin(), angles.cos()), dim=-1)
                timed_pairs = pair_states + self.time_projection(time_input)
                timed_pairs = torch.where(effective_pairs[..., None], timed_pairs, 0.0)
                center = context_state + self.center_query
                tokens = torch.cat((center[:, None], timed_pairs), dim=1)
                # The center is always a valid key, even when every pair is absent.
                padding = torch.cat(
                    (
                        torch.zeros((batch, 1), dtype=torch.bool, device=context.device),
                        ~effective_pairs,
                    ),
                    dim=1,
                )
                for block in self.attention_blocks:
                    tokens = block(tokens, src_key_padding_mask=padding)
                motion = tokens[:, 0]
        motion = self.capacity_blocks(motion)
        fused = self.fusion(torch.cat((context_state, motion), dim=-1))
        logits = self.classifier(fused).squeeze(-1)
        return {
            "motion_logits": logits,
            "motion_probability": logits.sigmoid(),
            "observation_available": available,
        }

    def conditional_probabilities(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Return [standing, walking/running] probabilities conditional on upright."""
        probability = self(**inputs)["motion_probability"]
        return torch.stack((1 - probability, probability), dim=-1)
