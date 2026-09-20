"""Pre-pooling fine-detail moments with a shared coarse first-order reference.

One model accepts either genuine 12x12 tokens or retained 3x3 tokens repeated
within each 4x4 macroregion. A linear channel projection preserves their matched
means. Nonlinear spatial dispersion and temporal residual products are formed
*before* pooling, so zero-mean fine detail can survive that otherwise lossy step.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

DETAIL_MODES = ("both", "spatial_only", "motion_only")
MATCH_MODES = ("fixed", "local_soft")


def group_fine_cells(values: torch.Tensor) -> torch.Tensor:
    """[B,T,144,D] -> [B,T,9,16,D], preserving row-major macroregion layout."""
    if values.ndim != 4 or values.shape[2] != 144:
        raise ValueError("Expected [batch,time,144,channels]")
    batch, times, _, channels = values.shape
    return (
        values.reshape(batch, times, 3, 4, 3, 4, channels)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(batch, times, 9, 16, channels)
    )


def ungroup_fine_cells(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 5 or values.shape[2:4] != (9, 16):
        raise ValueError("Expected [batch,time,9,16,channels]")
    batch, times, _, _, channels = values.shape
    return (
        values.reshape(batch, times, 3, 3, 4, 4, channels)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(batch, times, 144, channels)
    )


def expand_coarse_tokens(coarse: torch.Tensor) -> torch.Tensor:
    """Repeat every 3x3 cell into its corresponding 4x4 block, without interpolation."""
    if coarse.ndim != 4 or coarse.shape[2] != 9:
        raise ValueError("Expected retained coarse tokens [batch,time,9,channels]")
    batch, times, _, channels = coarse.shape
    return (
        coarse.reshape(batch, times, 3, 3, channels)
        .repeat_interleave(4, dim=2)
        .repeat_interleave(4, dim=3)
        .reshape(batch, times, 144, channels)
    )


def centered_second_moment(
    values: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-region centered fine-cell values and their weighted variance."""
    if values.shape[:-1] != weights.shape or values.shape[-2] != 16:
        raise ValueError("Fine values and weights must share a sixteen-cell axis")
    count = weights.sum(-1, keepdim=True).clamp_min(1e-6)
    mean = (values * weights[..., None]).sum(-2) / count
    centered = torch.where((weights > 0)[..., None], values - mean[..., None, :], 0.0)
    moment = (centered.square() * weights[..., None]).sum(-2) / count
    return centered, moment


class FineLocalMotion(nn.Module):
    """Identical parameterization for the two resolution conditions.

    ``coarse_reference`` should be the same original retained 3x3 tensor in both
    conditions. It anchors first-order appearance despite float16 re-averaging
    differences. ``detail_mode`` masks moment blocks for later ablations without
    changing model shape. Sparse matching remains inside each macroregion.
    """

    def __init__(
        self,
        input_dim: int = 768,
        *,
        rank: int = 32,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        quality_dim: int = 6,
        dropout: float = 0.1,
        match_mode: str = "fixed",
        detail_mode: str = "both",
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if min(input_dim, rank, width, layers, heads) < 1 or width % heads:
            raise ValueError("Dimensions must be positive and width divisible by heads")
        if (
            quality_dim < 0
            or not 0 <= dropout < 1
            or match_mode not in MATCH_MODES
            or detail_mode not in DETAIL_MODES
        ):
            raise ValueError("Invalid fine-detail configuration")
        self.input_dim, self.rank, self.width, self.quality_dim = (
            input_dim,
            rank,
            width,
            quality_dim,
        )
        self.match_mode, self.detail_mode = match_mode, detail_mode
        # Deliberately linear: any density effect in the shared-reference setup
        # must enter through explicit pre-pooling detail moments below.
        self.projection = nn.Linear(input_dim, rank, bias=False)
        self.first_order = nn.Sequential(
            nn.Linear(18 * rank + quality_dim, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.macro_projection = nn.Linear(rank, width)
        self.region_embedding = nn.Parameter(torch.empty(9, width))
        nn.init.normal_(self.region_embedding, std=0.02)
        self.time_embedding = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.time_gain = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.detail_projection = nn.Sequential(nn.Linear(4 * 9 * rank, width), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 3)
        )
        if match_mode == "local_soft":
            self.match_projection = nn.Linear(rank, rank, bias=False)
            self.unmatched_logit = nn.Parameter(torch.tensor(-1.0))
            coordinates = torch.stack(
                torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), -1
            ).reshape(16, 2)
            offsets = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
            targets = coordinates[:, None, :] + offsets[None, :, :]
            allowed = ((targets >= 0) & (targets < 4)).all(-1)
            indices = (targets[..., 0] * 4 + targets[..., 1]).clamp(0, 15)
            self.register_buffer("neighbor_indices", indices)
            self.register_buffer("neighbor_valid", allowed)
            self.register_buffer("neighbor_distance", offsets.square().sum(-1).float())
        if self.trainable_parameters > parameter_limit:
            raise ValueError("Fine-detail head exceeds declared parameter limit")

    @property
    def trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _transport(self, source, target, source_valid, target_valid):
        # Only five neighbors per fine cell; never form a 144x144 attention map.
        neighbors = target[:, :, :, self.neighbor_indices, :]
        available = target_valid[:, :, :, self.neighbor_indices] & self.neighbor_valid
        query = F.normalize(self.match_projection(source), dim=-1, eps=1e-6)
        key = F.normalize(self.match_projection(neighbors), dim=-1, eps=1e-6)
        scores = (query[..., None, :] * key).sum(-1) * math.sqrt(self.rank)
        scores = scores - self.neighbor_distance
        scores = scores.masked_fill(~available, -torch.inf)
        dustbin = self.unmatched_logit.expand(*scores.shape[:-1], 1)
        assignment = torch.cat((scores, dustbin), dim=-1).softmax(-1)
        assignment = torch.where(source_valid[..., None], assignment, 0.0)
        matched = assignment[..., :5]
        support = matched.sum(-1)
        conditional = matched / support.clamp_min(1e-6)[..., None]
        transported = (conditional[..., None] * neighbors).sum(-2)
        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(-1)
        return transported, support, entropy

    def _moments(self, fine, valid, times, center_index):
        weights = valid.to(fine.dtype)
        _, spatial = centered_second_moment(fine[:, center_index], weights[:, center_index])
        if self.match_mode == "fixed":
            change = fine[:, 1:] - fine[:, :-1]
            support = (valid[:, 1:] & valid[:, :-1]).to(fine.dtype)
            entropy = torch.zeros_like(support)
        else:
            transported, support, entropy = self._transport(
                fine[:, :-1], fine[:, 1:], valid[:, :-1], valid[:, 1:]
            )
            static, _, _ = self._transport(fine[:, :-1], fine[:, :-1], valid[:, :-1], valid[:, :-1])
            change = torch.where((support > 1e-6)[..., None], transported - static, 0.0)
        centered_change, _ = centered_second_moment(change, support)
        dt = times[:, 1:] - times[:, :-1]
        midpoint = (times[:, 1:] + times[:, :-1]) / 2
        moments = [spatial]
        for lag in (0, 1, 2):
            length = change.shape[1] - lag
            physical = torch.stack(
                (dt[:, :length], dt[:, lag:], midpoint[:, lag:] - midpoint[:, :length]), dim=-1
            )
            gain = F.softplus(self.time_gain(physical).squeeze(-1)) + 1e-6
            joint = support[:, :length] * support[:, lag:]
            product = centered_change[:, :length] * centered_change[:, lag:]
            numerator = (product * joint[..., None] * gain[:, :, None, None, None]).sum((1, 3))
            denominator = joint.sum((1, 3)).clamp_min(1e-6)
            moments.append(numerator / denominator[..., None])
        return torch.stack(moments, dim=1), entropy

    def forward(
        self,
        fine_tokens: torch.Tensor,
        times: torch.Tensor,
        quality: torch.Tensor | None = None,
        coarse_reference: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        center_index: int = 4,
    ) -> dict[str, torch.Tensor]:
        if fine_tokens.ndim != 4 or fine_tokens.shape[2:] != (144, self.input_dim):
            raise ValueError("fine_tokens must be [batch,time,144,input_dim]")
        batch, temporal, _, _ = fine_tokens.shape
        if batch < 1 or temporal < 4 or not 0 <= center_index < temporal:
            raise ValueError("Insufficient times or invalid center")
        if (
            times.shape != (batch, temporal)
            or not torch.isfinite(times).all()
            or torch.any(times[:, 1:] <= times[:, :-1])
        ):
            raise ValueError("Times must be finite, increasing [batch,time] seconds")
        if valid is None:
            valid = torch.ones(fine_tokens.shape[:3], dtype=torch.bool, device=fine_tokens.device)
        if valid.shape != fine_tokens.shape[:3] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch,time,144]")
        if not torch.isfinite(fine_tokens[valid]).all() or not valid[:, center_index].any(-1).all():
            raise ValueError("Valid fine tokens must be finite and the center observed")
        dtype = self.projection.weight.dtype
        safe = torch.where(valid[..., None], fine_tokens, 0.0).to(dtype)
        fine = group_fine_cells(self.projection(safe))
        grouped_valid = group_fine_cells(valid[..., None]).squeeze(-1)
        if coarse_reference is None:
            count = grouped_valid.sum(-1).clamp_min(1)[..., None]
            macro = (fine * grouped_valid[..., None]).sum(-2) / count
        else:
            if (
                coarse_reference.shape != (batch, temporal, 9, self.input_dim)
                or not torch.isfinite(coarse_reference).all()
            ):
                raise ValueError("Shared coarse reference must be finite [batch,time,9,input_dim]")
            macro = self.projection(coarse_reference.to(dtype))
        macro_valid = grouped_valid.any(-1)
        macro = torch.where(macro_valid[..., None], macro, 0.0)
        if quality is None:
            quality = torch.zeros(batch, self.quality_dim, dtype=dtype, device=fine_tokens.device)
        if quality.shape != (batch, self.quality_dim) or not torch.isfinite(quality).all():
            raise ValueError("Quality must be finite [batch,quality_dim]")
        mean_macro = macro.sum(1) / macro_valid.sum(1).clamp_min(1)[..., None]
        first = self.first_order(
            torch.cat(
                (mean_macro.flatten(1), macro[:, center_index].flatten(1), quality.to(dtype)),
                dim=-1,
            )
        )
        times = times.to(dtype)
        relative = times - times[:, center_index, None]
        position = torch.stack((relative, relative.abs(), relative.square()), dim=-1)
        macro_sequence = (
            self.macro_projection(macro)
            + self.region_embedding[None, None]
            + self.time_embedding(position)[:, :, None]
        )
        context = self.temporal(
            macro_sequence.flatten(1, 2), src_key_padding_mask=~macro_valid.flatten(1)
        ).reshape(batch, temporal, 9, self.width)
        context = (
            torch.where(macro_valid[..., None], context, 0.0).sum((1, 2))
            / macro_valid.sum((1, 2)).clamp_min(1)[:, None]
        )
        moments, entropy = self._moments(fine, grouped_valid, times, center_index)
        moment_mask = torch.ones(4, dtype=dtype, device=fine_tokens.device)
        if self.detail_mode == "spatial_only":
            moment_mask[1:] = 0
        elif self.detail_mode == "motion_only":
            moment_mask[0] = 0
        used_moments = moments * moment_mask[None, :, None, None]
        detail = self.detail_projection(used_moments.flatten(1))
        logits = self.classifier(torch.cat((first, context, detail), dim=-1))
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "macro_tokens": macro,
            "detail_moments": moments,
            "used_detail_moments": used_moments,
            "correspondence_entropy": entropy,
        }
