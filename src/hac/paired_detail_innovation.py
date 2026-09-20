"""ARFTR-preserving shared-weight fine/coarse detail innovation.

The paired arms pass a genuine dense observation and its repeated-coarse
counterfactual through the *same* nonlinear encoder.  Their representation
difference is therefore upstream of the task head and cannot contain a
separate-reader initialization or calibration contrast.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hac.bounded_factor_correction import factor_logits, probability_factor_scores
from hac.fine_local_motion import centered_second_moment, group_fine_cells


ARMS = ("coarse_residual", "paired_spatial", "paired_spatial_temporal")


def _check_probabilities(values: torch.Tensor, rows: int) -> None:
    if (
        values.shape != (rows, 3)
        or not values.is_floating_point()
        or not torch.isfinite(values).all()
        or torch.any(values < 0)
        or not torch.allclose(
            values.sum(1), torch.ones(rows, device=values.device), atol=1e-6, rtol=0
        )
    ):
        raise ValueError("anchor probabilities must be finite [batch,3] simplexes")


def anchor_context(probabilities: torch.Tensor) -> torch.Tensor:
    values = probabilities.clamp_min(1e-12)
    entropy = -(values * values.log()).sum(1, keepdim=True)
    return torch.cat((values, values.log(), entropy), dim=1)


def apply_factor_actions(
    anchor: torch.Tensor, correction: torch.Tensor
) -> torch.Tensor:
    """Return retain/posture/motion/both distributions as [B,4,3]."""

    _check_probabilities(anchor, len(anchor))
    if correction.shape != (len(anchor), 2) or not torch.isfinite(correction).all():
        raise ValueError("correction must be finite [batch,2]")
    posture, motion = probability_factor_scores(anchor)
    zero = torch.zeros_like(correction[:, 0])
    masks = ((correction[:, 0], zero), (zero, correction[:, 1]),
             (correction[:, 0], correction[:, 1]))
    candidates = []
    for dp, dm in masks:
        decoded = factor_logits(posture + dp, motion + dm).softmax(1)
        active = (dp != 0) | (dm != 0)
        candidates.append(torch.where(active[:, None], decoded, anchor))
    # Do not round-trip the explicit retain action through logit algebra.
    return torch.stack([anchor, *candidates], dim=1)


class PairedDetailInnovation(nn.Module):
    """Shared dense/coarse encoder with a zero-initialized factor residual."""

    def __init__(
        self,
        input_dim: int = 768,
        *,
        rank: int = 32,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        quality_dim: int = 6,
        correction_bound: float = 0.5,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if min(input_dim, rank, width, layers, heads) < 1 or width % heads:
            raise ValueError("invalid PDI dimensions")
        if quality_dim < 0 or correction_bound <= 0:
            raise ValueError("invalid PDI quality/bound")
        self.input_dim, self.rank, self.width, self.quality_dim = (
            input_dim, rank, width, quality_dim
        )
        self.projection = nn.Linear(input_dim, rank, bias=False)
        self.first_order = nn.Sequential(
            nn.Linear(18 * rank + quality_dim, width), nn.GELU()
        )
        self.macro_projection = nn.Linear(rank, width)
        self.region_embedding = nn.Parameter(torch.empty(9, width))
        nn.init.normal_(self.region_embedding, std=0.02)
        self.time_embedding = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.Linear(width, width)
        )
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, 0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.detail_projection = nn.Sequential(
            nn.Linear(4 * 9 * rank, width), nn.GELU()
        )
        self.merge = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        # Context must enter before this nonlinearity; appending it after a
        # paired subtraction would cancel it without conditioning innovation.
        self.condition = nn.Sequential(nn.Linear(width + 7, width), nn.GELU())
        # No bias: paired null innovation must remain an exact zero after any
        # optimizer update, rather than learning a global offset.
        self.correction_head = nn.Linear(width, 2, bias=False)
        nn.init.zeros_(self.correction_head.weight)
        self.register_buffer("correction_limit", torch.tensor(float(correction_bound)))
        if self.trainable_parameters > parameter_limit:
            raise ValueError("PDI exceeds declared parameter limit")

    @property
    def trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _validate(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
        times: torch.Tensor,
        quality: torch.Tensor,
        valid: torch.Tensor,
        center_index: int,
    ) -> None:
        batch, temporal = fine.shape[:2]
        if fine.shape != (batch, temporal, 144, self.input_dim):
            raise ValueError("fine must be [batch,time,144,input_dim]")
        if coarse.shape != (batch, temporal, 9, self.input_dim):
            raise ValueError("coarse must be [batch,time,9,input_dim]")
        if times.shape != (batch, temporal) or torch.any(times[:, 1:] <= times[:, :-1]):
            raise ValueError("times must be increasing [batch,time]")
        if quality.shape != (batch, self.quality_dim):
            raise ValueError("quality shape changed")
        if valid.shape != fine.shape[:3] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch,time,144]")
        if not 0 <= center_index < temporal or temporal < 4:
            raise ValueError("invalid center/time count")
        if not valid[:, center_index].any(1).all():
            raise ValueError("center observation missing")
        if not torch.isfinite(coarse).all() or not torch.isfinite(times).all() or not torch.isfinite(quality).all():
            raise ValueError("nonfinite PDI context")
        if not torch.isfinite(fine[valid]).all():
            raise ValueError("nonfinite valid dense token")

    def _detail_moments(
        self, fine: torch.Tensor, valid: torch.Tensor, center_index: int
    ) -> torch.Tensor:
        weights = valid.to(fine.dtype)
        centered, spatial = centered_second_moment(
            fine[:, center_index], weights[:, center_index]
        )
        del centered
        change = fine[:, 1:] - fine[:, :-1]
        support = (valid[:, 1:] & valid[:, :-1]).to(fine.dtype)
        centered_change, _ = centered_second_moment(change, support)
        moments = [spatial]
        # Fixed, interpretable temporal lags. The physical clock is represented
        # in Q and the temporal transformer; no learned pair-local transport.
        for lag in (0, 1, 2):
            length = change.shape[1] - lag
            joint = support[:, :length] * support[:, lag:]
            product = centered_change[:, :length] * centered_change[:, lag:]
            numerator = (product * joint[..., None]).sum((1, 3))
            denominator = joint.sum((1, 3)).clamp_min(1e-6)[..., None]
            moments.append(numerator / denominator)
        return torch.stack(moments, dim=1)

    def _coarse_scaffold(
        self, macro: torch.Tensor, times: torch.Tensor, center_index: int
    ) -> torch.Tensor:
        center = macro[:, center_index]
        spatial = (center - center.mean(1, keepdim=True)).square()
        temporal = (macro - macro.mean(1, keepdim=True)).square().mean(1)
        dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-6)
        velocity = (macro[:, 1:] - macro[:, :-1]) / dt[:, :, None, None]
        physical = velocity.square().mean(1)
        centered = velocity - velocity.mean(1, keepdim=True)
        lag = (centered[:, 1:] * centered[:, :-1]).mean(1)
        return torch.stack((spatial, temporal, physical, lag), dim=1)

    def _encode(
        self,
        macro: torch.Tensor,
        statistics: torch.Tensor,
        times: torch.Tensor,
        quality: torch.Tensor,
        context: torch.Tensor,
        center_index: int,
    ) -> torch.Tensor:
        mean_macro = macro.mean(1)
        first = self.first_order(torch.cat(
            (mean_macro.flatten(1), macro[:, center_index].flatten(1), quality), dim=1
        ))
        relative = times - times[:, center_index, None]
        position = torch.stack((relative, relative.abs(), relative.square()), dim=-1)
        sequence = (
            self.macro_projection(macro)
            + self.region_embedding[None, None]
            + self.time_embedding(position)[:, :, None]
        )
        temporal = self.temporal(sequence.flatten(1, 2)).mean(1)
        detail = self.detail_projection(statistics.flatten(1))
        merged = self.merge(torch.cat((first, temporal, detail), dim=1))
        return self.condition(torch.cat((merged, context), dim=1))

    def forward(
        self,
        fine_tokens: torch.Tensor,
        coarse_reference: torch.Tensor,
        times: torch.Tensor,
        quality: torch.Tensor,
        anchor_probabilities: torch.Tensor,
        *,
        arm: str,
        valid: torch.Tensor | None = None,
        center_index: int = 4,
    ) -> dict[str, torch.Tensor]:
        if arm not in ARMS:
            raise ValueError(f"unknown PDI arm: {arm}")
        if valid is None:
            valid = torch.ones(
                fine_tokens.shape[:3], dtype=torch.bool, device=fine_tokens.device
            )
        self._validate(fine_tokens, coarse_reference, times, quality, valid, center_index)
        _check_probabilities(anchor_probabilities, len(fine_tokens))
        dtype = self.projection.weight.dtype
        fine = self.projection(torch.where(valid[..., None], fine_tokens, 0).to(dtype))
        fine = group_fine_cells(fine)
        grouped_valid = group_fine_cells(valid[..., None]).squeeze(-1)
        macro = self.projection(coarse_reference.to(dtype))
        times, quality = times.to(dtype), quality.to(dtype)
        q = self._coarse_scaffold(macro, times, center_index)
        d = self._detail_moments(fine, grouped_valid, center_index)
        # Repeated cells yield D=0 analytically; snap numerical residue to zero
        # only at the tested FP tolerance, never on genuine observations.
        if arm == "paired_spatial":
            d = torch.cat((d[:, :1], torch.zeros_like(d[:, 1:])), dim=1)
        context = anchor_context(anchor_probabilities.to(dtype))
        base = self._encode(macro, q, times, quality, context, center_index)
        representation = base if arm == "coarse_residual" else (
            self._encode(macro, q + d, times, quality, context, center_index) - base
        )
        correction = self.correction_limit * torch.tanh(self.correction_head(representation))
        actions = apply_factor_actions(anchor_probabilities.to(dtype), correction)
        posture, motion = probability_factor_scores(anchor_probabilities.to(dtype))
        training_probabilities = factor_logits(
            posture + correction[:, 0], motion + correction[:, 1]
        ).softmax(1)
        q_norms = torch.linalg.vector_norm(q, dim=(2, 3))
        return {
            "probabilities": actions[:, 3],
            "training_probabilities": training_probabilities,
            "actions": actions,
            "correction": correction,
            "representation": representation,
            "coarse_scaffold": q,
            "coarse_scaffold_norms": q_norms,
            "detail_moments": d,
        }


def pdi_loss(
    output: dict[str, torch.Tensor],
    anchor: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor,
    *,
    anchor_kl_weight: float = 0.1,
    correction_l1_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probabilities = output.get("training_probabilities", output["probabilities"]).clamp_min(1e-12)
    ce = F.nll_loss(probabilities.log(), labels, weight=class_weight)
    kl = F.kl_div(probabilities.log(), anchor, reduction="batchmean")
    l1 = output["correction"].abs().mean()
    total = ce + anchor_kl_weight * kl + correction_l1_weight * l1
    return total, {"cross_entropy": ce, "anchor_kl": kl, "correction_l1": l1}
