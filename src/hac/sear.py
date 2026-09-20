"""Shared-evidence actor explanations over frozen center-frame patch features.

This module implements only SEAR and its relative-geometry-off control. Slots,
matchability, nuisance transforms, cost weights and temperatures have no class or
template axis. Class hypotheses differ only in their descriptor/layout templates.
Learned slots are visual roles, not anatomical keypoints or calibrated visibility.
No occupancy supervision, memory routing or backbone training is introduced here.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def square_coordinates(patches: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Row-major normalized patch centers; the production DINO grid is 27 by 27."""
    side = math.isqrt(patches)
    if patches < 1 or side * side != patches:
        raise ValueError("Default coordinates require a nonempty square patch grid")
    axis = (torch.arange(side, device=device, dtype=dtype) + 0.5) * (2 / side) - 1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=-1)


def shared_template_energy(
    descriptors: torch.Tensor,
    positions: torch.Tensor,
    masses: torch.Tensor,
    template_descriptors: torch.Tensor,
    template_positions: torch.Tensor,
    *,
    appearance_weight: torch.Tensor | float = 1.0,
    geometry_weight: torch.Tensor | float = 1.0,
) -> dict[str, torch.Tensor]:
    """Score every template against exactly the same observed slot evidence.

    Inputs have shapes [B,K,R], [B,K,2], [B,K], [C,M,K,R], [C,M,K,2].
    Masses are fractions of available patch support, not per-class attention.
    They are deliberately NOT renormalized across visible slots: confidence from
    vanishing coverage must vanish, rather than amplify its last surviving slot.
    Geometry uses ordered, non-self slot pairs and their shared product masses.
    Descriptor costs use 2-2*cosine, equivalent to squared distance for unit
    vectors. A zero/epsilon-normalized prototype therefore has neutral cost two,
    not an artificially cheap distance from reduced prototype norm. Relative
    coordinate squared distances are divided by eight on the normalized [-1,1]
    image coordinate scale.
    Zero-mass descriptors/positions may be NaN and are sanitized before arithmetic.
    """
    if descriptors.ndim != 3 or descriptors.shape[0] < 1:
        raise ValueError("Descriptors must be nonempty [batch,slots,rank]")
    batch, slots, rank = descriptors.shape
    if slots < 1 or rank < 1 or positions.shape != (batch, slots, 2):
        raise ValueError("Observed positions do not match descriptors")
    if masses.shape != (batch, slots):
        raise ValueError("Masses must have only batch and observed-slot axes")
    if (
        template_descriptors.ndim != 4
        or template_descriptors.shape[2:] != (slots, rank)
        or min(template_descriptors.shape[:2]) < 1
        or template_positions.shape != (*template_descriptors.shape[:3], 2)
    ):
        raise ValueError("Templates must share the observed slots and descriptor rank")
    if not torch.isfinite(masses).all() or torch.any(masses < 0):
        raise ValueError("Observed masses must be finite and nonnegative")
    if torch.any(masses.sum(-1) > 1 + 1e-5):
        raise ValueError("Observed masses cannot exceed the image support budget")
    observed = masses > 0
    if (
        not torch.isfinite(descriptors[observed]).all()
        or not torch.isfinite(positions[observed]).all()
    ):
        raise ValueError("Observed slot evidence must be finite")
    if (
        not torch.isfinite(template_descriptors).all()
        or not torch.isfinite(template_positions).all()
    ):
        raise ValueError("All templates must be finite")
    weights = []
    for value in (appearance_weight, geometry_weight):
        value = torch.as_tensor(value, dtype=descriptors.dtype, device=descriptors.device)
        if value.ndim != 0 or not torch.isfinite(value) or value < 0:
            raise ValueError("Cost weights must be shared finite nonnegative scalars")
        weights.append(value)
    safe_descriptor = torch.where(observed[..., None], descriptors, 0.0)
    safe_position = torch.where(observed[..., None], positions, 0.0)
    normalized = F.normalize(safe_descriptor, dim=-1, eps=1e-6)
    prototypes = F.normalize(template_descriptors, dim=-1, eps=1e-6)
    unary_cost = (2 - 2 * torch.einsum("bkr,cmkr->bcmk", normalized, prototypes)).clamp_min(0)
    unary_cost = torch.where(observed[:, None, None], unary_cost, 0.0)
    unary_energy = (unary_cost * masses[:, None, None]).sum(-1)
    joint_mass = masses[:, :, None] * masses[:, None, :]
    nonself = ~torch.eye(slots, device=masses.device, dtype=torch.bool)
    joint_mass = torch.where(nonself[None], joint_mass, 0.0)
    relative = safe_position[:, :, None] - safe_position[:, None, :]
    template_relative = template_positions[:, :, :, None] - template_positions[:, :, None, :]
    pair_cost = (relative[:, None, None] - template_relative[None]).square().sum(-1) / 8
    pair_cost = torch.where(joint_mass[:, None, None] > 0, pair_cost, 0.0)
    pair_energy = (pair_cost * joint_mass[:, None, None]).sum((-1, -2))
    return {
        "template_energies": weights[0] * unary_energy + weights[1] * pair_energy,
        "unary_energies": unary_energy,
        "geometry_energies": pair_energy,
        "unary_masses": masses,
        "pairwise_masses": joint_mass,
        "coverage": masses.sum(-1),
    }


class CommonVideoHead(nn.Module):
    """Reusable video-only logits for all subsequently implemented controls."""

    def __init__(self, input_dim: int = 1536, *, width: int = 128, dropout: float = 0.1):
        super().__init__()
        if min(input_dim, width) < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid common video dimensions or dropout")
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )

    def forward(self, video_features: torch.Tensor) -> torch.Tensor:
        if video_features.ndim != 2 or video_features.shape[1] != self.input_dim:
            raise ValueError("Video features must be [batch,video_dim]")
        if not torch.isfinite(video_features).all():
            raise ValueError("The mandatory video fallback must be finite")
        return self.network(video_features.to(self.network[1].weight.dtype))


class SharedSlotExtractor(nn.Module):
    """Class-independent competitive extraction with an explicit unmatched channel.

    Three recurrent refinements by default, followed by a final readout. Fixed
    learned slot seeds supply consistent parameter identities across images; this
    does not prove that learned roles are stable or noncollapsed on real data.
    No per-image slot-presence lower bound or class-specific mask is imposed.

    ``valid=False`` means nonexistent padding, removed from the support count.
    An existing but missing/occluded physical patch instead uses ``valid=True``
    and ``matchability=0``; it remains in the denominator and weakens coverage.
    In a complete 27x27 image, all729 slots are physical even when masked. These
    weights describe model input availability, not measured physical occlusion.
    """

    def __init__(
        self,
        input_dim: int = 768,
        *,
        rank: int = 32,
        slots: int = 6,
        width: int = 128,
        iterations: int = 3,
    ) -> None:
        super().__init__()
        if min(input_dim, rank, slots, width, iterations) < 1:
            raise ValueError("Slot extractor dimensions and refinements must be positive")
        self.input_dim, self.rank, self.slots, self.iterations = input_dim, rank, slots, iterations
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, rank),
            nn.LayerNorm(rank),
        )
        self.key = nn.Linear(rank, rank, bias=False)
        self.query = nn.Linear(rank, rank, bias=False)
        self.unmatched = nn.Linear(rank, 1)
        self.slot_seeds = nn.Parameter(torch.randn(slots, rank) * 0.1)
        angles = torch.arange(slots) * (2 * math.pi / slots)
        self.position_seeds = nn.Parameter(torch.stack((angles.cos(), angles.sin()), -1) * 0.5)
        self.raw_spatial_strength = nn.Parameter(torch.tensor(0.0))
        self.update = nn.GRUCell(rank, rank)
        self.refinement = nn.Sequential(
            nn.LayerNorm(rank), nn.Linear(rank, 2 * rank), nn.GELU(), nn.Linear(2 * rank, rank)
        )
        self.nuisance = nn.Sequential(nn.Linear(rank, 2 * rank), nn.GELU(), nn.Linear(2 * rank, 2))
        nn.init.normal_(self.nuisance[-1].weight, std=0.001)
        nn.init.zeros_(self.nuisance[-1].bias)

    def forward(
        self,
        patches: torch.Tensor,
        valid: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        matchability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[-1] != self.input_dim:
            raise ValueError("Patches must be [batch,patches,input_dim]")
        batch, count, _ = patches.shape
        if min(batch, count) < 1:
            raise ValueError("Empty batches or patch grids are unsupported")
        if valid is None:
            valid = torch.ones((batch, count), dtype=torch.bool, device=patches.device)
        if valid.shape != (batch, count) or valid.dtype != torch.bool:
            raise ValueError("Patch validity must be boolean [batch,patches]")
        dtype = self.projection[1].weight.dtype
        if matchability is None:
            matchability = valid.to(dtype)
        if matchability.shape != (batch, count):
            raise ValueError("Matchability must be class-independent [batch,patches]")
        if not torch.isfinite(matchability[valid]).all() or torch.any(
            (matchability[valid] < 0) | (matchability[valid] > 1)
        ):
            raise ValueError("Matchability must be finite within [0,1] on valid patches")
        matchability = torch.where(valid, matchability.to(dtype), 0.0)
        available = valid & (matchability > 0)
        if not torch.isfinite(patches[available]).all():
            raise ValueError("Available patches must be finite")
        if coordinates is None:
            coordinates = square_coordinates(count, device=patches.device, dtype=dtype)
        if coordinates.shape == (count, 2):
            coordinates = coordinates[None].expand(batch, -1, -1)
        if coordinates.shape != (batch, count, 2):
            raise ValueError("Coordinates must be [patches,2] or [batch,patches,2]")
        if not torch.isfinite(coordinates[available]).all():
            raise ValueError("Available patch coordinates must be finite")
        coordinates = torch.where(available[..., None], coordinates.to(dtype), 0.0)
        safe = torch.where(available[..., None], patches.to(dtype), 0.0)
        values = torch.where(available[..., None], self.projection(safe), 0.0)
        # Invalid padding adds no support. Duplicating every physical observation
        # and coordinate leaves these normalized patch budgets unchanged in sum.
        patch_mass = matchability / valid.sum(-1, keepdim=True).clamp_min(1)
        keys = F.normalize(self.key(values), dim=-1, eps=1e-6)
        unmatched = self.unmatched(values)
        query = self.slot_seeds[None].expand(batch, -1, -1)
        centers = self.position_seeds.tanh()[None].expand(batch, -1, -1)

        def readout(query, centers):
            normalized_query = F.normalize(self.query(query), dim=-1, eps=1e-6)
            scores = torch.einsum("bpr,bkr->bpk", keys, normalized_query) * math.sqrt(self.rank)
            distance = (coordinates[:, :, None] - centers[:, None]).square().sum(-1)
            scores = scores - (F.softplus(self.raw_spatial_strength) + 1e-4) * distance
            allocation = torch.cat((scores, unmatched), -1).softmax(-1) * patch_mass[..., None]
            assigned = allocation[..., :-1]
            mass = assigned.sum(1)
            denominator = mass.clamp_min(1e-8)[..., None]
            descriptor = torch.einsum("bpk,bpr->bkr", assigned, values) / denominator
            position = torch.einsum("bpk,bpd->bkd", assigned, coordinates) / denominator
            descriptor = torch.where((mass > 0)[..., None], descriptor, 0.0)
            position = torch.where((mass > 0)[..., None], position, 0.0)
            return allocation, mass, descriptor, position

        for _ in range(self.iterations):
            _, _, descriptor, centers = readout(query, centers)
            query = self.update(descriptor.reshape(-1, self.rank), query.reshape(-1, self.rank))
            query = query.reshape(batch, self.slots, self.rank)
            query = query + self.refinement(query)
        allocation, mass, descriptor, position = readout(query, centers)
        offset = coordinates[:, :, None] - position[:, None]
        spread = torch.einsum("bpk,bpkd->bkd", allocation[..., :-1], offset.square())
        spread = spread / mass.clamp_min(1e-8)[..., None]
        pooled = (values * patch_mass[..., None]).sum(1) / patch_mass.sum(-1).clamp_min(1e-8)[
            :, None
        ]
        nuisance = self.nuisance(pooled).tanh()
        angle, log_scale = nuisance[:, 0] * (math.pi / 6), nuisance[:, 1] * math.log(1.25)
        has_evidence = mass.sum(-1) > 0
        angle = torch.where(has_evidence, angle, 0.0)
        log_scale = torch.where(has_evidence, log_scale, 0.0)
        cosine, sine = angle.cos(), angle.sin()
        rotation = torch.stack((cosine, sine, -sine, cosine), -1).reshape(batch, 2, 2)
        transform = rotation / log_scale.exp()[:, None, None]
        canonical = torch.einsum("bij,bkj->bki", transform, position)
        return {
            "slot_descriptors": descriptor,
            "slot_positions": position,
            "canonical_slot_positions": canonical,
            "slot_masses": mass,
            "slot_spread": spread,
            "patch_allocation": allocation,
            "patch_matchability": allocation[..., :-1].sum(-1),
            "unmatched_mass": allocation[..., -1].sum(-1),
            "input_matchability": matchability,
            "nuisance_transform": transform,
            "nuisance_angle": angle,
            "nuisance_scale": log_scale.exp(),
        }


class SEAR(nn.Module):
    """A5 shared-evidence energy, or A4 with relative geometry disabled.

    Forward takes frozen center patches plus the common short/long video vector.
    All classification comes from visual inputs; no labels, scenario identities,
    per-class masks or externally selected templates are accepted by inference.
    No implicit slot regularizer is applied; the training protocol must predeclare
    any such loss identically for relevant template controls.
    ``valid`` denotes padding; use ``matchability`` for missing physical patches.
    """

    def __init__(
        self,
        input_dim: int = 768,
        video_dim: int = 1536,
        *,
        rank: int = 32,
        slots: int = 6,
        templates_per_class: int = 4,
        iterations: int = 3,
        width: int = 128,
        dropout: float = 0.1,
        geometry: bool = True,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if templates_per_class < 1 or parameter_limit < 1 or not isinstance(geometry, bool):
            raise ValueError("Invalid template count, parameter limit or geometry switch")
        self.geometry, self.templates_per_class = geometry, templates_per_class
        self.extractor = SharedSlotExtractor(
            input_dim, rank=rank, slots=slots, width=width, iterations=iterations
        )
        self.video = CommonVideoHead(video_dim, width=width, dropout=dropout)
        self.template_descriptors = nn.Parameter(torch.randn(3, templates_per_class, slots, rank))
        layout = torch.randn(3, templates_per_class, slots, 2) * 0.5
        if geometry:
            self.template_positions = nn.Parameter(layout)
        else:
            # These remain buffers, not falsely counted inactive trainable capacity.
            self.register_buffer("template_positions", layout)
            self.extractor.nuisance.requires_grad_(False)
        self.raw_cost_weights = nn.Parameter(
            torch.full((2 if geometry else 1,), math.log(math.expm1(1.0)))
        )
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(0.2))))
        if self.trainable_parameters > parameter_limit:
            raise ValueError(f"SEAR has {self.trainable_parameters} parameters, above the limit")

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": self.trainable_parameters,
            "video": sum(p.numel() for p in self.video.parameters() if p.requires_grad),
            "shared_extractor": sum(
                p.numel() for p in self.extractor.parameters() if p.requires_grad
            ),
            "templates_and_scales": sum(
                p.numel() for p in self.parameters(recurse=False) if p.requires_grad
            ),
        }

    def forward(
        self,
        patches: torch.Tensor,
        video_features: torch.Tensor,
        valid: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        matchability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        video_logits = self.video(video_features)
        if patches.ndim != 3 or patches.shape[0] != len(video_logits):
            raise ValueError("Local and video batches must align")
        observed = self.extractor(patches, valid, coordinates, matchability)
        positive = F.softplus(self.raw_cost_weights) + 1e-4
        # Turning geometry off does not renormalize or double the appearance cost.
        weights = torch.stack(
            (positive[0], positive[1] if self.geometry else positive.new_zeros(()))
        )
        energies = shared_template_energy(
            observed["slot_descriptors"],
            observed["canonical_slot_positions"],
            observed["slot_masses"],
            self.template_descriptors,
            self.template_positions.tanh(),
            appearance_weight=weights[0],
            geometry_weight=weights[1],
        )
        temperature = 0.05 + F.softplus(self.raw_temperature)
        local_logits = temperature * (
            torch.logsumexp(-energies["template_energies"] / temperature, -1)
            - math.log(self.templates_per_class)
        )
        available = energies["coverage"] > 0
        local_logits = torch.where(available[:, None], local_logits, 0.0)
        logits = torch.where(available[:, None], video_logits + local_logits, video_logits)
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "video_logits": video_logits,
            "local_logits": local_logits,
            "cost_weights": weights,
            "temperature": temperature,
            **observed,
            **energies,
        }
