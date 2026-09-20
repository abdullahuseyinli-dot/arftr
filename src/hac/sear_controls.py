"""Matched non-template controls for the SEAR center-evidence experiment.

These heads deliberately return additive local and video logits.  Keeping the
interface identical makes it possible to test whether dense center evidence is
useful before attributing a gain to the shared-evidence template mechanism.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from hac.sear import CommonVideoHead, SharedSlotExtractor

CONTROL_ARMS = ("center_cls", "dense_cnn", "dense_transformer")


def _validate_inputs(
    patches: torch.Tensor,
    video_features: torch.Tensor,
    valid: torch.Tensor | None,
    matchability: torch.Tensor | None,
    *,
    input_dim: int,
    video_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if patches.ndim != 3 or patches.shape[-1] != input_dim or patches.shape[0] < 1:
        raise ValueError("patches must be [batch,patch,input_dim]")
    if video_features.shape != (patches.shape[0], video_dim):
        raise ValueError("video_features must be [batch,video_dim]")
    if not patches.is_floating_point() or not video_features.is_floating_point():
        raise ValueError("SEAR control inputs must be floating point")
    if valid is None:
        valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
    if valid.shape != patches.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [batch,patch]")
    if matchability is None:
        matchability = valid.to(patches.dtype)
    if matchability.shape != patches.shape[:2]:
        raise ValueError("matchability must be class-independent [batch,patch]")
    if not torch.isfinite(matchability[valid]).all() or torch.any(
        (matchability[valid] < 0) | (matchability[valid] > 1)
    ):
        raise ValueError("matchability must be finite within [0,1] on valid patches")
    matchability = torch.where(valid, matchability.to(patches.dtype), 0.0)
    available = valid & (matchability > 0)
    if not torch.isfinite(patches[available]).all() or not torch.isfinite(video_features).all():
        raise ValueError("Available patches and video features must be finite")
    return valid, matchability


def _default_coordinates(patches: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = math.isqrt(patches)
    if side * side != patches:
        raise ValueError("Dense controls require a square patch grid")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(patches, 2)


def _masked_mean_and_max(
    values: torch.Tensor, valid: torch.Tensor, matchability: torch.Tensor
) -> torch.Tensor:
    weights = matchability.to(values.dtype)
    denominator = valid.sum(1).clamp_min(1).to(values.dtype)[:, None]
    mean = (values * weights[..., None]).sum(1) / denominator
    available = weights > 0
    maximum = values.masked_fill(~available[..., None], -torch.inf).amax(1)
    coverage = weights.sum(1, keepdim=True) / denominator
    maximum = torch.where(available.any(1, keepdim=True), maximum * coverage, 0.0)
    return torch.cat((mean, maximum), dim=-1)


class DenseCenterControl(nn.Module):
    """A0-A2 controls with common additive video evidence.

    ``center_cls`` consumes an explicitly supplied center CLS descriptor and is
    the intentional no-patch-information baseline.  The dense controls receive
    all patch tokens.  Local logits are multiplied by observed support so an
    all-missing patch input returns *exactly* the video branch logits.
    """

    def __init__(
        self,
        arm: str,
        *,
        input_dim: int = 768,
        video_dim: int = 1536,
        width: int = 128,
        video_width: int = 128,
        layers: int = 2,
        heads: int = 4,
        classes: int = 3,
        dropout: float = 0.1,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if arm not in CONTROL_ARMS:
            raise ValueError(f"Unknown SEAR control arm: {arm}")
        if (
            min(input_dim, video_dim, width, video_width, layers, heads, classes) < 1
            or width % heads
            or classes != 3
        ):
            raise ValueError("Invalid SEAR control dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.arm = arm
        self.input_dim, self.video_dim, self.width = input_dim, video_dim, width
        # This is byte-for-byte the video branch used by A4/A5.
        self.video_head = CommonVideoHead(video_dim, width=video_width, dropout=dropout)
        if arm == "center_cls":
            self.image_encoder = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.GELU(),
            )
            image_width = width
        elif arm == "dense_cnn":
            self.patch_projection = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, width)
            )
            blocks = []
            for _ in range(layers):
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(width, width, 3, padding=1, groups=width),
                        nn.GELU(),
                        nn.Conv2d(width, width, 1),
                        nn.GELU(),
                        nn.Dropout2d(dropout),
                    )
                )
            self.spatial_blocks = nn.ModuleList(blocks)
            image_width = 2 * width
        else:
            self.patch_projection = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, width)
            )
            self.position_projection = nn.Sequential(
                nn.Linear(2, width), nn.GELU(), nn.Linear(width, width)
            )
            layer = nn.TransformerEncoderLayer(
                width,
                heads,
                2 * width,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.spatial_transformer = nn.TransformerEncoder(
                layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
            )
            image_width = 2 * width
        self.local_head = nn.Sequential(
            nn.LayerNorm(image_width),
            nn.Linear(image_width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, classes),
        )
        if self.trainable_parameters > parameter_limit:
            raise ValueError("SEAR control exceeds the declared parameter limit")

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _dense_representation(
        self,
        patches: torch.Tensor,
        valid: torch.Tensor,
        matchability: torch.Tensor,
        coordinates: torch.Tensor | None,
    ) -> torch.Tensor:
        available = valid & (matchability > 0)
        safe = torch.where(available[..., None], patches, 0.0)
        values = self.patch_projection(safe.to(self.patch_projection[1].weight.dtype))
        values = torch.where(available[..., None], values, 0.0)
        if self.arm == "dense_cnn":
            side = math.isqrt(patches.shape[1])
            if side * side != patches.shape[1]:
                raise ValueError("Dense CNN requires a square patch grid")
            image = values.transpose(1, 2).reshape(len(values), self.width, side, side)
            mask = available.reshape(len(values), 1, side, side).to(image.dtype)
            for block in self.spatial_blocks:
                image = (image + block(image)) * mask
            values = image.flatten(2).transpose(1, 2)
        else:
            if coordinates is None:
                coordinates = _default_coordinates(
                    patches.shape[1], device=patches.device, dtype=values.dtype
                )
            if coordinates.shape != (patches.shape[1], 2) or not torch.isfinite(coordinates).all():
                raise ValueError("coordinates must be finite [patch,2]")
            values = values + self.position_projection(coordinates.to(values.dtype))[None]
            # TransformerEncoder produces NaNs for an entirely masked row.  Let
            # one zero-valued sentinel participate, then erase it before pooling.
            safe_valid = available.clone()
            missing = ~safe_valid.any(1)
            safe_valid[missing, 0] = True
            values = self.spatial_transformer(values, src_key_padding_mask=~safe_valid)
            values = torch.where(available[..., None], values, 0.0)
        return _masked_mean_and_max(values, valid, matchability)

    def forward(
        self,
        patches: torch.Tensor,
        video_features: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        matchability: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        center_cls: torch.Tensor | None = None,
        center_cls_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid, matchability = _validate_inputs(
            patches,
            video_features,
            valid,
            matchability,
            input_dim=self.input_dim,
            video_dim=self.video_dim,
        )
        dtype = next(self.parameters()).dtype
        if self.arm == "center_cls":
            if center_cls is None or center_cls.shape != (len(patches), self.input_dim):
                raise ValueError("center_cls arm requires [batch,input_dim] center_cls")
            if center_cls_valid is None:
                center_cls_valid = torch.ones(len(patches), dtype=torch.bool, device=patches.device)
            if center_cls_valid.shape != (len(patches),) or center_cls_valid.dtype != torch.bool:
                raise ValueError("center_cls_valid must be boolean [batch]")
            if not torch.isfinite(center_cls[center_cls_valid]).all():
                raise ValueError("Available center_cls values must be finite")
            safe_cls = torch.where(center_cls_valid[:, None], center_cls, 0.0)
            image = self.image_encoder(safe_cls.to(dtype))
            observed = center_cls_valid[:, None].to(dtype)
        else:
            image = self._dense_representation(patches, valid, matchability, coordinates)
            observed = (matchability.sum(1, keepdim=True) > 0).to(dtype)
        local_logits = self.local_head(image) * observed
        video_logits = self.video_head(video_features.to(dtype))
        logits = video_logits + local_logits
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "local_logits": local_logits,
            "video_logits": video_logits,
            "local_observed": observed.squeeze(-1).bool(),
        }


class UnrestrictedTemplateControl(nn.Module):
    """A3 control with class/template-specific hiding and nuisance privileges.

    It deliberately shares the initial visual-slot extractor with SEAR, then lets
    each class/template choose its own slot weights, viewpoint transform and cost
    scales.  A3 is therefore an adversarially useful control for the proposed
    shared-evidence constraint, not a recommended inference mechanism.
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
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if min(rank, slots, templates_per_class, parameter_limit) < 1:
            raise ValueError("Invalid unrestricted-template configuration")
        self.rank, self.slots = rank, slots
        self.templates_per_class = templates_per_class
        self.extractor = SharedSlotExtractor(
            input_dim, rank=rank, slots=slots, width=width, iterations=iterations
        )
        # A3 replaces this shared transform with a class/template-specific one.
        self.extractor.nuisance.requires_grad_(False)
        self.video = CommonVideoHead(video_dim, width=width, dropout=dropout)
        shape = (3, templates_per_class, slots)
        self.template_descriptors = nn.Parameter(torch.randn(*shape, rank))
        self.template_positions = nn.Parameter(torch.randn(*shape, 2) * 0.5)
        self.visibility_queries = nn.Parameter(torch.randn(*shape, rank) * 0.1)
        self.visibility_bias = nn.Parameter(torch.zeros(shape))
        self.nuisance = nn.Linear(rank, 3 * templates_per_class * 2)
        nn.init.normal_(self.nuisance.weight, std=0.001)
        nn.init.zeros_(self.nuisance.bias)
        self.raw_cost_weights = nn.Parameter(
            torch.full((3, templates_per_class, 2), math.log(math.expm1(1.0)))
        )
        self.raw_temperature = nn.Parameter(torch.full((3,), math.log(math.expm1(0.2))))
        if self.trainable_parameters > parameter_limit:
            raise ValueError("Unrestricted template control exceeds the parameter limit")

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
            "class_specific": sum(
                p.numel() for p in self.parameters(recurse=False) if p.requires_grad
            )
            + sum(p.numel() for p in self.nuisance.parameters() if p.requires_grad),
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
        descriptor = observed["slot_descriptors"]
        position = observed["slot_positions"]
        base_mass = observed["slot_masses"]
        normalized = F.normalize(descriptor, dim=-1, eps=1e-6)
        prototypes = F.normalize(self.template_descriptors, dim=-1, eps=1e-6)

        # Unlike A4/A5, each class/template can suppress inconvenient slots.
        visibility = torch.sigmoid(
            torch.einsum("bsr,ctsr->bcts", normalized, self.visibility_queries)
            / math.sqrt(self.rank)
            + self.visibility_bias[None]
        )
        masses = base_mass[:, None, None] * visibility
        unary_cost = (2 - 2 * torch.einsum("bsr,ctsr->bcts", normalized, prototypes)).clamp_min(0)
        unary_energy = (unary_cost * masses).sum(-1)

        pooled = (descriptor * base_mass[..., None]).sum(1) / base_mass.sum(-1).clamp_min(1e-8)[
            :, None
        ]
        raw_nuisance = self.nuisance(pooled).reshape(len(patches), 3, self.templates_per_class, 2)
        angle = raw_nuisance[..., 0].tanh() * (math.pi / 6)
        scale = (raw_nuisance[..., 1].tanh() * math.log(1.25)).exp()
        cosine, sine = angle.cos(), angle.sin()
        transform = torch.stack((cosine, sine, -sine, cosine), -1).reshape(
            len(patches), 3, self.templates_per_class, 2, 2
        )
        transform = transform / scale[..., None, None]
        canonical = torch.einsum("bctij,bsj->bctsi", transform, position)
        relative = canonical[..., :, None, :] - canonical[..., None, :, :]
        template_position = self.template_positions.tanh()
        template_relative = template_position[..., :, None, :] - template_position[..., None, :, :]
        pair_cost = (relative - template_relative[None]).square().sum(-1) / 8
        pair_mass = masses[..., :, None] * masses[..., None, :]
        nonself = ~torch.eye(self.slots, device=patches.device, dtype=torch.bool)
        pair_mass = torch.where(nonself, pair_mass, 0.0)
        geometry_energy = (pair_cost * pair_mass).sum((-1, -2))
        weights = F.softplus(self.raw_cost_weights) + 1e-4
        energies = weights[None, ..., 0] * unary_energy + weights[None, ..., 1] * geometry_energy
        temperatures = 0.05 + F.softplus(self.raw_temperature)
        local_logits = temperatures[None] * (
            torch.logsumexp(-energies / temperatures[None, :, None], -1)
            - math.log(self.templates_per_class)
        )
        available = base_mass.sum(-1) > 0
        local_logits = torch.where(available[:, None], local_logits, 0.0)
        logits = torch.where(available[:, None], video_logits + local_logits, video_logits)
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "local_logits": local_logits,
            "video_logits": video_logits,
            "template_energies": energies,
            "class_template_slot_masses": masses,
            "class_template_pairwise_masses": pair_mass,
            "class_template_nuisance_transform": transform,
            "class_template_cost_weights": weights,
            "class_template_temperature": temperatures,
            **observed,
        }
