"""Model factory, batching and regularizers for the locked SEAR comparison."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.nn import functional as F

from hac.sear import SEAR, square_coordinates
from hac.sear_controls import DenseCenterControl, UnrestrictedTemplateControl

ARMS = (
    "a0_center_cls",
    "a1_dense_cnn",
    "a2_dense_transformer",
    "a3_unrestricted_templates",
    "a4_shared_no_geometry",
    "a5_sear",
)
TEMPLATE_ARMS = ARMS[3:]


def seed_training(seed: int) -> None:
    """Seed all RNGs and require full-precision deterministic neural operations.

    The process entry point owns CUBLAS_WORKSPACE_CONFIG and must set it before
    initializing CUDA; this helper deliberately does not mutate process settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(protocol: dict, arm: str, *, device: str | torch.device = "cpu"):
    if arm not in ARMS or tuple(protocol["arms"]) != ARMS:
        raise ValueError("SEAR arm inventory or order changed")
    architecture = protocol["architecture"]
    declaration = protocol["arms"][arm]
    common = {
        "input_dim": architecture["input_dim"],
        "video_dim": architecture["video_dim"],
        "dropout": architecture["dropout"],
        "parameter_limit": architecture["maximum_parameters"],
    }
    if arm == "a0_center_cls":
        model = DenseCenterControl(
            "center_cls",
            width=declaration["image_width"],
            video_width=architecture["video_width"],
            layers=architecture["layers"],
            heads=architecture["heads"],
            **common,
        )
    elif arm == "a1_dense_cnn":
        model = DenseCenterControl(
            "dense_cnn",
            width=declaration["image_width"],
            video_width=architecture["video_width"],
            layers=architecture["layers"],
            heads=architecture["heads"],
            **common,
        )
    elif arm == "a2_dense_transformer":
        model = DenseCenterControl(
            "dense_transformer",
            width=declaration["image_width"],
            video_width=architecture["video_width"],
            layers=architecture["layers"],
            heads=architecture["heads"],
            **common,
        )
    elif arm == "a3_unrestricted_templates":
        model = UnrestrictedTemplateControl(
            rank=architecture["rank"],
            slots=architecture["slots"],
            templates_per_class=architecture["templates_per_class"],
            iterations=architecture["iterations"],
            width=declaration["image_width"],
            **common,
        )
    else:
        model = SEAR(
            rank=architecture["rank"],
            slots=architecture["slots"],
            templates_per_class=architecture["templates_per_class"],
            iterations=architecture["iterations"],
            width=declaration["image_width"],
            geometry=arm == "a5_sear",
            **common,
        )
    return model.to(device)


def video_module(model, arm: str):
    return model.video_head if arm in ARMS[:3] else model.video


def parameter_counts(protocol: dict) -> dict[str, int]:
    counts = {arm: make_model(protocol, arm).trainable_parameters for arm in ARMS}
    target = counts["a5_sear"]
    if any(abs(counts[arm] / target - 1) >= 0.05 for arm in ARMS[1:]):
        raise RuntimeError("A1-A5 active parameter counts are not within five percent")
    return counts


def forward_arm(
    model,
    arm: str,
    patches: torch.Tensor,
    center_cls: torch.Tensor,
    video: torch.Tensor,
    local_valid: torch.Tensor,
    *,
    coordinates: torch.Tensor | None = None,
    patch_matchability: torch.Tensor | None = None,
):
    if arm not in ARMS:
        raise ValueError("Unknown SEAR arm")
    if patches.ndim != 3 or min(patches.shape) < 1:
        raise ValueError("patches must be nonempty [batch,patch,input_dim]")
    if local_valid.shape != (len(patches),) or local_valid.dtype != torch.bool:
        raise ValueError("local_valid must be boolean [batch]")
    patch_valid = local_valid[:, None].expand(-1, patches.shape[1])
    if arm == "a0_center_cls":
        return model(
            patches,
            video,
            valid=patch_valid,
            matchability=patch_matchability,
            center_cls=center_cls,
            center_cls_valid=local_valid,
        )
    # The transformer control's standalone default uses a different convention;
    # explicitly bind every dense arm to the same physical patch-center grid.
    if coordinates is None:
        coordinates = square_coordinates(
            patches.shape[1], device=patches.device, dtype=next(model.parameters()).dtype
        )
    return model(
        patches,
        video,
        valid=patch_valid,
        coordinates=coordinates,
        matchability=patch_matchability,
    )


def slot_regularization(
    outputs: dict[str, torch.Tensor],
    protocol: dict,
) -> dict[str, torch.Tensor]:
    """Identical class-independent extractor regularization for A3-A5."""
    patch_allocation = outputs["patch_allocation"]
    if (
        patch_allocation.ndim != 3
        or min(patch_allocation.shape[:2]) < 1
        or patch_allocation.shape[2] < 2
    ):
        raise ValueError("Patch allocation must be nonempty [batch,patch,slots+unmatched]")
    allocation = patch_allocation[..., :-1]
    batch, _, slots = allocation.shape
    positions = outputs["slot_positions"]
    masses = outputs["slot_masses"]
    if positions.shape != (batch, slots, 2) or masses.shape != (batch, slots):
        raise ValueError("Slot allocation, positions and masses are inconsistent")
    coordinates = square_coordinates(
        allocation.shape[1], device=allocation.device, dtype=allocation.dtype
    )
    distance = (coordinates[None, :, None] - positions[:, None]).square().sum(-1)
    compactness = (allocation * distance).sum(-1).sum(-1).mean()
    mean_mass = masses.mean(0)
    distribution = mean_mass / mean_mass.sum().clamp_min(1e-8)
    balance = torch.where(
        mean_mass.sum() > 0,
        (distribution * (distribution.clamp_min(1e-8) * slots).log()).sum(),
        mean_mass.new_zeros(()),
    )
    weighted = (
        protocol["training"]["slot_compactness_weight"] * compactness
        + protocol["training"]["batch_slot_balance_weight"] * balance
    )
    return {"compactness": compactness, "batch_slot_balance": balance, "weighted": weighted}


def training_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    arm: str,
    protocol: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    classification = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    regularizers: dict[str, torch.Tensor] = {}
    if arm in TEMPLATE_ARMS:
        regularizers = slot_regularization(outputs, protocol)
    auxiliary = regularizers.get("weighted", classification.new_zeros(()))
    return classification + auxiliary, {"classification": classification, **regularizers}
