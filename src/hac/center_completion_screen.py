"""Label-blind reconstruction screen for visible-anchor center completion.

This module deliberately has no action-label, ARFTR, pose, or annotation-support
interface.  It operates on packed hidden-center target tokens so the 128-center
screen need not materialize dense multi-gigabyte tensors during optimization.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hac.okutama_native_video import EXPECTED_SCENARIOS

FEATURE_DIM = 768
PROJECTED_DIM = 128
DECODER_WIDTH = 256
PATCH_GRID = 27
MASK_COUNT = 2
DONOR_COUNT = 4
GATE_FEATURES = 21
TEMPORAL_FEATURES = 8
SCREEN_SEED = 42
MAX_UPDATES = 400
NEIGHBOR_OFFSETS = (-8.0, -4.0, 4.0, 7.0)

Arm = Literal["P0_center_only", "P1_unordered_pool", "P2_same_grid_pool", "P3_partial_transport"]
ARMS: tuple[Arm, ...] = (
    "P0_center_only",
    "P1_unordered_pool",
    "P2_same_grid_pool",
    "P3_partial_transport",
)


@dataclass(frozen=True)
class PackedScreenData:
    """An arm-specific, padded target-token population.

    The first two dimensions are center and packed target.  Padding is explicit
    in ``target_valid`` and never silently removed from a center denominator.
    """

    masked_tokens: torch.Tensor
    visible_context: torch.Tensor
    teacher_tokens: torch.Tensor
    positions_yx: torch.Tensor
    mask_ids: torch.Tensor
    target_valid: torch.Tensor
    evidence_tokens: torch.Tensor
    evidence_available: torch.Tensor
    temporal_metadata: torch.Tensor
    gate_features: torch.Tensor | None = None

    @property
    def centers(self) -> int:
        return int(self.masked_tokens.shape[0])

    @property
    def targets(self) -> int:
        return int(self.masked_tokens.shape[1])

    def validate(self, *, require_gate: bool = False, check_values: bool = True) -> None:
        prefix = self.masked_tokens.shape[:2]
        token_shape = (*prefix, FEATURE_DIM)
        if self.masked_tokens.shape != token_shape or self.masked_tokens.ndim != 3:
            raise ValueError("masked tokens must have shape N,T,768")
        for name, tensor in (
            ("visible context", self.visible_context),
            ("teacher tokens", self.teacher_tokens),
            ("evidence tokens", self.evidence_tokens),
        ):
            if tensor.shape != token_shape or not tensor.is_floating_point():
                raise ValueError(f"{name} must have shape N,T,768 and floating dtype")
        if not self.masked_tokens.is_floating_point():
            raise ValueError("masked tokens must have floating dtype")
        if self.positions_yx.shape != (*prefix, 2) or self.positions_yx.dtype != torch.int64:
            raise ValueError("positions must have shape N,T,2 and int64 dtype")
        if self.mask_ids.shape != prefix or self.mask_ids.dtype != torch.int64:
            raise ValueError("mask IDs must have shape N,T and int64 dtype")
        if self.target_valid.shape != prefix or self.target_valid.dtype != torch.bool:
            raise ValueError("target validity must have shape N,T and bool dtype")
        if self.evidence_available.shape != prefix or self.evidence_available.dtype != torch.bool:
            raise ValueError("evidence validity must have shape N,T and bool dtype")
        if self.temporal_metadata.shape != (*prefix, TEMPORAL_FEATURES):
            raise ValueError("temporal metadata must have shape N,T,8")
        if not self.temporal_metadata.is_floating_point():
            raise ValueError("temporal metadata must have floating dtype")
        if self.gate_features is not None and self.gate_features.shape != (
            *prefix,
            GATE_FEATURES,
        ):
            raise ValueError("gate features must contain exactly 21 values per target")
        if require_gate and self.gate_features is None:
            raise ValueError("P3 requires exactly 21 frozen inference features")
        if self.gate_features is not None and not self.gate_features.is_floating_point():
            raise ValueError("gate features must have floating dtype")
        if self.centers < 1 or self.targets < 1 or not self.target_valid.any(dim=1).all():
            raise ValueError("every center must contain at least one scored target")
        valid_positions = self.positions_yx[self.target_valid]
        if (
            (valid_positions < 0).any()
            or (valid_positions >= PATCH_GRID).any()
            or (self.mask_ids[self.target_valid] < 0).any()
            or (self.mask_ids[self.target_valid] >= MASK_COUNT).any()
        ):
            raise ValueError("valid target positions or mask IDs leave the locked grid")
        if check_values:
            values = [
                self.masked_tokens,
                self.visible_context,
                self.teacher_tokens,
                self.evidence_tokens,
                self.temporal_metadata,
            ]
            if self.gate_features is not None:
                values.append(self.gate_features)
            if not all(torch.isfinite(value).all() for value in values):
                raise ValueError("screen inputs must be finite")

    def select(self, rows: Sequence[int] | np.ndarray | torch.Tensor) -> PackedScreenData:
        index = torch.as_tensor(rows, dtype=torch.int64, device=self.masked_tokens.device)
        kwargs = {
            name: value.index_select(0, index) if value is not None else None
            for name, value in self.__dict__.items()
        }
        return PackedScreenData(**kwargs)

    def to(self, device: str | torch.device) -> PackedScreenData:
        return PackedScreenData(
            **{
                name: (
                    value.to(device=device, dtype=torch.float32)
                    if value is not None and value.is_floating_point()
                    else value.to(device)
                    if value is not None
                    else None
                )
                for name, value in self.__dict__.items()
            }
        )


@dataclass(frozen=True)
class ScreenOutput:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    candidate_prediction: torch.Tensor
    gate: torch.Tensor
    evidence_available: torch.Tensor


@dataclass(frozen=True)
class LossTerms:
    objective: torch.Tensor
    cosine_error: torch.Tensor
    layernorm_huber: torch.Tensor
    intervention_cost: torch.Tensor
    per_center_cosine: torch.Tensor
    per_center_huber: torch.Tensor


@dataclass(frozen=True)
class FitResult:
    model: nn.Module
    training_history: tuple[float, ...]
    train_rows: np.ndarray
    held_rows: np.ndarray
    seed: int
    updates: int


def _as_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))


def _dense_shape_check(
    masked_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    neighbor_tokens: torch.Tensor,
    neighbor_valid: torch.Tensor,
    target_masks: torch.Tensor,
    visible_masks: torch.Tensor,
) -> tuple[int, int]:
    if masked_tokens.ndim != 5:
        raise ValueError("masked dense tokens must have shape N,2,27,27,768")
    centers, masks, height, width, dimension = masked_tokens.shape
    if (masks, height, width, dimension) != (MASK_COUNT, PATCH_GRID, PATCH_GRID, FEATURE_DIM):
        raise ValueError("masked dense token geometry changed")
    if teacher_tokens.shape != (centers, PATCH_GRID, PATCH_GRID, FEATURE_DIM):
        raise ValueError("teacher dense token geometry changed")
    if neighbor_tokens.shape != (
        centers,
        DONOR_COUNT,
        PATCH_GRID,
        PATCH_GRID,
        FEATURE_DIM,
    ):
        raise ValueError("neighbor dense token geometry changed")
    if neighbor_valid.shape != neighbor_tokens.shape[:-1] or neighbor_valid.dtype != torch.bool:
        raise ValueError("neighbor validity geometry or dtype changed")
    if target_masks.shape != masked_tokens.shape[:-1] or target_masks.dtype != torch.bool:
        raise ValueError("target mask geometry or dtype changed")
    if visible_masks.shape != target_masks.shape or visible_masks.dtype != torch.bool:
        raise ValueError("visible mask geometry or dtype changed")
    if torch.logical_and(target_masks, visible_masks).any():
        raise ValueError("a hidden target was exposed as visible context")
    counts = target_masks.flatten(1).sum(dim=1)
    if (counts < 1).any():
        raise ValueError("every center must contain a target")
    return centers, int(counts.max())


def pack_dense_arm(
    *,
    arm: Arm,
    masked_tokens: torch.Tensor | np.ndarray,
    teacher_tokens: torch.Tensor | np.ndarray,
    neighbor_tokens: torch.Tensor | np.ndarray,
    neighbor_valid: torch.Tensor | np.ndarray,
    target_masks: torch.Tensor | np.ndarray,
    visible_masks: torch.Tensor | np.ndarray,
    transport_tokens: torch.Tensor | np.ndarray | None = None,
    transport_available: torch.Tensor | np.ndarray | None = None,
    gate_features: torch.Tensor | np.ndarray | None = None,
) -> PackedScreenData:
    """Pack one dense cache into the exact row-major target population for an arm."""

    if arm not in ARMS:
        raise ValueError("unknown reconstruction arm")
    masked = _as_tensor(masked_tokens)
    teacher = _as_tensor(teacher_tokens)
    neighbors = _as_tensor(neighbor_tokens)
    neighbor_ok = _as_tensor(neighbor_valid)
    targets = _as_tensor(target_masks)
    visible = _as_tensor(visible_masks)
    centers, packed_targets = _dense_shape_check(
        masked, teacher, neighbors, neighbor_ok, targets, visible
    )
    floating_dtype = masked.dtype
    device = masked.device
    shape = (centers, packed_targets)
    packed = {
        "masked_tokens": torch.zeros((*shape, FEATURE_DIM), dtype=floating_dtype, device=device),
        "visible_context": torch.zeros((*shape, FEATURE_DIM), dtype=floating_dtype, device=device),
        "teacher_tokens": torch.zeros((*shape, FEATURE_DIM), dtype=teacher.dtype, device=device),
        "positions_yx": torch.zeros((*shape, 2), dtype=torch.int64, device=device),
        "mask_ids": torch.zeros(shape, dtype=torch.int64, device=device),
        "target_valid": torch.zeros(shape, dtype=torch.bool, device=device),
        "evidence_tokens": torch.zeros((*shape, FEATURE_DIM), dtype=neighbors.dtype, device=device),
        "evidence_available": torch.zeros(shape, dtype=torch.bool, device=device),
        "temporal_metadata": torch.zeros(
            (*shape, TEMPORAL_FEATURES), dtype=torch.float32, device=device
        ),
    }
    dense_transport = _as_tensor(transport_tokens) if transport_tokens is not None else None
    dense_available = _as_tensor(transport_available) if transport_available is not None else None
    dense_gate = _as_tensor(gate_features) if gate_features is not None else None
    if arm == "P3_partial_transport":
        expected_prefix = (centers, MASK_COUNT, PATCH_GRID, PATCH_GRID)
        if dense_transport is None or dense_transport.shape != (*expected_prefix, FEATURE_DIM):
            raise ValueError("P3 transport tokens must have shape N,2,27,27,768")
        if dense_available is None or dense_available.shape != expected_prefix:
            raise ValueError("P3 transport availability must have shape N,2,27,27")
        if dense_available.dtype != torch.bool:
            raise ValueError("P3 transport availability must be boolean")
        if dense_gate is None or dense_gate.shape != (*expected_prefix, GATE_FEATURES):
            raise ValueError("P3 gate input must contain exactly 21 features")

    offsets = torch.tensor(NEIGHBOR_OFFSETS, dtype=torch.float32, device=device) / 8.0
    for center in range(centers):
        cursor = 0
        global_donor_valid = neighbor_ok[center].flatten(1).any(dim=1)
        global_valid_tokens = neighbor_ok[center]
        if global_valid_tokens.any():
            unordered = (
                neighbors[center][global_valid_tokens].float().mean(dim=0).to(neighbors.dtype)
            )
        else:
            unordered = torch.zeros(FEATURE_DIM, dtype=neighbors.dtype, device=device)
        for mask_id in range(MASK_COUNT):
            positions = torch.nonzero(targets[center, mask_id], as_tuple=False)
            length = len(positions)
            rows = slice(cursor, cursor + length)
            y, x = positions[:, 0], positions[:, 1]
            packed["masked_tokens"][center, rows] = masked[center, mask_id, y, x]
            packed["teacher_tokens"][center, rows] = teacher[center, y, x]
            packed["positions_yx"][center, rows] = positions
            packed["mask_ids"][center, rows] = mask_id
            packed["target_valid"][center, rows] = True
            visible_tokens = masked[center, mask_id][visible[center, mask_id]]
            if not len(visible_tokens):
                raise ValueError("each masked crop requires visible center context")
            context = visible_tokens.float().mean(dim=0).to(masked.dtype)
            packed["visible_context"][center, rows] = context
            if arm == "P1_unordered_pool":
                packed["evidence_tokens"][center, rows] = unordered
                packed["evidence_available"][center, rows] = global_donor_valid.any()
                valid_bits = global_donor_valid.float().expand(length, -1)
            elif arm == "P2_same_grid_pool":
                selected = neighbors[center, :, y, x].permute(1, 0, 2)
                selected_valid = neighbor_ok[center, :, y, x].transpose(0, 1)
                denominator = selected_valid.sum(dim=1, keepdim=True).clamp_min(1)
                same_grid = (
                    (selected.float() * selected_valid[..., None]).sum(dim=1) / denominator
                ).to(neighbors.dtype)
                packed["evidence_tokens"][center, rows] = same_grid
                packed["evidence_available"][center, rows] = selected_valid.any(dim=1)
                valid_bits = selected_valid.float()
            elif arm == "P3_partial_transport":
                packed["evidence_tokens"][center, rows] = dense_transport[center, mask_id, y, x]
                packed["evidence_available"][center, rows] = dense_available[center, mask_id, y, x]
                valid_bits = dense_gate[center, mask_id, y, x, 1::4][:, :DONOR_COUNT]
            else:
                valid_bits = torch.zeros((length, DONOR_COUNT), device=device)
            if arm != "P0_center_only":
                packed["temporal_metadata"][center, rows, :DONOR_COUNT] = offsets
                packed["temporal_metadata"][center, rows, DONOR_COUNT:] = valid_bits
            cursor += length
    packed_gate = None
    if arm == "P3_partial_transport":
        packed_gate = torch.zeros((*shape, GATE_FEATURES), dtype=torch.float32, device=device)
        for center in range(centers):
            cursor = 0
            for mask_id in range(MASK_COUNT):
                positions = torch.nonzero(targets[center, mask_id], as_tuple=False)
                length = len(positions)
                y, x = positions[:, 0], positions[:, 1]
                packed_gate[center, cursor : cursor + length] = dense_gate[center, mask_id, y, x]
                cursor += length
    result = PackedScreenData(**packed, gate_features=packed_gate)
    result.validate(require_gate=arm == "P3_partial_transport")
    return result


def build_gate_features_21(
    donor_tokens: torch.Tensor,
    donor_observed: torch.Tensor,
    residuals: torch.Tensor,
    local_anchor_cosine: torch.Tensor,
    nearest_anchor_distance: torch.Tensor,
) -> torch.Tensor:
    """Construct the locked 21 inference-only P3 gate features.

    The donor axis must be penultimate in tokens and last in the four scalar
    tensors.  Invalid-donor scalar values are replaced by fixed sentinels.
    """

    if donor_tokens.ndim < 2 or donor_tokens.shape[-2:] != (DONOR_COUNT, FEATURE_DIM):
        raise ValueError("donor tokens must end in 4,768")
    prefix = donor_tokens.shape[:-2]
    expected = (*prefix, DONOR_COUNT)
    scalar_inputs = (donor_observed, residuals, local_anchor_cosine, nearest_anchor_distance)
    if any(value.shape != expected for value in scalar_inputs):
        raise ValueError("donor gate statistics must share the token prefix and four donors")
    if donor_observed.dtype != torch.bool:
        raise ValueError("donor observations must be boolean")
    if not donor_tokens.is_floating_point() or not all(
        value.is_floating_point() for value in scalar_inputs[1:]
    ):
        raise ValueError("donor features and statistics must be floating point")
    if not torch.isfinite(donor_tokens).all():
        raise ValueError("donor tokens must be finite")
    for value in scalar_inputs[1:]:
        if not torch.isfinite(value[donor_observed]).all():
            raise ValueError("observed donor statistics must be finite")

    observed = donor_observed
    valid_fraction = observed.float().mean(dim=-1, keepdim=True)
    residual = torch.where(observed, (residuals / 2.0).clamp(0, 1), torch.ones_like(residuals))
    cosine = torch.where(observed, local_anchor_cosine.clamp(-1, 1), torch.zeros_like(residuals))
    distance = torch.where(
        observed,
        (nearest_anchor_distance / 4.0).clamp(0, 1),
        torch.ones_like(residuals),
    )
    blocks = torch.stack([observed.float(), residual, cosine, distance], dim=-1).flatten(-2)

    normalized = F.normalize(donor_tokens.float(), dim=-1, eps=1e-12)
    pair_distances = []
    pair_valid = []
    for left in range(DONOR_COUNT):
        for right in range(left + 1, DONOR_COUNT):
            pair_distances.append(
                1.0 - (normalized[..., left, :] * normalized[..., right, :]).sum(-1)
            )
            pair_valid.append(observed[..., left] & observed[..., right])
    distances = torch.stack(pair_distances, dim=-1)
    pairs = torch.stack(pair_valid, dim=-1)
    ordered = torch.sort(
        torch.where(pairs, distances, torch.full_like(distances, torch.inf)), dim=-1
    ).values
    counts = pairs.sum(dim=-1)
    lower = ((counts - 1).clamp_min(0) // 2).unsqueeze(-1)
    upper = (counts.clamp_min(1) // 2).unsqueeze(-1)
    median = 0.5 * (ordered.gather(-1, lower) + ordered.gather(-1, upper))
    median = torch.where(counts.unsqueeze(-1) > 0, median, torch.zeros_like(median))
    maximum = (
        torch.where(
            pairs,
            distances,
            torch.full_like(distances, -torch.inf),
        )
        .max(dim=-1, keepdim=True)
        .values
    )
    maximum = torch.where(counts.unsqueeze(-1) > 0, maximum, torch.zeros_like(maximum))
    past = observed[..., :2].float().mean(dim=-1, keepdim=True)
    future = observed[..., 2:].float().mean(dim=-1, keepdim=True)
    result = torch.cat([valid_fraction, blocks, median, maximum, past, future], dim=-1)
    if result.shape != (*prefix, GATE_FEATURES) or not torch.isfinite(result).all():
        raise RuntimeError("21-feature gate construction changed")
    return result.detach()


def _orthogonal_projection(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((FEATURE_DIM, PROJECTED_DIM))
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return torch.from_numpy((q * signs).astype(np.float32))


def _position_embedding() -> torch.Tensor:
    frequencies = np.exp(
        -np.log(10_000.0)
        * np.arange(PROJECTED_DIM // 4, dtype=np.float64)
        / max(PROJECTED_DIM // 4 - 1, 1)
    )
    yy, xx = np.mgrid[:PATCH_GRID, :PATCH_GRID]
    y = yy[..., None] * frequencies
    x = xx[..., None] * frequencies
    return torch.from_numpy(
        np.concatenate([np.sin(y), np.cos(y), np.sin(x), np.cos(x)], axis=-1).astype(np.float32)
    )


class FixedScreenEncoding(nn.Module):
    """Frozen seed-locked projections and embeddings."""

    def __init__(self, seed: int = SCREEN_SEED):
        super().__init__()
        self.register_buffer("projection", _orthogonal_projection(seed), persistent=True)
        self.register_buffer("position", _position_embedding(), persistent=True)
        rng = np.random.default_rng(seed + 1)
        mask = rng.standard_normal((MASK_COUNT, PROJECTED_DIM)).astype(np.float32)
        mask /= np.maximum(np.linalg.norm(mask, axis=1, keepdims=True), 1e-12)
        self.register_buffer("mask", torch.from_numpy(mask), persistent=True)
        temporal = rng.standard_normal((PROJECTED_DIM, TEMPORAL_FEATURES))
        temporal, _ = np.linalg.qr(temporal, mode="reduced")
        self.register_buffer(
            "temporal", torch.from_numpy(temporal.T.astype(np.float32)), persistent=True
        )

    def assemble(self, data: PackedScreenData, *, use_evidence: bool) -> torch.Tensor:
        # Immutable caches are float16; optimization is intentionally float32.
        projection = self.projection
        local = data.masked_tokens.float() @ projection
        context = data.visible_context.float() @ projection
        position = self.position[data.positions_yx[..., 0], data.positions_yx[..., 1]]
        mask = self.mask[data.mask_ids]
        if use_evidence:
            evidence = data.evidence_tokens.float() @ projection
            temporal = data.temporal_metadata.float() @ self.temporal
        else:
            evidence = torch.zeros_like(local)
            temporal = torch.zeros_like(local)
        return torch.cat([local, context, position, mask, evidence, temporal], dim=-1)


class CompletionDecoder(nn.Module):
    """The common LayerNorm/two-layer 256-wide GELU residual decoder."""

    input_dim = 6 * PROJECTED_DIM

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, DECODER_WIDTH),
            nn.GELU(),
            nn.Linear(DECODER_WIDTH, FEATURE_DIM),
        )

    def forward(self, inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if inputs.shape[:-1] != reference.shape[:-1] or inputs.shape[-1] != self.input_dim:
            raise ValueError("decoder input geometry changed")
        return reference + self.network(inputs)


class BaseCompletionModel(nn.Module):
    arm: Arm = "P0_center_only"

    def __init__(self, seed: int = SCREEN_SEED):
        super().__init__()
        self.fixed = FixedScreenEncoding(seed)
        self.decoder = CompletionDecoder()

    def forward(self, data: PackedScreenData) -> ScreenOutput:
        data.validate(check_values=False)
        assembled = self.fixed.assemble(data, use_evidence=False)
        prediction = self.decoder(assembled, data.masked_tokens.float())
        zeros = torch.zeros_like(data.target_valid, dtype=prediction.dtype)
        return ScreenOutput(prediction, prediction, prediction, zeros, data.target_valid)


class EvidenceCompletionModel(nn.Module):
    def __init__(self, base: BaseCompletionModel, arm: Arm):
        super().__init__()
        if arm not in ARMS[1:]:
            raise ValueError("an evidence model requires P1, P2, or P3")
        self.arm = arm
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.base.eval()
        self.decoder = CompletionDecoder()
        self.gate = (
            nn.Sequential(
                nn.LayerNorm(GATE_FEATURES),
                nn.Linear(GATE_FEATURES, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            if arm == "P3_partial_transport"
            else None
        )

    def train(self, mode: bool = True) -> EvidenceCompletionModel:
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, data: PackedScreenData) -> ScreenOutput:
        require_gate = self.arm == "P3_partial_transport"
        data.validate(require_gate=require_gate, check_values=False)
        with torch.no_grad():
            base = self.base(data).prediction
        assembled = self.base.fixed.assemble(data, use_evidence=True)
        candidate = self.decoder(assembled, base)
        available = data.target_valid & data.evidence_available
        if self.gate is None:
            prediction = torch.where(available[..., None], candidate, base)
            gate = available.to(candidate.dtype)
        else:
            frozen_features = data.gate_features.detach().to(candidate.dtype)
            soft_gate = torch.sigmoid(self.gate(frozen_features).squeeze(-1))
            gate = soft_gate * available.to(soft_gate.dtype)
            blended = (1.0 - gate[..., None]) * base + gate[..., None] * candidate
            prediction = torch.where(available[..., None], blended, base)
        return ScreenOutput(prediction, base, candidate, gate, available)


def new_screen_model(
    arm: Arm,
    *,
    base: BaseCompletionModel | None = None,
    seed: int = SCREEN_SEED,
    device: str | torch.device = "cpu",
) -> BaseCompletionModel | EvidenceCompletionModel:
    if arm not in ARMS:
        raise ValueError("unknown reconstruction arm")
    if seed != SCREEN_SEED:
        raise ValueError("the screen permits only seed 42")
    if arm == "P0_center_only" and base is not None:
        raise ValueError("P0 does not accept an ancestor model")
    if arm != "P0_center_only" and not isinstance(base, BaseCompletionModel):
        raise ValueError("P1/P2/P3 require the fitted P0 from the same outer fold")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = (
            BaseCompletionModel(seed)
            if arm == "P0_center_only"
            else EvidenceCompletionModel(base, arm)
        )
    return model.to(device)


def active_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def matched_parameter_counts(seed: int = SCREEN_SEED) -> dict[str, int | float]:
    base = new_screen_model("P0_center_only", seed=seed)
    counts: dict[str, int | float] = {"P0_center_only": active_parameter_count(base)}
    for arm in ARMS[1:]:
        counts[arm] = active_parameter_count(new_screen_model(arm, base=base, seed=seed))
    active = [int(counts[arm]) for arm in ARMS[1:]]
    counts["P1_P3_relative_range"] = (max(active) - min(active)) / min(active)
    if counts["P1_P3_relative_range"] > 0.10:
        raise RuntimeError("P1/P2/P3 active parameter counts differ by more than 10 percent")
    return counts


def center_mean(values: torch.Tensor, target_valid: torch.Tensor) -> torch.Tensor:
    """Average target contributions per center using the fixed target denominator."""

    if values.shape != target_valid.shape or target_valid.dtype != torch.bool:
        raise ValueError("center-mean values and target population differ")
    denominator = target_valid.sum(dim=1)
    if (denominator < 1).any():
        raise ValueError("a center has no fixed target denominator")
    contributions = torch.where(target_valid, values, torch.zeros_like(values))
    return contributions.sum(dim=1) / denominator


def reconstruction_loss(
    output: ScreenOutput,
    teacher_tokens: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    huber_weight: float = 0.25,
    intervention_weight: float = 0.01,
) -> LossTerms:
    if output.prediction.shape != teacher_tokens.shape or teacher_tokens.shape[-1] != FEATURE_DIM:
        raise ValueError("prediction and teacher geometry differ")
    teacher = teacher_tokens.detach().to(output.prediction.dtype)
    cosine = 1.0 - F.cosine_similarity(output.prediction, teacher, dim=-1, eps=1e-12)
    predicted_ln = F.layer_norm(output.prediction, (FEATURE_DIM,))
    teacher_ln = F.layer_norm(teacher, (FEATURE_DIM,))
    huber = F.smooth_l1_loss(predicted_ln, teacher_ln, reduction="none", beta=1.0).mean(-1)
    per_center_cosine = center_mean(cosine, target_valid)
    per_center_huber = center_mean(huber, target_valid)
    per_center_gate = center_mean(output.gate, target_valid)
    objective = (
        per_center_cosine.mean()
        + huber_weight * per_center_huber.mean()
        + intervention_weight * per_center_gate.mean()
    )
    return LossTerms(
        objective,
        per_center_cosine.mean(),
        per_center_huber.mean(),
        per_center_gate.mean(),
        per_center_cosine,
        per_center_huber,
    )


def outer_scenario_splits(
    scenarios: Sequence[str] | np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    values = np.asarray(scenarios).astype(str)
    if values.ndim != 1 or set(values.tolist()) != set(EXPECTED_SCENARIOS):
        raise ValueError("screen population must contain all and only the 11 locked scenarios")
    folds = np.array(
        [int(EXPECTED_SCENARIOS[scenario][0].split("-")[1]) for scenario in values],
        dtype=np.int64,
    )
    result = []
    for fold in range(5):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if not len(train) or not len(held) or set(values[train]) & set(values[held]):
            raise RuntimeError("outer scenario split leaked or became empty")
        result.append((train, held))
    return tuple(result)


def fit_outer_fold(
    data: PackedScreenData,
    scenarios: Sequence[str] | np.ndarray,
    held_fold: int,
    *,
    arm: Arm,
    fitted_p0: BaseCompletionModel | None = None,
    updates: int = MAX_UPDATES,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    seed: int = SCREEN_SEED,
    device: str | torch.device = "cpu",
) -> FitResult:
    """Fit only four-fold reconstruction targets; held targets are never read."""

    if held_fold not in range(5):
        raise ValueError("held fold must be 0..4")
    if seed != SCREEN_SEED or not 1 <= updates <= MAX_UPDATES:
        raise ValueError("screen permits seed 42 and at most 400 updates")
    if batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid optimizer contract")
    if data.centers != len(scenarios):
        raise ValueError("scenario identities and screen rows differ")
    train_rows, held_rows = outer_scenario_splits(scenarios)[held_fold]
    training = data.select(train_rows)
    training.validate(require_gate=arm == "P3_partial_transport")
    model = new_screen_model(arm, base=fitted_p0, seed=seed, device=device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = np.random.default_rng(seed)
    history = []
    order = np.empty(0, dtype=np.int64)
    cursor = 0
    for _ in range(updates):
        if cursor >= len(order):
            order = generator.permutation(training.centers)
            cursor = 0
        rows = order[cursor : cursor + batch_size]
        cursor += len(rows)
        batch = training.select(rows).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = reconstruction_loss(output, batch.teacher_tokens, batch.target_valid)
        if not torch.isfinite(loss.objective):
            raise RuntimeError("reconstruction loss became nonfinite")
        loss.objective.backward()
        optimizer.step()
        history.append(float(loss.objective.detach().cpu()))
    model.eval()
    return FitResult(model, tuple(history), train_rows, held_rows, seed, updates)


@torch.no_grad()
def evaluate_screen(
    model: BaseCompletionModel | EvidenceCompletionModel,
    data: PackedScreenData,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray | float]:
    data.validate(require_gate=getattr(model, "arm", None) == "P3_partial_transport")
    model.to(device).eval()
    batch = data.to(device)
    output = model(batch)
    loss = reconstruction_loss(output, batch.teacher_tokens, batch.target_valid)
    denominator = batch.target_valid.sum().item()
    coverage = float((output.evidence_available & batch.target_valid).sum().item() / denominator)
    return {
        "mean_target_token_cosine_error": float(loss.per_center_cosine.mean().cpu()),
        "layernorm_huber_error": float(loss.per_center_huber.mean().cpu()),
        "mean_intervention": float(loss.intervention_cost.cpu()),
        "coverage": coverage,
        "dustbin_rate": 1.0 - coverage,
        "per_center_cosine_error": loss.per_center_cosine.cpu().numpy(),
        "per_center_layernorm_huber": loss.per_center_huber.cpu().numpy(),
    }


def scenario_metrics(
    evaluation: dict[str, np.ndarray | float],
    scenarios: Sequence[str] | np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Aggregate already-held center metrics by the locked scenario IDs."""

    identities = np.asarray(scenarios).astype(str)
    cosine = np.asarray(evaluation["per_center_cosine_error"], dtype=np.float64)
    huber = np.asarray(evaluation["per_center_layernorm_huber"], dtype=np.float64)
    if identities.shape != cosine.shape or huber.shape != cosine.shape:
        raise ValueError("scenario identities and per-center metrics differ")
    if not set(identities).issubset(EXPECTED_SCENARIOS):
        raise ValueError("an evaluation scenario is outside the locked population")
    return {
        scenario: {
            "centers": int(np.sum(identities == scenario)),
            "mean_target_token_cosine_error": float(np.mean(cosine[identities == scenario])),
            "layernorm_huber_error": float(np.mean(huber[identities == scenario])),
        }
        for scenario in sorted(set(identities.tolist()))
    }


def relative_cosine_error_reduction(
    reference: dict[str, np.ndarray | float],
    candidate: dict[str, np.ndarray | float],
) -> float:
    reference_error = float(reference["mean_target_token_cosine_error"])
    candidate_error = float(candidate["mean_target_token_cosine_error"])
    if not np.isfinite([reference_error, candidate_error]).all() or reference_error <= 0:
        raise ValueError("relative reduction requires finite positive reference error")
    return (reference_error - candidate_error) / reference_error


def evaluate_p3_controls(
    model: EvidenceCompletionModel,
    nominal: PackedScreenData,
    *,
    sample_ids: Sequence[str],
    repeated_gate_features: torch.Tensor,
    repeated_available: torch.Tensor,
    wrong_track: PackedScreenData | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, dict[str, np.ndarray | float]]:
    """Evaluate the locked P3 controls through one unchanged fitted checkpoint."""

    if model.arm != "P3_partial_transport":
        raise ValueError("the transport controls require a fitted P3 model")
    views = {
        "nominal": nominal,
        "repeated_masked_center": repeated_masked_center_control(
            nominal,
            repeated_gate_features=repeated_gate_features,
            repeated_available=repeated_available,
        ),
        "spatial_reassignment": spatial_reassignment_control(nominal, sample_ids),
        "time_reversal": time_reversal_control(nominal),
    }
    if wrong_track is not None:
        views["wrong_track"] = wrong_track_control(nominal, wrong_track)
    return {name: evaluate_screen(model, view, device=device) for name, view in views.items()}


def repeated_masked_center_control(
    data: PackedScreenData,
    *,
    repeated_gate_features: torch.Tensor,
    repeated_available: torch.Tensor,
) -> PackedScreenData:
    """Replace P3 donors with masked-center copies and recomputed control metadata.

    Requiring independently constructed gate inputs prevents true-track affine
    confidence from leaking into this negative control.
    """

    data.validate(require_gate=True)
    if repeated_gate_features.shape != data.gate_features.shape:
        raise ValueError("repeated-donor control requires recomputed N,T,21 gate features")
    if (
        repeated_available.shape != data.evidence_available.shape
        or repeated_available.dtype != torch.bool
    ):
        raise ValueError("repeated-donor control availability must have shape N,T and bool dtype")
    if not torch.isfinite(repeated_gate_features).all():
        raise ValueError("repeated-donor gate features must be finite")
    temporal = data.temporal_metadata.clone()
    donor_valid = repeated_gate_features[..., 1:17].reshape(
        *repeated_gate_features.shape[:-1], DONOR_COUNT, 4
    )[..., 0]
    temporal[..., DONOR_COUNT:] = donor_valid
    return replace(
        data,
        evidence_tokens=data.masked_tokens.clone(),
        evidence_available=repeated_available.clone(),
        temporal_metadata=temporal,
        gate_features=repeated_gate_features.detach().clone(),
    )


def spatial_reassignment_control(
    data: PackedScreenData,
    sample_ids: Sequence[str],
    *,
    salt: str = "center-completion-spatial-reassignment-v1",
) -> PackedScreenData:
    """Hash-rotate complete transported evidence packets within crop/mask."""

    data.validate(require_gate=True)
    if len(sample_ids) != data.centers or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("spatial control requires one unique sample ID per center")
    evidence = data.evidence_tokens.clone()
    available = data.evidence_available.clone()
    temporal = data.temporal_metadata.clone()
    gate = data.gate_features.clone()
    for center, sample_id in enumerate(sample_ids):
        for mask_id in range(MASK_COUNT):
            indices = torch.nonzero(
                data.target_valid[center] & (data.mask_ids[center] == mask_id),
                as_tuple=False,
            ).flatten()
            if len(indices) < 2:
                continue
            keys = []
            for index in indices.tolist():
                y, x = data.positions_yx[center, index].tolist()
                payload = f"{salt}|{sample_id}|{mask_id}|{y}|{x}".encode()
                keys.append(hashlib.sha256(payload).digest())
            order = torch.tensor(
                np.argsort(np.asarray(keys, dtype="S32")),
                dtype=torch.int64,
                device=indices.device,
            )
            ordered = indices[order]
            source = torch.roll(ordered, shifts=1)
            evidence[center, ordered] = data.evidence_tokens[center, source]
            available[center, ordered] = data.evidence_available[center, source]
            temporal[center, ordered] = data.temporal_metadata[center, source]
            gate[center, ordered] = data.gate_features[center, source]
    return replace(
        data,
        evidence_tokens=evidence,
        evidence_available=available,
        temporal_metadata=temporal,
        gate_features=gate,
    )


def time_reversal_control(data: PackedScreenData) -> PackedScreenData:
    """Reverse offset/confidence metadata while leaving feature tensors untouched."""

    data.validate(require_gate=True)
    temporal = data.temporal_metadata.clone()
    temporal[..., :4] = torch.flip(data.temporal_metadata[..., :4], dims=(-1,))
    temporal[..., 4:] = torch.flip(data.temporal_metadata[..., 4:], dims=(-1,))
    gate = data.gate_features.clone()
    blocks = data.gate_features[..., 1:17].reshape(*data.gate_features.shape[:-1], 4, 4)
    gate[..., 1:17] = torch.flip(blocks, dims=(-2,)).flatten(-2)
    gate[..., 19] = data.gate_features[..., 20]
    gate[..., 20] = data.gate_features[..., 19]
    return replace(data, temporal_metadata=temporal, gate_features=gate)


def wrong_track_control(nominal: PackedScreenData, wrong: PackedScreenData) -> PackedScreenData:
    """Substitute a precomputed fixed-map wrong-track view without fitting."""

    nominal.validate(require_gate=True)
    wrong.validate(require_gate=True)
    immutable = (
        "masked_tokens",
        "visible_context",
        "teacher_tokens",
        "positions_yx",
        "mask_ids",
        "target_valid",
    )
    if any(not torch.equal(getattr(nominal, name), getattr(wrong, name)) for name in immutable):
        raise ValueError("wrong-track view changed the center target population")
    return replace(
        nominal,
        evidence_tokens=wrong.evidence_tokens,
        evidence_available=wrong.evidence_available,
        temporal_metadata=wrong.temporal_metadata,
        gate_features=wrong.gate_features,
    )
