"""Small, auditable heads for the body-witness experiment.

The predictive decoder is intentionally blind to the withheld RGB region.  Its
caller may pass only features encoded from the complementary visible pixels,
geometric crop metadata, acquisition-valid bits, a spatial-mask identifier and
one of the two posture hypotheses.  Pose heatmaps are targets, never decoder
inputs.  The final reader may consume their *precomputed* predictive energies.

This module does not choose interventions.  ``make_pair_only_candidate`` is the
only deployment transformation here: it can redistribute sitting/standing mass
on a common, observation-only eligibility mask, or copy ARFTR exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POSTURE_CLASSES = 2
LANDMARKS = 8
SPATIAL_BINS = 16 * 12 + 1


def _positive_dimension(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class ConditionalWitnessDecoder(nn.Module):
    """Shared width-64 decoder for categorical hidden-region pose mass.

    The observation axis can represent any fixed combination of crop extent and
    spatial mask. ``mask_ids`` identifies the upper/lower partition.  Setting
    ``condition_on_hypothesis=False`` supplies the prespecified class-independent
    control without changing the module shapes.
    """

    def __init__(
        self,
        visible_feature_dim: int,
        *,
        transform_dim: int = 6,
        validity_dim: int = 1,
        width: int = 64,
        hypothesis_dim: int = 8,
        mask_dim: int = 4,
        landmarks: int = LANDMARKS,
        spatial_bins: int = SPATIAL_BINS,
        condition_on_hypothesis: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visible_feature_dim = _positive_dimension(visible_feature_dim, "visible_feature_dim")
        self.transform_dim = _positive_dimension(transform_dim, "transform_dim")
        self.validity_dim = _positive_dimension(validity_dim, "validity_dim")
        self.width = _positive_dimension(width, "width")
        self.landmarks = _positive_dimension(landmarks, "landmarks")
        self.spatial_bins = _positive_dimension(spatial_bins, "spatial_bins")
        if hypothesis_dim != 8:
            raise ValueError("The locked posture-hypothesis embedding is 8-dimensional")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.condition_on_hypothesis = bool(condition_on_hypothesis)
        self.hypothesis_embedding = nn.Embedding(POSTURE_CLASSES, hypothesis_dim)
        self.mask_embedding = nn.Embedding(2, mask_dim)
        input_dim = (
            self.visible_feature_dim
            + self.transform_dim
            + self.validity_dim
            + hypothesis_dim
            + mask_dim
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, self.landmarks * self.spatial_bins),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        visible_features: torch.Tensor,
        mask_ids: torch.Tensor,
        crop_transforms: torch.Tensor,
        predictor_validity: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits shaped ``[batch, observations, 2, landmarks, bins]``."""

        if visible_features.ndim != 3 or visible_features.shape[-1] != self.visible_feature_dim:
            raise ValueError("visible_features must have shape [batch, observations, feature_dim]")
        batch, observations, _ = visible_features.shape
        if crop_transforms.shape != (batch, observations, self.transform_dim):
            raise ValueError("crop_transforms do not match the decoder contract")
        if predictor_validity.shape != (batch, observations, self.validity_dim):
            raise ValueError("predictor_validity does not match the decoder contract")
        if mask_ids.ndim == 1:
            if mask_ids.shape != (observations,):
                raise ValueError("one-dimensional mask_ids must match observations")
            mask_ids = mask_ids[None].expand(batch, -1)
        if mask_ids.shape != (batch, observations) or mask_ids.dtype != torch.long:
            raise ValueError("mask_ids must be long [observations] or [batch, observations]")
        tensors = (visible_features, crop_transforms, predictor_validity, mask_ids)
        if any(value.device != visible_features.device for value in tensors):
            raise ValueError("all decoder inputs must share a device")
        if torch.any((mask_ids < 0) | (mask_ids > 1)):
            raise ValueError("mask_ids must be upper/lower identifiers 0 or 1")
        for value, name in (
            (visible_features, "visible_features"),
            (crop_transforms, "crop_transforms"),
            (predictor_validity, "predictor_validity"),
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite floating point")

        dtype = self.decoder[1].weight.dtype
        visible_features = visible_features.to(dtype)
        crop_transforms = crop_transforms.to(dtype)
        predictor_validity = predictor_validity.to(dtype)
        mask_embedding = self.mask_embedding(mask_ids)
        hypotheses = torch.arange(POSTURE_CLASSES, device=visible_features.device)
        hypothesis_embedding = self.hypothesis_embedding(hypotheses)
        if not self.condition_on_hypothesis:
            hypothesis_embedding = hypothesis_embedding * 0.0

        shared = torch.cat(
            (visible_features, crop_transforms, predictor_validity, mask_embedding), dim=-1
        )
        shared = shared[:, :, None, :].expand(-1, -1, POSTURE_CLASSES, -1)
        hypothesis_embedding = hypothesis_embedding[None, None].expand(
            batch, observations, -1, -1
        )
        inputs = torch.cat((shared, hypothesis_embedding), dim=-1)
        logits = self.decoder(inputs)
        return logits.reshape(
            batch, observations, POSTURE_CLASSES, self.landmarks, self.spatial_bins
        )


class SharedTemplateDecoder(nn.Module):
    """SEAR-like control: class/mask templates without visible-pixel prediction."""

    def __init__(
        self,
        *,
        landmarks: int = LANDMARKS,
        spatial_bins: int = SPATIAL_BINS,
    ) -> None:
        super().__init__()
        self.landmarks = _positive_dimension(landmarks, "landmarks")
        self.spatial_bins = _positive_dimension(spatial_bins, "spatial_bins")
        self.template_logits = nn.Parameter(
            torch.zeros(2, POSTURE_CLASSES, self.landmarks, self.spatial_bins)
        )

    @property
    def trainable_parameters(self) -> int:
        return int(self.template_logits.numel())

    def forward(self, mask_ids: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        if mask_ids.ndim == 1:
            mask_ids = mask_ids[None].expand(batch_size, -1)
        if mask_ids.shape[0] != batch_size or mask_ids.dtype != torch.long:
            raise ValueError("mask_ids must be long [observations] or [batch, observations]")
        if torch.any((mask_ids < 0) | (mask_ids > 1)):
            raise ValueError("mask_ids must contain only 0 and 1")
        return self.template_logits[mask_ids]


def normalized_predictive_energy(
    observed_heatmap_mass: torch.Tensor,
    predicted_logits: torch.Tensor,
    landmark_weights: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score both hypotheses against the exact same observed witness.

    Returns ``(energy, available)`` with shapes ``[B,O,2]`` and ``[B,O]``.
    Inputs use an outside-region bin and are required to be normalized categorical
    descriptors. Availability requires total clipped peak mass at least 2.0.
    """

    if observed_heatmap_mass.ndim != 4:
        raise ValueError("observed_heatmap_mass must be [batch, observations, landmarks, bins]")
    batch, observations, landmarks, bins = observed_heatmap_mass.shape
    if predicted_logits.shape != (batch, observations, POSTURE_CLASSES, landmarks, bins):
        raise ValueError("predicted_logits do not match the observed witnesses")
    if landmark_weights.shape != (batch, observations, landmarks):
        raise ValueError("landmark_weights must be [batch, observations, landmarks]")
    if bins < 2 or not 0 < epsilon < 1:
        raise ValueError("at least two bins and a valid fixed epsilon are required")
    if any(
        value.device != observed_heatmap_mass.device
        for value in (predicted_logits, landmark_weights)
    ):
        raise ValueError("witness tensors must share a device")
    if not all(
        value.is_floating_point()
        and torch.isfinite(value).all()
        for value in (observed_heatmap_mass, predicted_logits, landmark_weights)
    ):
        raise ValueError("witness tensors must be finite floating point")
    if torch.any(observed_heatmap_mass < 0):
        raise ValueError("observed heatmap mass cannot be negative")
    if not torch.allclose(
        observed_heatmap_mass.sum(-1),
        torch.ones_like(observed_heatmap_mass[..., 0]),
        atol=5 * epsilon,
        rtol=0,
    ):
        raise ValueError("observed heatmap descriptors must sum to one")

    weights = landmark_weights.clamp(0.0, 1.0)
    weight_sum = weights.sum(-1)
    log_probabilities = F.log_softmax(predicted_logits, dim=-1)
    cross_entropy = -(
        observed_heatmap_mass[:, :, None] * log_probabilities
    ).sum(-1)
    numerator = (cross_entropy * weights[:, :, None]).sum(-1)
    energy = numerator / weight_sum.clamp_min(epsilon)[:, :, None]
    energy = energy / float(np.log(bins))
    available = weight_sum >= 2.0
    energy = torch.where(available[:, :, None], energy, 0.0)
    return energy, available


def energy_evidence(energy: torch.Tensor) -> torch.Tensor:
    """Flatten ``[E(sit), E(stand), E(stand)-E(sit)]`` per observation."""

    if energy.ndim != 3 or energy.shape[-1] != POSTURE_CLASSES:
        raise ValueError("energy must be [batch, observations, 2]")
    values = torch.stack((energy[..., 0], energy[..., 1], energy[..., 1] - energy[..., 0]), -1)
    return values.flatten(1)


class PostureReader(nn.Module):
    """Fixed width-64 two-layer binary reader used by all screen arms."""

    def __init__(self, input_dim: int, *, width: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = _positive_dimension(input_dim, "input_dim")
        if width != 64:
            raise ValueError("The locked reader width is 64")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        if self.trainable_parameters >= 1_000_000:
            raise ValueError("The body-witness reader exceeds the one-million parameter cap")

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[-1] != self.input_dim:
            raise ValueError("reader values must be [batch, input_dim]")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("reader inputs must be finite floating point")
        return self.network(values.to(self.network[1].weight.dtype)).squeeze(-1)


class _ExpansionResidual(nn.Module):
    """Useful direct-path capacity used to match lower-information controls."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class ModalityPostureReader(nn.Module):
    """Project available modalities to a shared width and predict posture.

    Modality sets, dimensions and ``extra_residual_blocks`` are protocol fields.
    Extra blocks operate on the real fused direct path; they are not unused
    parameters or zero-filled forbidden inputs.  This permits V1/V2/V3 capacity
    matching while V0 remains an intentionally smaller information control.
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        *,
        extra_residual_blocks: int = 0,
        width: int = 64,
        expansion: int = 512,
        dropout: float = 0.1,
        parameter_limit: int = 1_000_000,
    ) -> None:
        super().__init__()
        if not modality_dims or len(set(modality_dims)) != len(modality_dims):
            raise ValueError("At least one uniquely named modality is required")
        if width != 64 or min(expansion, parameter_limit) < 1 or extra_residual_blocks < 0:
            raise ValueError("Invalid locked modality-reader architecture")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.modality_dims = {
            str(name): _positive_dimension(dimension, f"{name}_dim")
            for name, dimension in modality_dims.items()
        }
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dimension), nn.Linear(dimension, width), nn.GELU()
                )
                for name, dimension in self.modality_dims.items()
            }
        )
        self.blocks = nn.ModuleList(
            _ExpansionResidual(width, expansion, dropout)
            for _ in range(extra_residual_blocks)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 1))
        if self.trainable_parameters >= parameter_limit:
            raise ValueError("The body-witness reader exceeds its parameter cap")

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, modalities: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(modalities) != set(self.modality_dims):
            raise ValueError("Reader modality names differ from its construction contract")
        batch = None
        fused = None
        for name, dimension in self.modality_dims.items():
            values = modalities[name]
            if values.ndim != 2 or values.shape[1] != dimension:
                raise ValueError(f"{name} must have shape [batch,{dimension}]")
            if batch is None:
                batch = values.shape[0]
            if values.shape[0] != batch or not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError("All modalities must share a batch and contain finite floats")
            projected = self.projections[name](values.to(self.output[-1].weight.dtype))
            fused = projected if fused is None else fused + projected
        fused = fused / (len(self.modality_dims) ** 0.5)
        for block in self.blocks:
            fused = block(fused)
        return self.output(fused).squeeze(-1)


def body_witness_training_loss(
    reader_logits: torch.Tensor,
    canonical_labels: torch.Tensor,
    sample_weights: torch.Tensor,
    energy: torch.Tensor,
    observation_available: torch.Tensor,
    *,
    energy_coefficient: float = 0.20,
    prediction_coefficient: float = 0.20,
) -> dict[str, torch.Tensor]:
    """Locked V3 objective for sitting/standing training populations only."""

    if reader_logits.ndim != 1:
        raise ValueError("reader_logits must be one-dimensional")
    batch = reader_logits.shape[0]
    if canonical_labels.shape != (batch,) or canonical_labels.dtype != torch.long:
        raise ValueError("canonical_labels must be long [batch]")
    if sample_weights.shape != (batch,) or not sample_weights.is_floating_point():
        raise ValueError("sample_weights must be floating [batch]")
    if energy.ndim != 3 or energy.shape[0] != batch or energy.shape[-1] != POSTURE_CLASSES:
        raise ValueError("energy must be [batch, observations, 2]")
    if observation_available.shape != energy.shape[:2] or observation_available.dtype != torch.bool:
        raise ValueError("observation_available must be boolean [batch, observations]")
    if torch.any((canonical_labels < 0) | (canonical_labels >= POSTURE_CLASSES)):
        raise ValueError("V3 training accepts only canonical sitting/standing labels")
    if not (
        torch.isfinite(reader_logits).all()
        and torch.isfinite(sample_weights).all()
        and torch.isfinite(energy).all()
    ) or torch.any(sample_weights < 0):
        raise ValueError("loss inputs must be finite with nonnegative sample weights")
    normalization = sample_weights.sum().clamp_min(1e-12)
    sitting_target = (canonical_labels == 0).to(reader_logits.dtype)
    binary = (
        F.binary_cross_entropy_with_logits(reader_logits, sitting_target, reduction="none")
        * sample_weights
    ).sum() / normalization

    counts = observation_available.sum(1)
    row_available = counts > 0
    mean_energy = (
        energy * observation_available[..., None]
    ).sum(1) / counts.clamp_min(1)[:, None]
    valid_weights = sample_weights * row_available.to(sample_weights.dtype)
    class_energy = (
        F.cross_entropy(-mean_energy, canonical_labels, reduction="none") * valid_weights
    ).sum() / normalization
    correct = energy.gather(
        2, canonical_labels[:, None, None].expand(-1, energy.shape[1], 1)
    ).squeeze(-1)
    # The teacher-prediction term uses only class-independent observed
    # reliability already folded into ``energy``.  Average observations within
    # each physical center, then average over *all* training rows.  Unavailable
    # rows contribute zero without changing the denominator.
    prediction = (
        (correct * observation_available).sum(1) / counts.clamp_min(1)
    ).mean()
    total = binary + energy_coefficient * class_energy + prediction_coefficient * prediction
    return {
        "loss": total,
        "binary_bce": binary,
        "energy_class_ce": class_energy,
        "observed_prediction_loss": prediction,
    }


@dataclass(frozen=True)
class PairOnlyCandidate:
    probabilities: np.ndarray
    eligible: np.ndarray
    raw_probabilities: np.ndarray


def _probability_matrix(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be finite three-class probability vectors")
    return values


def make_pair_only_candidate(
    arftr_probabilities: np.ndarray,
    sitting_probability: np.ndarray,
    *,
    acquired_measurement_valid: np.ndarray,
) -> PairOnlyCandidate:
    """Build the diagnostic sitting/standing candidate with exact retain fallback."""

    anchor = _probability_matrix(arftr_probabilities, "arftr_probabilities")
    sitting_probability = np.asarray(sitting_probability)
    acquired_measurement_valid = np.asarray(acquired_measurement_valid)
    if (
        sitting_probability.shape != (len(anchor),)
        or not np.issubdtype(sitting_probability.dtype, np.floating)
        or not np.isfinite(sitting_probability).all()
        or np.any((sitting_probability < 0) | (sitting_probability > 1))
    ):
        raise ValueError("sitting_probability must be finite in [0,1]")
    if acquired_measurement_valid.shape != (len(anchor),) or acquired_measurement_valid.dtype != np.bool_:
        raise ValueError("acquired_measurement_valid must be boolean [rows]")

    mass = anchor[:, 0] + anchor[:, 1]
    raw = np.empty_like(anchor)
    raw[:, 0] = mass * sitting_probability
    raw[:, 1] = mass * (1.0 - sitting_probability)
    raw[:, 2] = anchor[:, 2]
    anchor_class = anchor.argmax(1)
    raw_class = raw.argmax(1)
    eligible = (
        acquired_measurement_valid
        & (anchor_class < POSTURE_CLASSES)
        & (raw_class < POSTURE_CLASSES)
        & (anchor_class != raw_class)
    )
    output = anchor.copy()
    output[eligible] = raw[eligible]
    if not np.array_equal(output[~eligible], anchor[~eligible]):
        raise RuntimeError("retain action failed to copy ARFTR bytes exactly")
    if not np.array_equal(output[:, 2], anchor[:, 2]):
        raise RuntimeError("pair-only candidate changed locomotion probability")
    return PairOnlyCandidate(output, eligible, raw)
