from __future__ import annotations

import numpy as np
import pytest
import torch

from hac.body_witness import (
    ConditionalWitnessDecoder,
    ModalityPostureReader,
    PostureReader,
    SharedTemplateDecoder,
    body_witness_training_loss,
    energy_evidence,
    make_pair_only_candidate,
    normalized_predictive_energy,
)


def test_pair_only_candidate_has_exact_retain_and_locomotion_invariants():
    anchor = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.70, 0.10],
            [0.20, 0.10, 0.70],
            [0.55, 0.35, 0.10],
        ],
        dtype=np.float64,
    )
    result = make_pair_only_candidate(
        anchor,
        np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float64),
        acquired_measurement_valid=np.array([True, True, True, False]),
    )
    assert result.eligible.tolist() == [True, True, False, False]
    assert np.array_equal(result.probabilities[~result.eligible], anchor[~result.eligible])
    assert np.array_equal(result.probabilities[:, 2], anchor[:, 2])
    assert np.allclose(result.probabilities.sum(1), 1.0)


def test_pair_only_candidate_requires_common_observation_mask():
    anchor = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)
    with pytest.raises(ValueError, match="boolean"):
        make_pair_only_candidate(
            anchor,
            np.array([0.1]),
            acquired_measurement_valid=np.array([1], dtype=np.int64),
        )


def _witness_inputs(batch: int = 3, observations: int = 4, bins: int = 13):
    observed = torch.rand(batch, observations, 8, bins)
    observed /= observed.sum(-1, keepdim=True)
    logits = torch.randn(batch, observations, 2, 8, bins)
    weights = torch.ones(batch, observations, 8)
    return observed, logits, weights


def test_predictive_energy_scores_same_witness_for_both_hypotheses():
    observed, logits, weights = _witness_inputs()
    logits[:, :, 1] = logits[:, :, 0]
    energy, available = normalized_predictive_energy(observed, logits, weights)
    assert energy.shape == (3, 4, 2)
    assert available.all()
    assert torch.equal(energy[..., 0], energy[..., 1])
    evidence = energy_evidence(energy)
    assert evidence.shape == (3, 12)
    assert torch.equal(evidence.reshape(3, 4, 3)[..., 2], torch.zeros(3, 4))


def test_predictive_energy_has_no_free_variance_and_missing_fallback():
    observed, logits, weights = _witness_inputs(batch=2, observations=2)
    weights[0, 0] = 0.2  # total 1.6, below the locked 2.0 threshold
    energy, available = normalized_predictive_energy(observed, logits, weights)
    assert available.tolist() == [[False, True], [True, True]]
    assert torch.equal(energy[0, 0], torch.zeros(2))
    # The categorical head has only logits; there is no learned variance output.
    assert logits.shape[-1] == observed.shape[-1]


def test_conditional_decoder_shapes_and_class_independent_control():
    torch.manual_seed(4)
    visible = torch.randn(2, 4, 16)
    masks = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    transforms = torch.randn(2, 4, 6)
    validity = torch.ones(2, 4, 1)
    direct = ConditionalWitnessDecoder(
        16, landmarks=8, spatial_bins=13, dropout=0.0
    ).eval()
    control = ConditionalWitnessDecoder(
        16,
        landmarks=8,
        spatial_bins=13,
        condition_on_hypothesis=False,
        dropout=0.0,
    ).eval()
    assert direct(visible, masks, transforms, validity).shape == (2, 4, 2, 8, 13)
    output = control(visible, masks, transforms, validity)
    assert torch.equal(output[:, :, 0], output[:, :, 1])


def test_shared_template_is_not_conditioned_on_visible_pixels():
    control = SharedTemplateDecoder(landmarks=8, spatial_bins=13)
    masks = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    output = control(masks, batch_size=3)
    assert output.shape == (3, 4, 2, 8, 13)
    assert torch.equal(output[0], output[2])


def test_locked_reader_and_v3_loss_are_finite_with_all_missing_witnesses():
    reader = PostureReader(20, dropout=0.0)
    logits = reader(torch.randn(4, 20))
    energy = torch.zeros(4, 4, 2)
    available = torch.zeros(4, 4, dtype=torch.bool)
    result = body_witness_training_loss(
        logits,
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
        torch.ones(4),
        energy,
        available,
    )
    assert reader.trainable_parameters < 1_000_000
    assert torch.isfinite(result["loss"])
    assert result["energy_class_ce"].item() == 0.0
    assert result["observed_prediction_loss"].item() == 0.0


def test_v1_v2_v3_useful_capacity_is_matched_within_ten_percent():
    dimensions = {
        "context": 2304,
        "body": 6 * 768,
        "quality": 8 + 4 * 6 + 4,
        "pose": 2 * 8 * 193 + 16,
        "energy": 4 * 3,
    }
    readers = {
        "V1": ModalityPostureReader(
            {key: dimensions[key] for key in ("context", "body", "quality")},
            extra_residual_blocks=5,
            dropout=0.0,
        ),
        "V2": ModalityPostureReader(
            {key: dimensions[key] for key in ("context", "body", "quality", "pose")},
            extra_residual_blocks=2,
            dropout=0.0,
        ),
        "V3": ModalityPostureReader(dimensions, dropout=0.0),
    }
    decoder_parameters = ConditionalWitnessDecoder(768).trainable_parameters
    counts = np.array(
        [
            readers["V1"].trainable_parameters,
            readers["V2"].trainable_parameters,
            readers["V3"].trainable_parameters + decoder_parameters,
        ]
    )
    assert counts.max() / counts.min() - 1 <= 0.10
    values = {
        name: torch.randn(2, dimension) for name, dimension in dimensions.items()
    }
    assert readers["V1"]({key: values[key] for key in ("context", "body", "quality")}).shape == (2,)
    assert readers["V2"]({key: values[key] for key in ("context", "body", "quality", "pose")}).shape == (2,)
    assert readers["V3"](values).shape == (2,)


def test_v3_observed_prediction_loss_is_class_independent_training_row_mean():
    energy = torch.tensor(
        [
            [[0.2, 0.9], [0.4, 0.7]],
            [[0.5, 0.8], [0.1, 0.6]],
        ]
    )
    available = torch.tensor([[True, True], [True, False]])
    result = body_witness_training_loss(
        torch.zeros(2),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1.0, 9.0]),
        energy,
        available,
    )
    # Per-row correct-hypothesis means are 0.3 and 0.8. Class-frequency
    # weights must not reweight this teacher-prediction term.
    assert result["observed_prediction_loss"].item() == pytest.approx(0.55)

    missing = body_witness_training_loss(
        torch.zeros(2),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1.0, 9.0]),
        energy,
        torch.tensor([[True, True], [False, False]]),
    )
    # The unavailable second row contributes zero but remains in the fixed
    # training-row denominator: (0.3 + 0.0) / 2.
    assert missing["observed_prediction_loss"].item() == pytest.approx(0.15)
