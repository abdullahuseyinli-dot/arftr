import numpy as np
import pytest
import torch

from hac.evidence_decomposition import (
    ARM_NAMES,
    READER_INPUT_DIM,
    EvidenceUtilityReader,
    actor_roi_bin_map,
    apply_standardizer,
    bilinear_sample_grid,
    compose_arm_features,
    decode_factor_logits,
    decompose_evidence,
    fit_standardizer,
    phase_only_coordinates,
    weighted_factor_nll,
)


def test_roi_bins_use_true_patch_centers_and_actor_box():
    transform = np.eye(3)
    valid = np.ones((27, 27), dtype=bool)
    bins = actor_roi_bin_map(transform, (0.0, 0.0, 378.0, 378.0), valid)
    assert set(np.unique(bins)) == set(range(6))
    assert bins[0, 0] == 0 and bins[0, -1] == 1
    assert bins[-1, 0] == 4 and bins[-1, -1] == 5
    valid[0, 0] = False
    assert actor_roi_bin_map(transform, (0, 0, 378, 378), valid)[0, 0] == -1


def test_phase_formula_is_signed_around_nearest_integer():
    mapped = np.zeros((27, 27, 2), dtype=np.float64)
    y, x = np.mgrid[:27, :27]
    mapped[..., 0] = x + 1.75
    mapped[..., 1] = y - 2.20
    phase = phase_only_coordinates(mapped)
    assert np.allclose(phase[..., 0], x - 0.25)
    assert np.allclose(phase[..., 1], y - 0.20)


def test_bilinear_sampler_requires_all_four_support_cells():
    tokens = np.arange(27 * 27, dtype=np.float32).reshape(27, 27, 1)
    valid = np.ones((27, 27), dtype=bool)
    coordinates = np.zeros((27, 27, 2), dtype=np.float64)
    coordinates[..., :] = (3.5, 4.5)
    values, observed = bilinear_sample_grid(tokens, valid, coordinates)
    assert observed.all()
    assert np.allclose(
        values[..., 0], (tokens[4, 3, 0] + tokens[4, 4, 0] + tokens[5, 3, 0] + tokens[5, 4, 0]) / 4
    )
    valid[5, 4] = False
    values, observed = bilinear_sample_grid(tokens, valid, coordinates)
    assert not observed.any() and np.all(values == 0)


def test_decomposition_identity_has_shared_coverage_and_zero_contrasts():
    rng = np.random.default_rng(7)
    center = rng.normal(size=(27, 27, 16)).astype(np.float32)
    neighbors = np.repeat(center[None], 4, axis=0)
    valid = np.ones((27, 27), dtype=bool)
    bins = np.repeat(np.arange(6), 122)[:729].reshape(27, 27).astype(np.int8)
    result = decompose_evidence(center, neighbors, valid, np.repeat(valid[None], 4, 0), bins)
    assert result.transform_valid.all()
    assert np.allclose(
        result.transform_matrix, np.repeat(np.array([[[1, 0, 0], [0, 1, 0]]]), 4, 0), atol=1e-6
    )
    assert np.all(result.contrasts == 0)
    # Least-squares identity is accurate to floating precision, but tiny negative
    # edge coordinates correctly fail the strict four-corner support contract.
    assert np.all(result.coverage > 0.7)
    assert np.all(result.common_target_count <= result.roi_target_count)


def test_reader_features_scaler_factor_decode_and_loss_contracts():
    rng = np.random.default_rng(11)
    context = rng.normal(size=(3, 2304)).astype(np.float32)
    center = rng.normal(size=(3, 6, 768)).astype(np.float32)
    contrasts = rng.normal(size=(3, 4, 6, 768)).astype(np.float32)
    contrasts[:, 0] = 0
    coverage = rng.uniform(size=(3, 6)).astype(np.float32)
    features = compose_arm_features(context, center, contrasts, coverage)
    assert features.shape == (3, len(ARM_NAMES), READER_INPUT_DIM)
    mean, scale = fit_standardizer(features[:, 2])
    standardized = apply_standardizer(features[:, 2], mean, scale)
    assert standardized.dtype == np.float32 and np.isfinite(standardized).all()
    model = EvidenceUtilityReader()
    assert model.trainable_parameters == 737858
    logits = model(torch.from_numpy(standardized))
    probabilities = decode_factor_logits(logits)
    assert torch.allclose(probabilities.sum(1), torch.ones(3), atol=1e-6)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    loss = weighted_factor_nll(logits, labels, torch.ones(3))
    expected = -torch.log(probabilities[torch.arange(3), labels]).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_corruption_does_not_become_missing_evidence():
    center = np.zeros((27, 27, 8), dtype=np.float32)
    neighbors = np.zeros((4, 27, 27, 8), dtype=np.float32)
    neighbors[0, 0, 0, 0] = np.nan
    valid = np.ones((27, 27), dtype=bool)
    bins = np.zeros((27, 27), dtype=np.int8)
    with pytest.raises(ValueError, match="Malformed four-donor"):
        decompose_evidence(center, neighbors, valid, np.repeat(valid[None], 4, 0), bins)
