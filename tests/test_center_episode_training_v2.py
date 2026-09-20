"""Synthetic-only isolation, loss, gradient and resume tests for episode fitting."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from hac.center_episode_training_v2 import (
    EpisodeBatches,
    clip_episode_gradients,
    fit_episode,
    new_episode_model,
)


@pytest.fixture
def episode_inputs():
    torch.set_num_threads(2)
    rng = np.random.default_rng(931)
    rows = 36
    labels = np.tile(np.arange(3), rows // 3)
    groups = np.repeat(np.arange(6), 6)
    neighbors = np.arange(rows)[:, None] + np.arange(-2, 3)[None, :]
    valid = (neighbors >= 0) & (neighbors < rows)
    valid &= groups[neighbors.clip(0, rows - 1)] == groups[:, None]
    neighbors[~valid] = -1
    previous = np.full_like(neighbors, -1)
    for row in range(rows):
        slots = np.flatnonzero(valid[row])
        previous[row, slots[1:]] = slots[:-1]
    data = {
        "features": rng.normal(size=(rows, 12)).astype(np.float32),
        "labels": labels,
        "scenarios": groups.astype(str),
        "sample_ids": np.array([f"episode-{i}" for i in range(rows)]),
        "neighbor_indices": neighbors,
        "valid": valid,
        "times": np.broadcast_to(np.arange(-2, 3, dtype=np.float32), (rows, 5)).copy(),
        "boundary_targets": (previous >= 0).astype(np.float32),
        "boundary_valid": previous >= 0,
        "edge_source_slot": previous,
    }
    protocol = {
        "arms": ["log_survival_control", "expected_episode"],
        "matched_capacity": {
            "width": 8,
            "hazard_width": 4,
            "layers": 1,
            "heads": 2,
            "dropout": 0.1,
            "parameter_limit": 1000000,
        },
        "max_epochs": 2,
        "batch_size": 8,
        "early_stopping_patience": 1,
        "gradient_clip_norm": 1.0,
        "boundary_loss_weight": 0.2,
        "gate_square_penalty": 0.001,
        "coefficient_bound_tolerance": 1e-6,
    }
    return data, rng.dirichlet(np.ones(3), rows), np.arange(24), np.arange(24, 36), protocol


def execute(inputs, directory, arm="expected_episode", epochs=2):
    data, probabilities, train, held, protocol = inputs
    return fit_episode(
        data,
        probabilities,
        train,
        held,
        arm=arm,
        learning_rate=0.001,
        weight_decay=0.01,
        seed=42,
        protocol=protocol,
        ancestry=[{"synthetic": True}],
        directory=directory,
        epochs=epochs,
        device="cpu",
    )


@pytest.mark.parametrize("arm", ["log_survival_control", "expected_episode"])
def test_episode_fit_and_exact_resume(episode_inputs, tmp_path, arm):
    predicted, receipt = execute(episode_inputs, tmp_path / arm, arm)
    replay, replay_receipt = execute(episode_inputs, tmp_path / arm, arm)
    np.testing.assert_array_equal(predicted, replay)
    assert receipt == replay_receipt
    assert predicted.dtype == np.float64
    assert receipt["boundary_positive_weight"] is None
    assert receipt["request"]["selection_labels_sha256"] is None
    assert not any("validation_metrics" in row for row in receipt["history"])
    if arm == "expected_episode":
        assert max(receipt["coefficient_bound_audit"].values()) <= 1e-6


def test_outer_class_and_boundary_labels_cannot_change_weights(episode_inputs, tmp_path):
    first, receipt = execute(episode_inputs, tmp_path / "original")
    data, probabilities, train, held, protocol = episode_inputs
    changed = copy.deepcopy(data)
    changed["labels"][held] = (changed["labels"][held] + 1) % 3
    changed["boundary_targets"][held] = np.nan
    changed["boundary_valid"][held] = False
    second, second_receipt = execute(
        (changed, probabilities, train, held, protocol), tmp_path / "changed"
    )
    np.testing.assert_array_equal(first, second)
    assert receipt["request"] == second_receipt["request"]
    left = torch.load(tmp_path / "original/checkpoint.pt", weights_only=False)["model"]
    right = torch.load(tmp_path / "changed/checkpoint.pt", weights_only=False)["model"]
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)


def test_unweighted_bce_is_used_even_when_positive_ratio_is_extreme(episode_inputs):
    data, probabilities, train, _, protocol = episode_inputs
    batches = EpisodeBatches(data, probabilities, train, device="cpu")
    assert not hasattr(batches, "positive_weight")
    index = torch.tensor(train[:8])
    output = {
        "probabilities": torch.full((len(index), 3), 1 / 3, dtype=torch.float64),
        "gate": torch.zeros(len(index)),
        "boundary_valid": batches.boundary_valid[index],
        "boundary_left_index": batches.edge_source_slot[index],
        "boundary_logits": torch.zeros((len(index), 5), requires_grad=True),
    }
    primary = F.nll_loss(
        output["probabilities"].float().log(), batches.labels[index], weight=batches.class_weight
    )
    expected = primary + protocol["boundary_loss_weight"] * torch.log(torch.tensor(2.0))
    torch.testing.assert_close(batches.loss(output, index, protocol), expected)


def test_hazard_and_utility_clipping_are_independent(episode_inputs):
    _, _, _, _, protocol = episode_inputs
    model = new_episode_model(12, "expected_episode", protocol, "cpu")
    hazard, utility = list(model.hazard_parameters()), list(model.utility_parameters())
    for parameter in hazard:
        parameter.grad = torch.full_like(parameter, 1000)
    for parameter in utility:
        parameter.grad = torch.full_like(parameter, 0.001)
    before = [parameter.grad.clone() for parameter in utility]
    utility_norm = torch.sqrt(sum(gradient.square().sum() for gradient in before))
    assert utility_norm < 1
    norms = clip_episode_gradients(model, 1)
    assert norms["hazard"] > 1 and norms["utility"] < 1
    for parameter, expected in zip(utility, before, strict=True):
        torch.testing.assert_close(parameter.grad, expected, atol=0, rtol=0)
    assert torch.sqrt(sum(parameter.grad.square().sum() for parameter in hazard)) <= 1.00001


def test_training_scaler_ignores_outer_features_and_inner_selects_epochs(episode_inputs, tmp_path):
    data, p, train, held, _ = episode_inputs
    original = EpisodeBatches(data, p, train, device="cpu")
    changed = copy.deepcopy(data)
    changed["features"][held] += 10000
    other = EpisodeBatches(changed, p, train, device="cpu")
    np.testing.assert_array_equal(original.scaler.mean_, other.scaler.mean_)
    torch.testing.assert_close(original.features[train], other.features[train], atol=0, rtol=0)
    _, receipt = execute(episode_inputs, tmp_path, epochs=None)
    assert receipt["request"]["selection_labels_sha256"] is not None
    assert 1 <= receipt["selected_epoch"] <= 2


def test_changed_request_or_tampered_output_refused(episode_inputs, tmp_path):
    execute(episode_inputs, tmp_path)
    with pytest.raises(RuntimeError, match="different episode fit"):
        execute(episode_inputs, tmp_path, arm="log_survival_control")
    path = tmp_path / "predictions.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="output changed"):
        execute(episode_inputs, tmp_path)


def test_prefit_request_precedes_outputs_and_partial_outputs_are_retained(episode_inputs, tmp_path):
    import json

    _, receipt = execute(episode_inputs, tmp_path)
    assert json.loads((tmp_path / "request.json").read_text()) == receipt["request"]
    prediction_bytes = (tmp_path / "predictions.npz").read_bytes()
    (tmp_path / "receipt.json").unlink()
    with pytest.raises(RuntimeError, match="refusing to overwrite partial"):
        execute(episode_inputs, tmp_path)
    assert (tmp_path / "predictions.npz").read_bytes() == prediction_bytes


def test_foreign_artifacts_without_request_are_not_overwritten(episode_inputs, tmp_path):
    foreign = tmp_path / "checkpoint.pt"
    foreign.write_bytes(b"foreign partial artifact")
    with pytest.raises(RuntimeError, match="lack an immutable pre-fit request"):
        execute(episode_inputs, tmp_path)
    assert foreign.read_bytes() == b"foreign partial artifact"


def test_completed_fit_rejects_extra_inventory(episode_inputs, tmp_path):
    execute(episode_inputs, tmp_path)
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(RuntimeError, match="artifact inventory changed"):
        execute(episode_inputs, tmp_path)


def test_inner_evidence_is_reconstructed_from_actual_saved_predictions(episode_inputs, tmp_path):
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from run_okutama_center_episode_v2 import validate_fit_evidence

    execute(episode_inputs, tmp_path, epochs=None)
    data, _, train, held, protocol = episode_inputs
    args = (
        tmp_path,
        data,
        protocol,
        "expected_episode",
        train,
        held,
        0.001,
        0.01,
        42,
        None,
        [{"synthetic": True}],
    )
    probabilities, receipt = validate_fit_evidence(*args)
    assert probabilities.shape == (len(held), 3)
    receipt["selected_epoch"] = 99
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="selected inner checkpoint"):
        validate_fit_evidence(*args)


def test_cross_scenario_or_missing_training_neighbors_refused(episode_inputs, tmp_path):
    data, p, train, held, protocol = episode_inputs
    with pytest.raises(RuntimeError, match="scenario leakage"):
        execute((data, p, np.append(train, held[0]), held, protocol), tmp_path / "leak")
    with pytest.raises(RuntimeError, match="neighborhood escapes"):
        execute((data, p, train[1:], held, protocol), tmp_path / "escape")


def test_v2_refuses_reusing_v1_neural_checkpoint(episode_inputs, tmp_path):
    from hac.actor_memory_base import file_sha256
    from hac.center_episode_training import fit_episode as fit_v1

    data, probabilities, train, held, protocol = episode_inputs
    fit_v1(
        data,
        probabilities,
        train,
        held,
        arm="expected_episode",
        learning_rate=0.001,
        weight_decay=0.01,
        seed=42,
        protocol=protocol,
        ancestry=[{"synthetic": True}],
        directory=tmp_path,
        epochs=1,
        device="cpu",
    )
    before = {path.name: file_sha256(path) for path in tmp_path.iterdir()}
    with pytest.raises(RuntimeError, match="different episode fit request"):
        execute(episode_inputs, tmp_path, epochs=1)
    assert before == {path.name: file_sha256(path) for path in tmp_path.iterdir()}
