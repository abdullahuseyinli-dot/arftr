"""Training/selection plumbing on synthetic data only."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hac.actor_memory_training import MemoryBatches, fit_memory


@pytest.fixture
def inputs():
    torch.set_num_threads(2)
    rng = np.random.default_rng(73)
    rows = 36
    labels = np.tile(np.arange(3), rows // 3)
    groups = np.repeat(np.arange(6), 6)
    neighbors = np.arange(rows)[:, None] + np.arange(-2, 3)[None, :]
    valid = (neighbors >= 0) & (neighbors < rows)
    valid &= groups[neighbors.clip(0, rows - 1)] == groups[:, None]
    neighbors[~valid] = -1
    previous = np.full_like(neighbors, -1)
    target = np.zeros_like(neighbors, dtype=np.float32)
    for row in range(rows):
        observed = np.flatnonzero(valid[row])
        previous[row, observed[1:]] = observed[:-1]
        target[row, observed[1:]] = (
            labels[neighbors[row, observed[1:]]] != labels[neighbors[row, observed[:-1]]]
        )
    data = {
        "features": rng.normal(size=(rows, 12)).astype(np.float32),
        "labels": labels,
        "scenarios": groups.astype(str),
        "sample_ids": np.asarray([f"sample-{i}" for i in range(rows)]),
        "neighbor_indices": neighbors,
        "valid": valid,
        "times": np.broadcast_to(np.arange(-2, 3, dtype=np.float32), (rows, 5)).copy(),
        "boundary_targets": target,
        "boundary_valid": previous >= 0,
        "edge_source_slot": previous,
    }
    protocol = {
        "width": 8,
        "layers": 1,
        "heads": 2,
        "dropout": 0.1,
        "max_parameters": 1000000,
        "max_epochs": 2,
        "batch_size": 8,
        "early_stopping_patience": 1,
        "gradient_clip_norm": 1.0,
        "boundary_loss_weight": 0.2,
        "gate_square_penalty": 0.001,
    }
    return data, rng.dirichlet(np.ones(3), rows), np.arange(24), np.arange(24, 36), protocol


def run_fit(inputs, directory, *, arm="survival_memory", epochs=2):
    data, probabilities, train, held, protocol = inputs
    return fit_memory(
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


@pytest.mark.parametrize(
    "arm", ["temporal_conv", "query_attention", "survival_memory", "corroborated_memory"]
)
def test_fit_is_finite_and_resumes_exactly(inputs, tmp_path, arm):
    probabilities, receipt = run_fit(inputs, tmp_path / arm, arm=arm)
    replay, replay_receipt = run_fit(inputs, tmp_path / arm, arm=arm)
    np.testing.assert_array_equal(probabilities, replay)
    assert receipt == replay_receipt
    assert probabilities.dtype == np.float64
    assert receipt["selected_epoch"] == 2
    assert receipt["request"]["selection_labels_sha256"] is None
    assert np.isfinite([value["training_loss"] for value in receipt["history"]]).all()


def test_outer_labels_cannot_change_fitted_weights(inputs, tmp_path):
    run_fit(inputs, tmp_path / "first")
    data, p, train, held, protocol = inputs
    changed = {**data, "labels": data["labels"].copy()}
    changed["labels"][held] = (changed["labels"][held] + 1) % 3
    run_fit((changed, p, train, held, protocol), tmp_path / "second")
    first = torch.load(tmp_path / "first/checkpoint.pt", weights_only=False)["model"]
    second = torch.load(tmp_path / "second/checkpoint.pt", weights_only=False)["model"]
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_inner_epoch_selection_and_outer_scaler_are_isolated(inputs, tmp_path):
    _, receipt = run_fit(inputs, tmp_path / "selection", epochs=None)
    assert 1 <= receipt["selected_epoch"] <= 2
    assert receipt["request"]["selection_labels_sha256"] is not None
    data, p, train, held, _ = inputs
    batches = MemoryBatches(data, p, train, device="cpu")
    changed = {**data, "features": data["features"].copy()}
    changed["features"][held] += 1000
    poisoned = MemoryBatches(changed, p, train, device="cpu")
    np.testing.assert_array_equal(batches.scaler.mean_, poisoned.scaler.mean_)
    torch.testing.assert_close(batches.features[train], poisoned.features[train], rtol=0, atol=0)


def test_changed_fit_or_tampered_prediction_is_rejected(inputs, tmp_path):
    run_fit(inputs, tmp_path)
    with pytest.raises(RuntimeError, match="different memory fit"):
        run_fit(inputs, tmp_path, arm="query_attention")
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output_sha256"]["predictions.npz"] = "changed"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="output changed"):
        run_fit(inputs, tmp_path)
