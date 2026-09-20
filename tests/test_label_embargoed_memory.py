from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from hac.label_embargoed_memory import (
    _embargo_supervision,
    fit_outer_label_embargoed,
)

ROOT = Path(__file__).resolve().parents[1]


def test_embargo_view_retains_only_training_supervision() -> None:
    data = {
        "labels": np.asarray((0, 1, 2, 0)),
        "boundary_targets": np.asarray((0.0, 1.0, 1.0, 0.0)),
        "boundary_valid": np.asarray((True, True, True, True)),
        "edge_source_slot": np.asarray((1, 2, 3, 4)),
        "features": np.ones((4, 2)),
    }
    result = _embargo_supervision(data, np.asarray((0, 2)))

    assert np.array_equal(result["labels"], np.asarray((0, -1, 2, -1)))
    assert np.array_equal(
        result["boundary_valid"], np.asarray((True, False, True, False))
    )
    assert np.array_equal(result["edge_source_slot"], np.asarray((1, -1, 3, -1)))
    assert result["features"] is data["features"]
    assert np.array_equal(data["labels"], np.asarray((0, 1, 2, 0)))


def test_outer_fit_has_static_held_label_embargo() -> None:
    source = (ROOT / "src/hac/label_embargoed_memory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fit = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "fit_outer_label_embargoed"
    )
    fit_source = ast.get_source_segment(source, fit)
    assert fit_source is not None
    assert 'data["labels"][held]' not in fit_source
    assert "probability_metrics" not in fit_source
    assert '"held_metrics"' not in fit_source


def test_held_labels_cannot_change_outer_artifact(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    rows = 18
    train = np.arange(9)
    held = np.arange(9, 18)
    neighbors = np.repeat(np.arange(rows)[:, None], 5, axis=1)
    data = {
        "features": rng.normal(size=(rows, 12)).astype(np.float32),
        "labels": np.tile(np.arange(3), 6),
        "scenarios": np.asarray(["train"] * 9 + ["held"] * 9),
        "sample_ids": np.asarray([f"sample-{index}" for index in range(rows)]),
        "neighbor_indices": neighbors,
        "valid": np.ones((rows, 5), dtype=bool),
        "times": np.zeros((rows, 5), dtype=np.float32),
        "boundary_targets": np.zeros((rows, 5), dtype=np.float32),
        "boundary_valid": np.zeros((rows, 5), dtype=bool),
        "edge_source_slot": np.full((rows, 5), -1),
    }
    protocol = {
        "width": 8,
        "layers": 1,
        "heads": 2,
        "dropout": 0.1,
        "max_parameters": 1000000,
        "batch_size": 8,
        "gradient_clip_norm": 1.0,
        "boundary_loss_weight": 0.2,
        "gate_square_penalty": 0.001,
    }
    probabilities = rng.dirichlet(np.ones(3), rows)

    first, first_receipt = fit_outer_label_embargoed(
        tmp_path / "first",
        data,
        protocol,
        train,
        held,
        0.001,
        0.01,
        42,
        1,
        probabilities,
        [{"synthetic": True}],
        "lock",
        None,
        device="cpu",
    )
    changed = {**data, "labels": data["labels"].copy()}
    changed["labels"][held] = (changed["labels"][held] + 1) % 3
    second, second_receipt = fit_outer_label_embargoed(
        tmp_path / "second",
        changed,
        protocol,
        train,
        held,
        0.001,
        0.01,
        42,
        1,
        probabilities,
        [{"synthetic": True}],
        "lock",
        None,
        device="cpu",
    )

    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first_receipt["request"] == second_receipt["request"]
    assert first_receipt["outer_prediction_labels_read"] == 0
    assert "held_metrics" not in first_receipt
    first_state = torch.load(
        tmp_path / "first/checkpoint.pt", map_location="cpu", weights_only=False
    )["model"]
    second_state = torch.load(
        tmp_path / "second/checkpoint.pt", map_location="cpu", weights_only=False
    )["model"]
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)
