from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hac.generalized_arftr import PopulationInputs, execute_population, parameter_grid


def _data() -> dict[str, np.ndarray]:
    rows = 9
    return {
        "labels": np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        "scenarios": np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3),
        "folds": np.asarray([0] * 6 + [1] * 3),
        "recordings": np.asarray(["r"] * rows),
        "tracks": np.asarray([f"t{i}" for i in range(rows)]),
        "frames": np.arange(rows) * 30,
        "times": np.asarray([[-2, -1, 0, 1, 2]] * rows),
        "neighbor_indices": np.full((rows, 5), -1, dtype=np.int64),
        "valid": np.zeros((rows, 5), dtype=bool),
    }


def _probabilities(rows: int) -> np.ndarray:
    values = np.resize(
        np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]),
        (rows, 3),
    )
    return values.astype(np.float64)


def test_population_executor_is_deterministic_and_scenario_disjoint():
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments/okutama_arftr_protocol.json").read_text()
    )
    assert len(parameter_grid(protocol)) == 300
    train, held = np.arange(6), np.arange(6, 9)
    inputs = PopulationInputs(
        m4_inner=_probabilities(6),
        p6_inner=_probabilities(6),
        a3_inner=_probabilities(6),
        m4_outer=np.stack([_probabilities(3)] * 3),
        p6_outer=_probabilities(3),
        a3_outer=np.stack([_probabilities(3)] * 3),
    )
    first = execute_population(_data(), train, held, inputs, protocol)
    second = execute_population(_data(), train, held, inputs, protocol)
    assert np.array_equal(first.parameters.as_array(), second.parameters.as_array())
    assert np.array_equal(first.seed_probabilities, second.seed_probabilities)
    assert first.seed_probabilities.shape == (3, 3, 3)
    with pytest.raises(ValueError, match="disjoint"):
        execute_population(_data(), train, np.asarray([3, 4, 5]), inputs, protocol)
