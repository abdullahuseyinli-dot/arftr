from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hac.arftr import (
    ARFTRParameters,
    apply_arftr,
    decode_factor_scores,
    exact_track_neighbors,
    factor_scores,
    shuffled_neighbors,
)

ROOT = Path(__file__).resolve().parents[1]


def test_factor_scores_round_trip():
    probabilities = np.array([[0.2, 0.3, 0.5], [0.75, 0.2, 0.05]], dtype=np.float64)
    posture, motion = factor_scores(probabilities)
    assert np.allclose(decode_factor_scores(posture, motion), probabilities, atol=1e-15)


def test_zero_transform_preserves_exact_anchor_bytes_and_dtype():
    m4 = np.array([[0.2, 0.3, 0.5], [0.75, 0.2, 0.05]], dtype=np.float32)
    p6 = np.array([[0.3, 0.4, 0.3], [0.25, 0.5, 0.25]], dtype=np.float64)
    a3 = np.array([[0.4, 0.2, 0.4], [0.4, 0.4, 0.2]], dtype=np.float64)
    result = apply_arftr(
        m4,
        p6,
        a3,
        np.full((2, 2), -1),
        ARFTRParameters(0.0, 0.0, 0.0, 0.0),
    )
    assert result.dtype == m4.dtype
    assert np.array_equal(result, m4)


def _temporal_fixture() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows = 5
    neighbors = np.array(
        [[-1, -1, 0, 1, 2], [-1, 0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, -1], [2, 3, 4, -1, -1]]
    )
    valid = neighbors >= 0
    return (
        {
            "neighbor_indices": neighbors,
            "valid": valid,
            "times": np.tile(np.array([-2, -1, 0, 1, 2], dtype=np.float32), (rows, 1)),
            "frames": np.arange(30, 151, 30, dtype=np.int64),
            "recordings": np.repeat("r", rows),
            "tracks": np.repeat("t", rows),
            "scenarios": np.repeat("s", rows),
            "folds": np.zeros(rows, dtype=np.int64),
        },
        np.arange(rows, dtype=np.int64),
    )


def test_exact_neighbors_use_signed_one_second_edges_only():
    data, rows = _temporal_fixture()
    result = exact_track_neighbors(data, rows)
    assert np.array_equal(
        result,
        np.array([[-1, 1], [0, 2], [1, 3], [2, 4], [3, -1]], dtype=np.int64),
    )
    data["folds"][2] = 1
    with pytest.raises(RuntimeError, match="crosses folds"):
        exact_track_neighbors(data, rows)


def test_shuffle_is_deterministic_within_scenario_and_preserves_counts():
    actual = np.array([[-1, 1], [0, 2], [1, 3], [2, 4], [3, -1], [-1, 6], [5, 7], [6, -1]])
    scenarios = np.array(["a"] * 5 + ["b"] * 3)
    # Three-row scenario is rejected: pseudo-neighbours cannot safely exclude
    # self and both true neighbours.
    with pytest.raises(RuntimeError, match="too small"):
        shuffled_neighbors(actual, scenarios, seed=7)
    actual = actual[:5]
    scenarios = scenarios[:5]
    first = shuffled_neighbors(actual, scenarios, seed=7)
    second = shuffled_neighbors(actual, scenarios, seed=7)
    assert np.array_equal(first, second)
    assert np.array_equal((first >= 0).sum(1), (actual >= 0).sum(1))
    for row in range(len(first)):
        assert row not in first[row]
        assert not set(first[row, first[row] >= 0]) & set(actual[row, actual[row] >= 0])


def test_frozen_protocol_has_exact_grid_and_adaptive_statistics():
    protocol = json.loads((ROOT / "experiments/okutama_arftr_protocol.json").read_text())
    selection = protocol["selection"]
    size = np.prod(
        [
            len(selection["posture_restoration"]),
            len(selection["motion_restoration"]),
            len(selection["a3_motion_residual"]),
            len(selection["temporal_strength"]),
        ]
    )
    assert size == selection["candidates_per_outer_fold"] == 300
    assert protocol["population"]["classes"] == ["sitting", "standing", "walking_running"]
    assert protocol["statistics"]["exact_scenario_swap_assignments"] == 2048
    assert protocol["statistics"]["one_sided_significance_alpha"] == 0.05
    assert protocol["status"] == "adaptive_internal_development_not_independent_confirmation"
