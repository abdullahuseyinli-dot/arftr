"""Behavioral tests for identity, missing-time graph and dense-label safeguards."""

from __future__ import annotations

import numpy as np
import pytest

from hac.actor_memory_data import (
    actual_support_union,
    annotation_target,
    build_neighbor_graph,
    target_interval,
)
from hac.okutama_native_video import Annotation


def annotation(frame: int, actions: tuple[str, ...]) -> Annotation:
    return Annotation(0, (0, 0, 10, 20), frame, False, False, False, actions)


def test_graph_retains_missing_timestamp_and_bridges_it() -> None:
    graph = build_neighbor_graph(
        np.array(["r", "r", "r"]),
        np.array(["0", "0", "0"]),
        np.array([30, 90, 120]),
        np.zeros(3, int),
    )
    assert graph["neighbor_indices"][1].tolist() == [0, -1, 1, 2, -1]
    assert graph["edge_source_slot"][1].tolist() == [-1, -1, 0, 2, -1]
    assert graph["edge_dt_seconds"][1].tolist() == [0, 0, 2, 1, 0]
    assert graph["edge_bridges_missing_slot"][1].tolist() == [False, False, True, False, False]


def test_graph_does_not_join_equal_track_id_between_recordings() -> None:
    graph = build_neighbor_graph(
        np.array(["a", "b"]), np.array(["0", "0"]), np.array([30, 60]), np.array([0, 1])
    )
    assert graph["valid"].sum() == 2
    np.testing.assert_array_equal(graph["neighbor_indices"][:, 2], np.arange(2))


def test_graph_rejects_track_across_folds_and_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="crosses folds"):
        build_neighbor_graph(
            np.array(["a", "a"]), np.array(["0", "0"]), np.array([30, 60]), np.array([0, 1])
        )
    with pytest.raises(ValueError, match="Duplicate"):
        build_neighbor_graph(
            np.array(["a", "a"]), np.array(["0", "0"]), np.array([30, 30]), np.array([0, 0])
        )


def test_dense_target_detects_return_to_same_endpoint_state() -> None:
    stable = {(0, f): annotation(f, ("Standing",) if f != 2 else ("Walking",)) for f in range(5)}
    result = target_interval(stable, 0, 0, 4)
    assert result["complete"] and result["boundary"]
    assert result["labels"][0] == result["labels"][-1]


def test_dense_target_masks_missing_and_ambiguous_frames() -> None:
    stable = {(0, f): annotation(f, ("Standing",)) for f in range(5) if f != 2}
    result = target_interval(stable, 0, 0, 4)
    assert not result["complete"] and result["missing_count"] == 1
    stable[(0, 2)] = annotation(2, ("Standing", "Sitting"))
    result = target_interval(stable, 0, 0, 4)
    assert not result["complete"] and result["ambiguous_count"] == 1
    assert not result["boundary"]


def test_walking_running_collapse_is_not_ambiguous_or_transition() -> None:
    assert annotation_target(annotation(0, ("Walking", "Running"))) == 2
    stable = {(0, 0): annotation(0, ("Walking",)), (0, 1): annotation(1, ("Running",))}
    result = target_interval(stable, 0, 0, 1)
    assert result["complete"] and not result["boundary"]


def test_actual_support_respects_short_fallback_and_union() -> None:
    graph = build_neighbor_graph(
        np.array(["a", "a"]), np.array(["0", "0"]), np.array([60, 90]), np.zeros(2, int)
    )
    short = np.stack([np.arange(52, 68), np.arange(82, 98)])
    long = np.stack([np.arange(28, 92, 4), np.arange(58, 122, 4)])
    support = actual_support_union(short, long, np.array([False, True]), graph)
    actual = support["node_support_frames"][0, support["node_support_valid"][0]]
    np.testing.assert_array_equal(actual, short[0])
    union = np.unique(np.concatenate((short[0], short[1], long[1])))
    actual = support["memory_support_frames"][0, support["memory_support_valid"][0]]
    np.testing.assert_array_equal(actual, union)
    assert support["memory_support_start_frame"][0] == 52
    assert support["memory_support_end_frame"][0] == 118
