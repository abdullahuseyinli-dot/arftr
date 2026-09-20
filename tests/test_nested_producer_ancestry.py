from __future__ import annotations

import numpy as np
import pytest


def assert_recursive_scenario_exclusion(
    evaluation_scenarios: np.ndarray, ancestor_training_scenarios: list[list[str]]
) -> None:
    evaluated = set(np.asarray(evaluation_scenarios).astype(str).tolist())
    for depth, trained in enumerate(ancestor_training_scenarios):
        overlap = evaluated & set(map(str, trained))
        if overlap:
            raise RuntimeError(f"recursive ancestry overlap at depth {depth}: {sorted(overlap)}")


def test_recursive_ancestry_fixture_passes_clean_chain():
    assert_recursive_scenario_exclusion(np.asarray(["held"]), [["a", "b"], ["a"], []])


def test_recursive_ancestry_fixture_detects_indirect_contamination():
    with pytest.raises(RuntimeError, match="depth 1"):
        assert_recursive_scenario_exclusion(
            np.asarray(["held"]), [["a", "b"], ["a", "held"], ["a"]]
        )
