"""The source intervention cannot silently change fitting or resume ancestry."""

import numpy as np
import pytest

from hac.source_swap_data import immutable_json
from hac.source_swap_execution import expected_request, fit_guard, guard_attempt, selection_from


def fixture_data():
    return {
        "labels": np.array([0, 1, 2, 0, 1, 2]),
        "sample_ids": np.array([str(i) for i in range(6)]),
        "scenarios": np.array(["a"] * 3 + ["b"] * 3),
        "boundary_targets": np.zeros((6, 5), np.float32),
        "boundary_valid": np.ones((6, 5), bool),
    }


def test_outer_request_does_not_depend_on_held_labels_or_boundary_targets():
    data = fixture_data()
    train, held = np.arange(3), np.arange(3, 6)
    before = fit_guard(data, {}, train, held, 0.001, 0.01, 42, 3, [{"base": "only_train"}], "lock")
    data["labels"][held] = [2, 2, 2]
    data["boundary_targets"][held] = 1
    after = fit_guard(data, {}, train, held, 0.001, 0.01, 42, 3, [{"base": "only_train"}], "lock")
    assert before == after
    data["labels"][0] = 2
    assert before != fit_guard(
        data, {}, train, held, 0.001, 0.01, 42, 3, [{"base": "only_train"}], "lock"
    )


def test_inner_selection_labels_and_source_ancestry_bound():
    data = fixture_data()
    args = (data, {}, np.arange(3), np.arange(3, 6), 0.001, 0.01, 42, None)
    before = expected_request(*args, [{"cache": "new_source"}])
    data["labels"][3] = 1
    assert before != expected_request(*args, [{"cache": "new_source"}])
    assert expected_request(*args, [{"cache": "new_source"}]) != expected_request(
        *args, [{"cache": "old_source"}]
    )


def test_crossing_scenario_refused():
    data = fixture_data()
    data["scenarios"][3] = "a"
    with pytest.raises(RuntimeError, match="scenario leakage"):
        expected_request(data, {}, np.arange(3), np.arange(3, 6), 0.001, 0.01, 42, 3, [])


def test_partial_attempt_preserved_and_foreign_request_refused(tmp_path):
    directory = tmp_path / "fit"
    expected = {"receipt.json", "checkpoint.pt", "predictions.npz"}
    guard_attempt(directory, {"source": "new"}, complete_files=expected)
    guard_attempt(directory, {"source": "new"}, complete_files=expected)
    with pytest.raises(RuntimeError, match="changed"):
        guard_attempt(directory, {"source": "old"}, complete_files=expected)
    immutable_json(directory / "receipt.json", {"partial": True})
    old_bytes = (directory / "receipt.json").read_bytes()
    with pytest.raises(RuntimeError, match="Partial"):
        guard_attempt(directory, {"source": "new"}, complete_files=expected)
    assert (directory / "receipt.json").read_bytes() == old_bytes


def test_foreign_complete_inventory_requires_prefit_guard(tmp_path):
    expected = {"receipt.json", "checkpoint.pt", "predictions.npz"}
    for name in expected:
        immutable_json(tmp_path / name, {})
    with pytest.raises(RuntimeError, match="Partial or foreign"):
        guard_attempt(tmp_path, {}, complete_files=expected)


def test_selection_matches_original_tie_breaks_and_median_epochs():
    candidates = [
        {
            "config_index": i,
            "learning_rate": lr,
            "weight_decay": wd,
            "inner_metrics": {"macro_f1": f1, "nll": nll},
            "inner_epochs": epochs,
        }
        for i, (lr, wd, f1, nll, epochs) in enumerate(
            [
                (0.001, 0.01, 0.81, 0.4, [2, 3, 9]),
                (0.0003, 0.01, 0.81, 0.4, [1, 5, 8]),
                (0.0003, 0.0001, 0.81, 0.4, [2, 4, 11]),
                (0.0003, 0.0001, 0.8, 0.2, [7, 8, 9]),
            ]
        )
    ]
    result = selection_from(candidates)
    assert result["selected"]["config_index"] == 2
    assert result["refit_epochs"] == 4
    assert result["outer_held_used_for_selection"] is False
