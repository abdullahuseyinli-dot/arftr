from copy import deepcopy

import audit_okutama_correspondence_field as audit
import numpy as np
import pytest
import run_okutama_correspondence_field as runner

from hac.actor_memory_base import canonical_hash


def test_frozen_protocol_and_plan_have_no_training_or_metric_side_effect():
    protocol = runner.read_json(runner.PROTOCOL)
    runner.validate_protocol(protocol)
    run = runner.ROOT / ".runs/research_20260912/correspondence_field_plan_test_only"
    assert not run.exists()
    plan = runner.metadata_plan(protocol, run)
    assert plan["initial_fits"] == 15
    assert plan["total_initial_optimizer_steps"] == 12000
    assert plan["task_labels_read"] == 0
    assert plan["task_metrics_released"] == 0
    assert plan["stage2_status"] == "FORBIDDEN_UNTIL_STAGE1_GATE_PASSES"
    assert not run.exists()


def test_protocol_rejects_changed_update_count():
    protocol = deepcopy(runner.read_json(runner.PROTOCOL))
    protocol["training"]["optimizer_steps_per_fit"] = 799
    with pytest.raises(RuntimeError, match="protocol changed"):
        runner.validate_protocol(protocol)


def test_fit_request_never_reads_or_hashes_held_labels():
    data = {
        "sample_ids": np.asarray(["a", "b", "c", "d", "e", "f"]),
        "labels": np.asarray([1, 2, 1, 2, 99, 99]),
        "folds": np.asarray([1, 1, 2, 2, 0, 0]),
        "scenarios": np.asarray(["s1", "s1", "s2", "s2", "held", "held"]),
    }
    request, train, held = runner._fit_request(
        runner.read_json(runner.PROTOCOL), data, "a" * 64, "c2", 0, 42
    )
    assert train.tolist() == [0, 1, 2, 3]
    assert held.tolist() == [4, 5]
    assert request["train_upright_labels_sha256"] == canonical_hash([1, 2, 1, 2])
    assert request["held_labels_sha256"] is None


def _diagnostic_batch():
    points = np.arange(2 * 15 * 8 * 10, dtype=np.float32).reshape(2, 15, 8, 10)
    valid = np.zeros((2, 15, 8), dtype=bool)
    valid[..., :5] = True
    points[~valid] = 0
    return {
        "context": np.zeros((2, 2304), dtype=np.float32),
        "points": points,
        "point_valid": valid,
        "pair_features": np.ones((2, 15, 5), dtype=np.float32),
        "pair_valid": valid.any(2),
        "midpoint_times": np.broadcast_to(np.linspace(-1, 1, 15, dtype=np.float32), (2, 15)).copy(),
        "summaries": np.zeros((2, 31), dtype=np.float32),
    }


def test_audit_point_permutation_moves_masks_with_tokens_and_is_invertible_as_a_multiset():
    batch = _diagnostic_batch()
    changed = audit._permuted_points(batch)
    for row in range(2):
        for pair in range(15):
            original = sorted(
                map(tuple, batch["points"][row, pair, batch["point_valid"][row, pair]])
            )
            permuted = sorted(
                map(tuple, changed["points"][row, pair, changed["point_valid"][row, pair]])
            )
            assert original == permuted


def test_fixed_pair_reassignment_only_decouples_time_when_requested():
    batch = _diagnostic_batch()
    joint = audit._pair_reassignment(batch, keep_times=False)
    decoupled = audit._pair_reassignment(batch, keep_times=True)
    assert not np.array_equal(joint["points"], batch["points"])
    assert not np.array_equal(joint["midpoint_times"], batch["midpoint_times"])
    assert np.array_equal(decoupled["midpoint_times"], batch["midpoint_times"])
    assert np.array_equal(joint["points"], decoupled["points"])


def test_independent_audit_rejects_nonfinite_or_out_of_range_probabilities():
    with pytest.raises(RuntimeError, match="malformed or nonfinite"):
        audit._checked_probability(np.asarray([0.2, np.nan]), 2)
    with pytest.raises(RuntimeError, match="malformed or nonfinite"):
        audit._checked_probability(np.asarray([0.2, 1.1]), 2)
