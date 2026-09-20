"""Scheduling and immutable-completion tests for the fresh source-swap M4 arm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import run_okutama_source_swap_memory as runner  # noqa: E402


def workload():
    groups = np.repeat(np.arange(15), 3)
    data = {
        "labels": np.tile(np.arange(3), 15),
        "scenarios": groups.astype(str),
        "folds": groups % 5,
        "sample_ids": np.array([f"row-{i}" for i in range(45)]),
        "boundary_targets": np.zeros((45, 5), np.float32),
        "boundary_valid": np.ones((45, 5), bool),
    }
    protocol = {
        "learning_rates": [0.0003, 0.001],
        "weight_decays": [0.0001, 0.01],
        "inner_seed": 42,
        "outer_seeds": [42, 43, 44],
    }
    return data, protocol


def execute_synthetic_fold(tmp_path, monkeypatch):
    data, protocol = workload()
    calls, requests, evidence = [], [], {}

    class ReadOnlyCache:
        def meta_probabilities(self, rows):
            requests.append(rows.copy())
            return np.full((45, 3), 1 / 3), [{"population": rows.tolist()}]

        def prepare(self, _rows):
            pytest.fail("M4 source stage must never fit a base")

    def fake_fit(
        directory,
        passed_data,
        passed_protocol,
        train,
        held,
        lr,
        wd,
        seed,
        epochs,
        probabilities,
        ancestry,
        execution_hash,
        progress,
    ):
        assert passed_data is data and passed_protocol is protocol
        assert not set(data["scenarios"][train]) & set(data["scenarios"][held])
        assert ancestry == [{"population": train.tolist()}]
        calls.append((train.copy(), held.copy(), lr, wd, seed, epochs))
        epoch = [2, 3, 7][int(directory.name[-1])] if epochs is None else epochs
        outputs = {
            "probabilities": np.full((len(held), 3), 1 / 3),
            **{
                key: np.zeros((len(held), 5))
                for key in runner.FIELDS
                if key not in ("probabilities", "gate")
            },
            "gate": np.zeros(len(held)),
        }
        receipt = {"selected_epoch": epoch, "seconds": 0}
        for name in ("request_guard.json", "receipt.json", "checkpoint.pt", "predictions.npz"):
            runner.immutable_json(directory / name, {"synthetic": True})
        hashes = {str(path): runner.file_sha256(path) for path in directory.iterdir()}
        evidence[str(directory)] = (outputs, receipt, hashes, 0)
        return evidence[str(directory)]

    def fake_validate(directory, *_args, **_kwargs):
        return evidence[str(directory)]

    monkeypatch.setattr(runner, "fit_new", fake_fit)
    monkeypatch.setattr(runner, "validate_fit", fake_validate)
    cache = ReadOnlyCache()
    result = runner.fit_fold(tmp_path, data, cache, protocol, 0, "synthetic-lock", train=True)
    return data, protocol, cache, calls, requests, evidence, result


def test_exact_fifteen_fits_and_training_only_ancestry(tmp_path, monkeypatch):
    data, protocol, cache, calls, requests, _, result = execute_synthetic_fold(
        tmp_path, monkeypatch
    )
    assert len(calls) == 15 and len(requests) == 4
    outer_train, outer_held = np.flatnonzero(data["folds"] != 0), np.flatnonzero(data["folds"] == 0)
    for train, held, _lr, _wd, seed, epochs in calls[:12]:
        assert set(train) < set(outer_train) and set(held) < set(outer_train)
        assert seed == 42 and epochs is None
    for wanted_seed, (train, held, lr, wd, seed, epochs) in zip(
        [42, 43, 44], calls[12:], strict=True
    ):
        np.testing.assert_array_equal(train, outer_train)
        np.testing.assert_array_equal(held, outer_held)
        assert seed == wanted_seed and epochs == 3 and lr == 0.0003 and wd == 0.0001
    assert result[2]["fit_receipts"] == 15
    assert len(result[2]["files_sha256"]) == 61
    assert result[1]["probabilities"].shape == (3, len(outer_held), 3)
    checked = runner.fit_fold(tmp_path, data, cache, protocol, 0, "synthetic-lock", train=False)
    np.testing.assert_array_equal(checked[1]["probabilities"], result[1]["probabilities"])
    assert len(calls) == 15


def test_missing_inner_evidence_prevents_completed_fold_claim(tmp_path, monkeypatch):
    data, protocol, cache, _, _, evidence, _ = execute_synthetic_fold(tmp_path, monkeypatch)
    directory = tmp_path / "models/survival_memory/fold-0/config-2/inner-1"
    del evidence[str(directory)]
    with pytest.raises(KeyError):
        runner.fit_fold(tmp_path, data, cache, protocol, 0, "synthetic-lock", train=False)


def test_altered_inner_selected_epoch_prevents_selection_reconstruction(tmp_path, monkeypatch):
    data, protocol, cache, _, _, evidence, _ = execute_synthetic_fold(tmp_path, monkeypatch)
    directory = tmp_path / "models/survival_memory/fold-0/config-0/inner-0"
    evidence[str(directory)][1]["selected_epoch"] = 29
    with pytest.raises(RuntimeError, match="Selection not reproduced"):
        runner.fit_fold(tmp_path, data, cache, protocol, 0, "synthetic-lock", train=False)


def test_unexpected_receipt_prevents_exact_budget_claim(tmp_path, monkeypatch):
    data, protocol, cache, _, _, _, _ = execute_synthetic_fold(tmp_path, monkeypatch)
    runner.immutable_json(tmp_path / "models/survival_memory/fold-0/extra-fit/receipt.json", {})
    with pytest.raises(RuntimeError, match="exactly15"):
        runner.fit_fold(tmp_path, data, cache, protocol, 0, "synthetic-lock", train=False)


def test_legacy_strata_are_not_replaced_by_dense_or_sampled_masks():
    data, _ = workload()
    data["quality"] = np.zeros((45, 6), np.float32)
    for prefix in ("node_support", "memory_support", "node_interval", "memory_interval"):
        data[prefix + "_complete"] = np.ones(45, bool)
        data[prefix + "_boundary"] = np.zeros(45, bool)
    legacy = {"complete_target_boundary": np.arange(45) % 2 == 0}
    masks = {"historical_short_fallback": np.arange(45) % 3 == 0}
    strata = runner.comparison_strata(data, legacy, masks)
    np.testing.assert_array_equal(
        strata["complete_target_boundary"], legacy["complete_target_boundary"]
    )
    assert not strata["node_interval_complete_boundary"].any()
    assert strata["node_interval_complete_stable"].all()


def test_comparison_names_reference_without_mislabeling_it_p6():
    labels = np.arange(3)
    p = np.eye(3)
    reference = p[[1, 0, 2]]
    result = runner.compare_source(
        labels, p, reference, np.array([True, False, False]), np.ones(3, bool)
    )
    assert "p6_metrics" not in result and result["reference_errors"] == 2
    assert result["rescues"] == 2 and result["harms"] == 0
    assert result["historical_shared_P6_failures"] == 1
