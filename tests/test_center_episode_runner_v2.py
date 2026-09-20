"""Synthetic runner scheduling, immutable-output and selection contract tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import run_okutama_center_episode_v2 as runner  # noqa: E402


def test_numerical_revision_preserves_learning_contract_and_tolerance():
    import copy

    old = runner.audit.read_json(runner.ROOT / "experiments/okutama_center_episode_protocol.json")
    new = runner.audit.read_json(runner.PROTOCOL)
    assert runner.validate_unchanged_learning_contract(new, old)
    for key, changed in (
        ("coefficient_bound_tolerance", 0.001),
        ("max_epochs", 31),
        ("batch_size", 64),
    ):
        mutated = copy.deepcopy(new)
        mutated[key] = changed
        with pytest.raises(RuntimeError, match="learning contract"):
            runner.validate_unchanged_learning_contract(mutated, old)
    mutated = copy.deepcopy(new)
    mutated["attention"]["maximum_segments"] = 10
    with pytest.raises(RuntimeError, match="declared mechanism"):
        runner.validate_unchanged_learning_contract(mutated, old)


def synthetic_workload():
    groups = np.repeat(np.arange(15), 3)
    data = {
        "labels": np.tile(np.arange(3), 15),
        "scenarios": groups.astype(str),
        "folds": groups % 5,
        "sample_ids": np.array([f"row-{index}" for index in range(45)]),
    }
    protocol = {
        "arms": ["log_survival_control", "expected_episode"],
        "outer_folds": list(range(5)),
        "inner_folds": 3,
        "split_seed": 42,
        "learning_rates": [0.001, 0.0003],
        "weight_decays": [0.01, 0.0001],
        "inner_seed": 42,
        "outer_seeds": [42, 43, 44],
    }
    return data, protocol


def test_fold_schedules_exactly_twelve_inner_and_three_outer_fits(tmp_path):
    data, protocol = synthetic_workload()
    population_requests, calls = [], []

    class ReadOnlyCache:
        def meta_probabilities(self, population):
            population_requests.append(population.copy())
            return np.full((45, 3), 1 / 3), [{"synthetic_population": population.tolist()}]

        def prepare(self, *_args):
            pytest.fail("Episode runner may never fit a base model")

    def fake_fit(_data, _probabilities, train, held, **kwargs):
        calls.append((train.copy(), held.copy(), kwargs))
        assert not set(data["scenarios"][train]) & set(data["scenarios"][held])
        if "epochs" in kwargs:
            selected_epoch = kwargs["epochs"]
        else:
            inner = int(kwargs["directory"].name.removeprefix("inner-"))
            selected_epoch = [1, 2, 2][inner]
        probabilities = np.full((len(held), 3), 1 / 3)
        fit_dir = kwargs["directory"]
        request = {"synthetic": True, "train": train.tolist(), "held": held.tolist()}
        runner.write_immutable(fit_dir / "request.json", request)
        np.savez_compressed(
            fit_dir / "predictions.npz",
            probabilities=probabilities,
            sample_ids=data["sample_ids"][held],
            held_rows=held,
        )
        (fit_dir / "checkpoint.pt").write_bytes(b"synthetic scheduling test only")
        receipt = {
            "selected_epoch": selected_epoch,
            "seconds": 0.0,
            "request": request,
            "output_sha256": {
                name: runner.file_sha256(fit_dir / name)
                for name in ("predictions.npz", "checkpoint.pt")
            },
        }
        runner.write_immutable(fit_dir / "receipt.json", receipt)
        return probabilities, receipt

    runner.fit_fold(
        tmp_path,
        data,
        protocol,
        ReadOnlyCache(),
        "expected_episode",
        0,
        "synthetic-lock",
        fitter=fake_fit,
    )
    assert len(calls) == 15 and len(population_requests) == 4
    outer_train, outer_held = np.flatnonzero(data["folds"] != 0), np.flatnonzero(data["folds"] == 0)
    for train, held, kwargs in calls[:12]:
        assert set(train).issubset(outer_train) and set(held).issubset(outer_train)
        assert "epochs" not in kwargs and kwargs["seed"] == 42
    for seed, (train, held, kwargs) in zip([42, 43, 44], calls[12:], strict=True):
        np.testing.assert_array_equal(train, outer_train)
        np.testing.assert_array_equal(held, outer_held)
        assert kwargs["seed"] == seed and kwargs["epochs"] == 2
        assert kwargs["learning_rate"] == 0.0003 and kwargs["weight_decay"] == 0.0001
    directory = tmp_path / "models/expected_episode/fold-0"
    selection = json.loads((directory / "selection.json").read_text())
    assert selection["outer_held_used_for_selection"] is False
    with np.load(directory / "outer_predictions.npz") as saved:
        np.testing.assert_array_equal(saved["held_rows"], outer_held)
        np.testing.assert_array_equal(saved["probabilities"], saved["seed_probabilities"].mean(0))
        assert saved["seed_probabilities"].shape == (3, len(outer_held), 3)
    marker = json.loads((directory / "outer_complete.json").read_text())
    assert marker["arm"] == "expected_episode" and marker["fold"] == 0
    manifest = json.loads((directory / "completion_manifest.json").read_text())
    assert manifest["fit_receipts"] == 15 and len(manifest["files_sha256"]) == 61

    evidence_calls = []

    def synthetic_validator(fit_dir, *_args):
        evidence_calls.append(fit_dir)
        with np.load(fit_dir / "predictions.npz") as saved:
            predicted = saved["probabilities"]
        return predicted, runner.audit.read_json(fit_dir / "receipt.json")

    reconstructed = runner.validate_selection_evidence(
        directory,
        data,
        protocol,
        "expected_episode",
        0,
        ReadOnlyCache(),
        validator=synthetic_validator,
    )
    assert reconstructed == selection and len(evidence_calls) == 12
    selection["candidates"][0]["inner_metrics"]["macro_f1"] += 0.01
    (directory / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(RuntimeError, match="all 12 inner fits"):
        runner.validate_selection_evidence(
            directory,
            data,
            protocol,
            "expected_episode",
            0,
            ReadOnlyCache(),
            validator=synthetic_validator,
        )
    (directory / "config-0/inner-0/receipt.json").unlink()
    with pytest.raises(RuntimeError, match="exactly 15"):
        runner.completion_manifest(directory, protocol, "expected_episode", 0, "synthetic-lock")


def test_selection_tie_break_and_rounded_median():
    candidates = [
        {
            "learning_rate": lr,
            "weight_decay": wd,
            "inner_metrics": {"macro_f1": 0.8, "nll": 0.4},
            "inner_epochs": [1, 4, 5],
        }
        for lr in [0.001, 0.0003]
        for wd in [0.01, 0.0001]
    ]
    selected = runner.selection_from(candidates)
    assert selected["selected"]["learning_rate"] == 0.0003
    assert selected["selected"]["weight_decay"] == 0.0001
    assert selected["refit_epochs"] == 4


def test_immutable_receipt_cannot_be_replaced(tmp_path):
    path = tmp_path / "receipt.json"
    runner.write_immutable(path, {"locked": 1})
    runner.write_immutable(path, {"locked": 1})
    with pytest.raises(RuntimeError, match="Immutable"):
        runner.write_immutable(path, {"locked": 2})


def test_partial_arm_is_not_projected_to_whole_population(tmp_path):
    data, protocol = synthetic_workload()
    assert runner.completed_arm(tmp_path, data, protocol, "expected_episode", "lock") is None


def test_misplaced_outer_arm_marker_is_refused(tmp_path):
    data, protocol = synthetic_workload()
    marker = tmp_path / "models/expected_episode/fold-0/outer_complete.json"
    runner.write_immutable(
        marker,
        {
            "arm": "log_survival_control",
            "fold": 0,
            "execution_lock_sha256": "lock",
            "probabilities_sha256": "unused",
        },
    )
    with pytest.raises(RuntimeError, match="marker identity"):
        runner.completed_arm(tmp_path, data, protocol, "expected_episode", "lock")


def test_declared_budget_is_exactly_150_and_proposal_is_preserved():
    protocol = runner.audit.read_json(runner.PROTOCOL)
    assert (
        len(protocol["arms"])
        * len(protocol["outer_folds"])
        * (
            len(protocol["learning_rates"])
            * len(protocol["weight_decays"])
            * protocol["inner_folds"]
            + len(protocol["outer_seeds"])
        )
        == 150
    )
    assert (
        runner.file_sha256(runner.ROOT / protocol["proposal_snapshot"])
        == protocol["proposal_snapshot_sha256"]
    )
    assert protocol["gradient_clip_norm"] == 1.0
    assert protocol["hazard"]["loss"].startswith("Uniform unweighted")
