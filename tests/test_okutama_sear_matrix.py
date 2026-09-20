# ruff: noqa: E402, I001

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import run_okutama_sear_matrix as runner  # noqa: E402


PROTOCOL = json.loads(runner.PROTOCOL.read_text(encoding="utf-8"))


def tiny_data():
    scenarios = np.repeat(np.array([f"s{i}" for i in range(10)]), 6)
    return {
        "sample_ids": np.array([f"row-{i}" for i in range(60)]),
        "labels": np.tile(np.arange(3), 20),
        "scenarios": scenarios,
        "folds": np.repeat(np.arange(5), 12),
    }


def test_protocol_locks_exact_matrix_and_capacity():
    counts = runner.validate_protocol(PROTOCOL)
    assert len(counts) == 6
    assert sum(15 for _arm in runner.ARMS for _fold in range(5)) == 450
    assert counts["a5_sear"] == 551_558


def test_outer_request_is_independent_of_held_labels():
    data = tiny_data()
    train, held = np.arange(48), np.arange(48, 60)
    args = (data, "a5_sear", train, held, 0.001, 0.01, 42, "cuda", "lock")
    before = runner.build_fit_request(*args, outer_fold=0, epochs=4)
    data["labels"][held] = 2
    after = runner.build_fit_request(*args, outer_fold=0, epochs=4)
    assert before == after
    data["labels"][0] = 2
    assert before != runner.build_fit_request(*args, outer_fold=0, epochs=4)


def test_inner_request_binds_selection_labels():
    data = tiny_data()
    train, held = np.arange(42), np.arange(42, 48)
    args = (data, "a1_dense_cnn", train, held, 0.001, 0.01, 42, "cuda", "lock")
    before = runner.build_fit_request(*args, outer_fold=0, inner_fold=1)
    data["labels"][held] = np.roll(data["labels"][held], 1)
    assert before != runner.build_fit_request(*args, outer_fold=0, inner_fold=1)


def test_inner_splits_are_scenario_disjoint_and_cover_outer_train():
    data = tiny_data()
    train = np.arange(48)
    splits = runner.inner_splits(data, train, PROTOCOL)
    assert len(splits) == 3
    assert np.array_equal(np.sort(np.concatenate([held for _, held in splits])), train)
    for fit, held in splits:
        assert not set(data["scenarios"][fit]) & set(data["scenarios"][held])


def test_partial_workload_is_preserved_and_refused(tmp_path):
    data = tiny_data()
    directory = tmp_path / "workloads/a5_sear/fold-0"
    directory.mkdir(parents=True)
    marker = directory / "partial.bin"
    marker.write_bytes(b"preserve")
    with pytest.raises(RuntimeError, match="Partial SEAR workload"):
        runner.run_workload(tmp_path, {"protocol": PROTOCOL}, None, None, data, "a5_sear", 0, "cpu")
    assert marker.read_bytes() == b"preserve"


def test_batch_inputs_preserve_dense_half_precision():
    patches = np.zeros((4, 9, 8), np.float16)
    center = np.zeros((4, 8), np.float16)
    data = {"video": np.zeros((4, 10), np.float32), "local_valid": np.ones(4, bool)}
    values = runner.batch_inputs(patches, center, data, np.array([1, 3]), "cpu")
    assert str(values[0].dtype) == "torch.float16"
    assert str(values[1].dtype) == "torch.float16"
    assert str(values[2].dtype) == "torch.float32"
    assert values[3].dtype == torch.bool


def test_immutable_npz_resumes_exact_mixed_dtypes_and_rejects_stale(tmp_path):
    path = tmp_path / "values.npz"
    arrays = {
        "sample_ids": np.array(["row-a", "row-b"]),
        "labels": np.array([0, 1], dtype=np.int64),
        "valid": np.array([True, False]),
        "values": np.array([1.0, np.nan], dtype=np.float32),
    }
    runner.immutable_npz(path, arrays)
    runner.immutable_npz(path, {name: value.copy() for name, value in arrays.items()})
    stale = {name: value.copy() for name, value in arrays.items()}
    stale["labels"][0] = 2
    with pytest.raises(RuntimeError, match="Immutable SEAR array"):
        runner.immutable_npz(path, stale)


def test_fixed_patch_diagnostics_are_predeclared_and_geometrically_exact():
    count = 27 * 27
    coordinates = runner._diagnostic_coordinates(PROTOCOL, count, "cpu", torch.float32)
    assert coordinates.shape == (count, 2)
    assert len(torch.unique(coordinates, dim=0)) == count
    valid = torch.tensor([True, False])
    matchability = runner._outer_border_matchability(PROTOCOL, valid, count, torch.float32)
    assert matchability.shape == (2, count)
    assert matchability[0].sum() == 19 * 19
    assert matchability[1].sum() == 0
