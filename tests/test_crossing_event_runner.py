from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location("crossing_runner_test", ROOT / "experiments/run_okutama_crossing_event_verifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_population_contamination_detected():
    module = runner()
    folds = np.array([0, 0, 1, 1, 2, 2])
    module.check_training_rows(np.array([2, 3, 4, 5]), np.array([0, 1]), folds, 0)
    with pytest.raises(RuntimeError, match="training rows"):
        module.check_training_rows(np.array([1, 2, 3, 4, 5]), np.array([0, 1]), folds, 0)


def test_metrics_and_transition_fixture():
    module = runner()
    labels = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[[0, 1, 2, 1, 1, 2]] * .8 + .2 / 3
    result = module.metrics(labels, p)
    assert result["errors"] == 1
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["macro_f1"] == pytest.approx((2 / 3 + .8 + 1) / 3)
    assert module.transitions(labels, p, np.eye(3)[labels]) == {"rescues": 1, "harms": 0, "net": 1}


def test_cluster_interval_zero_and_perfect_improvement():
    module = runner()
    labels = np.tile(np.arange(3), 6)
    scenes = np.repeat(np.arange(6).astype(str), 3)
    p = np.eye(3)[labels] * .8 + .2 / 3
    result = module.clustered_interval(labels, p, p, scenes, draws=50)
    assert result["percentile_95"] == [0, 0]
    bad = np.roll(p, 1, axis=1)
    result = module.clustered_interval(labels, bad, p, scenes, draws=50)
    assert result["percentile_95"] == [1, 1]


def test_exclusive_json_numpy_boolean_safe(tmp_path):
    module = runner()
    path = tmp_path / "receipt.json"
    module.write_json(path, {"pass": np.bool_(True)})
    assert json.loads(path.read_text()) == {"pass": True}
    module.write_json(path, {"pass": True})
    with pytest.raises(RuntimeError, match="immutable"):
        module.write_json(path, {"pass": False})
    assert json.loads(path.read_text()) == {"pass": True}


def test_protocol_preserves_approved_promotion_and_no_grid():
    module = runner()
    spec = module.read_json(module.REVIEW / "NEXT_TRIAL_PROTOCOL.json")
    executable = module.read_json(module.PROTOCOL)
    for key in ("promotion", "readers", "new_verifiers", "minimum_training_support"):
        assert executable[key] == spec[key]
