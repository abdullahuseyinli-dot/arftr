from __future__ import annotations

import itertools
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_okutama_video_probe as runner
from sklearn.exceptions import ConvergenceWarning


def perfect_probabilities(labels):
    return np.eye(3, dtype=float)[labels]


def test_metrics_keep_fixed_classes_and_confusion():
    labels = np.asarray([0, 1, 2, 2])
    probabilities = perfect_probabilities(labels)
    score = runner.metrics(labels, probabilities)
    assert score["macro_f1"] == 1.0
    assert score["nll"] == 0.0 and score["brier"] == 0.0
    assert score["confusion"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert runner.metrics(np.asarray([0]), np.asarray([[1.0, 0.0, 0.0]]))["macro_f1"] == 1 / 3


@pytest.mark.parametrize(
    "probabilities", [np.ones((2, 3)), np.full((2, 3), np.nan), np.zeros((2, 2))]
)
def test_invalid_probabilities_fail_closed(probabilities):
    with pytest.raises(RuntimeError):
        runner.validate_probability_array(probabilities, 2)


def test_inner_splits_exclude_entire_outer_scenarios_and_cover_once():
    scenarios = np.repeat([f"scene-{number}" for number in range(15)], 6)
    labels = np.tile([0, 1, 2, 0, 1, 2], 15)
    folds = np.repeat(np.arange(15) % 5, 6)
    splits = runner.grouped_inner_splits(labels, scenarios, folds, 2, n_splits=3, seed=42)
    assert len(splits) == 3
    for fit, held in splits:
        assert not set(scenarios[fit]) & set(scenarios[held])
        assert not np.any(folds[np.concatenate((fit, held))] == 2)
    assert np.array_equal(
        np.sort(np.concatenate([held for _, held in splits])), np.flatnonzero(folds != 2)
    )


def test_outer_group_leakage_is_rejected():
    with pytest.raises(RuntimeError, match="crosses the outer"):
        runner.grouped_inner_splits(
            np.asarray([0, 1, 2, 0]),
            np.asarray(["a", "b", "a", "c"]),
            np.asarray([0, 1, 1, 1]),
            0,
            n_splits=2,
            seed=42,
        )


def test_feature_validation_preserves_missing_rows_but_rejects_bad_valid_values():
    values = np.zeros((2, 8, 9, 768), dtype=np.float16)
    values[1] = np.nan
    valid = np.asarray([True, False])
    runner.validate_feature_array("vjepa21_real_clip", values, valid, 2)
    with pytest.raises(RuntimeError, match="nonfinite"):
        runner.validate_feature_array("vjepa21_real_clip", values, np.ones(2, dtype=bool), 2)
    with pytest.raises(RuntimeError, match="shape"):
        runner.validate_feature_array("dinov2_native_frames", values, valid, 2)


def test_exact_group_swap_matches_brute_force_and_counts_rescue_harm():
    labels = np.tile([0, 1, 2, 1], 3)
    groups = np.repeat(["a", "b", "c"], 4)
    reference = perfect_probabilities(np.asarray([0, 0, 2, 2] * 3))
    candidate = perfect_probabilities(np.asarray([0, 1, 2, 1, 1, 1, 2, 1, 0, 1, 1, 1]))
    result = runner.paired_statistics(
        labels, candidate, reference, groups, bootstrap_resamples=200, bootstrap_seed=7
    )
    observed = (
        runner.metrics(labels, candidate)["macro_f1"]
        - runner.metrics(labels, reference)["macro_f1"]
    )
    samples = []
    for bits in itertools.product((False, True), repeat=3):
        mask = np.asarray(bits)[np.repeat(np.arange(3), 4)]
        left, right = (
            np.where(mask[:, None], reference, candidate),
            np.where(mask[:, None], candidate, reference),
        )
        samples.append(
            runner.metrics(labels, left)["macro_f1"] - runner.metrics(labels, right)["macro_f1"]
        )
    assert result["exact_swap_assignments"] == 8
    assert result["one_sided_exact_swap_pvalue"] == np.mean(np.asarray(samples) >= observed - 1e-12)
    assert result["rescued_errors"] == 6 and result["new_errors"] == 2
    assert result["net_correct_change"] == 4


def test_identical_predictions_have_null_paired_statistics():
    labels = np.tile(np.arange(3), 11)
    groups = np.repeat([str(index) for index in range(11)], 3)
    probabilities = perfect_probabilities(labels)
    result = runner.paired_statistics(
        labels,
        probabilities,
        probabilities,
        groups,
        bootstrap_resamples=10_000,
        bootstrap_seed=20260907,
    )
    assert result["macro_f1_delta_two_sided_95pct"] == [0.0, 0.0]
    assert result["exact_swap_assignments"] == 2048
    assert result["one_sided_exact_swap_pvalue"] == 1.0


def test_holm_adjustment_is_monotonic_and_bounded():
    assert runner.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.8}) == {"a": 0.03, "b": 0.06, "c": 0.8}


def test_atomic_write_refuses_existing_and_interrupted_artifacts(tmp_path):
    target = tmp_path / "result.json"
    runner.atomic_json(target, {"value": 1})
    with pytest.raises(FileExistsError):
        runner.atomic_json(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}
    interrupted = tmp_path / "other.json.tmp"
    interrupted.write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        runner.atomic_json(tmp_path / "other.json", {"value": 1})


def test_nonconverged_linear_fit_is_rejected_before_prediction(monkeypatch):
    def fail(*args, **kwargs):
        import warnings

        warnings.warn("no convergence", ConvergenceWarning, stacklevel=1)

    monkeypatch.setattr(runner.LogisticRegression, "fit", fail)
    with pytest.raises(RuntimeError, match="did not converge"):
        runner.fit_linear(
            np.eye(3),
            np.arange(3),
            {"solver": "lbfgs", "class_weight": "balanced", "max_iter": 2000, "tolerance": 0.0001},
            0.1,
            42,
        )


def test_no_fitting_when_p1_lock_validation_fails(tmp_path, monkeypatch):
    root = Path(runner.__file__).resolve().parents[1]
    monkeypatch.setattr(runner, "configure_determinism", lambda seed: None)

    def reject(*args):
        raise RuntimeError("invalid P1 lock")

    monkeypatch.setattr(runner, "_load_lock", reject)
    monkeypatch.setattr(
        runner, "load_primary_data", lambda *args: pytest.fail("decoded inputs before lock")
    )
    with pytest.raises(RuntimeError, match="invalid P1 lock"):
        runner.run(
            Namespace(
                output_dir=root / ".runs/test_unused_p1",
                protocol_lock=tmp_path / "bad.json",
                max_new_workloads=None,
            )
        )


def test_nested_selection_never_reads_baseline_or_held_features_for_fitting(monkeypatch):
    count = 45
    labels = np.tile(np.arange(3), 15)
    groups = np.repeat([f"scene-{index:02d}" for index in range(15)], 3)
    folds = np.repeat(np.arange(15) % 5, 3)
    values = np.arange(count, dtype=float).reshape(count, 1, 1, 1)
    valid = np.ones(count, dtype=bool)
    valid[[0, 6]] = False
    fit_calls = []

    class BaselineGuard:
        def __getitem__(self, indices):
            assert np.all(folds[indices] == 0), "Baseline entered inner selection"
            return np.tile([0.2, 0.3, 0.5], (len(indices), 1))

    def fake_fit(features, fit_labels, config, c, seed):
        indices = features[:, 0].astype(int)
        assert np.all(folds[indices] != 0)
        assert np.all(valid[indices])
        assert np.array_equal(fit_labels, labels[indices])
        fit_calls.append(indices)
        scaler = SimpleNamespace(
            transform=lambda array: array, mean_=np.zeros(1), scale_=np.ones(1), var_=np.ones(1)
        )
        model = SimpleNamespace(
            predict_proba=lambda array: np.tile([1 / 3] * 3, (len(array), 1)),
            n_iter_=np.asarray([3]),
            coef_=np.zeros((3, 1)),
            intercept_=np.zeros(3),
            classes_=np.arange(3),
        )
        return scaler, model

    monkeypatch.setattr(runner, "fit_linear", fake_fit)
    data = runner.PrimaryData(
        np.asarray([str(i) for i in range(count)]),
        labels,
        groups,
        folds,
        {"vjepa21_real_clip": values},
        {"vjepa21_real_clip": valid},
        BaselineGuard(),
        np.zeros(count, bool),
        np.zeros(count, bool),
    )
    protocol = {
        "seed": 42,
        "probes": {"linear": {"C_values": [0.01, 0.1, 1.0], "inner_folds": 3, "inner_seed": 42}},
    }
    predictions, details, _ = runner.linear_workload(data, "vjepa21_real_clip", 0, protocol)
    assert len(fit_calls) == 10
    assert details["selection"]["C"] == 0.01
    assert details["fallback_rows"] == 1
    assert details["inner_selection_rows"] == int(((folds != 0) & valid).sum())
    np.testing.assert_array_equal(predictions[0], [0.2, 0.3, 0.5])


def test_bootstrap_losses_weight_scenario_sizes_before_division():
    labels = np.tile(np.arange(3), 4)
    groups = np.asarray(["a"] * 3 + ["b"] * 9)
    reference = np.full((12, 3), 1 / 3)
    candidate = reference.copy()
    candidate[:3] = perfect_probabilities(labels[:3])
    report = runner.paired_statistics(
        labels, candidate, reference, groups, bootstrap_resamples=100, bootstrap_seed=8
    )
    assert report["nll_delta"] == pytest.approx(-np.log(3) / 4)
    assert report["brier_delta"] == pytest.approx(-1 / 6)
    assert (
        report["both_correct"]
        + report["shared_errors"]
        + report["new_errors"]
        + report["rescued_errors"]
        == 12
    )


def test_workload_resume_rejects_an_incomplete_artifact_inventory(tmp_path):
    labels = np.arange(3)
    data = runner.PrimaryData(
        np.asarray(["a", "b", "c"]),
        labels,
        np.asarray(["x"] * 3),
        np.zeros(3, dtype=int),
        {},
        {},
        perfect_probabilities(labels),
        np.zeros(3, bool),
        np.zeros(3, bool),
    )
    request = {
        "request_sha256": "fixed",
        "arm": "vjepa21_real_clip",
        "probe": "linear",
        "outer_fold": 0,
        "held_sample_ids": ["a", "b", "c"],
        "seed": 42,
    }
    path = tmp_path / "workloads/vjepa21_real_clip/linear/fold-0/receipt.json"
    runner.atomic_json(
        path, {"request": request, "status": "P1_WORKLOAD_COMPLETE", "artifacts": {}}
    )
    with pytest.raises(RuntimeError, match="inventory"):
        runner._workload(data, "vjepa21_real_clip", "linear", 0, {"seed": 42}, tmp_path, "fixed")


def test_workload_request_is_published_before_any_fit(tmp_path, monkeypatch):
    labels = np.arange(3)
    data = runner.PrimaryData(
        np.asarray(["a", "b", "c"]),
        labels,
        np.asarray(["x"] * 3),
        np.zeros(3, dtype=int),
        {},
        {},
        perfect_probabilities(labels),
        np.zeros(3, bool),
        np.zeros(3, bool),
    )

    def guarded_fit(*args):
        request_path = tmp_path / "workloads/vjepa21_real_clip/linear/fold-0/request.json"
        assert json.loads(request_path.read_text())["request_sha256"] == "fixed"
        raise RuntimeError("intentional pre-fit stop")

    monkeypatch.setattr(runner, "linear_workload", guarded_fit)
    with pytest.raises(RuntimeError, match="intentional pre-fit"):
        runner._workload(data, "vjepa21_real_clip", "linear", 0, {"seed": 42}, tmp_path, "fixed")
    assert not list(tmp_path.rglob("receipt.json"))


def test_iteration_ceiling_is_rejected_even_without_warning(monkeypatch):
    def at_ceiling(self, *args):
        self.n_iter_ = np.asarray([2000])
        self.classes_ = np.arange(3)
        return self

    monkeypatch.setattr(runner.LogisticRegression, "fit", at_ceiling)
    with pytest.raises(RuntimeError, match="iteration ceiling"):
        runner.fit_linear(
            np.eye(3),
            np.arange(3),
            {"solver": "lbfgs", "class_weight": "balanced", "max_iter": 2000, "tolerance": 0.0001},
            0.1,
            42,
        )
