from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_okutama_video_p2a as runner
from sklearn.exceptions import ConvergenceWarning

from hac.video_fusion import ARMS, DerivedFeatures


def protocol():
    comparisons = [
        {"name": f"{arm}_vs_{reference}", "candidate": arm, "reference": reference}
        for reference in ("baseline", "p1_real_linear")
        for arm in ARMS
    ]
    comparisons.extend(
        {"name": f"{arm}_vs_{ARMS[0]}", "candidate": arm, "reference": ARMS[0]} for arm in ARMS[1:]
    )
    comparisons.append(
        {"name": f"{ARMS[3]}_vs_{ARMS[2]}", "candidate": ARMS[3], "reference": ARMS[2]}
    )
    return {
        "seed": 42,
        "arms": list(ARMS),
        "probes": {
            "linear": {
                "C_values": list(runner.C_VALUES),
                "inner_folds": 3,
                "inner_seed": 42,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 2000,
                "tolerance": 0.0001,
                "inner_shuffle": True,
                "standardize": True,
            }
        },
        "statistics": {
            "comparisons": comparisons,
            "bootstrap_resamples": 100,
            "bootstrap_seed": 20260907,
        },
    }


def synthetic_data():
    count = 45
    labels = np.tile(np.arange(3), 15)
    groups = np.repeat([f"scene-{index:02d}" for index in range(15)], 3)
    folds = np.repeat(np.arange(15) % 5, 3)
    data = runner.p1.PrimaryData(
        np.asarray([str(index) for index in range(count)]),
        labels,
        groups,
        folds,
        {},
        {},
        np.full((count, 3), 1 / 3),
        np.zeros(count, bool),
        np.arange(count) % 7 == 0,
    )
    features = DerivedFeatures(
        np.arange(count, dtype=np.float32)[:, None],
        np.ones((count, 1), dtype=np.float32),
        np.ones((count, 2), dtype=np.float32),
        {arm: np.ones(count, bool) for arm in ARMS},
    )
    return data, features


def fake_fit_factory(data, calls):
    def fit(features, labels, config, c, seed, *, binary):
        rows = features[:, 0].astype(int)
        assert np.all(data.folds[rows] != 0), "Outer-held data entered a fitted transformation"
        if binary:
            if features.shape[1] == 2:
                np.testing.assert_array_equal(labels, data.labels[rows] != 0)
            else:
                assert np.all(data.labels[rows] != 0), "Sitting entered conditional motion fitting"
                np.testing.assert_array_equal(labels, data.labels[rows] == 2)
        else:
            np.testing.assert_array_equal(labels, data.labels[rows])
        calls.append((rows, binary, c))
        width, classes = features.shape[1], 2 if binary else 3
        scaler = SimpleNamespace(
            transform=lambda x: x, mean_=np.zeros(width), scale_=np.ones(width), var_=np.ones(width)
        )
        model = SimpleNamespace(
            predict_proba=lambda x: np.full((len(x), classes), 1 / classes),
            n_iter_=np.asarray([3]),
            coef_=np.zeros((1 if binary else 3, width)),
            intercept_=np.zeros(1 if binary else 3),
            classes_=np.arange(classes),
        )
        return scaler, model

    return fit


@pytest.mark.parametrize("arm,expected_fits,candidates", [(ARMS[0], 13, 4), (ARMS[3], 26, 16)])
def test_nested_selection_only_training_groups_and_no_reference_access(
    monkeypatch, arm, expected_fits, candidates
):
    data, features = synthetic_data()

    class ForbiddenReference:
        def __getitem__(self, _):
            pytest.fail("Historical OOF predictions entered fitting/selection")

    data.baseline = ForbiddenReference()
    calls = []
    monkeypatch.setattr(runner, "fit_logistic", fake_fit_factory(data, calls))
    predictions, details, checkpoint = runner.nested_workload(data, features, arm, 0, protocol())
    assert len(calls) == expected_fits and len(details["candidates"]) == candidates
    assert len(checkpoint) > 0 and predictions.shape == (9, 3)
    assert details["inner_selection_rows"] == 36 and details["fallback_rows"] == 0
    assert not details["baseline_access_during_fit"] and not details["p1_oof_access_during_fit"]
    selected = details["selection"]
    if arm == ARMS[3]:
        assert selected["posture_C"] == selected["motion_C"] == min(runner.C_VALUES)
        assert details["motion_fit_rows"] == 24
        np.testing.assert_allclose(predictions, np.tile([0.5, 0.25, 0.25], (9, 1)))
    else:
        assert selected["C"] == min(runner.C_VALUES)


def test_invalid_arm_rows_fail_before_fitting(monkeypatch):
    data, features = synthetic_data()
    features.validity[ARMS[0]][0] = False
    monkeypatch.setattr(
        runner, "fit_logistic", lambda *args: pytest.fail("fit despite invalid rows")
    )
    with pytest.raises(RuntimeError, match="every original"):
        runner.nested_workload(data, features, ARMS[0], 0, protocol())


def test_selection_tiebreaks_nll_then_each_regularization():
    candidates = [
        {"posture_C": p, "motion_C": m, "inner_metrics": {"macro_f1": f, "nll": n}}
        for p, m, f, n in [
            (1e-5, 1e-5, 0.7, 0.6),
            (0.01, 0.01, 0.8, 0.7),
            (0.001, 0.01, 0.8, 0.6),
            (0.001, 0.001, 0.8, 0.6),
        ]
    ]
    selected = runner.select_candidate(candidates, factorized=True)
    assert selected["posture_C"] == selected["motion_C"] == 0.001


@pytest.mark.parametrize("binary,labels", [(True, [0, 0]), (False, [0, 1])])
def test_missing_training_classes_fail_before_scaling(binary, labels):
    with pytest.raises(RuntimeError, match="missing required"):
        runner.fit_logistic(
            np.eye(2), np.asarray(labels), protocol()["probes"]["linear"], 0.01, 42, binary=binary
        )


def test_standardizer_is_fit_only_on_supplied_partition():
    values = np.asarray([[0, 2], [1, 4], [2, 6], [3, 8]], dtype=np.float32)
    scaler, model = runner.fit_logistic(
        values, np.asarray([0, 1, 0, 1]), protocol()["probes"]["linear"], 0.01, 42, binary=True
    )
    np.testing.assert_allclose(scaler.mean_, [1.5, 5])
    assert np.array_equal(model.classes_, [0, 1])
    assert scaler.transform(np.asarray([[100, 200]])).shape == (1, 2)
    np.testing.assert_allclose(scaler.mean_, [1.5, 5])


@pytest.mark.parametrize("ceiling", [False, True])
def test_nonconvergence_fails_closed(monkeypatch, ceiling):
    def fail(self, *args):
        if ceiling:
            self.n_iter_ = np.asarray([2000])
            self.classes_ = np.arange(2)
            return self
        import warnings

        warnings.warn("no convergence", ConvergenceWarning, stacklevel=1)

    monkeypatch.setattr(runner.LogisticRegression, "fit", fail)
    with pytest.raises(RuntimeError, match="ceiling" if ceiling else "did not converge"):
        runner.fit_logistic(
            np.eye(2), np.arange(2), protocol()["probes"]["linear"], 0.01, 42, binary=True
        )


def test_invalid_lock_prevents_input_decoding_and_fitting(tmp_path, monkeypatch):
    root = Path(runner.__file__).resolve().parents[1]

    def reject(*args):
        raise RuntimeError("invalid P2a lock")

    monkeypatch.setattr(runner, "_load_lock", reject)
    monkeypatch.setattr(
        runner.p1, "load_primary_data", lambda *args: pytest.fail("inputs before lock")
    )
    with pytest.raises(RuntimeError, match="invalid P2a lock"):
        runner.run(
            Namespace(
                output_dir=root / ".runs/test_unused_p2a",
                protocol_lock=tmp_path / "bad.json",
                max_new_workloads=None,
            )
        )


def test_workload_request_published_before_fit_and_partial_output_preserved(tmp_path, monkeypatch):
    data, features = synthetic_data()
    directory = tmp_path / "workloads" / ARMS[0] / "fold-0"

    def guarded(*args):
        assert json.loads((directory / "request.json").read_text())["request_sha256"] == "digest"
        raise RuntimeError("intentional pre-fit stop")

    monkeypatch.setattr(runner, "nested_workload", guarded)
    with pytest.raises(RuntimeError, match="intentional"):
        runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")
    (directory / "checkpoint.npz.tmp").write_bytes(b"partial evidence")
    with pytest.raises(RuntimeError, match="Incomplete"):
        runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")
    assert (directory / "checkpoint.npz.tmp").read_bytes() == b"partial evidence"


def test_completed_workload_resumes_without_fitting_and_detects_tampering(tmp_path, monkeypatch):
    data, features = synthetic_data()
    probabilities = np.full((9, 3), 1 / 3)
    monkeypatch.setattr(runner, "nested_workload", lambda *args: (probabilities, {}, b"checkpoint"))
    saved, fitted = runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")
    assert fitted
    monkeypatch.setattr(
        runner, "nested_workload", lambda *args: pytest.fail("refit completed evidence")
    )
    retained, fitted = runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")
    assert not fitted
    np.testing.assert_array_equal(saved, retained)
    (tmp_path / "workloads" / ARMS[0] / "fold-0" / "checkpoint.npz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bytes changed"):
        runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")


def test_subgroup_bootstrap_retains_scenarios_missing_stratum():
    labels = np.tile(np.arange(3), 11)
    scenarios = np.repeat([str(index) for index in range(11)], 3)
    mask = np.arange(33) < 3
    candidate, reference = np.eye(3)[labels], np.full((33, 3), 1 / 3)
    result = runner.subgroup_statistics(
        labels, candidate, reference, scenarios, mask, bootstrap_resamples=100, bootstrap_seed=4
    )
    assert len(result["scenario_order"]) == 11
    assert result["rows"] == 3 and result["bootstrap_draws_used"] == 100
    assert 0 < result["bootstrap_empty_draw_fraction"] < 1
    assert result["rescued_errors"] == 2 and result["new_errors"] == 0
    assert result["class_support"] == [1, 1, 1]


def test_summary_uses_exact_locked_twelve_comparisons_and_holm(monkeypatch):
    data, _ = synthetic_data()
    predictions = {arm: data.baseline.copy() for arm in ARMS}

    # Test comparison orchestration with small synthetic arrays; exact statistic
    # implementation is independently exercised in the P1 and subgroup tests.
    def paired(*args, **kwargs):
        return {"one_sided_exact_swap_pvalue": 0.1, "macro_f1_delta": 0.0}

    monkeypatch.setattr(runner.p1, "paired_statistics", paired)
    result = runner.summarize(data, predictions, protocol(), data.baseline)
    assert len(result["contrasts"]) == len(result["holm_family"]) == 12
    assert all(c["holm_adjusted_one_sided_pvalue"] == 1 for c in result["contrasts"].values())
    assert result["rows"] == 45 and not result["reference_predictions_used_for_fitting"]
    assert result["models"][ARMS[0]]["fallback_rows"] == 0


def test_protocol_rejects_posthoc_c_grid_change():
    spec = protocol()
    runner._validate_protocol(spec)
    spec["probes"]["linear"]["C_values"].append(0.1)
    with pytest.raises(RuntimeError, match="contract changed"):
        runner._validate_protocol(spec)
