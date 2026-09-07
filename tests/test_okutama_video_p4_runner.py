from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_okutama_video_p4 as runner

from hac.video_kinematic import ARMS, KinematicFeatures


def protocol() -> dict:
    from tools.lock_okutama_video_p4_materialization import expected_comparisons

    return {
        "seed": 42,
        "arms": list(ARMS),
        "probe": {
            "C_values": list(runner.C_VALUES),
            "inner_folds": 3,
            "inner_seed": 42,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 2000,
            "tolerance": 0.0001,
            "inner_shuffle": True,
            "standardize": True,
            "maximum_unique_estimator_fits": 910,
        },
        "statistics": {
            "comparisons": expected_comparisons(),
            "bootstrap_resamples": 100,
            "bootstrap_seed": 20260907,
            "breakthrough_target": 0.82,
        },
    }


def synthetic_data() -> tuple[runner.p1.PrimaryData, KinematicFeatures]:
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
        np.zeros(count, dtype=bool),
        np.arange(count) % 7 == 0,
    )
    row = np.column_stack((np.arange(count), np.ones(count))).astype(np.float32)
    features = KinematicFeatures(
        *(row.copy() for _ in range(8)),
        np.linspace(0, 1, count, dtype=np.float32),
    )
    return data, features


def fake_fit_factory(data, calls):
    def fit(features, labels, config, c, seed, *, binary):
        rows = features[:, 0].astype(int)
        assert np.all(data.folds[rows] != 0), "Outer-held row entered P4 fitting"
        if binary:
            if np.all(data.labels[rows] != 0):
                np.testing.assert_array_equal(labels, data.labels[rows] == 2)
            else:
                np.testing.assert_array_equal(labels, data.labels[rows] != 0)
        else:
            np.testing.assert_array_equal(labels, data.labels[rows])
        calls.append((rows, binary, c))
        width, classes = features.shape[1], 2 if binary else 3
        scaler = SimpleNamespace(
            transform=lambda x: x,
            mean_=np.zeros(width),
            scale_=np.ones(width),
            var_=np.ones(width),
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


@pytest.mark.parametrize(
    "arm,expected_fits,candidates,motion_references",
    [
        ("visual_factorized_refit", 26, 16, 1),
        ("camera_reference_marginalized", 39, 16, 2),
        ("visual_dual_track_multinomial", 13, 4, None),
    ],
)
def test_nested_fit_counts_outer_isolation_and_no_reference_access(
    monkeypatch, arm, expected_fits, candidates, motion_references
) -> None:
    data, features = synthetic_data()

    class ForbiddenReference:
        def __getitem__(self, _):
            pytest.fail("Historical probability entered P4 fitting")

    data.baseline = ForbiddenReference()
    calls = []
    monkeypatch.setattr(runner.p2, "fit_logistic", fake_fit_factory(data, calls))
    probabilities, details, checkpoint = runner.nested_workload(
        data, features, arm, 0, protocol()
    )
    assert len(calls) == expected_fits == details["unique_estimator_fits"]
    assert len(details["candidates"]) == candidates
    assert probabilities.shape == (9, 3) and checkpoint
    assert details.get("motion_references") == motion_references
    assert not details["baseline_access_during_fit"]
    assert not details["p3_oof_access_during_fit"]
    if motion_references == 2:
        assert details["probability_reduction"] == "unweighted_arithmetic_mean"


def test_protocol_rejects_budget_and_family_drift() -> None:
    spec = protocol()
    runner._validate_protocol(spec)
    spec["probe"]["maximum_unique_estimator_fits"] += 1
    with pytest.raises(RuntimeError, match="contract changed"):
        runner._validate_protocol(spec)
    spec = protocol()
    spec["statistics"]["comparisons"].pop()
    with pytest.raises(RuntimeError, match="contract changed"):
        runner._validate_protocol(spec)


def test_workload_request_precedes_fit_and_partial_is_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    data, features = synthetic_data()
    directory = tmp_path / "workloads" / ARMS[0] / "fold-0"

    def guarded(*args):
        assert json.loads((directory / "request.json").read_text())["request_sha256"] == "digest"
        raise RuntimeError("intentional pre-fit stop")

    monkeypatch.setattr(runner, "nested_workload", guarded)
    with pytest.raises(RuntimeError, match="intentional"):
        runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")
    (directory / "checkpoint.npz.tmp").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="Incomplete"):
        runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")


def test_completed_workload_resumes_without_refit_and_rejects_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    data, features = synthetic_data()
    probabilities = np.full((9, 3), 1 / 3)
    monkeypatch.setattr(
        runner, "nested_workload", lambda *args: (probabilities, {}, b"checkpoint")
    )
    _, fitted = runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")
    assert fitted
    monkeypatch.setattr(
        runner, "nested_workload", lambda *args: pytest.fail("refit completed evidence")
    )
    retained, fitted = runner._workload(
        data, features, ARMS[0], 0, protocol(), tmp_path, "digest"
    )
    assert not fitted
    np.testing.assert_array_equal(retained, probabilities)
    path = tmp_path / "workloads" / ARMS[0] / "fold-0" / "checkpoint.npz"
    path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact changed"):
        runner._workload(data, features, ARMS[0], 0, protocol(), tmp_path, "digest")
