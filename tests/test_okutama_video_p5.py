from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_okutama_video_p5 as runner

from hac.video_token_moments import ARMS, TokenMomentFeatures
from tools import lock_okutama_video_p5 as lock

ROOT = Path(__file__).resolve().parents[1]


def protocol() -> dict:
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
            "maximum_unique_estimator_fits": 585,
        },
        "statistics": {
            "comparisons": lock.expected_comparisons(),
            "bootstrap_resamples": 100,
            "bootstrap_seed": 20260907,
            "breakthrough_target": 0.82,
        },
    }


def synthetic_data() -> tuple[runner.p1.PrimaryData, TokenMomentFeatures]:
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
    features = TokenMomentFeatures(
        *(row.copy() for _ in range(5)), np.ones(count, dtype=bool)
    )
    return data, features


def fake_fit_factory(data, calls):
    def fit(features, labels, config, c, seed, *, binary):
        rows = features[:, 0].astype(int)
        assert np.all(data.folds[rows] != 0), "Outer-held row entered P5 fitting"
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
    "arm,expected_fits,candidates",
    [(ARMS[0], 26, 16), (ARMS[-1], 13, 4)],
)
def test_nested_fit_counts_outer_isolation_and_reference_exclusion(
    monkeypatch, arm, expected_fits, candidates
) -> None:
    data, features = synthetic_data()

    class ForbiddenReference:
        def __getitem__(self, _):
            pytest.fail("Historical probability entered P5 fitting")

    data.baseline = ForbiddenReference()
    calls = []
    monkeypatch.setattr(runner.p2, "fit_logistic", fake_fit_factory(data, calls))
    probabilities, details, checkpoint = runner.nested_workload(
        data, features, arm, 0, protocol()
    )
    assert len(calls) == expected_fits == details["unique_estimator_fits"]
    assert len(details["candidates"]) == candidates
    assert probabilities.shape == (9, 3) and checkpoint
    assert not details["baseline_access_during_fit"]
    assert not details["p3_oof_access_during_fit"]


def test_protocol_freezes_moments_budget_and_comparisons() -> None:
    spec = lock.load_protocol(ROOT)
    assert tuple(spec["arms"]) == ARMS
    assert spec["statistics"]["comparisons"] == lock.expected_comparisons()
    assert len(spec["statistics"]["comparisons"]) == 12
    assert spec["probe"]["maximum_unique_estimator_fits"] == 585
    assert spec["feature_recipe"]["dc_coefficient_excluded"]
    runner._validate_protocol(spec)


def test_protocol_drift_and_lock_overwrite_fail_closed(tmp_path: Path) -> None:
    spec = json.loads((ROOT / lock.PROTOCOL_PATH).read_text(encoding="utf-8"))
    protocol_path = tmp_path / lock.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    spec["probe"]["maximum_unique_estimator_fits"] += 1
    protocol_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        lock.load_protocol(tmp_path)
    output = tmp_path / "lock.json"
    lock.write_lock(tmp_path, output, {"status": lock.STATUS})
    with pytest.raises(FileExistsError, match="overwrite"):
        lock.write_lock(tmp_path, output, {"status": "changed"})
