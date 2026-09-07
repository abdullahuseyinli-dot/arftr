from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_okutama_video_p7 as runner

from hac.video_center import ARMS, CenterAwareFeatures
from hac.video_token_moments import TokenMomentFeatures
from tools import lock_okutama_video_p7 as lock

ROOT = Path(__file__).resolve().parents[1]


def protocol() -> dict:
    return {
        "seed": 42,
        "arms": list(ARMS),
        "primary_expert": "center_signed_factorized",
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
            "maximum_unique_estimator_fits": 715,
            "maximum_workloads": 30,
        },
        "statistics": {
            "comparisons": lock.expected_comparisons(),
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260907,
        },
    }


def synthetic_data() -> tuple[runner.p1.PrimaryData, CenterAwareFeatures]:
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
    moments = TokenMomentFeatures(
        np.zeros((count, 3072), np.float32),
        np.zeros((count, 4608), np.float32),
        np.zeros((count, 4608), np.float32),
        np.zeros((count, 6912), np.float32),
        np.zeros((count, 6144), np.float32),
        np.ones(count, dtype=bool),
    )
    moments.visual_posture[:, 0] = np.arange(count)
    moments.visual_motion[:, 0] = np.arange(count)
    center = CenterAwareFeatures(
        moments,
        *(np.zeros((count, 1536), np.float32) for _ in range(5)),
        np.ones(count, dtype=bool),
    )
    return data, center


def fake_fit_factory(data, calls):
    def fit(features, labels, config, c, seed, *, binary):
        rows = features[:, 0].astype(int)
        assert np.all(data.folds[rows] != 0), "Outer-held row entered P7 fitting"
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
    [(ARMS[2], 26, 16), (ARMS[-1], 13, 4)],
)
def test_nested_fit_counts_outer_isolation_and_reference_exclusion(
    monkeypatch, arm, expected_fits, candidates
) -> None:
    data, features = synthetic_data()

    class ForbiddenReference:
        def __getitem__(self, _):
            pytest.fail("Historical probability entered P7 fitting")

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
    assert not details["historical_oof_access_during_fit"]


def test_protocol_freezes_center_recipe_budget_and_comparisons() -> None:
    spec = lock.load_protocol(ROOT)
    assert tuple(spec["arms"]) == ARMS
    assert spec["statistics"]["comparisons"] == lock.expected_comparisons()
    assert len(spec["statistics"]["comparisons"]) == 6
    assert spec["probe"]["maximum_unique_estimator_fits"] == 715
    assert spec["feature_recipe"]["signed_change_width"] == 1536
    assert spec["systems"]["center_signed_replacement_triad"]["learned_parameters"] == 0
    runner._validate_protocol(spec)


def test_change_counts_separates_rescues_harms_and_still_wrong() -> None:
    labels = np.asarray([0, 0, 1, 2])
    reference = np.eye(3)[[1, 0, 1, 1]]
    candidate = np.eye(3)[[0, 2, 1, 0]]
    counts = runner._change_counts(
        labels, candidate, reference, np.ones(4, dtype=bool)
    )
    assert counts == {
        "rows": 4,
        "reference_errors": 2,
        "candidate_errors": 2,
        "rescues": 1,
        "harms": 1,
        "net_correct_change": 0,
        "changed_but_still_wrong": 1,
    }


def test_completed_workload_resumes_without_refitting_and_detects_corruption(
    monkeypatch, tmp_path: Path
) -> None:
    data, features = synthetic_data()
    calls = []

    def fake_nested(data, features, arm, fold, protocol):
        calls.append((arm, fold))
        rows = int((data.folds == fold).sum())
        return (
            np.full((rows, 3), 1 / 3),
            {"unique_estimator_fits": 13},
            b"checkpoint",
        )

    monkeypatch.setattr(runner, "nested_workload", fake_nested)
    first, fitted = runner._workload(
        data, features, ARMS[-1], 0, protocol(), tmp_path, "request"
    )
    second, refitted = runner._workload(
        data, features, ARMS[-1], 0, protocol(), tmp_path, "request"
    )
    np.testing.assert_array_equal(first, second)
    assert fitted and not refitted and calls == [(ARMS[-1], 0)]
    checkpoint = tmp_path / "workloads" / ARMS[-1] / "fold-0" / "checkpoint.npz"
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="artifact changed"):
        runner._workload(
            data, features, ARMS[-1], 0, protocol(), tmp_path, "request"
        )


@pytest.mark.parametrize(
    "path,value",
    [
        (("feature_recipe", "center", "dino_anchor"), 7),
        (("feature_recipe", "offcenter_control", "vjepa_tubelet"), 3),
        (("feature_recipe", "source_fps"), 25.0),
        (("feature_recipe", "long_source_stride_frames"), 8),
        (("fold_contract", "fold-0"), ["not-a-scenario"]),
        (("references", "p6", "reference_array"), "changed"),
        (("statistics", "bootstrap_seed"), 1),
        (("engineering_guardrails", "milestone_macro_f1"), 0.9),
    ],
)
def test_semantic_protocol_drift_fails_closed(
    tmp_path: Path, path: tuple[str, ...], value
) -> None:
    spec = json.loads((ROOT / lock.PROTOCOL_PATH).read_text(encoding="utf-8"))
    target = spec
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    protocol_path = tmp_path / lock.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        lock.load_protocol(tmp_path)


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
