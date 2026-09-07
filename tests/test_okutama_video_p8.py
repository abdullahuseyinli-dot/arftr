from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_okutama_video_p8 as runner

from hac.ccac_features import ARMS, CCACFeatures
from hac.video_kinematic import ArmInputs
from hac.video_token_moments import TokenMomentFeatures
from tools import lock_okutama_video_p8 as locker

ROOT = Path(__file__).resolve().parents[1]


def probe_config() -> dict:
    return {
        "C_values": [1e-5, 1e-4, 1e-3, 1e-2],
        "inner_folds": 3,
        "inner_seed": 42,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 2000,
        "tolerance": 0.0001,
        "inner_shuffle": True,
        "standardize": True,
    }


def feature_archive(rows: int = 6) -> dict[str, np.ndarray]:
    available = np.resize([15, 15, 14, 13, 0, 15], rows).astype(np.int16)
    camera = np.resize([15, 14, 12, 11, 0, 15], rows).astype(np.int16)
    translation = np.resize([15, 12, 10, 9, 0, 10], rows).astype(np.int16)
    articulation = np.resize([15, 10, 9, 6, 0, 10], rows).astype(np.int16)
    translation_valid = translation >= 10
    articulation_valid = articulation >= 10
    observed = available > 0
    quality = np.zeros((rows, 16), dtype=np.float32)
    quality[:, :4] = np.column_stack((available, camera, translation, articulation)) / 15
    quality[:, 4] = translation_valid
    quality[:, 5] = articulation_valid
    quality[:, 6:9] = observed[:, None] * np.float32(0.5)
    quality[:, 9] = observed * np.float32(0.1)
    quality[:, 10] = observed
    quality[:, 11] = observed * np.float32(0.2)
    quality[:, 12] = observed
    quality[:, 13] = observed * np.float32(0.5)
    quality[:, 14] = np.float32(0.5)
    quality[:, 15] = np.float32(0.04)
    speed = np.arange(1, rows + 1, dtype=np.float32)[:, None]
    component = np.repeat(speed, 3, axis=1)
    raw = np.concatenate((component, np.zeros_like(component), component), axis=1)
    compensated = raw * np.float32(0.5)
    within = np.concatenate((component * 0.1, component * 0.2), axis=1).astype(np.float32)
    raw[~translation_valid] = 0
    compensated[~translation_valid] = 0
    within[~articulation_valid] = 0
    return {
        "sample_ids": np.asarray([f"sample-{index:03d}" for index in range(rows)]),
        "long_valid": np.arange(rows) % 3 != 0,
        "quality": quality,
        "raw_translation": raw,
        "compensated_translation": compensated,
        "within_actor": within,
        "translation_feature_valid": translation_valid,
        "articulation_feature_valid": articulation_valid,
        "requested_pair_counts": np.full(rows, 15, dtype=np.int16),
        "available_pair_counts": available,
        "camera_pair_counts": camera,
        "translation_pair_counts": translation,
        "articulation_pair_counts": articulation,
    }


def align(arrays: dict[str, np.ndarray]) -> CCACFeatures:
    expected = feature_archive()
    return runner._align_ccac_arrays(arrays, expected["sample_ids"], expected["long_valid"])


def test_feature_alignment_joins_every_block_by_sample_identity() -> None:
    arrays = feature_archive()
    order = np.asarray([4, 1, 5, 0, 3, 2])
    shuffled = {name: values[order].copy() for name, values in arrays.items()}
    untouched = {name: values.copy() for name, values in shuffled.items()}
    actual = align(shuffled)
    for field, archive_name in (
        ("quality", "quality"),
        ("raw_translation", "raw_translation"),
        ("compensated_translation", "compensated_translation"),
        ("within_actor", "within_actor"),
        ("translation_valid", "translation_feature_valid"),
        ("articulation_valid", "articulation_feature_valid"),
    ):
        np.testing.assert_array_equal(getattr(actual, field), arrays[archive_name])
    for name, expected in untouched.items():
        np.testing.assert_array_equal(shuffled[name], expected)


@pytest.mark.parametrize("failure", ["duplicate_source", "missing", "extra", "unknown_id"])
def test_feature_alignment_requires_exact_identity_bijection(failure: str) -> None:
    arrays = feature_archive(7 if failure == "extra" else 6)
    if failure == "duplicate_source":
        arrays["sample_ids"][1] = arrays["sample_ids"][0]
    elif failure == "missing":
        arrays = {name: values[:-1] for name, values in arrays.items()}
    elif failure == "unknown_id":
        arrays["sample_ids"][0] = "unknown-id"
    with pytest.raises((RuntimeError, ValueError)):
        align(arrays)


def test_feature_alignment_rejects_duplicate_destination_and_changed_long_mask() -> None:
    arrays = feature_archive()
    destination = arrays["sample_ids"].copy()
    destination[1] = destination[0]
    with pytest.raises((RuntimeError, ValueError)):
        runner._align_ccac_arrays(arrays, destination, arrays["long_valid"])
    expected_validity = arrays["long_valid"].copy()
    expected_validity[0] = ~expected_validity[0]
    with pytest.raises((RuntimeError, ValueError)):
        runner._align_ccac_arrays(arrays, arrays["sample_ids"], expected_validity)


@pytest.mark.parametrize(
    "failure",
    [
        "label_array",
        "missing_array",
        "float64_quality",
        "float_counts",
        "integer_validity",
        "object_ids",
        "wrong_width",
        "nonfinite_quality",
        "nonfinite_motion",
        "nonzero_invalid_raw",
        "nonzero_invalid_compensated",
        "nonzero_invalid_within",
        "quality_validity_disagrees",
        "quality_fraction_disagrees",
        "count_order",
        "negative_count",
        "requested_count",
        "mask_count_disagrees",
        "articulation_without_translation",
    ],
)
def test_feature_alignment_rejects_corrupt_numeric_or_missingness_contract(failure: str) -> None:
    arrays = feature_archive()
    if failure == "label_array":
        arrays["labels"] = np.zeros(6, dtype=np.int64)
    elif failure == "missing_array":
        arrays.pop("camera_pair_counts")
    elif failure == "float64_quality":
        arrays["quality"] = arrays["quality"].astype(np.float64)
    elif failure == "float_counts":
        arrays["available_pair_counts"] = arrays["available_pair_counts"].astype(np.float32)
    elif failure == "integer_validity":
        arrays["translation_feature_valid"] = arrays["translation_feature_valid"].astype(np.uint8)
    elif failure == "object_ids":
        arrays["sample_ids"] = arrays["sample_ids"].astype(object)
    elif failure == "wrong_width":
        arrays["raw_translation"] = arrays["raw_translation"][:, :-1]
    elif failure == "nonfinite_quality":
        arrays["quality"][0, 9] = np.nan
    elif failure == "nonfinite_motion":
        arrays["raw_translation"][0, 0] = np.inf
    elif failure == "nonzero_invalid_raw":
        arrays["raw_translation"][3, 0] = 1
    elif failure == "nonzero_invalid_compensated":
        arrays["compensated_translation"][3, 0] = 1
    elif failure == "nonzero_invalid_within":
        arrays["within_actor"][2, 0] = 1
    elif failure == "quality_validity_disagrees":
        arrays["quality"][0, 4] = 0
    elif failure == "quality_fraction_disagrees":
        arrays["quality"][0, 0] = 0.5
    elif failure == "count_order":
        arrays["camera_pair_counts"][3] = 14
    elif failure == "negative_count":
        arrays["articulation_pair_counts"][4] = -1
    elif failure == "requested_count":
        arrays["requested_pair_counts"][0] = 14
    elif failure == "mask_count_disagrees":
        arrays["translation_pair_counts"][2] = 9
        arrays["quality"][2, 2] = np.float32(9 / 15)
    elif failure == "articulation_without_translation":
        arrays["articulation_feature_valid"][3] = True
        arrays["articulation_pair_counts"][3] = 10
        arrays["quality"][3, 3] = np.float32(10 / 15)
        arrays["quality"][3, 5] = 1
    else:
        raise AssertionError(failure)
    with pytest.raises((RuntimeError, ValueError)):
        align(arrays)


def synthetic_training_data() -> tuple[runner.p1.PrimaryData, TokenMomentFeatures, CCACFeatures]:
    count = 45
    data = runner.p1.PrimaryData(
        np.asarray([f"row-{index}" for index in range(count)]),
        np.tile(np.arange(3), 15),
        np.repeat([f"scene-{index:02d}" for index in range(15)], 3),
        np.repeat(np.arange(15) % 5, 3),
        {},
        {},
        np.full((count, 3), 1 / 3),
        np.zeros(count, dtype=bool),
        np.arange(count) % 7 == 0,
    )
    moments = TokenMomentFeatures(
        np.zeros((count, 3072), dtype=np.float32),
        np.zeros((count, 4608), dtype=np.float32),
        np.zeros((count, 4608), dtype=np.float32),
        np.zeros((count, 6912), dtype=np.float32),
        np.zeros((count, 6144), dtype=np.float32),
        np.ones(count, dtype=bool),
    )
    moments.visual_posture[:, 0] = np.arange(count)
    moments.visual_motion[:, 0] = np.arange(count)
    archive = feature_archive(count)
    ccac = CCACFeatures(
        archive["quality"],
        archive["raw_translation"],
        archive["compensated_translation"],
        archive["within_actor"],
        archive["translation_feature_valid"],
        archive["articulation_feature_valid"],
    )
    return data, moments, ccac


def fake_fit_factory(data, calls):
    def fit(features, labels, config, c, seed, *, binary):
        rows = features[:, 0].astype(int)
        assert np.all(data.folds[rows] != 0), "Outer-held row entered P8 fitting"
        assert binary
        motion = bool(np.all(data.labels[rows] != 0))
        expected = data.labels[rows] == 2 if motion else data.labels[rows] != 0
        np.testing.assert_array_equal(labels, expected)
        calls.append({"rows": rows.copy(), "motion": motion, "C": c})
        width = features.shape[1]
        scaler = SimpleNamespace(
            transform=lambda values: values,
            mean_=np.zeros(width),
            scale_=np.ones(width),
            var_=np.ones(width),
        )
        model = SimpleNamespace(
            predict_proba=lambda values: np.full((len(values), 2), 0.5),
            n_iter_=np.asarray([2]),
            coef_=np.zeros((1, width)),
            intercept_=np.zeros(1),
            classes_=np.arange(2),
        )
        return scaler, model

    return fit


@pytest.mark.parametrize(
    "arm,posture_c,expected_fits,candidates", [(ARMS[0], None, 26, 16), (ARMS[-1], 1e-3, 17, 4)]
)
def test_nested_workload_fit_budget_and_outer_isolation(
    monkeypatch, arm, posture_c, expected_fits, candidates
) -> None:
    data, moments, ccac = synthetic_training_data()

    class ForbiddenReference:
        def __getitem__(self, _):
            pytest.fail("Historical probabilities entered P8 fitting")

        def __array__(self, *_args, **_kwargs):
            pytest.fail("Historical probabilities were converted during P8 fitting")

    data.baseline = ForbiddenReference()
    calls = []
    monkeypatch.setattr(runner.p2, "fit_logistic", fake_fit_factory(data, calls))
    values, details, checkpoint = runner.nested_workload(
        data, moments, ccac, arm, 0, {"seed": 42, "probe": probe_config()}, posture_c=posture_c
    )
    assert len(calls) == expected_fits == details["unique_estimator_fits"]
    assert len(details["candidates"]) == candidates
    assert values.shape == (9, 3) and checkpoint
    np.testing.assert_allclose(values.sum(axis=1), 1)
    if posture_c is not None:
        posture_calls = [call for call in calls if not call["motion"]]
        assert len(posture_calls) == 4
        assert {call["C"] for call in posture_calls} == {posture_c}


def test_frozen_posture_probabilities_and_selection_ignore_outer_held_labels() -> None:
    data, _, _ = synthetic_training_data()
    row = np.arange(len(data.labels), dtype=np.float64)
    posture = np.column_stack((row % 3, np.sin(row), np.cos(row)))
    motion = np.column_stack((row % 3, row % 5, np.sin(row / 3), np.cos(row / 3)))
    inputs = ArmInputs("factorized", posture, (motion,))
    config = probe_config()

    def evaluate(current):
        splits = runner.p1.grouped_inner_splits(
            current.labels, current.scenarios, current.folds, 0, n_splits=3, seed=42
        )
        return runner._fit_frozen_posture(current, inputs, 0, config, 42, splits, 1e-3)

    first, first_details, first_checkpoint = evaluate(data)
    modified_labels = data.labels.copy()
    modified_labels[data.folds == 0] = (modified_labels[data.folds == 0] + 1) % 3
    second, second_details, second_checkpoint = evaluate(replace(data, labels=modified_labels))
    np.testing.assert_array_equal(first, second)
    assert first_details["selection"] == second_details["selection"]
    assert first_details["unique_estimator_fits"] == second_details["unique_estimator_fits"] == 17
    assert len(first_details["candidates"]) == len(second_details["candidates"]) == 4
    assert first_checkpoint == second_checkpoint


def test_protocol_budget_is_470_actual_fits_with_exactly_six_comparisons() -> None:
    spec = json.loads((ROOT / locker.PROTOCOL_PATH).read_text(encoding="utf-8"))
    locker.validate_spec(spec, require_bound=False)
    assert spec["statistics"]["comparisons"] == locker.expected_comparisons()
    assert len(spec["statistics"]["comparisons"]) == 6
    assert 5 * sum(spec["arm_contracts"][arm]["fits_per_fold"] for arm in ARMS) == 470
    assert spec["probe"]["maximum_unique_estimator_fits"] == 470
    assert spec["probe"]["maximum_estimator_fit_invocations"] == 470


@pytest.mark.parametrize(
    "receipt_name",
    ["execution_lock", "request", "summary", "completion", "feature_manifest", "features"],
)
def test_unset_full_extraction_hash_cannot_be_locked_before_external_access(
    monkeypatch, tmp_path: Path, receipt_name: str
) -> None:
    spec = json.loads((ROOT / locker.PROTOCOL_PATH).read_text(encoding="utf-8"))
    for receipt in spec["full_extraction"].values():
        receipt["sha256"] = "1" * 64
        receipt["size_bytes"] = 1
    spec["full_extraction"][receipt_name]["sha256"] = "unset_full_extraction"

    def forbidden_upstream(_root):
        pytest.fail("Unbound P8 protocol reached external lineage reads")

    monkeypatch.setattr(locker.previous, "load_protocol", forbidden_upstream)
    protocol_path = tmp_path / locker.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        locker.load_protocol(tmp_path)


def small_workload_data():
    count = 6
    data = runner.p1.PrimaryData(
        np.asarray([f"workload-row-{index}" for index in range(count)]),
        np.tile(np.arange(3), 2),
        np.repeat(["scene-a", "scene-b"], 3),
        np.repeat([0, 1], 3),
        {},
        {},
        np.full((count, 3), 1 / 3),
        np.zeros(count, dtype=bool),
        np.zeros(count, dtype=bool),
    )
    features = runner.P8Features(SimpleNamespace(long_valid=np.ones(count, dtype=bool)), object())
    return data, features


def mock_fit_details(arm, posture_c):
    candidates = [
        {
            "posture_C": posture,
            "motion_C": motion,
            "inner_metrics": {"macro_f1": 0.5, "nll": 1.0},
        }
        for posture in (runner.C_VALUES if posture_c is None else (posture_c,))
        for motion in runner.C_VALUES
    ]
    fits = 26 if arm == "spatial_refit" else 17
    return {
        "unique_estimator_fits": fits,
        "estimator_fit_invocations": fits,
        "candidates": candidates,
        "selection": runner.p2.select_candidate(candidates, factorized=True),
    }


def mock_nested_factory(calls, failure=None):
    def nested(data, _moments, _ccac, arm, fold, _protocol, posture_c=None):
        calls.append((arm, fold, posture_c))
        is_addon = arm != "spatial_refit"
        assert (posture_c is not None) == is_addon
        probabilities = np.tile(
            [0.2, 0.35, 0.45] if is_addon else [0.2, 0.3, 0.5],
            (np.count_nonzero(data.folds == fold), 1),
        )
        checkpoint = {
            "posture_coef": np.asarray([[0.1, 0.2]]),
            "posture_intercept": np.asarray([0.0]),
            "posture_scaler_mean": np.zeros(2),
            "posture_scaler_scale": np.ones(2),
            "motion_reference_0_coef": np.asarray([[float(is_addon)]]),
        }
        if is_addon and failure == "sitting":
            probabilities[:, 0] += 0.01
            probabilities[:, 1] -= 0.01
        elif is_addon and failure == "checkpoint":
            checkpoint["posture_coef"][0, 0] += 0.01
        elif is_addon and failure == "checkpoint_inventory":
            checkpoint.pop("posture_intercept")
        return probabilities, mock_fit_details(arm, posture_c), runner.p1.npz_bytes(**checkpoint)

    return nested


def execute_mock_workload(output, data, features, arm):
    return runner._workload(data, features, arm, 0, {"seed": 42}, output, "a" * 64)


def test_completed_workloads_resume_without_fitting_and_pin_current_a0(monkeypatch, tmp_path):
    data, features = small_workload_data()
    calls = []
    monkeypatch.setattr(runner, "nested_workload", mock_nested_factory(calls))
    first_a0, fitted = execute_mock_workload(tmp_path, data, features, ARMS[0])
    assert fitted
    first_addon, fitted = execute_mock_workload(tmp_path, data, features, ARMS[-1])
    assert fitted
    assert calls == [(ARMS[0], 0, None), (ARMS[-1], 0, runner.C_VALUES[0])]

    def forbidden_refit(*_args, **_kwargs):
        pytest.fail("A completed workload was fitted again")

    monkeypatch.setattr(runner, "nested_workload", forbidden_refit)
    for arm, expected in ((ARMS[0], first_a0), (ARMS[-1], first_addon)):
        retained, fitted = execute_mock_workload(tmp_path, data, features, arm)
        np.testing.assert_array_equal(retained, expected)
        assert not fitted
    receipt = json.loads(
        (tmp_path / "workloads" / ARMS[-1] / "fold-0" / "receipt.json").read_text()
    )
    a0_path = tmp_path / "workloads" / ARMS[0] / "fold-0" / "receipt.json"
    dependency = receipt["request"]["a0_dependency"]
    assert dependency["receipt_sha256"] == runner.p1.sha256_file(a0_path)
    assert dependency["posture_C"] == runner.C_VALUES[0]
    assert receipt["details"]["exact_a0_posture_checkpoint_and_sitting_probabilities"]


@pytest.mark.parametrize("failure", ["artifact", "receipt", "identity"])
def test_addon_resume_rejects_changed_a0_dependency(monkeypatch, tmp_path, failure):
    data, features = small_workload_data()
    calls = []
    monkeypatch.setattr(runner, "nested_workload", mock_nested_factory(calls))
    execute_mock_workload(tmp_path, data, features, ARMS[0])
    execute_mock_workload(tmp_path, data, features, ARMS[-1])
    directory = tmp_path / "workloads" / ARMS[0] / "fold-0"
    if failure == "artifact":
        (directory / "checkpoint.npz").write_bytes(b"changed A0 checkpoint")
    else:
        path = directory / "receipt.json"
        receipt = json.loads(path.read_text())
        if failure == "receipt":
            receipt["seconds"] += 1
        else:
            receipt["request"]["outer_fold"] = 1
        path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError):
        execute_mock_workload(tmp_path, data, features, ARMS[-1])
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["sitting", "checkpoint", "checkpoint_inventory"])
def test_addon_publication_requires_exact_a0_posture(monkeypatch, tmp_path, failure):
    data, features = small_workload_data()
    calls = []
    monkeypatch.setattr(runner, "nested_workload", mock_nested_factory(calls, failure))
    execute_mock_workload(tmp_path, data, features, ARMS[0])
    with pytest.raises(RuntimeError, match="sitting probabilities|posture checkpoint"):
        execute_mock_workload(tmp_path, data, features, ARMS[-1])
    directory = tmp_path / "workloads" / ARMS[-1] / "fold-0"
    assert (directory / "fit_started.json").exists()
    assert not (directory / "receipt.json").exists()
    with pytest.raises(RuntimeError, match="no automatic refitting"):
        execute_mock_workload(tmp_path, data, features, ARMS[-1])
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["sitting", "checkpoint"])
def test_addon_resume_rechecks_posture_even_if_local_hashes_are_updated(
    monkeypatch, tmp_path, failure
):
    data, features = small_workload_data()
    calls = []
    monkeypatch.setattr(runner, "nested_workload", mock_nested_factory(calls))
    execute_mock_workload(tmp_path, data, features, ARMS[0])
    execute_mock_workload(tmp_path, data, features, ARMS[-1])
    directory = tmp_path / "workloads" / ARMS[-1] / "fold-0"
    name = "predictions.npz" if failure == "sitting" else "checkpoint.npz"
    path = directory / name
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if failure == "sitting":
        arrays["probabilities"][:, 0] += 0.01
        arrays["probabilities"][:, 1] -= 0.01
    else:
        arrays["posture_coef"][0, 0] += 0.01
    path.write_bytes(runner.p1.npz_bytes(**arrays))
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][name] = runner.p1.sha256_file(path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sitting probabilities|posture checkpoint"):
        execute_mock_workload(tmp_path, data, features, ARMS[-1])
    assert len(calls) == 2


def test_fit_started_marker_prevents_any_automatic_refit(monkeypatch, tmp_path):
    data, features = small_workload_data()
    directory = tmp_path / "workloads" / ARMS[0] / "fold-0"
    directory.mkdir(parents=True)
    (directory / "fit_started.json").write_text("{}", encoding="utf-8")

    def forbidden_refit(*_args, **_kwargs):
        pytest.fail("An interrupted started workload was fitted again")

    monkeypatch.setattr(runner, "nested_workload", forbidden_refit)
    with pytest.raises(RuntimeError, match="no automatic refitting"):
        execute_mock_workload(tmp_path, data, features, ARMS[0])


def publication_fixture(output, *, published=True):
    names = {"oof_probabilities.npz", "metrics.csv", "paired_statistics.json", "diagnostics.json"}
    names.update(
        f"workloads/{arm}/fold-{fold}/{name}"
        for arm in ARMS
        for fold in range(5)
        for name in (
            "request.json",
            "receipt.json",
            "checkpoint.npz",
            "predictions.npz",
            "fit_started.json",
        )
    )
    for name in names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    (output / "request.json").write_text("{}", encoding="utf-8")
    summary = {
        "status": runner.STATUS,
        "request_sha256": "a" * 64,
        "artifacts": {name: runner.p1.sha256_file(output / name) for name in names},
    }
    if published:
        runner.p1.atomic_json(output / "summary.json", summary)
        runner.p1.atomic_json(
            output / "completion.json",
            {
                "status": "P8_PUBLICATION_COMPLETE",
                "request_sha256": "a" * 64,
                "summary_sha256": runner.p1.sha256_file(output / "summary.json"),
            },
        )
    return summary


@pytest.mark.parametrize("published", [False, True])
def test_validate_output_accepts_only_exact_stage_inventory(tmp_path, published):
    summary = publication_fixture(tmp_path, published=published)
    assert len(summary["artifacts"]) == 129
    runner._validate_output(tmp_path, summary, "a" * 64, published=published)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_entry",
        "extra_file",
        "artifact_bytes",
        "summary_bytes",
        "completion",
        "completion_extra",
    ],
)
def test_validate_output_rejects_changed_inventory_and_completion(tmp_path, failure):
    summary = publication_fixture(tmp_path)
    if failure == "missing_entry":
        summary["artifacts"].pop("metrics.csv")
    elif failure == "extra_file":
        (tmp_path / "undeclared.bin").write_bytes(b"undeclared")
    elif failure == "artifact_bytes":
        (tmp_path / "metrics.csv").write_bytes(b"changed")
    elif failure == "summary_bytes":
        (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    else:
        path = tmp_path / "completion.json"
        completion = json.loads(path.read_text())
        if failure == "completion":
            completion["summary_sha256"] = "0" * 64
        else:
            completion["undeclared"] = True
        path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(RuntimeError):
        runner._validate_output(tmp_path, summary, "a" * 64)


@pytest.mark.parametrize("reconstruction_matches", [False, True])
def test_postfit_references_require_exact_p6_reconstruction(monkeypatch, reconstruction_matches):
    data, features = small_workload_data()
    spec = json.loads((ROOT / locker.PROTOCOL_PATH).read_text(encoding="utf-8"))
    identity = {
        "sample_ids": data.sample_ids,
        "recording_ids": data.scenarios,
        "labels": data.labels,
        "folds": data.folds,
        "long_valid": features.long_valid,
    }
    references = {phase: dict(identity) for phase in ("p3", "p5", "p6")}
    long = np.tile([0.2, 0.3, 0.5], (len(data.labels), 1))
    dual = np.tile([0.3, 0.2, 0.5], (len(data.labels), 1))
    spatial = np.tile([0.1, 0.3, 0.6], (len(data.labels), 1))
    for phase, values in (("p3", (long, dual)), ("p5", (long, spatial))):
        for name, probabilities in zip(
            spec["references"][phase]["post_fit_arrays"], values, strict=True
        ):
            references[phase][name] = probabilities
    p6 = runner.uniform_consensus([long, dual, spatial])
    if not reconstruction_matches:
        p6 = np.tile([0.4, 0.4, 0.2], (len(data.labels), 1))
    references["p6"][spec["references"]["p6"]["reference_array"]] = p6
    metrics = runner.p1.metrics(data.labels, p6)
    for key in ("macro_f1", "nll", "brier"):
        spec["references"]["p6"][f"reference_{key}"] = metrics[key]
    lock = {"protocol": spec, "postfit_references": {phase: phase for phase in references}}
    accessed = []

    def read_oof(_root, phase):
        accessed.append(phase)
        return references[phase]

    monkeypatch.setattr(runner, "_read_oof", read_oof)
    if reconstruction_matches:
        evidence = runner._load_postfit_references(ROOT, lock, data, features)
        np.testing.assert_array_equal(evidence.p6_reference, p6)
    else:
        with pytest.raises(RuntimeError, match="exact P6 component reconstruction failed"):
            runner._load_postfit_references(ROOT, lock, data, features)
    assert accessed == ["p3", "p5", "p6"]
