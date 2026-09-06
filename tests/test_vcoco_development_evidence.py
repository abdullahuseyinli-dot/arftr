import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.build_vcoco_development_evidence as vcoco_builder
from experiments.build_vcoco_development_evidence import (
    CLASS_NAMES,
    DEFAULT_BASELINE,
    DEFAULT_CANDIDATE,
    METHOD_KEYS,
    RETAINED_DEVELOPMENT_SHA256,
    RETAINED_PREDICTION_SHA256,
    RETAINED_PREDICTION_SOURCE_SHA256,
    build_evidence_frame,
    load_and_align_nested_predictions,
    load_locked_development,
    validate_output_directory,
    write_development_evidence,
)
from hac.metrics import classification_metrics
from hac.polar import sha256_file


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def manifest_row(
    person: str,
    image: str,
    annotation: str,
    split: str,
    label: str,
    *,
    boundary: bool = False,
) -> dict:
    factor = {
        "sitting": ("seated", "stationary", "not_applicable"),
        "standing": ("upright", "stationary", "not_applicable"),
        "walking_running": ("upright", "locomoting", "walking"),
    }[label]
    xmin = 0.5 if boundary else 10.0
    ymin = 8.0
    xmax = 70.0
    ymax = 72.0
    actual_width = 100
    actual_height = 80
    box_width = xmax - xmin
    box_height = ymax - ymin
    return {
        "person_id": person,
        "image_id": image,
        "annotation_id": annotation,
        "external_split": split,
        "selection_role": "adaptation" if split == "train" else "selection_and_calibration",
        "file_name": f"{image}.jpg",
        "sha256": hashlib.sha256(image.encode("utf-8")).hexdigest(),
        "source_actions": {
            "sitting": "sit",
            "standing": "stand",
            "walking_running": "stand|walk",
        }[label],
        "label_3": label,
        "image_level_unambiguous": True,
        "posture_label": factor[0],
        "motion_label": factor[1],
        "gait_label": factor[2],
        "bbox_xmin": xmin,
        "bbox_ymin": ymin,
        "bbox_xmax": xmax,
        "bbox_ymax": ymax,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "bbox_area_fraction": box_width * box_height / (actual_width * actual_height),
        "bbox_aspect_ratio": box_width / box_height,
        "bbox_center_x_fraction": (xmin + xmax) / (2.0 * actual_width),
        "bbox_center_y_fraction": (ymin + ymax) / (2.0 * actual_height),
        "person_pixel_height": box_height,
    }


def probability_rows(predictions: list[int]) -> np.ndarray:
    values = np.full((len(predictions), len(CLASS_NAMES)), 0.05, dtype=np.float64)
    values[np.arange(len(predictions)), predictions] = 0.90
    return values


def make_fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        [
            manifest_row("p0", "train-image-a", "a0", "train", "sitting", boundary=True),
            manifest_row("p1", "train-image-a", "a1", "train", "standing"),
            manifest_row("p2", "train-image-b", "a2", "train", "walking_running"),
        ]
    )
    validation = pd.DataFrame(
        [
            manifest_row("p3", "val-image-a", "a3", "val", "sitting"),
            manifest_row("p4", "val-image-b", "a4", "val", "standing"),
            manifest_row("p5", "val-image-c", "a5", "val", "walking_running"),
        ]
    )
    train_path = tmp_path / "vcoco_train_clean.csv"
    val_path = tmp_path / "vcoco_val_clean.csv"
    train.to_csv(train_path, index=False, lineterminator="\n")
    validation.to_csv(val_path, index=False, lineterminator="\n")
    protocol_path = tmp_path / "vcoco_v2_protocol_lock.json"
    write_json(
        protocol_path,
        {
            "status": "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING",
            "artifact_sha256": {
                train_path.name: sha256_file(train_path),
                val_path.name: sha256_file(val_path),
            },
        },
    )

    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=int)
    baseline = probability_rows([0, 2, 2, 0, 1, 1])
    candidate = probability_rows([0, 1, 2, 1, 1, 2])
    matrices = {
        DEFAULT_BASELINE: baseline,
        "dino_factorized_probability_stack": baseline.copy(),
        DEFAULT_CANDIDATE: candidate,
        "dino_siglip_linear_svm_control": baseline.copy(),
    }
    canonical_people = np.asarray([f"p{index}" for index in range(6)])
    canonical_images = np.asarray(
        [
            "train-image-a",
            "train-image-a",
            "train-image-b",
            "val-image-a",
            "val-image-b",
            "val-image-c",
        ]
    )
    shuffled = np.asarray([2, 0, 1, 5, 3, 4])
    run_dir = tmp_path / "nested_stacks"
    run_dir.mkdir()
    predictions_path = run_dir / "nested_oof_probabilities.npz"
    np.savez_compressed(
        predictions_path,
        person_ids=canonical_people[shuffled],
        image_ids=canonical_images[shuffled],
        labels=labels[shuffled],
        class_names=np.asarray(CLASS_NAMES),
        **{name: values[shuffled] for name, values in matrices.items()},
    )
    metrics_rows = []
    for method, values in matrices.items():
        metrics_rows.append({"family": method, **classification_metrics(labels, values)})
    metrics_path = run_dir / "nested_source_tag_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, lineterminator="\n")
    candidate_grid_path = tmp_path / "vcoco_v3_candidate_grid.json"
    candidate_lock_path = tmp_path / "candidate_grid_lock.json"
    human_pilot_audit_path = tmp_path / "human_pilot_audit.json"
    write_json(candidate_grid_path, {"status": "SYNTHETIC_CANDIDATE_GRID"})
    write_json(candidate_lock_path, {"status": "SYNTHETIC_CANDIDATE_LOCK"})
    write_json(human_pilot_audit_path, {"status": "SYNTHETIC_HUMAN_PILOT_AUDIT"})
    prediction_source_paths = {
        "candidate_grid": candidate_grid_path,
        "candidate_lock": candidate_lock_path,
        "human_pilot_audit": human_pilot_audit_path,
    }
    prediction_source_sha256 = {
        name: sha256_file(path) for name, path in prediction_source_paths.items()
    }
    summary_path = run_dir / "summary.json"
    write_json(
        summary_path,
        {
            "status": "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE",
            "endpoint": "source_tag_macro_f1",
            "human_pilot_labels_used_for_selection": False,
            "people": 6,
            "source_images": 5,
            "outer_folds": 5,
            "inner_folds": 3,
            "stack_folds": 3,
            "best_family": DEFAULT_CANDIDATE,
            "best_macro_f1": classification_metrics(labels, candidate)["macro_f1"],
            "official_v2_test_rows_read": 0,
            "official_v2_test_predictions_run": False,
            "source_sha256": prediction_source_sha256,
            "artifact_sha256": {
                predictions_path.name: sha256_file(predictions_path),
                metrics_path.name: sha256_file(metrics_path),
            },
        },
    )
    return {
        "protocol": protocol_path,
        "train": train_path,
        "val": val_path,
        "run_dir": run_dir,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "run_summary": summary_path,
        "prediction_source_paths": prediction_source_paths,
        "prediction_source_sha256": prediction_source_sha256,
        "development_sha256": {
            "protocol_lock": sha256_file(protocol_path),
            train_path.name: sha256_file(train_path),
            val_path.name: sha256_file(val_path),
        },
        "prediction_run_sha256": {
            summary_path.name: sha256_file(summary_path),
            predictions_path.name: sha256_file(predictions_path),
            metrics_path.name: sha256_file(metrics_path),
        },
    }


def refresh_development_lock(paths: dict) -> None:
    paths["development_sha256"] = {
        "protocol_lock": sha256_file(paths["protocol"]),
        paths["train"].name: sha256_file(paths["train"]),
        paths["val"].name: sha256_file(paths["val"]),
    }


def refresh_prediction_lock(paths: dict) -> None:
    paths["prediction_run_sha256"] = {
        paths["run_summary"].name: sha256_file(paths["run_summary"]),
        paths["predictions"].name: sha256_file(paths["predictions"]),
        paths["metrics"].name: sha256_file(paths["metrics"]),
    }


def rewrite_manifest(paths: dict, name: str, transform) -> None:
    manifest_path = paths[name]
    frame = pd.read_csv(
        manifest_path,
        dtype={"person_id": str, "image_id": str, "annotation_id": str},
    )
    transform(frame)
    frame.to_csv(manifest_path, index=False, lineterminator="\n")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    protocol["artifact_sha256"][manifest_path.name] = sha256_file(manifest_path)
    write_json(paths["protocol"], protocol)
    refresh_development_lock(paths)


def rewrite_prediction_summary(paths: dict, transform) -> None:
    summary = json.loads(paths["run_summary"].read_text(encoding="utf-8"))
    transform(summary)
    write_json(paths["run_summary"], summary)
    refresh_prediction_lock(paths)


def set_manifest_value(frame: pd.DataFrame, column: str, value) -> None:
    if isinstance(value, float) and not np.isfinite(value):
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value


def update_prediction_hash(paths: dict, *, refresh_lock: bool = True) -> None:
    summary = json.loads(paths["run_summary"].read_text(encoding="utf-8"))
    summary["artifact_sha256"]["nested_oof_probabilities.npz"] = sha256_file(paths["predictions"])
    write_json(paths["run_summary"], summary)
    if refresh_lock:
        refresh_prediction_lock(paths)


def rewrite_predictions(paths: dict, transform, *, refresh_lock: bool = True) -> None:
    with np.load(paths["predictions"], allow_pickle=False) as payload:
        values = {name: payload[name].copy() for name in payload.files}
    transform(values)
    np.savez_compressed(paths["predictions"], **values)
    update_prediction_hash(paths, refresh_lock=refresh_lock)


def load_locked_fixture_rows(paths: dict):
    rows, protocol, development_hashes = load_locked_development(
        paths["protocol"],
        paths["train"],
        paths["val"],
        expected_sha256=paths["development_sha256"],
    )
    return rows, protocol, development_hashes


def load_fixture_predictions(paths: dict, rows: pd.DataFrame):
    return load_and_align_nested_predictions(
        paths["run_dir"],
        rows,
        expected_run_sha256=paths["prediction_run_sha256"],
        expected_source_sha256=paths["prediction_source_sha256"],
        source_paths=paths["prediction_source_paths"],
    )


def load_fixture(paths: dict):
    rows, _, _ = load_locked_fixture_rows(paths)
    probabilities, metrics, summary, hashes = load_fixture_predictions(paths, rows)
    return rows, probabilities, metrics, summary, hashes


def evidence_source_paths(paths: dict) -> dict[str, Path]:
    return {
        "protocol_lock": paths["protocol"],
        "train_manifest": paths["train"],
        "val_manifest": paths["val"],
        "prediction_summary": paths["run_summary"],
        "nested_oof_probabilities": paths["predictions"],
        "nested_source_tag_metrics": paths["metrics"],
        **paths["prediction_source_paths"],
    }


def evidence_source_hashes(paths: dict, prediction_hashes: dict[str, str]) -> dict[str, str]:
    return {
        "protocol_lock": sha256_file(paths["protocol"]),
        "train_manifest": sha256_file(paths["train"]),
        "val_manifest": sha256_file(paths["val"]),
        **prediction_hashes,
    }


def test_builds_deterministic_person_level_development_evidence(tmp_path):
    paths = make_fixture(tmp_path / "inputs")
    rows, probabilities, metrics, prediction_summary, hashes = load_fixture(paths)

    evidence = build_evidence_frame(
        rows,
        probabilities,
        baseline=DEFAULT_BASELINE,
        candidate=DEFAULT_CANDIDATE,
    )
    assert evidence["person_id"].tolist() == [f"p{index}" for index in range(6)]
    assert evidence.loc[0, "people_in_image"] == 2
    assert bool(evidence.loc[0, "boundary_contact_1px"])
    assert evidence.loc[1, f"{DEFAULT_CANDIDATE}__predicted_label"] == "standing"
    np.testing.assert_allclose(evidence.loc[1, f"{DEFAULT_CANDIDATE}__nll"], -np.log(0.9))
    np.testing.assert_allclose(evidence.loc[1, f"{DEFAULT_CANDIDATE}__brier"], 0.015)
    np.testing.assert_allclose(evidence.loc[1, f"{DEFAULT_BASELINE}__nll"], -np.log(0.05))
    np.testing.assert_allclose(
        evidence.loc[1, "candidate_minus_baseline_nll"],
        -np.log(0.9) + np.log(0.05),
    )
    assert evidence["candidate_vs_baseline_transition"].value_counts().to_dict() == {
        "both_correct": 3,
        "rescued": 2,
        "harmed": 1,
    }

    source_hashes = evidence_source_hashes(paths, hashes)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = write_development_evidence(
        first,
        evidence,
        probabilities=probabilities,
        replayed_metrics=metrics,
        prediction_summary=prediction_summary,
        baseline=DEFAULT_BASELINE,
        candidate=DEFAULT_CANDIDATE,
        source_hashes=source_hashes,
        source_paths=evidence_source_paths(paths),
    )
    second_summary = write_development_evidence(
        second,
        evidence,
        probabilities=probabilities,
        replayed_metrics=metrics,
        prediction_summary=prediction_summary,
        baseline=DEFAULT_BASELINE,
        candidate=DEFAULT_CANDIDATE,
        source_hashes=source_hashes,
        source_paths=evidence_source_paths(paths),
    )

    assert (first / "development_joined.csv").read_bytes() == (
        second / "development_joined.csv"
    ).read_bytes()
    assert first_summary == second_summary
    assert first_summary["official_v2_test_rows_read"] == 0
    assert not first_summary["official_v2_test_predictions_run"]
    assert first_summary["calibration_artifacts_read"] == 0
    assert first_summary["prediction_run_provenance"] == {
        "status": "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE",
        "endpoint": "source_tag_macro_f1",
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "outer_folds": 5,
        "inner_folds": 3,
        "stack_folds": 3,
        "source_sha256": paths["prediction_source_sha256"],
        "used_artifact_sha256": {
            "nested_oof_probabilities.npz": sha256_file(paths["predictions"]),
            "nested_source_tag_metrics.csv": sha256_file(paths["metrics"]),
        },
    }


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda values: values["person_ids"].__setitem__(1, values["person_ids"][0]), "duplicate"),
        (lambda values: values["person_ids"].__setitem__(1, "extra-person"), "ID sets differ"),
        (lambda values: values["image_ids"].__setitem__(0, "wrong-image"), "person-to-image"),
        (lambda values: values["labels"].__setitem__(0, 0), "labels disagree"),
    ],
)
def test_rejects_prediction_identity_mismatches(tmp_path, transform, message):
    paths = make_fixture(tmp_path)
    rewrite_predictions(paths, transform)
    rows, _, _ = load_locked_fixture_rows(paths)

    with pytest.raises(RuntimeError, match=message):
        load_fixture_predictions(paths, rows)


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (
            lambda values: values.__setitem__(
                "class_names", np.asarray(["standing", "sitting", "walking_running"])
            ),
            "class order",
        ),
        (
            lambda values: values[DEFAULT_CANDIDATE].__setitem__((0, 0), np.nan),
            "non-finite",
        ),
        (
            lambda values: (
                values[DEFAULT_CANDIDATE].__setitem__((0, 0), -0.1),
                values[DEFAULT_CANDIDATE].__setitem__((0, 1), 1.05),
            ),
            "outside",
        ),
        (
            lambda values: values[DEFAULT_CANDIDATE].__setitem__(
                0, values[DEFAULT_CANDIDATE][0] * 0.9
            ),
            "sum to one",
        ),
        (
            lambda values: values.__setitem__(DEFAULT_CANDIDATE, values[DEFAULT_CANDIDATE][:, :2]),
            "wrong shape",
        ),
    ],
)
def test_rejects_invalid_prediction_probabilities(tmp_path, transform, message):
    paths = make_fixture(tmp_path)
    rewrite_predictions(paths, transform)
    rows, _, _ = load_locked_fixture_rows(paths)

    with pytest.raises(RuntimeError, match=message):
        load_fixture_predictions(paths, rows)


def test_rejects_provenance_drift_test_access_and_cross_split_images(tmp_path):
    drift = make_fixture(tmp_path / "drift")
    with drift["predictions"].open("ab") as stream:
        stream.write(b"drift")
    rows, _, _ = load_locked_fixture_rows(drift)
    with pytest.raises(RuntimeError, match="content lock drift"):
        load_fixture_predictions(drift, rows)

    test_access = make_fixture(tmp_path / "test-access")
    summary = json.loads(test_access["run_summary"].read_text(encoding="utf-8"))
    summary["official_v2_test_rows_read"] = 1
    write_json(test_access["run_summary"], summary)
    refresh_prediction_lock(test_access)
    rows, _, _ = load_locked_fixture_rows(test_access)
    with pytest.raises(RuntimeError, match="official-test rows"):
        load_fixture_predictions(test_access, rows)

    overlap = make_fixture(tmp_path / "overlap")
    validation = pd.read_csv(overlap["val"], dtype={"person_id": str, "image_id": str})
    validation.loc[0, "image_id"] = "train-image-a"
    validation.to_csv(overlap["val"], index=False, lineterminator="\n")
    protocol = json.loads(overlap["protocol"].read_text(encoding="utf-8"))
    protocol["artifact_sha256"]["vcoco_val_clean.csv"] = sha256_file(overlap["val"])
    write_json(overlap["protocol"], protocol)
    refresh_development_lock(overlap)
    with pytest.raises(RuntimeError, match="crosses"):
        load_locked_fixture_rows(overlap)


def test_retained_content_locks_are_explicit_and_independent():
    assert dict(RETAINED_DEVELOPMENT_SHA256) == {
        "protocol_lock": "3a90d6720a6cf5250b995820801199eca611706d514e7b5de2c83bea03f5a143",
        "vcoco_train_clean.csv": "aa0919b4283dd683d1317b1cb1621af072800aa80d7a3bc1bd8a58271ff1514b",
        "vcoco_val_clean.csv": "837d5470e616b374471dd1b2d2723ba9e3a9861683cdf47a7c030a133a608097",
    }
    assert dict(RETAINED_PREDICTION_SHA256) == {
        "summary.json": "ef8a7c8704489c8d2c0a64469eadd7a05e8d1d937ae1ff88acc984296379885a",
        "nested_oof_probabilities.npz": (
            "a8af388b9ffd44037bbf3cb09d765db801ab4fefa9a9a0f61ffab8f499b9ea49"
        ),
        "nested_source_tag_metrics.csv": (
            "eee7b3fe590bc3144d4c65f7bce16dca138ed5047f17b806b96403514edf503d"
        ),
    }
    assert dict(RETAINED_PREDICTION_SOURCE_SHA256) == {
        "candidate_grid": "cc3527dd7218499ca350d8ff37c4d77a19f5e8175ab583ebfe093ac7b295eba1",
        "candidate_lock": "462311d9d55581c838072cbce276c9717b922bf17ad01279059583419c22d827",
        "human_pilot_audit": ("f7f7e8eb5e8192cb6e275fb821847e0d252849de96ed51bca7d7aa0b0849c0f4"),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("endpoint", "official_test_macro_f1", "selection endpoint"),
        ("human_pilot_labels_used_for_selection", True, "human-pilot label use"),
        ("outer_folds", 1, "outer_folds"),
        ("inner_folds", 2, "inner_folds"),
        ("stack_folds", 4, "stack_folds"),
    ],
)
def test_rejects_adverse_prediction_protocol_metadata(tmp_path, field, value, message):
    paths = make_fixture(tmp_path)
    rewrite_prediction_summary(paths, lambda summary: summary.__setitem__(field, value))
    rows, _, _ = load_locked_fixture_rows(paths)

    with pytest.raises(RuntimeError, match=message):
        load_fixture_predictions(paths, rows)


def test_rejects_unbound_prediction_sources_and_unexpected_arrays(tmp_path):
    declaration_drift = make_fixture(tmp_path / "declaration")
    rewrite_prediction_summary(
        declaration_drift,
        lambda summary: summary["source_sha256"].__setitem__("candidate_grid", "0" * 64),
    )
    rows, _, _ = load_locked_fixture_rows(declaration_drift)
    with pytest.raises(RuntimeError, match="source hash differs"):
        load_fixture_predictions(declaration_drift, rows)

    source_drift = make_fixture(tmp_path / "source")
    write_json(
        source_drift["prediction_source_paths"]["candidate_grid"],
        {"status": "DRIFTED"},
    )
    rows, _, _ = load_locked_fixture_rows(source_drift)
    with pytest.raises(RuntimeError, match="content lock drift"):
        load_fixture_predictions(source_drift, rows)

    extra_array = make_fixture(tmp_path / "extra-array")
    rewrite_predictions(
        extra_array,
        lambda values: values.__setitem__(
            "official_v2_test_probabilities", np.full((1, len(CLASS_NAMES)), 1.0 / 3.0)
        ),
    )
    rows, _, _ = load_locked_fixture_rows(extra_array)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_fixture_predictions(extra_array, rows)


def test_rejects_content_hash_aliases_and_inconsistent_image_hashes(tmp_path):
    cross_split = make_fixture(tmp_path / "cross-split")
    train = pd.read_csv(cross_split["train"])
    train_hash = train.loc[0, "sha256"]
    rewrite_manifest(
        cross_split,
        "val",
        lambda frame: frame.loc.__setitem__((0, "sha256"), train_hash),
    )
    with pytest.raises(RuntimeError, match="content crosses"):
        load_locked_fixture_rows(cross_split)

    inconsistent = make_fixture(tmp_path / "inconsistent")
    different_hash = hashlib.sha256(b"different-content").hexdigest()
    rewrite_manifest(
        inconsistent,
        "train",
        lambda frame: frame.loc.__setitem__((1, "sha256"), different_hash),
    )
    with pytest.raises(RuntimeError, match="maps to multiple image content hashes"):
        load_locked_fixture_rows(inconsistent)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("file_name", "different.jpg"),
        ("actual_width", 101.0),
        ("actual_height", 81.0),
    ],
)
def test_rejects_inconsistent_image_level_metadata(tmp_path, column, value):
    paths = make_fixture(tmp_path)

    def make_inconsistent(frame):
        frame.loc[1, column] = value
        if column in {"actual_width", "actual_height"}:
            width = float(frame.loc[1, "actual_width"])
            height = float(frame.loc[1, "actual_height"])
            box_width = float(frame.loc[1, "bbox_xmax"] - frame.loc[1, "bbox_xmin"])
            box_height = float(frame.loc[1, "bbox_ymax"] - frame.loc[1, "bbox_ymin"])
            frame.loc[1, "bbox_area_fraction"] = box_width * box_height / (width * height)
            frame.loc[1, "bbox_center_x_fraction"] = (
                frame.loc[1, "bbox_xmin"] + frame.loc[1, "bbox_xmax"]
            ) / (2.0 * width)
            frame.loc[1, "bbox_center_y_fraction"] = (
                frame.loc[1, "bbox_ymin"] + frame.loc[1, "bbox_ymax"]
            ) / (2.0 * height)

    rewrite_manifest(paths, "train", make_inconsistent)
    with pytest.raises(RuntimeError, match=f"inconsistent image-level {column}"):
        load_locked_fixture_rows(paths)


def test_rejects_development_input_mutation_during_parse(tmp_path, monkeypatch):
    paths = make_fixture(tmp_path)
    original_read_json = vcoco_builder._read_json

    def read_then_mutate(path):
        payload = original_read_json(path)
        if Path(path).resolve() == paths["protocol"].resolve():
            with paths["protocol"].open("a", encoding="utf-8") as stream:
                stream.write(" ")
        return payload

    monkeypatch.setattr(vcoco_builder, "_read_json", read_then_mutate)
    with pytest.raises(RuntimeError, match="Development input changed while being read"):
        load_locked_fixture_rows(paths)


def test_rejects_prediction_input_mutation_during_parse(tmp_path, monkeypatch):
    paths = make_fixture(tmp_path)
    rows, _, _ = load_locked_fixture_rows(paths)
    original_read_json = vcoco_builder._read_json

    def read_then_mutate(path):
        payload = original_read_json(path)
        if Path(path).resolve() == paths["run_summary"].resolve():
            with paths["run_summary"].open("a", encoding="utf-8") as stream:
                stream.write(" ")
        return payload

    monkeypatch.setattr(vcoco_builder, "_read_json", read_then_mutate)
    with pytest.raises(RuntimeError, match="prediction run input changed while being read"):
        load_fixture_predictions(paths, rows)


@pytest.mark.parametrize(
    ("column", "value"),
    [("person_id", None), ("image_id", ""), ("annotation_id", "   ")],
)
def test_rejects_missing_or_blank_manifest_identities(tmp_path, column, value):
    paths = make_fixture(tmp_path)
    rewrite_manifest(
        paths,
        "train",
        lambda frame: frame.loc.__setitem__((0, column), value),
    )

    with pytest.raises(RuntimeError, match=f"missing or blank {column}"):
        load_locked_fixture_rows(paths)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("actual_width", 0.0, "image dimensions must be positive"),
        ("actual_height", np.inf, "non-finite"),
        ("bbox_xmax", 0.5, "ordered, nondegenerate, and in bounds"),
        ("bbox_ymax", 81.0, "ordered, nondegenerate, and in bounds"),
        ("bbox_area_fraction", 0.49, "bbox_area_fraction is inconsistent"),
        ("bbox_aspect_ratio", 0.5, "bbox_aspect_ratio is inconsistent"),
        ("bbox_center_x_fraction", 0.5, "bbox_center_x_fraction is inconsistent"),
        ("bbox_center_y_fraction", 0.6, "bbox_center_y_fraction is inconsistent"),
        ("person_pixel_height", 63.0, "person_pixel_height is inconsistent"),
    ],
)
def test_rejects_invalid_or_inconsistent_manifest_geometry(tmp_path, column, value, message):
    paths = make_fixture(tmp_path)
    rewrite_manifest(
        paths,
        "train",
        lambda frame: set_manifest_value(frame, column, value),
    )

    with pytest.raises(RuntimeError, match=message):
        load_locked_fixture_rows(paths)


@pytest.mark.parametrize(
    "file_name",
    [
        r"Z:\private\subject.jpg",
        "/private/subject.jpg",
        "../private/subject.jpg",
        "subdirectory/subject.jpg",
        r"subdirectory\subject.jpg",
    ],
)
def test_rejects_non_basename_manifest_file_names(tmp_path, file_name):
    paths = make_fixture(tmp_path)
    rewrite_manifest(
        paths,
        "train",
        lambda frame: frame.loc.__setitem__((0, "file_name"), file_name),
    )

    with pytest.raises(RuntimeError, match="safe basename"):
        load_locked_fixture_rows(paths)


def test_declared_methods_remain_the_matched_nested_families():
    assert METHOD_KEYS == (
        "dino_flat_probability_stack",
        "dino_factorized_probability_stack",
        "dino_siglip_factorized_reliability_stack",
        "dino_siglip_linear_svm_control",
    )


def test_rejects_identical_baseline_and_candidate(tmp_path):
    paths = make_fixture(tmp_path)
    rows, probabilities, _metrics, _summary, _hashes = load_fixture(paths)

    with pytest.raises(ValueError, match="different methods"):
        build_evidence_frame(
            rows,
            probabilities,
            baseline=DEFAULT_BASELINE,
            candidate=DEFAULT_BASELINE,
        )


def test_rejects_output_directory_inside_locked_input_location(tmp_path):
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    locked_input = locked_dir / "summary.json"
    write_json(locked_input, {"status": "LOCKED"})
    with pytest.raises(RuntimeError, match="output directory overlaps"):
        validate_output_directory(locked_dir, [locked_input])
    with pytest.raises(RuntimeError, match="output directory overlaps"):
        validate_output_directory(locked_dir / "derived", [locked_input])
    assert (
        validate_output_directory(tmp_path / "safe", [locked_input])
        == (tmp_path / "safe").resolve()
    )


@pytest.mark.parametrize("suffix", [Path(), Path("derived")])
def test_writer_enforces_locked_input_output_collision(tmp_path, suffix):
    paths = make_fixture(tmp_path)
    rows, probabilities, metrics, prediction_summary, hashes = load_fixture(paths)
    evidence = build_evidence_frame(
        rows,
        probabilities,
        baseline=DEFAULT_BASELINE,
        candidate=DEFAULT_CANDIDATE,
    )
    output_dir = paths["run_dir"] / suffix

    with pytest.raises(RuntimeError, match="output directory overlaps"):
        write_development_evidence(
            output_dir,
            evidence,
            probabilities=probabilities,
            replayed_metrics=metrics,
            prediction_summary=prediction_summary,
            baseline=DEFAULT_BASELINE,
            candidate=DEFAULT_CANDIDATE,
            source_hashes=evidence_source_hashes(paths, hashes),
            source_paths=evidence_source_paths(paths),
        )


def test_writer_rechecks_sources_and_preserves_existing_artifact_on_write_failure(
    tmp_path,
    monkeypatch,
):
    paths = make_fixture(tmp_path / "source-drift")
    rows, probabilities, metrics, prediction_summary, hashes = load_fixture(paths)
    evidence = build_evidence_frame(
        rows,
        probabilities,
        baseline=DEFAULT_BASELINE,
        candidate=DEFAULT_CANDIDATE,
    )
    source_hashes = evidence_source_hashes(paths, hashes)
    with paths["run_summary"].open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(RuntimeError, match="content lock drift"):
        write_development_evidence(
            tmp_path / "source-drift-output",
            evidence,
            probabilities=probabilities,
            replayed_metrics=metrics,
            prediction_summary=prediction_summary,
            baseline=DEFAULT_BASELINE,
            candidate=DEFAULT_CANDIDATE,
            source_hashes=source_hashes,
            source_paths=evidence_source_paths(paths),
        )

    stable_paths = make_fixture(tmp_path / "atomic")
    rows, probabilities, metrics, prediction_summary, hashes = load_fixture(stable_paths)
    evidence = build_evidence_frame(
        rows,
        probabilities,
        baseline=DEFAULT_BASELINE,
        candidate=DEFAULT_CANDIDATE,
    )
    output_dir = tmp_path / "atomic-output"
    output_dir.mkdir()
    evidence_path = output_dir / "development_joined.csv"
    evidence_path.write_bytes(b"stable-existing-artifact")

    def fail_to_csv(*_args, **_kwargs):
        raise RuntimeError("synthetic CSV failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)
    with pytest.raises(RuntimeError, match="synthetic CSV failure"):
        write_development_evidence(
            output_dir,
            evidence,
            probabilities=probabilities,
            replayed_metrics=metrics,
            prediction_summary=prediction_summary,
            baseline=DEFAULT_BASELINE,
            candidate=DEFAULT_CANDIDATE,
            source_hashes=evidence_source_hashes(stable_paths, hashes),
            source_paths=evidence_source_paths(stable_paths),
        )
    assert evidence_path.read_bytes() == b"stable-existing-artifact"
    assert not list(output_dir.glob(".development_joined.csv.*.tmp"))
