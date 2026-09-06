import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.analyze_vcoco_development_evidence as vcoco_analysis
from experiments.analyze_vcoco_development_evidence import (
    ANALYSIS_STATUS,
    AREA_QUARTILES,
    CLASS_NAMES,
    RETAINED_EVIDENCE_SHA256,
    RETAINED_PREDICTION_RUN_PROVENANCE,
    RETAINED_SOURCE_SHA256,
    add_area_quartiles,
    compute_stratum_tables,
    load_development_evidence,
    shared_image_cluster_bootstrap,
    write_analysis,
)
from hac.polar import sha256_file

BASELINE = "dino_flat_probability_stack"
CANDIDATE = "dino_siglip_factorized_reliability_stack"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def probability_rows(predictions: np.ndarray) -> np.ndarray:
    values = np.full((len(predictions), len(CLASS_NAMES)), 0.05, dtype=float)
    values[np.arange(len(predictions)), predictions] = 0.90
    return values


def error_directions(labels: np.ndarray, predictions: np.ndarray) -> list[str]:
    names = ("sitting", "standing", "locomotion")
    return [
        "correct" if truth == predicted else f"{names[truth]}_to_{names[predicted]}"
        for truth, predicted in zip(labels, predictions, strict=True)
    ]


def add_method_columns(
    frame: pd.DataFrame, method: str, probabilities: np.ndarray, labels: np.ndarray
) -> None:
    predictions = probabilities.argmax(axis=1)
    for index, class_name in enumerate(CLASS_NAMES):
        frame[f"{method}__p_{class_name}"] = probabilities[:, index]
    frame[f"{method}__predicted_label"] = [CLASS_NAMES[index] for index in predictions]
    frame[f"{method}__correct"] = predictions == labels
    frame[f"{method}__error_direction"] = error_directions(labels, predictions)
    frame[f"{method}__nll"] = -np.log(probabilities[np.arange(len(frame)), labels])


def make_evidence(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = 24
    labels = np.arange(rows) % len(CLASS_NAMES)
    baseline_predictions = labels.copy()
    baseline_predictions[[7, 19, 20, 23]] = [2, 2, 1, 1]
    candidate_predictions = baseline_predictions.copy()
    candidate_predictions[[20, 23]] = labels[[20, 23]]
    candidate_predictions[1] = 2
    baseline_probabilities = probability_rows(baseline_predictions)
    candidate_probabilities = probability_rows(candidate_predictions)
    image_ids = np.asarray([f"image-{index // 2:02d}" for index in range(rows)])
    frame = pd.DataFrame(
        {
            "person_id": [f"person-{index:02d}" for index in range(rows)],
            "image_id": image_ids,
            "annotation_id": [f"annotation-{index:02d}" for index in range(rows)],
            "file_name": [f"{image_id}.jpg" for image_id in image_ids],
            "image_sha256": [f"{index // 2:064x}" for index in range(rows)],
            "v2_split": ["train"] * 12 + ["val"] * 12,
            "label_3": [CLASS_NAMES[index] for index in labels],
            "label_index": labels,
            "actual_width": [100.0] * rows,
            "actual_height": [80.0] * rows,
            "bbox_area_fraction": np.linspace(0.01, 0.24, rows),
        }
    )
    add_method_columns(frame, BASELINE, baseline_probabilities, labels)
    add_method_columns(frame, CANDIDATE, candidate_probabilities, labels)
    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    frame["candidate_vs_baseline_transition"] = np.select(
        [
            baseline_correct & candidate_correct,
            ~baseline_correct & candidate_correct,
            baseline_correct & ~candidate_correct,
        ],
        ["both_correct", "rescued", "harmed"],
        default="both_wrong",
    )
    frame["candidate_minus_baseline_nll"] = frame[f"{CANDIDATE}__nll"] - frame[f"{BASELINE}__nll"]
    evidence_path = tmp_path / "development_joined.csv"
    frame.to_csv(
        evidence_path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    transition_counts = {
        str(name): int(count)
        for name, count in frame["candidate_vs_baseline_transition"]
        .value_counts()
        .sort_index()
        .items()
    }
    summary = {
        "status": "VCOCO_DEVELOPMENT_JOINED_EVIDENCE_COMPLETE",
        "scope": "byte_locked_v2_train_val_rows_with_attested_v3_grouped_oof_predictions",
        "row_unit": "person_instance_with_image_id",
        "row_order": "locked_train_manifest_then_locked_val_manifest",
        "people": rows,
        "source_images": int(frame["image_id"].nunique()),
        "v2_split_counts": {"train": 12, "val": 12},
        "class_names": list(CLASS_NAMES),
        "methods": {
            method: {"artifact": "nested_oof_probabilities.npz", "key": method}
            for method in (
                BASELINE,
                "dino_factorized_probability_stack",
                CANDIDATE,
                "dino_siglip_linear_svm_control",
            )
        },
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "candidate_vs_baseline_transition_counts": transition_counts,
        "mean_candidate_minus_baseline_nll": float(frame["candidate_minus_baseline_nll"].mean()),
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "calibration_artifacts_read": 0,
        "source_sha256": {
            "protocol_lock": "a" * 64,
            "train_manifest": "b" * 64,
            "val_manifest": "c" * 64,
            "prediction_summary": "d" * 64,
            "nested_oof_probabilities": "e" * 64,
            "nested_source_tag_metrics": "f" * 64,
            "candidate_grid": "1" * 64,
            "candidate_lock": "2" * 64,
            "human_pilot_audit": "3" * 64,
        },
        "artifact_sha256": {evidence_path.name: sha256_file(evidence_path)},
    }
    summary["prediction_run_provenance"] = {
        "status": "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE",
        "endpoint": "source_tag_macro_f1",
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "outer_folds": 5,
        "inner_folds": 3,
        "stack_folds": 3,
        "source_sha256": {
            "candidate_grid": summary["source_sha256"]["candidate_grid"],
            "candidate_lock": summary["source_sha256"]["candidate_lock"],
            "human_pilot_audit": summary["source_sha256"]["human_pilot_audit"],
        },
        "used_artifact_sha256": {
            "nested_oof_probabilities.npz": summary["source_sha256"]["nested_oof_probabilities"],
            "nested_source_tag_metrics.csv": summary["source_sha256"]["nested_source_tag_metrics"],
        },
    }
    write_json(tmp_path / "summary.json", summary)
    return tmp_path


def load_fixture(evidence_dir: Path):
    frame, summary, baseline, candidate, probabilities = load_fixture_evidence(evidence_dir)
    frame, edges = add_area_quartiles(frame)
    return frame, summary, baseline, candidate, probabilities, edges


def load_fixture_evidence(
    evidence_dir: Path,
    *,
    expected_evidence_sha256: str | None = None,
    expected_summary_sha256: str | None = None,
    expected_source_sha256: dict[str, str] | None = None,
    expected_prediction_run_provenance: dict | None = None,
):
    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    return load_development_evidence(
        evidence_dir,
        expected_evidence_sha256=expected_evidence_sha256
        or sha256_file(evidence_dir / "development_joined.csv"),
        expected_summary_sha256=expected_summary_sha256
        or sha256_file(evidence_dir / "summary.json"),
        expected_source_sha256=expected_source_sha256 or summary["source_sha256"],
        expected_prediction_run_provenance=expected_prediction_run_provenance
        or summary["prediction_run_provenance"],
    )


def rewrite_evidence(evidence_dir: Path, transform) -> None:
    evidence_path = evidence_dir / "development_joined.csv"
    frame = pd.read_csv(
        evidence_path,
        dtype={
            "person_id": str,
            "image_id": str,
            "annotation_id": str,
            "image_sha256": str,
        },
    )
    transform(frame)
    frame.to_csv(evidence_path, index=False, lineterminator="\n", float_format="%.17g")
    summary_path = evidence_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifact_sha256"][evidence_path.name] = sha256_file(evidence_path)
    write_json(summary_path, summary)


def evidence_hashes(evidence_dir: Path) -> tuple[str, str]:
    return (
        sha256_file(evidence_dir / "development_joined.csv"),
        sha256_file(evidence_dir / "summary.json"),
    )


def test_computes_global_split_and_area_class_metrics_and_transitions(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    frame, _summary, baseline, candidate, probabilities, edges = load_fixture(evidence_dir)

    aggregate, classes, transitions = compute_stratum_tables(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
    )

    assert len(edges) == 5
    assert set(frame["area_quartile"]) == set(AREA_QUARTILES)
    assert len(aggregate) == 14
    assert len(classes) == 42
    assert len(transitions) == 7
    assert set(aggregate["scope"]) == {"global", "v2_split", "area_quartile"}
    assert set(classes["class"]) == set(CLASS_NAMES)
    assert set(aggregate["interpretation"]) == {"exploratory_association_not_causal"}
    assert (
        aggregate.query("scope == 'v2_split'")["scope_note"].str.contains("provenance_only").all()
    )
    overall = transitions.query("scope == 'global' and value == 'all'").iloc[0]
    assert overall["rescued"] == 2
    assert overall["harmed"] == 1
    assert overall["baseline_standing_to_locomotion"] == 2
    assert overall["baseline_locomotion_to_standing"] == 2
    assert overall["candidate_standing_to_locomotion"] == 3
    assert overall["candidate_locomotion_to_standing"] == 0


def test_shared_cluster_bootstrap_is_deterministic_and_reports_scale_recall(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    frame, _summary, baseline, candidate, probabilities, _edges = load_fixture(evidence_dir)

    first, first_scale = shared_image_cluster_bootstrap(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
        resamples=500,
        seed=11,
    )
    second, second_scale = shared_image_cluster_bootstrap(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
        resamples=500,
        seed=11,
    )

    assert first == second
    pd.testing.assert_frame_equal(first_scale, second_scale)
    assert first["shared_resample_stream"]
    assert first["interpretation"] == "exploratory_association_not_causal"
    assert first["clusters"] == 12
    assert first["macro_f1"]["point_estimate"] > 0
    assert first["mean_nll"]["point_estimate"] == pytest.approx(
        frame["candidate_minus_baseline_nll"].mean()
    )
    baseline_locomotion = first_scale.query(
        "method == @baseline and `class` == 'walking_running'"
    ).iloc[0]
    candidate_locomotion = first_scale.query(
        "method == @candidate and `class` == 'walking_running'"
    ).iloc[0]
    assert baseline_locomotion["point_estimate"] == pytest.approx(-1.0)
    assert candidate_locomotion["point_estimate"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "INCOMPLETE", "completed status"),
        ("official_v2_test_rows_read", 1, "official-test rows"),
        ("official_v2_test_predictions_run", True, "official-test predictions"),
        ("calibration_artifacts_read", 1, "calibration-artifact"),
    ],
)
def test_rejects_ineligible_builder_summary(tmp_path, field, value, message):
    evidence_dir = make_evidence(tmp_path)
    summary_path = evidence_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match=message):
        load_fixture_evidence(evidence_dir)


def test_rejects_csv_hash_and_source_hash_drift(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    evidence_path = evidence_dir / "development_joined.csv"
    with evidence_path.open("a", encoding="utf-8") as stream:
        stream.write("drift\n")
    with pytest.raises(RuntimeError, match="CSV hash"):
        load_fixture_evidence(evidence_dir)

    evidence_dir = make_evidence(tmp_path / "source")
    summary_path = evidence_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    retained_source_contract = dict(summary["source_sha256"])
    summary["source_sha256"]["protocol_lock"] = "not-a-hash"
    write_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="source hashes differ"):
        load_fixture_evidence(evidence_dir, expected_source_sha256=retained_source_contract)


def test_rejects_builder_summary_hash_drift(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    evidence_sha256, summary_sha256 = evidence_hashes(evidence_dir)
    summary_path = evidence_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["people"] += 1
    write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="summary.*independent content lock"):
        load_fixture_evidence(
            evidence_dir,
            expected_evidence_sha256=evidence_sha256,
            expected_summary_sha256=summary_sha256,
        )


def test_rejects_nonretained_baseline_candidate_pair_even_when_rehashed(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    summary_path = evidence_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["baseline"] = "dino_factorized_probability_stack"
    write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="retained method pair"):
        load_fixture_evidence(evidence_dir)


def test_rejects_evidence_input_mutation_during_parse(tmp_path, monkeypatch):
    evidence_dir = make_evidence(tmp_path)
    evidence_sha256, summary_sha256 = evidence_hashes(evidence_dir)
    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    original_read_csv = vcoco_analysis.pd.read_csv

    def read_then_mutate(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        with (evidence_dir / "summary.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        return frame

    monkeypatch.setattr(vcoco_analysis.pd, "read_csv", read_then_mutate)
    with pytest.raises(RuntimeError, match="input changed while being read"):
        load_development_evidence(
            evidence_dir,
            expected_evidence_sha256=evidence_sha256,
            expected_summary_sha256=summary_sha256,
            expected_source_sha256=summary["source_sha256"],
            expected_prediction_run_provenance=summary["prediction_run_provenance"],
        )


def test_rejects_probability_and_derived_evidence_drift(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    evidence_path = evidence_dir / "development_joined.csv"
    frame = pd.read_csv(evidence_path)
    frame.loc[0, f"{CANDIDATE}__predicted_label"] = "standing"
    frame.to_csv(evidence_path, index=False, lineterminator="\n", float_format="%.17g")
    summary_path = evidence_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifact_sha256"][evidence_path.name] = sha256_file(evidence_path)
    write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="predicted labels do not replay"):
        load_fixture_evidence(evidence_dir)


def test_production_defaults_are_independent_retained_content_locks(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    with pytest.raises(RuntimeError, match="independent content lock"):
        load_development_evidence(evidence_dir)
    assert (
        RETAINED_EVIDENCE_SHA256
        == "ca32a6cd9f60524d9cbb122dce969ebede167cd6900d5772c757f1131a94ec9b"
    )
    assert RETAINED_SOURCE_SHA256["candidate_grid"] == (
        "cc3527dd7218499ca350d8ff37c4d77a19f5e8175ab583ebfe093ac7b295eba1"
    )
    assert RETAINED_PREDICTION_RUN_PROVENANCE["outer_folds"] == 5


def test_rejects_self_attested_prediction_run_provenance_drift(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    summary_path = evidence_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    retained_provenance = dict(summary["prediction_run_provenance"])
    summary["prediction_run_provenance"]["endpoint"] = "official_test_macro_f1"
    write_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="not the retained OOF contract"):
        load_fixture_evidence(
            evidence_dir,
            expected_prediction_run_provenance=retained_provenance,
        )


def test_rejects_fractional_label_index_even_when_rehashed(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    rewrite_evidence(evidence_dir, lambda frame: frame.__setitem__("label_index", 0.5))
    with pytest.raises(RuntimeError, match="not finite integers"):
        load_fixture_evidence(evidence_dir)


@pytest.mark.parametrize("column", ["person_id", "image_id", "annotation_id"])
def test_rejects_missing_evidence_identities_even_when_rehashed(tmp_path, column):
    evidence_dir = make_evidence(tmp_path)

    def remove_identity(frame):
        frame.loc[0, column] = None

    rewrite_evidence(evidence_dir, remove_identity)
    with pytest.raises(RuntimeError, match=f"missing or blank {column}"):
        load_fixture_evidence(evidence_dir)


def test_rejects_content_alias_across_splits_even_when_rehashed(tmp_path):
    evidence_dir = make_evidence(tmp_path)

    def alias_content(frame):
        frame.loc[frame["v2_split"].eq("val"), "image_sha256"] = frame.loc[
            frame["v2_split"].eq("train"), "image_sha256"
        ].iloc[0]

    rewrite_evidence(evidence_dir, alias_content)
    with pytest.raises(RuntimeError, match="content hash maps to multiple image IDs"):
        load_fixture_evidence(evidence_dir)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("file_name", "different.jpg"),
        ("actual_width", 101.0),
        ("actual_height", 81.0),
    ],
)
def test_rejects_inconsistent_evidence_image_metadata_even_when_rehashed(
    tmp_path,
    column,
    value,
):
    evidence_dir = make_evidence(tmp_path)

    def make_inconsistent(frame):
        frame.loc[1, column] = value

    rewrite_evidence(evidence_dir, make_inconsistent)
    with pytest.raises(RuntimeError, match=f"inconsistent image-level {column}"):
        load_fixture_evidence(evidence_dir)


def test_writes_deterministic_exploratory_analysis(tmp_path):
    evidence_dir = make_evidence(tmp_path / "evidence")
    evidence_sha256, summary_sha256 = evidence_hashes(evidence_dir)
    frame, evidence_summary, baseline, candidate, probabilities, edges = load_fixture(evidence_dir)
    aggregate, classes, transitions = compute_stratum_tables(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
    )
    paired, scale = shared_image_cluster_bootstrap(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
        resamples=100,
        seed=19,
    )

    summaries = []
    for name in ("first", "second"):
        summaries.append(
            write_analysis(
                tmp_path / name,
                evidence_dir=evidence_dir,
                evidence_summary=evidence_summary,
                frame=frame,
                quartile_edges=edges,
                aggregate_metrics=aggregate,
                class_metrics=classes,
                transitions=transitions,
                paired_uncertainty=paired,
                scale_contrasts=scale,
                expected_evidence_sha256=evidence_sha256,
                expected_summary_sha256=summary_sha256,
            )
        )
    assert summaries[0] == summaries[1]
    assert summaries[0]["status"] == ANALYSIS_STATUS
    assert summaries[0]["interpretation"] == "exploratory_association_not_causal"
    assert "provenance_only" in summaries[0]["v2_split_interpretation"]
    assert not summaries[0]["model_fitting_performed"]
    assert not summaries[0]["regression_performed"]
    assert summaries[0]["official_v2_test_rows_read"] == 0
    assert (tmp_path / "first" / "aggregate_metrics.csv").read_bytes() == (
        tmp_path / "second" / "aggregate_metrics.csv"
    ).read_bytes()
    for unsafe_output in (evidence_dir, evidence_dir / "derived"):
        with pytest.raises(RuntimeError, match="output directory overlaps"):
            write_analysis(
                unsafe_output,
                evidence_dir=evidence_dir,
                evidence_summary=evidence_summary,
                frame=frame,
                quartile_edges=edges,
                aggregate_metrics=aggregate,
                class_metrics=classes,
                transitions=transitions,
                paired_uncertainty=paired,
                scale_contrasts=scale,
                expected_evidence_sha256=evidence_sha256,
                expected_summary_sha256=summary_sha256,
            )


def test_analysis_writer_rechecks_evidence_sources(tmp_path):
    evidence_dir = make_evidence(tmp_path / "evidence")
    evidence_sha256, summary_sha256 = evidence_hashes(evidence_dir)
    frame, evidence_summary, baseline, candidate, probabilities, edges = load_fixture(evidence_dir)
    aggregate, classes, transitions = compute_stratum_tables(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
    )
    paired, scale = shared_image_cluster_bootstrap(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
        resamples=20,
        seed=23,
    )
    with (evidence_dir / "summary.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(RuntimeError, match="summary.*independent content lock"):
        write_analysis(
            tmp_path / "output",
            evidence_dir=evidence_dir,
            evidence_summary=evidence_summary,
            frame=frame,
            quartile_edges=edges,
            aggregate_metrics=aggregate,
            class_metrics=classes,
            transitions=transitions,
            paired_uncertainty=paired,
            scale_contrasts=scale,
            expected_evidence_sha256=evidence_sha256,
            expected_summary_sha256=summary_sha256,
        )
    assert not (tmp_path / "output").exists()


def test_analysis_csv_write_is_atomic(tmp_path, monkeypatch):
    evidence_dir = make_evidence(tmp_path / "evidence")
    evidence_sha256, summary_sha256 = evidence_hashes(evidence_dir)
    frame, evidence_summary, baseline, candidate, probabilities, edges = load_fixture(evidence_dir)
    aggregate, classes, transitions = compute_stratum_tables(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
    )
    paired, scale = shared_image_cluster_bootstrap(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
        resamples=20,
        seed=29,
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    aggregate_path = output_dir / "aggregate_metrics.csv"
    aggregate_path.write_bytes(b"stable-existing-artifact")

    def fail_to_csv(*_args, **_kwargs):
        raise RuntimeError("synthetic CSV failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)
    with pytest.raises(RuntimeError, match="synthetic CSV failure"):
        write_analysis(
            output_dir,
            evidence_dir=evidence_dir,
            evidence_summary=evidence_summary,
            frame=frame,
            quartile_edges=edges,
            aggregate_metrics=aggregate,
            class_metrics=classes,
            transitions=transitions,
            paired_uncertainty=paired,
            scale_contrasts=scale,
            expected_evidence_sha256=evidence_sha256,
            expected_summary_sha256=summary_sha256,
        )
    assert aggregate_path.read_bytes() == b"stable-existing-artifact"
    assert not list(output_dir.glob(".aggregate_metrics.csv.*.tmp"))


def test_bootstrap_rejects_nonpositive_resample_count(tmp_path):
    evidence_dir = make_evidence(tmp_path)
    frame, _summary, baseline, candidate, probabilities, _edges = load_fixture(evidence_dir)

    with pytest.raises(ValueError, match="positive"):
        shared_image_cluster_bootstrap(
            frame,
            probabilities,
            baseline=baseline,
            candidate=candidate,
            resamples=0,
            seed=1,
        )
