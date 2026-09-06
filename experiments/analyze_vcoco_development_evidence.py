"""Analyze the hash-bound V-COCO grouped-OOF development evidence table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics

CLASS_NAMES = ("sitting", "standing", "walking_running")
AREA_QUARTILES = ("Q1_small", "Q2", "Q3", "Q4_large")
RETAINED_METHODS = (
    "dino_flat_probability_stack",
    "dino_factorized_probability_stack",
    "dino_siglip_factorized_reliability_stack",
    "dino_siglip_linear_svm_control",
)
BUILDER_STATUS = "VCOCO_DEVELOPMENT_JOINED_EVIDENCE_COMPLETE"
ANALYSIS_STATUS = "VCOCO_DEVELOPMENT_EVIDENCE_ANALYSIS_COMPLETE"
BUILDER_SCOPE = "byte_locked_v2_train_val_rows_with_attested_v3_grouped_oof_predictions"
BUILDER_ROW_UNIT = "person_instance_with_image_id"
PROBABILITY_TOLERANCE = 1e-6
NLL_FLOOR = 1e-12
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_906
INTERPRETATION = "exploratory_association_not_causal"
GROUPED_OOF_SCOPE = "attested_v3_combined_development_image_grouped_oof_predictions"
BUILDER_ROW_ORDER = "locked_train_manifest_then_locked_val_manifest"
RETAINED_EVIDENCE_SHA256 = "ca32a6cd9f60524d9cbb122dce969ebede167cd6900d5772c757f1131a94ec9b"
# Fail closed until the scope-hardened builder output is rerun and this digest is replaced.
RETAINED_EVIDENCE_SUMMARY_SHA256 = (
    "1a5e7c4d307e6fa872a8fc25a352502bda93b35583cfb24c10c702c062dbf7e3"
)
RETAINED_BASELINE = "dino_flat_probability_stack"
RETAINED_CANDIDATE = "dino_siglip_factorized_reliability_stack"
RETAINED_SOURCE_SHA256 = MappingProxyType(
    {
        "protocol_lock": "3a90d6720a6cf5250b995820801199eca611706d514e7b5de2c83bea03f5a143",
        "train_manifest": "aa0919b4283dd683d1317b1cb1621af072800aa80d7a3bc1bd8a58271ff1514b",
        "val_manifest": "837d5470e616b374471dd1b2d2723ba9e3a9861683cdf47a7c030a133a608097",
        "prediction_summary": "ef8a7c8704489c8d2c0a64469eadd7a05e8d1d937ae1ff88acc984296379885a",
        "nested_oof_probabilities": (
            "a8af388b9ffd44037bbf3cb09d765db801ab4fefa9a9a0f61ffab8f499b9ea49"
        ),
        "nested_source_tag_metrics": (
            "eee7b3fe590bc3144d4c65f7bce16dca138ed5047f17b806b96403514edf503d"
        ),
        "candidate_grid": "cc3527dd7218499ca350d8ff37c4d77a19f5e8175ab583ebfe093ac7b295eba1",
        "candidate_lock": "462311d9d55581c838072cbce276c9717b922bf17ad01279059583419c22d827",
        "human_pilot_audit": ("f7f7e8eb5e8192cb6e275fb821847e0d252849de96ed51bca7d7aa0b0849c0f4"),
    }
)
RETAINED_PREDICTION_RUN_PROVENANCE = MappingProxyType(
    {
        "status": "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE",
        "endpoint": "source_tag_macro_f1",
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "outer_folds": 5,
        "inner_folds": 3,
        "stack_folds": 3,
        "source_sha256": {
            "candidate_grid": RETAINED_SOURCE_SHA256["candidate_grid"],
            "candidate_lock": RETAINED_SOURCE_SHA256["candidate_lock"],
            "human_pilot_audit": RETAINED_SOURCE_SHA256["human_pilot_audit"],
        },
        "used_artifact_sha256": {
            "nested_oof_probabilities.npz": RETAINED_SOURCE_SHA256["nested_oof_probabilities"],
            "nested_source_tag_metrics.csv": RETAINED_SOURCE_SHA256["nested_source_tag_metrics"],
        },
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(".runs/research_20260906/vcoco_development_evidence"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/research_20260906/vcoco_development_analysis"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Expected {name} SHA-256 is invalid")
    return value


def _validate_evidence_content_lock(
    evidence_dir: Path,
    *,
    expected_evidence_sha256: str,
    expected_summary_sha256: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {
        "development_joined.csv": evidence_dir / "development_joined.csv",
        "summary.json": evidence_dir / "summary.json",
    }
    expected = {
        "development_joined.csv": _require_sha256(
            expected_evidence_sha256,
            name="development evidence",
        ),
        "summary.json": _require_sha256(
            expected_summary_sha256,
            name="development evidence summary",
        ),
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    for name, digest in actual.items():
        if digest != expected[name]:
            label = (
                "Development evidence CSV hash"
                if name == "development_joined.csv"
                else "Development evidence summary"
            )
            raise RuntimeError(f"{label} differs from its independent content lock")
    return paths, actual


def _validate_content_unchanged(
    paths: Mapping[str, Path],
    before_sha256: Mapping[str, str],
    *,
    scope: str,
) -> None:
    if set(paths) != set(before_sha256):
        raise RuntimeError(f"{scope} stability-check keys differ")
    for name, path in paths.items():
        if sha256_file(path) != before_sha256[name]:
            raise RuntimeError(f"{scope} changed while being read: {name}")


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(value, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(
            temporary_path,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_columns(frame: pd.DataFrame, columns: set[str], *, source: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"Evidence table lacks required columns {sorted(missing)}: {source}")


def _as_boolean(series: pd.Series, *, column: str) -> np.ndarray:
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise RuntimeError(f"Evidence column {column} contains invalid booleans")
    return mapped.to_numpy(dtype=bool)


def _probability_columns(method: str) -> list[str]:
    return [f"{method}__p_{class_name}" for class_name in CLASS_NAMES]


def _probability_matrix(frame: pd.DataFrame, method: str) -> np.ndarray:
    values = frame.loc[:, _probability_columns(method)].to_numpy(dtype=np.float64)
    if values.shape != (len(frame), len(CLASS_NAMES)):
        raise RuntimeError(f"Evidence probabilities have the wrong shape for {method}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Evidence probabilities contain non-finite values for {method}")
    if (values < 0.0).any() or (values > 1.0 + PROBABILITY_TOLERANCE).any():
        raise RuntimeError(f"Evidence probabilities fall outside [0, 1] for {method}")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise RuntimeError(f"Evidence probabilities do not sum to one for {method}")
    return values


def _error_directions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    error_names = ("sitting", "standing", "locomotion")
    output = np.full(len(labels), "correct", dtype=object)
    wrong = labels != predictions
    output[wrong] = [
        f"{error_names[truth]}_to_{error_names[predicted]}"
        for truth, predicted in zip(labels[wrong], predictions[wrong], strict=True)
    ]
    return output


def _transition_labels(baseline_correct: np.ndarray, candidate_correct: np.ndarray) -> np.ndarray:
    return np.select(
        [
            baseline_correct & candidate_correct,
            ~baseline_correct & candidate_correct,
            baseline_correct & ~candidate_correct,
        ],
        ["both_correct", "rescued", "harmed"],
        default="both_wrong",
    )


def load_development_evidence(
    evidence_dir: Path,
    *,
    expected_evidence_sha256: str = RETAINED_EVIDENCE_SHA256,
    expected_summary_sha256: str = RETAINED_EVIDENCE_SUMMARY_SHA256,
    expected_source_sha256: Mapping[str, str] = RETAINED_SOURCE_SHA256,
    expected_prediction_run_provenance: Mapping = RETAINED_PREDICTION_RUN_PROVENANCE,
) -> tuple[pd.DataFrame, dict, str, str, dict[str, np.ndarray]]:
    """Load and independently validate the development-only builder output."""

    evidence_dir = evidence_dir.resolve()
    summary_path = evidence_dir / "summary.json"
    evidence_path = evidence_dir / "development_joined.csv"
    locked_paths, initial_hashes = _validate_evidence_content_lock(
        evidence_dir,
        expected_evidence_sha256=expected_evidence_sha256,
        expected_summary_sha256=expected_summary_sha256,
    )
    summary = _read_json(summary_path)
    if summary.get("status") != BUILDER_STATUS:
        raise RuntimeError("Evidence builder summary does not have completed status")
    if summary.get("scope") != BUILDER_SCOPE or summary.get("row_unit") != BUILDER_ROW_UNIT:
        raise RuntimeError("Evidence builder summary has an unexpected scope or row unit")
    if summary.get("row_order") != BUILDER_ROW_ORDER:
        raise RuntimeError("Evidence builder summary has an unexpected row order")
    if summary.get("official_v2_test_rows_read") != 0:
        raise RuntimeError("Evidence builder reports official-test rows read")
    if summary.get("official_v2_test_predictions_run") is not False:
        raise RuntimeError("Evidence builder reports official-test predictions")
    if summary.get("calibration_artifacts_read") != 0:
        raise RuntimeError("Evidence builder reports calibration-artifact access")
    expected_csv_hash = summary.get("artifact_sha256", {}).get(evidence_path.name)
    if expected_csv_hash != initial_hashes[evidence_path.name]:
        raise RuntimeError("Development evidence CSV hash differs from its builder summary")
    source_hashes = summary.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("Evidence builder summary lacks source hashes")
    if source_hashes != dict(expected_source_sha256):
        raise RuntimeError(
            "Evidence builder source hashes differ from the retained source contract"
        )
    prediction_run_provenance = summary.get("prediction_run_provenance")
    if prediction_run_provenance != dict(expected_prediction_run_provenance):
        raise RuntimeError(
            "Evidence builder prediction-run provenance is not the retained OOF contract"
        )
    if tuple(summary.get("class_names", ())) != CLASS_NAMES:
        raise RuntimeError("Evidence builder summary has an unexpected class order")

    baseline = str(summary.get("baseline", ""))
    candidate = str(summary.get("candidate", ""))
    methods = summary.get("methods")
    if (
        not baseline
        or not candidate
        or baseline == candidate
        or not isinstance(methods, dict)
        or baseline not in methods
        or candidate not in methods
    ):
        raise RuntimeError("Evidence builder summary has an invalid baseline/candidate pair")
    if baseline != RETAINED_BASELINE or candidate != RETAINED_CANDIDATE:
        raise RuntimeError(
            "Evidence builder baseline/candidate differ from the retained method pair"
        )
    expected_methods = {
        method: {"artifact": "nested_oof_probabilities.npz", "key": method}
        for method in RETAINED_METHODS
    }
    if methods != expected_methods:
        raise RuntimeError("Evidence builder methods differ from the retained prediction contract")

    required_columns = {
        "person_id",
        "image_id",
        "annotation_id",
        "file_name",
        "image_sha256",
        "v2_split",
        "label_3",
        "label_index",
        "actual_width",
        "actual_height",
        "bbox_area_fraction",
        "candidate_vs_baseline_transition",
        "candidate_minus_baseline_nll",
    }
    for method in (baseline, candidate):
        required_columns.update(_probability_columns(method))
        required_columns.update(
            {
                f"{method}__predicted_label",
                f"{method}__correct",
                f"{method}__error_direction",
                f"{method}__nll",
            }
        )
    frame = pd.read_csv(
        evidence_path,
        dtype={
            "person_id": str,
            "image_id": str,
            "annotation_id": str,
            "image_sha256": str,
        },
    )
    _require_columns(frame, required_columns, source=evidence_path)
    if len(frame) != int(summary.get("people", -1)):
        raise RuntimeError("Evidence row count differs from its builder summary")
    for column in ("person_id", "image_id", "annotation_id"):
        identities = frame[column].astype("string")
        if identities.isna().any() or identities.str.strip().eq("").fillna(True).any():
            raise RuntimeError(f"Evidence table contains a missing or blank {column}")
        frame[column] = identities.astype(str)
    if frame["person_id"].duplicated().any() or frame["annotation_id"].duplicated().any():
        raise RuntimeError("Evidence table contains duplicate person or annotation IDs")
    file_names = frame["file_name"].astype("string")
    if file_names.isna().any() or file_names.str.strip().eq("").fillna(True).any():
        raise RuntimeError("Evidence table contains a missing or blank file_name")
    frame["file_name"] = file_names.astype(str)
    for column in ("actual_width", "actual_height"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise RuntimeError(f"Evidence image-level {column} values must be finite and positive")
        frame[column] = values
    for column in ("file_name", "actual_width", "actual_height", "v2_split", "image_sha256"):
        if frame.groupby("image_id", sort=False)[column].nunique(dropna=False).gt(1).any():
            raise RuntimeError(
                f"Evidence image IDs map to inconsistent image-level {column} values"
            )
    if int(frame["image_id"].nunique()) != int(summary.get("source_images", -1)):
        raise RuntimeError("Evidence image count differs from its builder summary")
    if set(frame["v2_split"].astype(str)) != {"train", "val"}:
        raise RuntimeError("Evidence table is not exactly the locked train/val development union")
    split_counts = {
        str(name): int(count)
        for name, count in frame["v2_split"].value_counts().sort_index().items()
    }
    if split_counts != summary.get("v2_split_counts"):
        raise RuntimeError("Evidence split counts differ from the builder summary")
    image_hashes = frame["image_sha256"].astype("string")
    if (
        image_hashes.isna().any()
        or not image_hashes.str.fullmatch(r"[0-9a-f]{64}").fillna(False).all()
    ):
        raise RuntimeError("Evidence table contains an invalid image SHA-256")
    frame["image_sha256"] = image_hashes.astype(str)
    if frame.groupby("image_id", sort=False)["image_sha256"].nunique().gt(1).any():
        raise RuntimeError("An evidence image ID maps to multiple content hashes")
    if frame.groupby("image_sha256", sort=False)["image_id"].nunique().gt(1).any():
        raise RuntimeError("An evidence image content hash maps to multiple image IDs")
    if frame.groupby("image_id", sort=False)["v2_split"].nunique().gt(1).any():
        raise RuntimeError("An evidence source image crosses the train/val boundary")
    hashes_by_split = {
        split: set(frame.loc[frame["v2_split"].eq(split), "image_sha256"])
        for split in ("train", "val")
    }
    if hashes_by_split["train"].intersection(hashes_by_split["val"]):
        raise RuntimeError("Evidence image content crosses the train/val boundary")

    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    mapped_labels = frame["label_3"].map(label_map)
    if mapped_labels.isna().any():
        raise RuntimeError("Evidence table contains an unknown source label")
    numeric_labels = pd.to_numeric(frame["label_index"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(numeric_labels).all()
        or not np.equal(numeric_labels, np.floor(numeric_labels)).all()
    ):
        raise RuntimeError("Evidence label indices are not finite integers")
    labels = numeric_labels.astype(int)
    if not np.array_equal(labels, mapped_labels.to_numpy(dtype=int)):
        raise RuntimeError("Evidence label indices disagree with source labels")
    area = pd.to_numeric(frame["bbox_area_fraction"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(area).all() or (area <= 0.0).any() or (area > 1.0).any():
        raise RuntimeError("Evidence bounding-box areas must be finite and in (0, 1]")

    probabilities = {}
    correct_by_method = {}
    nll_by_method = {}
    for method in (baseline, candidate):
        values = _probability_matrix(frame, method)
        predictions = values.argmax(axis=1)
        predicted_labels = np.asarray([CLASS_NAMES[index] for index in predictions])
        if not np.array_equal(
            predicted_labels, frame[f"{method}__predicted_label"].astype(str).to_numpy()
        ):
            raise RuntimeError(f"Stored predicted labels do not replay for {method}")
        correct = predictions == labels
        if not np.array_equal(correct, _as_boolean(frame[f"{method}__correct"], column=method)):
            raise RuntimeError(f"Stored correctness does not replay for {method}")
        directions = _error_directions(labels, predictions)
        if not np.array_equal(
            directions, frame[f"{method}__error_direction"].astype(str).to_numpy()
        ):
            raise RuntimeError(f"Stored error directions do not replay for {method}")
        nll = -np.log(np.clip(values[np.arange(len(frame)), labels], NLL_FLOOR, 1.0))
        stored_nll = pd.to_numeric(frame[f"{method}__nll"], errors="raise").to_numpy(float)
        if not np.allclose(nll, stored_nll, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Stored NLL does not replay for {method}")
        probabilities[method] = values
        correct_by_method[method] = correct
        nll_by_method[method] = nll

    transitions = _transition_labels(correct_by_method[baseline], correct_by_method[candidate])
    if not np.array_equal(
        transitions, frame["candidate_vs_baseline_transition"].astype(str).to_numpy()
    ):
        raise RuntimeError("Stored candidate/baseline transitions do not replay")
    nll_delta = nll_by_method[candidate] - nll_by_method[baseline]
    stored_delta = pd.to_numeric(frame["candidate_minus_baseline_nll"], errors="raise").to_numpy(
        float
    )
    if not np.allclose(nll_delta, stored_delta, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Stored candidate-minus-baseline NLL does not replay")
    transition_counts = {
        str(name): int(count)
        for name, count in pd.Series(transitions).value_counts().sort_index().items()
    }
    if transition_counts != summary.get("candidate_vs_baseline_transition_counts"):
        raise RuntimeError("Transition counts differ from the builder summary")
    if not np.isclose(
        float(summary.get("mean_candidate_minus_baseline_nll", np.nan)),
        float(nll_delta.mean()),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Mean NLL delta differs from the builder summary")
    _validate_content_unchanged(
        locked_paths,
        initial_hashes,
        scope="Development evidence input",
    )
    return frame, summary, baseline, candidate, probabilities


def add_area_quartiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Assign one set of global development area quartiles."""

    output = frame.copy()
    try:
        quartiles, edges = pd.qcut(
            output["bbox_area_fraction"],
            4,
            labels=AREA_QUARTILES,
            retbins=True,
            duplicates="raise",
        )
    except ValueError as error:
        raise RuntimeError("Development area quartiles are not uniquely defined") from error
    output["area_quartile"] = quartiles.astype(str)
    if output["area_quartile"].isna().any() or set(output["area_quartile"]) != set(AREA_QUARTILES):
        raise RuntimeError("Development area quartiles are incomplete")
    return output, [float(value) for value in edges]


def _strata(frame: pd.DataFrame):
    yield "global", "all", np.ones(len(frame), dtype=bool)
    for split in ("train", "val"):
        yield "v2_split", split, frame["v2_split"].eq(split).to_numpy()
    for quartile in AREA_QUARTILES:
        yield "area_quartile", quartile, frame["area_quartile"].eq(quartile).to_numpy()


def _scope_note(scope: str) -> str:
    if scope == "v2_split":
        return (
            "v2_split_provenance_only; predictions remain attested v3 "
            "combined-development grouped OOF"
        )
    return GROUPED_OOF_SCOPE


def _fixed_class_metrics(labels: np.ndarray, probabilities: np.ndarray) -> tuple[dict, list[dict]]:
    aggregate = classification_metrics(labels, probabilities)
    classes = per_class_metrics(labels, probabilities, CLASS_NAMES)
    aggregate["macro_f1"] = float(np.mean([row["f1"] for row in classes]))
    aggregate["balanced_accuracy"] = float(np.mean([row["recall"] for row in classes]))
    total_support = sum(int(row["support"]) for row in classes)
    aggregate["weighted_f1"] = float(
        sum(float(row["f1"]) * int(row["support"]) for row in classes) / total_support
    )
    return aggregate, classes


def compute_stratum_tables(
    frame: pd.DataFrame,
    probabilities: Mapping[str, np.ndarray],
    *,
    baseline: str,
    candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute descriptive metrics and directional error transitions by stratum."""

    labels = frame["label_index"].to_numpy(dtype=int)
    predictions = {method: probabilities[method].argmax(axis=1) for method in (baseline, candidate)}
    correctness = {method: values == labels for method, values in predictions.items()}
    aggregate_rows = []
    class_rows = []
    transition_rows = []
    for scope, value, mask in _strata(frame):
        if not mask.any():
            raise RuntimeError(f"Empty requested development stratum: {scope}={value}")
        people = int(mask.sum())
        images = int(frame.loc[mask, "image_id"].nunique())
        for method in (baseline, candidate):
            metrics, classes = _fixed_class_metrics(labels[mask], probabilities[method][mask])
            aggregate_rows.append(
                {
                    "scope": scope,
                    "value": value,
                    "scope_note": _scope_note(scope),
                    "interpretation": INTERPRETATION,
                    "method": method,
                    "people": people,
                    "images": images,
                    **metrics,
                }
            )
            class_rows.extend(
                {
                    "scope": scope,
                    "value": value,
                    "scope_note": _scope_note(scope),
                    "interpretation": INTERPRETATION,
                    "method": method,
                    "people": people,
                    "images": images,
                    **row,
                }
                for row in classes
            )

        baseline_correct = correctness[baseline] & mask
        candidate_correct = correctness[candidate] & mask
        baseline_predictions = predictions[baseline]
        candidate_predictions = predictions[candidate]
        transition_rows.append(
            {
                "scope": scope,
                "value": value,
                "scope_note": _scope_note(scope),
                "interpretation": INTERPRETATION,
                "people": people,
                "images": images,
                "baseline_standing_to_locomotion": int(
                    (mask & (labels == 1) & (baseline_predictions == 2)).sum()
                ),
                "baseline_locomotion_to_standing": int(
                    (mask & (labels == 2) & (baseline_predictions == 1)).sum()
                ),
                "candidate_standing_to_locomotion": int(
                    (mask & (labels == 1) & (candidate_predictions == 2)).sum()
                ),
                "candidate_locomotion_to_standing": int(
                    (mask & (labels == 2) & (candidate_predictions == 1)).sum()
                ),
                "baseline_errors": int((mask & ~correctness[baseline]).sum()),
                "candidate_errors": int((mask & ~correctness[candidate]).sum()),
                "rescued": int((~correctness[baseline] & correctness[candidate] & mask).sum()),
                "harmed": int((correctness[baseline] & ~correctness[candidate] & mask).sum()),
                "both_correct": int((baseline_correct & candidate_correct).sum()),
                "both_wrong": int((mask & ~correctness[baseline] & ~correctness[candidate]).sum()),
            }
        )
    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(transition_rows),
    )


def _class_f1(confusions: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(confusions, axis1=-2, axis2=-1).astype(float)
    denominator = confusions.sum(axis=-2) + confusions.sum(axis=-1)
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )


def _interval(values: np.ndarray, point_estimate: float) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise RuntimeError("No finite cluster-bootstrap replicates are available")
    return {
        "point_estimate": float(point_estimate),
        "ci_95_low": float(np.quantile(finite, 0.025)),
        "ci_95_high": float(np.quantile(finite, 0.975)),
        "valid_resamples": int(len(finite)),
    }


def shared_image_cluster_bootstrap(
    frame: pd.DataFrame,
    probabilities: Mapping[str, np.ndarray],
    *,
    baseline: str,
    candidate: str,
    resamples: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Use one image-cluster resample stream for paired and scale-recall contrasts."""

    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    labels = frame["label_index"].to_numpy(dtype=int)
    groups = frame["image_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Image-cluster bootstrap requires at least two source images")
    group_lookup = {group: index for index, group in enumerate(unique_groups)}
    encoded_groups = np.asarray([group_lookup[group] for group in groups], dtype=int)
    classes = len(CLASS_NAMES)
    predictions = {method: probabilities[method].argmax(axis=1) for method in (baseline, candidate)}

    group_confusions = {}
    for method in (baseline, candidate):
        values = np.zeros((len(unique_groups), classes, classes), dtype=np.int64)
        np.add.at(values, (encoded_groups, labels, predictions[method]), 1)
        group_confusions[method] = values
    observed_f1 = {
        method: _class_f1(group_confusions[method].sum(axis=0, keepdims=True))[0]
        for method in (baseline, candidate)
    }

    candidate_nll = -np.log(
        np.clip(probabilities[candidate][np.arange(len(frame)), labels], NLL_FLOOR, 1.0)
    )
    baseline_nll = -np.log(
        np.clip(probabilities[baseline][np.arange(len(frame)), labels], NLL_FLOOR, 1.0)
    )
    nll_delta = candidate_nll - baseline_nll
    group_nll_delta = np.zeros(len(unique_groups), dtype=float)
    group_people = np.zeros(len(unique_groups), dtype=np.int64)
    np.add.at(group_nll_delta, encoded_groups, nll_delta)
    np.add.at(group_people, encoded_groups, 1)

    scale_correct = {}
    scale_support = np.zeros((len(unique_groups), 2, classes), dtype=np.int64)
    quartile_index = np.full(len(frame), -1, dtype=int)
    quartile_index[frame["area_quartile"].eq("Q1_small").to_numpy()] = 0
    quartile_index[frame["area_quartile"].eq("Q4_large").to_numpy()] = 1
    extreme = quartile_index >= 0
    np.add.at(
        scale_support,
        (encoded_groups[extreme], quartile_index[extreme], labels[extreme]),
        1,
    )
    for method in (baseline, candidate):
        correct = predictions[method] == labels
        values = np.zeros_like(scale_support)
        selected = extreme & correct
        np.add.at(
            values,
            (encoded_groups[selected], quartile_index[selected], labels[selected]),
            1,
        )
        scale_correct[method] = values
    observed_support = scale_support.sum(axis=0)
    if (observed_support == 0).any():
        raise RuntimeError("Q1/Q4 scale contrast lacks support for one or more classes")
    observed_scale = {
        method: scale_correct[method].sum(axis=0)[1] / observed_support[1]
        - scale_correct[method].sum(axis=0)[0] / observed_support[0]
        for method in (baseline, candidate)
    }

    f1_deltas = np.empty((resamples, classes), dtype=float)
    nll_deltas = np.empty(resamples, dtype=float)
    scale_deltas = {
        method: np.full((resamples, classes), np.nan, dtype=float)
        for method in (baseline, candidate)
    }
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    batch_size = 128
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        sampled = rng.integers(
            0,
            len(unique_groups),
            size=(stop - start, len(unique_groups)),
            dtype=np.int64,
        )
        digest.update(sampled.tobytes())
        candidate_confusions = group_confusions[candidate][sampled].sum(axis=1)
        baseline_confusions = group_confusions[baseline][sampled].sum(axis=1)
        f1_deltas[start:stop] = _class_f1(candidate_confusions) - _class_f1(baseline_confusions)
        sampled_people = group_people[sampled].sum(axis=1)
        nll_deltas[start:stop] = group_nll_delta[sampled].sum(axis=1) / sampled_people
        sampled_support = scale_support[sampled].sum(axis=1)
        for method in (baseline, candidate):
            sampled_correct = scale_correct[method][sampled].sum(axis=1)
            recalls = np.divide(
                sampled_correct,
                sampled_support,
                out=np.full_like(sampled_correct, np.nan, dtype=float),
                where=sampled_support != 0,
            )
            scale_deltas[method][start:stop] = recalls[:, 1, :] - recalls[:, 0, :]

    observed_f1_delta = observed_f1[candidate] - observed_f1[baseline]
    paired = {
        "status": "VCOCO_DEVELOPMENT_SHARED_IMAGE_CLUSTER_BOOTSTRAP_COMPLETE",
        "orientation": {
            "f1": "candidate_minus_baseline_positive_favors_candidate",
            "mean_nll": "candidate_minus_baseline_negative_favors_candidate",
        },
        "interpretation": INTERPRETATION,
        "prediction_scope": GROUPED_OOF_SCOPE,
        "macro_f1": _interval(f1_deltas.mean(axis=1), observed_f1_delta.mean()),
        "per_class_f1": {
            class_name: _interval(f1_deltas[:, index], observed_f1_delta[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "mean_nll": _interval(nll_deltas, nll_delta.mean()),
        "clusters": int(len(unique_groups)),
        "resamples": int(resamples),
        "seed": int(seed),
        "shared_resample_stream": True,
        "resample_index_sha256": digest.hexdigest(),
        "interval_method": "percentile_95_image_cluster_bootstrap",
    }
    scale_rows = []
    for method in (baseline, candidate):
        for index, class_name in enumerate(CLASS_NAMES):
            interval = _interval(scale_deltas[method][:, index], observed_scale[method][index])
            scale_rows.append(
                {
                    "method": method,
                    "class": class_name,
                    "contrast": "Q4_large_minus_Q1_small_recall",
                    "prediction_scope": GROUPED_OOF_SCOPE,
                    "interpretation": INTERPRETATION,
                    **interval,
                    "clusters": int(len(unique_groups)),
                    "resamples": int(resamples),
                    "seed": int(seed),
                }
            )
    return paired, pd.DataFrame(scale_rows)


def _write_json(path: Path, payload: Mapping) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def write_analysis(
    output_dir: Path,
    *,
    evidence_dir: Path,
    evidence_summary: Mapping,
    frame: pd.DataFrame,
    quartile_edges: list[float],
    aggregate_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    transitions: pd.DataFrame,
    paired_uncertainty: Mapping,
    scale_contrasts: pd.DataFrame,
    expected_evidence_sha256: str = RETAINED_EVIDENCE_SHA256,
    expected_summary_sha256: str = RETAINED_EVIDENCE_SUMMARY_SHA256,
) -> dict:
    """Write deterministic descriptive analysis outputs and a provenance summary."""

    output_dir = output_dir.resolve()
    evidence_dir = evidence_dir.resolve()
    if output_dir == evidence_dir or evidence_dir in output_dir.parents:
        raise RuntimeError("Analysis output directory overlaps the locked evidence input directory")
    locked_paths, write_input_hashes = _validate_evidence_content_lock(
        evidence_dir,
        expected_evidence_sha256=expected_evidence_sha256,
        expected_summary_sha256=expected_summary_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        evidence_summary.get("baseline") != RETAINED_BASELINE
        or evidence_summary.get("candidate") != RETAINED_CANDIDATE
    ):
        raise RuntimeError("Analysis writer requires the retained baseline/candidate pair")
    csv_artifacts = {
        "aggregate_metrics.csv": aggregate_metrics,
        "per_class_metrics.csv": class_metrics,
        "error_transitions.csv": transitions,
        "scale_recall_contrasts.csv": scale_contrasts,
    }
    artifact_hashes = {}
    for name, table in csv_artifacts.items():
        path = output_dir / name
        _atomic_write_csv(path, table)
        artifact_hashes[name] = sha256_file(path)
    uncertainty_path = output_dir / "paired_uncertainty.json"
    _write_json(uncertainty_path, paired_uncertainty)
    artifact_hashes[uncertainty_path.name] = sha256_file(uncertainty_path)

    baseline = str(evidence_summary["baseline"])
    candidate = str(evidence_summary["candidate"])
    global_metrics = aggregate_metrics[
        aggregate_metrics["scope"].eq("global") & aggregate_metrics["value"].eq("all")
    ].set_index("method")
    global_transitions = transitions[
        transitions["scope"].eq("global") & transitions["value"].eq("all")
    ].iloc[0]
    summary = {
        "status": ANALYSIS_STATUS,
        "scope": BUILDER_SCOPE,
        "interpretation": INTERPRETATION,
        "v2_split_interpretation": (
            "provenance_only; all rows are predictions from the attested v3 "
            "combined-development grouped OOF run"
        ),
        "area_quartile_definition": "global development bbox_area_fraction quartiles",
        "area_quartile_edges": quartile_edges,
        "people": int(len(frame)),
        "source_images": int(frame["image_id"].nunique()),
        "baseline": baseline,
        "candidate": candidate,
        "prediction_run_provenance": evidence_summary["prediction_run_provenance"],
        "headline": {
            "baseline_macro_f1": float(global_metrics.loc[baseline, "macro_f1"]),
            "candidate_macro_f1": float(global_metrics.loc[candidate, "macro_f1"]),
            "rescued": int(global_transitions["rescued"]),
            "harmed": int(global_transitions["harmed"]),
            "paired_macro_f1": paired_uncertainty["macro_f1"],
            "paired_mean_nll": paired_uncertainty["mean_nll"],
        },
        "bootstrap": {
            "unit": "image_id",
            "clusters": int(paired_uncertainty["clusters"]),
            "resamples": int(paired_uncertainty["resamples"]),
            "seed": int(paired_uncertainty["seed"]),
            "shared_resample_stream": True,
            "resample_index_sha256": paired_uncertainty["resample_index_sha256"],
        },
        "model_fitting_performed": False,
        "regression_performed": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "calibration_artifacts_read": 0,
        "source_sha256": {
            "development_evidence_summary": write_input_hashes["summary.json"],
            "development_joined.csv": write_input_hashes["development_joined.csv"],
        },
        "artifact_sha256": artifact_hashes,
    }
    _validate_content_unchanged(
        locked_paths,
        write_input_hashes,
        scope="Development evidence write input",
    )
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    frame, evidence_summary, baseline, candidate, probabilities = load_development_evidence(
        args.evidence_dir
    )
    frame, quartile_edges = add_area_quartiles(frame)
    aggregate, classes, transitions = compute_stratum_tables(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
    )
    uncertainty, scale = shared_image_cluster_bootstrap(
        frame,
        probabilities,
        baseline=baseline,
        candidate=candidate,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    summary = write_analysis(
        args.output_dir,
        evidence_dir=args.evidence_dir,
        evidence_summary=evidence_summary,
        frame=frame,
        quartile_edges=quartile_edges,
        aggregate_metrics=aggregate,
        class_metrics=classes,
        transitions=transitions,
        paired_uncertainty=uncertainty,
        scale_contrasts=scale,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
