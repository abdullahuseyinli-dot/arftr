"""Audit CPTR baseline preservation from locked development predictions without fitting.

The grouped cross-fit OOF scope is primary.  The fixed development-validation scope is
descriptive only.  This program deliberately has no calibration or confirmation input and
never opens feature stores, checkpoints, or training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

CLASS_NAMES = ("sitting", "standing", "walking_running")
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
STATUS = "OKUTAMA_CPTR_BASELINE_PRESERVATION_DIAGNOSTIC_COMPLETE"
DEFAULT_BOOTSTRAP_SEED = 20260919
PROBABILITY_TOLERANCE = 1e-6
EXPECTED_FOLDS = 5
EXPECTED_ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)
ATTESTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "5997e412ea8823bff9787aea9b825f465096dfbc25ccebad0bfe97104a4ae4fa"
)
OUTPUT_FILE_NAMES = (
    "paired_rows.csv",
    "slice_metrics.csv",
    "cluster_uncertainty.json",
    "recording_resample_indices.npz",
    "summary.json",
)

# These are content locks, not values inferred from mutable summaries at run time.  Every
# file is hashed before JSON/CSV/NPZ parsing begins.
EXPECTED_SOURCE_SHA256 = {
    "development_summary": "4bfde80da8f0ab0ad3ebc95482e31d9df9fbfa5104b69c4831b72ab8468961c4",
    "crossfit_oof_predictions": "8f45ea78960a52a863ea17dd0c40890b07602c5b6b7b11cb90841b76826e6ee8",
    "validation_predictions": "27f8027e7033de809156da87d3f92766414735c7f7e4f672120cd9dfb31500be",
    "crossfit_oof_anchor": "3c27148171708919087df64dc155db0493dfa5c11cd51ed8219527c7efa82775",
    "validation_anchor": "34dfeb8d0642e28b6f02db8ea3a8118917c2440d56c929db28e64f2324bbb616",
}

FOLD_RECORDINGS = {
    "fold-0": ("1.4", "2.2", "2.5"),
    "fold-1": ("1.5", "2.11"),
    "fold-2": ("1.10", "1.2"),
    "fold-3": ("2.7", "2.8"),
    "fold-4": ("1.11", "1.3"),
}
FOLD_BY_RECORDING = {
    recording: fold for fold, recordings in FOLD_RECORDINGS.items() for recording in recordings
}

EXPECTED_SCOPE = {
    "grouped_crossfit_oof": {
        "development_role": "train",
        "rows": 4977,
        "recordings": tuple(sorted(FOLD_BY_RECORDING)),
        "tracks": 444,
        "label_counts": {0: 734, 1: 2118, 2: 2125},
        "transition_rows": 398,
        "occluded_rows": 412,
    },
    "fixed_development_validation": {
        "development_role": "validation",
        "rows": 1383,
        "recordings": ("1.7", "2.4", "2.9"),
        "tracks": 128,
        "label_counts": {0: 193, 1: 513, 2: 677},
        "transition_rows": 101,
        "occluded_rows": 25,
    },
}


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: str


OOF_PREDICTION_SCHEMA = {
    "sample_ids": ArraySpec((4977,), "<U38"),
    "recording_ids": ArraySpec((4977,), "<U4"),
    "track_ids": ArraySpec((4977,), "<U11"),
    "labels": ArraySpec((4977,), "int64"),
    "transition_targets": ArraySpec((4977,), "bool"),
    "occlusion_targets": ArraySpec((4977,), "bool"),
    "probabilities": ArraySpec((4977, 3), "float32"),
    "baseline_probabilities": ArraySpec((4977, 3), "float64"),
}
VALIDATION_PREDICTION_SCHEMA = {
    "sample_ids": ArraySpec((1383,), "<U36"),
    "recording_ids": ArraySpec((1383,), "<U3"),
    "track_ids": ArraySpec((1383,), "<U9"),
    "labels": ArraySpec((1383,), "int64"),
    "probabilities": ArraySpec((1383, 3), "float32"),
    "baseline_probabilities": ArraySpec((1383, 3), "float32"),
    "transition_targets": ArraySpec((1383,), "bool"),
    "occlusion_targets": ArraySpec((1383,), "bool"),
}
OOF_ANCHOR_SCHEMA = {
    "sample_ids": ArraySpec((4977,), "<U38"),
    "recording_ids": ArraySpec((4977,), "<U4"),
    "track_ids": ArraySpec((4977,), "<U11"),
    "labels": ArraySpec((4977,), "int64"),
    "static_probabilities": ArraySpec((4977, 3), "float64"),
    "teacher_probabilities": ArraySpec((4977, 3), "float64"),
    "teacher_advantage_targets": ArraySpec((4977,), "int64"),
}
VALIDATION_ANCHOR_SCHEMA = {
    "sample_ids": ArraySpec((1383,), "<U36"),
    "labels": ArraySpec((1383,), "int64"),
    "static_probabilities": ArraySpec((1383, 3), "float32"),
    "teacher__real_all_frames": ArraySpec((1383, 3), "float32"),
    "teacher__occlusion_masked": ArraySpec((1383, 3), "float32"),
    "teacher__repeat_centre_frame": ArraySpec((1383, 3), "float32"),
    "teacher__reverse_temporal_order": ArraySpec((1383, 3), "float32"),
    "teacher__deterministic_temporal_shuffle": ArraySpec((1383, 3), "float32"),
    "teacher__zero_geometry": ArraySpec((1383, 3), "float32"),
    "teacher__zero_appearance_dynamics": ArraySpec((1383, 3), "float32"),
    "teacher__geometry_only": ArraySpec((1383, 3), "float32"),
    "teacher__coherent_camera_jitter": ArraySpec((1383, 3), "float32"),
}


@dataclass(frozen=True)
class InputPaths:
    development_summary: Path
    crossfit_oof_predictions: Path
    validation_predictions: Path
    crossfit_oof_anchor: Path
    validation_anchor: Path

    @classmethod
    def from_root(cls, root: Path) -> InputPaths:
        return cls(
            development_summary=root / ".runs/cptr/development_final/summary.json",
            crossfit_oof_predictions=root
            / ".runs/cptr/development_final/crossfit_oof_predictions.npz",
            validation_predictions=root / ".runs/cptr/development_final/validation_predictions.npz",
            crossfit_oof_anchor=root
            / ".runs/vcoco_v3/temporal/crossfit_aggregate/crossfit_targets.npz",
            validation_anchor=root / ".runs/cptr/baseline/baseline_and_diagnostic_predictions.npz",
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "development_summary": self.development_summary,
            "crossfit_oof_predictions": self.crossfit_oof_predictions,
            "validation_predictions": self.validation_predictions,
            "crossfit_oof_anchor": self.crossfit_oof_anchor,
            "validation_anchor": self.validation_anchor,
        }


@dataclass(frozen=True)
class ScopeData:
    scope: str
    development_role: str
    sample_ids: np.ndarray
    recording_ids: np.ndarray
    track_ids: np.ndarray
    folds: np.ndarray
    labels: np.ndarray
    transition: np.ndarray
    occluded: np.ndarray
    baseline: np.ndarray
    candidate: np.ndarray
    static: np.ndarray


def parse_args() -> argparse.Namespace:
    defaults = InputPaths.from_root(Path("."))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, default=defaults.development_summary)
    parser.add_argument(
        "--crossfit-oof-predictions", type=Path, default=defaults.crossfit_oof_predictions
    )
    parser.add_argument(
        "--validation-predictions", type=Path, default=defaults.validation_predictions
    )
    parser.add_argument("--crossfit-oof-anchor", type=Path, default=defaults.crossfit_oof_anchor)
    parser.add_argument("--validation-anchor", type=Path, default=defaults.validation_anchor)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/cptr/baseline_preservation_diagnostic"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def ordered_id_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(("\n".join(np.asarray(values).astype(str)) + "\n").encode()).hexdigest()


def capture_source_bytes(paths: InputPaths) -> dict[str, bytes]:
    """Read each explicitly declared source exactly once into an immutable byte snapshot."""

    captured: dict[str, bytes] = {}
    for name, path in paths.as_dict().items():
        try:
            captured[name] = path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"Could not capture required locked input {name}: {path}") from error
    return captured


def verify_source_hashes(captured: Mapping[str, bytes]) -> dict[str, str]:
    if set(captured) != set(EXPECTED_SOURCE_SHA256):
        missing = sorted(set(EXPECTED_SOURCE_SHA256).difference(captured))
        extra = sorted(set(captured).difference(EXPECTED_SOURCE_SHA256))
        raise RuntimeError(f"Captured source keys changed; missing={missing}, extra={extra}")
    actual: dict[str, str] = {}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        content = captured[name]
        if not isinstance(content, bytes):
            raise TypeError(f"Captured source {name} is not immutable bytes")
        actual[name] = sha256_bytes(content)
        if actual[name] != expected:
            raise RuntimeError(
                f"Locked SHA-256 mismatch for {name}: expected {expected}, got {actual[name]}"
            )
    return actual


def validate_development_summary(content: bytes) -> dict[str, Any]:
    try:
        summary = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("The locked CPTR development summary is not valid UTF-8 JSON") from error
    if not isinstance(summary, dict):
        raise RuntimeError("The locked CPTR development summary must be a JSON object")
    if summary.get("status") != "OKUTAMA_CPTR_DEVELOPMENT_COMPLETE_NO_PROMOTION":
        raise RuntimeError("The locked CPTR development summary has an unexpected status")
    for protected in ("calibration_samples_read", "confirmation_samples_read"):
        if summary.get(protected) != 0:
            raise RuntimeError(f"The development summary reports nonzero {protected}")
    if summary.get("folds") != EXPECTED_FOLDS:
        raise RuntimeError("The development summary fold count changed")
    if tuple(summary.get("seeds", ())) != EXPECTED_ENSEMBLE_SEEDS:
        raise RuntimeError("The development summary ensemble seeds changed")
    for scope, summary_key in (
        ("grouped_crossfit_oof", "grouped_crossfit_oof"),
        ("fixed_development_validation", "development_validation"),
    ):
        declared = summary.get(summary_key, {})
        expected = EXPECTED_SCOPE[scope]
        if declared.get("samples") != expected["rows"]:
            raise RuntimeError(f"Development-summary row count changed for {scope}")
        if declared.get("recordings") != len(expected["recordings"]):
            raise RuntimeError(f"Development-summary recording count changed for {scope}")
    expected_artifacts = {
        "crossfit_oof_predictions.npz": EXPECTED_SOURCE_SHA256["crossfit_oof_predictions"],
        "validation_predictions.npz": EXPECTED_SOURCE_SHA256["validation_predictions"],
    }
    artifacts = summary.get("artifact_sha256", {})
    for name, expected in expected_artifacts.items():
        if artifacts.get(name) != expected:
            raise RuntimeError(f"Development-summary artifact lock mismatch for {name}")
    sources = summary.get("source_sha256", {})
    if sources.get("manifest") != ATTESTED_DEVELOPMENT_MANIFEST_SHA256:
        raise RuntimeError("Development-summary manifest lock mismatch")
    if sources.get("baseline_predictions") != EXPECTED_SOURCE_SHA256["validation_anchor"]:
        raise RuntimeError("Development-summary validation-anchor lock mismatch")
    return summary


def validate_prediction_scope(
    prediction: Mapping[str, np.ndarray],
    scope: str,
    *,
    expectation: Mapping[str, Any] | None = None,
    fold_by_recording: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a locked development scope without consulting the mixed-role manifest."""

    expected = EXPECTED_SCOPE[scope] if expectation is None else expectation
    sample_ids = _unique_ids(prediction["sample_ids"], name=scope)
    recording_ids = np.asarray(prediction["recording_ids"]).astype(str)
    track_ids = np.asarray(prediction["track_ids"]).astype(str)
    labels = np.asarray(prediction["labels"], dtype=np.int64)
    transition = np.asarray(prediction["transition_targets"], dtype=bool)
    occluded = np.asarray(prediction["occlusion_targets"], dtype=bool)
    if len(sample_ids) != expected["rows"]:
        raise RuntimeError(f"{scope} prediction row count changed")
    for field, values in (("recording", recording_ids), ("track", track_ids)):
        if np.any(np.char.strip(values) == ""):
            raise RuntimeError(f"{scope} has blank {field} IDs")
    recordings = tuple(sorted(np.unique(recording_ids)))
    if recordings != expected["recordings"]:
        raise RuntimeError(f"{scope} recording IDs changed: {recordings}")
    if len(np.unique(track_ids)) != expected["tracks"]:
        raise RuntimeError(f"{scope} track count changed")
    track_recording_pairs = pd.DataFrame(
        {"track_id": track_ids, "recording_id": recording_ids}
    ).drop_duplicates()
    if track_recording_pairs["track_id"].duplicated().any():
        raise RuntimeError(f"{scope} track IDs cross recording boundaries")
    counts = {
        int(index): int(count)
        for index, count in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    if counts != expected["label_counts"]:
        raise RuntimeError(f"{scope} label counts changed: {counts}")
    if int(transition.sum()) != expected["transition_rows"]:
        raise RuntimeError(f"{scope} transition count changed")
    if int(occluded.sum()) != expected["occluded_rows"]:
        raise RuntimeError(f"{scope} occlusion count changed")
    if scope == "grouped_crossfit_oof":
        mapping = FOLD_BY_RECORDING if fold_by_recording is None else fold_by_recording
        if set(mapping) != set(recordings):
            raise RuntimeError("Grouped OOF recording-to-fold contract changed")
        folds = np.asarray([mapping[recording] for recording in recording_ids])
        observed = {
            fold: tuple(sorted(np.unique(recording_ids[folds == fold])))
            for fold in sorted(set(mapping.values()))
        }
        expected_folds = {
            fold: tuple(sorted(recording for recording, value in mapping.items() if value == fold))
            for fold in sorted(set(mapping.values()))
        }
        if observed != expected_folds:
            raise RuntimeError("Grouped OOF recording-to-fold membership changed")
        if expectation is None and expected_folds != FOLD_RECORDINGS:
            raise RuntimeError("Grouped OOF fold contract differs from the immutable lock")
        return sample_ids, folds
    return sample_ids, np.full(len(sample_ids), "not_applicable", dtype="<U14")


def validate_npz_arrays(
    arrays: dict[str, np.ndarray], schema: dict[str, ArraySpec], *, name: str
) -> None:
    if set(arrays) != set(schema):
        missing = sorted(set(schema).difference(arrays))
        extra = sorted(set(arrays).difference(schema))
        raise RuntimeError(f"{name} NPZ keys changed; missing={missing}, extra={extra}")
    for key, spec in schema.items():
        value = arrays[key]
        if value.shape != spec.shape:
            raise RuntimeError(
                f"{name}.{key} has wrong shape: expected {spec.shape}, got {value.shape}"
            )
        if str(value.dtype) != spec.dtype:
            raise RuntimeError(
                f"{name}.{key} has wrong dtype: expected {spec.dtype}, got {value.dtype}"
            )


def load_npz(content: bytes, schema: dict[str, ArraySpec], *, name: str) -> dict[str, np.ndarray]:
    expected_members = {f"{key}.npy" for key in schema}
    try:
        with ZipFile(BytesIO(content), mode="r") as archive:
            members = [member.filename for member in archive.infolist()]
    except (BadZipFile, OSError) as error:
        raise RuntimeError(f"{name} is not a valid NPZ archive") from error
    if len(members) != len(set(members)) or set(members) != expected_members:
        missing = sorted(expected_members.difference(members))
        extra = sorted(set(members).difference(expected_members))
        duplicates = sorted(member for member in set(members) if members.count(member) > 1)
        raise RuntimeError(
            f"{name} NPZ member names changed; missing={missing}, extra={extra}, "
            f"duplicates={duplicates}"
        )
    with np.load(BytesIO(content), allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    validate_npz_arrays(arrays, schema, name=name)
    return arrays


def validate_probabilities(values: np.ndarray, *, name: str) -> None:
    probabilities = np.asarray(values)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise RuntimeError(f"{name} is not an N-by-{len(CLASS_NAMES)} matrix")
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{name} contains non-finite values")
    if np.any(probabilities < -PROBABILITY_TOLERANCE) or np.any(
        probabilities > 1.0 + PROBABILITY_TOLERANCE
    ):
        raise RuntimeError(f"{name} has probabilities outside the tolerated [0, 1] bounds")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise RuntimeError(f"{name} probability rows do not sum to one")


def _unique_ids(values: np.ndarray, *, name: str) -> np.ndarray:
    identifiers = np.asarray(values).astype(str)
    if np.any(np.char.strip(identifiers) == ""):
        raise RuntimeError(f"{name} sample IDs contain blanks")
    if len(np.unique(identifiers)) != len(identifiers):
        raise RuntimeError(f"{name} sample IDs are not unique")
    return identifiers


def align_scope(
    *,
    scope: str,
    prediction: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    expectation: Mapping[str, Any] | None = None,
    fold_by_recording: Mapping[str, str] | None = None,
) -> ScopeData:
    expected = EXPECTED_SCOPE[scope] if expectation is None else expectation
    sample_ids, folds = validate_prediction_scope(
        prediction,
        scope,
        expectation=expectation,
        fold_by_recording=fold_by_recording,
    )

    anchor_ids = _unique_ids(anchor["sample_ids"], name=f"{scope} anchor")
    if set(anchor_ids) != set(sample_ids):
        raise RuntimeError(f"{scope} anchor/prediction ID sets differ")
    anchor_position = {sample_id: index for index, sample_id in enumerate(anchor_ids)}
    order = np.asarray([anchor_position[sample_id] for sample_id in sample_ids], dtype=int)
    if not np.array_equal(anchor["labels"][order], prediction["labels"]):
        raise RuntimeError(f"{scope} anchor labels disagree after sample-ID join")
    if scope == "grouped_crossfit_oof":
        if not np.array_equal(
            anchor["recording_ids"][order].astype(str), prediction["recording_ids"].astype(str)
        ):
            raise RuntimeError("Grouped OOF anchor recording IDs disagree after sample-ID join")
        if not np.array_equal(
            anchor["track_ids"][order].astype(str), prediction["track_ids"].astype(str)
        ):
            raise RuntimeError("Grouped OOF anchor track IDs disagree after sample-ID join")
        anchor_baseline = anchor["teacher_probabilities"][order]
    else:
        anchor_baseline = anchor["teacher__real_all_frames"][order]
    if not np.array_equal(anchor_baseline, prediction["baseline_probabilities"]):
        maximum = float(np.max(np.abs(anchor_baseline - prediction["baseline_probabilities"])))
        raise RuntimeError(
            f"{scope} saved baseline differs from locked teacher anchor (max={maximum})"
        )

    matrices = {
        "baseline": np.asarray(prediction["baseline_probabilities"], dtype=np.float64),
        "candidate": np.asarray(prediction["probabilities"], dtype=np.float64),
        "static": np.asarray(anchor["static_probabilities"][order], dtype=np.float64),
    }
    for name, values in matrices.items():
        validate_probabilities(values, name=f"{scope}.{name}")
    labels = np.asarray(prediction["labels"], dtype=np.int64)
    if set(np.unique(labels)) != set(range(len(CLASS_NAMES))):
        raise RuntimeError(f"{scope} does not contain the locked three-class order")
    return ScopeData(
        scope=scope,
        development_role=str(expected["development_role"]),
        sample_ids=sample_ids,
        recording_ids=prediction["recording_ids"].astype(str),
        track_ids=prediction["track_ids"].astype(str),
        folds=folds,
        labels=labels,
        transition=np.asarray(prediction["transition_targets"], dtype=bool),
        occluded=np.asarray(prediction["occlusion_targets"], dtype=bool),
        baseline=matrices["baseline"],
        candidate=matrices["candidate"],
        static=matrices["static"],
    )


def load_locked_development(paths: InputPaths) -> tuple[dict[str, ScopeData], dict[str, Any]]:
    captured = capture_source_bytes(paths)
    source_hashes = verify_source_hashes(captured)
    development_summary = validate_development_summary(captured["development_summary"])
    oof_prediction = load_npz(
        captured["crossfit_oof_predictions"],
        OOF_PREDICTION_SCHEMA,
        name="crossfit_oof_predictions",
    )
    validation_prediction = load_npz(
        captured["validation_predictions"],
        VALIDATION_PREDICTION_SCHEMA,
        name="validation_predictions",
    )
    oof_anchor = load_npz(
        captured["crossfit_oof_anchor"], OOF_ANCHOR_SCHEMA, name="crossfit_oof_anchor"
    )
    validation_anchor = load_npz(
        captured["validation_anchor"],
        VALIDATION_ANCHOR_SCHEMA,
        name="validation_anchor",
    )
    scopes = {
        "grouped_crossfit_oof": align_scope(
            scope="grouped_crossfit_oof",
            prediction=oof_prediction,
            anchor=oof_anchor,
        ),
        "fixed_development_validation": align_scope(
            scope="fixed_development_validation",
            prediction=validation_prediction,
            anchor=validation_anchor,
        ),
    }
    if set(scopes["grouped_crossfit_oof"].sample_ids).intersection(
        scopes["fixed_development_validation"].sample_ids
    ):
        raise RuntimeError("Development sample IDs cross analysis scopes")
    if set(scopes["grouped_crossfit_oof"].recording_ids).intersection(
        scopes["fixed_development_validation"].recording_ids
    ):
        raise RuntimeError("Development recording IDs cross analysis scopes")
    audit = {
        "source_sha256": source_hashes,
        "development_summary_status": development_summary["status"],
        "attested_development_manifest_sha256": ATTESTED_DEVELOPMENT_MANIFEST_SHA256,
        "source_capture": {
            "declared_source_count": len(captured),
            "path_reads_per_declared_source": 1,
            "hash_input": "captured_immutable_bytes",
            "parser_input": "same_captured_immutable_bytes",
            "npz_member_names_checked_before_array_materialization": True,
        },
        "protected_raw_access_by_current_analysis": {
            "rows_read": 0,
            "fields_read": 0,
            "arrays_read": 0,
        },
    }
    return scopes, audit


def class_f1(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    values = np.zeros(len(CLASS_NAMES), dtype=float)
    for class_index in range(len(CLASS_NAMES)):
        true_positive = np.sum((labels == class_index) & (predictions == class_index))
        denominator = np.sum(labels == class_index) + np.sum(predictions == class_index)
        values[class_index] = 2.0 * true_positive / denominator if denominator else 0.0
    return values


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    rows = np.arange(len(labels))
    one_hot = np.eye(len(CLASS_NAMES), dtype=float)[labels]
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "negative_log_likelihood": float(
            np.mean(-np.log(np.clip(probabilities[rows, labels], 1e-12, 1.0)))
        ),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def model_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    per_class = class_f1(labels, predictions)
    output = {
        **probability_metrics(labels, probabilities),
        "macro_f1": float(per_class.mean()),
    }
    output.update(
        {
            f"{class_name}_f1": float(per_class[index])
            for index, class_name in enumerate(CLASS_NAMES)
        }
    )
    return output


def pair_outcomes(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    baseline_correct = np.asarray(baseline).argmax(axis=1) == labels
    candidate_correct = np.asarray(candidate).argmax(axis=1) == labels
    outcome = np.full(len(labels), "both_wrong", dtype="<U12")
    outcome[baseline_correct & candidate_correct] = "both_correct"
    outcome[~baseline_correct & candidate_correct] = "rescued"
    outcome[baseline_correct & ~candidate_correct] = "harmed"
    return outcome, baseline_correct != candidate_correct


def static_direction_diagnostics(
    baseline: np.ndarray, candidate: np.ndarray, static: np.ndarray
) -> dict[str, np.ndarray]:
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    static = np.asarray(static, dtype=float)
    direction = static - baseline
    movement = candidate - baseline
    dot = np.sum(direction * movement, axis=1)
    direction_norm_sq = np.sum(direction**2, axis=1)
    movement_norm_sq = np.sum(movement**2, axis=1)
    defined = direction_norm_sq > 1e-24
    projection = np.full(len(baseline), np.nan, dtype=float)
    projection[defined] = dot[defined] / direction_norm_sq[defined]
    cosine_defined = defined & (movement_norm_sq > 1e-24)
    cosine = np.full(len(baseline), np.nan, dtype=float)
    cosine[cosine_defined] = dot[cosine_defined] / np.sqrt(
        direction_norm_sq[cosine_defined] * movement_norm_sq[cosine_defined]
    )
    return {
        "defined": defined,
        "projection": projection,
        "cosine": cosine,
        "toward_static": defined & (dot > 0.0),
        "closer_to_static": defined
        & (np.sum((candidate - static) ** 2, axis=1) < direction_norm_sq),
    }


def oracle_occlusion_fallback(data: ScopeData) -> np.ndarray:
    return np.where(data.occluded[:, None], data.baseline, data.candidate)


def build_paired_rows(data: ScopeData) -> pd.DataFrame:
    fallback = oracle_occlusion_fallback(data)
    baseline_predictions = data.baseline.argmax(axis=1)
    candidate_predictions = data.candidate.argmax(axis=1)
    fallback_predictions = fallback.argmax(axis=1)
    outcomes, _ = pair_outcomes(data.labels, data.baseline, data.candidate)
    row_index = np.arange(len(data.labels))
    one_hot = np.eye(len(CLASS_NAMES), dtype=float)[data.labels]
    baseline_nll = -np.log(np.clip(data.baseline[row_index, data.labels], 1e-12, 1.0))
    candidate_nll = -np.log(np.clip(data.candidate[row_index, data.labels], 1e-12, 1.0))
    baseline_brier = np.sum((data.baseline - one_hot) ** 2, axis=1)
    candidate_brier = np.sum((data.candidate - one_hot) ** 2, axis=1)
    direction = static_direction_diagnostics(data.baseline, data.candidate, data.static)
    frame = pd.DataFrame(
        {
            "scope": data.scope,
            "development_role": data.development_role,
            "sample_id": data.sample_ids,
            "recording_id": data.recording_ids,
            "track_id": data.track_ids,
            "fold": data.folds,
            "label_index": data.labels,
            "label": np.asarray(CLASS_NAMES)[data.labels],
            "transition_window": data.transition,
            "window_any_occluded": data.occluded,
            "baseline_prediction_index": baseline_predictions,
            "baseline_prediction": np.asarray(CLASS_NAMES)[baseline_predictions],
            "candidate_prediction_index": candidate_predictions,
            "candidate_prediction": np.asarray(CLASS_NAMES)[candidate_predictions],
            "oracle_occlusion_fallback_prediction_index": fallback_predictions,
            "oracle_occlusion_fallback_prediction": np.asarray(CLASS_NAMES)[fallback_predictions],
            "baseline_correct": baseline_predictions == data.labels,
            "candidate_correct": candidate_predictions == data.labels,
            "oracle_occlusion_fallback_correct": fallback_predictions == data.labels,
            "candidate_vs_baseline_outcome": outcomes,
            "candidate_changed_argmax": candidate_predictions != baseline_predictions,
            "candidate_minus_baseline_nll": candidate_nll - baseline_nll,
            "candidate_minus_baseline_brier": candidate_brier - baseline_brier,
            "candidate_l1_shift_from_baseline": np.sum(
                np.abs(data.candidate - data.baseline), axis=1
            ),
            "static_direction_defined": direction["defined"],
            "static_direction_projection": direction["projection"],
            "static_direction_cosine": direction["cosine"],
            "candidate_move_toward_static": direction["toward_static"],
            "candidate_closer_to_static": direction["closer_to_static"],
        }
    )
    matrices = {
        "baseline": data.baseline,
        "candidate": data.candidate,
        "static_anchor": data.static,
        "oracle_occlusion_fallback": fallback,
    }
    for model, probabilities in matrices.items():
        for index, class_name in enumerate(CLASS_NAMES):
            frame[f"{model}_p_{class_name}"] = probabilities[:, index]
    return frame


def _slice_definitions(data: ScopeData) -> list[tuple[str, str, np.ndarray]]:
    masks: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(len(data.labels), dtype=bool)),
        ("occlusion", "clear", ~data.occluded),
        ("occlusion", "occluded", data.occluded),
        ("transition", "non_transition", ~data.transition),
        ("transition", "transition", data.transition),
    ]
    for index, class_name in enumerate(CLASS_NAMES):
        masks.append(("class", class_name, data.labels == index))
    for recording in sorted(np.unique(data.recording_ids)):
        masks.append(("recording", recording, data.recording_ids == recording))
    if data.scope == "grouped_crossfit_oof":
        for fold in FOLD_RECORDINGS:
            masks.append(("fold", fold, data.folds == fold))
    for index, class_name in enumerate(CLASS_NAMES):
        masks.append(
            ("occlusion_x_class", f"clear__{class_name}", (~data.occluded) & (data.labels == index))
        )
        masks.append(
            (
                "occlusion_x_class",
                f"occluded__{class_name}",
                data.occluded & (data.labels == index),
            )
        )
    return masks


def build_slice_metrics(data: ScopeData) -> pd.DataFrame:
    fallback = oracle_occlusion_fallback(data)
    direction = static_direction_diagnostics(data.baseline, data.candidate, data.static)
    outcomes, _ = pair_outcomes(data.labels, data.baseline, data.candidate)
    candidate_predictions = data.candidate.argmax(axis=1)
    baseline_predictions = data.baseline.argmax(axis=1)
    matrices = {
        "baseline": data.baseline,
        "candidate": data.candidate,
        "oracle_occlusion_fallback": fallback,
    }
    rows: list[dict[str, Any]] = []
    for slice_type, slice_value, mask in _slice_definitions(data):
        if not np.any(mask):
            continue
        selected_labels = data.labels[mask]
        observed_classes = set(np.unique(selected_labels))
        all_label_scope = observed_classes == set(range(len(CLASS_NAMES)))
        true_class_index = (
            int(next(iter(observed_classes)))
            if len(observed_classes) == 1 and slice_type in {"class", "occlusion_x_class"}
            else None
        )
        metrics_by_model: dict[str, dict[str, float | int]] = {}
        for model, probabilities in matrices.items():
            selected_probabilities = probabilities[mask]
            metrics: dict[str, float | int] = probability_metrics(
                selected_labels, selected_probabilities
            )
            predictions = selected_probabilities.argmax(axis=1)
            if all_label_scope:
                metrics.update(model_metrics(selected_labels, selected_probabilities))
            elif true_class_index is not None:
                metrics["true_class_recall"] = float(np.mean(predictions == true_class_index))
                for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                    count = int(np.sum(predictions == predicted_index))
                    metrics[f"predicted_{predicted_name}_count"] = count
                    metrics[f"predicted_{predicted_name}_fraction"] = float(
                        count / len(predictions)
                    )
            metrics_by_model[model] = metrics
        row: dict[str, Any] = {
            "scope": data.scope,
            "scope_role": "primary" if data.scope == "grouped_crossfit_oof" else "descriptive",
            "slice_type": slice_type,
            "slice_value": slice_value,
            "rows": int(np.sum(mask)),
            "recordings": int(len(np.unique(data.recording_ids[mask]))),
            "tracks": int(len(np.unique(data.track_ids[mask]))),
            "class_metric_status": (
                "all_three_true_classes_supported"
                if all_label_scope
                else (
                    "true_class_recall_and_confusion_only"
                    if true_class_index is not None
                    else "f1_unsupported_missing_true_classes"
                )
            ),
            "true_class": CLASS_NAMES[true_class_index] if true_class_index is not None else None,
        }
        for model, metrics in metrics_by_model.items():
            row.update({f"{model}_{name}": value for name, value in metrics.items()})
        paired_metrics = ["accuracy", "negative_log_likelihood", "brier_score"]
        if all_label_scope:
            paired_metrics.extend(["macro_f1", *(f"{name}_f1" for name in CLASS_NAMES)])
        elif true_class_index is not None:
            paired_metrics.append("true_class_recall")
        for metric in paired_metrics:
            row[f"candidate_minus_baseline_{metric}"] = (
                metrics_by_model["candidate"][metric] - metrics_by_model["baseline"][metric]
            )
        selected_outcomes = outcomes[mask]
        for outcome in ("both_correct", "rescued", "harmed", "both_wrong"):
            row[outcome] = int(np.sum(selected_outcomes == outcome))
        row["double_faults"] = row["both_wrong"]
        row["candidate_argmax_changes"] = int(
            np.sum(candidate_predictions[mask] != baseline_predictions[mask])
        )
        l1 = np.sum(np.abs(data.candidate[mask] - data.baseline[mask]), axis=1)
        row["candidate_l1_shift_mean"] = float(np.mean(l1))
        row["candidate_l1_shift_median"] = float(np.median(l1))
        defined = mask & direction["defined"]
        row["static_direction_defined_rows"] = int(np.sum(defined))
        row["static_direction_projection_mean"] = (
            float(np.mean(direction["projection"][defined])) if np.any(defined) else None
        )
        row["static_direction_projection_median"] = (
            float(np.median(direction["projection"][defined])) if np.any(defined) else None
        )
        row["candidate_move_toward_static_fraction"] = (
            float(np.mean(direction["toward_static"][defined])) if np.any(defined) else None
        )
        row["candidate_closer_to_static_fraction"] = (
            float(np.mean(direction["closer_to_static"][defined])) if np.any(defined) else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _confusion_by_group(
    labels: np.ndarray, probabilities: np.ndarray, encoded_groups: np.ndarray, clusters: int
) -> np.ndarray:
    predictions = probabilities.argmax(axis=1)
    matrices = np.zeros((clusters, len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(matrices, (encoded_groups, labels, predictions), 1)
    return matrices


def _class_f1_from_confusion(matrices: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1).astype(float)
    denominator = matrices.sum(axis=-2) + matrices.sum(axis=-1)
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )


def _interval(
    values: np.ndarray,
    point: float,
    *,
    total_resamples: int | None = None,
) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    supported = values[np.isfinite(values)]
    total = len(values) if total_resamples is None else int(total_resamples)
    if not len(supported):
        return {
            "point_estimate": float(point),
            "ci_95_low": None,
            "ci_95_high": None,
            "resample_min": None,
            "resample_max": None,
            "valid_resamples": 0,
            "unsupported_resamples": total,
        }
    return {
        "point_estimate": float(point),
        "ci_95_low": float(np.quantile(supported, 0.025)),
        "ci_95_high": float(np.quantile(supported, 0.975)),
        "resample_min": float(np.min(supported)),
        "resample_max": float(np.max(supported)),
        "valid_resamples": int(len(supported)),
        "unsupported_resamples": int(total - len(supported)),
    }


def recording_resample_indices(
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    unique_groups = np.unique(np.asarray(groups).astype(str))
    if len(unique_groups) < 2:
        raise ValueError("Cluster bootstrap requires at least two recordings")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(unique_groups),
        size=(resamples, len(unique_groups)),
        dtype=np.int64,
    )
    return unique_groups, indices


def recording_resample_plan_sha256(
    recording_order: np.ndarray,
    sampled_recording_indices: np.ndarray,
) -> str:
    recording_order = np.asarray(recording_order).astype(str)
    indices = np.asarray(sampled_recording_indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"okutama-recording-bootstrap-v1\n")
    digest.update(("\n".join(recording_order) + "\n").encode("utf-8"))
    digest.update(np.asarray(indices.shape, dtype="<i8").tobytes())
    digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


def _validate_resample_plan(
    groups: np.ndarray,
    recording_order: np.ndarray,
    sampled_recording_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(np.asarray(groups).astype(str))
    recording_order = np.asarray(recording_order).astype(str)
    sampled = np.asarray(sampled_recording_indices)
    if not np.array_equal(recording_order, unique_groups):
        raise ValueError("Resample recording order disagrees with cluster inputs")
    if sampled.ndim != 2 or sampled.shape[1] != len(unique_groups) or not len(sampled):
        raise ValueError("Resample index matrix has the wrong shape")
    if not np.issubdtype(sampled.dtype, np.integer):
        raise ValueError("Resample indices must be integers")
    sampled = sampled.astype(np.int64, copy=False)
    if (sampled < 0).any() or (sampled >= len(unique_groups)).any():
        raise ValueError("Resample indices fall outside the recording order")
    return unique_groups, sampled


def recording_cluster_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    recording_order: np.ndarray,
    sampled_recording_indices: np.ndarray,
    challenger_name: str = "candidate",
    reference_name: str = "baseline",
    subgroup_mask: np.ndarray | None = None,
    estimand: str = "all_rows_in_scope",
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    groups = np.asarray(groups).astype(str)
    if not (len(labels) == len(candidate) == len(baseline) == len(groups)):
        raise ValueError("Cluster-bootstrap inputs do not align")
    unique_groups, sampled = _validate_resample_plan(
        groups, recording_order, sampled_recording_indices
    )
    selected = (
        np.ones(len(labels), dtype=bool)
        if subgroup_mask is None
        else np.asarray(subgroup_mask, dtype=bool)
    )
    if selected.shape != labels.shape:
        raise ValueError("Cluster-bootstrap subgroup mask does not align")
    if not selected.any():
        return {
            "method": "paired_recording_cluster_percentile_bootstrap",
            "available": False,
            "estimand": estimand,
            "reason": "subgroup_has_no_rows",
        }
    group_index = {group: index for index, group in enumerate(unique_groups)}
    encoded = np.asarray([group_index[group] for group in groups], dtype=int)
    selected_labels = labels[selected]
    selected_candidate = candidate[selected]
    selected_baseline = baseline[selected]
    selected_encoded = encoded[selected]
    candidate_confusion = _confusion_by_group(
        selected_labels, selected_candidate, selected_encoded, len(unique_groups)
    )
    baseline_confusion = _confusion_by_group(
        selected_labels, selected_baseline, selected_encoded, len(unique_groups)
    )
    row_index = np.arange(len(selected_labels))
    one_hot = np.eye(len(CLASS_NAMES), dtype=float)[selected_labels]
    candidate_row_nll = -np.log(np.clip(selected_candidate[row_index, selected_labels], 1e-12, 1.0))
    baseline_row_nll = -np.log(np.clip(selected_baseline[row_index, selected_labels], 1e-12, 1.0))
    candidate_row_brier = np.sum((selected_candidate - one_hot) ** 2, axis=1)
    baseline_row_brier = np.sum((selected_baseline - one_hot) ** 2, axis=1)
    counts = np.bincount(selected_encoded, minlength=len(unique_groups)).astype(float)
    candidate_nll_sum = np.bincount(
        selected_encoded, weights=candidate_row_nll, minlength=len(unique_groups)
    )
    baseline_nll_sum = np.bincount(
        selected_encoded, weights=baseline_row_nll, minlength=len(unique_groups)
    )
    candidate_brier_sum = np.bincount(
        selected_encoded, weights=candidate_row_brier, minlength=len(unique_groups)
    )
    baseline_brier_sum = np.bincount(
        selected_encoded, weights=baseline_row_brier, minlength=len(unique_groups)
    )
    resamples = len(sampled)
    macro_delta = np.empty(resamples, dtype=float)
    per_class_delta = np.empty((resamples, len(CLASS_NAMES)), dtype=float)
    nll_delta = np.empty(resamples, dtype=float)
    brier_delta = np.empty(resamples, dtype=float)
    macro_delta.fill(np.nan)
    per_class_delta.fill(np.nan)
    nll_delta.fill(np.nan)
    brier_delta.fill(np.nan)
    batch_size = 256
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        batch = sampled[start:stop]
        candidate_matrix = candidate_confusion[batch].sum(axis=1)
        baseline_matrix = baseline_confusion[batch].sum(axis=1)
        candidate_f1 = _class_f1_from_confusion(candidate_matrix)
        baseline_f1 = _class_f1_from_confusion(baseline_matrix)
        true_support = candidate_matrix.sum(axis=2)
        class_supported = true_support > 0
        per_class_values = candidate_f1 - baseline_f1
        per_class_values[~class_supported] = np.nan
        per_class_delta[start:stop] = per_class_values
        macro_supported = class_supported.all(axis=1)
        macro_values = np.full(len(batch), np.nan, dtype=float)
        macro_values[macro_supported] = per_class_values[macro_supported].mean(axis=1)
        macro_delta[start:stop] = macro_values
        sampled_rows = counts[batch].sum(axis=1)
        row_supported = sampled_rows > 0
        nll_values = np.full(len(batch), np.nan, dtype=float)
        brier_values = np.full(len(batch), np.nan, dtype=float)
        nll_values[row_supported] = (
            candidate_nll_sum[batch].sum(axis=1)[row_supported]
            - baseline_nll_sum[batch].sum(axis=1)[row_supported]
        ) / sampled_rows[row_supported]
        brier_values[row_supported] = (
            candidate_brier_sum[batch].sum(axis=1)[row_supported]
            - baseline_brier_sum[batch].sum(axis=1)[row_supported]
        ) / sampled_rows[row_supported]
        nll_delta[start:stop] = nll_values
        brier_delta[start:stop] = brier_values
    candidate_metrics = model_metrics(selected_labels, selected_candidate)
    baseline_metrics = model_metrics(selected_labels, selected_baseline)
    group_estimates: dict[str, Any] = {}
    for group in unique_groups:
        mask = selected & (groups == group)
        if not mask.any():
            continue
        candidate_group = model_metrics(labels[mask], candidate[mask])
        baseline_group = model_metrics(labels[mask], baseline[mask])
        all_classes_supported = set(np.unique(labels[mask])) == set(range(len(CLASS_NAMES)))
        group_estimates[str(group)] = {
            "rows": int(np.sum(mask)),
            "all_true_classes_supported": all_classes_supported,
            "macro_f1": (
                candidate_group["macro_f1"] - baseline_group["macro_f1"]
                if all_classes_supported
                else None
            ),
            "negative_log_likelihood": candidate_group["negative_log_likelihood"]
            - baseline_group["negative_log_likelihood"],
            "brier_score": candidate_group["brier_score"] - baseline_group["brier_score"],
        }
    return {
        "method": "paired_recording_cluster_percentile_bootstrap",
        "available": True,
        "estimand": estimand,
        "inference": "percentile_interval_only_no_null_p_value",
        "delta_definition": f"{challenger_name}_minus_{reference_name}",
        "proper_loss_orientation": "negative_favors_challenger",
        "clusters": int(len(unique_groups)),
        "clusters_with_rows": int(np.count_nonzero(counts)),
        "resamples": int(resamples),
        "resample_plan_sha256": recording_resample_plan_sha256(unique_groups, sampled),
        "point_all_true_classes_supported": set(np.unique(selected_labels))
        == set(range(len(CLASS_NAMES))),
        "macro_f1_resample_support": "all_three_true_classes_present",
        "loss_resample_support": "at_least_one_estimand_row_present",
        "macro_f1": _interval(
            macro_delta, candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]
        ),
        "per_class_f1": {
            class_name: _interval(
                per_class_delta[:, index],
                candidate_metrics[f"{class_name}_f1"] - baseline_metrics[f"{class_name}_f1"],
            )
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "negative_log_likelihood": _interval(
            nll_delta,
            candidate_metrics["negative_log_likelihood"]
            - baseline_metrics["negative_log_likelihood"],
        ),
        "brier_score": _interval(
            brier_delta, candidate_metrics["brier_score"] - baseline_metrics["brier_score"]
        ),
        "recording_estimates": group_estimates,
    }


def exact_recording_swap(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    maximum_recordings: int = 20,
    challenger_name: str = "candidate",
    reference_name: str = "baseline",
    subgroup_mask: np.ndarray | None = None,
    estimand: str = "all_rows_in_scope",
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    groups = np.asarray(groups).astype(str)
    if not (len(labels) == len(candidate) == len(baseline) == len(groups)):
        raise ValueError("Exact-swap inputs do not align")
    selected = (
        np.ones(len(labels), dtype=bool)
        if subgroup_mask is None
        else np.asarray(subgroup_mask, dtype=bool)
    )
    if selected.shape != labels.shape:
        raise ValueError("Exact-swap subgroup mask does not align")
    if not selected.any():
        return {
            "method": "exact_paired_recording_prediction_swap",
            "available": False,
            "estimand": estimand,
            "reason": "subgroup_has_no_rows",
        }
    # Only recordings that contribute rows to the estimand are exchangeable units.  Keeping
    # empty-slice recordings would duplicate every null assignment and overstate the nominal
    # permutation count (even though the tail proportion happens to remain unchanged).
    unique_groups = np.unique(groups[selected])
    if len(unique_groups) > maximum_recordings:
        return {
            "method": "exact_paired_recording_prediction_swap",
            "available": False,
            "estimand": estimand,
            "recordings": int(len(unique_groups)),
            "reason": f"more_than_{maximum_recordings}_recordings",
        }
    group_index = {group: index for index, group in enumerate(unique_groups)}
    selected_labels = labels[selected]
    selected_candidate = candidate[selected]
    selected_baseline = baseline[selected]
    selected_encoded = np.asarray([group_index[group] for group in groups[selected]], dtype=int)
    candidate_confusion = _confusion_by_group(
        selected_labels, selected_candidate, selected_encoded, len(unique_groups)
    )
    baseline_confusion = _confusion_by_group(
        selected_labels, selected_baseline, selected_encoded, len(unique_groups)
    )
    row_index = np.arange(len(selected_labels))
    one_hot = np.eye(len(CLASS_NAMES), dtype=float)[selected_labels]
    loss_deltas = {
        "negative_log_likelihood": -np.log(
            np.clip(selected_candidate[row_index, selected_labels], 1e-12, 1.0)
        )
        + np.log(np.clip(selected_baseline[row_index, selected_labels], 1e-12, 1.0)),
        "brier_score": np.sum((selected_candidate - one_hot) ** 2, axis=1)
        - np.sum((selected_baseline - one_hot) ** 2, axis=1),
    }
    group_loss_deltas = {
        name: np.bincount(selected_encoded, weights=values, minlength=len(unique_groups))
        for name, values in loss_deltas.items()
    }
    permutations = 1 << len(unique_groups)
    assignments = (
        (
            np.arange(permutations, dtype=np.uint64)[:, None]
            >> np.arange(len(unique_groups), dtype=np.uint64)[None, :]
        )
        & 1
    ).astype(bool)
    chosen_candidate = np.where(
        assignments[:, :, None, None],
        candidate_confusion[None, :, :, :],
        baseline_confusion[None, :, :, :],
    ).sum(axis=1)
    chosen_baseline = np.where(
        assignments[:, :, None, None],
        baseline_confusion[None, :, :, :],
        candidate_confusion[None, :, :, :],
    ).sum(axis=1)
    macro_distribution = _class_f1_from_confusion(chosen_candidate).mean(
        axis=1
    ) - _class_f1_from_confusion(chosen_baseline).mean(axis=1)
    signs = np.where(assignments, 1.0, -1.0)
    loss_distributions = {
        name: signs @ values / len(selected_labels) for name, values in group_loss_deltas.items()
    }
    metrics_candidate = model_metrics(selected_labels, selected_candidate)
    metrics_baseline = model_metrics(selected_labels, selected_baseline)
    points = {
        "macro_f1": metrics_candidate["macro_f1"] - metrics_baseline["macro_f1"],
        "negative_log_likelihood": metrics_candidate["negative_log_likelihood"]
        - metrics_baseline["negative_log_likelihood"],
        "brier_score": metrics_candidate["brier_score"] - metrics_baseline["brier_score"],
    }

    def exact_result(values: np.ndarray, point: float) -> dict[str, float | bool]:
        return {
            "point_estimate": float(point),
            "two_sided_p": float(np.mean(np.abs(values) >= abs(point) - 1e-15)),
            "null_min": float(np.min(values)),
            "null_max": float(np.max(values)),
        }

    macro_result = exact_result(macro_distribution, points["macro_f1"])
    macro_result["all_true_classes_supported"] = set(np.unique(selected_labels)) == set(
        range(len(CLASS_NAMES))
    )
    return {
        "method": "exact_paired_recording_prediction_swap",
        "available": True,
        "estimand": estimand,
        "inference": "exact_recording_level_prediction_swap_null",
        "delta_definition": f"{challenger_name}_minus_{reference_name}",
        "proper_loss_orientation": "negative_favors_challenger",
        "recordings": int(len(unique_groups)),
        "recordings_with_rows": int(len(np.unique(groups[selected]))),
        "rows": int(np.sum(selected)),
        "permutations": int(permutations),
        "point_estimate": macro_result["point_estimate"],
        "two_sided_p": macro_result["two_sided_p"],
        "macro_f1": macro_result,
        "negative_log_likelihood": exact_result(
            loss_distributions["negative_log_likelihood"], points["negative_log_likelihood"]
        ),
        "brier_score": exact_result(loss_distributions["brier_score"], points["brier_score"]),
    }


def _headline(data: ScopeData) -> dict[str, Any]:
    fallback = oracle_occlusion_fallback(data)
    baseline = model_metrics(data.labels, data.baseline)
    candidate = model_metrics(data.labels, data.candidate)
    oracle = model_metrics(data.labels, fallback)
    outcomes, _ = pair_outcomes(data.labels, data.baseline, data.candidate)
    return {
        "scope_role": "primary" if data.scope == "grouped_crossfit_oof" else "descriptive",
        "rows": int(len(data.labels)),
        "recordings": int(len(np.unique(data.recording_ids))),
        "tracks": int(len(np.unique(data.track_ids))),
        "ordered_sample_id_sha256_lf": ordered_id_sha256(data.sample_ids),
        "baseline": baseline,
        "candidate": candidate,
        "oracle_occlusion_fallback": oracle,
        "candidate_minus_baseline": {
            metric: candidate[metric] - baseline[metric]
            for metric in ("macro_f1", "accuracy", "negative_log_likelihood", "brier_score")
        },
        "oracle_occlusion_fallback_minus_baseline": {
            metric: oracle[metric] - baseline[metric]
            for metric in ("macro_f1", "accuracy", "negative_log_likelihood", "brier_score")
        },
        "oracle_occlusion_fallback_minus_candidate": {
            metric: oracle[metric] - candidate[metric]
            for metric in ("macro_f1", "accuracy", "negative_log_likelihood", "brier_score")
        },
        "pair_outcomes": {
            outcome: int(np.sum(outcomes == outcome))
            for outcome in ("both_correct", "rescued", "harmed", "both_wrong")
        },
        "candidate_argmax_changes": int(
            np.sum(data.candidate.argmax(axis=1) != data.baseline.argmax(axis=1))
        ),
    }


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    )
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def validate_output_isolated(paths: InputPaths, output_dir: Path) -> Path:
    """Reject any output location that could overwrite or contain a locked input."""

    resolved_output = output_dir.resolve()
    input_files = {name: path.resolve() for name, path in paths.as_dict().items()}
    output_targets = [resolved_output]
    for name in OUTPUT_FILE_NAMES:
        target = (resolved_output / name).resolve()
        output_targets.extend((target, target.with_suffix(target.suffix + ".tmp")))
    for output_target in output_targets:
        for input_name, input_path in input_files.items():
            if _paths_overlap(output_target, input_path) or _paths_overlap(
                output_target, input_path.parent
            ):
                raise RuntimeError(
                    f"Output path overlaps locked input location {input_name}: {output_target}"
                )
    return resolved_output


def fold_level_sensitivity(
    data: ScopeData,
    *,
    fold_recordings: Mapping[str, tuple[str, ...]] = FOLD_RECORDINGS,
) -> dict[str, Any]:
    if data.scope != "grouped_crossfit_oof":
        raise ValueError("Fold sensitivity is defined only for grouped cross-fit OOF data")
    all_candidate = model_metrics(data.labels, data.candidate)
    all_baseline = model_metrics(data.labels, data.baseline)
    overall_delta = all_candidate["macro_f1"] - all_baseline["macro_f1"]
    fold_rows: dict[str, Any] = {}
    for fold in fold_recordings:
        held = data.folds == fold
        retained = ~held
        held_candidate = model_metrics(data.labels[held], data.candidate[held])
        held_baseline = model_metrics(data.labels[held], data.baseline[held])
        retained_candidate = model_metrics(data.labels[retained], data.candidate[retained])
        retained_baseline = model_metrics(data.labels[retained], data.baseline[retained])
        fold_delta = held_candidate["macro_f1"] - held_baseline["macro_f1"]
        leave_one_fold_out_delta = retained_candidate["macro_f1"] - retained_baseline["macro_f1"]
        fold_rows[fold] = {
            "held_out_rows": int(np.sum(held)),
            "held_out_recordings": list(fold_recordings[fold]),
            "all_true_classes_supported": set(np.unique(data.labels[held]))
            == set(range(len(CLASS_NAMES))),
            "held_out_macro_f1_delta": float(fold_delta),
            "leave_one_fold_out_macro_f1_delta": float(leave_one_fold_out_delta),
            "leave_one_fold_out_influence_vs_overall": float(
                leave_one_fold_out_delta - overall_delta
            ),
        }
    held_deltas = np.asarray(
        [row["held_out_macro_f1_delta"] for row in fold_rows.values()], dtype=float
    )
    leave_one_out_deltas = np.asarray(
        [row["leave_one_fold_out_macro_f1_delta"] for row in fold_rows.values()], dtype=float
    )
    return {
        "estimand": "candidate_minus_baseline_macro_f1",
        "macro_f1_semantics": {
            "class_order": list(CLASS_NAMES),
            "averaging": "unweighted_mean_over_all_three_fixed_labels",
            "zero_division": 0.0,
            "missing_true_class_policy": (
                "Every locked class remains in the fold metric. Its F1 is zero when its "
                "precision/recall denominator is zero; a true-absent class predicted by the "
                "model also has zero F1. Inspect all_true_classes_supported before interpreting "
                "a held-out-fold delta."
            ),
        },
        "overall_delta": float(overall_delta),
        "folds": fold_rows,
        "held_out_fold_delta_min": float(held_deltas.min()),
        "held_out_fold_delta_max": float(held_deltas.max()),
        "leave_one_fold_out_delta_min": float(leave_one_out_deltas.min()),
        "leave_one_fold_out_delta_max": float(leave_one_out_deltas.max()),
        "held_out_fold_sign_reversal_present": bool(np.any(held_deltas * overall_delta < 0.0)),
        "conditional_inference_limit": (
            "Sensitivity is conditional on the retained predictions from the fixed five-fold, "
            "five-seed fitted ensemble; folds are not independent confirmation datasets and "
            "training-stage uncertainty is not propagated."
        ),
    }


def run_analysis(
    paths: InputPaths,
    output_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    output_dir = validate_output_isolated(paths, output_dir)
    scopes, source_audit = load_locked_development(paths)
    scope_order = ("grouped_crossfit_oof", "fixed_development_validation")
    paired = pd.concat([build_paired_rows(scopes[name]) for name in scope_order], ignore_index=True)
    slices = pd.concat(
        [build_slice_metrics(scopes[name]) for name in scope_order], ignore_index=True
    )
    uncertainty: dict[str, Any] = {
        "schema_version": 2,
        "delta_orientation": "challenger_minus_reference",
        "proper_loss_orientation": "negative deltas favor challenger",
        "bootstrap_unit": "recording_id",
        "bootstrap_inference": (
            "Paired recording-cluster percentile intervals only; the uncentered bootstrap is not "
            "a null distribution and no bootstrap p-value is reported."
        ),
        "conditional_inference_limit": (
            "All inference is conditional on retained predictions from the fixed five-fold, "
            "five-seed fitted ensemble and does not propagate training-stage uncertainty."
        ),
        "scopes": {},
    }
    uncertainty_seeds: dict[str, int] = {}
    resample_plans: dict[str, np.ndarray] = {}
    resample_plan_hashes: dict[str, str] = {}
    for scope_offset, scope in enumerate(scope_order):
        data = scopes[scope]
        fallback = oracle_occlusion_fallback(data)
        scope_seed = bootstrap_seed + scope_offset
        recording_order, sampled_recording_indices = recording_resample_indices(
            data.recording_ids,
            resamples=bootstrap_resamples,
            seed=scope_seed,
        )
        plan_hash = recording_resample_plan_sha256(recording_order, sampled_recording_indices)
        uncertainty_seeds[scope] = int(scope_seed)
        resample_plan_hashes[scope] = plan_hash
        resample_plans[f"{scope}__recording_order"] = recording_order
        resample_plans[f"{scope}__sampled_recording_indices"] = sampled_recording_indices
        comparisons = (
            ("candidate_vs_baseline", "candidate", data.candidate, "baseline", data.baseline),
            (
                "oracle_occlusion_fallback_vs_baseline",
                "oracle_occlusion_fallback",
                fallback,
                "baseline",
                data.baseline,
            ),
            (
                "oracle_occlusion_fallback_vs_candidate",
                "oracle_occlusion_fallback",
                fallback,
                "candidate",
                data.candidate,
            ),
        )
        comparison_results: dict[str, Any] = {}
        for (
            comparison_id,
            challenger_name,
            challenger,
            reference_name,
            reference,
        ) in comparisons:
            oracle_labeled = comparison_id.startswith("oracle_occlusion_fallback")
            comparison_results[comparison_id] = {
                "delta_orientation": f"{challenger_name}_minus_{reference_name}",
                "oracle_labeled": oracle_labeled,
                "nondeployable": oracle_labeled,
                "selection_eligible": False,
                "claim_limit": (
                    "Annotation-conditioned retrospective diagnostic; unavailable at deployment "
                    "and ineligible for model selection or promotion."
                    if oracle_labeled
                    else "Retrospective no-fit development comparison; not a promotion decision."
                ),
                "resample_plan_sha256": plan_hash,
                "cluster_bootstrap": recording_cluster_bootstrap(
                    data.labels,
                    challenger,
                    reference,
                    data.recording_ids,
                    recording_order=recording_order,
                    sampled_recording_indices=sampled_recording_indices,
                    challenger_name=challenger_name,
                    reference_name=reference_name,
                ),
                "exact_recording_swap": exact_recording_swap(
                    data.labels,
                    challenger,
                    reference,
                    data.recording_ids,
                    challenger_name=challenger_name,
                    reference_name=reference_name,
                ),
                "occluded_subgroup": {
                    "estimand": (
                        "challenger-minus-reference performance among locked development rows "
                        "with window_any_occluded=true"
                    ),
                    "cluster_bootstrap": recording_cluster_bootstrap(
                        data.labels,
                        challenger,
                        reference,
                        data.recording_ids,
                        recording_order=recording_order,
                        sampled_recording_indices=sampled_recording_indices,
                        challenger_name=challenger_name,
                        reference_name=reference_name,
                        subgroup_mask=data.occluded,
                        estimand="window_any_occluded_true_rows",
                    ),
                    "exact_recording_swap": exact_recording_swap(
                        data.labels,
                        challenger,
                        reference,
                        data.recording_ids,
                        challenger_name=challenger_name,
                        reference_name=reference_name,
                        subgroup_mask=data.occluded,
                        estimand="window_any_occluded_true_rows",
                    ),
                },
            }
        uncertainty["scopes"][scope] = {
            "scope_role": "primary" if scope == "grouped_crossfit_oof" else "descriptive",
            "resample_seed": int(scope_seed),
            "resample_plan_sha256": plan_hash,
            "recording_order": recording_order.tolist(),
            "precision_limited": scope == "fixed_development_validation",
            "precision_limit": (
                "Only three validation recordings are available; percentile intervals are "
                "precision-limited and the exact recording-swap p-value has 1/8 resolution."
                if scope == "fixed_development_validation"
                else None
            ),
            "comparisons": comparison_results,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_rows.csv"
    slices_path = output_dir / "slice_metrics.csv"
    uncertainty_path = output_dir / "cluster_uncertainty.json"
    resample_path = output_dir / "recording_resample_indices.npz"
    _atomic_csv(paired_path, paired)
    _atomic_csv(slices_path, slices)
    _atomic_npz(resample_path, resample_plans)
    uncertainty["resample_plan_artifact"] = {
        "file": resample_path.name,
        "sha256": sha256_file(resample_path),
        "scope_plan_sha256": resample_plan_hashes,
    }
    _atomic_json(uncertainty_path, uncertainty)
    artifacts = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in (paired_path, slices_path, uncertainty_path, resample_path)
    }
    summary: dict[str, Any] = {
        "schema_version": 2,
        "status": STATUS,
        "analysis_mode": "development_only_no_fit_replay",
        "candidate_id": "centre_short_parts",
        "baseline_id": "v3_temporal_8f_050s_five_seed_ensemble",
        "class_order": list(CLASS_NAMES),
        "scope_policy": {
            "primary": "grouped_crossfit_oof",
            "descriptive_only": "fixed_development_validation",
            "selection_or_promotion_decision": False,
        },
        "claim_limit": (
            "Retained development predictions support a baseline-preservation diagnostic only; "
            "they do not establish independent confirmation, deployment utility, or a causal gate."
        ),
        "oracle_occlusion_fallback": {
            "oracle_labeled": True,
            "requires_ground_truth_occlusion": True,
            "nondeployable": True,
            "selection_eligible": False,
            "promotion_eligible": False,
            "rule": "use exact locked temporal baseline when window_any_occluded is true; otherwise use candidate",
            "purpose": "annotation-conditioned retrospective fallback diagnostic",
            "claim_limit": (
                "This oracle-labeled comparison is nondeployable, was not used for selection, "
                "cannot support promotion, and is not guaranteed to upper-bound either model."
            ),
        },
        "static_direction_diagnostic": {
            "space": "three-class probability simplex",
            "direction": "static_anchor_minus_temporal_baseline",
            "movement": "candidate_minus_temporal_baseline",
            "projection": "dot(movement,direction)/dot(direction,direction)",
            "interpretation_limit": (
                "A descriptive probability-space signature; not a recovered logit gate coefficient "
                "and not causal evidence."
            ),
        },
        "source_sha256": source_audit["source_sha256"],
        "source_integrity": {
            "development_summary_status": source_audit["development_summary_status"],
            "baseline_anchor_join_max_absolute_difference": 0.0,
            "source_capture": source_audit["source_capture"],
            "mixed_role_development_manifest_opened": False,
            "development_manifest_sha256_attested_by_locked_summary": source_audit[
                "attested_development_manifest_sha256"
            ],
        },
        "analysis_access_accounting": {
            "protected_raw_rows_read_by_current_analysis": source_audit[
                "protected_raw_access_by_current_analysis"
            ]["rows_read"],
            "protected_raw_fields_read_by_current_analysis": source_audit[
                "protected_raw_access_by_current_analysis"
            ]["fields_read"],
            "protected_raw_arrays_read_by_current_analysis": source_audit[
                "protected_raw_access_by_current_analysis"
            ]["arrays_read"],
            "features_or_images_read_by_analysis": 0,
            "checkpoints_read_by_analysis": 0,
        },
        "external_operator_context": {
            "prior_interim_protected_raw_rows_scanned": 1979,
            "prior_interim_protected_rows_displayed": 5,
            "performed_by_current_analyzer": False,
            "used_by_current_analysis": False,
            "note": (
                "Before this refactored analysis, an interim implementation scanned 1,979 "
                "protected mixed-manifest raw rows and a separate operator view displayed five "
                "rows. The current analyzer has no mixed-manifest input and reads zero protected "
                "raw rows, fields, or arrays."
            ),
        },
        "headline": {scope: _headline(scopes[scope]) for scope in scope_order},
        "fold_level_sensitivity": fold_level_sensitivity(scopes["grouped_crossfit_oof"]),
        "uncertainty": {
            "resamples": int(bootstrap_resamples),
            "deterministic_scope_seeds": uncertainty_seeds,
            "shared_resample_plan_within_each_scope": True,
            "scope_plan_sha256": resample_plan_hashes,
            "resample_plan_artifact": resample_path.name,
            "recording_clusters": {
                scope: int(len(np.unique(scopes[scope].recording_ids))) for scope in scope_order
            },
        },
        "artifacts": artifacts,
        "model_fits": 0,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    paths = InputPaths(
        development_summary=args.development_summary.resolve(),
        crossfit_oof_predictions=args.crossfit_oof_predictions.resolve(),
        validation_predictions=args.validation_predictions.resolve(),
        crossfit_oof_anchor=args.crossfit_oof_anchor.resolve(),
        validation_anchor=args.validation_anchor.resolve(),
    )
    summary = run_analysis(
        paths,
        args.output_dir.resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    primary = summary["headline"]["grouped_crossfit_oof"]["candidate_minus_baseline"]
    validation = summary["headline"]["fixed_development_validation"]["candidate_minus_baseline"]
    print(f"status={summary['status']}")
    print(f"oof_macro_f1_delta={primary['macro_f1']:.12f}")
    print(f"validation_macro_f1_delta={validation['macro_f1']:.12f}")
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
