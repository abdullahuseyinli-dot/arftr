"""Build a provenance-bound, person-level V-COCO development evidence table."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file

CLASS_NAMES = ("sitting", "standing", "walking_running")
ERROR_CLASS_NAMES = ("sitting", "standing", "locomotion")
METHOD_KEYS = (
    "dino_flat_probability_stack",
    "dino_factorized_probability_stack",
    "dino_siglip_factorized_reliability_stack",
    "dino_siglip_linear_svm_control",
)
DEFAULT_BASELINE = "dino_flat_probability_stack"
DEFAULT_CANDIDATE = "dino_siglip_factorized_reliability_stack"
PROTOCOL_STATUS = "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING"
PREDICTION_STATUS = "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE"
OUTPUT_STATUS = "VCOCO_DEVELOPMENT_JOINED_EVIDENCE_COMPLETE"
OUTPUT_SCOPE = "byte_locked_v2_train_val_rows_with_attested_v3_grouped_oof_predictions"
PROBABILITY_TOLERANCE = 1e-6
NLL_FLOOR = 1e-12
GEOMETRY_RTOL = 1e-10
GEOMETRY_ATOL = 1e-12
PREDICTION_ENDPOINT = "source_tag_macro_f1"
EXPECTED_FOLD_COUNTS = MappingProxyType({"outer_folds": 5, "inner_folds": 3, "stack_folds": 3})
RETAINED_DEVELOPMENT_SHA256 = MappingProxyType(
    {
        "protocol_lock": "3a90d6720a6cf5250b995820801199eca611706d514e7b5de2c83bea03f5a143",
        "vcoco_train_clean.csv": "aa0919b4283dd683d1317b1cb1621af072800aa80d7a3bc1bd8a58271ff1514b",
        "vcoco_val_clean.csv": "837d5470e616b374471dd1b2d2723ba9e3a9861683cdf47a7c030a133a608097",
    }
)
RETAINED_PREDICTION_SHA256 = MappingProxyType(
    {
        "summary.json": "ef8a7c8704489c8d2c0a64469eadd7a05e8d1d937ae1ff88acc984296379885a",
        "nested_oof_probabilities.npz": (
            "a8af388b9ffd44037bbf3cb09d765db801ab4fefa9a9a0f61ffab8f499b9ea49"
        ),
        "nested_source_tag_metrics.csv": (
            "eee7b3fe590bc3144d4c65f7bce16dca138ed5047f17b806b96403514edf503d"
        ),
    }
)
RETAINED_PREDICTION_SOURCE_SHA256 = MappingProxyType(
    {
        "candidate_grid": "cc3527dd7218499ca350d8ff37c4d77a19f5e8175ab583ebfe093ac7b295eba1",
        "candidate_lock": "462311d9d55581c838072cbce276c9717b922bf17ad01279059583419c22d827",
        "human_pilot_audit": ("f7f7e8eb5e8192cb6e275fb821847e0d252849de96ed51bca7d7aa0b0849c0f4"),
    }
)
DEFAULT_PREDICTION_SOURCE_PATHS = MappingProxyType(
    {
        "candidate_grid": Path("experiments/vcoco_v3_candidate_grid.json"),
        "candidate_lock": Path(".runs/vcoco_v3/candidates/candidate_grid_lock.json"),
        "human_pilot_audit": Path(".runs/vcoco_v3/annotation/final/summary.json"),
    }
)

MANIFEST_REQUIRED_COLUMNS = (
    "person_id",
    "image_id",
    "annotation_id",
    "external_split",
    "selection_role",
    "file_name",
    "sha256",
    "source_actions",
    "label_3",
    "image_level_unambiguous",
    "posture_label",
    "motion_label",
    "gait_label",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "actual_width",
    "actual_height",
    "bbox_area_fraction",
    "bbox_aspect_ratio",
    "bbox_center_x_fraction",
    "bbox_center_y_fraction",
    "person_pixel_height",
)
NUMERIC_MANIFEST_COLUMNS = (
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "actual_width",
    "actual_height",
    "bbox_area_fraction",
    "bbox_aspect_ratio",
    "bbox_center_x_fraction",
    "bbox_center_y_fraction",
    "person_pixel_height",
)
OUTPUT_MANIFEST_COLUMNS = (
    "person_id",
    "image_id",
    "annotation_id",
    "v2_split",
    "selection_role",
    "file_name",
    "image_sha256",
    "source_actions",
    "label_3",
    "label_index",
    "image_level_unambiguous",
    "posture_label",
    "motion_label",
    "gait_label",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "actual_width",
    "actual_height",
    "bbox_area_fraction",
    "bbox_aspect_ratio",
    "bbox_center_x_fraction",
    "bbox_center_y_fraction",
    "person_pixel_height",
    "people_in_image",
    "boundary_contact_1px",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_v2_protocol_lock.json"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_train_clean.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_val_clean.csv"),
    )
    parser.add_argument(
        "--prediction-run-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/nested_stacks"),
    )
    parser.add_argument(
        "--candidate-grid",
        type=Path,
        default=Path("experiments/vcoco_v3_candidate_grid.json"),
    )
    parser.add_argument(
        "--candidate-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/candidates/candidate_grid_lock.json"),
    )
    parser.add_argument(
        "--human-pilot-audit",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/final/summary.json"),
    )
    parser.add_argument("--baseline", choices=METHOD_KEYS, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", choices=METHOD_KEYS, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/research_20260906/vcoco_development_evidence"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, source: Path) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Manifest lacks required columns {sorted(missing)}: {source}")


def validate_output_directory(output_dir: Path, input_paths: Sequence[Path]) -> Path:
    """Reject writes into a directory that contains a locked input."""

    resolved_output = output_dir.resolve()
    for input_path in input_paths:
        resolved_input = input_path.resolve()
        input_parent = resolved_input.parent
        if (
            resolved_output == resolved_input
            or resolved_output == input_parent
            or input_parent in resolved_output.parents
        ):
            raise RuntimeError(
                f"Evidence output directory overlaps a locked input location: {resolved_input}"
            )
    return resolved_output


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"Invalid SHA-256 for {name}")
    return value


def _validate_content_lock(
    paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
    *,
    scope: str,
) -> dict[str, str]:
    missing_hashes = set(paths) - set(expected_sha256)
    missing_paths = set(expected_sha256) - set(paths)
    if missing_hashes or missing_paths:
        raise RuntimeError(
            f"{scope} content lock keys differ: "
            f"missing_hashes={sorted(missing_hashes)}, missing_paths={sorted(missing_paths)}"
        )
    actual = {}
    for name, path in paths.items():
        expected = _require_sha256(expected_sha256[name], name=f"{scope}:{name}")
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"{scope} content lock drift: {name}")
        actual[name] = digest
    return actual


def _validate_content_unchanged(
    paths: Mapping[str, Path],
    before_sha256: Mapping[str, str],
    *,
    scope: str,
) -> None:
    """Reject inputs whose bytes changed after their initial content-lock check."""

    if set(paths) != set(before_sha256):
        raise RuntimeError(f"{scope} stability-check keys differ")
    for name, path in paths.items():
        if sha256_file(path) != before_sha256[name]:
            raise RuntimeError(f"{scope} changed while being read: {name}")


def _atomic_write_text(path: Path, value: str) -> None:
    """Replace one text artifact atomically while preserving Path.write_text bytes."""

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
    """Replace one deterministic CSV artifact atomically."""

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


def _validate_safe_basename(value: object, *, path: Path) -> str:
    if pd.isna(value):
        raise RuntimeError(f"Manifest contains a missing file_name: {path}")
    text = str(value)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or text != text.strip()
        or text in {".", ".."}
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.name != text
        or windows.name != text
    ):
        raise RuntimeError(f"Manifest file_name is not a safe basename: {path}")
    return text


def _validate_geometry(rows: pd.DataFrame, *, path: Path) -> None:
    xmin = rows["bbox_xmin"].to_numpy(dtype=float)
    ymin = rows["bbox_ymin"].to_numpy(dtype=float)
    xmax = rows["bbox_xmax"].to_numpy(dtype=float)
    ymax = rows["bbox_ymax"].to_numpy(dtype=float)
    image_width = rows["actual_width"].to_numpy(dtype=float)
    image_height = rows["actual_height"].to_numpy(dtype=float)
    box_width = xmax - xmin
    box_height = ymax - ymin
    if (image_width <= 0.0).any() or (image_height <= 0.0).any():
        raise RuntimeError(f"Manifest image dimensions must be positive: {path}")
    if (
        (xmin < 0.0).any()
        or (ymin < 0.0).any()
        or (xmax > image_width).any()
        or (ymax > image_height).any()
        or (box_width <= 0.0).any()
        or (box_height <= 0.0).any()
    ):
        raise RuntimeError(
            f"Manifest bounding boxes must be ordered, nondegenerate, and in bounds: {path}"
        )

    expected = {
        "bbox_area_fraction": box_width * box_height / (image_width * image_height),
        "bbox_aspect_ratio": box_width / box_height,
        "bbox_center_x_fraction": (xmin + xmax) / (2.0 * image_width),
        "bbox_center_y_fraction": (ymin + ymax) / (2.0 * image_height),
        "person_pixel_height": box_height,
    }
    domains = {
        "bbox_area_fraction": (0.0, 1.0),
        "bbox_aspect_ratio": (0.0, np.inf),
        "bbox_center_x_fraction": (0.0, 1.0),
        "bbox_center_y_fraction": (0.0, 1.0),
        "person_pixel_height": (0.0, np.inf),
    }
    for column, expected_values in expected.items():
        actual_values = rows[column].to_numpy(dtype=float)
        lower, upper = domains[column]
        if (actual_values <= lower).any() or (actual_values > upper).any():
            raise RuntimeError(f"Manifest geometry column {column} is outside its domain: {path}")
        if not np.allclose(
            actual_values,
            expected_values,
            rtol=GEOMETRY_RTOL,
            atol=GEOMETRY_ATOL,
        ):
            raise RuntimeError(f"Manifest geometry column {column} is inconsistent: {path}")


def _coerce_boolean(series: pd.Series, *, name: str) -> pd.Series:
    mapping = {"true": True, "false": False}
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        bad = sorted(series.loc[converted.isna()].astype(str).unique())
        raise RuntimeError(f"Column {name} contains invalid booleans: {bad}")
    return converted.astype(bool)


def _load_manifest(path: Path, expected_split: str) -> pd.DataFrame:
    rows = pd.read_csv(
        path,
        dtype={"person_id": str, "image_id": str, "annotation_id": str},
    )
    _require_columns(rows, MANIFEST_REQUIRED_COLUMNS, source=path)
    for column in ("person_id", "image_id", "annotation_id"):
        values = rows[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").fillna(True).any():
            raise RuntimeError(f"Manifest contains a missing or blank {column}: {path}")
        rows[column] = values.astype(str)
    if not rows["external_split"].astype(str).eq(expected_split).all():
        raise RuntimeError(f"Manifest is not exclusively split={expected_split!r}: {path}")
    expected_selection_role = (
        "adaptation" if expected_split == "train" else "selection_and_calibration"
    )
    if not rows["selection_role"].astype(str).eq(expected_selection_role).all():
        raise RuntimeError(
            f"Manifest has an invalid selection role for split={expected_split!r}: {path}"
        )
    if rows["person_id"].duplicated().any():
        raise RuntimeError(f"Manifest contains duplicate person IDs: {path}")
    if rows["annotation_id"].duplicated().any():
        raise RuntimeError(f"Manifest contains duplicate annotation IDs: {path}")
    hashes = rows["sha256"].astype("string")
    if hashes.isna().any() or not hashes.str.fullmatch(r"[0-9a-f]{64}").fillna(False).all():
        raise RuntimeError(f"Manifest contains an invalid image SHA-256: {path}")
    rows["sha256"] = hashes.astype(str)
    rows["file_name"] = rows["file_name"].map(
        lambda value: _validate_safe_basename(value, path=path)
    )
    for column in NUMERIC_MANIFEST_COLUMNS:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if not np.isfinite(rows[column].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Manifest column {column} contains non-finite values: {path}")
    _validate_geometry(rows, path=path)
    if rows.groupby("image_id", sort=False)["sha256"].nunique(dropna=False).gt(1).any():
        raise RuntimeError(f"A manifest image ID maps to multiple image content hashes: {path}")
    for column in ("file_name", "actual_width", "actual_height", "external_split"):
        if rows.groupby("image_id", sort=False)[column].nunique(dropna=False).gt(1).any():
            raise RuntimeError(
                f"Manifest image IDs map to inconsistent image-level {column} values: {path}"
            )
    rows["image_level_unambiguous"] = _coerce_boolean(
        rows["image_level_unambiguous"], name="image_level_unambiguous"
    )
    return rows


def load_locked_development(
    protocol_lock_path: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    *,
    expected_sha256: Mapping[str, str] = RETAINED_DEVELOPMENT_SHA256,
) -> tuple[pd.DataFrame, dict, dict[str, str]]:
    """Load the byte-locked V-COCO development manifests in canonical order."""

    protocol_lock_path = protocol_lock_path.resolve()
    train_manifest_path = train_manifest_path.resolve()
    val_manifest_path = val_manifest_path.resolve()
    development_paths = {
        "protocol_lock": protocol_lock_path,
        "vcoco_train_clean.csv": train_manifest_path,
        "vcoco_val_clean.csv": val_manifest_path,
    }
    development_hashes = _validate_content_lock(
        development_paths,
        expected_sha256,
        scope="Development",
    )
    protocol = _read_json(protocol_lock_path)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise RuntimeError("Development evidence requires the locked V-COCO v2 protocol")
    expected_hashes = protocol.get("artifact_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("Protocol lock lacks artifact hashes")
    manifest_paths = {
        "vcoco_train_clean.csv": train_manifest_path,
        "vcoco_val_clean.csv": val_manifest_path,
    }
    for name, path in manifest_paths.items():
        if expected_hashes.get(name) != sha256_file(path):
            raise RuntimeError(f"Locked development manifest drift: {name}")

    train = _load_manifest(train_manifest_path, "train")
    validation = _load_manifest(val_manifest_path, "val")
    if set(train["image_id"]).intersection(validation["image_id"]):
        raise RuntimeError("A source image crosses the locked train/validation boundary")
    if set(train["sha256"]).intersection(validation["sha256"]):
        raise RuntimeError("Source image content crosses the locked train/validation boundary")
    rows = pd.concat([train, validation], ignore_index=True)
    if rows["person_id"].duplicated().any():
        raise RuntimeError("Development manifests contain duplicate person IDs")
    if rows["annotation_id"].duplicated().any():
        raise RuntimeError("Development manifests contain duplicate annotation IDs")
    if rows.groupby("image_id", sort=False)["sha256"].nunique().gt(1).any():
        raise RuntimeError("A development image ID maps to multiple image content hashes")
    if rows.groupby("sha256", sort=False)["image_id"].nunique().gt(1).any():
        raise RuntimeError("A development image content hash maps to multiple image IDs")
    for column in ("file_name", "actual_width", "actual_height", "external_split", "sha256"):
        if rows.groupby("image_id", sort=False)[column].nunique(dropna=False).gt(1).any():
            raise RuntimeError(
                f"Development image IDs map to inconsistent image-level {column} values"
            )

    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = rows["label_3"].map(label_map)
    if labels.isna().any():
        unknown = sorted(rows.loc[labels.isna(), "label_3"].astype(str).unique())
        raise RuntimeError(f"Development manifest contains unknown labels: {unknown}")
    rows["label_index"] = labels.astype(int)
    expected_factors = {
        "sitting": ("seated", "stationary"),
        "standing": ("upright", "stationary"),
        "walking_running": ("upright", "locomoting"),
    }
    for label, (posture, motion) in expected_factors.items():
        selected = rows["label_3"].eq(label)
        if not (
            rows.loc[selected, "posture_label"].astype(str).eq(posture).all()
            and rows.loc[selected, "motion_label"].astype(str).eq(motion).all()
        ):
            raise RuntimeError(f"Source-tag factorization disagrees with label {label!r}")

    rows["v2_split"] = rows["external_split"].astype(str)
    rows["image_sha256"] = rows["sha256"].astype(str)
    rows["people_in_image"] = (
        rows.groupby("image_id", sort=False)["person_id"].transform("size").astype(int)
    )
    rows["boundary_contact_1px"] = (
        rows["bbox_xmin"].le(1.0)
        | rows["bbox_ymin"].le(1.0)
        | rows["bbox_xmax"].ge(rows["actual_width"] - 1.0)
        | rows["bbox_ymax"].ge(rows["actual_height"] - 1.0)
    )
    _validate_content_unchanged(
        development_paths,
        development_hashes,
        scope="Development input",
    )
    return rows, protocol, development_hashes


def _validate_probability_matrix(values: np.ndarray, *, method: str, rows: int) -> None:
    if values.shape != (rows, len(CLASS_NAMES)):
        raise RuntimeError(f"Prediction matrix has the wrong shape for {method}: {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Prediction matrix contains non-finite values: {method}")
    if (values < 0.0).any() or (values > 1.0 + PROBABILITY_TOLERANCE).any():
        raise RuntimeError(f"Prediction matrix contains values outside [0, 1]: {method}")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise RuntimeError(f"Prediction rows do not sum to one: {method}")


def load_and_align_nested_predictions(
    prediction_run_dir: Path,
    rows: pd.DataFrame,
    *,
    methods: Sequence[str] = METHOD_KEYS,
    expected_run_sha256: Mapping[str, str] = RETAINED_PREDICTION_SHA256,
    expected_source_sha256: Mapping[str, str] = RETAINED_PREDICTION_SOURCE_SHA256,
    source_paths: Mapping[str, Path] = DEFAULT_PREDICTION_SOURCE_PATHS,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict, dict[str, str]]:
    """Validate and align the byte-locked run that attests grouped-OOF predictions."""

    prediction_run_dir = prediction_run_dir.resolve()
    summary_path = prediction_run_dir / "summary.json"
    predictions_path = prediction_run_dir / "nested_oof_probabilities.npz"
    metrics_path = prediction_run_dir / "nested_source_tag_metrics.csv"
    run_paths = {
        summary_path.name: summary_path,
        predictions_path.name: predictions_path,
        metrics_path.name: metrics_path,
    }
    run_hashes = _validate_content_lock(
        run_paths,
        expected_run_sha256,
        scope="Retained prediction run",
    )
    resolved_source_paths = {name: path.resolve() for name, path in source_paths.items()}
    source_file_hashes = _validate_content_lock(
        resolved_source_paths,
        expected_source_sha256,
        scope="Retained prediction source",
    )
    summary = _read_json(summary_path)
    if summary.get("status") != PREDICTION_STATUS:
        raise RuntimeError("Prediction run is not a completed nested V-COCO development run")
    if summary.get("endpoint") != PREDICTION_ENDPOINT:
        raise RuntimeError("Prediction run has an unexpected selection endpoint")
    if summary.get("human_pilot_labels_used_for_selection") is not False:
        raise RuntimeError("Prediction run reports human-pilot label use for selection")
    for name, expected in EXPECTED_FOLD_COUNTS.items():
        value = summary.get(name)
        if type(value) is not int or value != expected:
            raise RuntimeError(f"Prediction run has an unexpected {name}")
    test_rows = summary.get("official_v2_test_rows_read")
    if type(test_rows) is not int or test_rows != 0:
        raise RuntimeError("Prediction run reports official-test rows read")
    if summary.get("official_v2_test_predictions_run") is not False:
        raise RuntimeError("Prediction run reports official-test prediction access")
    people = summary.get("people")
    if type(people) is not int or people != len(rows):
        raise RuntimeError("Prediction summary person count disagrees with the manifests")
    source_images = summary.get("source_images")
    if type(source_images) is not int or source_images != int(rows["image_id"].nunique()):
        raise RuntimeError("Prediction summary image count disagrees with the manifests")

    declared_sources = summary.get("source_sha256")
    if not isinstance(declared_sources, dict):
        raise RuntimeError("Prediction summary lacks source hashes")
    for name, digest in source_file_hashes.items():
        if declared_sources.get(name) != digest:
            raise RuntimeError(f"Prediction summary source hash differs: {name}")

    artifact_hashes = summary.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError("Prediction summary lacks artifact hashes")
    for path in (predictions_path, metrics_path):
        if artifact_hashes.get(path.name) != run_hashes[path.name]:
            raise RuntimeError(f"Nested prediction artifact drift: {path.name}")

    with np.load(predictions_path, allow_pickle=False) as payload:
        required = {"person_ids", "image_ids", "labels", "class_names", *methods}
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"Nested prediction artifact lacks keys: {sorted(missing)}")
        unexpected = set(payload.files) - required
        if unexpected:
            raise RuntimeError(
                f"Nested prediction artifact has unexpected keys: {sorted(unexpected)}"
            )
        class_names = tuple(str(value) for value in payload["class_names"])
        if class_names != CLASS_NAMES:
            raise RuntimeError(f"Unexpected prediction class order: {class_names}")
        person_ids = np.asarray(payload["person_ids"]).astype(str)
        image_ids = np.asarray(payload["image_ids"]).astype(str)
        labels = np.asarray(payload["labels"])
        if person_ids.ndim != 1 or image_ids.ndim != 1 or labels.ndim != 1:
            raise RuntimeError("Prediction identity arrays must be one-dimensional")
        if not (len(person_ids) == len(image_ids) == len(labels) == len(rows)):
            raise RuntimeError("Prediction identity arrays do not match manifest row count")
        if len(np.unique(person_ids)) != len(person_ids):
            raise RuntimeError("Prediction artifact contains duplicate person IDs")
        manifest_ids = rows["person_id"].astype(str).to_numpy()
        missing_ids = set(manifest_ids) - set(person_ids)
        extra_ids = set(person_ids) - set(manifest_ids)
        if missing_ids or extra_ids:
            raise RuntimeError(
                "Prediction and manifest person ID sets differ: "
                f"missing={len(missing_ids)}, extra={len(extra_ids)}"
            )
        position = {value: index for index, value in enumerate(person_ids)}
        order = np.asarray([position[value] for value in manifest_ids], dtype=int)
        if not np.array_equal(image_ids[order], rows["image_id"].astype(str).to_numpy()):
            raise RuntimeError("Prediction person-to-image mapping disagrees with the manifests")
        if not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError("Prediction labels are not integer class indices")
        aligned_labels = labels[order].astype(int, copy=False)
        if not np.array_equal(aligned_labels, rows["label_index"].to_numpy(dtype=int)):
            raise RuntimeError("Prediction labels disagree with the locked source tags")
        probabilities = {}
        for method in methods:
            values = np.asarray(payload[method], dtype=np.float64)
            _validate_probability_matrix(values, method=method, rows=len(rows))
            probabilities[method] = values[order]

    metrics_frame = pd.read_csv(metrics_path)
    _require_columns(metrics_frame, ("family",), source=metrics_path)
    if metrics_frame["family"].duplicated().any():
        raise RuntimeError("Nested metrics contain duplicate family rows")
    metrics_frame = metrics_frame.set_index("family", drop=False)
    replayed_metrics = {}
    labels_array = rows["label_index"].to_numpy(dtype=int)
    for method, values in probabilities.items():
        if method not in metrics_frame.index:
            raise RuntimeError(f"Nested metrics omit prediction family: {method}")
        replayed = classification_metrics(labels_array, values)
        source_row = metrics_frame.loc[method]
        for metric, value in replayed.items():
            if metric not in metrics_frame.columns:
                raise RuntimeError(f"Nested metrics omit column: {metric}")
            source_value = float(source_row[metric])
            if not np.isclose(value, source_value, rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"Replayed metric differs for {method}:{metric}")
        replayed_metrics[method] = replayed

    best_family = str(summary.get("best_family"))
    if best_family not in replayed_metrics:
        raise RuntimeError("Prediction summary names an unavailable best family")
    best_replayed_macro_f1 = max(values["macro_f1"] for values in replayed_metrics.values())
    if not np.isclose(
        replayed_metrics[best_family]["macro_f1"],
        best_replayed_macro_f1,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Prediction summary best family disagrees with replayed metrics")
    if not np.isclose(
        float(summary.get("best_macro_f1", np.nan)),
        replayed_metrics[best_family]["macro_f1"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Prediction summary best macro-F1 does not replay")

    source_hashes = {
        "prediction_summary": run_hashes[summary_path.name],
        "nested_oof_probabilities": run_hashes[predictions_path.name],
        "nested_source_tag_metrics": run_hashes[metrics_path.name],
        **source_file_hashes,
    }
    _validate_content_unchanged(
        run_paths,
        run_hashes,
        scope="Retained prediction run input",
    )
    _validate_content_unchanged(
        resolved_source_paths,
        source_file_hashes,
        scope="Retained prediction source input",
    )
    return probabilities, replayed_metrics, summary, source_hashes


def _error_directions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    output = np.full(len(labels), "correct", dtype=object)
    wrong = labels != predictions
    output[wrong] = [
        f"{ERROR_CLASS_NAMES[truth]}_to_{ERROR_CLASS_NAMES[predicted]}"
        for truth, predicted in zip(labels[wrong], predictions[wrong], strict=True)
    ]
    return output


def build_evidence_frame(
    rows: pd.DataFrame,
    probabilities: Mapping[str, np.ndarray],
    *,
    baseline: str,
    candidate: str,
) -> pd.DataFrame:
    """Create the deterministic wide evidence table from aligned probabilities."""

    if baseline == candidate:
        raise ValueError("Baseline and candidate must be different methods")
    if baseline not in probabilities or candidate not in probabilities:
        raise ValueError("Baseline and candidate must name supplied probability matrices")
    output = rows.loc[:, OUTPUT_MANIFEST_COLUMNS].copy()
    labels = rows["label_index"].to_numpy(dtype=int)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels]
    correct_by_method = {}
    nll_by_method = {}
    for method, matrix in probabilities.items():
        values = np.asarray(matrix, dtype=np.float64)
        _validate_probability_matrix(values, method=method, rows=len(rows))
        predictions = values.argmax(axis=1)
        correct = predictions == labels
        true_class_probability = values[np.arange(len(values)), labels]
        nll = -np.log(np.clip(true_class_probability, NLL_FLOOR, 1.0))
        for index, class_name in enumerate(CLASS_NAMES):
            output[f"{method}__p_{class_name}"] = values[:, index]
        output[f"{method}__predicted_label"] = [CLASS_NAMES[index] for index in predictions]
        output[f"{method}__confidence"] = values.max(axis=1)
        output[f"{method}__correct"] = correct
        output[f"{method}__error_direction"] = _error_directions(labels, predictions)
        output[f"{method}__nll"] = nll
        output[f"{method}__brier"] = np.square(values - one_hot).sum(axis=1)
        correct_by_method[method] = correct
        nll_by_method[method] = nll

    baseline_correct = correct_by_method[baseline]
    candidate_correct = correct_by_method[candidate]
    output["candidate_vs_baseline_transition"] = np.select(
        [
            baseline_correct & candidate_correct,
            ~baseline_correct & candidate_correct,
            baseline_correct & ~candidate_correct,
        ],
        ["both_correct", "rescued", "harmed"],
        default="both_wrong",
    )
    output["candidate_minus_baseline_nll"] = nll_by_method[candidate] - nll_by_method[baseline]
    return output


def write_development_evidence(
    output_dir: Path,
    evidence: pd.DataFrame,
    *,
    probabilities: Mapping[str, np.ndarray],
    replayed_metrics: Mapping[str, Mapping[str, float]],
    prediction_summary: Mapping,
    baseline: str,
    candidate: str,
    source_hashes: Mapping[str, str],
    source_paths: Mapping[str, Path],
) -> dict:
    """Write deterministic row evidence and its provenance summary."""

    output_dir = validate_output_directory(output_dir, tuple(source_paths.values()))
    write_input_hashes = _validate_content_lock(
        source_paths,
        source_hashes,
        scope="Evidence write input",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "development_joined.csv"
    _atomic_write_csv(evidence_path, evidence)
    transitions = {
        str(name): int(count)
        for name, count in evidence["candidate_vs_baseline_transition"]
        .value_counts()
        .sort_index()
        .items()
    }
    prediction_run_provenance = {
        "status": str(prediction_summary["status"]),
        "endpoint": str(prediction_summary["endpoint"]),
        "human_pilot_labels_used_for_selection": bool(
            prediction_summary["human_pilot_labels_used_for_selection"]
        ),
        "official_v2_test_rows_read": int(prediction_summary["official_v2_test_rows_read"]),
        "official_v2_test_predictions_run": bool(
            prediction_summary["official_v2_test_predictions_run"]
        ),
        "outer_folds": int(prediction_summary["outer_folds"]),
        "inner_folds": int(prediction_summary["inner_folds"]),
        "stack_folds": int(prediction_summary["stack_folds"]),
        "source_sha256": dict(prediction_summary["source_sha256"]),
        "used_artifact_sha256": {
            name: str(prediction_summary["artifact_sha256"][name])
            for name in ("nested_oof_probabilities.npz", "nested_source_tag_metrics.csv")
        },
    }
    summary = {
        "status": OUTPUT_STATUS,
        "scope": OUTPUT_SCOPE,
        "row_unit": "person_instance_with_image_id",
        "row_order": "locked_train_manifest_then_locked_val_manifest",
        "people": int(len(evidence)),
        "source_images": int(evidence["image_id"].nunique()),
        "v2_split_counts": {
            str(name): int(count)
            for name, count in evidence["v2_split"].value_counts().sort_index().items()
        },
        "class_names": list(CLASS_NAMES),
        "methods": {
            method: {"artifact": "nested_oof_probabilities.npz", "key": method}
            for method in probabilities
        },
        "baseline": baseline,
        "candidate": candidate,
        "prediction_run_best_family": str(prediction_summary["best_family"]),
        "prediction_run_provenance": prediction_run_provenance,
        "replayed_metrics": {
            method: {metric: float(value) for metric, value in metrics.items()}
            for method, metrics in replayed_metrics.items()
        },
        "candidate_vs_baseline_transition_counts": transitions,
        "mean_candidate_minus_baseline_nll": float(evidence["candidate_minus_baseline_nll"].mean()),
        "nll_probability_floor": NLL_FLOOR,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "calibration_artifacts_read": 0,
        "source_sha256": dict(source_hashes),
        "artifact_sha256": {"development_joined.csv": sha256_file(evidence_path)},
    }
    summary_path = output_dir / "summary.json"
    _validate_content_unchanged(
        source_paths,
        write_input_hashes,
        scope="Evidence write input",
    )
    _atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_lock.resolve()
    train_path = args.train_manifest.resolve()
    val_path = args.val_manifest.resolve()
    prediction_run_dir = args.prediction_run_dir.resolve()
    output_dir = validate_output_directory(
        args.output_dir,
        (
            protocol_path,
            train_path,
            val_path,
            prediction_run_dir / "summary.json",
            prediction_run_dir / "nested_oof_probabilities.npz",
            prediction_run_dir / "nested_source_tag_metrics.csv",
            args.candidate_grid,
            args.candidate_lock,
            args.human_pilot_audit,
        ),
    )
    rows, _protocol, development_hashes = load_locked_development(
        protocol_path,
        train_path,
        val_path,
    )
    probabilities, replayed_metrics, prediction_summary, prediction_hashes = (
        load_and_align_nested_predictions(
            prediction_run_dir,
            rows,
            source_paths={
                "candidate_grid": args.candidate_grid,
                "candidate_lock": args.candidate_lock,
                "human_pilot_audit": args.human_pilot_audit,
            },
        )
    )
    evidence = build_evidence_frame(
        rows,
        probabilities,
        baseline=args.baseline,
        candidate=args.candidate,
    )
    source_hashes = {
        "protocol_lock": development_hashes["protocol_lock"],
        "train_manifest": development_hashes["vcoco_train_clean.csv"],
        "val_manifest": development_hashes["vcoco_val_clean.csv"],
        **prediction_hashes,
    }
    source_paths = {
        "protocol_lock": protocol_path,
        "train_manifest": train_path,
        "val_manifest": val_path,
        "prediction_summary": prediction_run_dir / "summary.json",
        "nested_oof_probabilities": prediction_run_dir / "nested_oof_probabilities.npz",
        "nested_source_tag_metrics": prediction_run_dir / "nested_source_tag_metrics.csv",
        "candidate_grid": args.candidate_grid.resolve(),
        "candidate_lock": args.candidate_lock.resolve(),
        "human_pilot_audit": args.human_pilot_audit.resolve(),
    }
    summary = write_development_evidence(
        output_dir,
        evidence,
        probabilities=probabilities,
        replayed_metrics=replayed_metrics,
        prediction_summary=prediction_summary,
        baseline=args.baseline,
        candidate=args.candidate,
        source_hashes=source_hashes,
        source_paths=source_paths,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
