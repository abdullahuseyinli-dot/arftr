"""Build a role-safe input inventory for the frozen Okutama CPTR replay.

The historical mixed development manifest and the packed store's broad metadata export
are intentionally not inputs.  Eligible packed-array positions are reconstructed from
the already separated prediction IDs, the audit's per-recording counts, and the exact
sort used by the original feature cacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from hac.cptr_features import sample_indices_with_centre

STATUS = "OKUTAMA_CPTR_REPLAY_INPUT_INVENTORY_COMPLETE"
INDEX_STATUS = "OKUTAMA_CPTR_ROLE_SAFE_INDEX_COMPLETE"
MATERIALIZATION_LOCK_STATUS = "OKUTAMA_CPTR_REPLAY_MATERIALIZATION_LOCKED_BEFORE_FEATURE_ACCESS"
BUNDLE_STATUS = "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE"
EXPECTED_ROWS = 6360
EXPECTED_SCENARIOS = 14
EXPECTED_OOF_ROWS = 4977
EXPECTED_VALIDATION_ROWS = 1383
EXPECTED_FEATURE_ROWS = 8339
EXPECTED_ARCHIVE_BYTES = 5_770_432_522
EXPECTED_ARCHIVE_SHA256 = "c021ce8a12c84e083f359023ffd41c145561aaedb48b118e7c5416d5ddcecb73"
EXPECTED_FOLDS = tuple(f"fold-{value}" for value in range(5))
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_FIXED_EPOCHS = {42: 2, 43: 0, 44: 1, 45: 9, 46: 1}
EXPECTED_PAIRED_SHA256 = "4be1537f19fba6f4524618485da65044d1645228ae56fbbc1a11e4a64fbfdc46"
EXPECTED_AUDIT_SHA256 = "212942e26b2a66df46435b770458adbd6da7632a9c3ba7b8dce51db25cf6af0f"
EXPECTED_BASE_STORE_SHA256 = "e58fd947683b02183571296dd6a4599f8005f470763198674a9a91b93e2d8cd3"
EXPECTED_PART_STORE_SHA256 = "f6d7c3650575470cc10fb74c5e0b771dd69abbe81183fd091a4b739dd13399f2"
FORBIDDEN_NAMES = {
    "development_manifest.csv",
    "development_metadata.csv",
    "calibration_manifest.csv",
    "confirmation_manifest.csv",
    "test_manifest.csv",
}
SAMPLE_PATTERN = re.compile(
    r"^train__(?P<recording>[12]\.[12]\.(?:[1-9]|1[01]))__track-"
    r"(?P<track>\d+)__frame-(?P<frame>\d{6})$"
)
BASE_ACTIONS = {"Sitting", "Standing", "Walking", "Running"}
TARGET_LABEL = {
    "Sitting": "sitting",
    "Standing": "standing",
    "Walking": "walking_running",
    "Running": "walking_running",
}
WINDOW_OFFSETS = tuple(np.rint(np.arange(-8, 9) * 30.0 / 16.0).astype(int))
DISTINCT_SHORT_INDICES = np.asarray([4, 5, 6, 7, 8, 10, 11, 12], dtype=np.int64)
DISTINCT_SHORT_CENTRE = 4


@dataclass(frozen=True)
class Paths:
    paired_rows: Path
    paired_summary: Path
    audit_summary: Path
    base_store: Path
    part_store: Path
    cptr_crossfit_root: Path
    baseline_crossfit_root: Path
    crossfit_plan: Path
    candidate_grid: Path
    cptr_protocol: Path
    temporal_grid: Path

    @classmethod
    def defaults(cls, root: Path) -> Paths:
        return cls(
            paired_rows=root / ".runs/cptr/baseline_preservation_diagnostic/paired_rows.csv",
            paired_summary=root / ".runs/cptr/baseline_preservation_diagnostic/summary.json",
            audit_summary=root / ".runs/vcoco_v3/okutama/development_audit/summary.json",
            base_store=root / ".runs/vcoco_v3/okutama/features/dinov2_base/store.json",
            part_store=root / ".runs/cptr/part_features/store.json",
            cptr_crossfit_root=root / ".runs/cptr/crossfit/centre_short_parts",
            baseline_crossfit_root=root / ".runs/vcoco_v3/temporal/crossfit",
            crossfit_plan=root / "experiments/okutama_cptr_crossfit_plan.json",
            candidate_grid=root / "experiments/okutama_cptr_adaptive_grid.json",
            cptr_protocol=root / "experiments/okutama_cptr_protocol.json",
            temporal_grid=root / "experiments/okutama_temporal_grid.json",
        )


@dataclass(frozen=True)
class Annotation:
    track_id: int
    bbox: tuple[int, int, int, int]
    frame: int
    lost: bool
    occluded: bool
    generated: bool
    actions: tuple[str, ...]

    @property
    def base_actions(self) -> tuple[str, ...]:
        return tuple(action for action in self.actions if action in BASE_ACTIONS)


def parse_args() -> argparse.Namespace:
    defaults = Paths.defaults(Path("."))
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in defaults.__dict__.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=value)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0"),
    )
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument(
        "--materialization-lock",
        type=Path,
        default=Path(".runs/research_20260907/protocol_locks/okutama_materialization_lock.json"),
    )
    parser.add_argument("--window-masks", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--eligible-index", type=Path)
    parser.add_argument("--bundle-output", type=Path)
    parser.add_argument("--bundle-summary", type=Path)
    parser.add_argument("--extract-window-masks", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--window-masks-output", type=Path)
    parser.add_argument("--window-mask-summary", type=Path)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while it was hashed: {path}")
    return digest.hexdigest()


def file_evidence(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(relative_to.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def capture(path: Path) -> bytes:
    resolved = path.resolve()
    if resolved.name.lower() in FORBIDDEN_NAMES:
        raise RuntimeError(f"Forbidden mixed/protected input: {resolved.name}")
    return resolved.read_bytes()


def load_json_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return value


def load_npz_bytes(raw: bytes, *, name: str) -> dict[str, np.ndarray]:
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise RuntimeError(f"{name} contains duplicate members")
        return {key: archive[key] for key in archive.files}


def parse_sample_id(value: str) -> tuple[str, str, int]:
    match = SAMPLE_PATTERN.fullmatch(str(value))
    if match is None:
        raise RuntimeError(f"Unexpected Okutama sample ID: {value!r}")
    return match["recording"], match["track"], int(match["frame"])


def parse_annotation(line: str) -> Annotation:
    fields = shlex.split(line)
    if len(fields) < 10 or fields[9] != "Person":
        raise RuntimeError(f"Malformed Okutama annotation row: {line[:80]!r}")
    return Annotation(
        track_id=int(fields[0]),
        bbox=tuple(map(int, fields[1:5])),  # type: ignore[arg-type]
        frame=int(fields[5]),
        lost=bool(int(fields[6])),
        occluded=bool(int(fields[7])),
        generated=bool(int(fields[8])),
        actions=tuple(fields[10:]),
    )


def read_annotation_member(archive: zipfile.ZipFile, member: str) -> list[Annotation]:
    if PurePosixPath(member).suffix != ".txt":
        raise RuntimeError("Only Okutama annotation text members may be read")
    raw = archive.read(member)
    return [parse_annotation(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    xmin = max(left[0], right[0])
    ymin = max(left[1], right[1])
    xmax = min(left[2], right[2])
    ymax = min(left[3], right[3])
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def stable_annotations_for_recording(
    archive: zipfile.ZipFile, recording_id: str
) -> dict[tuple[int, int], Annotation]:
    multi_member = f"Labels/MultiActionLabels/3840x2160/{recording_id}.txt"
    tracking_member = f"Labels/SingleActionTrackingLabels/3840x2160/{recording_id}.txt"
    multi = read_annotation_member(archive, multi_member)
    tracking = read_annotation_member(archive, tracking_member)
    tracking_by_key: dict[tuple[Any, ...], Annotation] = {}
    tracking_by_frame: dict[int, list[Annotation]] = {}
    for row in tracking:
        key = (row.frame, row.bbox, row.occluded, row.generated)
        if key in tracking_by_key:
            raise RuntimeError(f"Duplicate tracking join key in allowed recording {recording_id}")
        tracking_by_key[key] = row
        tracking_by_frame.setdefault(row.frame, []).append(row)
    by_track_frame: dict[tuple[int, int], Annotation] = {}
    for source in multi:
        if source.lost:
            continue
        join_key = (source.frame, source.bbox, source.occluded, source.generated)
        identity = tracking_by_key.get(join_key)
        if identity is None:
            nearest = max(
                tracking_by_frame.get(source.frame, ()),
                key=lambda row: box_iou(source.bbox, row.bbox),
                default=None,
            )
            overlap = box_iou(source.bbox, nearest.bbox) if nearest is not None else 0.0
            if source.frame % 180 == 0 and overlap >= 0.2:
                continue
            # Historical selected centres cannot depend on an unmatched non-boundary row.
            continue
        key = (identity.track_id, source.frame)
        if key in by_track_frame:
            raise RuntimeError(f"Duplicate stable track/frame in allowed recording {recording_id}")
        by_track_frame[key] = Annotation(
            track_id=identity.track_id,
            bbox=source.bbox,
            frame=source.frame,
            lost=source.lost,
            occluded=source.occluded,
            generated=source.generated,
            actions=source.actions,
        )
    return by_track_frame


def extract_role_safe_window_masks(
    *,
    archive_path: Path,
    inventory_path: Path,
    eligible_index_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    inventory = load_json_bytes(capture(inventory_path), name="input inventory")
    if inventory.get("status") != STATUS:
        raise RuntimeError("Window-mask extraction requires a complete R0 inventory")
    if inventory.get("artifacts", {}).get("eligible_index.csv", {}).get("sha256") != sha256_file(
        eligible_index_path
    ):
        raise RuntimeError("Eligible index differs from the R0 inventory")
    index = pd.read_csv(eligible_index_path, dtype=str, keep_default_na=False)
    if len(index) != EXPECTED_ROWS or index["sample_id"].duplicated().any():
        raise RuntimeError("Eligible index cardinality or identity changed")
    archive_path = archive_path.resolve()
    if (
        archive_path.name != "TrainSetFrames.zip"
        or archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES
    ):
        raise RuntimeError("Okutama development archive name or byte count changed")
    archive_hash = sha256_file(archive_path)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Okutama development archive hash changed")

    grouped = {
        recording: rows.copy()
        for recording, rows in index.groupby("provider_recording_id", observed=True)
    }
    masks: dict[str, np.ndarray] = {}
    annotation_members_read: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        # Do not call testzip() or enumerate/read any disallowed annotation member.
        for recording in sorted(grouped):
            expected_members = (
                f"Labels/MultiActionLabels/3840x2160/{recording}.txt",
                f"Labels/SingleActionTrackingLabels/3840x2160/{recording}.txt",
            )
            for member in expected_members:
                archive.getinfo(member)
            stable = stable_annotations_for_recording(archive, recording)
            annotation_members_read.extend(expected_members)
            for row in grouped[recording].itertuples(index=False):
                _recording, track, frame = parse_sample_id(str(row.sample_id))
                window = [stable.get((int(track), frame + offset)) for offset in WINDOW_OFFSETS]
                if any(value is None or value.lost for value in window):
                    raise RuntimeError(
                        f"Allowed row has an incomplete historical window: {row.sample_id}"
                    )
                typed_window = [value for value in window if value is not None]
                center = stable.get((int(track), frame))
                if center is None or len(center.base_actions) != 1:
                    raise RuntimeError(f"Allowed center annotation changed: {row.sample_id}")
                base_action = center.base_actions[0]
                transition = (
                    len({action for value in typed_window for action in value.base_actions}) > 1
                )
                occluded = np.asarray([value.occluded for value in typed_window], dtype=bool)
                if TARGET_LABEL[base_action] != str(row.label):
                    raise RuntimeError(f"Allowed row label differs from source: {row.sample_id}")
                if transition != (str(row.transition_window).lower() == "true"):
                    raise RuntimeError(
                        f"Allowed row transition differs from source: {row.sample_id}"
                    )
                if bool(occluded.any()) != (str(row.window_any_occluded).lower() == "true"):
                    raise RuntimeError(
                        f"Allowed row occlusion differs from source: {row.sample_id}"
                    )
                masks[str(row.sample_id)] = occluded
    ids = np.asarray(index["sample_id"].astype(str), dtype=str)
    if set(masks) != set(ids) or len(masks) != EXPECTED_ROWS:
        raise RuntimeError("Exact-mask extraction did not cover every permitted row once")
    values = np.stack([masks[value] for value in ids])
    output_path = output_path.resolve()
    summary_path = summary_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("Role-safe mask outputs already exist")
    temporary = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez_compressed(temporary, sample_ids=ids, window_occluded=values)
    temporary.replace(output_path)
    summary = {
        "status": "OKUTAMA_CPTR_ROLE_SAFE_EXACT_WINDOW_MASKS_COMPLETE",
        "rows": EXPECTED_ROWS,
        "scenarios": EXPECTED_SCENARIOS,
        "allowed_recordings": sorted(grouped),
        "annotation_members_read": annotation_members_read,
        "annotation_members_read_count": len(annotation_members_read),
        "archive_wide_crc_scan_performed": False,
        "source_sha256": {
            "archive": archive_hash,
            "input_inventory": sha256_file(inventory_path),
            "eligible_index": sha256_file(eligible_index_path),
            "extractor": sha256_file(Path(__file__)),
        },
        "artifact_sha256": {output_path.name: sha256_file(output_path)},
        "access_accounting": {
            "allowed_annotation_members_read": len(annotation_members_read),
            "disallowed_annotation_members_read": 0,
            "image_members_read": 0,
            "mixed_development_manifest_rows_read": 0,
            "broad_development_metadata_rows_read": 0,
            "confirmation_rows_read": 0,
            "test_rows_read": 0,
            "feature_array_values_read": 0,
            "checkpoints_loaded": 0,
        },
    }
    write_text_atomic(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def derive_role_safe_index(
    paired: pd.DataFrame,
    audit: dict[str, Any],
) -> pd.DataFrame:
    required = {
        "scope",
        "development_role",
        "sample_id",
        "recording_id",
        "track_id",
        "fold",
        "label_index",
        "label",
        "transition_window",
        "window_any_occluded",
    }
    missing = required - set(paired.columns)
    if missing:
        raise RuntimeError(f"Permitted paired rows are missing columns: {sorted(missing)}")
    if len(paired) != EXPECTED_ROWS or paired["sample_id"].duplicated().any():
        raise RuntimeError("Permitted paired-row cardinality or identity changed")
    scope_counts = paired.groupby("scope", observed=True).size().to_dict()
    if scope_counts != {
        "fixed_development_validation": EXPECTED_VALIDATION_ROWS,
        "grouped_crossfit_oof": EXPECTED_OOF_ROWS,
    }:
        raise RuntimeError(f"Unexpected role-safe scope counts: {scope_counts}")

    parsed = [parse_sample_id(value) for value in paired["sample_id"].astype(str)]
    result = paired[list(sorted(required))].copy()
    result["provider_recording_id"] = [item[0] for item in parsed]
    result["provider_track_id"] = [item[1] for item in parsed]
    result["center_frame"] = [item[2] for item in parsed]
    expected_track = (
        result["provider_recording_id"].astype(str) + "::" + result["provider_track_id"].astype(str)
    )
    if not np.array_equal(expected_track.to_numpy(), result["track_id"].astype(str).to_numpy()):
        raise RuntimeError("Sample IDs and role-safe composite track IDs disagree")

    evidence = audit.get("recording_evidence")
    if audit.get("status") != "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED" or not isinstance(
        evidence, list
    ):
        raise RuntimeError("The aggregate Okutama audit is not complete")
    if int(audit.get("selected_centres", -1)) != EXPECTED_FEATURE_ROWS:
        raise RuntimeError("The historical packed feature cardinality changed")
    recording_rows: dict[str, tuple[str, int]] = {}
    for item in evidence:
        recording = str(item["recording_id"])
        if recording in recording_rows:
            raise RuntimeError(f"Duplicate audit recording: {recording}")
        recording_rows[recording] = (str(item["scenario_id"]), int(item["selected_centres"]))
    if sum(count for _, count in recording_rows.values()) != EXPECTED_FEATURE_ROWS:
        raise RuntimeError("Audit per-recording counts do not sum to the packed store")

    allowed_scenarios = set(result["recording_id"].astype(str))
    if len(allowed_scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError("The role-safe row set does not contain 14 scenarios")
    allowed_recordings = {
        recording
        for recording, (scenario, _) in recording_rows.items()
        if scenario in allowed_scenarios
    }
    if set(result["provider_recording_id"].astype(str)) != allowed_recordings:
        raise RuntimeError(
            "Permitted sample IDs do not contain every view of each allowed scenario"
        )
    for recording in sorted(allowed_recordings):
        observed = int(result["provider_recording_id"].eq(recording).sum())
        expected = recording_rows[recording][1]
        if observed != expected:
            raise RuntimeError(
                f"Allowed recording {recording} is incomplete: observed={observed}, expected={expected}"
            )
    if sum(recording_rows[value][1] for value in allowed_recordings) != EXPECTED_ROWS:
        raise RuntimeError("Allowed audit counts do not sum to 6,360")

    offsets: dict[str, int] = {}
    offset = 0
    for recording, (_scenario, count) in sorted(
        recording_rows.items(), key=lambda item: (item[1][0], item[0])
    ):
        offsets[recording] = offset
        offset += count

    result["feature_index"] = -1
    for recording in sorted(allowed_recordings):
        selector = result["provider_recording_id"].eq(recording)
        ordered = result.loc[selector].sort_values(
            ["provider_track_id", "center_frame"], kind="stable"
        )
        positions = offsets[recording] + np.arange(len(ordered), dtype=np.int64)
        result.loc[ordered.index, "feature_index"] = positions
    indices = result["feature_index"].to_numpy(dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= EXPECTED_FEATURE_ROWS):
        raise RuntimeError("A reconstructed feature index is out of bounds")
    if len(np.unique(indices)) != EXPECTED_ROWS:
        raise RuntimeError("Reconstructed feature indices are not one-to-one")

    result = result.sort_values("feature_index", kind="stable", ignore_index=True)
    return result[
        [
            "scope",
            "development_role",
            "sample_id",
            "recording_id",
            "provider_recording_id",
            "track_id",
            "provider_track_id",
            "center_frame",
            "feature_index",
            "fold",
            "label_index",
            "label",
            "transition_window",
            "window_any_occluded",
        ]
    ]


def validate_store(
    raw: bytes,
    *,
    name: str,
    expected_status: str,
    expected_sha256: str,
) -> dict[str, Any]:
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"{name} declaration hash changed")
    store = load_json_bytes(raw, name=name)
    if (
        store.get("status") != expected_status
        or int(store.get("samples", -1)) != EXPECTED_FEATURE_ROWS
    ):
        raise RuntimeError(f"{name} declaration is incomplete or has the wrong row count")
    arrays = store.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise RuntimeError(f"{name} declaration has no arrays")
    return store


def verify_declared_array_receipts(
    declaration_path: Path,
    declaration: dict[str, Any],
    *,
    store_name: str,
    expected_names: set[str],
    materialization_lock: dict[str, Any],
    repository_root: Path,
) -> dict[str, dict[str, Any]]:
    """Opaque-hash source arrays against declarations and the pre-access lock."""

    arrays = declaration.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != expected_names:
        raise RuntimeError(f"Unexpected {store_name} feature-array declaration")
    locked = materialization_lock.get("materialization_inputs", {}).get(
        "feature_arrays", {}
    )
    receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        relative = Path(str(arrays[name].get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe declared array path for {name}")
        path = (declaration_path.resolve().parent / relative).resolve()
        try:
            repository_relative = path.relative_to(repository_root.resolve()).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"Declared feature array is outside the repository: {path}"
            ) from error
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != arrays[name].get("sha256"):
            raise RuntimeError(f"Declared {store_name}/{name} feature-array digest changed")
        key = f"{store_name}_{name}"
        locked_item = locked.get(key, {})
        if (
            locked_item.get("path") != repository_relative
            or locked_item.get("sha256") != digest
            or int(locked_item.get("size_bytes", -1)) != size
            or locked_item.get("verification")
            != "opaque_byte_stream_no_array_decoding_or_indexing"
        ):
            raise RuntimeError(f"Materialization lock does not bind {store_name}/{name}")
        receipts[key] = {
            "path": repository_relative,
            "sha256": digest,
            "size_bytes": size,
            "verification": "opaque_byte_stream_no_array_decoding_or_indexing",
        }
    return receipts


def validate_prediction_arrays(
    arrays: dict[str, np.ndarray],
    *,
    name: str,
    expected: pd.DataFrame,
    baseline: bool,
) -> None:
    keys = {
        "sample_ids",
        "recording_ids",
        "track_ids",
        "labels",
        "probabilities",
    }
    if not baseline:
        keys |= {
            "baseline_probabilities",
            "transition_targets",
            "occlusion_targets",
        }
    if set(arrays) != keys:
        raise RuntimeError(f"{name} NPZ members changed: {sorted(arrays)}")
    rows = len(expected)
    if arrays["probabilities"].shape != (rows, 3):
        raise RuntimeError(f"{name} probability shape changed")
    ids = arrays["sample_ids"].astype(str)
    if len(np.unique(ids)) != rows or set(ids) != set(expected["sample_id"].astype(str)):
        raise RuntimeError(f"{name} sample IDs differ from the reconstructed fold")
    position = expected.set_index("sample_id", drop=False)
    aligned = position.loc[ids]
    if not np.array_equal(arrays["recording_ids"].astype(str), aligned["recording_id"].astype(str)):
        raise RuntimeError(f"{name} scenario IDs disagree")
    if not np.array_equal(arrays["track_ids"].astype(str), aligned["track_id"].astype(str)):
        raise RuntimeError(f"{name} track IDs disagree")
    if not np.array_equal(arrays["labels"].astype(int), aligned["label_index"].astype(int)):
        raise RuntimeError(f"{name} labels disagree")
    probabilities = arrays["probabilities"].astype(np.float64)
    if not np.isfinite(probabilities).all() or np.max(np.abs(probabilities.sum(1) - 1.0)) > 1e-5:
        raise RuntimeError(f"{name} contains invalid probabilities")


def checkpoint_inventory(
    paths: Paths,
    eligible_index: pd.DataFrame,
    *,
    repository_root: Path,
) -> list[dict[str, Any]]:
    oof = eligible_index.loc[eligible_index["scope"].eq("grouped_crossfit_oof")]
    entries: list[dict[str, Any]] = []
    for fold in range(5):
        fold_name = f"fold-{fold}"
        expected = oof.loc[oof["fold"].eq(fold_name)]
        if expected.empty:
            raise RuntimeError(f"The role-safe index has no rows for {fold_name}")
        reference_ids: np.ndarray | None = None
        for seed in EXPECTED_SEEDS:
            candidate_dir = paths.cptr_crossfit_root / fold_name / f"seed-{seed}"
            baseline_dir = paths.baseline_crossfit_root
            files = {
                "candidate_checkpoint": candidate_dir / "checkpoint.pt",
                "candidate_predictions": candidate_dir / "held_predictions.npz",
                "candidate_request": candidate_dir / "request.json",
                "candidate_summary": candidate_dir / "summary.json",
                "static_checkpoint": baseline_dir
                / "static"
                / fold_name
                / f"seed-{seed}"
                / "checkpoint.pt",
                "teacher_checkpoint": baseline_dir
                / "teacher"
                / fold_name
                / f"seed-{seed}"
                / "checkpoint.pt",
                "teacher_predictions": baseline_dir
                / "teacher"
                / fold_name
                / f"seed-{seed}"
                / "held_predictions.npz",
            }
            evidence = {
                name: file_evidence(path, relative_to=repository_root)
                for name, path in files.items()
            }
            request = load_json_bytes(capture(files["candidate_request"]), name="candidate request")
            summary = load_json_bytes(capture(files["candidate_summary"]), name="candidate summary")
            request_payload = {
                key: value for key, value in request.items() if key != "request_sha256"
            }
            canonical_request_sha256 = sha256_bytes(
                json.dumps(request_payload, sort_keys=True).encode("utf-8")
            )
            if (
                request.get("request_sha256") != canonical_request_sha256
                or summary.get("request_sha256") != canonical_request_sha256
            ):
                raise RuntimeError(
                    "Historical candidate request, canonical hash, and summary disagree"
                )
            if any(
                int(request.get(key, -1)) != 0
                for key in (
                    "validation_samples_read",
                    "calibration_samples_read",
                    "confirmation_samples_read",
                )
            ):
                raise RuntimeError("A historical candidate request reports protected-role access")
            if (
                request.get("candidate_id") != "centre_short_parts"
                or int(request.get("fold", -1)) != fold
                or int(request.get("seed", -1)) != seed
                or int(request.get("fixed_epochs", -1)) != EXPECTED_FIXED_EPOCHS[seed]
            ):
                raise RuntimeError(
                    "A historical candidate request differs from the replay contract"
                )
            if summary.get("status") != "OKUTAMA_CPTR_CROSSFIT_RUN_COMPLETE":
                raise RuntimeError("A historical candidate run is incomplete")
            if (
                summary.get("artifact_sha256", {}).get("checkpoint.pt")
                != evidence["candidate_checkpoint"]["sha256"]
            ):
                raise RuntimeError("A candidate checkpoint differs from its historical summary")
            if (
                summary.get("artifact_sha256", {}).get("held_predictions.npz")
                != evidence["candidate_predictions"]["sha256"]
            ):
                raise RuntimeError("Candidate predictions differ from their historical summary")
            if (
                summary.get("baseline_checkpoint_sha256", {}).get("static")
                != evidence["static_checkpoint"]["sha256"]
                or summary.get("baseline_checkpoint_sha256", {}).get("teacher")
                != evidence["teacher_checkpoint"]["sha256"]
            ):
                raise RuntimeError("Baseline checkpoint hashes differ from the candidate summary")

            candidate = load_npz_bytes(
                capture(files["candidate_predictions"]), name="candidate predictions"
            )
            teacher = load_npz_bytes(
                capture(files["teacher_predictions"]), name="teacher predictions"
            )
            validate_prediction_arrays(
                candidate,
                name=f"{fold_name}/seed-{seed} candidate",
                expected=expected,
                baseline=False,
            )
            validate_prediction_arrays(
                teacher, name=f"{fold_name}/seed-{seed} teacher", expected=expected, baseline=True
            )
            candidate_ids = candidate["sample_ids"].astype(str)
            teacher_position = {
                value: index for index, value in enumerate(teacher["sample_ids"].astype(str))
            }
            order = np.asarray([teacher_position[value] for value in candidate_ids], dtype=np.int64)
            if not np.array_equal(
                candidate["baseline_probabilities"], teacher["probabilities"][order].astype(float)
            ):
                raise RuntimeError("Candidate and separately retained teacher predictions differ")
            if reference_ids is None:
                reference_ids = candidate_ids
            elif not np.array_equal(reference_ids, candidate_ids):
                raise RuntimeError(f"Held row order differs across seeds in {fold_name}")
            entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "fixed_epochs": EXPECTED_FIXED_EPOCHS[seed],
                    "request_sha256": canonical_request_sha256,
                    "held_rows": len(expected),
                    "files": evidence,
                }
            )
    return entries


def write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="")
    temporary.replace(path)


def _validate_materialization_authorization(lock: dict[str, Any]) -> None:
    protected = lock.get("protected_access", {})
    for name in (
        "mixed_manifest_rows_read",
        "mixed_development_metadata_rows_read",
        "calibration_rows_or_arrays_read",
        "confirmation_rows_or_arrays_read",
        "test_rows_or_arrays_read",
    ):
        if int(protected.get(name, -1)) != 0:
            raise RuntimeError(f"Materialization lock does not attest {name}=0")
    authorization = lock.get("authorization", {})
    if authorization.get("eligible_feature_bundle_construction") is not True:
        raise RuntimeError("Source lock does not authorize eligible feature bundle construction")
    if any(
        authorization.get(name) is not False
        for name in ("R1a_execution", "R1b_execution", "R2_fitting")
    ):
        raise RuntimeError("Materialization source lock improperly authorizes model execution")


def _load_declared_array(
    declaration_path: Path,
    declaration: dict[str, Any],
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> np.ndarray:
    item = declaration["arrays"][name]
    relative = Path(str(item["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe declared array path for {name}")
    path = (declaration_path.resolve().parent / relative).resolve()
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != expected_shape or array.dtype != np.dtype(expected_dtype):
        raise RuntimeError(f"Declared {name} shape/dtype changed: {array.shape}, {array.dtype}")
    return array


def _load_window_masks(path: Path, eligible_ids: np.ndarray) -> tuple[np.ndarray, str]:
    resolved = path.resolve()
    if resolved.name.lower() in FORBIDDEN_NAMES:
        raise RuntimeError("A forbidden broad metadata source cannot supply replay masks")
    raw = resolved.read_bytes()
    arrays = load_npz_bytes(raw, name="role-safe window masks")
    if set(arrays) != {"sample_ids", "window_occluded"}:
        raise RuntimeError("Role-safe mask artifact must contain sample_ids and window_occluded")
    ids = arrays["sample_ids"].astype(str)
    masks = arrays["window_occluded"]
    if (
        len(ids) != EXPECTED_ROWS
        or len(np.unique(ids)) != EXPECTED_ROWS
        or set(ids) != set(eligible_ids)
        or masks.shape != (EXPECTED_ROWS, 17)
        or masks.dtype != np.dtype("bool")
    ):
        raise RuntimeError("Role-safe mask artifact identity, shape, or dtype changed")
    positions = {value: index for index, value in enumerate(ids)}
    order = np.asarray([positions[value] for value in eligible_ids], dtype=np.int64)
    return masks[order], sha256_bytes(raw)


def _load_primary_retained_teacher_probabilities(
    inventory: dict[str, Any],
    eligible_index: pd.DataFrame,
    *,
    repository_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only inventory-bound OOF teacher predictions into primary bundle order."""

    primary = eligible_index.loc[
        eligible_index["scope"].eq("grouped_crossfit_oof"), ["sample_id", "fold"]
    ]
    if len(primary) != EXPECTED_OOF_ROWS or primary["sample_id"].duplicated().any():
        raise RuntimeError("Primary retained-teacher row identity changed")
    primary_ids = primary["sample_id"].to_numpy(dtype=str)
    primary_position = {value: index for index, value in enumerate(primary_ids)}
    probabilities = np.empty(
        (EXPECTED_OOF_ROWS, len(EXPECTED_SEEDS), 3), dtype=np.float32
    )
    assigned = np.zeros((EXPECTED_OOF_ROWS, len(EXPECTED_SEEDS)), dtype=bool)
    entries = inventory.get("checkpoints")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_FOLDS) * len(
        EXPECTED_SEEDS
    ):
        raise RuntimeError("Checkpoint inventory cannot supply retained teacher predictions")
    seed_position = {seed: index for index, seed in enumerate(EXPECTED_SEEDS)}
    for entry in entries:
        fold = int(entry.get("fold", -1))
        seed = int(entry.get("seed", -1))
        if fold not in range(len(EXPECTED_FOLDS)) or seed not in seed_position:
            raise RuntimeError("Checkpoint inventory fold/seed changed")
        evidence = entry.get("files", {}).get("teacher_predictions", {})
        relative = Path(str(evidence.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe retained-teacher prediction path")
        path = (repository_root / relative).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(evidence.get("bytes", -1))
            or sha256_file(path) != evidence.get("sha256")
        ):
            raise RuntimeError("Retained-teacher prediction file differs from inventory")
        arrays = load_npz_bytes(capture(path), name="retained teacher predictions")
        expected_fold = primary.loc[primary["fold"].eq(f"fold-{fold}"), "sample_id"].astype(
            str
        )
        validate_prediction_arrays(
            arrays,
            name=f"fold-{fold}/seed-{seed} retained teacher",
            expected=eligible_index.loc[eligible_index["sample_id"].isin(expected_fold)],
            baseline=True,
        )
        ids = arrays["sample_ids"].astype(str)
        if set(ids) != set(expected_fold):
            raise RuntimeError("Retained-teacher prediction fold membership changed")
        rows = np.asarray([primary_position[value] for value in ids], dtype=np.int64)
        column = seed_position[seed]
        if assigned[rows, column].any():
            raise RuntimeError("Duplicate retained-teacher row/seed assignment")
        values = arrays["probabilities"].astype(np.float32, copy=False)
        probabilities[rows, column] = values
        assigned[rows, column] = True
    if not assigned.all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Retained-teacher prediction matrix is incomplete or non-finite")
    if np.max(np.abs(probabilities.sum(axis=2) - 1.0)) > 1e-5:
        raise RuntimeError("Retained-teacher probabilities are not normalized")
    return primary_ids, np.asarray(EXPECTED_SEEDS, dtype=np.int64), probabilities


def materialize_role_safe_bundle(
    paths: Paths,
    *,
    inventory_path: Path,
    eligible_index_path: Path,
    window_masks_path: Path,
    window_mask_summary_path: Path,
    lock_path: Path,
    bundle_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    inventory = load_json_bytes(capture(inventory_path), name="input inventory")
    lock = load_json_bytes(capture(lock_path), name="materialization lock")
    if inventory.get("status") != STATUS or lock.get("status") != MATERIALIZATION_LOCK_STATUS:
        raise RuntimeError("The R0 inventory or pre-access materialization lock is incomplete")
    _validate_materialization_authorization(lock)
    locked = lock.get("source_sha256", {})
    required = {
        "input_inventory": inventory_path,
        "eligible_index": eligible_index_path,
        "eligible_window_masks": window_masks_path,
        "window_mask_summary": window_mask_summary_path,
        "bundle_builder": Path(__file__),
        "cptr_feature_module": repository_root / "src/hac/cptr_features.py",
    }
    for name, path in required.items():
        observed = sha256_file(path.resolve())
        if locked.get(name) != observed:
            raise RuntimeError(f"Materialization lock mismatch for {name}: {observed}")
    if inventory.get("artifacts", {}).get("eligible_index.csv", {}).get("sha256") != sha256_file(
        eligible_index_path
    ):
        raise RuntimeError("Eligible index differs from its R0 inventory")

    mask_summary = load_json_bytes(
        capture(window_mask_summary_path), name="role-safe window-mask summary"
    )
    if (
        mask_summary.get("status")
        != "OKUTAMA_CPTR_ROLE_SAFE_EXACT_WINDOW_MASKS_COMPLETE"
        or int(mask_summary.get("rows", -1)) != EXPECTED_ROWS
        or int(mask_summary.get("scenarios", -1)) != EXPECTED_SCENARIOS
    ):
        raise RuntimeError("Role-safe window-mask summary status or scope changed")
    mask_summary_sources = mask_summary.get("source_sha256", {})
    expected_mask_sources = {
        "archive": EXPECTED_ARCHIVE_SHA256,
        "input_inventory": sha256_file(inventory_path),
        "eligible_index": sha256_file(eligible_index_path),
        "extractor": sha256_file(Path(__file__)),
    }
    for name, expected in expected_mask_sources.items():
        if mask_summary_sources.get(name) != expected:
            raise RuntimeError(f"Role-safe window-mask source mismatch: {name}")
    for name in (
        "disallowed_annotation_members_read",
        "image_members_read",
        "mixed_development_manifest_rows_read",
        "broad_development_metadata_rows_read",
        "confirmation_rows_read",
        "test_rows_read",
        "feature_array_values_read",
        "checkpoints_loaded",
    ):
        if int(mask_summary.get("access_accounting", {}).get(name, -1)) != 0:
            raise RuntimeError(f"Unsafe access reported by role-safe window masks: {name}")

    index = pd.read_csv(eligible_index_path, dtype=str, keep_default_na=False)
    if len(index) != EXPECTED_ROWS or index["sample_id"].duplicated().any():
        raise RuntimeError("Eligible index cardinality or identity changed")
    feature_indices = pd.to_numeric(index["feature_index"], errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(feature_indices)) != EXPECTED_ROWS:
        raise RuntimeError("Eligible feature positions are not one-to-one")
    window_occluded, mask_hash = _load_window_masks(
        window_masks_path, index["sample_id"].astype(str).to_numpy()
    )
    if mask_summary.get("artifact_sha256", {}).get(window_masks_path.name) != mask_hash:
        raise RuntimeError("Role-safe window-mask artifact differs from its summary")

    base_raw = capture(paths.base_store)
    part_raw = capture(paths.part_store)
    base_store = validate_store(
        base_raw,
        name="base store",
        expected_status="VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE",
        expected_sha256=EXPECTED_BASE_STORE_SHA256,
    )
    part_store = validate_store(
        part_raw,
        name="part store",
        expected_status="OKUTAMA_CPTR_FEATURE_STORE_COMPLETE",
        expected_sha256=EXPECTED_PART_STORE_SHA256,
    )
    base_array_receipts = verify_declared_array_receipts(
        paths.base_store,
        base_store,
        store_name="base",
        expected_names={"tight", "context", "geometry"},
        materialization_lock=lock,
        repository_root=repository_root,
    )
    part_array_receipts = verify_declared_array_receipts(
        paths.part_store,
        part_store,
        store_name="part",
        expected_names={"part_tokens", "part_confidence"},
        materialization_lock=lock,
        repository_root=repository_root,
    )
    dimensions = int(base_store["feature_dimensions"])
    part_dimensions = int(part_store["part_token_dim"])
    tight = _load_declared_array(
        paths.base_store,
        base_store,
        "tight",
        (EXPECTED_FEATURE_ROWS, 17, dimensions),
        "float32",
    )
    context = _load_declared_array(
        paths.base_store,
        base_store,
        "context",
        (EXPECTED_FEATURE_ROWS, 17, dimensions),
        "float32",
    )
    geometry = _load_declared_array(
        paths.base_store,
        base_store,
        "geometry",
        (EXPECTED_FEATURE_ROWS, 17, 6),
        "float32",
    )
    parts = _load_declared_array(
        paths.part_store,
        part_store,
        "part_tokens",
        (EXPECTED_FEATURE_ROWS, 17, 7, part_dimensions),
        "float16",
    )
    part_confidence_all = _load_declared_array(
        paths.part_store,
        part_store,
        "part_confidence",
        (EXPECTED_FEATURE_ROWS, 17, 7),
        "float16",
    )

    short_indices, short_centre = sample_indices_with_centre(
        17, centre_index=8, samples=8, span_frames=8
    )
    long_indices, long_centre = sample_indices_with_centre(
        17, centre_index=8, samples=8, span_frames=16
    )
    if short_indices.tolist() != [4, 6, 6, 8, 8, 10, 10, 12]:
        raise RuntimeError("Historical short-window preprocessing changed")
    if int(short_centre) != 4 or int(long_centre) != 3:
        raise RuntimeError("Historical centre-slot preprocessing changed")

    # Two-axis advanced indexing reads only eligible rows and requested temporal slots.
    # In particular, it avoids materializing the 17-frame part tensor (~2.3 GB float32)
    # before retaining the historical eight long-window slots.
    legacy_positions_in_distinct = np.asarray(
        [int(np.flatnonzero(DISTINCT_SHORT_INDICES == value)[0]) for value in short_indices],
        dtype=np.int64,
    )
    if DISTINCT_SHORT_INDICES[DISTINCT_SHORT_CENTRE] != 8:
        raise RuntimeError("Declared distinct-window center changed")
    row_indices = feature_indices[:, None]
    short_slots = DISTINCT_SHORT_INDICES[None, :]
    long_slots = long_indices[None, :]
    selected_tight = np.asarray(tight[row_indices, short_slots], dtype=np.float32)
    selected_context = np.asarray(context[row_indices, short_slots], dtype=np.float32)
    selected_geometry = np.asarray(geometry[row_indices, short_slots], dtype=np.float32)
    selected_parts = np.asarray(parts[row_indices, long_slots], dtype=np.float32)
    selected_part_confidence_full = np.asarray(
        part_confidence_all[feature_indices], dtype=np.float32
    )
    selected_part_confidence = selected_part_confidence_full[:, long_indices]
    distinct_combined = np.concatenate(
        (selected_tight, selected_context, selected_geometry), axis=2
    )
    combined = distinct_combined[:, legacy_positions_in_distinct]
    valid = ~window_occluded
    camera_quality = valid.any(axis=1).astype(np.float32)
    quality = np.stack(
        (
            window_occluded[:, 8].astype(np.float32),
            window_occluded.mean(axis=1, dtype=np.float32),
            selected_geometry[:, DISTINCT_SHORT_CENTRE, 0],
            selected_geometry[:, DISTINCT_SHORT_CENTRE, 1],
            selected_geometry[:, DISTINCT_SHORT_CENTRE, 5],
            camera_quality,
            selected_part_confidence_full.mean(axis=(1, 2), dtype=np.float32),
            np.zeros(EXPECTED_ROWS, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    labels = pd.to_numeric(index["label_index"], errors="raise").to_numpy(dtype=np.int64)
    transition = index["transition_window"].str.lower().eq("true").to_numpy(dtype=bool)
    occlusion_target = index["window_any_occluded"].str.lower().eq("true").to_numpy(dtype=bool)
    if not np.array_equal(occlusion_target, window_occluded.any(axis=1)):
        raise RuntimeError("Role-safe exact masks disagree with retained occlusion targets")
    retained_teacher_ids, retained_teacher_seeds, retained_teacher_probabilities = (
        _load_primary_retained_teacher_probabilities(
            inventory,
            index,
            repository_root=repository_root,
        )
    )
    # Close the hash-to-mmap window: verify every mixed source file again after the
    # selective copies have completed and before any bundle output is committed.
    post_copy_base_receipts = verify_declared_array_receipts(
        paths.base_store,
        base_store,
        store_name="base",
        expected_names={"tight", "context", "geometry"},
        materialization_lock=lock,
        repository_root=repository_root,
    )
    post_copy_part_receipts = verify_declared_array_receipts(
        paths.part_store,
        part_store,
        store_name="part",
        expected_names={"part_tokens", "part_confidence"},
        materialization_lock=lock,
        repository_root=repository_root,
    )
    if (
        post_copy_base_receipts != base_array_receipts
        or post_copy_part_receipts != part_array_receipts
    ):
        raise RuntimeError("A feature-array source changed during selective materialization")

    bundle_path = bundle_path.resolve()
    summary_path = summary_path.resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists() or summary_path.exists():
        raise FileExistsError("Role-safe bundle outputs already exist")
    temporary = bundle_path.with_name(bundle_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_ids=np.asarray(index["sample_id"].astype(str), dtype=str),
        recording_ids=np.asarray(index["recording_id"].astype(str), dtype=str),
        track_ids=np.asarray(index["track_id"].astype(str), dtype=str),
        scope=np.asarray(index["scope"].astype(str), dtype=str),
        fold=np.asarray(index["fold"].astype(str), dtype=str),
        labels=labels,
        transition_targets=transition,
        occlusion_targets=occlusion_target,
        source_feature_indices=feature_indices,
        window_occluded=window_occluded,
        static_features=distinct_combined[:, DISTINCT_SHORT_CENTRE],
        short_features=combined,
        short_valid_mask=valid[:, short_indices],
        short_centre_index=np.asarray(short_centre, dtype=np.int64),
        distinct_short_features=distinct_combined,
        distinct_short_valid_mask=valid[:, DISTINCT_SHORT_INDICES],
        distinct_short_indices=DISTINCT_SHORT_INDICES,
        distinct_short_centre_index=np.asarray(DISTINCT_SHORT_CENTRE, dtype=np.int64),
        part_tokens=selected_parts,
        part_confidence=selected_part_confidence,
        part_valid_mask=valid[:, long_indices],
        part_centre_index=np.asarray(long_centre, dtype=np.int64),
        quality_features=quality,
        primary_retained_teacher_sample_ids=retained_teacher_ids,
        retained_teacher_seeds=retained_teacher_seeds,
        primary_retained_teacher_probabilities=retained_teacher_probabilities,
    )
    temporary.replace(bundle_path)
    summary = {
        "status": BUNDLE_STATUS,
        "rows": EXPECTED_ROWS,
        "scenarios": EXPECTED_SCENARIOS,
        "source_feature_rows": EXPECTED_FEATURE_ROWS,
        "source_feature_values_read": "allowed_indices_only_via_numpy_mmap",
        "excluded_feature_indices_count": EXPECTED_FEATURE_ROWS - EXPECTED_ROWS,
        "excluded_feature_identities_or_values_read": 0,
        "short_indices": short_indices.tolist(),
        "short_centre_index": int(short_centre),
        "distinct_short_indices": DISTINCT_SHORT_INDICES.tolist(),
        "distinct_short_centre_index": DISTINCT_SHORT_CENTRE,
        "distinct_short_role": "preauthorized_T2_input_only_not_used_by_R1",
        "retained_teacher_probabilities": {
            "scope": "4977_grouped_crossfit_oof_rows_only",
            "shape": [EXPECTED_OOF_ROWS, len(EXPECTED_SEEDS), 3],
            "dtype": "float32",
            "seed_order": list(EXPECTED_SEEDS),
            "row_order": "primary_rows_in_eligible_feature_bundle_order",
            "source": "inventory_bound_role_safe_teacher_held_predictions",
        },
        "part_indices": long_indices.tolist(),
        "part_centre_index": int(long_centre),
        "q_values": sorted(np.unique(valid[:, short_indices].mean(axis=1)).tolist()),
        "source_sha256": {
            "protocol_lock": sha256_file(lock_path),
            "input_inventory": sha256_file(inventory_path),
            "eligible_index": sha256_file(eligible_index_path),
            "eligible_window_masks": mask_hash,
            "window_mask_summary": sha256_file(window_mask_summary_path),
            "base_store_declaration": sha256_bytes(base_raw),
            "part_store_declaration": sha256_bytes(part_raw),
            "bundle_builder": sha256_file(Path(__file__)),
            "cptr_feature_module": sha256_file(repository_root / "src/hac/cptr_features.py"),
            **{
                f"feature_array_{name}": receipt["sha256"]
                for name, receipt in {
                    **base_array_receipts,
                    **part_array_receipts,
                }.items()
            },
        },
        "source_store_receipts": {
            "base_declared_arrays": base_store["arrays"],
            "part_declared_arrays": part_store["arrays"],
            "observed_opaque_file_receipts": {
                **base_array_receipts,
                **part_array_receipts,
            },
        },
        "artifact_sha256": {bundle_path.name: sha256_file(bundle_path)},
        "access_accounting": {
            "mixed_development_manifest_rows_read": 0,
            "broad_development_metadata_rows_read": 0,
            "protected_calibration_rows_read": 0,
            "confirmation_rows_read": 0,
            "test_rows_read": 0,
            "checkpoints_loaded": 0,
            "role_safe_teacher_prediction_arrays_read": len(EXPECTED_FOLDS)
            * len(EXPECTED_SEEDS),
            "feature_array_headers_opened_during_materialization": 5,
            "feature_array_files_opaque_hashed": 5,
            "feature_array_opaque_hash_passes": 2,
            "feature_array_values_interpreted_during_opaque_hash": 0,
            "feature_array_values_read": "eligible_indices_only",
            "known_feature_array_header_opens_before_materialization_lock": 5,
            "known_feature_array_values_read_before_materialization_lock": 0,
        },
        "known_operator_exposure_before_materialization_lock": {
            "development_metadata_csv": {
                "path": (
                    ".runs/vcoco_v3/okutama/features/dinov2_base/"
                    "development_metadata.csv"
                ),
                "rows_loaded": 8339,
                "displayed": "column_names_shape_nunique_and_five_first_row_examples",
                "used_by_bundle": False,
            },
            "mixed_feature_store_headers": {
                "headers_opened": 5,
                "values_read": 0,
                "method": "numpy_load_mmap_mode_r_header_only",
            },
            "timing": "before_source_or_materialization_lock",
        },
    }
    write_text_atomic(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def run(paths: Paths, output_dir: Path) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    paired_raw = capture(paths.paired_rows)
    audit_raw = capture(paths.audit_summary)
    if sha256_bytes(paired_raw) != EXPECTED_PAIRED_SHA256:
        raise RuntimeError("The permitted paired-row source changed")
    if sha256_bytes(audit_raw) != EXPECTED_AUDIT_SHA256:
        raise RuntimeError("The aggregate Okutama audit changed")
    paired = pd.read_csv(BytesIO(paired_raw), dtype=str, keep_default_na=False)
    audit = load_json_bytes(audit_raw, name="audit summary")
    eligible_index = derive_role_safe_index(paired, audit)

    base_raw = capture(paths.base_store)
    part_raw = capture(paths.part_store)
    base_store = validate_store(
        base_raw,
        name="base store",
        expected_status="VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE",
        expected_sha256=EXPECTED_BASE_STORE_SHA256,
    )
    part_store = validate_store(
        part_raw,
        name="part store",
        expected_status="OKUTAMA_CPTR_FEATURE_STORE_COMPLETE",
        expected_sha256=EXPECTED_PART_STORE_SHA256,
    )
    entries = checkpoint_inventory(
        paths,
        eligible_index,
        repository_root=repository_root,
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "eligible_index.csv"
    inventory_path = output_dir / "input_inventory.json"
    if index_path.exists() or inventory_path.exists():
        raise FileExistsError(
            "Replay inventory outputs already exist; use a fresh output directory"
        )
    index_csv = eligible_index.to_csv(index=False, lineterminator="\n")
    write_text_atomic(index_path, index_csv)
    source_paths = {
        "paired_rows": paths.paired_rows,
        "paired_summary": paths.paired_summary,
        "audit_summary": paths.audit_summary,
        "base_store": paths.base_store,
        "part_store": paths.part_store,
        "crossfit_plan": paths.crossfit_plan,
        "candidate_grid": paths.candidate_grid,
        "cptr_protocol": paths.cptr_protocol,
        "temporal_grid": paths.temporal_grid,
        "bundle_builder": Path(__file__),
        "replay_runner": repository_root / "experiments/replay_okutama_cptr_interventions.py",
        "cptr_model_module": repository_root / "src/hac/cptr.py",
        "cptr_feature_module": repository_root / "src/hac/cptr_features.py",
        "cptr_training_module": repository_root / "src/hac/cptr_training.py",
    }
    inventory = {
        "status": STATUS,
        "index_status": INDEX_STATUS,
        "role": "development_only_train_oof_plus_fixed_validation",
        "rows": len(eligible_index),
        "eligible_rows": len(eligible_index),
        "oof_rows": int(eligible_index["scope"].eq("grouped_crossfit_oof").sum()),
        "fixed_validation_rows": int(
            eligible_index["scope"].eq("fixed_development_validation").sum()
        ),
        "scenarios": int(eligible_index["recording_id"].nunique()),
        "eligible_scenarios": int(eligible_index["recording_id"].nunique()),
        "source_feature_rows": EXPECTED_FEATURE_ROWS,
        "excluded_feature_indices_count": EXPECTED_FEATURE_ROWS - len(eligible_index),
        "excluded_feature_identities_or_values_read": 0,
        "feature_index_derivation": {
            "global_sort": ["scenario_id", "recording_id", "track_id_as_string", "center_frame"],
            "block_offsets": "aggregate_audit_recording_evidence_selected_centres",
            "within_block_order": "permitted_sample_id_track_as_string_then_center_frame",
            "all_allowed_scenarios_complete": True,
        },
        "feature_store_declarations": {
            "base": {
                "sha256": sha256_bytes(base_raw),
                "declared_arrays": base_store["arrays"],
                "feature_dimensions": int(base_store["feature_dimensions"]),
                "frames_per_sample": int(base_store["frames_per_sample"]),
            },
            "parts": {
                "sha256": sha256_bytes(part_raw),
                "declared_arrays": part_store["arrays"],
                "part_token_dim": int(part_store["part_token_dim"]),
                "frames_per_sample": int(part_store["frames_per_sample"]),
            },
        },
        "checkpoints": entries,
        "source_files": {
            name: file_evidence(path, relative_to=repository_root)
            for name, path in source_paths.items()
        },
        "artifacts": {
            "eligible_index.csv": {
                "bytes": index_path.stat().st_size,
                "sha256": sha256_file(index_path),
            }
        },
        "artifact_sha256": {"eligible_index.csv": sha256_file(index_path)},
        "access_accounting": {
            "mixed_development_manifest_rows_read": 0,
            "protected_calibration_rows_read": 0,
            "confirmation_rows_read": 0,
            "test_rows_read": 0,
            "feature_array_values_read": 0,
            "known_feature_array_header_opens_before_materialization_lock": 5,
            "known_feature_array_values_read_before_materialization_lock": 0,
            "checkpoints_loaded": 0,
        },
        "operator_exposure_before_lock": {
            "path": ".runs/vcoco_v3/okutama/features/dinov2_base/development_metadata.csv",
            "rows_loaded": 8339,
            "displayed": "column_names_shape_nunique_and_five_first_row_examples",
            "feature_array_values_images_or_checkpoints_loaded": False,
            "used_by_inventory_or_replay": False,
            "timing": "before_source_or_materialization_lock",
        },
        "additional_operator_exposure_before_materialization_lock": {
            "feature_array_headers_opened": 5,
            "feature_array_values_read": 0,
            "method": "numpy_load_mmap_mode_r_header_only",
            "purpose": "confirm_declared_shapes_and_dtypes",
            "arrays": [
                "base:tight",
                "base:context",
                "base:geometry",
                "parts:part_tokens",
                "parts:part_confidence",
            ],
        },
        "window_mask_status": "missing_role_safe_exact_17_frame_masks",
        "execution_status": "R0_INVENTORY_ONLY_R1_NOT_AUTHORIZED_BY_THIS_ARTIFACT",
    }
    write_text_atomic(
        inventory_path,
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return inventory


def main() -> None:
    args = parse_args()
    names = Paths.__dataclass_fields__
    paths = Paths(**{name: getattr(args, name) for name in names})
    if args.extract_window_masks:
        output_dir = args.output_dir.resolve()
        inventory_path = (args.inventory or output_dir / "input_inventory.json").resolve()
        eligible_index_path = (args.eligible_index or output_dir / "eligible_index.csv").resolve()
        mask_path = (args.window_masks_output or output_dir / "eligible_window_masks.npz").resolve()
        mask_summary_path = (
            args.window_mask_summary or output_dir / "window_mask_summary.json"
        ).resolve()
        if args.archive is None:
            raise ValueError("Exact-window-mask extraction requires --archive")
        result = extract_role_safe_window_masks(
            archive_path=args.archive,
            inventory_path=inventory_path,
            eligible_index_path=eligible_index_path,
            output_path=mask_path,
            summary_path=mask_summary_path,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "rows": result["rows"],
                    "annotation_members_read_count": result["annotation_members_read_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.materialize:
        output_dir = args.output_dir.resolve()
        inventory_path = (args.inventory or output_dir / "input_inventory.json").resolve()
        eligible_index_path = (args.eligible_index or output_dir / "eligible_index.csv").resolve()
        mask_summary_path = (
            args.window_mask_summary or output_dir / "window_mask_summary.json"
        ).resolve()
        bundle_path = (args.bundle_output or output_dir / "eligible_feature_bundle.npz").resolve()
        summary_path = (args.bundle_summary or output_dir / "bundle_summary.json").resolve()
        if args.materialization_lock is None or args.window_masks is None:
            raise ValueError("Materialization requires --materialization-lock and --window-masks")
        result = materialize_role_safe_bundle(
            paths,
            inventory_path=inventory_path,
            eligible_index_path=eligible_index_path,
            window_masks_path=args.window_masks,
            window_mask_summary_path=mask_summary_path,
            lock_path=args.materialization_lock,
            bundle_path=bundle_path,
            summary_path=summary_path,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "rows": result["rows"],
                    "scenarios": result["scenarios"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run(paths, args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "scenarios": result["scenarios"],
                "checkpoint_sets": len(result["checkpoints"]),
                "execution_status": result["execution_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
