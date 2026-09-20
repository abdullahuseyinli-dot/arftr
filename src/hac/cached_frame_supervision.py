"""Leakage-safe labels and identities for the frozen Okutama DINO frame caches.

This module materializes metadata only.  It opens the allowlisted provider
annotation members, but never reads an image payload, decodes a frame, runs a
backbone, or fits a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hac.actor_memory_data import annotation_target
from hac.okutama_native_video import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_PRIMARY_ROWS,
    EXPECTED_RECORDINGS,
    read_stable_annotations,
    sha256_file,
)

CENTER_INDEX = 8
FRAME_COUNT = 16
FEATURE_DIM = 768
TRACK_EFFECTIVE_FRAME_CAP = 128.0
STREAMS = ("short", "long")

SOURCE_PATHS = {
    "eligible_index": ".runs/research_20260907/cptr_replay_r0/eligible_index.csv",
    "short_manifest": (
        ".runs/research_20260907/okutama_native_video_p0_r1/manifest/frame_manifest.csv"
    ),
    "short_feature": (
        ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/"
        "dinov2_native_frames.npy"
    ),
    "short_summary": (
        ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json"
    ),
    "short_validity": (
        ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/validity.npy"
    ),
    "short_completed": (
        ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/completed.npy"
    ),
    "long_manifest": (
        ".runs/research_20260907/okutama_native_video_p3/manifest/frame_manifest.csv"
    ),
    "long_feature": (
        ".runs/research_20260907/okutama_native_video_p3/dinov2_full/"
        "dinov2_long16_native_frames.npy"
    ),
    "long_summary": (
        ".runs/research_20260907/okutama_native_video_p3/dinov2_full/summary.json"
    ),
    "long_validity": (
        ".runs/research_20260907/okutama_native_video_p3/dinov2_full/validity.npy"
    ),
    "long_completed": (
        ".runs/research_20260907/okutama_native_video_p3/dinov2_full/completed.npy"
    ),
}

# These immutable inputs were already sealed by the P0/P2/P3/P5 execution locks.
SOURCE_SHA256 = {
    "eligible_index": "5bf6d0cc11a18d3e986b714f8a3de71cadd9fef3bd233eb9b5f212f701aadbbb",
    "short_manifest": "b69fe9e00057ff535698afb02cb3e9031ce3f53246c7e7ba64f061d1d557427f",
    "short_feature": "b40cfea1659795bf7554b0e77f7f6d95226c1745038baca884460065546ec2f2",
    "short_summary": "17086c64f9627b2a20865976c698f5541f954967ab96fb4961bd2831519b6267",
    "short_validity": "dbc5751701174ec2ed9e60adcd129d2868eeebb9099310f48b57b9145ef6e732",
    "short_completed": "f33c11f806e4c6913c5e64b1220a0499732e9fde7f40b1584b15969cd43a3c27",
    "long_manifest": "68a6f62dd0c033fd05421272345587f369da8d5ab04668d1f368477866c9cb9c",
    "long_feature": "c762f9254ebfc23e617d76c31bce85faabd7ff629a4f641cc7d018800f96f838",
    "long_summary": "2ee40807ff9c7e9e472c64088659bc2a9782264ec86699ea52c3a1805c2c16d7",
    "long_validity": "45b23fbc3aeb1fe8a38b78c1127dd815a3068af8f6dfb109ac8638478e51bd5d",
    "long_completed": "f33c11f806e4c6913c5e64b1220a0499732e9fde7f40b1584b15969cd43a3c27",
}

REQUIRED_KEYS = {
    "sample_ids",
    "labels",
    "scenarios",
    "folds",
    "recordings",
    "tracks",
    "center_frames",
    "short_source_frames",
    "short_valid",
    "short_frame_labels",
    "long_source_frames",
    "long_valid",
    "long_frame_labels",
    "physical_weight",
}


def _receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _checked_source(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    path = (root / SOURCE_PATHS[name]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached-frame source: {name}")
    receipt = _receipt(path)
    if receipt["sha256"] != SOURCE_SHA256[name]:
        raise RuntimeError(f"Cached-frame source digest changed: {name}")
    receipt["path"] = SOURCE_PATHS[name]
    return path, receipt


def _read_primary(index_path: Path) -> dict[str, np.ndarray]:
    # Import locally to make the identity validator explicit at the boundary.
    from hac.okutama_native_video import load_primary_index

    rows = load_primary_index(index_path)
    folds = np.asarray([int(row["fold"].removeprefix("fold-")) for row in rows], np.int8)
    return {
        "sample_ids": np.asarray([row["sample_id"] for row in rows]),
        "labels": np.asarray([int(row["label_index"]) for row in rows], np.int8),
        "scenarios": np.asarray([row["recording_id"] for row in rows]),
        "folds": folds,
        "recordings": np.asarray([row["provider_recording_id"] for row in rows]),
        "tracks": np.asarray([row["provider_track_id"] for row in rows]),
        "center_frames": np.asarray([int(row["center_frame"]) for row in rows], np.int32),
    }


def read_ordered_frame_manifest(
    path: Path,
    primary: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Read exactly N x 16 manifest rows in cache order, without image access."""
    n = len(primary["sample_ids"])
    source_frames = np.empty((n, FRAME_COUNT), np.int32)
    manifest_valid = np.empty((n, FRAME_COUNT), bool)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = iter(csv.DictReader(stream))
        for row_index in range(n):
            for slot in range(FRAME_COUNT):
                try:
                    row = next(rows)
                except StopIteration as error:
                    raise RuntimeError("Frame manifest ended before N x 16 rows") from error
                expected = (
                    str(primary["sample_ids"][row_index]),
                    str(primary["recordings"][row_index]),
                    str(primary["tracks"][row_index]),
                )
                actual = (
                    row.get("sample_id", ""),
                    row.get("provider_recording_id", ""),
                    row.get("provider_track_id", ""),
                )
                if actual != expected or int(row.get("time_index", -1)) != slot:
                    raise RuntimeError("Frame manifest order or physical identity changed")
                source_frames[row_index, slot] = int(row["source_frame"])
                value = row.get("valid_frame", "").casefold()
                if value not in {"0", "1", "false", "true"}:
                    raise RuntimeError("Frame manifest validity is not Boolean")
                manifest_valid[row_index, slot] = value in {"1", "true"}
        try:
            next(rows)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Frame manifest contains rows beyond N x 16")
    if not np.array_equal(source_frames[:, CENTER_INDEX], primary["center_frames"]):
        raise RuntimeError("Frame manifest no longer places the center at index 8")
    if not manifest_valid[:, CENTER_INDEX].all():
        raise RuntimeError("Every immutable center frame must remain valid")
    return source_frames, manifest_valid


def _read_cache_contract(
    feature_path: Path,
    summary_path: Path,
    validity_path: Path,
    completed_path: Path,
    sample_ids: np.ndarray,
) -> np.ndarray:
    n = len(sample_ids)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in {
        "OKUTAMA_DINOV2_FROZEN_FEATURE_CACHE_COMPLETE",
        "OKUTAMA_P3_LONG_FROZEN_FEATURE_CACHE_COMPLETE",
    } or summary.get("sample_ids") != sample_ids.tolist():
        raise RuntimeError("DINO cache status or row identity changed")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (n, FRAME_COUNT, 1, FEATURE_DIM) or features.dtype != np.float16:
        raise RuntimeError("DINO frame cache shape or dtype changed")
    completed = np.load(completed_path, allow_pickle=False)
    if completed.shape != (n,) or completed.dtype != np.bool_ or not completed.all():
        raise RuntimeError("DINO cache is not completely materialized")
    validity = np.load(validity_path, allow_pickle=False)
    if validity.shape == (n, 1):
        validity = validity[:, 0]
    if validity.shape != (n,) or validity.dtype != np.bool_:
        raise RuntimeError("DINO cache validity contract changed")
    return validity.copy()


class _AnnotationOnlyArchive:
    """Proxy that makes reading any non-annotation member impossible."""

    def __init__(self, archive: zipfile.ZipFile, allowed: set[str]) -> None:
        self.archive = archive
        self.allowed = allowed
        self.member_sha256: dict[str, str] = {}
        self.payload_bytes = 0

    def read(self, member: str) -> bytes:
        if member not in self.allowed:
            raise RuntimeError("Attempted to read a non-permitted archive payload")
        payload = self.archive.read(member)
        self.member_sha256[member] = hashlib.sha256(payload).hexdigest()
        self.payload_bytes += len(payload)
        return payload


def _read_permitted_annotations(
    archive_path: Path,
) -> tuple[dict[str, dict], dict[str, Any]]:
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("Okutama training archive size changed")
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Okutama training archive digest changed")
    from hac.okutama_native_video import annotation_members

    allowed = {
        member for recording in EXPECTED_RECORDINGS for member in annotation_members(recording)
    }
    accessed: list[str] = []
    stable_by_recording: dict[str, dict] = {}
    with zipfile.ZipFile(archive_path) as source:
        guarded = _AnnotationOnlyArchive(source, allowed)
        for recording in sorted(EXPECTED_RECORDINGS):
            stable_by_recording[recording] = read_stable_annotations(
                guarded, recording, accessed  # type: ignore[arg-type]
            )
    if len(accessed) != 42 or set(accessed) != allowed or len(set(accessed)) != len(accessed):
        raise RuntimeError("Annotation access differed from the exact 42-member allowlist")
    return stable_by_recording, {
        "archive": _receipt(archive_path),
        "annotation_members": accessed,
        "annotation_member_sha256": guarded.member_sha256,
        "annotation_payload_reads": len(accessed),
        "annotation_payload_bytes": guarded.payload_bytes,
    }


def frame_targets(
    primary: dict[str, np.ndarray],
    source_frames: np.ndarray,
    input_valid: np.ndarray,
    stable_by_recording: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Map exact provider frames to one class; mask all other targets."""
    n = len(primary["sample_ids"])
    if source_frames.shape != (n, FRAME_COUNT) or input_valid.shape != source_frames.shape:
        raise ValueError("Frame targets require aligned N x 16 frames and input masks")
    labels = np.full(source_frames.shape, -1, np.int8)
    reasons: Counter[str] = Counter()
    for row in range(n):
        recording = str(primary["recordings"][row])
        track = int(primary["tracks"][row])
        stable = stable_by_recording[recording]
        for slot in range(FRAME_COUNT):
            target = annotation_target(stable.get((track, int(source_frames[row, slot]))))
            if target < 0:
                reasons["missing_lost_unsupported_or_multibase"] += 1
                continue
            labels[row, slot] = target
            if not input_valid[row, slot]:
                reasons["known_target_but_cached_input_invalid"] += 1
    valid = input_valid & (labels >= 0)
    if not np.array_equal(labels[:, CENTER_INDEX], primary["labels"]):
        raise RuntimeError("Exact cached center annotations disagree with immutable labels")
    reasons["valid"] = int(valid.sum())
    return labels, valid, dict(reasons)


def physical_multiplicity_weights(
    recordings: np.ndarray,
    tracks: np.ndarray,
    source_frames: np.ndarray,
    valid: np.ndarray,
    *,
    track_cap: float = TRACK_EFFECTIVE_FRAME_CAP,
) -> np.ndarray:
    """Split one physical-frame vote across crop variants, then cap each track.

    ``source_frames`` and ``valid`` have stream-major shape ``[S, N, 16]``.
    Thus a physical frame shared by short/long caches and/or overlapping clips
    still contributes a total pre-cap weight of exactly one.
    """
    if track_cap <= 0 or source_frames.ndim != 3 or source_frames.shape != valid.shape:
        raise ValueError("Physical weights need positive cap and aligned [S,N,T] arrays")
    streams, n, width = source_frames.shape
    if width != FRAME_COUNT or recordings.shape != (n,) or tracks.shape != (n,):
        raise ValueError("Physical identity vectors do not align with frame arrays")
    identities: dict[tuple[str, str, int], list[tuple[int, int, int]]] = defaultdict(list)
    for stream in range(streams):
        for row in range(n):
            for slot in np.flatnonzero(valid[stream, row]):
                key = (
                    str(recordings[row]),
                    str(tracks[row]),
                    int(source_frames[stream, row, slot]),
                )
                identities[key].append((stream, row, int(slot)))
    weights = np.zeros(source_frames.shape, np.float32)
    for occurrences in identities.values():
        value = np.float32(1.0 / len(occurrences))
        for index in occurrences:
            weights[index] = value
    track_totals: dict[tuple[str, str], float] = defaultdict(float)
    for (recording, track, _frame) in identities:
        track_totals[(recording, track)] += 1.0
    track_scale = {
        key: min(1.0, track_cap / total) for key, total in track_totals.items()
    }
    for (recording, track, _frame), occurrences in identities.items():
        scale = np.float32(track_scale[(recording, track)])
        for index in occurrences:
            weights[index] *= scale
    return weights


def _validate_partition_and_weights(data: dict[str, np.ndarray]) -> dict[str, float | int]:
    frames = np.stack((data["short_source_frames"], data["long_source_frames"]))
    valid = np.stack((data["short_valid"], data["long_valid"]))
    weights = data["physical_weight"]
    partitions: dict[tuple[str, str, int], set[tuple[str, int]]] = defaultdict(set)
    totals: dict[tuple[str, str, int], float] = defaultdict(float)
    track_totals: dict[tuple[str, str], float] = defaultdict(float)
    track_partitions: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for stream in range(len(STREAMS)):
        for row in range(len(data["sample_ids"])):
            for slot in np.flatnonzero(valid[stream, row]):
                key = (
                    str(data["recordings"][row]),
                    str(data["tracks"][row]),
                    int(frames[stream, row, slot]),
                )
                partitions[key].add((str(data["scenarios"][row]), int(data["folds"][row])))
                track_partitions[(str(data["recordings"][row]), str(data["tracks"][row]))].add(
                    (str(data["scenarios"][row]), int(data["folds"][row]))
                )
                totals[key] += float(weights[stream, row, slot])
    if any(len(values) != 1 for values in partitions.values()):
        raise RuntimeError("A physical frame crosses scenario/fold partitions")
    if any(len(values) != 1 for values in track_partitions.values()):
        raise RuntimeError("A provider track crosses scenario/fold partitions")
    for (recording, track, _frame), value in totals.items():
        if value > 1.0 + 1e-5:
            raise RuntimeError("A physical frame has more than one effective vote")
        track_totals[(recording, track)] += value
    maximum = max(track_totals.values(), default=0.0)
    if maximum > TRACK_EFFECTIVE_FRAME_CAP + 1e-4:
        raise RuntimeError("Per-track effective frame cap was exceeded")
    return {
        "unique_valid_physical_frames": len(partitions),
        "unique_tracks_with_supervision": len(track_totals),
        "maximum_track_effective_frames": maximum,
        "cross_partition_physical_frames": 0,
        "cross_partition_tracks": 0,
    }


def validate_cached_frame_supervision(data: dict[str, np.ndarray]) -> dict[str, float | int]:
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise RuntimeError(f"Cached-frame metadata keys are missing: {sorted(missing)}")
    n = len(data["sample_ids"])
    if n != EXPECTED_PRIMARY_ROWS:
        raise RuntimeError("Cached-frame metadata row count changed")
    for name in ("labels", "scenarios", "folds", "recordings", "tracks", "center_frames"):
        if data[name].shape != (n,):
            raise RuntimeError(f"Cached-frame vector shape changed: {name}")
    for stream in STREAMS:
        for suffix in ("source_frames", "valid", "frame_labels"):
            if data[f"{stream}_{suffix}"].shape != (n, FRAME_COUNT):
                raise RuntimeError(f"Cached-frame matrix shape changed: {stream}_{suffix}")
        if data[f"{stream}_valid"].dtype != np.bool_:
            raise RuntimeError("Cached-frame valid masks must be Boolean")
        if not np.array_equal(
            data[f"{stream}_source_frames"][:, CENTER_INDEX], data["center_frames"]
        ):
            raise RuntimeError("Cached-frame center index changed")
        if not np.array_equal(
            data[f"{stream}_frame_labels"][:, CENTER_INDEX], data["labels"]
        ):
            raise RuntimeError("Cached-frame center targets changed")
        if stream == "short" and not data[f"{stream}_valid"][:, CENTER_INDEX].all():
            raise RuntimeError("Cached-frame center masks changed")
        labels = data[f"{stream}_frame_labels"]
        if np.any(labels < -1) or np.any(labels > 2):
            raise RuntimeError("Cached-frame labels must use the fixed three-class order")
        if np.any(labels[data[f"{stream}_valid"]] < 0):
            raise RuntimeError("Valid cached frames must have known labels")
    weights = data["physical_weight"]
    valid = np.stack((data["short_valid"], data["long_valid"]))
    if (
        weights.shape != (len(STREAMS), n, FRAME_COUNT)
        or weights.dtype != np.float32
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or np.any(weights[~valid] != 0)
        or np.any(weights[valid] <= 0)
    ):
        raise RuntimeError("Cached-frame physical weights are invalid")
    return _validate_partition_and_weights(data)


def load_cached_frame_supervision(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a portable, pickle-free metadata artifact."""
    path = Path(path)
    if path.is_dir():
        path = path / "frame_supervision.npz"
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name].copy() for name in source.files}
    validate_cached_frame_supervision(data)
    return data


def validate_cached_frame_supervision_receipt(
    root: Path, directory: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bind a completed metadata artifact to all materialization evidence."""

    root, directory = Path(root).resolve(), Path(directory).resolve()
    receipt_path, artifact_path = directory / "receipt.json", directory / "frame_supervision.npz"
    if not receipt_path.is_file() or not artifact_path.is_file():
        raise RuntimeError("Cached-frame metadata receipt or artifact is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "OKUTAMA_CACHED_FRAME_SUPERVISION_METADATA_COMPLETE"
        or receipt.get("rows") != EXPECTED_PRIMARY_ROWS
        or receipt.get("center_index") != CENTER_INDEX
        or receipt.get("streams") != list(STREAMS)
        or receipt.get("class_order") != ["sitting", "standing", "walking_running"]
        or receipt.get("unknown_label") != -1
    ):
        raise RuntimeError("Cached-frame metadata receipt contract changed")
    artifact = receipt.get("artifact", {})
    if (
        Path(artifact.get("path", "")).resolve() != artifact_path
        or artifact.get("size_bytes") != artifact_path.stat().st_size
        or artifact.get("sha256") != sha256_file(artifact_path)
    ):
        raise RuntimeError("Cached-frame metadata artifact is not bound by its receipt")
    source_receipts = receipt.get("source_receipts", {})
    if set(source_receipts) != set(SOURCE_PATHS):
        raise RuntimeError("Cached-frame source receipt inventory changed")
    for name in SOURCE_PATHS:
        path = (root / SOURCE_PATHS[name]).resolve()
        record = source_receipts[name]
        if (
            record.get("path") != SOURCE_PATHS[name]
            or record.get("sha256") != SOURCE_SHA256[name]
            or not path.is_file()
            or record.get("size_bytes") != path.stat().st_size
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Cached-frame source receipt changed: {name}")
    expected_code = {
        "protocol": root / "experiments/okutama_cached_frame_supervision_protocol.json",
        "module": Path(__file__).resolve(),
        "builder": root / "experiments/build_okutama_cached_frame_supervision.py",
        "annotation_targets": root / "src/hac/actor_memory_data.py",
        "annotation_reader": root / "src/hac/okutama_native_video.py",
    }
    if set(receipt.get("source_code", {})) != set(expected_code):
        raise RuntimeError("Cached-frame materialization code inventory changed")
    for name, path in expected_code.items():
        record = receipt["source_code"][name]
        if (
            Path(record.get("path", "")).resolve() != path.resolve()
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"Cached-frame materialization code changed: {name}")
    annotation = receipt.get("annotation_receipt", {})
    archive = annotation.get("archive", {})
    if (
        archive.get("sha256") != EXPECTED_ARCHIVE_SHA256
        or archive.get("size_bytes") != EXPECTED_ARCHIVE_BYTES
        or len(annotation.get("annotation_members", ())) != 42
        or len(annotation.get("annotation_member_sha256", {})) != 42
        or annotation.get("annotation_payload_reads") != 42
    ):
        raise RuntimeError("Cached-frame annotation provenance changed")
    data = load_cached_frame_supervision(artifact_path)
    observed_weight_audit = validate_cached_frame_supervision(data)
    expected_weight_audit = receipt.get("weight_audit", {})
    if set(observed_weight_audit) != set(expected_weight_audit):
        raise RuntimeError("Cached-frame weight-audit fields changed")
    for name, value in observed_weight_audit.items():
        expected = expected_weight_audit[name]
        if isinstance(value, float):
            if not np.isclose(value, expected, atol=1e-7, rtol=0):
                raise RuntimeError(f"Cached-frame weight audit changed: {name}")
        elif value != expected:
            raise RuntimeError(f"Cached-frame weight audit changed: {name}")
    target_counts = {}
    for stream in STREAMS:
        valid = data[f"{stream}_valid"]
        labels = data[f"{stream}_frame_labels"]
        target_counts[stream] = {
            "missing_lost_unsupported_or_multibase": int((labels < 0).sum()),
            "valid": int(valid.sum()),
        }
        known_invalid = int(((labels >= 0) & ~valid).sum())
        if known_invalid:
            target_counts[stream]["known_target_but_cached_input_invalid"] = known_invalid
    if target_counts != receipt.get("target_counts"):
        raise RuntimeError("Cached-frame target counts do not replay from the artifact")
    accounting = receipt.get("access_accounting", {})
    expected_zero = (
        "image_payloads_read",
        "image_payloads_decoded",
        "cached_feature_values_read",
        "backbone_forward_passes",
        "model_fits",
        "protected_rows_read",
    )
    if any(accounting.get(name) != 0 for name in expected_zero) or accounting.get(
        "permitted_annotation_members_read"
    ) != 42:
        raise RuntimeError("Cached-frame access accounting changed")
    return data, receipt


def materialize_cached_frame_supervision(
    root: Path,
    output: Path,
    archive_path: Path,
) -> dict[str, Any]:
    """Build compact metadata for FSAR without touching cached feature values."""
    root, output, archive_path = root.resolve(), output.resolve(), archive_path.resolve()
    if output.exists():
        if (output / "receipt.json").is_file() and (output / "frame_supervision.npz").is_file():
            _, receipt = validate_cached_frame_supervision_receipt(root, output)
            return receipt
        if any(output.iterdir()):
            raise RuntimeError("Refusing to overwrite a partial cached-frame metadata directory")
        output.rmdir()
    paths: dict[str, Path] = {}
    source_receipts: dict[str, dict[str, Any]] = {}
    for name in SOURCE_PATHS:
        paths[name], source_receipts[name] = _checked_source(root, name)
    primary = _read_primary(paths["eligible_index"])
    frames: dict[str, np.ndarray] = {}
    manifest_valid: dict[str, np.ndarray] = {}
    cache_valid: dict[str, np.ndarray] = {}
    for stream in STREAMS:
        frames[stream], manifest_valid[stream] = read_ordered_frame_manifest(
            paths[f"{stream}_manifest"], primary
        )
        cache_valid[stream] = _read_cache_contract(
            paths[f"{stream}_feature"],
            paths[f"{stream}_summary"],
            paths[f"{stream}_validity"],
            paths[f"{stream}_completed"],
            primary["sample_ids"],
        )
    stable, annotation_receipt = _read_permitted_annotations(archive_path)
    data = dict(primary)
    target_counts: dict[str, dict[str, int]] = {}
    for stream in STREAMS:
        input_valid = manifest_valid[stream] & cache_valid[stream][:, None]
        labels, valid, target_counts[stream] = frame_targets(
            primary, frames[stream], input_valid, stable
        )
        data[f"{stream}_source_frames"] = frames[stream]
        data[f"{stream}_valid"] = valid
        data[f"{stream}_frame_labels"] = labels
    data["physical_weight"] = physical_multiplicity_weights(
        data["recordings"],
        data["tracks"],
        np.stack((frames["short"], frames["long"])),
        np.stack((data["short_valid"], data["long_valid"])),
    )
    weight_audit = validate_cached_frame_supervision(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise RuntimeError("A stale cached-frame metadata staging directory exists")
    staging.mkdir()
    artifact_path = staging / "frame_supervision.npz"
    np.savez_compressed(artifact_path, **data)
    protocol_path = root / "experiments/okutama_cached_frame_supervision_protocol.json"
    module_path = Path(__file__).resolve()
    builder_path = root / "experiments/build_okutama_cached_frame_supervision.py"
    receipt = {
        "status": "OKUTAMA_CACHED_FRAME_SUPERVISION_METADATA_COMPLETE",
        "rows": len(data["sample_ids"]),
        "streams": list(STREAMS),
        "center_index": CENTER_INDEX,
        "class_order": ["sitting", "standing", "walking_running"],
        "unknown_label": -1,
        "physical_identity": ["provider_recording_id", "provider_track_id", "source_frame"],
        "physical_multiplicity_scope": "joint_short_and_long_occurrences",
        "per_track_effective_frame_cap": TRACK_EFFECTIVE_FRAME_CAP,
        "feature_paths": {
            stream: SOURCE_PATHS[f"{stream}_feature"] for stream in STREAMS
        },
        "source_receipts": source_receipts,
        "annotation_receipt": annotation_receipt,
        "target_counts": target_counts,
        "weight_audit": weight_audit,
        "artifact": {
            **_receipt(artifact_path),
            "path": str((output / "frame_supervision.npz").resolve()),
        },
        "source_code": {
            "protocol": _receipt(protocol_path),
            "module": _receipt(module_path),
            "builder": _receipt(builder_path),
            "annotation_targets": _receipt(root / "src/hac/actor_memory_data.py"),
            "annotation_reader": _receipt(root / "src/hac/okutama_native_video.py"),
        },
        "access_accounting": {
            "image_payloads_read": 0,
            "image_payloads_decoded": 0,
            "cached_feature_values_read": 0,
            "backbone_forward_passes": 0,
            "model_fits": 0,
            "protected_rows_read": 0,
            "permitted_annotation_members_read": 42,
        },
    }
    (staging / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    validate_cached_frame_supervision_receipt(root, output)
    return receipt
