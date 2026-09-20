"""Observation-only center sources and raw masks for the body-witness pilot.

This module does not load pose models, extract features, or fit task models.
Training labels are deliberately separate from ``CenterObservation`` records.
The native source map reuses the pinned within-recording source audit, not the
old annotation-derived whole-window fallback. Failed exact centers are retained
as unavailable observations; no neighboring frame or alternate source is chosen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from hac.okutama_native_video import (
    CENTER_SLOT,
    EXPECTED_PRIMARY_ROWS,
    EXPECTED_RECORDINGS,
    EXPECTED_SCENARIOS,
    LABEL_INDEX,
    frame_member,
    load_primary_index,
    parse_sample_id,
    sha256_file,
)

ARFTR_PATH = ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
ARFTR_SHA256 = "ec9957a9393e6f1803d274d281549c20290ed59343b308c5be3446be37cec720"
MANIFEST_LOCK_PATH = ".runs/research_20260907/okutama_native_video_p3/extraction_lock.json"
MANIFEST_LOCK_SHA256 = "3ef99679f90f18e38b741177b3cc82aff3bd195a8b49ed86be7d3d5b34d8052a"
SOURCE_RECEIPTS = {
    "alignment": (
        ".runs/research_20260908/source_4k_alignment_validation_r1/summary.json",
        "f84184b6fe4aae1e55d7e4a091afd539ca4dbcbdabd919cc0572ee934ed8b654",
    ),
    "fidelity": (
        ".runs/research_20260908/source_4k_fidelity/summary.json",
        "7533b9625cca86fb457e9545695327aad2a64ef77da8246054af1e7aea217a12",
    ),
    "support": (
        ".runs/research_20260908/native4k_support_census/summary.json",
        "131426a48456a179d47b4a25326329c6fac6589443c1d5ee464f754aac7825f8",
    ),
}
DEFAULT_SOURCE_DIRECTORY = Path(
    r"C:\Users\DELL\hac_external_data\OkutamaAction\source4k\allowed_videos"
)
CROP_EXTENTS = (1.0, 1.25)
MASK_IDS = ("upper_visible", "lower_visible")
MASKED_SLOT_ORDER = tuple((extent, mask_id) for extent in CROP_EXTENTS for mask_id in MASK_IDS)
GAP_FRACTION = 0.05
PADDING_RGB = (124, 116, 104)  # Historical frozen RGB mean padding; never image-derived.
PILOT_HASH_PREFIX = "hac-body-witness-v1|"


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays are not auditable body-witness inputs")
    digest = hashlib.sha256(canonical_digest([array.dtype.str, list(array.shape)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _checked_file(path: Path, digest: str) -> Path:
    path = Path(path).resolve()
    if sha256_file(path) != digest:
        raise RuntimeError(f"Body-witness source hash changed: {path}")
    return path


def _repository_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Body-witness artifact reference escapes the repository")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass(frozen=True)
class CenterObservation:
    """Label-free identity/geometry metadata, not a vector of model predictors."""

    sample_id: str
    recording: str
    track: str
    scenario: str
    fold: int
    center_frame: int
    image_member: str
    box_720: tuple[float, float, float, float] | None

    def __post_init__(self) -> None:
        recording, track, frame = parse_sample_id(self.sample_id)
        scenario = ".".join(recording.split(".")[1:])
        if (
            recording not in EXPECTED_RECORDINGS
            or (self.recording, self.track, self.center_frame) != (recording, track, frame)
            or self.scenario != scenario
            or self.fold != int(EXPECTED_SCENARIOS[scenario][0].split("-")[1])
            or self.image_member != frame_member(recording, frame)
        ):
            raise ValueError("Center identity, scenario, fold, or exact frame member changed")

    def observation_metadata(self) -> dict[str, Any]:
        """An explicit field list prevents future diagnostic fields leaking out."""
        return {
            "sample_id": self.sample_id,
            "recording": self.recording,
            "track": self.track,
            "scenario": self.scenario,
            "fold": self.fold,
            "center_frame": self.center_frame,
            "image_member": self.image_member,
            "box_720": self.box_720,
        }


@dataclass(frozen=True)
class BodyWitnessCohort:
    observations: tuple[CenterObservation, ...]
    training_labels: np.ndarray
    arftr_probabilities: np.ndarray
    provenance: dict[str, Any]


def join_canonical_cohort(
    identities: Mapping[str, np.ndarray],
    primary_rows: Sequence[Mapping[str, Any]],
    center_rows: Sequence[Mapping[str, Any]],
    *,
    allowed_members: set[str],
) -> tuple[tuple[CenterObservation, ...], np.ndarray]:
    """Strict sample-ID join; no row-order, numeric-scenario, or validity join.

    Extra annotation fields in a manifest are ignored. This function consumes
    labels only to validate the separate canonical training-label array.
    """
    arrays = {
        name: np.asarray(identities[name])
        for name in ("sample_ids", "labels", "scenarios", "folds")
    }
    if any(value.shape != (EXPECTED_PRIMARY_ROWS,) for value in arrays.values()):
        raise ValueError("Body-witness cohort must contain exactly 4,977 canonical centers")
    if arrays["sample_ids"].dtype.kind != "U" or arrays["scenarios"].dtype.kind != "U":
        raise ValueError("Sample IDs and scenario IDs must be Unicode strings, not numbers")
    if arrays["labels"].dtype.kind not in "iu" or arrays["folds"].dtype.kind not in "iu":
        raise ValueError("Canonical labels and folds must be integer arrays")
    ids = arrays["sample_ids"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate canonical sample ID")
    primary = {row["sample_id"]: row for row in primary_rows}
    centers = {row["sample_id"]: row for row in center_rows}
    if len(primary) != len(primary_rows) or len(centers) != len(center_rows):
        raise ValueError("Duplicate primary or center-manifest sample ID")
    if set(primary) != set(ids) or set(centers) != set(ids):
        raise ValueError("Canonical/manifest sample-ID populations differ")
    observed = []
    for index, sample_id in enumerate(ids):
        recording, track, frame = parse_sample_id(sample_id)
        scenario = ".".join(recording.split(".")[1:])
        row, center = primary[sample_id], centers[sample_id]
        if (
            recording not in EXPECTED_RECORDINGS
            or scenario not in EXPECTED_SCENARIOS
            or arrays["scenarios"][index] != scenario
            or arrays["folds"][index] != int(EXPECTED_SCENARIOS[scenario][0].split("-")[1])
            or row["label"] not in LABEL_INDEX
            or int(row["label_index"]) != LABEL_INDEX[row["label"]]
            or arrays["labels"][index] != LABEL_INDEX[row["label"]]
            or center["provider_recording_id"] != recording
            or str(center["provider_track_id"]) != track
            or int(center["source_frame"]) != frame
            or int(center["time_index"]) != CENTER_SLOT
            or center["image_member"] != frame_member(recording, frame)
            or center["image_member"] not in allowed_members
            or (int(center["image_width"]), int(center["image_height"])) != (1280, 720)
        ):
            raise ValueError(f"Canonical center identity/class/source mismatch: {sample_id}")
        try:
            box = tuple(float(center[f"bbox_{axis}"]) for axis in ("xmin", "ymin", "xmax", "ymax"))
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                box = None
        except (KeyError, TypeError, ValueError):
            box = None
        observed.append(
            CenterObservation(
                sample_id,
                recording,
                track,
                scenario,
                int(arrays["folds"][index]),
                frame,
                center["image_member"],
                box,
            )
        )
    if Counter(row.scenario for row in observed) != {
        key: value[1] for key, value in EXPECTED_SCENARIOS.items()
    }:
        raise ValueError("Canonical scenario counts changed")
    if {row.recording for row in observed} != EXPECTED_RECORDINGS:
        raise ValueError("Canonical recording population changed")
    if not np.array_equal(np.bincount(arrays["labels"], minlength=3), [734, 2118, 2125]):
        raise ValueError("Canonical class counts changed")
    labels = arrays["labels"].copy()
    labels.setflags(write=False)
    return tuple(observed), labels


def load_body_witness_cohort(repository_root: Path) -> BodyWitnessCohort:
    """Load pinned identities and exact centers; reads no raw annotation/JPEG payloads."""
    root = Path(repository_root).resolve()
    lock_path = _checked_file(root / MANIFEST_LOCK_PATH, MANIFEST_LOCK_SHA256)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    paths = {}
    for name in ("frame_manifest", "image_allowlist"):
        receipt = lock["manifest"]["artifacts"][name]
        paths[name] = _checked_file(_repository_path(root, receipt["path"]), receipt["sha256"])
    primary = load_primary_index(_repository_path(root, lock["inputs"]["eligible_index"]["path"]))
    anchor_path = _checked_file(root / ARFTR_PATH, ARFTR_SHA256)
    with np.load(anchor_path, allow_pickle=False) as saved:
        identities = {name: saved[name] for name in ("sample_ids", "labels", "scenarios", "folds")}
        arm = np.flatnonzero(saved["arms"] == "r5_arftr_full")
        if len(arm) != 1:
            raise RuntimeError("Retained ARFTR arm missing or duplicated")
        probabilities = saved["mean_probabilities"][int(arm[0])].copy()
    centers = [
        row for row in _read_csv(paths["frame_manifest"]) if int(row["time_index"]) == CENTER_SLOT
    ]
    allowed = {row["image_member"] for row in _read_csv(paths["image_allowlist"])}
    observations, labels = join_canonical_cohort(
        identities, primary, centers, allowed_members=allowed
    )
    if (
        probabilities.shape != (EXPECTED_PRIMARY_ROWS, 3)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise RuntimeError("Malformed retained ARFTR probabilities")
    probabilities.setflags(write=False)
    return BodyWitnessCohort(
        observations,
        labels,
        probabilities,
        {
            "arftr_sha256": ARFTR_SHA256,
            "manifest_lock_sha256": MANIFEST_LOCK_SHA256,
            "sample_ids_sha256": canonical_digest([row.sample_id for row in observations]),
            "observation_metadata_sha256": canonical_digest(
                [row.observation_metadata() for row in observations]
            ),
            "image_payloads_read": 0,
            "annotation_payloads_read": 0,
            "annotation_flags_used_for_acquisition": False,
        },
    )


def select_measurement_pilot(
    observations: Sequence[CenterObservation],
) -> tuple[CenterObservation, ...]:
    """Exactly 128 label-blind centers: scenario quota, hash rank, track-first pass."""
    if len({row.sample_id for row in observations}) != len(observations):
        raise ValueError("Pilot population contains duplicate sample IDs")
    scenarios = sorted(EXPECTED_SCENARIOS)
    if {row.scenario for row in observations} != set(scenarios):
        raise ValueError("Pilot requires all eleven canonical scenarios")
    selected = []
    for scenario_index, scenario in enumerate(scenarios):
        quota = 12 if scenario_index < 7 else 11
        ranked = sorted(
            (row for row in observations if row.scenario == scenario),
            key=lambda row: (
                hashlib.sha256((PILOT_HASH_PREFIX + row.sample_id).encode()).digest(),
                row.sample_id,
            ),
        )
        chosen, used_tracks = [], set()
        for row in ranked:
            track_key = (row.recording, row.track)
            if track_key not in used_tracks:
                chosen.append(row)
                used_tracks.add(track_key)
                if len(chosen) == quota:
                    break
        chosen_ids = {row.sample_id for row in chosen}
        for row in ranked:
            if len(chosen) == quota:
                break
            if row.sample_id not in chosen_ids:
                chosen.append(row)
        if len(chosen) != quota:
            raise ValueError(f"Insufficient centers for the fixed scenario quota: {scenario}")
        selected.extend(chosen)
    return tuple(selected)


@dataclass(frozen=True)
class NativeVideoSource:
    recording: str
    path: Path
    sha256: str
    size_bytes: int
    source_size: tuple[int, int]
    fps: Fraction
    time_base: Fraction
    offset: int
    minimum_index: int
    maximum_index: int
    missing_indices: tuple[int, ...]
    map_sha256: str


def load_verified_source_map(
    repository_root: Path, *, source_directory: Path = DEFAULT_SOURCE_DIRECTORY
) -> dict[str, NativeVideoSource]:
    """Bind pinned map/packet receipts, without hashing/decoding 21 large videos yet."""
    root, directory = Path(repository_root).resolve(), Path(source_directory).resolve()
    metadata = {}
    for name, (relative, digest) in SOURCE_RECEIPTS.items():
        path = _checked_file(root / relative, digest)
        metadata[name] = json.loads(path.read_text(encoding="utf-8"))
    alignment, fidelity, support = (metadata[name] for name in ("alignment", "fidelity", "support"))
    if (
        alignment.get("all_recordings_validated") is not True
        or alignment["fidelity_summary_sha256"] != SOURCE_RECEIPTS["fidelity"][1]
        or support["alignment_sha256"] != SOURCE_RECEIPTS["alignment"][1]
        or support["fidelity_sha256"] != SOURCE_RECEIPTS["fidelity"][1]
    ):
        raise RuntimeError("Source map/fidelity/support ancestry mismatch")
    videos = {row["recording"]: row for row in fidelity["videos"]}
    packets = {row["recording"]: row for row in support["videos"]}
    mappings = {row["recording"]: row for row in alignment["mappings"]}
    if any(set(group) != EXPECTED_RECORDINGS for group in (videos, packets, mappings)):
        raise RuntimeError("Native source map does not cover exactly the permitted recordings")
    result = {}
    for recording in sorted(EXPECTED_RECORDINGS):
        video, packet, mapping = videos[recording], packets[recording], mappings[recording]
        model = mapping["selected_model"]
        bases = {Fraction(row["pyav_time_base"]) for row in mapping["validation_checks"]}
        if (
            mapping.get("validated") is not True
            or model["scale"] != 1.0
            or int(model["offset"]) != model["offset"]
            or len(bases) != 1
            or packet["sha256"] != video["sha256"]
        ):
            raise RuntimeError("Unsupported or unvalidated exact native time mapping")
        path = directory / Path(video["zip_member"]).name
        if path.stem != recording or path.suffix.lower() not in {".mp4", ".mov"}:
            raise RuntimeError("Native video reference is outside its recording")
        default = packet["default"]
        result[recording] = NativeVideoSource(
            recording,
            path,
            video["sha256"],
            int(video["bytes"]),
            tuple(video["size"]),
            Fraction(str(video["fps"])).limit_denominator(1_000_000),
            bases.pop(),
            int(model["offset"]),
            max(0, int(default["min_index"])),
            int(default["max_index"]),
            tuple(int(value) for value in default["interior_missing_indices"]),
            SOURCE_RECEIPTS["alignment"][1],
        )
    return result


@dataclass(frozen=True)
class CenterSourceRequest:
    observation: CenterObservation
    video: NativeVideoSource
    native_index: int
    target_pts: int
    nominal_seconds: float
    native_seconds: float
    native_box: tuple[float, float, float, float] | None
    available: bool
    unavailable_reason: str | None


def resolve_center_source(
    observation: CenterObservation, source_map: Mapping[str, NativeVideoSource]
) -> CenterSourceRequest:
    source = source_map[observation.recording]
    if source.recording != observation.recording or source.fps <= 0 or source.time_base <= 0:
        raise ValueError("Source recording/timebase mismatch")
    native_index = observation.center_frame + source.offset
    pts = Fraction(native_index) / source.fps / source.time_base
    reason = None
    if pts.denominator != 1:
        reason = "nonintegral_exact_timestamp"
    elif (
        native_index < source.minimum_index
        or native_index > source.maximum_index
        or native_index in source.missing_indices
    ):
        reason = "exact_native_center_absent"
    box = None
    if observation.box_720 is None:
        reason = reason or "invalid_supplied_center_geometry"
    else:
        sx, sy = source.source_size[0] / 1280, source.source_size[1] / 720
        box = tuple(
            value * scale
            for value, scale in zip(observation.box_720, (sx, sy, sx, sy), strict=True)
        )
        if not _valid_box(box, source.source_size):
            reason = reason or "invalid_supplied_center_geometry"
    return CenterSourceRequest(
        observation,
        source,
        native_index,
        int(pts) if pts.denominator == 1 else -1,
        observation.center_frame / 30.0,
        float(Fraction(native_index) / source.fps),
        box,
        reason is None,
        reason,
    )


def verify_video_file(request: CenterSourceRequest) -> dict[str, Any]:
    """Hash once per recording; callers may reuse the stat-bound receipt for decoding."""
    source = request.video
    before = source.path.stat()
    path = _checked_file(source.path, source.sha256)
    info = path.stat()
    if info.st_size != source.size_bytes or (before.st_size, before.st_mtime_ns) != (
        info.st_size,
        info.st_mtime_ns,
    ):
        raise RuntimeError("Native video changed during verification")
    return {
        "path": str(path),
        "sha256": source.sha256,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def validate_decoded_center(
    request: CenterSourceRequest, rgb: np.ndarray, *, pts: int | None, time_base: Fraction
) -> dict[str, Any]:
    """Reject later/nearby frames: exact PTS equality, dimensions and uint8 RGB only."""
    image = np.asarray(rgb)
    valid = bool(
        request.available
        and pts == request.target_pts
        and Fraction(time_base) == request.video.time_base
        and image.dtype == np.uint8
        and image.shape == (request.video.source_size[1], request.video.source_size[0], 3)
    )
    return {
        "sample_id": request.observation.sample_id,
        "decode_valid": valid,
        "requested_native_index": request.native_index,
        "requested_pts": request.target_pts,
        "actual_pts": pts,
        "actual_time_base": str(time_base),
        "nominal_seconds": request.nominal_seconds,
        "native_seconds": request.native_seconds,
        "source_size": list(request.video.source_size),
        "source_sha256": request.video.sha256,
        "alignment_sha256": request.video.map_sha256,
        "image_sha256": array_digest(image) if valid else None,
        "fallback": "none" if valid else "retain_arftr_no_alternate_frame",
        "reason": None if valid else request.unavailable_reason or "exact_decode_contract_failed",
    }


def decode_exact_center(
    request: CenterSourceRequest,
    *,
    verified_video: Mapping[str, Any] | None = None,
    maximum_decoded_frames: int = 256,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Bounded lazy PyAV decode. This function is never invoked by a metadata loader."""
    if maximum_decoded_frames < 1:
        raise ValueError("A positive decode budget is required")
    if not request.available:
        return None, validate_decoded_center(
            request,
            np.empty((0, 0, 3), dtype=np.uint8),
            pts=None,
            time_base=request.video.time_base,
        )
    receipt = verify_video_file(request) if verified_video is None else verified_video
    info = request.video.path.stat()
    if (
        receipt.get("path") != str(request.video.path.resolve())
        or receipt.get("sha256") != request.video.sha256
        or receipt.get("size_bytes") != info.st_size
        or receipt.get("mtime_ns") != info.st_mtime_ns
    ):
        raise RuntimeError("Native video verification receipt no longer matches source bytes")
    import av

    image, actual_pts = None, None
    with av.open(str(request.video.path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if (
            Fraction(stream.time_base) != request.video.time_base
            or Fraction(stream.average_rate) != request.video.fps
        ):
            raise RuntimeError("Native decoder timebase/rate differs from the verified source map")
        container.seek(request.target_pts, stream=stream, backward=True)
        for count, frame in enumerate(container.decode(stream), 1):
            if frame.pts is not None and frame.pts >= request.target_pts:
                actual_pts = frame.pts
                if actual_pts == request.target_pts:
                    image = frame.to_ndarray(format="rgb24")
                break
            if count >= maximum_decoded_frames:
                break
    after = request.video.path.stat()
    if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Native video changed during exact decoding")
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    audit = validate_decoded_center(
        request,
        image if image is not None else empty,
        pts=actual_pts,
        time_base=request.video.time_base,
    )
    audit["video_verification"] = dict(receipt)
    return (image if audit["decode_valid"] else None), audit


def _valid_box(box: Sequence[float], source_size: tuple[int, int]) -> bool:
    if len(box) != 4 or not np.isfinite(box).all():
        return False
    x1, y1, x2, y2 = box
    width, height = source_size
    return bool(
        width > 0
        and height > 0
        and x2 > x1
        and y2 > y1
        and min(x2, width) > max(0, x1)
        and min(y2, height) > max(0, y1)
    )


@dataclass(frozen=True)
class BodyCropGeometry:
    native_box: tuple[float, float, float, float]
    crop_box: tuple[int, int, int, int]
    source_size: tuple[int, int]
    extent: float

    @property
    def raw_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.crop_box
        return right - left, bottom - top

    @property
    def image_to_raw(self) -> np.ndarray:
        left, top, _, _ = self.crop_box
        return np.asarray([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)

    def provenance(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "image_to_raw": self.image_to_raw.tolist(),
            "interpolation": "none_raw_integer_crop",
            "pixel_coordinates": "pixel_edges;mask_membership_uses_pixel_centers",
            "padding_rgb": PADDING_RGB,
        }


def make_body_crop_geometry(
    native_box: Sequence[float], *, extent: float, source_size: tuple[int, int]
) -> BodyCropGeometry:
    """Native floor/ceil crop, same-center extent, no encoder-specific resize.

    Like historical ``crop_geometry``, fractional boundaries round outward and
    out-of-image regions are padded. Unlike that 720-only helper this operates
    directly in native source coordinates and does not force a square output.
    """
    box = tuple(float(value) for value in native_box)
    if extent not in CROP_EXTENTS or not _valid_box(box, source_size):
        raise ValueError("Invalid native body box or unprescribed crop extent")
    x1, y1, x2, y2 = box
    margin_x, margin_y = (extent - 1) * (x2 - x1) / 2, (extent - 1) * (y2 - y1) / 2
    crop = (
        math.floor(x1 - margin_x),
        math.floor(y1 - margin_y),
        math.ceil(x2 + margin_x),
        math.ceil(y2 + margin_y),
    )
    return BodyCropGeometry(box, crop, tuple(source_size), float(extent))


def crop_raw_body(rgb: np.ndarray, geometry: BodyCropGeometry) -> np.ndarray:
    """Return native uint8 pixels with constant padding; never resize before masking."""
    image = np.asarray(rgb)
    width, height = geometry.source_size
    if image.dtype != np.uint8 or image.shape != (height, width, 3):
        raise ValueError("Decoded RGB does not match native crop geometry")
    left, top, right, bottom = geometry.crop_box
    raw_width, raw_height = geometry.raw_size
    output = np.empty((raw_height, raw_width, 3), dtype=np.uint8)
    output[:] = PADDING_RGB
    x1, y1, x2, y2 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    output[y1 - top : y2 - top, x1 - left : x2 - left] = image[y1:y2, x1:x2]
    return output


def normalized_crop_transform(geometry: BodyCropGeometry) -> np.ndarray:
    """Six finite geometry fields consumed by the conditional decoder."""

    left, top, _, _ = geometry.crop_box
    crop_width, crop_height = geometry.raw_size
    source_width, source_height = geometry.source_size
    x1, y1, x2, y2 = geometry.native_box
    result = np.asarray(
        [
            (x1 - left) / crop_width,
            (y1 - top) / crop_height,
            (x2 - left) / crop_width,
            (y2 - top) / crop_height,
            crop_width / source_width,
            crop_height / source_height,
        ],
        dtype=np.float32,
    )
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("Crop transform must be six finite normalized fields")
    return result


def reader_quality_features(
    geometries: Sequence[BodyCropGeometry], predictor_validity: np.ndarray
) -> np.ndarray:
    """Fixed 36D geometry/availability vector shared by V1, V2 and V3."""

    if len(geometries) != len(CROP_EXTENTS):
        raise ValueError("Reader quality requires both prescribed crop extents")
    by_extent = {geometry.extent: geometry for geometry in geometries}
    if set(by_extent) != set(CROP_EXTENTS):
        raise ValueError("Reader quality crop extents differ from the fixed contract")
    reference = by_extent[CROP_EXTENTS[0]]
    if any(
        geometry.native_box != reference.native_box
        or geometry.source_size != reference.source_size
        for geometry in by_extent.values()
    ):
        raise ValueError("Reader quality geometries must describe one physical center")
    validity = np.asarray(predictor_validity)
    if validity.shape != (4,) or validity.dtype != np.bool_:
        raise ValueError("Predictor validity must be four fixed-slot booleans")

    source_width, source_height = reference.source_size
    x1, y1, x2, y2 = reference.native_box
    actor_area = (x2 - x1) * (y2 - y1)
    source_area = source_width * source_height
    inside_fractions = []
    for extent in CROP_EXTENTS:
        left, top, right, bottom = by_extent[extent].crop_box
        inside = max(0, min(source_width, right) - max(0, left)) * max(
            0, min(source_height, bottom) - max(0, top)
        )
        inside_fractions.append(inside / ((right - left) * (bottom - top)))
    base = np.asarray(
        [
            x1 / source_width,
            y1 / source_height,
            x2 / source_width,
            y2 / source_height,
            actor_area / source_area,
            np.log1p(actor_area) / np.log1p(source_area),
            *inside_fractions,
        ],
        dtype=np.float32,
    )
    transforms = np.stack(
        [normalized_crop_transform(by_extent[extent]) for extent, _ in MASKED_SLOT_ORDER]
    )
    result = np.concatenate((base, transforms.reshape(-1), validity.astype(np.float32)))
    if result.shape != (36,) or not np.isfinite(result).all():
        raise RuntimeError("Reader quality vector violates its fixed 36D contract")
    return result


def raw_body_masks(geometry: BodyCropGeometry) -> dict[str, np.ndarray]:
    """Spatial masks before interpolation; gap is excluded from both visible views.

    Each horizontal partition spans the whole crop width, including its fixed
    context margin. Dividing lines are defined by the unexpanded actor box.
    """
    width, height = geometry.raw_size
    top = geometry.crop_box[1]
    actor_top, actor_bottom = geometry.native_box[1], geometry.native_box[3]
    center, actor_height = (actor_top + actor_bottom) / 2, actor_bottom - actor_top
    y = top + np.arange(height, dtype=np.float64) + 0.5
    upper = np.broadcast_to(
        (y < center - GAP_FRACTION * actor_height / 2)[:, None], (height, width)
    ).copy()
    lower = np.broadcast_to(
        (y >= center + GAP_FRACTION * actor_height / 2)[:, None], (height, width)
    ).copy()
    return {"upper_visible": upper, "lower_visible": lower, "gap": ~(upper | lower)}


def apply_raw_body_mask(
    raw_rgb: np.ndarray, geometry: BodyCropGeometry, mask_id: str
) -> np.ndarray:
    """All hidden/gap pixels get a fixed fill, independent of their original values."""
    if mask_id not in MASK_IDS:
        raise ValueError("Unknown body-witness mask ID")
    raw = np.asarray(raw_rgb)
    if raw.dtype != np.uint8 or raw.shape != (geometry.raw_size[1], geometry.raw_size[0], 3):
        raise ValueError("Mask input must be the unresized native body crop")
    output = raw.copy()
    output[~raw_body_masks(geometry)[mask_id]] = PADDING_RGB
    return output


def corrupt_withheld_for_teacher(
    raw_rgb: np.ndarray, geometry: BodyCropGeometry, mask_id: str
) -> np.ndarray:
    """Target-side diagnostic only: replace the opposite partition by visible mean."""
    masked = apply_raw_body_mask(raw_rgb, geometry, mask_id)  # Validate the raw contract.
    masks = raw_body_masks(geometry)
    visible = masks[mask_id]
    hidden = masks[MASK_IDS[1 - MASK_IDS.index(mask_id)]]
    if not visible.any():
        raise ValueError("No visible pixels for the prescribed target-side corruption")
    result = np.asarray(raw_rgb).copy()
    result[hidden] = np.rint(masked[visible].mean(axis=0)).clip(0, 255).astype(np.uint8)
    return result


def predictor_input_valid(*, decode_succeeded: bool, geometry: BodyCropGeometry | None) -> bool:
    """Geometry/decode only: pose confidence and withheld pixels are not arguments."""
    if not isinstance(decode_succeeded, (bool, np.bool_)):
        raise ValueError("Decode availability must be an observed boolean")
    return bool(
        decode_succeeded
        and geometry is not None
        and geometry.extent in CROP_EXTENTS
        and _valid_box(geometry.native_box, geometry.source_size)
        and np.isfinite(normalized_crop_transform(geometry)).all()
    )


def common_acquisition_mask(
    crop_decode_valid: np.ndarray, pose_peak_magnitudes: np.ndarray
) -> np.ndarray:
    """Final candidate availability shared by V0-V3; NEVER a predictor validity bit.

    Shapes are [N,2] crops and [N,2,8] observed pose peaks. Each crop must have
    clipped total landmark weight >=2. No annotation or manual-review flag is
    accepted, and nonfinite observations fail availability instead of vanishing.
    """
    valid, peaks = np.asarray(crop_decode_valid), np.asarray(pose_peak_magnitudes)
    if (
        valid.dtype != np.bool_
        or valid.ndim != 2
        or valid.shape[1] != 2
        or peaks.shape != (*valid.shape, 8)
    ):
        raise ValueError("Malformed two-view acquisition/pose arrays")
    finite = np.isfinite(peaks).all(axis=-1)
    weights = np.clip(np.where(np.isfinite(peaks), peaks, 0), 0, 1).sum(axis=-1)
    return (valid & finite & (weights >= 2.0)).all(axis=1)


def predictor_payload(
    raw_rgb: np.ndarray, geometry: BodyCropGeometry, mask_id: str, *, decode_succeeded: bool
) -> dict[str, Any]:
    """Only decoder-permitted observations; not labels, IDs, reliability or context."""
    return {
        "masked_rgb": apply_raw_body_mask(raw_rgb, geometry, mask_id),
        "mask_id": mask_id,
        "crop_transform": normalized_crop_transform(geometry),
        "crop_extent": geometry.extent,
        "input_valid": predictor_input_valid(decode_succeeded=decode_succeeded, geometry=geometry),
    }
