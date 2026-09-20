"""Extract pooled, observation-only features for the aligned-evidence utility study.

The 16-center modes are the mandatory source/geometry/timing preflight.  ``shard``
materializes at most 128 consecutive full-cohort rows and performs no task fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.actor_memory_base import file_sha256
from hac.center_completion_data import (
    AllowlistedMember,
    decode_allowlisted_jpeg,
    prepare_neighbor,
)
from hac.correspondence_field_data import load_frozen_context
from hac.evidence_decomposition import (
    ARM_NAMES,
    actor_roi_bin_map,
    compose_arm_features,
    decompose_evidence,
)
from hac.image_encoders import load_dinov2_encoder, validate_dinov2_snapshot
from hac.okutama_native_video import (
    CENTER_SLOT,
    SAMPLE_PATTERN,
    SOURCE_OFFSETS,
    frame_member,
    valid_box,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_evidence_utility_protocol.json"
COUNCIL_PROTOCOL = (
    ROOT / ".runs/research_20260913/invention_council_20260913_135728/NEXT_PHASE_PROTOCOL.json"
)
PREFLIGHT_PLAN = ROOT / ".runs/research_20260913/center_evidence_completion_v1/plan_v8.json"
PILOT_SELECTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"
PILOT_CACHE = ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1"
FRAME_MANIFEST = (
    ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/frame_manifest.csv"
)
IMAGE_ALLOWLIST = (
    ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/image_allowlist.csv"
)
MATERIALIZATION_LOCK = (
    ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/materialization_lock.json"
)
DINO_SUMMARY = ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json"
DEFAULT_PREFLIGHT = ROOT / ".runs/research_20260913/evidence_utility_preflight_v1"
ARCHIVE_SHA256 = "c021ce8a12c84e083f359023ffd41c145561aaedb48b118e7c5416d5ddcecb73"
ARCHIVE_BYTES = 5_770_432_522
NEIGHBOR_TIME_INDICES = (0, 4, 12, 15)
OBSERVATION_COLUMNS = (
    "sample_id",
    "time_index",
    "provider_recording_id",
    "provider_track_id",
    "source_frame",
    "image_member",
    "image_width",
    "image_height",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "valid_geometry",
    "image_present",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cached-preflight", "fresh-preflight", "shard"),
        default="fresh-preflight",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=8)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _stable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype.str}|{list(value.shape)}".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _all_sample_ids() -> np.ndarray:
    summary = _read_json(DINO_SUMMARY)
    sample_ids = np.asarray(summary["sample_ids"], dtype=str)
    if sample_ids.shape != (4977,) or len(np.unique(sample_ids)) != 4977:
        raise RuntimeError("Frozen observation ID population changed")
    return sample_ids


def _selected_ids(mode: str, all_ids: np.ndarray, start: int, count: int) -> np.ndarray:
    if mode.endswith("preflight"):
        selected = np.asarray(
            _read_json(PREFLIGHT_PLAN)["preflight_selection"]["sample_ids"], dtype=str
        )
        if selected.shape != (16,) or not set(selected) <= set(all_ids):
            raise RuntimeError("Frozen preflight selection changed")
        return selected
    if start < 0 or count < 1 or count > 128 or start + count > len(all_ids):
        raise ValueError("A shard must be 1..128 consecutive full-cohort rows")
    return all_ids[start : start + count]


def _observation_rows(sample_ids: np.ndarray) -> dict[str, dict[int, dict[str, str]]]:
    selected = set(sample_ids.tolist())
    result: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with FRAME_MANIFEST.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)) or not set(OBSERVATION_COLUMNS) <= set(header):
            raise RuntimeError("Malformed observation manifest header")
        positions = {name: header.index(name) for name in OBSERVATION_COLUMNS}
        for values in reader:
            if len(values) != len(header):
                raise RuntimeError("Malformed observation manifest row")
            if values[positions["sample_id"]] not in selected:
                continue
            row = {name: values[index] for name, index in positions.items()}
            time_index = int(row["time_index"])
            if time_index in (CENTER_SLOT, *NEIGHBOR_TIME_INDICES):
                if time_index in result[row["sample_id"]]:
                    raise RuntimeError("Duplicate observation slot")
                result[row["sample_id"]][time_index] = row
    expected_slots = {CENTER_SLOT, *NEIGHBOR_TIME_INDICES}
    if set(result) != selected or any(set(rows) != expected_slots for rows in result.values()):
        raise RuntimeError("Observation manifest coverage changed")
    for sample_id in sample_ids:
        match = SAMPLE_PATTERN.fullmatch(str(sample_id))
        if match is None:
            raise RuntimeError("Sample identity grammar changed")
        recording, track, center = (
            match.group("recording"),
            match.group("track"),
            int(match.group("frame")),
        )
        for slot, row in result[str(sample_id)].items():
            expected_frame = center + SOURCE_OFFSETS[slot]
            box = None
            if row["valid_geometry"] == "1":
                try:
                    box = tuple(
                        float(row[name])
                        for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
                    )
                except ValueError as error:
                    raise RuntimeError(
                        f"Available geometry is malformed: {sample_id}/{slot}"
                    ) from error
            if (
                row["provider_recording_id"] != recording
                or row["provider_track_id"] != track
                or int(row["source_frame"]) != expected_frame
                or row["image_member"] != frame_member(recording, expected_frame)
                or (int(row["image_width"]), int(row["image_height"])) != (1280, 720)
                or row["valid_geometry"] not in {"0", "1"}
                or row["image_present"] not in {"0", "1"}
                or (row["valid_geometry"] == "1" and (box is None or not valid_box(box)))
            ):
                raise RuntimeError(f"Physical observation identity changed: {sample_id}/{slot}")
    return result


def _allowlist() -> dict[str, AllowlistedMember]:
    result = {}
    with IMAGE_ALLOWLIST.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result[row["image_member"]] = AllowlistedMember(
                row["image_member"], int(row["crc32"]), int(row["size_bytes"])
            )
    if not result:
        raise RuntimeError("Image allowlist is empty")
    return result


def _available(row: dict[str, str]) -> bool:
    if row["valid_geometry"] != "1" or row["image_present"] != "1":
        return False
    try:
        box = tuple(
            float(row[name]) for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
        )
    except ValueError:
        return False
    return valid_box(box)


def _sanitized(row: dict[str, str]) -> dict[str, str]:
    if not _available(row):
        raise ValueError("Cannot sanitize an unavailable physical observation")
    return {**row, "valid_frame": "1"}


@torch.inference_mode()
def _dense_forward(encoder, pixels: torch.Tensor) -> np.ndarray:
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 384, 384):
        raise ValueError("DINO input must be B,3,384,384")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = encoder.backbone(
            pixel_values=pixels.to("cuda", non_blocking=True), return_dict=True
        ).last_hidden_state
    if hidden.shape != (len(pixels), 730, 768) or not torch.isfinite(hidden).all():
        raise RuntimeError("Frozen dense DINO output changed")
    result = hidden[:, 1:].reshape(-1, 27, 27, 768).to(torch.float16).cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("Dense DINO output is nonfinite")
    return result


def _pilot_cache(sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summary = _read_json(PILOT_CACHE / "summary.json")
    if summary.get("status") != "CENTER_EVIDENCE_COMPLETION_FULL_CACHE_COMPLETE":
        raise RuntimeError("Dense pilot cache is not sealed complete")
    for name in ("teacher_tokens.npy", "neighbor_tokens.npy", "neighbor_valid.npy"):
        path = PILOT_CACHE / name
        expected = summary.get("artifacts", {}).get(name, {})
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("size_bytes")
            or file_sha256(path) != expected.get("sha256")
        ):
            raise RuntimeError(f"Audited dense pilot artifact changed: {name}")
    pilot_ids = np.asarray([row["sample_id"] for row in _read_json(PILOT_SELECTION)], dtype=str)
    index = {sample_id: row for row, sample_id in enumerate(pilot_ids)}
    if not set(sample_ids) <= set(pilot_ids):
        raise RuntimeError("Preflight row is absent from the audited dense pilot")
    rows = np.asarray([index[value] for value in sample_ids], dtype=np.int64)
    center = np.load(PILOT_CACHE / "teacher_tokens.npy", mmap_mode="r", allow_pickle=False)[rows]
    neighbors = np.load(PILOT_CACHE / "neighbor_tokens.npy", mmap_mode="r", allow_pickle=False)[
        rows
    ]
    valid = np.load(PILOT_CACHE / "neighbor_valid.npy", mmap_mode="r", allow_pickle=False)[rows]
    return np.asarray(center), np.asarray(neighbors), np.asarray(valid)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.relative_to(ROOT.resolve())
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Evidence output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    protocol = _read_json(PROTOCOL)
    if protocol.get("study_id") != "HAC_ALIGNED_EVIDENCE_ACTION_BRIDGE_V1":
        raise RuntimeError("Evidence protocol identity changed")
    all_ids = _all_sample_ids()
    selected = _selected_ids(args.mode, all_ids, args.start, args.count)
    rows = _observation_rows(selected)
    context_started = time.perf_counter()
    full_context, context_audit = load_frozen_context(ROOT, all_ids)
    context_seconds = time.perf_counter() - context_started
    all_index = {sample_id: row for row, sample_id in enumerate(all_ids.tolist())}
    selected_context = full_context[[all_index[value] for value in selected]]
    allowlist = _allowlist()
    completion_protocol = _read_json(
        ROOT / "experiments/okutama_center_evidence_completion_protocol.json"
    )
    archive_path = Path(completion_protocol["pinned_evidence"]["provider_frame_archive"]["path"])
    archive_started = time.perf_counter()
    if (
        archive_path.stat().st_size != ARCHIVE_BYTES
        or _stable_sha256(archive_path) != ARCHIVE_SHA256
    ):
        raise RuntimeError("Provider frame archive changed")
    archive_hash_seconds = time.perf_counter() - archive_started
    lock = _read_json(MATERIALIZATION_LOCK)
    model_root = Path(lock["inputs"]["dinov2_snapshot"]["path"])
    model_snapshot = validate_dinov2_snapshot(model_root)
    source_paths = [
        PROTOCOL,
        COUNCIL_PROTOCOL,
        Path(__file__),
        ROOT / "src/hac/actor_memory_base.py",
        ROOT / "src/hac/evidence_decomposition.py",
        ROOT / "src/hac/center_completion_data.py",
        ROOT / "src/hac/center_evidence_completion.py",
        ROOT / "src/hac/correspondence_field_data.py",
        ROOT / "src/hac/image_encoders.py",
        ROOT / "src/hac/okutama_native_video.py",
        ROOT / "experiments/okutama_center_evidence_completion_protocol.json",
        MATERIALIZATION_LOCK,
        FRAME_MANIFEST,
        IMAGE_ALLOWLIST,
        DINO_SUMMARY,
    ]
    if args.mode.endswith("preflight"):
        source_paths.extend(
            [
                PILOT_SELECTION,
                PILOT_CACHE / "summary.json",
                PILOT_CACHE / "teacher_tokens.npy",
                PILOT_CACHE / "neighbor_tokens.npy",
                PILOT_CACHE / "neighbor_valid.npy",
            ]
        )
    request = {
        "status": "EVIDENCE_UTILITY_OBSERVATION_REQUEST_LOCKED",
        "mode": args.mode,
        "sample_ids": selected.tolist(),
        "sample_ids_sha256": _array_sha256(selected),
        "rows": len(selected),
        "start": args.start if args.mode == "shard" else None,
        "microbatch": args.microbatch,
        "source_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in source_paths
        },
        "archive": {"path": str(archive_path), "bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256},
        "model_snapshot": model_snapshot,
        "task_label_fields_used": [],
        "ARFTR_outputs_used": [],
        "annotation_fields_used_for_acceptance": [],
    }
    _write_json(output / "request.json", request)

    cached_center = cached_neighbors = cached_neighbor_valid = None
    if args.mode.endswith("preflight"):
        cached_center, cached_neighbors, cached_neighbor_valid = _pilot_cache(selected)
    fresh = args.mode != "cached-preflight"
    encoder = model_receipt = None
    if fresh:
        if not torch.cuda.is_available():
            raise RuntimeError("Fresh dense extraction requires CUDA")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        encoder, model_receipt = load_dinov2_encoder(model_root, device="cuda")

    centers = np.zeros((len(selected), 6, 768), dtype=np.float32)
    contrasts = np.zeros((len(selected), 4, 6, 768), dtype=np.float32)
    coverage = np.zeros((len(selected), 6), dtype=np.float32)
    transforms = np.full((len(selected), 4, 2, 3), np.nan, dtype=np.float64)
    transform_valid = np.zeros((len(selected), 4), dtype=bool)
    matches = np.zeros((len(selected), 4), dtype=np.int64)
    residuals = np.full((len(selected), 4), np.inf, dtype=np.float64)
    common_counts = np.zeros((len(selected), 6), dtype=np.int64)
    roi_counts = np.zeros((len(selected), 6), dtype=np.int64)
    source_receipts = []
    prepare_seconds = encode_seconds = geometry_seconds = 0.0
    views_encoded = affine_fits_attempted = 0
    parity = []
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) != len({item.filename for item in archive.infolist()}):
            raise RuntimeError("Provider archive contains duplicate members")
        for output_row, sample_id in enumerate(selected):
            prepared = {}
            receipts = []
            begin = time.perf_counter()
            for slot in (CENTER_SLOT, *NEIGHBOR_TIME_INDICES):
                row = rows[str(sample_id)][slot]
                if not _available(row):
                    prepared[slot] = None
                    continue
                image, receipt = decode_allowlisted_jpeg(archive, _sanitized(row), allowlist)
                prepared[slot] = prepare_neighbor(image, row)
                receipts.append({"slot": slot, **receipt})
            prepare_seconds += time.perf_counter() - begin
            if prepared[CENTER_SLOT] is None:
                source_receipts.append(
                    {"sample_id": str(sample_id), "available_slots": [], "members": receipts}
                )
                continue
            input_slots = [
                slot for slot in (CENTER_SLOT, *NEIGHBOR_TIME_INDICES) if prepared[slot] is not None
            ]
            if fresh:
                pixels = torch.stack([prepared[slot].pixels for slot in input_slots])
                token_parts = []
                torch.cuda.synchronize()
                begin = time.perf_counter()
                for start in range(0, len(pixels), args.microbatch):
                    token_parts.append(
                        _dense_forward(encoder, pixels[start : start + args.microbatch])
                    )
                torch.cuda.synchronize()
                encode_seconds += time.perf_counter() - begin
                encoded = dict(zip(input_slots, np.concatenate(token_parts), strict=True))
                views_encoded += len(input_slots)
                center_tokens = encoded[CENTER_SLOT]
                neighbor_tokens = np.zeros((4, 27, 27, 768), dtype=np.float16)
                neighbor_source_valid = np.zeros((4, 27, 27), dtype=bool)
                for donor, slot in enumerate(NEIGHBOR_TIME_INDICES):
                    if slot in encoded:
                        neighbor_tokens[donor] = encoded[slot]
                        neighbor_source_valid[donor] = prepared[slot].valid_patch_mask
            else:
                center_tokens = cached_center[output_row]
                neighbor_tokens = cached_neighbors[output_row]
                neighbor_source_valid = cached_neighbor_valid[output_row]
            center_source_valid = prepared[CENTER_SLOT].valid_patch_mask
            if fresh and cached_center is not None:
                center_equal = np.array_equal(center_tokens, cached_center[output_row])
                neighbors_equal = np.array_equal(neighbor_tokens, cached_neighbors[output_row])
                valid_equal = np.array_equal(
                    neighbor_source_valid, cached_neighbor_valid[output_row]
                )
                parity.append(
                    {
                        "sample_id": str(sample_id),
                        "center_exact": center_equal,
                        "neighbors_exact": neighbors_equal,
                        "valid_exact": valid_equal,
                        "max_abs_center": float(
                            np.max(
                                np.abs(
                                    center_tokens.astype(np.float32)
                                    - cached_center[output_row].astype(np.float32)
                                )
                            )
                        ),
                        "max_abs_neighbors": float(
                            np.max(
                                np.abs(
                                    neighbor_tokens.astype(np.float32)
                                    - cached_neighbors[output_row].astype(np.float32)
                                )
                            )
                        ),
                    }
                )
                if not (center_equal and neighbors_equal and valid_equal):
                    raise RuntimeError("Fresh dense tokens differ from the audited cache")
            center_row = rows[str(sample_id)][CENTER_SLOT]
            actor_box = tuple(
                float(center_row[name])
                for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
            )
            bin_map = actor_roi_bin_map(
                prepared[CENTER_SLOT].raw_crop.geometry.image_to_crop,
                actor_box,
                center_source_valid,
            )
            begin = time.perf_counter()
            result = decompose_evidence(
                center_tokens, neighbor_tokens, center_source_valid, neighbor_source_valid, bin_map
            )
            affine_fits_attempted += len(NEIGHBOR_TIME_INDICES)
            geometry_seconds += time.perf_counter() - begin
            centers[output_row], contrasts[output_row], coverage[output_row] = (
                result.center_bins,
                result.contrasts,
                result.coverage,
            )
            transforms[output_row], transform_valid[output_row] = (
                result.transform_matrix,
                result.transform_valid,
            )
            matches[output_row], residuals[output_row] = (
                result.match_count,
                result.weighted_residual,
            )
            common_counts[output_row], roi_counts[output_row] = (
                result.common_target_count,
                result.roi_target_count,
            )
            source_receipts.append(
                {"sample_id": str(sample_id), "available_slots": input_slots, "members": receipts}
            )

    if not np.all(contrasts[:, 0] == 0) or not all(
        np.isfinite(value).all() for value in (centers, contrasts, coverage)
    ):
        raise RuntimeError("Pooled evidence violates its finite/B0 contract")
    arm_features = compose_arm_features(selected_context, centers, contrasts, coverage)
    feature_payload = {
        "sample_ids": selected,
        "arm_names": np.asarray(ARM_NAMES),
        "context": selected_context,
        "center_bins": centers,
        "contrasts": contrasts,
        "coverage": coverage,
        "transform_matrix": transforms,
        "transform_valid": transform_valid,
        "match_count": matches,
        "weighted_residual": residuals,
        "common_target_count": common_counts,
        "roi_target_count": roi_counts,
    }
    # The small preflight retains fully composed inputs for direct inspection.
    # Full shards keep the factored representation to avoid ~550 MB duplication.
    if args.mode.endswith("preflight"):
        feature_payload["arm_features"] = arm_features
    with (output / "features.npz").open("xb") as stream:
        np.savez_compressed(stream, **feature_payload)
    _write_json(output / "source_receipts.json", {"rows": source_receipts})
    if parity:
        _write_json(
            output / "fresh_cache_parity.json",
            {
                "rows": parity,
                "all_exact": all(
                    all(row[key] for key in ("center_exact", "neighbors_exact", "valid_exact"))
                    for row in parity
                ),
            },
        )
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": _stable_sha256(path)}
        for path in output.iterdir()
        if path.is_file()
    }
    summary = {
        "status": "EVIDENCE_UTILITY_PREFLIGHT_PASS"
        if args.mode.endswith("preflight")
        else "EVIDENCE_UTILITY_OBSERVATION_SHARD_COMPLETE",
        "mode": args.mode,
        "rows": len(selected),
        "views_encoded": views_encoded,
        "unmasked_affine_fits": affine_fits_attempted,
        "valid_affine_fits": int(transform_valid.sum()),
        "all_four_valid_rows": int(transform_valid.all(1).sum()),
        "coverage_mean": float(coverage.mean()),
        "reader_input_shape": list(arm_features.shape),
        "B0_contrast_exact_zero": bool(np.all(contrasts[:, 0] == 0)),
        "fresh_cache_parity_exact": None if not parity else True,
        "task_labels_read": 0,
        "ARFTR_outputs_read": 0,
        "annotation_support_fields_used": [],
        "timing_seconds": {
            "context_validation": context_seconds,
            "archive_hash": archive_hash_seconds,
            "jpeg_decode_and_prepare": prepare_seconds,
            "dense_dino": encode_seconds,
            "geometry_and_pooling": geometry_seconds,
        },
        "runtime_seconds_excluding_model_load": context_seconds
        + archive_hash_seconds
        + prepare_seconds
        + encode_seconds
        + geometry_seconds,
        "model": None
        if model_receipt is None
        else {
            "base_loader_receipt": model_receipt,
            "actual_cached_representation": "last_hidden_state[:,1:,:] reshaped row-major to 27x27x768",
            "actual_cached_dtype": "float16",
            "actual_matching_and_interpolation_dtype": "float32",
            "actual_bin_accumulation_dtype": "float64, cast to float32 output",
        },
        "device": None if not fresh else torch.cuda.get_device_name(),
        "peak_cuda_bytes": 0 if not fresh else int(torch.cuda.max_memory_allocated()),
        "context_audit": context_audit,
        "artifacts_before_summary": artifacts,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    _write_json(output / "summary.json", summary)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in (PROTOCOL, Path(__file__), ROOT / "src/hac/evidence_decomposition.py"):
        shutil.copyfile(path, snapshot / path.name)
    return summary


def main() -> None:
    args = parse_args()
    if args.microbatch < 1:
        raise ValueError("Microbatch must be positive")
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
