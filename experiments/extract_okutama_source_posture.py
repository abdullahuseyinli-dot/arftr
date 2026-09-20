"""Shardable label-blind source/posture feature extraction.

The default range is one center for bounded startup verification. ``--all`` is
the prepared long extraction and is intentionally not launched automatically.
No task label, ARFTR probability, support diagnostic or review field is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.absolute_body_evidence import (
    FEATURE_DIM,
    FrozenAbsoluteDinoCLS,
    compose_source_images,
    prepare_crops,
    prepare_source_crops,
    preprocess_rgb,
    source_actor_boxes,
)
from hac.body_witness_data import (
    CenterObservation,
    decode_exact_center,
    load_verified_source_map,
    resolve_center_source,
)
from hac.center_completion_data import AllowlistedMember, decode_allowlisted_jpeg
from hac.image_encoders import load_dinov2_encoder
from hac.okutama_native_video import CENTER_SLOT, frame_member, parse_sample_id

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_source_posture_protocol.json"
COMPLETION_PROTOCOL = ROOT / "experiments/okutama_center_evidence_completion_protocol.json"
METADATA = ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz"
FRAME_MANIFEST = ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/frame_manifest.csv"
IMAGE_ALLOWLIST = ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/image_allowlist.csv"
OLD_EXTRACTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/extraction_receipt.json"
DINO_SUMMARY = ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/source_posture_cache_v1_startup"
ABSOLUTE_MODULE = ROOT / "src/hac/absolute_body_evidence.py"
CROP_KEYS = (
    "J_whole", "J_upper", "J_lower", "R_whole", "R_upper", "R_lower",
    "N_whole", "N_upper", "N_lower", "R_whole_1p0", "R_whole_1p5",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed during hashing: {path}")
    return digest.hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_npz_new(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_observations() -> list[CenterObservation]:
    # Read only identity/split arrays, never the labels/features in this NPZ.
    with np.load(METADATA, allow_pickle=False) as saved:
        sample_ids, scenarios, folds = (saved[name] for name in ("sample_ids", "scenarios", "folds"))
    centers = {
        row["sample_id"]: row
        for row in read_csv(FRAME_MANIFEST)
        if int(row["time_index"]) == CENTER_SLOT
    }
    if len(sample_ids) != 4977 or set(sample_ids.tolist()) != set(centers):
        raise RuntimeError("Canonical center and extraction populations differ")
    result = []
    for sample_id, scenario, fold in zip(sample_ids, scenarios, folds, strict=True):
        row = centers[str(sample_id)]
        recording, track, center = parse_sample_id(str(sample_id))
        if (
            row["provider_recording_id"] != recording
            or row["provider_track_id"] != track
            or int(row["source_frame"]) != center
            or row["image_member"] != frame_member(recording, center)
            or row["valid_geometry"] != "1"
            or row["image_present"] != "1"
        ):
            raise RuntimeError("Center source identity/availability changed")
        box = tuple(float(row[name]) for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
        result.append(
            CenterObservation(
                str(sample_id), recording, track, str(scenario), int(fold), center,
                row["image_member"], box
            )
        )
    return result


def allowlist() -> dict[str, AllowlistedMember]:
    return {
        row["image_member"]: AllowlistedMember(
            row["image_member"], int(row["crc32"]), int(row["size_bytes"])
        )
        for row in read_csv(IMAGE_ALLOWLIST)
    }


def lock_value(*, shard_size: int) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETION_PROTOCOL.read_text(encoding="utf-8"))
    archive = completion["pinned_evidence"]["provider_frame_archive"]
    authority = ROOT / protocol["design_authority"]["path"]
    if sha256(authority) != protocol["design_authority"]["sha256"]:
        raise RuntimeError("Design authority changed")
    if sha256(METADATA) != protocol["population"]["metadata_sha256"]:
        raise RuntimeError("Canonical population metadata changed")
    return {
        "status": "SOURCE_POSTURE_LABEL_BLIND_EXTRACTION_LOCK",
        "rows": 4977,
        "shard_size": shard_size,
        "crop_keys": list(CROP_KEYS),
        "features_per_crop": FEATURE_DIM,
        "protocol_sha256": sha256(PROTOCOL),
        "design_authority_sha256": sha256(authority),
        "metadata_sha256": sha256(METADATA),
        "frame_manifest_sha256": sha256(FRAME_MANIFEST),
        "image_allowlist_sha256": sha256(IMAGE_ALLOWLIST),
        "source_resolution_receipt_sha256": sha256(OLD_EXTRACTION),
        "dino_summary_sha256": sha256(DINO_SUMMARY),
        "absolute_body_evidence_sha256": sha256(ABSOLUTE_MODULE),
        "provider_archive": {
            "path": archive["path"], "bytes": archive["size_bytes"],
            "expected_sha256": archive["sha256"], "full_hash_per_shard": False,
        },
        "labels_read": 0,
        "ARFTR_outputs_read": 0,
        "model_fits": 0,
    }


def establish_lock(output: Path, *, shard_size: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    expected = lock_value(shard_size=shard_size)
    path = output / "execution_lock.json"
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError("Existing extraction lock differs")
    else:
        write_json_new(path, expected)
    return expected


@torch.inference_mode()
def encode_tensors(
    encoder: FrozenAbsoluteDinoCLS, tensors: list[torch.Tensor], device: str
) -> np.ndarray:
    if not tensors:
        return np.empty((0, FEATURE_DIM), dtype=np.float16)
    values = []
    for start in range(0, len(tensors), 8):
        pixels = torch.stack(tensors[start : start + 8]).to(device, non_blocking=True)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = encoder(pixels)
        else:
            hidden = encoder(pixels)
        values.append(hidden.float().cpu().numpy())
    result = np.concatenate(values).astype(np.float16)
    if result.shape != (len(tensors), FEATURE_DIM) or not np.isfinite(result).all():
        raise RuntimeError("Source-posture row features are malformed")
    return result


def extract_shard(
    output: Path,
    observations: list[CenterObservation],
    start: int,
    count: int,
    encoder: FrozenAbsoluteDinoCLS,
    device: str,
    archive: zipfile.ZipFile,
    allowed: dict[str, AllowlistedMember],
    source_map: dict,
    video_receipts: dict,
) -> dict[str, Any]:
    stop = min(len(observations), start + count)
    if not 0 <= start < stop <= len(observations) or count > 64:
        raise ValueError("One shard must contain1..64 canonical consecutive rows")
    name = f"shard-{start:04d}-{stop:04d}"
    npz_path, receipt_path = output / f"{name}.npz", output / f"{name}.json"
    if npz_path.exists() != receipt_path.exists():
        raise RuntimeError(f"Incomplete existing shard pair: {name}")
    if npz_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "SOURCE_POSTURE_SHARD_COMPLETE"
            or (existing.get("start"), existing.get("stop")) != (start, stop)
            or existing.get("npz_sha256") != sha256(npz_path)
        ):
            raise RuntimeError(f"Existing shard cannot be safely resumed: {name}")
        return {**existing, "reused_after_hash_verification": True}
    features = np.zeros((stop - start, len(CROP_KEYS), FEATURE_DIM), dtype=np.float16)
    available = np.zeros((stop - start, len(CROP_KEYS)), dtype=bool)
    receipts = []
    started = time.perf_counter()
    for local, observation in enumerate(observations[start:stop]):
        request = resolve_center_source(observation, source_map)
        native, native_audit = decode_exact_center(
            request, verified_video=video_receipts[observation.recording]
        )
        jpeg, jpeg_audit = decode_allowlisted_jpeg(
            archive,
            {"valid_frame": "1", "image_width": "1280", "image_height": "720", "image_member": observation.image_member},
            allowed,
        )
        jpeg_rgb = np.asarray(jpeg, dtype=np.uint8)
        boxes = source_actor_boxes(observation.box_720, request.video.source_size)
        crops, crop_receipts = prepare_source_crops(
            jpeg_rgb,
            boxes["J"],
            source_id="J",
        )
        row = {
            "sample_id": observation.sample_id,
            "native": native_audit,
            "supplied": jpeg_audit,
        }
        if native is not None:
            native_crops, native_receipts = prepare_crops(
                compose_source_images(jpeg_rgb, native),
                boxes,
            )
            for key, crop in native_crops.items():
                if key.startswith(("R_", "N_")):
                    crops[key] = crop
                    crop_receipts[key] = native_receipts[key]
        present_keys = [key for key in CROP_KEYS if key in crops]
        present_indices = [CROP_KEYS.index(key) for key in present_keys]
        row_features = encode_tensors(
            encoder,
            [preprocess_rgb(crops[key]) for key in present_keys],
            device,
        )
        features[local, present_indices] = row_features
        available[local, present_indices] = True
        row["crops"] = {key: crop_receipts[key] for key in present_keys}
        receipts.append(row)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    ids = np.asarray([item.sample_id for item in observations[start:stop]])
    save_npz_new(npz_path, sample_ids=ids, crop_keys=np.asarray(CROP_KEYS), features=features, available=available)
    result = {
        "status": "SOURCE_POSTURE_SHARD_COMPLETE",
        "start": start, "stop": stop, "rows": stop - start,
        "sample_ids_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
        "npz_sha256": sha256(npz_path),
        "available_rows": int(available.all(1).sum()),
        "available_views": int(available.sum()),
        "elapsed_seconds": elapsed,
        "seconds_per_row": elapsed / len(ids),
        "labels_read": 0, "ARFTR_outputs_read": 0, "model_fits": 0,
        "row_receipts": receipts,
    }
    write_json_new(receipt_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--shard-size", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.shard_size <= 64:
        raise ValueError("Shard size must be1..64")
    output = args.output_dir.resolve()
    establish_lock(output, shard_size=args.shard_size)
    observations = load_observations()
    source_map = load_verified_source_map(ROOT)
    video_receipts = json.loads(OLD_EXTRACTION.read_text(encoding="utf-8"))["video_receipts"]
    completion = json.loads(COMPLETION_PROTOCOL.read_text(encoding="utf-8"))
    archive_path = Path(completion["pinned_evidence"]["provider_frame_archive"]["path"])
    if archive_path.stat().st_size != completion["pinned_evidence"]["provider_frame_archive"]["size_bytes"]:
        raise RuntimeError("Provider archive stat changed")
    dino = json.loads(DINO_SUMMARY.read_text(encoding="utf-8"))["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded, _ = load_dinov2_encoder(Path(dino["snapshot"]["path"]), device=device)
    encoder = FrozenAbsoluteDinoCLS(loaded.backbone).to(device).eval()
    del loaded
    results = []
    with zipfile.ZipFile(archive_path) as archive:
        if args.all:
            ranges = [(start, min(args.shard_size, len(observations) - start)) for start in range(0, len(observations), args.shard_size)]
        else:
            ranges = [(args.start, args.count)]
        fixed_allowlist = allowlist()
        for start, count in ranges:
            results.append(
                extract_shard(output, observations, start, count, encoder, device, archive, fixed_allowlist, source_map, video_receipts)
            )
    print(json.dumps({"status": "REQUESTED_SOURCE_POSTURE_SHARDS_COMPLETE", "shards": [{k: r[k] for k in ("start", "stop", "rows", "available_rows", "elapsed_seconds")} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
