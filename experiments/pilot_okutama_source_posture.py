"""Run or independently replay the16-center source/posture mechanics smoke.

This script reads no task label, ARFTR probability, annotation support, review or
error field.  It verifies exact source pixels, prepares crop-before-encoding
views and runs the pinned frozen DINO encoder only.  It never fits a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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
    DINO_INPUT_SIZE,
    FEATURE_DIM,
    FrozenAbsoluteDinoCLS,
    array_digest,
    compose_source_images,
    prepare_crops,
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
from hac.okutama_native_video import EXPECTED_SCENARIOS, frame_member

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_source_posture_protocol.json"
COMPLETION_PROTOCOL = ROOT / "experiments/okutama_center_evidence_completion_protocol.json"
PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"
PILOT_RECEIPT = ROOT / ".runs/research_20260913/body_witness_pilot_v1/selection_receipt.json"
CROP_MANIFEST = ROOT / ".runs/research_20260913/body_witness_pilot_v1/crop_manifest.json"
OLD_EXTRACTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/extraction_receipt.json"
IMAGE_ALLOWLIST = ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/image_allowlist.csv"
DINO_SUMMARY = ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/source_posture_preflight_v1"
CROP_KEYS = (
    "J_whole", "J_upper", "J_lower",
    "R_whole", "R_upper", "R_lower",
    "N_whole", "N_upper", "N_lower",
    "R_whole_1p0", "R_whole_1p5",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_new_npz(path: Path, **values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _observations() -> tuple[list[CenterObservation], list[dict[str, Any]], dict[str, Any]]:
    selection = json.loads(PILOT.read_text(encoding="utf-8"))
    receipt = json.loads(PILOT_RECEIPT.read_text(encoding="utf-8"))
    crops = json.loads(CROP_MANIFEST.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "LABEL_BLIND_SELECTION_COMPLETE"
        or receipt.get("training_labels_read_by_selection") != 0
        or receipt.get("arftr_outputs_read_by_selection") != 0
        or len(selection) != 128
        or len(crops) != 128
    ):
        raise RuntimeError("Existing pilot is not the sealed label-blind128 selection")
    by_id = {row["sample_id"]: row for row in crops}
    observations, records = [], []
    for selected in selection[:16]:
        sample_id = selected["sample_id"]
        row = by_id[sample_id]
        if row["selection_index"] != selected["selection_index"] or row["decode_valid"] is not True:
            raise RuntimeError("Pilot crop manifest identity or prior exact decode changed")
        native_box = tuple(float(x) for x in row["source_request"]["native_box"])
        box720 = tuple(x / 3 for x in native_box)
        scenario = selected["scenario"]
        observations.append(
            CenterObservation(
                sample_id=sample_id,
                recording=selected["recording"],
                track=selected["track"],
                scenario=scenario,
                fold=int(EXPECTED_SCENARIOS[scenario][0].split("-")[1]),
                center_frame=int(selected["center_frame"]),
                image_member=frame_member(selected["recording"], int(selected["center_frame"])),
                box_720=box720,
            )
        )
        records.append(row)
    return observations, records, receipt


def _allowlist() -> dict[str, AllowlistedMember]:
    result = {}
    for row in read_csv(IMAGE_ALLOWLIST):
        result[row["image_member"]] = AllowlistedMember(
            row["image_member"], int(row["crc32"]), int(row["size_bytes"])
        )
    return result


@torch.inference_mode()
def _encode(encoder: FrozenAbsoluteDinoCLS, tensors: list[torch.Tensor], device: str) -> tuple[np.ndarray, float]:
    output = []
    started = time.perf_counter()
    for start in range(0, len(tensors), 8):
        pixels = torch.stack(tensors[start : start + 8]).to(device, non_blocking=True)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = encoder(pixels)
        else:
            values = encoder(pixels)
        output.append(values.float().cpu().numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    values = np.concatenate(output).astype(np.float16)
    if values.shape != (len(tensors), FEATURE_DIM) or not np.isfinite(values).all():
        raise RuntimeError("Frozen source-posture features changed shape or became nonfinite")
    return values, elapsed


def execute(output: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    authority = ROOT / protocol["design_authority"]["path"]
    if sha256(authority) != protocol["design_authority"]["sha256"]:
        raise RuntimeError("Source-posture design authority changed")
    completion = json.loads(COMPLETION_PROTOCOL.read_text(encoding="utf-8"))
    archive_path = Path(completion["pinned_evidence"]["provider_frame_archive"]["path"])
    expected_archive = completion["pinned_evidence"]["provider_frame_archive"]
    if not archive_path.is_file() or archive_path.stat().st_size != expected_archive["size_bytes"]:
        raise RuntimeError("Provider JPEG archive stat changed")
    observations, historical_rows, selection_receipt = _observations()
    source_map = load_verified_source_map(ROOT)
    prior_video_receipts = json.loads(OLD_EXTRACTION.read_text(encoding="utf-8"))["video_receipts"]
    dino = json.loads(DINO_SUMMARY.read_text(encoding="utf-8"))["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded, model_receipt = load_dinov2_encoder(Path(dino["snapshot"]["path"]), device=device)
    encoder = FrozenAbsoluteDinoCLS(loaded.backbone).to(device).eval()
    del loaded
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    all_tensors: list[torch.Tensor] = []
    rows, source_audits = [], []
    allowlist = _allowlist()
    decode_seconds = 0.0
    preparation_seconds = 0.0
    started_all = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive:
        for observation, historical in zip(observations, historical_rows, strict=True):
            started = time.perf_counter()
            request = resolve_center_source(observation, source_map)
            previous_request = historical["source_request"]
            if (
                request.native_index != previous_request["native_index"]
                or request.target_pts != previous_request["target_pts"]
                or not np.array_equal(request.native_box, previous_request["native_box"])
            ):
                raise RuntimeError("Reconstructed native source request changed")
            native, native_audit = decode_exact_center(
                request, verified_video=prior_video_receipts[observation.recording]
            )
            jpeg_row = {
                "valid_frame": "1",
                "image_width": "1280",
                "image_height": "720",
                "image_member": observation.image_member,
            }
            jpeg, jpeg_audit = decode_allowlisted_jpeg(archive, jpeg_row, allowlist)
            supplied = np.asarray(jpeg, dtype=np.uint8).copy()
            decode_seconds += time.perf_counter() - started
            if native is None:
                raise RuntimeError("One of the sealed16 exact native centers is now unavailable")
            started = time.perf_counter()
            images = compose_source_images(supplied, native)
            boxes = source_actor_boxes(observation.box_720, request.video.source_size)
            crops, crop_receipts = prepare_crops(images, boxes)
            tensors = [preprocess_rgb(crops[key]) for key in CROP_KEYS]
            all_tensors.extend(tensors)
            preparation_seconds += time.perf_counter() - started
            scale = request.video.source_size[0] / 1280
            parity = {}
            for region in ("whole", "upper", "lower"):
                native_box = np.asarray(crop_receipts[f"N_{region}"]["continuous_box"])
                regen_box = np.asarray(crop_receipts[f"R_{region}"]["continuous_box"])
                parity[region] = bool(np.allclose(native_box, scale * regen_box, atol=1e-10, rtol=0))
            if not all(parity.values()):
                raise RuntimeError("Continuous native/regenerated physical crop parity failed")
            rows.append(
                {
                    "sample_id": observation.sample_id,
                    "selection_index": historical["selection_index"],
                    "source_request": previous_request,
                    "native_decode": native_audit,
                    "supplied_decode": jpeg_audit,
                    "continuous_coordinate_parity": parity,
                    "source_image_sha256": {key: array_digest(value) for key, value in images.items()},
                    "crops": {key: crop_receipts[key] for key in CROP_KEYS},
                }
            )
            source_audits.append(native_audit)
    features_flat, encode_seconds = _encode(encoder, all_tensors, device)
    features = features_flat.reshape(len(rows), len(CROP_KEYS), FEATURE_DIM)
    output.mkdir(parents=True, exist_ok=True)
    save_new_npz(
        output / "features.npz",
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        crop_keys=np.asarray(CROP_KEYS),
        features=features,
    )
    write_new_json(output / "crop_receipts.json", rows)
    total = time.perf_counter() - started_all
    summary = {
        "status": "SOURCE_POSTURE_16_MECHANICS_AND_RESOURCE_SMOKE_COMPLETE",
        "centers": len(rows),
        "views": len(all_tensors),
        "sources": 3,
        "unique_views_per_center": len(CROP_KEYS),
        "model_fits": 0,
        "task_labels_read": 0,
        "ARFTR_outputs_read": 0,
        "annotation_support_read": 0,
        "manual_review_fields_read": 0,
        "selection_receipt_sha256": sha256(PILOT_RECEIPT),
        "selection_sample_ids_sha256": selection_receipt["sample_ids_sha256"],
        "protocol_sha256": sha256(PROTOCOL),
        "design_authority_sha256": sha256(authority),
        "source_receipts": {
            "archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "expected_sha256": expected_archive["sha256"],
                "full_sha256_recomputed_this_invocation": False,
                "per_member_crc_and_payload_sha256_verified": True,
            },
            "native_alignment": source_audits,
            "DINO": {
                "snapshot": model_receipt["snapshot"],
                "dependencies": model_receipt["dependencies"],
                "encoder_parameters": model_receipt["encoder_parameters"],
                "frozen": True,
                "actual_input_size": [DINO_INPUT_SIZE, DINO_INPUT_SIZE],
                "actual_patch_size": 14,
                "actual_patch_grid": [27, 27],
                "actual_trailing_unpatchified_pixels": 0,
                "actual_representation": "L2-normalized float32 last_hidden_state[:,0,:] after complete27x27 patch grid",
                "actual_input_transform": "source-specific physical raw crop then378 letterbox/ImageNet normalization",
                "position_encoding": "upstream Dinov2 interpolation to27x27",
                "source_processor_defaults_applied": False,
                "validated_snapshot_loader_original_contract": {
                    "input_size": model_receipt["input_size"],
                    "unpatchified_trailing_pixels_per_axis": model_receipt[
                        "unpatchified_trailing_pixels_per_axis"
                    ],
                    "used_for_this_forward": False,
                },
            },
        },
        "DINO_input": [3, DINO_INPUT_SIZE, DINO_INPUT_SIZE],
        "DINO_patch_grid": [27, 27],
        "crop_keys": list(CROP_KEYS),
        "features_sha256": sha256(output / "features.npz"),
        "features_array_sha256": array_digest(features),
        "crop_receipts_sha256": sha256(output / "crop_receipts.json"),
        "timing_seconds": {
            "decode": decode_seconds,
            "crop_and_preprocess": preparation_seconds,
            "DINO_encode": encode_seconds,
            "total": total,
            "per_center": total / len(rows),
            "linear_4977_projection": total / len(rows) * 4977,
        },
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "peak_GPU_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
        "full_task_or_extraction_authorized_by_smoke": False,
        "next_gate": "independent full replay, then generalized F(S) dependency/parity resource audit",
    }
    write_new_json(output / "summary.json", summary)
    return summary


def replay(output: Path, reference: Path) -> dict[str, Any]:
    temporary = output / "fresh"
    summary = execute(temporary)
    expected_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    with np.load(reference / "features.npz", allow_pickle=False) as old, np.load(
        temporary / "features.npz", allow_pickle=False
    ) as new:
        identity = np.array_equal(old["sample_ids"], new["sample_ids"]) and np.array_equal(
            old["crop_keys"], new["crop_keys"]
        )
        exact_features = np.array_equal(old["features"], new["features"])
        maximum_difference = float(np.max(np.abs(old["features"].astype(np.float32) - new["features"].astype(np.float32))))
    old_crops = json.loads((reference / "crop_receipts.json").read_text(encoding="utf-8"))
    new_crops = json.loads((temporary / "crop_receipts.json").read_text(encoding="utf-8"))
    crop_exact = old_crops == new_crops
    result = {
        "status": "SOURCE_POSTURE_16_INDEPENDENT_REPLAY_PASS" if identity and exact_features and crop_exact else "FAIL",
        "identity_exact": identity,
        "features_exact_float16": exact_features,
        "feature_max_abs_difference": maximum_difference,
        "crop_and_source_receipts_exact": crop_exact,
        "reference_features_sha256": expected_summary["features_sha256"],
        "fresh_features_sha256": summary["features_sha256"],
        "reference_protocol_sha256": expected_summary["protocol_sha256"],
        "fresh_protocol_sha256": summary["protocol_sha256"],
        "task_labels_read": 0,
        "ARFTR_outputs_read": 0,
        "model_fits": 0,
    }
    if result["status"] == "FAIL":
        raise RuntimeError(f"Source-posture preflight replay failed: {result}")
    write_new_json(output / "replay_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extract", "replay"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "extract":
        result = execute(args.output_dir.resolve())
    else:
        if args.reference_dir is None:
            raise ValueError("Replay requires --reference-dir")
        result = replay(args.output_dir.resolve(), args.reference_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
