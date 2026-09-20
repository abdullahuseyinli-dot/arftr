"""Run the bounded label-blind center-evidence completion preflight.

The executable mode validates sources, preprocessing, target exclusion, frozen
DINO shape/runtime, and transport mechanics for exactly 16 centers.  It has no
training, reconstruction comparison, task-label, router, or full-cache path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.center_completion_data import (  # noqa: E402
    AllowlistedMember,
    apply_raw_mask,
    decode_allowlisted_jpeg,
    letterbox_rgb,
    perturb_hidden_raw_pixels,
    prepare_center,
    prepare_neighbor,
    preprocess_rgb,
    raw_fill_mask,
)
from hac.center_evidence_completion import (  # noqa: E402
    AffineTransport,
    bilinear_transport,
    donor_confidence_logits,
    fit_visible_anchor_transport,
    robust_affine_fit,
    transport_consensus,
)
from hac.image_encoders import load_dinov2_encoder, validate_dinov2_snapshot  # noqa: E402
from hac.okutama_native_video import (  # noqa: E402
    CENTER_SLOT,
    SAMPLE_PATTERN,
    SOURCE_FPS,
    SOURCE_OFFSETS,
    frame_member,
    valid_box,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_center_evidence_completion_protocol.json"
PLAN = ROOT / ".runs/research_20260913/center_evidence_completion_v1/plan_v7.json"
FRAME_MANIFEST = (
    ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/frame_manifest.csv"
)
IMAGE_ALLOWLIST = (
    ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/image_allowlist.csv"
)
MATERIALIZATION_LOCK = (
    ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/materialization_lock.json"
)
DEFAULT_OUTPUT = (
    ROOT / ".runs/research_20260913/center_evidence_completion_preflight_v4"
)
EXPECTED_PROTOCOL_SHA256 = "930fc6c9e9fd9dabb92065804a34b722065e8222acdeadf1affe75cd9a2b2c1c"
EXPECTED_PLAN_SHA256 = "3a218aed27bfc67439748b9f685d322d6edeb1b181a29830d0bc383c83adf03d"
SOURCE_TIME_INDICES = (CENTER_SLOT, 0, 4, 12, 15)
NEIGHBOR_TIME_INDICES = (0, 4, 12, 15)
MICROBATCH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file_stable(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_array_exclusive(path: Path, values: np.ndarray) -> dict[str, Any]:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": sha256_file_stable(path),
        "content_sha256": sha256_array(values),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"Malformed CSV header: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError(f"Malformed CSV rows: {path}")
    return rows


def validate_protocol_and_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file_stable(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Center-completion protocol changed after preflight lock")
    if sha256_file_stable(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Center-completion plan changed after preflight lock")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (
        protocol["schema_version"] != 7
        or protocol["compute_boundary"]["preflight_centers"] != 16
        or protocol["compute_boundary"]["preflight_self_supervised_updates_per_arm"] != 0
        or protocol["compute_boundary"]["preflight_scientific_gate_evaluation"] is not False
        or protocol["compute_boundary"]["preflight_encoder_inputs"] != 176
        or plan["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
        or plan["preflight_selection"]["centers"] != 16
        or plan["task_training_authorized"] is not False
    ):
        raise RuntimeError("Preflight authorization or fixed population changed")
    return protocol, plan


def load_allowlist() -> dict[str, AllowlistedMember]:
    rows = read_csv(IMAGE_ALLOWLIST)
    allowed = {
        row["image_member"]: AllowlistedMember(
            image_member=row["image_member"],
            crc32=int(row["crc32"]),
            size_bytes=int(row["size_bytes"]),
        )
        for row in rows
    }
    if len(allowed) != len(rows):
        raise RuntimeError("Pinned image allowlist contains duplicate member names")
    return allowed


def selected_source_rows(sample_ids: list[str]) -> dict[str, dict[int, dict[str, str]]]:
    selected = set(sample_ids)
    frames: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    forbidden = {"label", "label_index", "action", "support_category"}
    rows = read_csv(FRAME_MANIFEST)
    if forbidden & set(rows[0]):
        raise RuntimeError("Frame manifest unexpectedly contains task-label fields")
    for row in rows:
        if row["sample_id"] not in selected:
            continue
        index = int(row["time_index"])
        if index not in SOURCE_TIME_INDICES:
            continue
        if index in frames[row["sample_id"]]:
            raise RuntimeError("Duplicate selected time index")
        frames[row["sample_id"]][index] = row
    if set(frames) != selected or any(set(values) != set(SOURCE_TIME_INDICES) for values in frames.values()):
        raise RuntimeError("Frozen preflight source support is incomplete")
    for sample_id in sample_ids:
        match = SAMPLE_PATTERN.fullmatch(sample_id)
        if match is None:
            raise RuntimeError("Frozen sample ID no longer satisfies the source grammar")
        recording = match.group("recording")
        track = match.group("track")
        center_frame = int(match.group("frame"))
        for index in SOURCE_TIME_INDICES:
            row = frames[sample_id][index]
            offset = SOURCE_OFFSETS[index]
            source_frame = center_frame + offset
            box = tuple(
                float(row[key])
                for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
            )
            checks = (
                row["provider_recording_id"] == recording,
                row["provider_track_id"] == track,
                int(row["source_frame"]) == source_frame,
                row["image_member"] == frame_member(recording, source_frame),
                math.isclose(float(row["nominal_time_seconds"]), source_frame / SOURCE_FPS),
                math.isclose(float(row["offset_seconds"]), offset / SOURCE_FPS),
                (int(row["image_width"]), int(row["image_height"])) == (1280, 720),
                valid_box(box),
                row["valid_geometry"] == "1",
                row["image_present"] == "1",
                row["valid_frame"] == "1",
                row["missing_reason"] == "",
            )
            if not all(checks):
                raise RuntimeError(f"Selected source row violates its exact contract: {sample_id}/{index}")
    return frames


@torch.inference_mode()
def dense_forward(encoder, pixels: torch.Tensor) -> np.ndarray:
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 384, 384):
        raise ValueError("Preflight DINO input must be B,3,384,384")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = encoder.backbone(
            pixel_values=pixels.to("cuda", non_blocking=True), return_dict=True
        ).last_hidden_state
    if output.shape != (len(pixels), 730, 768) or not torch.isfinite(output).all():
        raise RuntimeError("Frozen DINO dense output shape or finiteness changed")
    dense = output[:, 1:].reshape(-1, 27, 27, 768).to(torch.float16).cpu().numpy()
    if not np.isfinite(dense).all():
        raise RuntimeError("Preflight token cache overflowed float16")
    return dense


def synthetic_transport_checks() -> dict[str, bool]:
    yy, xx = np.mgrid[1:26:4, 1:26:4]
    center_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    expected = np.array([[1.0, 0.05, 0.5], [-0.03, 1.0, -0.25]])
    neighbor_xy = np.column_stack([center_xy, np.ones(len(center_xy))]) @ expected.T
    fitted, residual = robust_affine_fit(center_xy, neighbor_xy)
    affine = bool(np.allclose(fitted, expected, atol=1e-10) and residual < 1e-10)

    grid_y, grid_x = np.mgrid[:27, :27]
    tokens = np.stack([grid_x, grid_y], axis=2).astype(np.float32)
    target = np.zeros((27, 27), dtype=bool)
    target[10, 11] = True
    fractional = AffineTransport(
        np.array([[1, 0, 0.5], [0, 1, 0.25]], dtype=np.float64),
        12,
        0.0,
        True,
        "ok",
    )
    values, observed = bilinear_transport(tokens, target, fractional)
    bilinear = bool(observed[0] and np.allclose(values[0], [11.5, 10.25]))

    donor_features = np.arange(4 * 1 * 3, dtype=np.float32).reshape(4, 1, 3)
    unavailable = np.zeros((4, 1), dtype=bool)
    logits = np.full((4, 1), -np.inf)
    center_only = np.array([[19.0, 23.0, 29.0]], dtype=np.float32)
    aggregate = transport_consensus(donor_features, unavailable, logits, center_only)
    all_invalid = bool(
        np.array_equal(aggregate.features, center_only)
        and not aggregate.available.any()
    )
    return {
        "affine_scale_shear_exact": affine,
        "fractional_bilinear_exact": bilinear,
        "all_invalid_copies_center_only_exact": all_invalid,
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("Preflight output directory must be new or empty")
    else:
        output.mkdir(parents=True)
    protocol, plan = validate_protocol_and_plan()
    sample_ids = plan["preflight_selection"]["sample_ids"]
    if len(sample_ids) != 16 or len(set(sample_ids)) != 16:
        raise RuntimeError("Frozen preflight identities are invalid")
    wrong_track_receipt = plan["neighbor_geometry"]["fixed_wrong_track_map"]
    wrong_track_path = ROOT / wrong_track_receipt["path"]
    if sha256_file_stable(wrong_track_path) != wrong_track_receipt["sha256"]:
        raise RuntimeError("Fixed wrong-track map changed after planning")
    wrong_track_rows = json.loads(wrong_track_path.read_text(encoding="utf-8"))["rows"]
    wrong_track_map = {
        (row["sample_id"], int(row["time_index"])): row
        for row in wrong_track_rows
        if row["sample_id"] in set(sample_ids)
    }
    if (
        len(wrong_track_map) != 16 * 4
        or not all(row["donor_available"] for row in wrong_track_map.values())
    ):
        raise RuntimeError("Preflight wrong-track control support is incomplete")

    archive_path = Path(protocol["pinned_evidence"]["provider_frame_archive"]["path"])
    archive_stat_before = archive_path.stat()
    hash_started = time.perf_counter()
    archive_hash = sha256_file_stable(archive_path)
    archive_hash_seconds = time.perf_counter() - hash_started
    expected_archive = protocol["pinned_evidence"]["provider_frame_archive"]
    if (
        archive_hash != expected_archive["sha256"]
        or archive_stat_before.st_size != expected_archive["size_bytes"]
    ):
        raise RuntimeError("Provider frame archive differs from the locked source")

    lock = json.loads(MATERIALIZATION_LOCK.read_text(encoding="utf-8"))
    model_root = Path(lock["inputs"]["dinov2_snapshot"]["path"])
    model_snapshot = validate_dinov2_snapshot(model_root)
    source_files = [
        Path(__file__),
        ROOT / "src/hac/center_completion_data.py",
        ROOT / "src/hac/center_evidence_completion.py",
        ROOT / "src/hac/image_encoders.py",
        ROOT / "src/hac/okutama_native_video.py",
        ROOT / "src/hac/video_encoders.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file_stable(path) for path in source_files}
    request = {
        "status": "CENTER_EVIDENCE_COMPLETION_PREFLIGHT_REQUEST_LOCKED",
        "scope": protocol["compute_boundary"]["preflight_scope"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "frame_manifest_sha256": sha256_file_stable(FRAME_MANIFEST),
        "image_allowlist_sha256": sha256_file_stable(IMAGE_ALLOWLIST),
        "materialization_lock_sha256": sha256_file_stable(MATERIALIZATION_LOCK),
        "wrong_track_map_sha256": wrong_track_receipt["sha256"],
        "archive": {
            "path": str(archive_path),
            "size_bytes": archive_stat_before.st_size,
            "sha256": archive_hash,
            "hash_seconds": archive_hash_seconds,
        },
        "model_snapshot": model_snapshot,
        "source_hashes": source_hashes,
        "sample_ids": sample_ids,
        "canonical_time_indices": list(SOURCE_TIME_INDICES),
        "neighbor_offsets_frames": protocol["pilot"]["neighbor_offsets_frames_at_30fps"],
        "microbatch": MICROBATCH,
        "precision": "cuda_bfloat16_forward_float16_cache",
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "Pillow", "torch", "transformers", "safetensors")
        },
        "optimizer_updates": 0,
        "task_labels_read": 0,
        "arftr_arrays_read": 0,
        "pose_outputs_read": 0,
    }
    write_json_exclusive(output / "request.json", request)

    frames = selected_source_rows(sample_ids)
    allowlist = load_allowlist()
    inputs: list[torch.Tensor] = []
    input_order: list[dict[str, Any]] = []
    centers = []
    neighbors_by_center = []
    wrong_neighbors_by_center = []
    view_receipts = []
    member_receipts: dict[str, dict[str, Any]] = {}
    member_accesses = Counter()
    dependency_checks = []
    decoded: dict[str, Any] = {}
    prepare_started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise RuntimeError("Provider archive contains duplicate ZIP member names")

        def get_image(row: dict[str, str]):
            member = row["image_member"]
            member_accesses[member] += 1
            if member not in decoded:
                image, receipt = decode_allowlisted_jpeg(archive, row, allowlist)
                decoded[member] = image
                member_receipts[member] = receipt
            return decoded[member]

        for center_index, sample_id in enumerate(sample_ids):
            center_row = frames[sample_id][CENTER_SLOT]
            center = prepare_center(get_image(center_row), center_row)
            center_neighbors = [
                prepare_neighbor(get_image(frames[sample_id][index]), frames[sample_id][index])
                for index in NEIGHBOR_TIME_INDICES
            ]
            wrong_neighbors = []
            for index in NEIGHBOR_TIME_INDICES:
                source_row = frames[sample_id][index]
                mapped = wrong_track_map[(sample_id, index)]
                wrong_row = dict(source_row)
                for key, value in zip(
                    ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"),
                    mapped["donor_box"],
                    strict=True,
                ):
                    wrong_row[key] = format(value, ".17g")
                wrong_neighbors.append(prepare_neighbor(get_image(source_row), wrong_row))
            centers.append(center)
            neighbors_by_center.append(center_neighbors)
            wrong_neighbors_by_center.append(wrong_neighbors)
            input_tensors = [
                center.unmasked_pixels,
                center.masked_pixels[0],
                center.masked_pixels[1],
                *(neighbor.pixels for neighbor in center_neighbors),
                *(neighbor.pixels for neighbor in wrong_neighbors),
            ]
            roles = [
                "teacher",
                "masked_upper",
                "masked_lower",
                "neighbor_-8",
                "neighbor_-4",
                "neighbor_+4",
                "neighbor_+7",
                "wrong_track_-8",
                "wrong_track_-4",
                "wrong_track_+4",
                "wrong_track_+7",
            ]
            for role, tensor in zip(roles, input_tensors, strict=True):
                inputs.append(tensor)
                input_order.append(
                    {"input_index": len(inputs) - 1, "center_index": center_index, "sample_id": sample_id, "role": role}
                )
            for mask_id in ("upper", "lower"):
                fill_mask = raw_fill_mask(center.raw_crop, mask_id)
                altered = perturb_hidden_raw_pixels(center.raw_crop.rgb, fill_mask)
                original_masked = apply_raw_mask(center.raw_crop.rgb, fill_mask)
                altered_masked = apply_raw_mask(altered, fill_mask)
                exact_raw = np.array_equal(original_masked, altered_masked)
                exact_letterbox = np.array_equal(
                    letterbox_rgb(original_masked), letterbox_rgb(altered_masked)
                )
                exact_normalized = torch.equal(
                    preprocess_rgb(original_masked), preprocess_rgb(altered_masked)
                )
                if not (exact_raw and exact_letterbox and exact_normalized):
                    raise RuntimeError("Hidden target pixels survived the pre-encoder mask")
                dependency_checks.append(
                    {
                        "sample_id": sample_id,
                        "mask_id": mask_id,
                        "altered_hidden_pixels": int(fill_mask.sum()),
                        "masked_raw_bit_exact": exact_raw,
                        "letterbox_bit_exact": exact_letterbox,
                        "normalized_bit_exact": exact_normalized,
                        "masked_raw_sha256": sha256_array(original_masked),
                    }
                )
            view_receipts.append(
                {
                    "sample_id": sample_id,
                    "center_box": [
                        float(center_row[key])
                        for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
                    ],
                    "center_crop_box": list(center.raw_crop.geometry.crop_box),
                    "center_letterbox_resized": list(center.raw_crop.geometry.resized_size),
                    "center_letterbox_padding": list(center.raw_crop.geometry.padding),
                    "target_patch_counts": center.target_patch_masks.reshape(2, -1).sum(axis=1).astype(int).tolist(),
                    "visible_anchor_patch_counts": center.visible_patch_masks.reshape(2, -1).sum(axis=1).astype(int).tolist(),
                    "target_masks_sha256": sha256_array(center.target_patch_masks),
                    "visible_masks_sha256": sha256_array(center.visible_patch_masks),
                    "wrong_track_boxes": [
                        wrong_track_map[(sample_id, index)]["donor_box"]
                        for index in NEIGHBOR_TIME_INDICES
                    ],
                    "normalized_inputs_sha256": [sha256_array(tensor.numpy()) for tensor in input_tensors],
                }
            )
    prepare_seconds = time.perf_counter() - prepare_started
    if len(inputs) != 16 * 11 or len(dependency_checks) != 16 * 2:
        raise RuntimeError("Preflight input or dependency-audit population changed")
    for member, count in member_accesses.items():
        member_receipts[member]["view_access_count"] = count

    if not torch.cuda.is_available():
        raise RuntimeError("Locked preflight requires CUDA bfloat16 extraction")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    encoder, model_receipt = load_dinov2_encoder(model_root, device="cuda")
    pixel_batch = torch.stack(inputs)
    token_parts = []
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    for start in range(0, len(pixel_batch), MICROBATCH):
        token_parts.append(dense_forward(encoder, pixel_batch[start : start + MICROBATCH]))
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started
    tokens = np.concatenate(token_parts)
    if tokens.shape != (176, 27, 27, 768) or tokens.dtype != np.float16:
        raise RuntimeError("Preflight token population, shape, or dtype changed")
    tokens = tokens.reshape(16, 11, 27, 27, 768)
    teacher = tokens[:, 0]
    masked = tokens[:, 1:3]
    neighbors = tokens[:, 3:7]
    wrong_neighbors = tokens[:, 7:11]
    target_masks = np.stack([center.target_patch_masks for center in centers])
    visible_masks = np.stack([center.visible_patch_masks for center in centers])
    neighbor_valid = np.stack(
        [[neighbor.valid_patch_mask for neighbor in values] for values in neighbors_by_center]
    )
    wrong_neighbor_valid = np.stack(
        [
            [neighbor.valid_patch_mask for neighbor in values]
            for values in wrong_neighbors_by_center
        ]
    )

    transport_started = time.perf_counter()
    transport_rows = []
    for center_index, sample_id in enumerate(sample_ids):
        for mask_index, mask_id in enumerate(("upper", "lower")):
            def process_donors(
                center_tokens: np.ndarray,
                target_mask: np.ndarray,
                center_valid: np.ndarray,
                token_block: np.ndarray,
                valid_block: np.ndarray,
            ) -> tuple[list[dict[str, Any]], Any]:
                donor_values = []
                donor_observed = []
                donor_logits = []
                donor_receipts = []
                for donor_index, offset in enumerate((-8, -4, 4, 7)):
                    transform, matches = fit_visible_anchor_transport(
                        center_tokens.astype(np.float32),
                        token_block[donor_index].astype(np.float32),
                        target_mask,
                        center_valid=center_valid,
                        neighbor_valid=valid_block[donor_index],
                    )
                    values, observed = bilinear_transport(
                        token_block[donor_index].astype(np.float32),
                        target_mask,
                        transform,
                        neighbor_valid=valid_block[donor_index],
                    )
                    logits = donor_confidence_logits(
                        matches,
                        transform,
                        target_mask,
                        observed,
                    )
                    donor_values.append(values)
                    donor_observed.append(observed)
                    donor_logits.append(logits)
                    donor_receipts.append(
                        {
                            "offset_frames": offset,
                            "match_count": transform.match_count,
                            "affine_matrix": (
                                transform.matrix.tolist() if transform.valid else None
                            ),
                            "weighted_residual": (
                                transform.weighted_residual
                                if math.isfinite(transform.weighted_residual)
                                else None
                            ),
                            "valid": transform.valid,
                            "reason": transform.reason,
                            "transported_targets": int(observed.sum()),
                        }
                    )
                stacked_values = np.stack(donor_values)
                aggregate = transport_consensus(
                    stacked_values,
                    np.stack(donor_observed),
                    np.stack(donor_logits),
                    np.zeros_like(stacked_values[0]),
                )
                if not np.isfinite(aggregate.features).all():
                    raise RuntimeError("Real-data transport smoke emitted nonfinite evidence")
                return donor_receipts, aggregate

            true_receipts, true_consensus = process_donors(
                masked[center_index, mask_index],
                target_masks[center_index, mask_index],
                visible_masks[center_index, mask_index],
                neighbors[center_index],
                neighbor_valid[center_index],
            )
            wrong_receipts, wrong_consensus = process_donors(
                masked[center_index, mask_index],
                target_masks[center_index, mask_index],
                visible_masks[center_index, mask_index],
                wrong_neighbors[center_index],
                wrong_neighbor_valid[center_index],
            )
            transport_rows.append(
                {
                    "sample_id": sample_id,
                    "mask_id": mask_id,
                    "target_count": int(target_masks[center_index, mask_index].sum()),
                    "true_track_donors": true_receipts,
                    "wrong_track_donors": wrong_receipts,
                    "true_consensus_available_targets": int(true_consensus.available.sum()),
                    "wrong_consensus_available_targets": int(wrong_consensus.available.sum()),
                    "true_hard_retain_targets": int((~true_consensus.available).sum()),
                    "wrong_hard_retain_targets": int((~wrong_consensus.available).sum()),
                    "true_mean_donor_weight_entropy": float(
                        true_consensus.donor_weight_entropy.mean()
                    ),
                    "wrong_mean_donor_weight_entropy": float(
                        wrong_consensus.donor_weight_entropy.mean()
                    ),
                }
            )
    transport_seconds = time.perf_counter() - transport_started
    true_mask_rows_with_consensus = sum(
        row["true_consensus_available_targets"] > 0 for row in transport_rows
    )
    wrong_mask_rows_with_consensus = sum(
        row["wrong_consensus_available_targets"] > 0 for row in transport_rows
    )
    if true_mask_rows_with_consensus != 32:
        raise RuntimeError("True-track consensus path was not exercised for every mask row")
    synthetic = synthetic_transport_checks()
    if not all(synthetic.values()):
        raise RuntimeError("Synthetic affine/transport preflight failed")

    artifacts = {
        "teacher": save_array_exclusive(output / "teacher_tokens.npy", teacher),
        "masked": save_array_exclusive(output / "masked_tokens.npy", masked),
        "neighbors": save_array_exclusive(output / "neighbor_tokens.npy", neighbors),
        "wrong_neighbors": save_array_exclusive(
            output / "wrong_neighbor_tokens.npy", wrong_neighbors
        ),
        "target_masks": save_array_exclusive(output / "target_masks.npy", target_masks),
        "visible_masks": save_array_exclusive(output / "visible_masks.npy", visible_masks),
        "neighbor_valid": save_array_exclusive(output / "neighbor_valid.npy", neighbor_valid),
        "wrong_neighbor_valid": save_array_exclusive(
            output / "wrong_neighbor_valid.npy", wrong_neighbor_valid
        ),
    }
    write_json_exclusive(output / "view_receipts.json", {"rows": view_receipts})
    write_json_exclusive(output / "member_receipts.json", {"members": member_receipts})
    write_json_exclusive(output / "dependency_audit.json", {"rows": dependency_checks})
    write_json_exclusive(output / "transport_smoke.json", {"rows": transport_rows})
    archive_stat_after = archive_path.stat()
    if (
        archive_stat_after.st_size,
        archive_stat_after.st_mtime_ns,
    ) != (archive_stat_before.st_size, archive_stat_before.st_mtime_ns):
        raise RuntimeError("Provider frame archive changed during preflight")
    if any(sha256_file_stable(path) != source_hashes[str(path.relative_to(ROOT))] for path in source_files):
        raise RuntimeError("Executable preflight source changed during execution")
    variable_seconds = prepare_seconds + encode_seconds + transport_seconds
    projected_full_seconds = archive_hash_seconds + variable_seconds * (128 / 16)
    summary = {
        "status": "CENTER_EVIDENCE_COMPLETION_OPERATIONAL_PREFLIGHT_PASS",
        "scientific_gate_evaluated": False,
        "full_extraction_authorized": False,
        "task_training_authorized": False,
        "centers": 16,
        "scenarios": len(plan["preflight_selection"]["scenario_counts"]),
        "encoded_inputs": len(inputs),
        "microbatch": MICROBATCH,
        "artifacts": artifacts,
        "source_members": {
            "unique_decoded": len(member_receipts),
            "view_accesses": int(sum(member_accesses.values())),
        },
        "target_patch_count_range": [
            int(target_masks.reshape(32, -1).sum(axis=1).min()),
            int(target_masks.reshape(32, -1).sum(axis=1).max()),
        ],
        "transport_code_path": {
            "true_track_donor_attempts": 16 * 2 * 4,
            "wrong_track_donor_attempts": 16 * 2 * 4,
            "true_track_valid_affine_fits": sum(
                donor["valid"]
                for row in transport_rows
                for donor in row["true_track_donors"]
            ),
            "wrong_track_valid_affine_fits": sum(
                donor["valid"]
                for row in transport_rows
                for donor in row["wrong_track_donors"]
            ),
            "true_mask_rows_with_any_consensus": true_mask_rows_with_consensus,
            "wrong_mask_rows_with_any_consensus": wrong_mask_rows_with_consensus,
            "learned_retain_transport_gate_fitted": False,
        },
        "synthetic_checks": synthetic,
        "runtime_seconds": {
            "archive_hash": archive_hash_seconds,
            "prepare_and_dependency": prepare_seconds,
            "dino_encode": encode_seconds,
            "transport_smoke": transport_seconds,
            "projected_128_center_total": projected_full_seconds,
        },
        "projected_full_under_20_minutes": projected_full_seconds <= 1200,
        "cuda": {
            "device": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "model": {
            **model_receipt,
            "preflight_representation": "last_hidden_state[:,1:] reshaped row-major to 27x27x768",
        },
        "zero_access_counters": {
            "task_labels": 0,
            "arftr_probability_arrays": 0,
            "pose_outputs": 0,
            "support_categories": 0,
            "annotation_flags_interpreted": 0,
            "optimizer_updates": 0,
            "task_fits": 0,
            "router_fits": 0,
        },
        "request_sha256": sha256_file_stable(output / "request.json"),
        "next_authorization": "independent artifact/code audit only; the 128-center screen remains unlaunched",
    }
    write_json_exclusive(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def main() -> None:
    run(parse_args().output_dir)


if __name__ == "__main__":
    main()
