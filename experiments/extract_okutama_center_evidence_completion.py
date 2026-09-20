"""Extract the locked 128-center label-blind completion cache; perform no fits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
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
PLAN = ROOT / ".runs/research_20260913/center_evidence_completion_v1/plan_v8.json"
PILOT_SELECTION = (
    ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"
)
FRAME_MANIFEST = (
    ROOT
    / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/frame_manifest.csv"
)
IMAGE_ALLOWLIST = (
    ROOT
    / ".runs/research_20260907/okutama_native_video_p0_r1/manifest/image_allowlist.csv"
)
MATERIALIZATION_LOCK = (
    ROOT
    / ".runs/research_20260907/okutama_native_video_p0_r1/materialization_lock.json"
)
DEFAULT_OUTPUT = (
    ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1"
)
EXPECTED_PROTOCOL_SHA256 = (
    "f2e95656cfd179e72522d02103d2542e96b4124ccb76c5f4c0c1706783505244"
)
EXPECTED_PLAN_SHA256 = (
    "b7e148c8efe1e72fc0000ffd0713aecf16a3a4e21387d9f520a9dac6e5e6b63c"
)
EXPECTED_WRONG_MAP_SHA256 = (
    "fd578ddd89d866ae80bf107975216195ef0ea96d8353d2178b9428f0418c27c6"
)
SOURCE_TIME_INDICES = (CENTER_SLOT, 0, 4, 12, 15)
NEIGHBOR_TIME_INDICES = (0, 4, 12, 15)
MICROBATCH = 8
ROWS = 128
ARRAY_SPECS = {
    "teacher_tokens.npy": (np.float16, (ROWS, 27, 27, 768)),
    "masked_tokens.npy": (np.float16, (ROWS, 2, 27, 27, 768)),
    "neighbor_tokens.npy": (np.float16, (ROWS, 4, 27, 27, 768)),
    "wrong_neighbor_tokens.npy": (np.float16, (ROWS, 4, 27, 27, 768)),
    "target_masks.npy": (np.bool_, (ROWS, 2, 27, 27)),
    "visible_masks.npy": (np.bool_, (ROWS, 2, 27, 27)),
    "neighbor_valid.npy": (np.bool_, (ROWS, 4, 27, 27)),
    "wrong_neighbor_valid.npy": (np.bool_, (ROWS, 4, 27, 27)),
    "wrong_slot_available.npy": (np.bool_, (ROWS, 4)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
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


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise RuntimeError(f"Malformed CSV header: {path}")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise RuntimeError(f"Malformed CSV rows: {path}")
    return rows


def _box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(
        float(row[key])
        for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
    )


def _intersection_area(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def validate_locks() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if sha256_file_stable(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Center-completion protocol changed after extraction lock")
    if sha256_file_stable(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Center-completion plan changed after extraction lock")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    selection = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    if (
        protocol["schema_version"] != 8
        or protocol["status"]
        != "LABEL_BLIND_128_CENTER_EXTRACTION_AUTHORIZED_NO_TASK_FIT"
        or plan["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
        or plan["full_extraction_authorized"] is not True
        or plan["task_training_authorized"] is not False
        or len(selection) != ROWS
        or len({row["sample_id"] for row in selection}) != ROWS
        or sha256_file_stable(PILOT_SELECTION)
        != plan["population"]["selection_sha256"]
    ):
        raise RuntimeError("Full extraction authorization or population changed")
    map_receipt = plan["neighbor_geometry"]["fixed_wrong_track_map"]
    wrong_path = ROOT / map_receipt["path"]
    if (
        map_receipt["sha256"] != EXPECTED_WRONG_MAP_SHA256
        or sha256_file_stable(wrong_path) != EXPECTED_WRONG_MAP_SHA256
    ):
        raise RuntimeError("Wrong-track map changed after extraction lock")
    wrong_rows = json.loads(wrong_path.read_text(encoding="utf-8"))["rows"]
    if (
        len(wrong_rows) != ROWS * 4
        or sum(row["donor_available"] for row in wrong_rows) != 496
    ):
        raise RuntimeError("Wrong-track control population changed")
    return protocol, plan, selection, wrong_rows


def selected_source_rows(
    sample_ids: list[str],
) -> dict[str, dict[int, dict[str, str]]]:
    selected = set(sample_ids)
    frames: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    rows = read_csv(FRAME_MANIFEST)
    if {"label", "label_index", "action", "support_category"} & set(rows[0]):
        raise RuntimeError("Frame manifest unexpectedly contains task-label fields")
    for row in rows:
        if row["sample_id"] not in selected:
            continue
        index = int(row["time_index"])
        if index not in SOURCE_TIME_INDICES:
            continue
        if index in frames[row["sample_id"]]:
            raise RuntimeError("Duplicate selected source row")
        frames[row["sample_id"]][index] = row
    if set(frames) != selected or any(
        set(values) != set(SOURCE_TIME_INDICES) for values in frames.values()
    ):
        raise RuntimeError("Frozen full-cache source support is incomplete")
    for sample_id in sample_ids:
        match = SAMPLE_PATTERN.fullmatch(sample_id)
        if match is None:
            raise RuntimeError("Frozen sample ID violates the source grammar")
        recording = match.group("recording")
        track = match.group("track")
        center_frame = int(match.group("frame"))
        for index in SOURCE_TIME_INDICES:
            row = frames[sample_id][index]
            offset = SOURCE_OFFSETS[index]
            source_frame = center_frame + offset
            checks = (
                row["provider_recording_id"] == recording,
                row["provider_track_id"] == track,
                int(row["source_frame"]) == source_frame,
                row["image_member"] == frame_member(recording, source_frame),
                math.isclose(
                    float(row["nominal_time_seconds"]), source_frame / SOURCE_FPS
                ),
                math.isclose(float(row["offset_seconds"]), offset / SOURCE_FPS),
                (int(row["image_width"]), int(row["image_height"])) == (1280, 720),
                valid_box(_box(row)),
                row["valid_geometry"] == "1",
                row["image_present"] == "1",
                row["valid_frame"] == "1",
                row["missing_reason"] == "",
            )
            if not all(checks):
                raise RuntimeError(f"Source contract failed: {sample_id}/{index}")
    return frames


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
        raise RuntimeError("Image allowlist contains duplicate members")
    return allowed


def validate_wrong_map(
    sample_ids: list[str],
    frames: dict[str, dict[int, dict[str, str]]],
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {(sample_id, index) for sample_id in sample_ids for index in NEIGHBOR_TIME_INDICES}
    mapped = {(row["sample_id"], int(row["time_index"])): row for row in rows}
    if set(mapped) != expected:
        raise RuntimeError("Wrong-track map keys changed")
    for key, row in mapped.items():
        source = frames[key[0]][key[1]]
        if (
            int(row["source_frame"]) != int(source["source_frame"])
            or row["image_member"] != source["image_member"]
            or row["source_track_id"] != source["provider_track_id"]
        ):
            raise RuntimeError("Wrong-track map source ancestry changed")
        if row["donor_available"]:
            donor_box = tuple(float(value) for value in row["donor_box"])
            if (
                row["donor_track_id"] == source["provider_track_id"]
                or not valid_box(donor_box)
                or _intersection_area(_box(source), donor_box) != 0
            ):
                raise RuntimeError("Wrong-track donor violates its control contract")
        elif any(
            row[name] is not None
            for name in ("donor_track_id", "donor_box", "selection_sha256")
        ):
            raise RuntimeError("Unavailable wrong-track slot contains a donor")
    return mapped


@torch.inference_mode()
def dense_forward(encoder, pixels: torch.Tensor) -> np.ndarray:
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 384, 384):
        raise ValueError("DINO input must be B,3,384,384")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = encoder.backbone(
            pixel_values=pixels.to("cuda", non_blocking=True), return_dict=True
        ).last_hidden_state
    if output.shape != (len(pixels), 730, 768) or not torch.isfinite(output).all():
        raise RuntimeError("Frozen DINO dense output shape or finiteness changed")
    values = output[:, 1:].reshape(-1, 27, 27, 768).to(torch.float16)
    result = values.cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("Dense cache overflowed float16")
    return result


def validate_only_receipt() -> dict[str, Any]:
    protocol, plan, selection, wrong_rows = validate_locks()
    sample_ids = [row["sample_id"] for row in selection]
    frames = selected_source_rows(sample_ids)
    validate_wrong_map(sample_ids, frames, wrong_rows)
    allowlist = load_allowlist()
    selected_members = {
        frames[sample_id][index]["image_member"]
        for sample_id in sample_ids
        for index in SOURCE_TIME_INDICES
    }
    if not selected_members <= set(allowlist):
        raise RuntimeError("A selected source member is not allowlisted")
    archive = Path(protocol["pinned_evidence"]["provider_frame_archive"]["path"])
    expected_archive = protocol["pinned_evidence"]["provider_frame_archive"]
    if archive.stat().st_size != expected_archive["size_bytes"]:
        raise RuntimeError("Provider archive size changed")
    return {
        "status": "CENTER_EVIDENCE_COMPLETION_FULL_CACHE_VALIDATION_PASS",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "wrong_track_map_sha256": EXPECTED_WRONG_MAP_SHA256,
        "centers": len(sample_ids),
        "source_rows": sum(len(rows) for rows in frames.values()),
        "unique_selected_members": len(selected_members),
        "wrong_track_available_slots": sum(
            row["donor_available"] for row in wrong_rows
        ),
        "planned_encoded_inputs": plan["planned_cost"]["total_frozen_dino_forwards"],
        "task_training_authorized": False,
    }


def _open_arrays(output: Path) -> dict[str, np.memmap]:
    arrays = {}
    for name, (dtype, shape) in ARRAY_SPECS.items():
        arrays[name] = np.lib.format.open_memmap(
            output / name, mode="w+", dtype=dtype, shape=shape
        )
        arrays[name][...] = False if dtype is np.bool_ else 0
        arrays[name].flush()
    return arrays


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("Full-cache output directory must be new or empty")
    else:
        output.mkdir(parents=True)
    protocol, plan, selection, wrong_rows = validate_locks()
    sample_ids = [row["sample_id"] for row in selection]
    frames = selected_source_rows(sample_ids)
    wrong_map = validate_wrong_map(sample_ids, frames, wrong_rows)
    allowlist = load_allowlist()
    archive_path = Path(protocol["pinned_evidence"]["provider_frame_archive"]["path"])
    archive_before = archive_path.stat()
    hash_started = time.perf_counter()
    archive_hash = sha256_file_stable(archive_path)
    archive_hash_seconds = time.perf_counter() - hash_started
    expected_archive = protocol["pinned_evidence"]["provider_frame_archive"]
    if (
        archive_hash != expected_archive["sha256"]
        or archive_before.st_size != expected_archive["size_bytes"]
    ):
        raise RuntimeError("Provider frame archive differs from the locked source")

    lock = json.loads(MATERIALIZATION_LOCK.read_text(encoding="utf-8"))
    model_root = Path(lock["inputs"]["dinov2_snapshot"]["path"])
    model_snapshot = validate_dinov2_snapshot(model_root)
    source_files = [
        Path(__file__),
        ROOT / "src/hac/center_completion_data.py",
        ROOT / "src/hac/image_encoders.py",
        ROOT / "src/hac/okutama_native_video.py",
        ROOT / "src/hac/video_encoders.py",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file_stable(path)
        for path in source_files
    }
    request = {
        "status": "CENTER_EVIDENCE_COMPLETION_FULL_CACHE_REQUEST_LOCKED",
        "scope": "128-center label-blind frozen-DINO extraction; no fitting",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "wrong_track_map_sha256": EXPECTED_WRONG_MAP_SHA256,
        "pilot_selection_sha256": sha256_file_stable(PILOT_SELECTION),
        "frame_manifest_sha256": sha256_file_stable(FRAME_MANIFEST),
        "image_allowlist_sha256": sha256_file_stable(IMAGE_ALLOWLIST),
        "materialization_lock_sha256": sha256_file_stable(MATERIALIZATION_LOCK),
        "archive": {
            "path": str(archive_path),
            "size_bytes": archive_before.st_size,
            "sha256": archive_hash,
            "hash_seconds": archive_hash_seconds,
        },
        "model_snapshot": model_snapshot,
        "source_hashes": source_hashes,
        "sample_ids": sample_ids,
        "canonical_time_indices": list(SOURCE_TIME_INDICES),
        "microbatch": MICROBATCH,
        "expected_encoded_inputs": 1392,
        "missing_wrong_track_slots_are_zero_and_invalid": True,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "Pillow", "torch", "transformers", "safetensors")
        },
        "zero_access_contract": {
            "clip_index": 0,
            "task_labels": 0,
            "arftr_probability_arrays": 0,
            "pose_outputs": 0,
            "support_categories": 0,
            "optimizer_updates": 0,
        },
    }
    write_json_exclusive(output / "request.json", request)
    snapshot_dir = output / "source_snapshot"
    snapshot_dir.mkdir()
    for path in source_files:
        target = snapshot_dir / path.name
        shutil.copyfile(path, target)
        if sha256_file_stable(target) != source_hashes[
            str(path.relative_to(ROOT)).replace("\\", "/")
        ]:
            raise RuntimeError("Executed-source snapshot changed while copying")

    arrays = _open_arrays(output)
    view_receipts = []
    dependency_rows = []
    member_receipts: dict[str, dict[str, Any]] = {}
    member_accesses = Counter()
    encoded_inputs = 0
    prepare_seconds = 0.0
    encode_seconds = 0.0

    if not torch.cuda.is_available():
        raise RuntimeError("Locked full extraction requires CUDA bfloat16")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    encoder, model_receipt = load_dinov2_encoder(model_root, device="cuda")

    with zipfile.ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise RuntimeError("Provider archive contains duplicate member names")
        for center_index, sample_id in enumerate(sample_ids):
            prepare_started = time.perf_counter()
            images: dict[str, Any] = {}

            def get_image(row: dict[str, str], images=images):
                member = row["image_member"]
                if member not in images:
                    image, receipt = decode_allowlisted_jpeg(archive, row, allowlist)
                    images[member] = image
                    if member in member_receipts:
                        prior = dict(member_receipts[member])
                        prior.pop("view_access_count", None)
                        if prior != receipt:
                            raise RuntimeError("Repeated member decode receipt changed")
                    else:
                        member_receipts[member] = receipt
                member_accesses[member] += 1
                return images[member]

            center_row = frames[sample_id][CENTER_SLOT]
            center = prepare_center(get_image(center_row), center_row)
            true_neighbors = [
                prepare_neighbor(
                    get_image(frames[sample_id][index]), frames[sample_id][index]
                )
                for index in NEIGHBOR_TIME_INDICES
            ]
            input_tensors = [
                center.unmasked_pixels,
                center.masked_pixels[0],
                center.masked_pixels[1],
                *(neighbor.pixels for neighbor in true_neighbors),
            ]
            roles: list[tuple[str, int | None]] = [
                ("teacher", None),
                ("masked", 0),
                ("masked", 1),
                *(("neighbor", index) for index in range(4)),
            ]
            wrong_boxes: list[list[float] | None] = []
            for donor_index, time_index in enumerate(NEIGHBOR_TIME_INDICES):
                source_row = frames[sample_id][time_index]
                mapped = wrong_map[(sample_id, time_index)]
                wrong_boxes.append(mapped["donor_box"])
                if not mapped["donor_available"]:
                    continue
                wrong_row = dict(source_row)
                for key, value in zip(
                    ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"),
                    mapped["donor_box"],
                    strict=True,
                ):
                    wrong_row[key] = format(value, ".17g")
                wrong = prepare_neighbor(get_image(source_row), wrong_row)
                input_tensors.append(wrong.pixels)
                roles.append(("wrong_neighbor", donor_index))
                arrays["wrong_neighbor_valid.npy"][center_index, donor_index] = (
                    wrong.valid_patch_mask
                )
                arrays["wrong_slot_available.npy"][center_index, donor_index] = True

            arrays["target_masks.npy"][center_index] = center.target_patch_masks
            arrays["visible_masks.npy"][center_index] = center.visible_patch_masks
            arrays["neighbor_valid.npy"][center_index] = np.stack(
                [neighbor.valid_patch_mask for neighbor in true_neighbors]
            )
            for mask_id in ("upper", "lower"):
                fill_mask = raw_fill_mask(center.raw_crop, mask_id)
                altered = perturb_hidden_raw_pixels(center.raw_crop.rgb, fill_mask)
                original_masked = apply_raw_mask(center.raw_crop.rgb, fill_mask)
                altered_masked = apply_raw_mask(altered, fill_mask)
                checks = {
                    "masked_raw_bit_exact": np.array_equal(
                        original_masked, altered_masked
                    ),
                    "letterbox_bit_exact": np.array_equal(
                        letterbox_rgb(original_masked), letterbox_rgb(altered_masked)
                    ),
                    "normalized_bit_exact": torch.equal(
                        preprocess_rgb(original_masked),
                        preprocess_rgb(altered_masked),
                    ),
                }
                if not all(checks.values()):
                    raise RuntimeError("Hidden target pixels survived pre-encoder masking")
                dependency_rows.append(
                    {
                        "sample_id": sample_id,
                        "mask_id": mask_id,
                        "altered_hidden_pixels": int(fill_mask.sum()),
                        **checks,
                        "masked_raw_sha256": sha256_array(original_masked),
                    }
                )
            prepare_seconds += time.perf_counter() - prepare_started

            pixel_batch = torch.stack(input_tensors)
            token_parts = []
            torch.cuda.synchronize()
            encode_started = time.perf_counter()
            for start in range(0, len(pixel_batch), MICROBATCH):
                token_parts.append(
                    dense_forward(encoder, pixel_batch[start : start + MICROBATCH])
                )
            torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - encode_started
            tokens = np.concatenate(token_parts)
            if len(tokens) != len(roles):
                raise RuntimeError("Encoded token/role count changed")
            for token, (role, donor_index) in zip(tokens, roles, strict=True):
                if role == "teacher":
                    arrays["teacher_tokens.npy"][center_index] = token
                elif role == "masked":
                    arrays["masked_tokens.npy"][center_index, donor_index] = token
                elif role == "neighbor":
                    arrays["neighbor_tokens.npy"][center_index, donor_index] = token
                else:
                    arrays["wrong_neighbor_tokens.npy"][
                        center_index, donor_index
                    ] = token
            encoded_inputs += len(tokens)
            view_receipts.append(
                {
                    "center_index": center_index,
                    "sample_id": sample_id,
                    "scenario": selection[center_index]["scenario"],
                    "encoded_inputs": len(tokens),
                    "center_box": list(_box(center_row)),
                    "center_crop_box": list(center.raw_crop.geometry.crop_box),
                    "center_letterbox_resized": list(
                        center.raw_crop.geometry.resized_size
                    ),
                    "center_letterbox_padding": list(
                        center.raw_crop.geometry.padding
                    ),
                    "target_patch_counts": center.target_patch_masks.reshape(
                        2, -1
                    )
                    .sum(axis=1)
                    .astype(int)
                    .tolist(),
                    "visible_anchor_patch_counts": center.visible_patch_masks.reshape(
                        2, -1
                    )
                    .sum(axis=1)
                    .astype(int)
                    .tolist(),
                    "wrong_track_boxes": wrong_boxes,
                    "normalized_inputs_sha256": [
                        sha256_array(tensor.numpy()) for tensor in input_tensors
                    ],
                }
            )
            if (center_index + 1) % 16 == 0:
                for values in arrays.values():
                    values.flush()
                print(
                    json.dumps(
                        {
                            "completed_centers": center_index + 1,
                            "encoded_inputs": encoded_inputs,
                        }
                    ),
                    flush=True,
                )

    if encoded_inputs != 1392 or len(dependency_rows) != ROWS * 2:
        raise RuntimeError("Full-cache input or dependency population changed")
    if not np.all(arrays["wrong_neighbor_tokens.npy"][~arrays["wrong_slot_available.npy"]] == 0):
        raise RuntimeError("Unavailable wrong-track slots contain token values")
    if arrays["wrong_neighbor_valid.npy"][~arrays["wrong_slot_available.npy"]].any():
        raise RuntimeError("Unavailable wrong-track slots contain valid patches")
    for values in arrays.values():
        values.flush()
    for member, receipt in member_receipts.items():
        receipt["view_access_count"] = int(member_accesses[member])
    write_json_exclusive(output / "view_receipts.json", {"rows": view_receipts})
    write_json_exclusive(output / "member_receipts.json", {"members": member_receipts})
    write_json_exclusive(output / "dependency_audit.json", {"rows": dependency_rows})

    archive_after = archive_path.stat()
    if (archive_after.st_size, archive_after.st_mtime_ns) != (
        archive_before.st_size,
        archive_before.st_mtime_ns,
    ):
        raise RuntimeError("Provider archive changed during extraction")
    if any(
        sha256_file_stable(path)
        != source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")]
        for path in source_files
    ):
        raise RuntimeError("Executed source changed during extraction")

    artifacts = {}
    for name, (_, shape) in ARRAY_SPECS.items():
        path = output / name
        artifacts[name] = {
            "shape": list(shape),
            "dtype": str(arrays[name].dtype),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file_stable(path),
        }
    summary = {
        "status": "CENTER_EVIDENCE_COMPLETION_FULL_CACHE_COMPLETE",
        "scientific_gate_evaluated": False,
        "task_training_authorized": False,
        "centers": ROWS,
        "scenarios": len({row["scenario"] for row in selection}),
        "encoded_inputs": encoded_inputs,
        "encoded_input_breakdown": {
            "teacher_center": 128,
            "masked_centers": 256,
            "true_track_neighbors": 512,
            "wrong_track_controls": 496,
        },
        "missing_wrong_track_slots": int(
            (~arrays["wrong_slot_available.npy"]).sum()
        ),
        "source_members": {
            "unique_decoded": len(member_receipts),
            "decode_operations": ROWS * len(SOURCE_TIME_INDICES),
            "view_accesses": int(sum(member_accesses.values())),
        },
        "target_patch_count_range": [
            int(arrays["target_masks.npy"].reshape(ROWS * 2, -1).sum(axis=1).min()),
            int(arrays["target_masks.npy"].reshape(ROWS * 2, -1).sum(axis=1).max()),
        ],
        "artifacts": artifacts,
        "runtime_seconds": {
            "archive_hash": archive_hash_seconds,
            "prepare_and_dependency": prepare_seconds,
            "dino_encode": encode_seconds,
        },
        "cuda": {
            "device": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "model": {
            **model_receipt,
            "cache_representation": (
                "last_hidden_state[:,1:] reshaped row-major to 27x27x768"
            ),
        },
        "zero_access_counters": {
            "clip_index": 0,
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
        "next_authorization": (
            "independent cache/dependency replay, then the locked label-blind "
            "reconstruction screen only"
        ),
    }
    write_json_exclusive(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def main() -> None:
    args = parse_args()
    if args.validate_only:
        print(json.dumps(validate_only_receipt(), indent=2, sort_keys=True))
    else:
        run(args.output_dir)


if __name__ == "__main__":
    main()
