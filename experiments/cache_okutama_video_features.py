"""Cache frozen V-JEPA 2.1-B features for the locked Okutama native clips.

This stage never fits a task model.  It reads only JPEG members named by the
role-safe extraction lock and retains every selected clip in manifest order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from hac.okutama_native_video import CENTER_SLOT, SOURCE_OFFSETS, crop_geometry
from hac.video_encoders import (
    CHECKPOINT_SHA256,
    PADDING_RGB,
    load_vjepa21_encoder,
    pool_spatiotemporal_tokens,
    preprocess_rgb_clip,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--extraction-lock", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    return parser.parse_args()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"Malformed CSV header: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError(f"Malformed CSV rows: {path}")
    return rows


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def _save_state(
    output: Path,
    arrays: dict[str, np.memmap],
    completed: np.ndarray,
    validity: np.ndarray,
) -> None:
    for values in arrays.values():
        values.flush()
    # A crash can leave a row safely eligible for recomputation, but must never
    # publish completion before the corresponding validity state is durable.
    _array_atomic(output / "validity.npy", validity)
    _array_atomic(output / "completed.npy", completed)


def _prepare_cache(
    output: Path,
    request: dict[str, Any],
    count: int,
) -> tuple[dict[str, np.memmap], np.ndarray, np.ndarray, str]:
    output.mkdir(parents=True, exist_ok=True)
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request_path = output / "request.json"
    feature_paths = {
        "vjepa21_real_clip": output / "vjepa21_real_clip.npy",
        "vjepa21_repeated_center": output / "vjepa21_repeated_center.npy",
    }
    completed_path, validity_path = output / "completed.npy", output / "validity.npy"
    if request_path.exists():
        prior = json.loads(request_path.read_text(encoding="utf-8"))
        if prior.get("request_sha256") != request_hash:
            raise RuntimeError("Output directory contains a different extraction request")
        required = (*feature_paths.values(), completed_path, validity_path)
        if not all(path.is_file() for path in required):
            raise RuntimeError("Video cache initialization is incomplete; existing files preserved")
        completed = np.load(completed_path, allow_pickle=False)
        validity = np.load(validity_path, allow_pickle=False)
        arrays = {
            name: np.load(path, allow_pickle=False, mmap_mode="r+")
            for name, path in feature_paths.items()
        }
        if (
            completed.shape != (count,)
            or completed.dtype != np.bool_
            or validity.shape != (count, 2)
            or validity.dtype != np.bool_
            or any(
                values.shape != (count, 8, 9, 768) or values.dtype != np.float16
                for values in arrays.values()
            )
        ):
            raise RuntimeError("Resumable video feature output shape or dtype changed")
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("request_sha256") != request_hash or not completed.all():
                raise RuntimeError("Completed video summary disagrees with its resume state")
            expected = {
                **{
                    feature_paths[name]: summary.get("arms", {}).get(name, {}).get("sha256")
                    for name in feature_paths
                },
                completed_path: summary.get("completed", {}).get("sha256"),
                validity_path: summary.get("validity", {}).get("sha256"),
            }
            if any(sha256_file(path) != value for path, value in expected.items()):
                raise RuntimeError("Completed video artifact bytes changed")
        # Recover conservatively if interrupted after validity but before completion.
        validity[~completed] = False
    else:
        if any(output.iterdir()):
            raise RuntimeError("Unrecognized files in video output directory; refusing overwrite")
        completed = np.zeros(count, dtype=bool)
        validity = np.zeros((count, 2), dtype=bool)
        arrays = {
            name: np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(count, 8, 9, 768)
            )
            for name, path in feature_paths.items()
        }
        for values in arrays.values():
            values[:] = 0
        _save_state(output, arrays, completed, validity)
        _json_atomic(request_path, {**request, "request_sha256": request_hash})
    for start in range(0, count, 128):
        for index, values in enumerate(arrays.values()):
            selected = completed[start : start + 128] & validity[start : start + 128, index]
            if selected.any() and not np.isfinite(values[start : start + 128][selected]).all():
                raise RuntimeError("Completed video cache rows contain nonfinite features")
    return arrays, completed, validity, request_hash


def _crop_with_mean_padding(
    image: Image.Image, box: tuple[float, float, float, float]
) -> Image.Image:
    geometry = crop_geometry(box, output_size=384, context_fraction=0.25)
    left, top, right, bottom = geometry.crop_box
    canvas = Image.new("RGB", (right - left, bottom - top), PADDING_RGB)
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        raise RuntimeError("Locked actor crop does not intersect the source image")
    canvas.paste(image.crop(source_box), (source_box[0] - left, source_box[1] - top))
    return canvas


class LockedClipDataset(Dataset):
    def __init__(
        self,
        archive: Path,
        clips: list[dict[str, str]],
        frames_by_sample: dict[str, list[dict[str, str]]],
        allowed_members: set[str],
    ) -> None:
        self.archive_path = archive
        self.clips = clips
        self.frames_by_sample = frames_by_sample
        self.allowed_members = allowed_members
        self._archive: zipfile.ZipFile | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_archive"] = None
        return state

    def _zip(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        return self._archive

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clips[index]
        sample_id = clip["sample_id"]
        rows = self.frames_by_sample[sample_id]
        if len(rows) != len(SOURCE_OFFSETS):
            raise RuntimeError("A locked clip no longer contains sixteen frame rows")
        valid = np.asarray([row["valid_frame"] in {"1", "True", "true"} for row in rows])
        center_valid = bool(valid[CENTER_SLOT])
        decoded: list[Image.Image | None] = []
        encoded_bytes = 0
        for row, is_valid in zip(rows, valid, strict=True):
            if not is_valid:
                decoded.append(None)
                continue
            member = row["image_member"]
            if member not in self.allowed_members:
                raise RuntimeError("Frame reader attempted a member outside the extraction lock")
            raw = self._zip().read(member)
            encoded_bytes += len(raw)
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                if source.size != (1280, 720) or source.format != "JPEG":
                    raise RuntimeError("Allowlisted source image format or dimensions changed")
                image = source.convert("RGB")
            box = tuple(
                float(row[key]) for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
            )
            decoded.append(_crop_with_mean_padding(image, box))
        if center_valid:
            center = decoded[CENTER_SLOT]
            assert center is not None
        else:
            center = None
        actual_valid = bool(valid.all())
        if actual_valid:
            actual_frames = [frame for frame in decoded if frame is not None]
            actual_pixels = preprocess_rgb_clip(actual_frames)
        else:
            actual_pixels = torch.zeros((3, len(SOURCE_OFFSETS), 384, 384), dtype=torch.float32)
        if center is not None:
            repeated_pixels = preprocess_rgb_clip([center] * len(SOURCE_OFFSETS))
        else:
            repeated_pixels = torch.zeros_like(actual_pixels)
        return {
            "row_index": index,
            "actual_pixels": actual_pixels,
            "repeated_pixels": repeated_pixels,
            "actual_valid": actual_valid,
            "repeated_valid": center_valid,
            "encoded_bytes": encoded_bytes,
        }


def _validate_and_select(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, list[dict[str, str]]], set[str]]:
    if args.workers < 0 or args.checkpoint_interval < 1:
        raise ValueError("workers must be nonnegative and checkpoint interval positive")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from lock_okutama_video_protocol import validate_extraction_lock

    lock = validate_extraction_lock(root, args.extraction_lock.resolve())
    if lock.get("authorization", {}).get("frozen_extraction") is not True:
        raise RuntimeError("Extraction lock does not authorize frozen feature computation")
    if lock.get("authorization", {}).get("model_fitting") is not False:
        raise RuntimeError("P0 extraction lock unexpectedly authorizes model fitting")
    expected_inputs = lock["inputs"]
    for name, path in (("archive", args.archive), ("checkpoint", args.checkpoint)):
        receipt = expected_inputs[name]
        if Path(receipt["path"]).resolve() != path.resolve():
            raise RuntimeError(f"{name} path differs from the extraction lock")
        if path.stat().st_size != receipt["size_bytes"] or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"{name} bytes differ from the extraction lock")
    if Path(lock["upstream"]["path"]).resolve() != args.source_root.resolve():
        raise RuntimeError("Upstream source path differs from the extraction lock")
    artifact_receipts = lock["manifest"]["artifacts"]
    for name in ("clip_index", "frame_manifest", "image_allowlist"):
        path = args.manifest_dir / f"{name}.csv"
        receipt = artifact_receipts[name]
        if (
            path.resolve() != (root / receipt["path"]).resolve()
            or sha256_file(path) != receipt["sha256"]
        ):
            raise RuntimeError(f"Manifest artifact differs from extraction lock: {name}")
    clips = _csv(args.manifest_dir / "clip_index.csv")
    frames = _csv(args.manifest_dir / "frame_manifest.csv")
    allowlist = _csv(args.manifest_dir / "image_allowlist.csv")
    by_sample: dict[str, list[dict[str, str]]] = {}
    for row in frames:
        by_sample.setdefault(row["sample_id"], []).append(row)
    for values in by_sample.values():
        values.sort(key=lambda row: int(row["time_index"]))
    if args.mode == "pilot":
        wanted = set(lock["pilot"]["sample_ids"])
        clips = [row for row in clips if row["sample_id"] in wanted]
        if len(clips) != lock["pilot"]["rows"] or {row["sample_id"] for row in clips} != wanted:
            raise RuntimeError("Label-blind pilot selection differs from the extraction lock")
    elif len(clips) != lock["primary_rows"]:
        raise RuntimeError("Full extraction must retain every locked primary center")
    return lock, clips, by_sample, {row["image_member"] for row in allowlist}


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config not in (None, ":4096:8"):
        raise RuntimeError("Locked extraction requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("Locked extraction requires CUDA; no CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Locked extraction requires CUDA BF16 support")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock, clips, frames_by_sample, allowed = _validate_and_select(args)
    output = args.output_dir.resolve()
    lock_hash = sha256_file(args.extraction_lock)
    encoder_module = Path(__file__).resolve().parents[1] / "src/hac/video_encoders.py"
    request = {
        "status": "OKUTAMA_VIDEO_FROZEN_FEATURE_REQUEST",
        "mode": args.mode,
        "sample_ids_sha256": hashlib.sha256(
            json.dumps([row["sample_id"] for row in clips], separators=(",", ":")).encode()
        ).hexdigest(),
        "extraction_lock_sha256": lock_hash,
        "extractor_sha256": sha256_file(Path(__file__)),
        "encoder_module_sha256": sha256_file(encoder_module),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "arms": ["vjepa21_real_clip", "vjepa21_repeated_center"],
        "precision": "bfloat16",
        "cache_dtype": "float16",
    }
    count = len(clips)
    feature_paths = {
        "vjepa21_real_clip": output / "vjepa21_real_clip.npy",
        "vjepa21_repeated_center": output / "vjepa21_repeated_center.npy",
    }
    completed_path, validity_path = output / "completed.npy", output / "validity.npy"
    arrays, completed, validity, request_hash = _prepare_cache(output, request, count)
    pending = np.flatnonzero(~completed)
    pending_clips = [clips[int(index)] for index in pending]
    pending_frames = {row["sample_id"]: frames_by_sample[row["sample_id"]] for row in pending_clips}
    dataset = LockedClipDataset(args.archive.resolve(), pending_clips, pending_frames, allowed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoded_bytes = 0
    processed = 0
    model_receipt = None
    if len(pending):
        encoder, model_receipt = load_vjepa21_encoder(
            args.source_root.resolve(),
            args.checkpoint.resolve(),
            checkpoint_sha256=CHECKPOINT_SHA256,
        )
        _json_atomic(
            output / "model_receipt.json",
            {"request_sha256": request_hash, "model": model_receipt},
        )
        for batch in tqdm(loader, desc=f"V-JEPA {args.mode}", unit="clip"):
            target_index = int(pending[int(batch["row_index"].item())])
            actual_valid = bool(batch["actual_valid"].item())
            repeated_valid = bool(batch["repeated_valid"].item())
            inputs = []
            names = []
            if actual_valid:
                inputs.append(batch["actual_pixels"].squeeze(0))
                names.append("vjepa21_real_clip")
            if repeated_valid:
                inputs.append(batch["repeated_pixels"].squeeze(0))
                names.append("vjepa21_repeated_center")
            if inputs:
                pixels = torch.stack(inputs).to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dense = encoder(pixels)
                pooled = pool_spatiotemporal_tokens(dense).to(torch.float16).cpu().numpy()
                for local, name in enumerate(names):
                    if not np.isfinite(pooled[local]).all():
                        raise RuntimeError("V-JEPA features overflowed the float16 cache")
                    arrays[name][target_index] = pooled[local]
            validity[target_index] = (actual_valid, repeated_valid)
            completed[target_index] = True
            encoded_bytes += int(batch["encoded_bytes"].item())
            processed += 1
            if processed % args.checkpoint_interval == 0:
                _save_state(output, arrays, completed, validity)
        torch.cuda.synchronize()
    _save_state(output, arrays, completed, validity)
    if not completed.all():
        raise RuntimeError("Frozen extraction ended with incomplete clips")
    if (
        sha256_file(args.extraction_lock) != lock_hash
        or sha256_file(Path(__file__)) != request["extractor_sha256"]
        or sha256_file(encoder_module) != request["encoder_module_sha256"]
    ):
        raise RuntimeError("Locked V-JEPA extraction inputs changed during computation")
    if model_receipt is None:
        receipt_path = output / "model_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError("Completed V-JEPA extraction is missing its model provenance")
        prior_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prior_receipt.get("request_sha256") != request_hash:
            raise RuntimeError("V-JEPA model provenance belongs to a different request")
        model_receipt = prior_receipt["model"]
    elapsed = time.perf_counter() - started
    summary = {
        "status": "OKUTAMA_VIDEO_FROZEN_FEATURE_CACHE_COMPLETE",
        "mode": args.mode,
        "rows": count,
        "all_primary_rows": args.mode == "full" and count == 4977,
        "sample_ids": [row["sample_id"] for row in clips],
        "arms": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "valid_rows": int(validity[:, index].sum()),
                "path": feature_paths[name].name,
                "sha256": sha256_file(feature_paths[name]),
            }
            for index, (name, value) in enumerate(arrays.items())
        },
        "validity": {"path": validity_path.name, "sha256": sha256_file(validity_path)},
        "completed": {"path": completed_path.name, "sha256": sha256_file(completed_path)},
        "request_sha256": request_hash,
        "extraction_lock_sha256": sha256_file(args.extraction_lock),
        "model": model_receipt,
        "encoded_member_bytes_this_invocation": encoded_bytes,
        "runtime_seconds_this_invocation": elapsed,
        "clips_per_second_this_invocation": processed / elapsed if elapsed else 0.0,
        "processed_clips_this_invocation": processed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device": torch.cuda.get_device_name(),
        "candidate_fits": 0,
        "labels_used_for_fitting_or_selection": 0,
        "protected_rows_read": 0,
    }
    _json_atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
