"""Cache DINOv2 frame CLS features from exactly the locked native video crops."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.cache_okutama_video_features import (  # noqa: E402
    LockedClipDataset,
    _json_atomic,
    _validate_and_select,
)
from hac.image_encoders import (  # noqa: E402
    DINO_REVISION,
    load_dinov2_encoder,
    validate_dinov2_lock_receipt,
)
from hac.video_encoders import sha256_file  # noqa: E402

ARM = "dinov2_native_frames"
CACHE_SHAPE = (16, 1, 768)
FRAME_MICROBATCH = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "manifest-dir", "extraction-lock", "model-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace):
    # Shared validation checks the materialization lock, extraction lock, live sources,
    # manifest hashes and exact pilot/cohort BEFORE the dataset opens any JPEG payload.
    declared = json.loads(args.extraction_lock.read_text(encoding="utf-8"))
    shared_args = copy.copy(args)
    shared_args.checkpoint = Path(declared["inputs"]["checkpoint"]["path"])
    shared_args.source_root = Path(declared["upstream"]["path"])
    lock, clips, frames, allowed = _validate_and_select(shared_args)
    receipt = validate_dinov2_lock_receipt(
        args.model_root.resolve(),
        lock["inputs"].get("dinov2_snapshot"),
    )
    return lock, clips, frames, allowed, receipt


def _array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def _save_state(output: Path, features, completed: np.ndarray, validity: np.ndarray) -> None:
    features.flush()
    # Commit validity before completion: an interrupted update can cause a safe repeat
    # of an unfinished extraction row, but cannot mark a stale validity row complete.
    _array_atomic(output / "validity.npy", validity)
    _array_atomic(output / "completed.npy", completed)


def prepare_cache(output: Path, request: dict[str, Any], count: int):
    output.mkdir(parents=True, exist_ok=True)
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request_path = output / "request.json"
    feature_path = output / f"{ARM}.npy"
    completed_path, validity_path = output / "completed.npy", output / "validity.npy"
    if request_path.exists():
        prior = json.loads(request_path.read_text(encoding="utf-8"))
        if prior.get("request_sha256") != request_hash:
            raise RuntimeError("DINO output contains a different extraction request")
        if not all(path.is_file() for path in (feature_path, completed_path, validity_path)):
            raise RuntimeError("DINO cache initialization is incomplete; existing files preserved")
        completed = np.load(completed_path, allow_pickle=False)
        validity = np.load(validity_path, allow_pickle=False)
        features = np.load(feature_path, allow_pickle=False, mmap_mode="r+")
        if (
            completed.shape != (count,)
            or completed.dtype != np.bool_
            or validity.shape != (count, 1)
            or validity.dtype != np.bool_
            or features.shape != (count, *CACHE_SHAPE)
            or features.dtype != np.float16
        ):
            raise RuntimeError("DINO resumable cache shape or dtype changed")
        summary_path = output / "summary.json"
        if summary_path.exists():
            prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if prior_summary.get("request_sha256") != request_hash or not completed.all():
                raise RuntimeError("Completed DINO summary disagrees with its resume state")
            expected_hashes = {
                feature_path: prior_summary.get("arms", {}).get(ARM, {}).get("sha256"),
                completed_path: prior_summary.get("completed", {}).get("sha256"),
                validity_path: prior_summary.get("validity", {}).get("sha256"),
            }
            if any(sha256_file(path) != digest for path, digest in expected_hashes.items()):
                raise RuntimeError("Completed DINO artifact bytes changed")
        if np.any(validity[:, 0] & ~completed):
            # This is an interrupted validity-before-completion checkpoint; recompute
            # those rows, retaining their stale feature bytes only until overwrite.
            validity[~completed] = False
    else:
        if any(output.iterdir()):
            raise RuntimeError("Unrecognized files in DINO output directory; refusing overwrite")
        completed = np.zeros(count, dtype=bool)
        validity = np.zeros((count, 1), dtype=bool)
        features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float16,
            shape=(count, *CACHE_SHAPE),
        )
        features[:] = 0
        _save_state(output, features, completed, validity)
        _json_atomic(request_path, {**request, "request_sha256": request_hash})
    for start in range(0, count, 128):
        selected = completed[start : start + 128] & validity[start : start + 128, 0]
        if selected.any() and not np.isfinite(features[start : start + 128][selected]).all():
            raise RuntimeError("Completed DINO cache rows contain nonfinite features")
    return features, completed, validity, request_hash


@torch.inference_mode()
def encode_native_clip(encoder, actual_pixels: torch.Tensor, *, device="cuda") -> np.ndarray:
    if tuple(actual_pixels.shape) != (3, 16, 384, 384):
        raise ValueError("DINO control requires the exact C,16,384,384 video input")
    frames = actual_pixels.permute(1, 0, 2, 3).contiguous()
    values = []
    for start in range(0, 16, FRAME_MICROBATCH):
        pixels = frames[start : start + FRAME_MICROBATCH].to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            cls = encoder(pixels)
        if cls.shape != (len(pixels), 768) or not torch.isfinite(cls).all():
            raise RuntimeError("DINO per-frame CLS features changed shape or finiteness")
        cached = cls.to(torch.float16).cpu().numpy()
        if not np.isfinite(cached).all():
            raise RuntimeError("DINO CLS values overflowed the float16 cache")
        values.append(cached)
    return np.concatenate(values, axis=0)[:, None, :]


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config not in (None, ":4096:8"):
        raise RuntimeError("DINO extraction requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("DINO extraction requires CUDA; no CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("DINO extraction requires CUDA BF16 support")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock, clips, frames, allowed, snapshot = validate_inputs(args)
    output = args.output_dir.resolve()
    lock_hash = sha256_file(args.extraction_lock)
    shared_extractor = Path(__file__).with_name("cache_okutama_video_features.py")
    image_module = Path(__file__).resolve().parents[1] / "src/hac/image_encoders.py"
    request = {
        "status": "OKUTAMA_DINOV2_FROZEN_FEATURE_REQUEST",
        "mode": args.mode,
        "sample_ids_sha256": hashlib.sha256(
            json.dumps([row["sample_id"] for row in clips], separators=(",", ":")).encode()
        ).hexdigest(),
        "extraction_lock_sha256": lock_hash,
        "extractor_sha256": sha256_file(Path(__file__)),
        "shared_video_extractor_sha256": sha256_file(shared_extractor),
        "image_encoder_module_sha256": sha256_file(image_module),
        "snapshot": snapshot,
        "revision": DINO_REVISION,
        "arms": [ARM],
        "precision": "bfloat16",
        "cache_dtype": "float16",
        "frame_microbatch": FRAME_MICROBATCH,
        "shape_per_clip": list(CACHE_SHAPE),
        "input_transform": "identical_LockedClipDataset_actual_pixels",
        "missing_policy": "whole_clip_zero_invalid;preserve_center_for_locked_downstream_fallback",
    }
    features, completed, validity, request_hash = prepare_cache(output, request, len(clips))
    pending = np.flatnonzero(~completed)
    processed, encoded_bytes = 0, 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model_receipt = None
    if len(pending):
        encoder, model_receipt = load_dinov2_encoder(args.model_root.resolve())
        _json_atomic(
            output / "model_receipt.json",
            {
                "request_sha256": request_hash,
                "model": model_receipt,
            },
        )
        pending_clips = [clips[int(index)] for index in pending]
        dataset = LockedClipDataset(args.archive.resolve(), pending_clips, frames, allowed)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for batch in tqdm(loader, desc=f"DINO {args.mode}", unit="clip"):
            target = int(pending[int(batch["row_index"].item())])
            valid = bool(batch["actual_valid"].item())
            features[target] = (
                encode_native_clip(encoder, batch["actual_pixels"].squeeze(0)) if valid else 0
            )
            validity[target, 0] = valid
            completed[target] = True
            encoded_bytes += int(batch["encoded_bytes"].item())
            processed += 1
            if processed % args.checkpoint_interval == 0:
                _save_state(output, features, completed, validity)
        torch.cuda.synchronize()
    _save_state(output, features, completed, validity)
    if not completed.all():
        raise RuntimeError("DINO extraction ended with incomplete rows")
    if (
        sha256_file(args.extraction_lock) != lock_hash
        or sha256_file(shared_extractor) != request["shared_video_extractor_sha256"]
        or sha256_file(image_module) != request["image_encoder_module_sha256"]
        or sha256_file(Path(__file__)) != request["extractor_sha256"]
    ):
        raise RuntimeError("Locked DINO extraction inputs changed during computation")
    validate_dinov2_lock_receipt(args.model_root.resolve(), lock["inputs"]["dinov2_snapshot"])
    elapsed = time.perf_counter() - started
    if model_receipt is None:
        receipt_path = output / "model_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError("Completed DINO extraction is missing its model provenance")
        prior_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prior_receipt.get("request_sha256") != request_hash:
            raise RuntimeError("DINO model provenance belongs to a different request")
        model_receipt = prior_receipt["model"]
    summary = {
        "status": "OKUTAMA_DINOV2_FROZEN_FEATURE_CACHE_COMPLETE",
        "mode": args.mode,
        "rows": len(clips),
        "all_primary_rows": args.mode == "full" and len(clips) == 4977,
        "sample_ids": [row["sample_id"] for row in clips],
        "arms": {
            ARM: {
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "valid_rows": int(validity.sum()),
                "path": f"{ARM}.npy",
                "sha256": sha256_file(output / f"{ARM}.npy"),
            }
        },
        "validity": {"path": "validity.npy", "sha256": sha256_file(output / "validity.npy")},
        "completed": {"path": "completed.npy", "sha256": sha256_file(output / "completed.npy")},
        "request_sha256": request_hash,
        "extraction_lock_sha256": lock_hash,
        "model": model_receipt,
        "encoded_member_bytes_this_invocation": encoded_bytes,
        "runtime_seconds_this_invocation": elapsed,
        "clips_per_second_this_invocation": processed / elapsed if elapsed else 0,
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
