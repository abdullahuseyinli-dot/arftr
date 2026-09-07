"""Label-blind synthetic smoke for the pinned frozen V-JEPA 2.1-B encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hac.video_encoders import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    load_vjepa21_encoder,
    pool_spatiotemporal_tokens,
    preprocess_rgb_clip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--source-commit", choices=[SOURCE_COMMIT], default=SOURCE_COMMIT)
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.add_argument("--frames", type=int, choices=[16], default=16)
    parser.add_argument("--size", type=int, choices=[384], default=384)
    parser.add_argument("--batch-size", type=int, choices=[1], default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def synthetic_clip(frames: int = 16) -> list[np.ndarray]:
    """Deterministic coloured pattern with motion; no target data or random I/O."""
    y, x = np.indices((96, 48))
    return [
        np.stack(((x * 5 + t * 7) % 256, (y * 2) % 256, ((x + y + t) * 3) % 256), axis=-1).astype(
            np.uint8
        )
        for t in range(frames)
    ]


def run(args: argparse.Namespace) -> dict:
    if args.output is not None and args.output.exists():
        raise FileExistsError("Smoke output already exists; choose a fresh output path")
    if torch.cuda.is_initialized():
        raise RuntimeError("Smoke must configure determinism before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Synthetic video smoke requires CUDA; no CPU fallback")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The requested BF16 precision is not supported")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    encoder, provenance = load_vjepa21_encoder(
        args.source_root, args.checkpoint, checkpoint_sha256=args.checkpoint_sha256
    )
    pixels = preprocess_rgb_clip(synthetic_clip(args.frames), size=args.size).unsqueeze(0)
    input_sha256 = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
    pixels = pixels.to("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    timings = []
    retained = None
    pooled = None
    max_difference = 0.0
    with torch.inference_mode():
        for repetition in range(3):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype, enabled=args.dtype != "float32"):
                dense = encoder(pixels)
            pooled = pool_spatiotemporal_tokens(dense, frames=args.frames, image_size=args.size)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if repetition:
                timings.append(elapsed)
                if retained is None:
                    retained = dense.detach().clone()
                else:
                    max_difference = float((dense.float() - retained.float()).abs().max())
                    if not torch.equal(dense, retained):
                        raise RuntimeError(
                            f"Same-process encoder repeat was not exact: {max_difference}"
                        )
    assert pooled is not None and retained is not None
    return {
        "status": "SYNTHETIC_VJEPA21_ENCODER_SMOKE_PASSED",
        "synthetic_only": True,
        "target_images_read": 0,
        "target_labels_read": 0,
        "candidate_fits": 0,
        "source_commit_argument": args.source_commit,
        "provenance": provenance,
        "input_shape": list(pixels.shape),
        "input_sha256": input_sha256,
        "preprocessing": {
            "policy": "aspect_preserving_centered_padding_PIL_bilinear",
            "padding_rgb": [124, 116, 104],
            "normalization": "ImageNet",
            "upstream_center_crop_reproduced": False,
        },
        "dense_shape": list(retained.shape),
        "pooled_shape": list(pooled.shape),
        "pooling": "3x3_equal_area_spatial_means_per_tubelet_float32",
        "pooled_sha256": hashlib.sha256(pooled.cpu().numpy().tobytes()).hexdigest(),
        "precision": args.dtype,
        "repeat_max_abs_difference": max_difference,
        "same_process_bit_exact": True,
        "warmup_passes": 1,
        "measured_forward_and_pool_seconds": timings,
        "total_seconds_including_integrity_checks": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_runtime": torch.version.cuda,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resource_scope": "encoder_load_plus_synthetic_preprocess_forward_pool_and_repeat_comparison",
        "cuda_determinism": {
            "algorithms": "strict",
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "tf32": False,
        },
    }


def main() -> None:
    args = parse_args()
    summary = run(args)
    serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
