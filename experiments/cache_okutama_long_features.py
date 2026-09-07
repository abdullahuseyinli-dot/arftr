"""Cache frozen V-JEPA or DINO features for the locked two-second Okutama clips."""

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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.cache_okutama_dinov2_features import encode_native_clip  # noqa: E402
from experiments.cache_okutama_video_features import (  # noqa: E402
    _crop_with_mean_padding,
)
from hac.image_encoders import load_dinov2_encoder, validate_dinov2_lock_receipt  # noqa: E402
from hac.okutama_long_video import CENTER_SLOT, SOURCE_OFFSETS  # noqa: E402
from hac.video_encoders import (  # noqa: E402
    CHECKPOINT_SHA256,
    load_vjepa21_encoder,
    pool_spatiotemporal_tokens,
    preprocess_rgb_clip,
    sha256_file,
)

ENCODERS = {
    "vjepa21": {"arm": "vjepa21_long16_real_clip", "shape": (8, 9, 768)},
    "dinov2": {"arm": "dinov2_long16_native_frames", "shape": (16, 1, 768)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--extraction-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=tuple(ENCODERS), required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    return parser.parse_args()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError("Malformed locked P3 CSV header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError("Malformed locked P3 CSV rows")
    return rows


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
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
    output: Path, features: np.memmap, completed: np.ndarray, validity: np.ndarray
) -> None:
    features.flush()
    _array_atomic(output / "validity.npy", validity)
    _array_atomic(output / "completed.npy", completed)


def prepare_cache(
    output: Path,
    request: dict[str, Any],
    count: int,
    *,
    arm: str,
    shape: tuple[int, ...],
) -> tuple[np.memmap, np.ndarray, np.ndarray, str]:
    output.mkdir(parents=True, exist_ok=True)
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request_path = output / "request.json"
    feature_path = output / f"{arm}.npy"
    completed_path, validity_path = output / "completed.npy", output / "validity.npy"
    if request_path.exists():
        prior = json.loads(request_path.read_text(encoding="utf-8"))
        if prior.get("request_sha256") != request_hash:
            raise RuntimeError("P3 cache contains a different extraction request")
        if not all(path.is_file() for path in (feature_path, completed_path, validity_path)):
            raise RuntimeError("P3 cache initialization is incomplete; files retained")
        completed = np.load(completed_path, allow_pickle=False)
        validity = np.load(validity_path, allow_pickle=False)
        features = np.load(feature_path, allow_pickle=False, mmap_mode="r+")
        if (
            completed.shape != (count,)
            or completed.dtype != np.bool_
            or validity.shape != (count,)
            or validity.dtype != np.bool_
            or features.shape != (count, *shape)
            or features.dtype != np.float16
        ):
            raise RuntimeError("P3 resumable cache shape or dtype changed")
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                feature_path: summary.get("arms", {}).get(arm, {}).get("sha256"),
                completed_path: summary.get("completed", {}).get("sha256"),
                validity_path: summary.get("validity", {}).get("sha256"),
            }
            if summary.get("request_sha256") != request_hash or not completed.all():
                raise RuntimeError("Completed P3 summary disagrees with resume state")
            if any(sha256_file(path) != digest for path, digest in expected.items()):
                raise RuntimeError("Completed P3 cache artifact bytes changed")
        validity[~completed] = False
    else:
        if any(output.iterdir()):
            raise RuntimeError("Unrecognized P3 output files; refusing overwrite")
        completed = np.zeros(count, dtype=bool)
        validity = np.zeros(count, dtype=bool)
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float16, shape=(count, *shape)
        )
        features[:] = 0
        _save_state(output, features, completed, validity)
        _json_atomic(request_path, {**request, "request_sha256": request_hash})
    for start in range(0, count, 128):
        chosen = completed[start : start + 128] & validity[start : start + 128]
        if chosen.any() and not np.isfinite(features[start : start + 128][chosen]).all():
            raise RuntimeError("Completed P3 feature rows contain nonfinite values")
    return features, completed, validity, request_hash


class LongClipDataset(Dataset):
    """Decode only complete long clips; incomplete rows use later short-cache fallback."""

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
        sample_id = self.clips[index]["sample_id"]
        rows = self.frames_by_sample[sample_id]
        if len(rows) != 16 or [int(row["time_index"]) for row in rows] != list(range(16)):
            raise RuntimeError("Locked P3 clip no longer has ordered long16 rows")
        valid = np.asarray([row["valid_frame"].lower() in {"1", "true"} for row in rows])
        if not valid.all():
            return {
                "row_index": index,
                "pixels": torch.zeros((3, 16, 384, 384), dtype=torch.float32),
                "valid": False,
                "encoded_bytes": 0,
                "decoded_members": 0,
            }
        frames, encoded_bytes = [], 0
        for row in rows:
            member = row["image_member"]
            if member not in self.allowed_members:
                raise RuntimeError("P3 reader attempted a JPEG outside its exact allowlist")
            raw = self._zip().read(member)
            encoded_bytes += len(raw)
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                if source.size != (1280, 720) or source.format != "JPEG":
                    raise RuntimeError("P3 source image format or dimensions changed")
                image = source.convert("RGB")
            box = tuple(
                float(row[key]) for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
            )
            frames.append(_crop_with_mean_padding(image, box))
        return {
            "row_index": index,
            "pixels": preprocess_rgb_clip(frames),
            "valid": True,
            "encoded_bytes": encoded_bytes,
            "decoded_members": 16,
        }


def validate_and_select(args: argparse.Namespace):
    if args.workers < 0 or args.checkpoint_interval < 1:
        raise ValueError("workers must be nonnegative and checkpoint interval positive")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from lock_okutama_video_p3_extraction import validate_extraction_lock

    lock = validate_extraction_lock(root, args.extraction_lock.resolve())
    if lock.get("authorization") != {
        "frozen_extraction": True,
        "image_payload_access": True,
        "annotation_payload_access": False,
        "model_fitting": False,
        "probe_fitting": False,
        "backbone_fitting": False,
        "protected_data_access": False,
    }:
        raise RuntimeError("P3 lock does not grant only frozen extraction")
    artifacts = lock["manifest"]["artifacts"]
    for name in ("clip_index", "frame_manifest", "image_allowlist"):
        path = (args.manifest_dir / f"{name}.csv").resolve()
        receipt = artifacts[name]
        expected_path = (root / receipt["path"]).resolve()
        if path != expected_path or path.stat().st_size != receipt["size_bytes"] or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"P3 manifest artifact differs from extraction lock: {name}")
    clips = _csv(args.manifest_dir / "clip_index.csv")
    frames = _csv(args.manifest_dir / "frame_manifest.csv")
    allowlist = _csv(args.manifest_dir / "image_allowlist.csv")
    by_sample: dict[str, list[dict[str, str]]] = {}
    for row in frames:
        by_sample.setdefault(row["sample_id"], []).append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda row: int(row["time_index"]))
    if args.mode == "pilot":
        wanted = set(lock["manifest"]["pilot_sample_ids"])
        clips = [row for row in clips if row["sample_id"] in wanted]
        if len(clips) != 128 or {row["sample_id"] for row in clips} != wanted:
            raise RuntimeError("P3 pilot selection changed")
    elif len(clips) != 4977:
        raise RuntimeError("P3 full extraction must retain all original centers")
    return lock, clips, by_sample, {row["image_member"] for row in allowlist}


def _load_encoder(name: str, lock: dict[str, Any]):
    if name == "vjepa21":
        source = Path(lock["upstream"]["path"])
        checkpoint = Path(lock["inputs"]["checkpoint"]["path"])
        return load_vjepa21_encoder(
            source, checkpoint, checkpoint_sha256=CHECKPOINT_SHA256
        )
    snapshot = Path(lock["inputs"]["dinov2_snapshot"]["path"])
    validate_dinov2_lock_receipt(snapshot, lock["inputs"]["dinov2_snapshot"])
    return load_dinov2_encoder(snapshot)


def _encode(name: str, encoder, pixels: torch.Tensor) -> np.ndarray:
    if name == "dinov2":
        return encode_native_clip(encoder, pixels.squeeze(0))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense = encoder(pixels.to("cuda", non_blocking=True))
    pooled = pool_spatiotemporal_tokens(dense).to(torch.float16).cpu().numpy()[0]
    if pooled.shape != ENCODERS[name]["shape"] or not np.isfinite(pooled).all():
        raise RuntimeError("P3 V-JEPA feature shape, finiteness or float16 range changed")
    return pooled


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config not in (None, ":4096:8"):
        raise RuntimeError("P3 extraction requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("P3 extraction requires a CUDA BF16 device; no CPU fallback")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock, clips, frames, allowed = validate_and_select(args)
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root / ".runs" or not output.is_relative_to(root / ".runs"):
        raise RuntimeError("P3 cache output must use a dedicated directory below .runs")
    selected = ENCODERS[args.encoder]
    arm, shape = selected["arm"], selected["shape"]
    lock_hash = sha256_file(args.extraction_lock)
    shared = Path(__file__).with_name("cache_okutama_video_features.py")
    request = {
        "status": "OKUTAMA_P3_LONG_FROZEN_FEATURE_REQUEST",
        "mode": args.mode,
        "encoder": args.encoder,
        "arm": arm,
        "shape_per_clip": list(shape),
        "sample_ids_sha256": hashlib.sha256(
            json.dumps([row["sample_id"] for row in clips], separators=(",", ":")).encode()
        ).hexdigest(),
        "extraction_lock_sha256": lock_hash,
        "extractor_sha256": sha256_file(Path(__file__)),
        "shared_crop_extractor_sha256": sha256_file(shared),
        "precision": "bfloat16",
        "cache_dtype": "float16",
        "source_offsets": list(SOURCE_OFFSETS),
        "center_slot": CENTER_SLOT,
        "missing_policy": "incomplete_long_clip_zero_invalid;downstream_short_anchor_separately_locked",
    }
    features, completed, validity, request_hash = prepare_cache(
        output, request, len(clips), arm=arm, shape=shape
    )
    pending = np.flatnonzero(~completed)
    pending_clips = [clips[int(index)] for index in pending]
    dataset = LongClipDataset(
        Path(lock["inputs"]["archive"]["path"]), pending_clips, frames, allowed
    )
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
    processed = encoded_bytes = decoded_members = 0
    model_receipt = None
    if len(pending):
        encoder, model_receipt = _load_encoder(args.encoder, lock)
        _json_atomic(
            output / "model_receipt.json",
            {"request_sha256": request_hash, "model": model_receipt},
        )
        for batch in tqdm(loader, desc=f"P3 {args.encoder} {args.mode}", unit="clip"):
            target = int(pending[int(batch["row_index"].item())])
            valid = bool(batch["valid"].item())
            features[target] = _encode(args.encoder, encoder, batch["pixels"]) if valid else 0
            validity[target] = valid
            completed[target] = True
            encoded_bytes += int(batch["encoded_bytes"].item())
            decoded_members += int(batch["decoded_members"].item())
            processed += 1
            if processed % args.checkpoint_interval == 0:
                _save_state(output, features, completed, validity)
        torch.cuda.synchronize()
    _save_state(output, features, completed, validity)
    if not completed.all():
        raise RuntimeError("P3 frozen extraction ended with incomplete rows")
    expected_valid = (
        sum(
            all(row["valid_frame"].lower() in {"1", "true"} for row in frames[clip["sample_id"]])
            for clip in clips
        )
    )
    if int(validity.sum()) != expected_valid:
        raise RuntimeError("P3 cache validity differs from its locked manifest")
    invalid = ~validity
    if invalid.any() and np.any(features[invalid] != 0):
        raise RuntimeError("P3 invalid feature rows must remain exact zero sentinels")
    if (
        sha256_file(args.extraction_lock) != lock_hash
        or sha256_file(Path(__file__)) != request["extractor_sha256"]
        or sha256_file(shared) != request["shared_crop_extractor_sha256"]
    ):
        raise RuntimeError("P3 extraction code or lock changed during computation")
    if model_receipt is None:
        receipt_path = output / "model_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError("Completed P3 cache lacks model provenance")
        retained = json.loads(receipt_path.read_text(encoding="utf-8"))
        if retained.get("request_sha256") != request_hash:
            raise RuntimeError("P3 model provenance belongs to another request")
        model_receipt = retained["model"]
    elapsed = time.perf_counter() - started
    feature_path = output / f"{arm}.npy"
    summary = {
        "status": "OKUTAMA_P3_LONG_FROZEN_FEATURE_CACHE_COMPLETE",
        "mode": args.mode,
        "encoder": args.encoder,
        "rows": len(clips),
        "all_primary_rows": args.mode == "full" and len(clips) == 4977,
        "sample_ids": [row["sample_id"] for row in clips],
        "arms": {
            arm: {
                "path": feature_path.name,
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "valid_rows": int(validity.sum()),
                "invalid_rows": int((~validity).sum()),
                "sha256": sha256_file(feature_path),
            }
        },
        "validity": {"path": "validity.npy", "sha256": sha256_file(output / "validity.npy")},
        "completed": {"path": "completed.npy", "sha256": sha256_file(output / "completed.npy")},
        "request_sha256": request_hash,
        "extraction_lock_sha256": lock_hash,
        "model": model_receipt,
        "processed_clips_this_invocation": processed,
        "valid_clips_encoded_this_invocation": decoded_members // 16,
        "image_payloads_read_this_invocation": decoded_members,
        "encoded_member_bytes_this_invocation": encoded_bytes,
        "runtime_seconds_this_invocation": elapsed,
        "clips_per_second_this_invocation": processed / elapsed if elapsed else 0.0,
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
