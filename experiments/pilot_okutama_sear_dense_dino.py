"""Label-blind dense DINO center pilot with exact historical CLS and pixel replay.

This utility deliberately has no full-cache or classifier-training mode. The small
ordinary heads are resource surrogates, not frozen definitions of the final arms.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import cache_okutama_dinov2_features as historical  # noqa: E402
from experiments.cache_okutama_video_features import (  # noqa: E402
    LockedClipDataset,
    _crop_with_mean_padding,
    _csv,
    _json_atomic,
)
from hac.image_encoders import (  # noqa: E402
    DINO_REVISION,
    load_dinov2_encoder,
    validate_dinov2_lock_receipt,
)
from hac.video_encoders import preprocess_rgb_clip, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("okutama_sear_dense_dino_protocol.json")
SOURCE_FILES = [
    Path(__file__),
    Path(historical.__file__),
    ROOT / "experiments/cache_okutama_video_features.py",
    ROOT / "src/hac/image_encoders.py",
    ROOT / "src/hac/video_encoders.py",
    ROOT / "src/hac/okutama_native_video.py",
]
IMPORT_SOURCE_HASHES = {str(p.resolve()): sha256_file(p) for p in SOURCE_FILES}


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "source_condition": "historical_supplied_720_jpeg",
        "cohort_rows": 4977,
        "historical_long_fallback_rows": 467,
        "pilot_rows": 32,
        "center_slot": 8,
        "image_size": 384,
        "patch_size": 14,
        "patch_grid": [27, 27],
        "feature_dim": 768,
        "precision": "bfloat16",
        "cache_dtype": "float16",
        "frame_microbatch": 4,
        "cls_replay_max_absolute_difference": 0.0,
    }
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Dense pilot protocol changes the frozen extraction/replay contract")
    auth = protocol.get("authorization", {})
    if auth != {
        "frozen_pilot_extraction": True,
        "synthetic_backward_resource_probe": True,
        "full_cache_extraction": False,
        "classifier_fitting": False,
        "backbone_fitting": False,
        "protected_payload_access": False,
        "external_downloads": False,
    }:
        raise RuntimeError("Dense pilot authorization must exclude full extraction and fitting")
    if protocol.get("selection") != (
        "one_sha256_minimum_per_scenario_and_long_fallback_stratum_then_global_sha256_order"
    ):
        raise RuntimeError("Dense pilot selection rule changed")
    resources = protocol["ordinary_head_benchmarks"]
    if (
        resources["optimizer_steps"] != 0
        or resources["video_dim"] != 1536
        or resources["rank"] != 32
        or resources["batches"] != [8, 32]
        or resources["warmup_steps"] != 1
        or resources["measured_steps"] != 3
    ):
        raise RuntimeError("Resource pilot is bounded and cannot perform optimizer steps")


def select_label_blind(
    sample_ids: list[str], scenarios: list[str], short_fallback: np.ndarray, count: int, salt: str
) -> np.ndarray:
    if (
        len(sample_ids) != len(set(sample_ids))
        or len(scenarios) != len(sample_ids)
        or short_fallback.shape != (len(sample_ids),)
        or short_fallback.dtype != np.bool_
        or not 0 < count <= len(sample_ids)
    ):
        raise ValueError("Pilot IDs, strata, fallback mask or count are invalid")
    order = sorted(
        range(len(sample_ids)),
        key=lambda i: hashlib.sha256(f"{salt}|{sample_ids[i]}".encode()).hexdigest(),
    )
    strata: dict[tuple[str, bool], int] = {}
    for i in order:
        strata.setdefault((scenarios[i], bool(short_fallback[i])), i)
    if len(strata) > count:
        raise ValueError("Pilot budget cannot cover every scenario/fallback stratum")
    chosen = set(strata.values())
    for i in order:
        if len(chosen) == count:
            break
        chosen.add(i)
    return np.asarray(sorted(chosen), dtype=np.int64)


def center_pixels(
    archive: zipfile.ZipFile, row: dict[str, str], allowed: set[str]
) -> tuple[torch.Tensor, bool, int]:
    if row["valid_frame"].lower() not in {"1", "true"}:
        return torch.zeros(3, 384, 384), False, 0
    member = row["image_member"]
    if member not in allowed:
        raise RuntimeError("Dense center attempted a JPEG outside the original allowlist")
    raw = archive.read(member)
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        if source.size != (1280, 720) or source.format != "JPEG":
            raise RuntimeError("Dense center source dimensions or format changed")
        rgb = source.convert("RGB")
    box = tuple(float(row[k]) for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
    crop = _crop_with_mean_padding(rgb, box)
    # Shared preprocessing requires an even clip length. Duplicate only for this
    # per-image transform, then discard the second identical normalized image.
    pixels = preprocess_rgb_clip([crop, crop])[:, 0].contiguous()
    return pixels, True, len(raw)


@torch.inference_mode()
def dense_forward(encoder, pixels: torch.Tensor, *, device: str = "cuda"):
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 384, 384):
        raise ValueError("Dense DINO requires B,3,384,384")
    if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
        raise ValueError("Dense DINO pixels must be finite normalized floats")
    if encoder.training or encoder.backbone.training:
        raise RuntimeError("Dense DINO must remain frozen in evaluation mode")
    pixels = pixels.to(device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        output = encoder.backbone(pixel_values=pixels, return_dict=True).last_hidden_state
    if output.shape != (len(pixels), 730, 768) or not torch.isfinite(output).all():
        raise RuntimeError("Dense DINO must emit finite CLS plus 27x27x768 tokens")
    values = output.to(torch.float16).cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("Dense DINO values overflowed float16")
    return values[:, 0], values[:, 1:].reshape(-1, 27, 27, 768)


def assert_exact_replay(actual: np.ndarray, expected: np.ndarray, *, description: str) -> float:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f"{description}: replay shape/dtype changed")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"{description}: nonfinite replay")
    difference = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{description}: exact replay failed, max absolute difference {difference}"
        )
    return difference


class OrdinaryDenseResourceHead(nn.Module):
    """Provisional capacity/activation surrogate; not a classifier training arm."""

    def __init__(self, kind: str):
        super().__init__()
        if kind not in {"cnn", "transformer"}:
            raise ValueError("Unknown resource surrogate")
        self.kind = kind
        self.project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 32))
        if kind == "cnn":
            self.spatial = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GELU(),
            )
            self.local = nn.Linear(64, 3)
        else:
            self.position = nn.Linear(2, 32, bias=False)
            self.spatial = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(32, 4, 128, 0.1, batch_first=True),
                2,
                enable_nested_tensor=False,
            )
            self.local = nn.Linear(32, 3)
        axis = torch.linspace(-1, 1, 27)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy), -1).reshape(729, 2))
        # Same proposed video-branch architecture, copied locally only to avoid
        # importing the independently changing SEAR implementation during this run.
        self.video = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, patches: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        features = self.project(patches)
        if self.kind == "cnn":
            features = self.spatial(features.transpose(1, 2).reshape(-1, 32, 27, 27))
            pooled = features.mean((-1, -2))
        else:
            pooled = self.spatial(features + self.position(self.coordinates)).mean(1)
        return self.local(pooled) + self.video(video)


def benchmark_ordinary_heads(tokens: np.ndarray, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    config = protocol["ordinary_head_benchmarks"]
    outputs = []
    torch.manual_seed(42)
    for kind in ("cnn", "transformer"):
        for batch_size in config["batches"]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = OrdinaryDenseResourceHead(kind).cuda().train()
            x = torch.from_numpy(tokens[np.arange(batch_size) % len(tokens)].copy())
            x = x.reshape(batch_size, 729, 768).cuda().float()
            video = torch.randn(batch_size, config["video_dim"], device="cuda")
            start = None
            for step in range(config["warmup_steps"] + config["measured_steps"]):
                if step == config["warmup_steps"]:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x, video)
                    loss = logits.float().square().mean()
                loss.backward()
                if not torch.isfinite(logits).all() or not all(
                    p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise RuntimeError("Ordinary resource head has nonfinite/missing gradients")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated()
            outputs.append(
                {
                    "kind": kind,
                    "batch_size": batch_size,
                    "active_parameters": sum(p.numel() for p in model.parameters()),
                    "measured_backward_steps": config["measured_steps"],
                    "seconds_per_forward_backward": elapsed / config["measured_steps"],
                    "peak_cuda_allocated_bytes": peak,
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "within_4gib_budget": peak < config["maximum_peak_allocated_bytes"],
                    "optimizer_steps": 0,
                    "action_labels_used": 0,
                    "all_parameters_have_finite_gradients": True,
                }
            )
            del model, x, video, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def verified_inputs(protocol: dict[str, Any]):
    paths = protocol["inputs"]
    extraction_lock = ROOT / paths["extraction_lock"]
    if sha256_file(extraction_lock) != paths["extraction_lock_sha256"]:
        raise RuntimeError("Historical extraction lock bytes changed")
    declared = json.loads(extraction_lock.read_text(encoding="utf-8"))
    args = argparse.Namespace(
        extraction_lock=extraction_lock,
        archive=Path(declared["inputs"]["archive"]["path"]),
        manifest_dir=ROOT / paths["manifest_dir"],
        model_root=Path(declared["inputs"]["dinov2_snapshot"]["path"]),
        mode="full",
        workers=0,
        checkpoint_interval=1,
    )
    # This new continuation has uncommitted authorized experiment files. Never
    # invoke or weaken the old clean-committed-worktree authorization gate. Bind
    # the immutable historical data and exact live preprocessing in a new receipt.
    lock = declared
    if lock["primary_rows"] != 4977 or lock["authorization"]["model_fitting"] is not False:
        raise RuntimeError("Historical lock population or fitting authorization differs")
    receipts = [lock["source_lock"], lock["inputs"]["archive"]]
    receipts += list(lock["manifest"]["artifacts"].values())
    receipts += [
        lock["sources"][key]
        for key in (
            "encoder_module",
            "extractor",
            "image_encoder_module",
            "image_extractor",
            "native_video_module",
        )
    ]
    for receipt in receipts:
        checked_path = ROOT / receipt["path"]
        if (
            checked_path.stat().st_size != receipt["size_bytes"]
            or sha256_file(checked_path) != receipt["sha256"]
        ):
            raise RuntimeError(f"Historical continuation input changed: {receipt['path']}")
    for name in ("clip_index", "frame_manifest", "image_allowlist"):
        if (args.manifest_dir / f"{name}.csv").resolve() != (
            ROOT / lock["manifest"]["artifacts"][name]["path"]
        ).resolve():
            raise RuntimeError("Continuation manifest path differs from immutable lock")
    clips = _csv(args.manifest_dir / "clip_index.csv")
    frame_rows = _csv(args.manifest_dir / "frame_manifest.csv")
    allowlist = _csv(args.manifest_dir / "image_allowlist.csv")
    allowed = {row["image_member"] for row in allowlist}
    if len(allowed) != lock["manifest"]["allowlist_rows"]:
        raise RuntimeError("Continuation allowlist population differs")
    frames: dict[str, list[dict[str, str]]] = {}
    for row in frame_rows:
        frames.setdefault(row["sample_id"], []).append(row)
        if row["valid_frame"].lower() in {"1", "true"} and row["image_member"] not in allowed:
            raise RuntimeError("Continuation valid frame is outside the immutable allowlist")
    ids = [row["sample_id"] for row in clips]
    if len(ids) != len(set(ids)) or set(frames) != set(ids):
        raise RuntimeError("Continuation clip/frame identities differ")
    if (
        len({row["recording_id"] for row in clips}) != lock["primary_scenarios"]
        or len({row["provider_recording_id"] for row in clips}) != 21
    ):
        raise RuntimeError("Continuation scenario allowlist differs")
    for values in frames.values():
        values.sort(key=lambda row: int(row["time_index"]))
        if [int(row["time_index"]) for row in values] != list(range(16)):
            raise RuntimeError("Continuation short clip time order differs")
    snapshot = validate_dinov2_lock_receipt(args.model_root, lock["inputs"]["dinov2_snapshot"])
    cache_dir = ROOT / paths["historical_cache_dir"]
    summary_path = cache_dir / "summary.json"
    if sha256_file(summary_path) != paths["historical_summary_sha256"]:
        raise RuntimeError("Historical DINO summary changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ids = [row["sample_id"] for row in clips]
    if summary["sample_ids"] != ids or len(ids) != protocol["cohort_rows"]:
        raise RuntimeError("Historical DINO and original manifest identities differ")
    for entry in [summary["arms"][historical.ARM], summary["validity"], summary["completed"]]:
        if sha256_file(cache_dir / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Historical DINO cache artifact changed")
    valid = np.load(cache_dir / "validity.npy", allow_pickle=False)
    completed = np.load(cache_dir / "completed.npy", allow_pickle=False)
    features = np.load(cache_dir / f"{historical.ARM}.npy", allow_pickle=False, mmap_mode="r")
    if (
        valid.shape != (len(ids), 1)
        or valid.dtype != np.bool_
        or completed.shape != (len(ids),)
        or completed.dtype != np.bool_
        or not completed.all()
        or features.shape != (len(ids), 16, 1, 768)
        or features.dtype != np.float16
    ):
        raise RuntimeError("Historical DINO cache shape/dtype/completion differs")
    if sha256_file(ROOT / paths["long_validity"]) != paths["long_validity_sha256"]:
        raise RuntimeError("Historical long fallback mask changed")
    if sha256_file(ROOT / paths["long_summary"]) != paths["long_summary_sha256"]:
        raise RuntimeError("Historical long identity receipt changed")
    long_summary = json.loads((ROOT / paths["long_summary"]).read_text(encoding="utf-8"))
    long_valid = np.load(ROOT / paths["long_validity"], allow_pickle=False)
    if (
        long_summary["sample_ids"] != ids
        or long_valid.shape != (len(ids),)
        or long_valid.dtype != np.bool_
    ):
        raise RuntimeError("Long fallback identities or dtype differ")
    fallback = ~long_valid
    if fallback.sum() != protocol["historical_long_fallback_rows"]:
        raise RuntimeError("Historical fallback population changed")
    scenarios = [row["recording_id"] for row in clips]
    indices = select_label_blind(
        ids, scenarios, fallback, protocol["pilot_rows"], protocol["selection_salt"]
    )
    return args, lock, clips, frames, allowed, snapshot, features, valid, fallback, indices


def run(protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol_hash = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("Dense pilot requires deterministic CUBLAS configuration")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Dense pilot requires the existing CUDA BF16 environment")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Dense pilot output is not empty; existing receipts preserved")
    output.mkdir(parents=True, exist_ok=True)
    args, lock, clips, frames, allowed, snapshot, old, old_valid, fallback, indices = (
        verified_inputs(protocol)
    )
    if any(sha256_file(Path(p)) != h for p, h in IMPORT_SOURCE_HASHES.items()):
        raise RuntimeError("Pilot source changed after import")
    if sha256_file(protocol_path) != protocol_hash:
        raise RuntimeError("Pilot protocol changed during validation")
    selected = [clips[int(i)] for i in indices]
    ids = np.asarray([row["sample_id"] for row in selected])
    source_receipts = dict(IMPORT_SOURCE_HASHES)
    source_receipts[str(protocol_path.resolve())] = protocol_hash
    snapshots = output / "source_snapshots"
    snapshots.mkdir()
    for path, digest in source_receipts.items():
        target = snapshots / f"{digest[:12]}_{Path(path).name}"
        shutil.copyfile(path, target)
        if sha256_file(target) != digest:
            raise RuntimeError("Executable source snapshot mismatch")
    request = {
        "status": "DENSE_DINO_PILOT_REQUEST_FROZEN_BEFORE_DECODING",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "loaded_source_hashes": source_receipts,
        "source_lock_sha256": sha256_file(args.extraction_lock),
        "snapshot": snapshot,
        "revision": DINO_REVISION,
        "sample_ids": ids.tolist(),
        "cohort_indices": indices.tolist(),
        "selected_long_fallback": fallback[indices].tolist(),
        "selection_uses_action_labels": False,
        "continuation_validation": protocol["continuation_validation"],
        "protected_payloads_allowed": False,
        "classifier_fits": 0,
    }
    _json_atomic(output / "request.json", request)
    np.savez(
        output / "cohort_metadata.npz",
        sample_ids=np.asarray([x["sample_id"] for x in clips]),
        historical_short_fallback=fallback,
        selected_indices=indices,
    )
    started = time.perf_counter()
    encoder, model_receipt = load_dinov2_encoder(args.model_root)
    torch.cuda.reset_peak_memory_stats()
    dense_tokens = np.zeros((len(indices), 27, 27, 768), np.float16)
    cls_tokens = np.zeros((len(indices), 768), np.float16)
    local_valid = np.zeros(len(indices), dtype=bool)
    original_dataset = LockedClipDataset(args.archive, selected, frames, allowed)
    all_pixels = []
    replay = []
    decode_seconds = 0.0
    reference_seconds = 0.0
    decoded_center_bytes = 0
    with zipfile.ZipFile(args.archive) as archive:
        for j, row in enumerate(selected):
            center = frames[row["sample_id"]][protocol["center_slot"]]
            if int(center["time_index"]) != 8 or int(center["source_frame"]) != int(
                row["center_frame"]
            ):
                raise RuntimeError("Pilot center timestamp or slot changed")
            start = time.perf_counter()
            pixels, valid, nbytes = center_pixels(archive, center, allowed)
            decode_seconds += time.perf_counter() - start
            decoded_center_bytes += nbytes
            local_valid[j] = valid
            all_pixels.append(pixels)
            start = time.perf_counter()
            reference = original_dataset[j]
            if bool(reference["actual_valid"]) != bool(old_valid[int(indices[j]), 0]):
                raise RuntimeError("Historical short-clip validity replay changed")
            if valid != bool(reference["repeated_valid"]):
                raise RuntimeError("Historical center validity changed")
            if valid and not torch.equal(pixels, reference["repeated_pixels"][:, 8]):
                raise RuntimeError("Center preprocessing differs from historical pixels")
            if reference["actual_valid"]:
                if not torch.equal(pixels, reference["actual_pixels"][:, 8]):
                    raise RuntimeError("Center preprocessing differs from actual clip pixels")
                # Same original four-frame microbatch containing center slot8.
                batch = reference["actual_pixels"][:, 8:12].permute(1, 0, 2, 3).contiguous().cuda()
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    legacy_cls = encoder(batch).half().cpu().numpy()[0]
                maximum = assert_exact_replay(
                    legacy_cls, old[int(indices[j]), 8, 0], description="historical microbatch CLS"
                )
                replay.append({"sample_id": row["sample_id"], "historical_cls_max_abs": maximum})
                del batch, legacy_cls
            reference_seconds += time.perf_counter() - start
            print(f"pixel/CLS replay {j + 1}/{len(indices)}", flush=True)
    if original_dataset._archive is not None:
        original_dataset._archive.close()
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    valid_indices = np.flatnonzero(local_valid)
    for start in range(0, len(valid_indices), protocol["frame_microbatch"]):
        rows = valid_indices[start : start + protocol["frame_microbatch"]]
        pixels = torch.stack([all_pixels[int(j)] for j in rows])
        cls, dense = dense_forward(encoder, pixels)
        for k, j in enumerate(rows):
            cls_tokens[j] = cls[k]
            dense_tokens[j] = dense[k]
            if old_valid[int(indices[j]), 0]:
                assert_exact_replay(
                    cls[k], old[int(indices[j]), 8, 0], description="dense center-only CLS"
                )
    torch.cuda.synchronize()
    encoding_seconds = time.perf_counter() - encode_started
    encoder_peak = torch.cuda.max_memory_allocated()
    encoder_reserved = torch.cuda.max_memory_reserved()
    del encoder, original_dataset, all_pixels
    gc.collect()
    torch.cuda.empty_cache()
    bench = benchmark_ordinary_heads(dense_tokens[local_valid], protocol)
    for name, values in {
        "sample_ids": ids,
        "cohort_indices": indices,
        "historical_short_fallback": fallback[indices],
        "local_valid": local_valid,
        "cls": cls_tokens,
        "patches": dense_tokens,
    }.items():
        np.save(output / f"{name}.npy", values, allow_pickle=False)
    if any(sha256_file(Path(p)) != h for p, h in source_receipts.items()):
        raise RuntimeError("Frozen executable/protocol changed during pilot")
    validate_dinov2_lock_receipt(args.model_root, lock["inputs"]["dinov2_snapshot"])
    payload_seconds = decode_seconds + encoding_seconds
    summary = {
        "status": "DENSE_DINO_CENTER_FEASIBILITY_COMPLETE",
        "request_sha256": sha256_file(output / "request.json"),
        "rows": len(ids),
        "all_original_cohort_rows_preserved_in_metadata": len(clips),
        "selected_long_fallback_rows": int(fallback[indices].sum()),
        "cohort_long_fallback_rows": int(fallback.sum()),
        "local_valid_rows": int(local_valid.sum()),
        "scenario_counts": dict(collections.Counter(x["recording_id"] for x in selected)),
        "pixel_replay_exact": True,
        "historical_cls_replayed_rows": len(replay),
        "historical_cls_max_abs": max((x["historical_cls_max_abs"] for x in replay), default=None),
        "dense_center_only_cls_replay_exact": True,
        "tokens_shape": list(dense_tokens.shape),
        "cache_dtype": str(dense_tokens.dtype),
        "model_receipt": model_receipt,
        "center_decode_seconds": decode_seconds,
        "center_jpeg_bytes": decoded_center_bytes,
        "historical_pixel_and_microbatch_replay_seconds": reference_seconds,
        "dense_center_encoder_seconds": encoding_seconds,
        "center_decode_plus_encoder_seconds": payload_seconds,
        "center_decode_plus_encoder_rows_per_second": len(ids) / payload_seconds,
        "projected_4977_center_decode_plus_encoder_seconds": payload_seconds
        * len(clips)
        / len(ids),
        "projected_4977_raw_patch_bytes": len(clips) * 27 * 27 * 768 * 2,
        "projection_caveat": "32 cold-cache pilot rows; excludes validation, model load, full output writes and historical replay; not a measured full-run runtime",
        "encoder_peak_cuda_allocated_bytes": encoder_peak,
        "encoder_peak_cuda_reserved_bytes": encoder_reserved,
        "ordinary_resource_surrogates": bench,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "elapsed_seconds_including_replay_and_resource_checks": time.perf_counter() - started,
        "artifacts": {
            p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in output.iterdir()
            if p.is_file()
        },
        "classifier_fits": 0,
        "backbone_updates": 0,
        "optimizer_steps": 0,
        "labels_used_for_fitting_or_selection": 0,
        "protected_payloads_read": 0,
        "full_cache_started": False,
    }
    _json_atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
