"""Immutable, chunk-resumable full-cohort dense DINO center extraction; zero fits."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import pilot_okutama_sear_dense_dino as pilot  # noqa: E402
from hac.video_encoders import sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("okutama_sear_dense_dino_full_protocol.json")
SELF_HASH_AT_IMPORT = sha256_file(Path(__file__))
ARRAYS = {
    "patch_tokens": (np.dtype("float16"), (729, 768)),
    "center_cls": (np.dtype("float16"), (768,)),
    "local_valid": (np.dtype("bool"), ()),
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def payload_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "cohort_rows": 4977,
        "historical_short_fallback_rows": 467,
        "source_condition": "historical_supplied_720_jpeg",
        "shape_per_row": [729, 768],
        "center_cls_shape_per_row": [768],
        "cache_dtype": "float16",
        "precision": "bfloat16",
        "image_size": 384,
        "patch_size": 14,
        "center_slot": 8,
        "microbatch": 4,
        "commit_chunk_rows": 64,
        "cls_max_absolute_difference": 0.0,
        "final_batch_policy": "repeat_last_valid_center_to_exactly_four_inputs;discard_padding_outputs",
    }
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Full dense protocol changes its frozen input/replay contract")
    if protocol.get("authorization") != {
        "full_frozen_center_extraction": True,
        "classifier_fitting": False,
        "optimizer_steps": False,
        "backbone_updates": False,
        "protected_payload_access": False,
        "external_downloads": False,
    }:
        raise RuntimeError("Full dense protocol authorizes only frozen extraction")


def validate_pilot_receipt(protocol: dict[str, Any]) -> dict[str, Any]:
    entry = protocol["verified_pilot"]
    for key in ("source", "protocol", "summary"):
        if sha256_file(ROOT / entry[key]) != entry[f"{key}_sha256"]:
            raise RuntimeError(f"Verified pilot {key} changed")
    receipt = json.loads((ROOT / entry["summary"]).read_text(encoding="utf-8"))
    if (
        receipt["status"] != "DENSE_DINO_CENTER_FEASIBILITY_COMPLETE"
        or receipt["rows"] != 32
        or receipt["historical_cls_replayed_rows"] != 32
        or receipt["historical_cls_max_abs"] != 0.0
        or receipt["pixel_replay_exact"] is not True
        or receipt["dense_center_only_cls_replay_exact"] is not True
        or receipt["classifier_fits"] != 0
    ):
        raise RuntimeError("Pilot does not establish the required exact replay gates")
    pilot_protocol = json.loads((ROOT / entry["protocol"]).read_text(encoding="utf-8"))
    pilot.validate_protocol(pilot_protocol)
    return pilot_protocol


def chunk_ranges(count: int, chunk_rows: int):
    if count < 1 or chunk_rows < 1:
        raise ValueError("Cache count and chunk size must be positive")
    return [(start, min(count, start + chunk_rows)) for start in range(0, count, chunk_rows)]


def receipt_name(start: int, stop: int) -> str:
    return f"chunk_{start:06d}_{stop:06d}.json"


def check_directory_members(output: Path, count: int, chunk_rows: int) -> None:
    files = {
        "execution_lock.json",
        "sample_ids.npy",
        "historical_short_fallback.npy",
        "source_frames.npy",
        "completed.npy",
        "completed.npy.tmp",
        "progress.json",
        "progress.json.tmp",
        "summary.json",
        "summary.json.tmp",
        "model_receipt.json",
        "model_receipt.json.tmp",
        "preflight.json",
        "preflight.json.tmp",
        *[f"{name}.npy" for name in ARRAYS],
    }
    directories = {"chunks", "source_snapshots"}
    for path in output.iterdir():
        if (path.is_dir() and path.name not in directories) or (
            path.is_file() and path.name not in files
        ):
            raise RuntimeError(f"Foreign full-cache artifact preserved: {path.name}")
    chunks = output / "chunks"
    if chunks.exists():
        allowed = {receipt_name(a, b) for a, b in chunk_ranges(count, chunk_rows)}
        allowed |= {name + ".tmp" for name in allowed}
        if any(not p.is_file() or p.name not in allowed for p in chunks.iterdir()):
            raise RuntimeError("Foreign chunk artifact preserved")


def array_atomic(path: Path, values: np.ndarray) -> None:
    pilot.historical._array_atomic(path, values)


def prepare_cache(
    output: Path,
    request: dict[str, Any],
    ids: np.ndarray,
    fallback: np.ndarray,
    frames: np.ndarray,
) -> tuple[dict[str, np.memmap], np.ndarray]:
    count, chunk_rows = len(ids), int(request["chunk_rows"])
    if ids.ndim != 1 or len(set(ids.tolist())) != count or ids.dtype.kind != "U":
        raise ValueError("Full cache requires unique ordered Unicode IDs")
    if (
        fallback.shape != (count,)
        or fallback.dtype != np.bool_
        or frames.shape != (count,)
        or frames.dtype != np.int64
    ):
        raise ValueError("Full cache metadata shapes/dtypes changed")
    output.mkdir(parents=True, exist_ok=True)
    request_hash = canonical_hash(request)
    lock_path = output / "execution_lock.json"
    arrays = {}
    if lock_path.exists():
        check_directory_members(output, count, chunk_rows)
        prior = json.loads(lock_path.read_text(encoding="utf-8"))
        if prior != {**request, "request_sha256": request_hash}:
            raise RuntimeError("Full cache contains a different immutable request")
        for name, values in {
            "sample_ids": ids,
            "historical_short_fallback": fallback,
            "source_frames": frames,
        }.items():
            saved = np.load(output / f"{name}.npy", allow_pickle=False)
            if saved.dtype != values.dtype or not np.array_equal(saved, values):
                raise RuntimeError(f"Full cache identity metadata changed: {name}")
        for name, (dtype, shape) in ARRAYS.items():
            arrays[name] = np.load(output / f"{name}.npy", allow_pickle=False, mmap_mode="r+")
            if arrays[name].shape != (count, *shape) or arrays[name].dtype != dtype:
                raise RuntimeError(f"Full cache shape/dtype changed: {name}")
        completed = np.load(output / "completed.npy", allow_pickle=False)
        if completed.shape != (count,) or completed.dtype != np.bool_:
            raise RuntimeError("Full cache completion bitmap changed shape/dtype")
    else:
        if any(output.iterdir()):
            raise RuntimeError("Partial foreign initialization preserved; refusing overwrite")
        for name, (dtype, shape) in ARRAYS.items():
            arrays[name] = np.lib.format.open_memmap(
                output / f"{name}.npy", mode="w+", dtype=dtype, shape=(count, *shape)
            )
            arrays[name].flush()
        for name, values in {
            "sample_ids": ids,
            "historical_short_fallback": fallback,
            "source_frames": frames,
        }.items():
            array_atomic(output / f"{name}.npy", values)
        completed = np.zeros(count, bool)
        array_atomic(output / "completed.npy", completed)
        (output / "chunks").mkdir()
        (output / "source_snapshots").mkdir()
        for source, digest in request["loaded_source_hashes"].items():
            target = output / "source_snapshots" / f"{digest[:12]}_{Path(source).name}"
            shutil.copyfile(source, target)
            if sha256_file(target) != digest:
                raise RuntimeError("Full cache executed-source snapshot changed")
        pilot._json_atomic(lock_path, {**request, "request_sha256": request_hash})
    expected_snapshots = {
        f"{h[:12]}_{Path(p).name}": h for p, h in request["loaded_source_hashes"].items()
    }
    actual_snapshots = list((output / "source_snapshots").iterdir())
    if {p.name for p in actual_snapshots} != set(expected_snapshots) or any(
        not p.is_file() or sha256_file(p) != expected_snapshots[p.name] for p in actual_snapshots
    ):
        raise RuntimeError("Full cache executed-source snapshots differ")
    recovered = np.zeros(count, bool)
    for start, stop in chunk_ranges(count, chunk_rows):
        receipt_path = output / "chunks" / receipt_name(start, stop)
        if not receipt_path.exists():
            if completed[start:stop].any():
                raise RuntimeError("Completion bit exists without a committed chunk receipt")
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("request_sha256") != request_hash
            or receipt.get("start") != start
            or receipt.get("stop") != stop
            or receipt.get("exact_historical_cls_replay") is not True
            or receipt.get("sample_ids_sha256") != payload_hash(ids[start:stop])
        ):
            raise RuntimeError("Committed chunk receipt identity differs")
        for name, values in arrays.items():
            if payload_hash(values[start:stop]) != receipt["payload_sha256"].get(name):
                raise RuntimeError("Committed chunk bytes changed")
            if not np.isfinite(values[start:stop]).all():
                raise RuntimeError("Committed chunk contains nonfinite values")
        recovered[start:stop] = True
    if not np.array_equal(completed, recovered):
        # Receipt is committed before bitmap: a crash in between is recoverable.
        completed = recovered
        array_atomic(output / "completed.npy", completed)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not completed.all() or summary.get("request_sha256") != request_hash:
            raise RuntimeError("Completed summary disagrees with immutable cache")
        for name, entry in summary["artifacts"].items():
            path = output / name
            if (
                not path.is_file()
                or path.stat().st_size != entry["size_bytes"]
                or sha256_file(path) != entry["sha256"]
            ):
                raise RuntimeError("Completed full-cache artifact hash changed")
    return arrays, completed


def commit_chunk(
    output: Path,
    request: dict[str, Any],
    arrays,
    completed,
    ids,
    start: int,
    stop: int,
    statistics: dict[str, Any],
) -> None:
    if (start, stop) not in chunk_ranges(len(ids), int(request["chunk_rows"])):
        raise ValueError("Commit must cover exactly one declared complete chunk")
    path = output / "chunks" / receipt_name(start, stop)
    if path.exists() or completed[start:stop].any():
        raise RuntimeError("Refusing to overwrite an immutable committed chunk")
    for values in arrays.values():
        if not np.isfinite(values[start:stop]).all():
            raise RuntimeError("Cannot commit nonfinite full-cache values")
        values.flush()
    receipt = {
        "request_sha256": canonical_hash(request),
        "start": start,
        "stop": stop,
        "sample_ids_sha256": payload_hash(ids[start:stop]),
        "payload_sha256": {
            name: payload_hash(values[start:stop]) for name, values in arrays.items()
        },
        "exact_historical_cls_replay": True,
        "cls_max_absolute_difference": 0.0,
        **statistics,
    }
    pilot._json_atomic(path, receipt)
    completed[start:stop] = True
    array_atomic(output / "completed.npy", completed)


def padded_center_batch(pixels: list[torch.Tensor], microbatch: int = 4) -> torch.Tensor:
    if not 0 < len(pixels) <= microbatch or microbatch != 4:
        raise ValueError("Center batch must have one to four inputs")
    return torch.stack(pixels + [pixels[-1]] * (microbatch - len(pixels)))


def run(protocol_path: Path, output: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    protocol_hash = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    pilot_protocol = validate_pilot_receipt(protocol)
    inputs = pilot.verified_inputs(pilot_protocol)
    args, old_lock, clips, frame_rows, allowed, snapshot, old_cls, old_valid, fallback, _ = inputs
    if not old_valid.all():
        raise RuntimeError("This full-cache contract requires all original short CLS targets valid")
    ids = np.asarray([row["sample_id"] for row in clips])
    centers = [frame_rows[str(sample_id)][8] for sample_id in ids]
    frames = np.asarray([int(row["center_frame"]) for row in clips], dtype=np.int64)
    for i, row in enumerate(centers):
        if (
            int(row["time_index"]) != 8
            or int(row["source_frame"]) != frames[i]
            or row["valid_frame"].lower() not in {"1", "true"}
        ):
            raise RuntimeError("An immutable center frame/validity differs from replay contract")
    sources = {
        **pilot.IMPORT_SOURCE_HASHES,
        str(Path(__file__).resolve()): SELF_HASH_AT_IMPORT,
        str(protocol_path.resolve()): protocol_hash,
        str((ROOT / protocol["verified_pilot"]["protocol"]).resolve()): protocol["verified_pilot"][
            "protocol_sha256"
        ],
    }
    if any(sha256_file(Path(p)) != h for p, h in sources.items()):
        raise RuntimeError("Full-cache executable/protocol changed after import")
    request = {
        "status": "FULL_DENSE_CENTER_REQUEST_BEFORE_DECODING",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "source_condition": protocol["source_condition"],
        "loaded_source_hashes": sources,
        "historical_extraction_lock_sha256": sha256_file(args.extraction_lock),
        "model_snapshot": snapshot,
        "rows": len(ids),
        "chunk_rows": protocol["commit_chunk_rows"],
        "sample_ids_sha256": payload_hash(ids),
        "source_frames_sha256": payload_hash(frames),
        "historical_short_fallback_sha256": payload_hash(fallback),
        "missing_policy": "all4977historical_short_centers_valid;halt_if_current_validity_changes;no_dropped_rows",
        "classifier_fits": 0,
        "backbone_updates": 0,
    }
    arrays, completed = prepare_cache(output, request, ids, fallback, frames)
    if preflight_only:
        receipt = {
            "status": "FULL_DENSE_PREFLIGHT_COMPLETE_NO_ENCODER_OR_CUDA_USE",
            "request_sha256": canonical_hash(request),
            "execution_lock_sha256": sha256_file(output / "execution_lock.json"),
            "rows": len(ids),
            "historical_short_fallback_rows": int(fallback.sum()),
            "historical_cls_valid_rows": int(old_valid.sum()),
            "patch_shape": list(arrays["patch_tokens"].shape),
            "cls_shape": list(arrays["center_cls"].shape),
            "completed_rows": int(completed.sum()),
            "encoder_loaded": False,
            "cuda_used": False,
            "image_payloads_decoded": 0,
            "classifier_fits": 0,
        }
        receipt_path = output / "preflight.json"
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior["request_sha256"] != receipt["request_sha256"]:
                raise RuntimeError("Previous full-cache preflight belongs to a different request")
        else:
            pilot._json_atomic(receipt_path, receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        return receipt
    if completed.all() and (output / "summary.json").exists():
        print(
            "Full dense cache already complete; hashes and chunk receipts reverified, no model loaded",
            flush=True,
        )
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("Full cache requires deterministic CUBLAS configuration")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full dense cache requires the pinned CUDA BF16 environment")
    started = time.perf_counter()
    encoder, model_receipt = pilot.load_dinov2_encoder(args.model_root)
    model_path = output / "model_receipt.json"
    expected_model = {"request_sha256": canonical_hash(request), "model": model_receipt}
    if model_path.exists():
        if json.loads(model_path.read_text(encoding="utf-8")) != expected_model:
            raise RuntimeError("Full cache resumed with different model provenance")
    else:
        pilot._json_atomic(model_path, expected_model)
    torch.cuda.reset_peak_memory_stats()
    processed = 0
    initially_completed = int(completed.sum())
    with zipfile.ZipFile(args.archive) as archive:
        for start, stop in chunk_ranges(len(ids), request["chunk_rows"]):
            if completed[start:stop].all():
                continue
            chunk_started = time.perf_counter()
            decode_seconds, encode_seconds, encoded_bytes = 0.0, 0.0, 0
            for first in range(start, stop, protocol["microbatch"]):
                last = min(stop, first + protocol["microbatch"])
                tick = time.perf_counter()
                pixels = []
                for i in range(first, last):
                    image, valid, nbytes = pilot.center_pixels(archive, centers[i], allowed)
                    if not valid:
                        raise RuntimeError(
                            "Historical valid center became unavailable; row retained, run halted"
                        )
                    pixels.append(image)
                    encoded_bytes += nbytes
                decode_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                cls, patches = pilot.dense_forward(encoder, padded_center_batch(pixels))
                size = last - first
                pilot.assert_exact_replay(
                    cls[:size],
                    old_cls[first:last, 8, 0],
                    description=f"full dense center CLS rows{first}:{last}",
                )
                arrays["center_cls"][first:last] = cls[:size]
                arrays["patch_tokens"][first:last] = patches[:size].reshape(size, 729, 768)
                arrays["local_valid"][first:last] = True
                encode_seconds += time.perf_counter() - tick
            stats = {
                "center_jpeg_payloads_decoded": stop - start,
                "center_jpeg_bytes": encoded_bytes,
                "center_decode_seconds": decode_seconds,
                "encode_replay_and_memmap_assignment_seconds": encode_seconds,
                "compute_seconds_before_flush_and_receipt": time.perf_counter() - chunk_started,
            }
            commit_chunk(output, request, arrays, completed, ids, start, stop, stats)
            processed += stop - start
            elapsed = time.perf_counter() - started
            progress = {
                "completed_rows": int(completed.sum()),
                "total_rows": len(ids),
                "processed_this_invocation": processed,
                "elapsed_seconds_this_invocation": elapsed,
                "rows_per_second_this_invocation": processed / elapsed,
                "eta_seconds": (len(ids) - completed.sum()) * elapsed / processed,
                "last_committed_chunk": [start, stop],
                "classifier_fits": 0,
            }
            progress["eta_seconds"] = float(progress["eta_seconds"])
            pilot._json_atomic(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    torch.cuda.synchronize()
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    if not completed.all() or not arrays["local_valid"].all():
        raise RuntimeError("Full cache did not preserve every valid center")
    if any(sha256_file(Path(p)) != h for p, h in sources.items()):
        raise RuntimeError("Frozen full-cache executable changed during extraction")
    if sha256_file(args.extraction_lock) != request["historical_extraction_lock_sha256"]:
        raise RuntimeError("Historical source lock changed during extraction")
    pilot.validate_dinov2_lock_receipt(args.model_root, old_lock["inputs"]["dinov2_snapshot"])
    # Re-open and verify every committed chunk independently of the extraction loop.
    _, verified_completed = prepare_cache(output, request, ids, fallback, frames)
    if not verified_completed.all():
        raise RuntimeError("Independent chunk verification found missing centers")
    whole_cls_difference = pilot.assert_exact_replay(
        arrays["center_cls"], old_cls[:, 8, 0], description="complete4977centerCLS"
    )
    chunk_stats = [
        json.loads((output / "chunks" / receipt_name(a, b)).read_text(encoding="utf-8"))
        for a, b in chunk_ranges(len(ids), request["chunk_rows"])
    ]
    artifact_paths = [p for p in output.iterdir() if p.is_file() and not p.name.endswith(".tmp")]
    artifact_paths += list((output / "chunks").glob("*.json"))
    artifact_paths += list((output / "source_snapshots").iterdir())
    artifacts = {
        p.relative_to(output).as_posix(): {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        for p in artifact_paths
    }
    summary = {
        "status": "FULL_DENSE_DINO_CENTER_CACHE_COMPLETE",
        "rows": len(ids),
        "all_primary_rows": True,
        "sample_ids": ids.tolist(),
        "source_condition": protocol["source_condition"],
        "request_sha256": canonical_hash(request),
        "execution_lock_sha256": sha256_file(output / "execution_lock.json"),
        "loaded_source_hashes": sources,
        "artifacts": artifacts,
        "patch_tokens": {
            "path": "patch_tokens.npy",
            "shape": list(arrays["patch_tokens"].shape),
            "dtype": "float16",
        },
        "center_cls": {
            "path": "center_cls.npy",
            "shape": list(arrays["center_cls"].shape),
            "dtype": "float16",
        },
        "completed_rows": int(completed.sum()),
        "valid_local_rows": int(arrays["local_valid"].sum()),
        "historical_short_fallback_rows": int(fallback.sum()),
        "center_cls_max_absolute_difference": whole_cls_difference,
        "historical_cls_replay_exact_all4977": True,
        "verified_immutable_chunks": len(chunk_stats),
        "center_jpeg_payloads_decoded_total": sum(
            x["center_jpeg_payloads_decoded"] for x in chunk_stats
        ),
        "center_jpeg_bytes_total": sum(x["center_jpeg_bytes"] for x in chunk_stats),
        "center_decode_seconds_total": sum(x["center_decode_seconds"] for x in chunk_stats),
        "encode_replay_and_memmap_assignment_seconds_total": sum(
            x["encode_replay_and_memmap_assignment_seconds"] for x in chunk_stats
        ),
        "elapsed_seconds_this_invocation_including_model_chunk_verification_and_final_hashing": time.perf_counter()
        - started,
        "timing_caveat": "excludes initial immutable input validation and cache initialization; OS page-cache condition uncontrolled",
        "initially_completed_rows": initially_completed,
        "processed_rows_this_invocation": processed,
        "peak_cuda_allocated_bytes": peak,
        "peak_cuda_reserved_bytes": reserved,
        "device": torch.cuda.get_device_name(),
        "classifier_fits": 0,
        "optimizer_steps": 0,
        "backbone_updates": 0,
        "labels_used_for_fitting_or_selection": 0,
        "protected_payloads_read": 0,
        "decoded_frames_per_center": 1,
    }
    pilot._json_atomic(output / "summary.json", summary)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"sample_ids", "artifacts", "loaded_source_hashes"}
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output_dir.resolve(), preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
