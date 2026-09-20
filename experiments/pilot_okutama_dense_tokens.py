"""Frozen 3x3/12x12 V-JEPA pilot or all-center cache with exact short fallback."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.audit_okutama_source_views import (
    P3,
    ROOT,
    SOURCE_DIR,
    load_allowed_manifest,
    read_csv,
    select_pilot,
    write_json,
)
from experiments.cache_okutama_long_features import LongClipDataset
from experiments.cache_okutama_video_features import _crop_with_mean_padding
from hac.okutama_native_video import EXPECTED_RECORDINGS, crop_geometry
from hac.video_encoders import load_vjepa21_encoder, pool_spatiotemporal_tokens, sha256_file


def native4k_prepare(output: Path, alignment_path: Path, fidelity_path: Path) -> dict:
    """CPU-only exact-time pixel gate and matched physical crops for 64 centers."""
    import cv2
    from PIL import Image

    from hac.video_encoders import PADDING_RGB, preprocess_rgb_clip

    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if not alignment["all_recordings_validated"]:
        raise RuntimeError("Native feature pilot requires all-view held-frame timing validation")
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    if alignment["fidelity_summary_sha256"] != sha256_file(fidelity_path):
        raise RuntimeError("Timing map and native video lineage differ")
    models = {m["recording"]: m["selected_model"] for m in alignment["mappings"]}
    videos = {v["recording"]: v for v in fidelity["videos"]}
    lock, clips, frames, allowed = load_allowed_manifest()
    p0_path = ROOT / lock["p0_extraction_lock"]["path"]
    if sha256_file(p0_path) != lock["p0_extraction_lock"]["sha256"]:
        raise RuntimeError("Fallback input lock changed")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    for name in ("clip_index", "frame_manifest", "image_allowlist"):
        artifact = p0["manifest"]["artifacts"][name]
        if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("Fallback manifest changed")
    short = defaultdict(list)
    for row in read_csv(ROOT / p0["manifest"]["artifacts"]["frame_manifest"]["path"]):
        short[row["sample_id"]].append(row)
    allowed |= {
        r["image_member"]
        for r in read_csv(ROOT / p0["manifest"]["artifacts"]["image_allowlist"]["path"])
    }
    for clip in clips:
        if clip["all_frames_valid"].lower() not in {"1", "true"}:
            frames[clip["sample_id"]] = sorted(
                short[clip["sample_id"]], key=lambda r: int(r["time_index"])
            )
    groups = defaultdict(list)
    excluded_support = []
    for clip in clips:
        recording = clip["provider_recording_id"]
        model = models[recording]
        mapped = [
            round(model["scale"] * int(f["source_frame"]) + model["offset"])
            for f in frames[clip["sample_id"]]
        ]
        if min(mapped) < 0 or max(mapped) >= videos[recording]["frame_count"]:
            excluded_support.append(clip["sample_id"])
        else:
            groups[recording].append(clip)
    for rows in groups.values():
        rows.sort(
            key=lambda c: hashlib.sha256(
                ("hac-native64-20260908:" + c["sample_id"]).encode()
            ).digest()
        )
    selected = []
    for recording in sorted(EXPECTED_RECORDINGS):
        if len(groups[recording]) < 2:
            raise RuntimeError("A retained view lacks native pilot support")
        selected.extend(groups[recording][:2])
    selected_ids = {c["sample_id"] for c in selected}
    scenarios = sorted({c["recording_id"] for c in clips})
    while len(selected) < 64:
        for scenario in scenarios:
            remaining = [
                c
                for rows in groups.values()
                for c in rows
                if c["recording_id"] == scenario and c["sample_id"] not in selected_ids
            ]
            remaining.sort(
                key=lambda c: hashlib.sha256(
                    ("hac-native64-20260908:" + c["sample_id"]).encode()
                ).digest()
            )
            chosen = remaining[0]
            selected.append(chosen)
            selected_ids.add(chosen["sample_id"])
            if len(selected) == 64:
                break
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output / "pixels").mkdir()
    selection = [
        {
            **c,
            "short_fallback": c["all_frames_valid"].lower() not in {"1", "true"},
            "frames": frames[c["sample_id"]],
        }
        for c in selected
    ]
    write_json(output / "selection_before_image_access.json", selection)
    write_json(
        output / "request.json",
        {
            "alignment_sha256": sha256_file(alignment_path),
            "fidelity_sha256": sha256_file(fidelity_path),
            "extraction_lock_sha256": sha256_file(P3 / "extraction_lock.json"),
            "selection_rule": "two hash-selected centers per allowed recording, then scenario round robin to 64; pixel support only",
            "centers_excluded_for_out_of_range_native_support": excluded_support,
            "action_values_used": 0,
            "models_fitted": 0,
            "arms": ["existing720", "native4k", "exact4k_downsample720"],
        },
    )
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    started = time.perf_counter()
    receipts = []
    with zipfile.ZipFile(lock["inputs"]["archive"]["path"]) as archive:
        for recording in sorted(EXPECTED_RECORDINGS):
            path = SOURCE_DIR / "allowed_videos" / Path(videos[recording]["zip_member"]).name
            if sha256_file(path) != videos[recording]["sha256"]:
                raise RuntimeError("Native video bytes changed")
            cap = cv2.VideoCapture(str(path))
            for clip_index, clip in enumerate(selection):
                if clip["provider_recording_id"] != recording:
                    continue
                tick = time.perf_counter()
                model = models[recording]
                mapped = [
                    round(model["scale"] * int(f["source_frame"]) + model["offset"])
                    for f in clip["frames"]
                ]
                wanted = set(mapped)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(mapped))
                source_frames = {}
                for index in range(min(mapped), max(mapped) + 1):
                    ok, bgr = cap.read()
                    if not ok:
                        raise RuntimeError(f"Native source support absent: {recording}/{index}")
                    if index in wanted:
                        source_frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                native_crops, low_crops, checks = [], [], []
                for row, index in zip(clip["frames"], mapped, strict=True):
                    if row["image_member"] not in allowed:
                        raise RuntimeError("Exact frame image outside authorized allowlist")
                    raw = archive.read(row["image_member"])
                    with Image.open(io.BytesIO(raw)) as im:
                        old_rgb = np.asarray(im.convert("RGB"))
                    native = source_frames[index]
                    low = cv2.resize(native, (1280, 720), interpolation=cv2.INTER_AREA)
                    old_small = cv2.resize(old_rgb, (320, 180), interpolation=cv2.INTER_AREA)
                    low_small = cv2.resize(low, (320, 180), interpolation=cv2.INTER_AREA)
                    mae = float(
                        np.abs(old_small.astype(np.float32) - low_small.astype(np.float32)).mean()
                    )
                    checks.append(
                        {
                            "jpeg_frame": int(row["source_frame"]),
                            "native_index": index,
                            "mae320": mae,
                            "jpeg_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    )
                    box = tuple(
                        float(row[k]) for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
                    )
                    bounds = tuple(
                        3 * x
                        for x in crop_geometry(box, output_size=384, context_fraction=0.25).crop_box
                    )
                    left, top, right, bottom = bounds
                    tile = Image.new("RGB", (right - left, bottom - top), PADDING_RGB)
                    valid = (max(0, left), max(0, top), min(3840, right), min(2160, bottom))
                    tile.paste(
                        Image.fromarray(native).crop(valid), (valid[0] - left, valid[1] - top)
                    )
                    native_crops.append(tile)
                    low_crops.append(_crop_with_mean_padding(Image.fromarray(low), box))
                receipt = {
                    "clip_index": clip_index,
                    "sample_id": clip["sample_id"],
                    "recording": recording,
                    "frames": checks,
                    "passed": all(c["mae320"] < 6 for c in checks),
                }
                if receipt["passed"]:
                    for arm, crops in (
                        ("native4k", native_crops),
                        ("exact4k_downsample720", low_crops),
                    ):
                        np.save(
                            output / "pixels" / f"{clip_index}_{arm}.npy",
                            preprocess_rgb_clip(crops).numpy(),
                            allow_pickle=False,
                        )
                receipt["seconds"] = time.perf_counter() - tick
                receipts.append(receipt)
                write_json(output / "checks" / f"{clip_index}.json", receipt)
            cap.release()
            print(
                json.dumps(
                    {
                        "native_prepare_recording": recording,
                        "centers": len(receipts),
                        "passed": sum(r["passed"] for r in receipts),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    result = {
        "status": "NATIVE64_PIXEL_GATE_COMPLETE",
        "passed": all(r["passed"] for r in receipts),
        "centers": len(receipts),
        "frames": 16 * len(receipts),
        "short_fallback_centers": sum(c["short_fallback"] for c in selection),
        "elapsed_seconds": time.perf_counter() - started,
        "median_clip_seconds": float(np.median([r["seconds"] for r in receipts])),
        "max_frame_mae320": max(f["mae320"] for r in receipts for f in r["frames"]),
        "median_frame_mae320": float(
            np.median([f["mae320"] for r in receipts for f in r["frames"]])
        ),
        "models_loaded": 0,
        "selection_sha256": sha256_file(output / "selection_before_image_access.json"),
        "pixel_artifacts": {
            p.name: sha256_file(p) for p in sorted((output / "pixels").glob("*.npy"))
        },
        "protected_data_payloads_read": 0,
        "rows": receipts,
    }
    write_json(output / "pixel_gate.json", result)
    return result


def native4k_encode(output: Path) -> dict:
    """Encode only after every exact pilot input passes the CPU pixel gate."""
    gate = json.loads((output / "pixel_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["centers"] != 64 or gate["frames"] != 1024:
        raise RuntimeError("Native64 pixel gate did not pass")
    if sha256_file(output / "selection_before_image_access.json") != gate["selection_sha256"]:
        raise RuntimeError("Pixel-gated selection changed")
    for name, digest in gate["pixel_artifacts"].items():
        if sha256_file(output / "pixels" / name) != digest:
            raise RuntimeError("Pixel-gated native input bytes changed")
    feature_dir = output / "features"
    feature_dir.mkdir(exist_ok=False)
    (feature_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    selected = json.loads(
        (output / "selection_before_image_access.json").read_text(encoding="utf-8")
    )
    lock, all_clips, _, allowed = load_allowed_manifest()
    p0_path = ROOT / lock["p0_extraction_lock"]["path"]
    if sha256_file(p0_path) != lock["p0_extraction_lock"]["sha256"]:
        raise RuntimeError("Historical fallback input lock changed")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    artifact = p0["manifest"]["artifacts"]["image_allowlist"]
    if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
        raise RuntimeError("Short fallback image allowlist changed")
    allowed |= {r["image_member"] for r in read_csv(ROOT / artifact["path"])}
    retained_ids = {c["sample_id"] for c in all_clips}
    for clip in selected:
        if (
            clip["sample_id"] not in retained_ids
            or clip["provider_recording_id"] not in EXPECTED_RECORDINGS
            or any(r["image_member"] not in allowed for r in clip["frames"])
        ):
            raise RuntimeError("Native64 selection outside immutable allowed cohort")
    dataset = LongClipDataset(
        Path(lock["inputs"]["archive"]["path"]),
        selected,
        {c["sample_id"]: c["frames"] for c in selected},
        allowed,
    )
    dense_dir = ROOT / ".runs/research_20260908/dense_tokens_full"
    verification = json.loads((dense_dir / "verification.json").read_text(encoding="utf-8"))
    if verification["grid3_exact_rows"] != 4977:
        raise RuntimeError("Historical replay control missing")
    old = np.load(dense_dir / "tokens_grid3.npy", mmap_mode="r", allow_pickle=False)
    old_index = {c["sample_id"]: i for i, c in enumerate(all_clips)}
    arms = ("existing720", "native4k", "exact4k_downsample720")
    arrays = {
        (arm, grid): np.lib.format.open_memmap(
            feature_dir / f"{arm}_grid{grid}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(64, 8, grid * grid, 768),
        )
        for arm in arms
        for grid in (3, 12)
    }
    write_json(
        feature_dir / "request.json",
        {
            "pixel_gate_sha256": sha256_file(output / "pixel_gate.json"),
            "selection_sha256": sha256_file(output / "selection_before_image_access.json"),
            "prepared_pixels": {
                p.name: sha256_file(p) for p in sorted((output / "pixels").glob("*.npy"))
            },
            "arms": list(arms),
            "grids": [3, 12],
            "batch_size": 1,
            "models_fitted": 0,
            "timing_concurrency": "Main-agent evidence-memory and ordered-motion small-head runs overlapped by explicit GPU coordination; not isolated hardware benchmark",
        },
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    encoder, provenance = load_vjepa21_encoder(
        Path(lock["upstream"]["path"]), Path(lock["inputs"]["checkpoint"]["path"])
    )
    write_json(feature_dir / "model_receipt.json", provenance)
    comparisons, timings = [], []
    for index, clip in enumerate(selected):
        tick = time.perf_counter()
        for arm in arms:
            arm_tick = time.perf_counter()
            pixels = (
                dataset[index]["pixels"]
                if arm == "existing720"
                else torch.from_numpy(
                    np.load(output / "pixels" / f"{index}_{arm}.npy", allow_pickle=False)
                )
            )
            pixels = pixels.unsqueeze(0).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                dense = encoder(pixels)
            for grid in (3, 12):
                arrays[(arm, grid)][index] = (
                    pool_spatiotemporal_tokens(dense, frames=16, spatial_grid=grid)[0]
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            torch.cuda.synchronize()
            timings.append(
                {"clip_index": index, "arm": arm, "seconds": time.perf_counter() - arm_tick}
            )
        exact = np.array_equal(arrays[("existing720", 3)][index], old[old_index[clip["sample_id"]]])
        if not exact:
            raise RuntimeError("Matched existing720 control did not reproduce historical grid3")
        row = {
            "sample_id": clip["sample_id"],
            "short_fallback": clip["short_fallback"],
            "existing720_grid3_exact": exact,
            "seconds": time.perf_counter() - tick,
        }
        for control in ("existing720", "exact4k_downsample720"):
            a = arrays[("native4k", 3)][index].astype(np.float32).reshape(-1)
            b = arrays[(control, 3)][index].astype(np.float32).reshape(-1)
            row[f"native4k_vs_{control}_feature_cosine"] = float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            )
            row[f"native4k_vs_{control}_feature_rms"] = float(np.sqrt(np.mean((a - b) ** 2)))
        comparisons.append(row)
        if (index + 1) % 8 == 0:
            print(
                json.dumps(
                    {
                        "native64_encoded": index + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    for array in arrays.values():
        array.flush()
    np.save(
        feature_dir / "sample_ids.npy",
        np.asarray([c["sample_id"] for c in selected]),
        allow_pickle=False,
    )
    result = {
        "status": "MATCHED_NATIVE64_FEATURE_RESOURCE_PILOT_COMPLETE",
        "centers": 64,
        "arms": list(arms),
        "existing720_grid3_exact_rows": sum(r["existing720_grid3_exact"] for r in comparisons),
        "encoding_seconds_including_load": time.perf_counter() - started,
        "pixel_preparation_seconds": gate["elapsed_seconds"],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "median_arm_seconds": {
            arm: float(np.median([t["seconds"] for t in timings if t["arm"] == arm]))
            for arm in arms
        },
        "median_native_vs_exactdownsample_feature_cosine": float(
            np.median([r["native4k_vs_exact4k_downsample720_feature_cosine"] for r in comparisons])
        ),
        "median_native_vs_existing720_feature_cosine": float(
            np.median([r["native4k_vs_existing720_feature_cosine"] for r in comparisons])
        ),
        "artifacts": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(feature_dir.glob("*.npy"))
        },
        "rows": comparisons,
        "timings": timings,
        "classifier_fits": 0,
        "action_values_used": 0,
        "protected_data_payloads_read": 0,
        "interpretation": "Representation and resource comparison only; no classification improvement claim; no full4k extraction authorized by this utility",
    }
    write_json(feature_dir / "summary.json", result)
    return result


def verify_cache(output: Path) -> dict:
    """Check completed outputs independently, without loading a model or images."""
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    _, all_clips, manifest_frames, _ = load_allowed_manifest()
    ids = np.load(output / "sample_ids.npy", allow_pickle=False)
    flags = np.load(output / "short_fallback.npy", allow_pickle=False)
    complete = np.load(output / "completed.npy", allow_pickle=False)
    if request["mode"] != "full" or list(ids) != [r["sample_id"] for r in all_clips]:
        raise RuntimeError("Full-cache identity order changed")
    expected_flags = np.asarray(
        [r["all_frames_valid"].lower() not in {"1", "true"} for r in all_clips]
    )
    if not np.array_equal(flags, expected_flags) or not complete.all() or flags.sum() != 467:
        raise RuntimeError("Full-cache completion/fallback policy changed")
    short_frames = {}
    for row in read_csv(P3.parent / "okutama_native_video_p0_r1/manifest/frame_manifest.csv"):
        short_frames.setdefault(row["sample_id"], []).append(row)
    expected_times = []
    for i, sample_id in enumerate(ids):
        frame_rows = short_frames[sample_id] if flags[i] else manifest_frames[sample_id]
        frame_rows = sorted(frame_rows, key=lambda row: int(row["time_index"]))
        expected_times.append([int(row["source_frame"]) for row in frame_rows])
    if not np.array_equal(
        np.load(output / "source_frames.npy", allow_pickle=False),
        np.asarray(expected_times, dtype=np.int32),
    ):
        raise RuntimeError("Exact long/short source frame support changed")
    for name, receipt in result["artifacts"].items():
        if sha256_file(output / name) != receipt["sha256"]:
            raise RuntimeError(f"Completed artifact bytes changed: {name}")
    old_long = np.load(
        P3 / "vjepa_full/vjepa21_long16_real_clip.npy", mmap_mode="r", allow_pickle=False
    )
    old_short = np.load(
        P3.parent / "okutama_native_video_p0_r1/vjepa_full/vjepa21_real_clip.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    actual = np.load(output / "tokens_grid3.npy", mmap_mode="r", allow_pickle=False)
    exact = sum(
        np.array_equal(actual[i], old_short[i] if flags[i] else old_long[i])
        for i in range(len(ids))
    )
    if exact != 4977:
        raise RuntimeError(f"Historical grid3 reconstruction mismatch: {exact}/4977 exact")
    receipt = {
        "status": "FULL_DENSE_CACHE_INDEPENDENT_REPLAY_VERIFIED",
        "rows": len(ids),
        "grid3_exact_rows": exact,
        "short_fallback_rows": int(flags.sum()),
        "models_loaded": 0,
        "images_decoded": 0,
        "artifacts": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in (
                output / name
                for name in (
                    "sample_ids.npy",
                    "short_fallback.npy",
                    "source_frames.npy",
                    "completed.npy",
                    "request.json",
                    "summary.json",
                )
            )
        },
    }
    write_json(output / "verification.json", receipt)
    return receipt


def native_full_encode(output: Path) -> dict:
    """All4977 paired native/downsample features from verified shared uint8 crops."""
    from hac.video_encoders import preprocess_rgb_clip

    crop_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    replay = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if (
        crop_summary["status"] != "ALL21_CHRONOLOGICAL_UINT8_CROP_CACHE_COMPLETE"
        or not replay["all_bit_exact"]
        or replay["clips"] != 64
        or replay["source_summary_sha256"] != sha256_file(output / "summary.json")
    ):
        raise RuntimeError("Full crop cache is not independently verified against native64")
    metadata = crop_summary["metadata"]
    for name, digest in replay["metadata_artifacts"].items():
        if sha256_file(output / name) != digest:
            raise RuntimeError("Independently verified crop metadata changed")
    if (
        metadata["native_centers"] != 4948
        or len(metadata["whole_clip_existing720_fallbacks"]) != 29
    ):
        raise RuntimeError("All4977 native/fallback denominator changed")
    for name, receipt in crop_summary["artifacts"].items():
        if sha256_file(output / name) != receipt["sha256"]:
            raise RuntimeError("Verified uint8 source shard changed")
    feature_dir = output / "features"
    feature_dir.mkdir(exist_ok=False)
    (feature_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lock, clips, _, _ = load_allowed_manifest()
    crop_index = json.loads((output / "cohort_crop_index.json").read_text(encoding="utf-8"))
    clip_crops = crop_index["clip_crops"]
    crop_rows = crop_index["crops"]
    stored = {
        r["crop_index"]: r
        for r in json.loads((output / "stored_crop_offsets.json").read_text(encoding="utf-8"))
    }
    fallback_ids = set(metadata["whole_clip_existing720_fallbacks"])
    dense_dir = ROOT / ".runs/research_20260908/dense_tokens_full"
    dense_summary = json.loads((dense_dir / "summary.json").read_text(encoding="utf-8"))
    dense_replay = json.loads((dense_dir / "verification.json").read_text(encoding="utf-8"))
    if (
        dense_replay["grid3_exact_rows"] != 4977
        or sha256_file(dense_dir / "summary.json")
        != dense_replay["artifacts"]["summary.json"]["sha256"]
    ):
        raise RuntimeError("Existing720 control lineage changed")
    existing = {}
    for name in ("sample_ids.npy", "source_frames.npy", "short_fallback.npy"):
        if sha256_file(dense_dir / name) != dense_replay["artifacts"][name]["sha256"]:
            raise RuntimeError("Existing720 identity/fallback/source-time receipt changed")
    for grid in (3, 12):
        path = dense_dir / f"tokens_grid{grid}.npy"
        if sha256_file(path) != dense_summary["artifacts"][path.name]["sha256"]:
            raise RuntimeError("Existing720 feature bytes changed")
        existing[grid] = np.load(path, mmap_mode="r", allow_pickle=False)
    ids = np.asarray([c["sample_id"] for c in clips])
    if not np.array_equal(ids, np.load(dense_dir / "sample_ids.npy", allow_pickle=False)):
        raise RuntimeError("Existing720 identity order changed")
    original_frames = np.load(dense_dir / "source_frames.npy", allow_pickle=False)
    native_frames = np.full((4977, 16), -1, dtype=np.int32)
    source_fallback = np.asarray([c["sample_id"] in fallback_ids for c in clips])
    historical_short = np.load(dense_dir / "short_fallback.npy", allow_pickle=False)
    expected_short = np.asarray([c["all_frames_valid"].lower() not in {"1", "true"} for c in clips])
    if (
        not fallback_ids <= set(ids)
        or source_fallback.dtype != np.bool_
        or source_fallback.shape != (4977,)
        or source_fallback.sum() != 29
        or historical_short.dtype != np.bool_
        or historical_short.sum() != 467
        or not np.array_equal(historical_short, expected_short)
    ):
        raise RuntimeError("Explicit native29/historical467 fallback identity check failed")
    for index, clip in enumerate(clips):
        if not source_fallback[index]:
            rows = [crop_rows[c] for c in clip_crops[clip["sample_id"]]]
            if not np.array_equal(original_frames[index], [r["source_frame"] for r in rows]):
                raise RuntimeError("Chronological crops changed exact short/long frame support")
            native_frames[index] = [r["native_frame"] for r in rows]
    arms = ("native4k", "exact4k_downsample720")
    arrays = {
        (arm, grid): np.lib.format.open_memmap(
            feature_dir / f"{arm}_grid{grid}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(4977, 8, grid * grid, 768),
        )
        for arm in arms
        for grid in (3, 12)
    }
    completed = np.zeros(4977, dtype=bool)
    for name, values in (
        ("sample_ids.npy", ids),
        ("source_frames.npy", original_frames),
        ("native_frames.npy", native_frames),
        ("native_source_fallback.npy", source_fallback),
        (
            "historical_short_fallback.npy",
            np.load(dense_dir / "short_fallback.npy", allow_pickle=False),
        ),
        ("completed.npy", completed),
    ):
        np.save(feature_dir / name, values, allow_pickle=False)
    write_json(
        feature_dir / "request.json",
        {
            "crop_summary_sha256": sha256_file(output / "summary.json"),
            "crop_verification_sha256": sha256_file(output / "verification.json"),
            "existing720_summary_sha256": sha256_file(dense_dir / "summary.json"),
            "sample_ids_sha256": sha256_file(feature_dir / "sample_ids.npy"),
            "source_frames_sha256": sha256_file(feature_dir / "source_frames.npy"),
            "cohort_crop_index_sha256": sha256_file(output / "cohort_crop_index.json"),
            "stored_crop_offsets_sha256": sha256_file(output / "stored_crop_offsets.json"),
            "centers": 4977,
            "native_centers": 4948,
            "exact720_whole_clip_fallbacks": 29,
            "historical_short_fallbacks": 467,
            "arms": list(arms),
            "grids": [3, 12],
            "batch_size": 1,
            "input_dtype": "uint8 immutable physical actor crops",
            "precision": "bfloat16",
            "cache_dtype": "float16",
            "source_timing": "verified default MP4 edit-list timeline with fixed per-recording offsets",
            "timing_concurrency": "Main-agent small-head jobs may overlap; GPU budget at most1GiB",
            "classifier_fits": 0,
            "action_values_used": 0,
        },
    )
    pilot_dir = ROOT / ".runs/research_20260908/native4k_matched_pilot/features"
    pilot_ids = np.load(pilot_dir / "sample_ids.npy", allow_pickle=False)
    pilot_summary = json.loads((pilot_dir / "summary.json").read_text(encoding="utf-8"))
    for name, receipt in pilot_summary["artifacts"].items():
        if sha256_file(pilot_dir / name) != receipt["sha256"]:
            raise RuntimeError("Native64 reference feature artifact changed")
    pilot_index = {sample_id: index for index, sample_id in enumerate(pilot_ids)}
    pilot_arrays = {
        (arm, grid): np.load(pilot_dir / f"{arm}_grid{grid}.npy", mmap_mode="r", allow_pickle=False)
        for arm in arms
        for grid in (3, 12)
    }
    source_maps = {
        (recording, arm): np.memmap(
            output / f"{recording}_{'native' if arm == 'native4k' else 'down720'}.uint8",
            mode="r",
            dtype=np.uint8,
        )
        for recording in EXPECTED_RECORDINGS
        for arm in arms
    }
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder, provenance = load_vjepa21_encoder(
        Path(lock["upstream"]["path"]), Path(lock["inputs"]["checkpoint"]["path"])
    )
    write_json(feature_dir / "model_receipt.json", provenance)
    timing, pilot_checks = [], []
    for index, clip in enumerate(clips):
        tick = time.perf_counter()
        if source_fallback[index]:
            for arm in arms:
                for grid in (3, 12):
                    arrays[(arm, grid)][index] = existing[grid][index]
        else:
            for arm in arms:
                arm_tick = time.perf_counter()
                native = arm == "native4k"
                mmap = source_maps[(clip["provider_recording_id"], arm)]
                images = []
                for crop_id in clip_crops[clip["sample_id"]]:
                    receipt = stored[crop_id]
                    offset = receipt["native_offset" if native else "low_offset"]
                    shape = tuple(receipt["native_shape" if native else "low_shape"])
                    images.append(
                        np.asarray(mmap[offset : offset + int(np.prod(shape))]).reshape(shape)
                    )
                pixels = preprocess_rgb_clip(images).unsqueeze(0).to("cuda")
                torch.cuda.synchronize()
                forward_tick = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder(pixels)
                for grid in (3, 12):
                    arrays[(arm, grid)][index] = (
                        pool_spatiotemporal_tokens(tokens, frames=16, spatial_grid=grid)[0]
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                torch.cuda.synchronize()
                if (
                    torch.cuda.max_memory_allocated() > 1024**3
                    or torch.cuda.max_memory_reserved() > 1024**3
                ):
                    raise RuntimeError("Native paired encoder exceeded its declared1GiB GPU budget")
                timing.append(
                    {
                        "row": index,
                        "arm": arm,
                        "total_seconds": time.perf_counter() - arm_tick,
                        "forward_pool_save_seconds": time.perf_counter() - forward_tick,
                    }
                )
        if clip["sample_id"] in pilot_index:
            for arm in arms:
                exact = all(
                    np.array_equal(
                        arrays[(arm, grid)][index],
                        pilot_arrays[(arm, grid)][pilot_index[clip["sample_id"]]],
                    )
                    for grid in (3, 12)
                )
                if not exact:
                    raise RuntimeError("Full feature cache changed exact bounded native64 features")
                pilot_checks.append(
                    {"sample_id": clip["sample_id"], "arm": arm, "both_grids_exact": exact}
                )
        completed[index] = True
        if (index + 1) % 64 == 0 or index + 1 == 4977:
            for array in arrays.values():
                array.flush()
            temporary = feature_dir / "completed.tmp.npy"
            np.save(temporary, completed, allow_pickle=False)
            os.replace(temporary, feature_dir / "completed.npy")
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "native_full_encoded": index + 1,
                        "of": 4977,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed / (index + 1) * (4977 - index - 1),
                        "last_row_seconds": time.perf_counter() - tick,
                    }
                ),
                flush=True,
            )
    for arm in arms:
        for grid in (3, 12):
            if not np.array_equal(
                arrays[(arm, grid)][source_fallback], existing[grid][source_fallback]
            ):
                raise RuntimeError("Native source fallback changed existing720 control")
    if len(pilot_checks) != 128:
        raise RuntimeError("Native64 exact feature replay coverage incomplete")
    result = {
        "status": "FULL_PAIRED_NATIVE4K_FEATURE_EXTRACTION_COMPLETE",
        "centers": 4977,
        "native_centers": 4948,
        "exact720_fallbacks": 29,
        "both_grids_native64_bit_exact_checks": len(pilot_checks),
        "elapsed_seconds_including_model_load": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "median_arm_seconds": {
            arm: float(np.median([t["total_seconds"] for t in timing if t["arm"] == arm]))
            for arm in arms
        },
        "median_forward_pool_save_seconds": float(
            np.median([t["forward_pool_save_seconds"] for t in timing])
        ),
        "artifacts": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(feature_dir.glob("*.npy"))
        },
        "pilot_checks": pilot_checks,
        "timing": timing,
        "action_values_used": 0,
        "classifier_fits": 0,
        "protected_data_payloads_read": 0,
    }
    write_json(feature_dir / "summary.json", result)
    return result


def declare_source_effect(output: Path) -> dict:
    """Freeze the small source-effect study before accessing any new accuracy."""
    reference_path = ROOT / "experiments/okutama_video_p3_probe_protocol.json"
    historical = json.loads(reference_path.read_text(encoding="utf-8"))
    original_manifest = json.loads(
        (ROOT / "experiments/okutama_video_p3_protocol.json").read_text(encoding="utf-8")
    )
    lock, clips, _, _ = load_allowed_manifest()
    config = {
        key: historical["probe"][key]
        for key in (
            "C_values",
            "solver",
            "class_weight",
            "max_iter",
            "tolerance",
            "standardize",
            "inner_folds",
            "inner_splitter",
            "inner_shuffle",
            "inner_seed",
            "selection",
        )
    }
    if config["C_values"] != [1e-5, 1e-4, 1e-3, 1e-2] or config["inner_folds"] != 3:
        raise RuntimeError("Original small-probe configuration changed")
    protocol = {
        "status": "DECLARED_BEFORE_FULL_NATIVE_FEATURE_COMPLETION_OR_ANY_NEW_ACCURACY",
        "study": "controlled_source_effect_not_new_architecture",
        "sources": ["existing720", "native4k", "exact4k_downsample720"],
        "heads": {
            "global_mean": {
                "dimensions": 768,
                "recipe": "float32 effective grid3 tokens mean over time and space",
            },
            "global_plus_spatial_contrast": {
                "dimensions": 7680,
                "recipe": "concat global_mean,flatten_region_major(time_mean_grid3 - its region_mean),all arithmetic float32",
            },
        },
        "seed": 42,
        "probe": config,
        "fold_contract": original_manifest["fold_contract"],
        "rows": len(clips),
        "historical_short_fallbacks": 467,
        "native_source_whole_clip_720_fallbacks": 29,
        "class_order": original_manifest["data_contract"]["class_order"],
        "fit_counts": {
            "per_source_per_head": 65,
            "two_new_sources_two_heads": 260,
            "two_existing720_control_replays": 130,
            "maximum_including_independent_control_replays": 390,
        },
        "control_policy": "Both existing720 heads are fixed controls; a byte/recipe/fold/config-identical historical global-mean receipt may be replayed, otherwise fit all390 estimators and report count",
        "fixed_comparisons": [
            "native4k minus exact4k_downsample720 within each fixed head",
            "native4k minus existing720 within each fixed head",
            "exact4k_downsample720 minus existing720 within each fixed head",
        ],
        "reporting": [
            "macro_f1",
            "accuracy",
            "per_class_f1",
            "confusion",
            "nll",
            "brier",
            "per_fold_metrics",
            "paired_rescues_and_harms",
            "native_fallback_stratum",
        ],
        "no_outer_label_source_or_head_selection": True,
        "no_learned_oof_fusion": True,
        "no_threshold_tuning": True,
        "no_new_backbone_training": True,
        "no_new_dino_features": True,
        "no_observation_duration_change": True,
        "fold_standardizer_fit_on_training_rows_only": True,
        "convergence_policy": "fail closed on ConvergenceWarning or max_iter reached",
        "existing720_control_grid3_sha256": "da6af8f2fd879fe06dee36e9991f499ab6f6fac48b44f0f5679538d8d64687c5",
        "manifest_sha256": lock["manifest"]["artifacts"]["clip_index"]["sha256"],
        "original_probe_protocol_sha256": sha256_file(reference_path),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(output / "protocol.json", protocol)
    return {
        "status": protocol["status"],
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "fit_counts": protocol["fit_counts"],
    }


def run_source_effect(output: Path, native_dir: Path) -> dict:
    """Fixed six-arm source screen; outer outcomes revealed only after all fits."""
    if str(ROOT / "experiments") not in sys.path:
        sys.path.insert(0, str(ROOT / "experiments"))
    import run_okutama_video_p2a as p2

    p1 = p2.p1
    protocol_path = output / "protocol.json"
    if (
        sha256_file(protocol_path)
        != "e62c30fa2687f8762eefed32353a1c3f64ba9994f1c3037a94f3ad3601afd546"
    ):
        raise RuntimeError("Declared source-effect protocol changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    replay_addendum_path = output / "replay_gate_addendum.json"
    replay_addendum = json.loads(replay_addendum_path.read_text(encoding="utf-8"))
    if (
        replay_addendum["maximum_absolute_probability_difference"] != 1e-8
        or not replay_addendum["require_exact_identity_labels_scenarios_folds"]
        or replay_addendum["new_source_accuracy_viewed"]
    ):
        raise RuntimeError("Prospective historical-replay integrity gate changed")
    native_summary_path = native_dir / "features/summary.json"
    native_summary = json.loads(native_summary_path.read_text(encoding="utf-8"))
    if (
        native_summary["status"] != "FULL_PAIRED_NATIVE4K_FEATURE_EXTRACTION_COMPLETE"
        or native_summary["centers"] != 4977
        or native_summary["exact720_fallbacks"] != 29
        or native_summary["both_grids_native64_bit_exact_checks"] != 128
    ):
        raise RuntimeError("Full source extraction was not verified")
    lock, retained, _, _ = load_allowed_manifest()
    if lock["manifest"]["artifacts"]["clip_index"]["sha256"] != protocol["manifest_sha256"]:
        raise RuntimeError("Frozen classification cohort changed")
    rows = read_csv(P3 / "manifest/clip_index.csv")
    ids = np.asarray([r["sample_id"] for r in rows])
    labels = np.asarray([int(r["label_index"]) for r in rows], dtype=np.int64)
    scenarios = np.asarray([r["recording_id"] for r in rows])
    folds = np.asarray([int(r["fold"].removeprefix("fold-")) for r in rows], dtype=np.int64)
    expected_folds = {
        scenario: int(fold.removeprefix("fold-"))
        for fold, groups in protocol["fold_contract"].items()
        for scenario in groups
    }
    if (
        not np.array_equal(ids, [r["sample_id"] for r in retained])
        or len(set(ids)) != 4977
        or set(labels) != {0, 1, 2}
        or any(
            expected_folds[r["recording_id"]] != int(r["fold"].removeprefix("fold-"))
            or r["scope"] != "grouped_crossfit_oof"
            or r["development_role"] != "train"
            for r in rows
        )
    ):
        raise RuntimeError("Historical class/fold/primary-row contract changed")
    if not np.array_equal(ids, np.load(native_dir / "features/sample_ids.npy", allow_pickle=False)):
        raise RuntimeError("Source feature identity order differs")
    existing_path = ROOT / ".runs/research_20260908/dense_tokens_full/tokens_grid3.npy"
    paths = {
        "existing720": existing_path,
        **{
            arm: native_dir / "features" / f"{arm}_grid3.npy"
            for arm in ("native4k", "exact4k_downsample720")
        },
    }
    feature_receipts = {}
    for arm, path in paths.items():
        expected = (
            protocol["existing720_control_grid3_sha256"]
            if arm == "existing720"
            else native_summary["artifacts"][path.name]["sha256"]
        )
        if sha256_file(path) != expected:
            raise RuntimeError("Classification source feature bytes changed")
        feature_receipts[arm] = {"path": str(path), "sha256": expected}
    result_dir = output / "results"
    if (
        result_dir.exists()
        and any(result_dir.iterdir())
        and not (result_dir / "request.json").exists()
    ):
        raise RuntimeError(
            "Unrecognized partial source-screen directory; retained without overwrite"
        )
    result_dir.mkdir(exist_ok=True)
    snapshot = result_dir / Path(__file__).name
    if snapshot.exists():
        if sha256_file(snapshot) != sha256_file(Path(__file__)):
            raise RuntimeError("Source-screen executable changed since saved fold receipts")
    else:
        snapshot.write_bytes(Path(__file__).read_bytes())

    def fixed_json(path: Path, value: dict) -> None:
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise RuntimeError(f"Saved source-screen receipt changed: {path.name}")
        else:
            write_json(path, value)

    fixed_json(
        result_dir / "request.json",
        {
            "protocol_sha256": sha256_file(protocol_path),
            "source_summary_sha256": sha256_file(native_summary_path),
            "features": feature_receipts,
            "classifier_code_sha256": sha256_file(Path(p2.__file__)),
            "split_metrics_code_sha256": sha256_file(Path(p1.__file__)),
            "actual_scheduled_estimator_fits": 390,
            "replay_gate_addendum_sha256": sha256_file(replay_addendum_path),
            "existing720_controls": "independent full nested replay, not copied probabilities",
            "outer_accuracy_access_during_fitting": False,
        },
    )
    config = protocol["probe"]
    splits = {
        fold: p1.grouped_inner_splits(labels, scenarios, folds, fold, n_splits=3, seed=42)
        for fold in range(5)
    }
    fixed_json(
        result_dir / "split_indices.json",
        {
            str(fold): [{"train": train.tolist(), "held": held.tolist()} for train, held in group]
            for fold, group in splits.items()
        },
    )
    oof, fold_receipts = {}, []
    fit_count = 0
    resumed_fits = 0
    started = time.perf_counter()
    for source in protocol["sources"]:
        values = np.load(paths[source], mmap_mode="r", allow_pickle=False)
        if values.shape != (4977, 8, 9, 768) or values.dtype != np.float16:
            raise RuntimeError("Classification grid3 feature contract changed")
        global_mean = np.empty((4977, 768), dtype=np.float32)
        contrast = np.empty((4977, 6912), dtype=np.float32)
        for start in range(0, 4977, 64):
            block = np.asarray(values[start : start + 64], dtype=np.float32)
            if not np.isfinite(block).all():
                raise RuntimeError("Nonfinite frozen source features")
            region = block.mean(axis=1, dtype=np.float32)
            global_mean[start : start + len(block)] = block.mean(axis=(1, 2), dtype=np.float32)
            contrast[start : start + len(block)] = (
                region - region.mean(axis=1, keepdims=True, dtype=np.float32)
            ).reshape(len(block), -1)
        heads = {
            "global_mean": global_mean,
            "global_plus_spatial_contrast": np.concatenate((global_mean, contrast), axis=1),
        }
        for head in protocol["heads"]:
            x = heads[head]
            if x.shape[1] != protocol["heads"][head]["dimensions"]:
                raise RuntimeError("Predeclared fixed head dimensions changed")
            arm = f"{source}__{head}"
            probabilities = np.full((4977, 3), np.nan, dtype=np.float64)
            for fold in range(5):
                tick = time.perf_counter()
                fit_rows, held_rows = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                prefix = f"{arm}__fold{fold}"
                checkpoint_path = result_dir / f"{prefix}.npz"
                receipt_path = result_dir / f"{prefix}.json"
                if checkpoint_path.exists() or receipt_path.exists():
                    if not checkpoint_path.exists() or not receipt_path.exists():
                        raise RuntimeError(
                            "Incomplete per-fold source receipt retained; refusing overwrite"
                        )
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if (
                        receipt["source"] != source
                        or receipt["head"] != head
                        or receipt["outer_fold"] != fold
                        or receipt["fit_rows"] != len(fit_rows)
                        or receipt["held_rows"] != len(held_rows)
                        or sha256_file(checkpoint_path) != receipt["checkpoint_sha256"]
                    ):
                        raise RuntimeError("Saved source-screen fold identity or bytes changed")
                    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                        if not np.array_equal(checkpoint["held_rows"], held_rows):
                            raise RuntimeError("Saved source-screen held rows changed")
                        p1.validate_probability_array(checkpoint["probabilities"], len(held_rows))
                        probabilities[held_rows] = checkpoint["probabilities"]
                    fit_count += 13
                    resumed_fits += 13
                    fold_receipts.append(receipt)
                    print(
                        json.dumps(
                            {
                                "verified_saved_fold": prefix,
                                "completed_unique_estimator_fits": fit_count,
                            }
                        ),
                        flush=True,
                    )
                    continue
                candidates = []
                for c in config["C_values"]:
                    inner = np.full((4977, 3), np.nan, dtype=np.float64)
                    iterations = []
                    for train, held in splits[fold]:
                        fitted = p2.fit_logistic(
                            x[train], labels[train], config, float(c), 42, binary=False
                        )
                        fit_count += 1
                        inner[held] = p2._predict(fitted, x[held])
                        iterations.append(int(np.max(fitted[1].n_iter_)))
                    candidates.append(
                        {
                            "C": float(c),
                            "inner_metrics": p1.metrics(labels[fit_rows], inner[fit_rows]),
                            "inner_iterations": iterations,
                        }
                    )
                chosen = p2.select_candidate(candidates, factorized=False)
                fitted = p2.fit_logistic(
                    x[fit_rows], labels[fit_rows], config, chosen["C"], 42, binary=False
                )
                fit_count += 1
                probabilities[held_rows] = p2._predict(fitted, x[held_rows])
                p1.validate_probability_array(probabilities[held_rows], len(held_rows))
                np.savez_compressed(
                    checkpoint_path,
                    held_rows=held_rows,
                    probabilities=probabilities[held_rows],
                    **p2._checkpoint_arrays(fitted, "model"),
                )
                receipt = {
                    "source": source,
                    "head": head,
                    "outer_fold": fold,
                    "fit_rows": len(fit_rows),
                    "held_rows": len(held_rows),
                    "selected": chosen,
                    "candidates": candidates,
                    "final_iterations": int(np.max(fitted[1].n_iter_)),
                    "elapsed_seconds": time.perf_counter() - tick,
                    "outer_metrics_calculated": False,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
                write_json(receipt_path, receipt)
                fold_receipts.append(receipt)
                print(
                    json.dumps(
                        {
                            "source": source,
                            "head": head,
                            "outer_fold_complete": fold,
                            "estimator_fits": fit_count,
                            "of": 390,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            p1.validate_probability_array(probabilities, 4977)
            oof[arm] = probabilities
    if fit_count != 390:
        raise RuntimeError("Fixed source screen fit count mismatch")
    all_arrays = {
        "sample_ids": ids,
        "recording_ids": scenarios,
        "labels": labels,
        "folds": folds,
        **oof,
    }
    oof_path = result_dir / "oof_probabilities.npz"
    if oof_path.exists():
        with np.load(oof_path, allow_pickle=False) as previous:
            if set(previous.files) != set(all_arrays) or any(
                not np.array_equal(previous[name], values) for name, values in all_arrays.items()
            ):
                raise RuntimeError(
                    "Existing complete OOF bytes/values differ from verified fold replay"
                )
    else:
        np.savez_compressed(oof_path, **all_arrays)
    source_fallback = np.load(
        native_dir / "features/native_source_fallback.npy", allow_pickle=False
    )
    if (
        source_fallback.sum() != 29
        or sha256_file(native_dir / "features/native_source_fallback.npy")
        != native_summary["artifacts"]["native_source_fallback.npy"]["sha256"]
    ):
        raise RuntimeError("Native fallback-stratum identity changed")
    metrics = {
        arm: {
            "overall": p1.metrics(labels, probabilities),
            "folds": {
                str(fold): p1.metrics(labels[folds == fold], probabilities[folds == fold])
                for fold in range(5)
            },
            "native_source_fallback": p1.metrics(
                labels[source_fallback], probabilities[source_fallback]
            ),
        }
        for arm, probabilities in oof.items()
    }
    comparisons = []
    for head in protocol["heads"]:
        for candidate, reference in (
            ("native4k", "exact4k_downsample720"),
            ("native4k", "existing720"),
            ("exact4k_downsample720", "existing720"),
        ):
            cand, ref = f"{candidate}__{head}", f"{reference}__{head}"
            correct_c, correct_r = oof[cand].argmax(1) == labels, oof[ref].argmax(1) == labels
            comparisons.append(
                {
                    "head": head,
                    "candidate": candidate,
                    "reference": reference,
                    "macro_f1_delta": metrics[cand]["overall"]["macro_f1"]
                    - metrics[ref]["overall"]["macro_f1"],
                    "rescues": int((correct_c & ~correct_r).sum()),
                    "harms": int((~correct_c & correct_r).sum()),
                    "both_wrong": int((~correct_c & ~correct_r).sum()),
                    "source_fallback_rescues": int(
                        (correct_c & ~correct_r & source_fallback).sum()
                    ),
                    "source_fallback_harms": int((~correct_c & correct_r & source_fallback).sum()),
                    "native_available_rescues": int(
                        (correct_c & ~correct_r & ~source_fallback).sum()
                    ),
                    "native_available_harms": int(
                        (~correct_c & correct_r & ~source_fallback).sum()
                    ),
                }
            )
    historical_path = P3 / "results/oof_probabilities.npz"
    with np.load(historical_path, allow_pickle=False) as historical:
        for name, expected in (
            ("sample_ids", ids),
            ("recording_ids", scenarios),
            ("labels", labels),
            ("folds", folds),
        ):
            if not np.array_equal(historical[name], expected):
                raise RuntimeError("Historical existing720 probability replay identity changed")
        replay_error = float(
            np.max(np.abs(oof["existing720__global_mean"] - historical["long_vjepa_mean"]))
        )
    replay_receipt = {
        "declared_absolute_tolerance": 1e-8,
        "max_abs_difference": replay_error,
        "passed": replay_error <= 1e-8,
        "historical_oof_sha256": sha256_file(historical_path),
        "source": "existing720",
        "head": "global_mean",
    }
    fixed_json(result_dir / "historical_replay_verification.json", replay_receipt)
    if not replay_receipt["passed"]:
        raise RuntimeError(
            "Historical existing720 global-mean replay failed its prospective1e-8 gate; source screen remains unverified"
        )
    result = {
        "status": "FIXED_NESTED_SOURCE_EFFECT_SCREEN_COMPLETE",
        "protocol_sha256": sha256_file(protocol_path),
        "estimator_fits": fit_count,
        "new_estimator_fits_this_invocation": fit_count - resumed_fits,
        "verified_saved_estimator_fits": resumed_fits,
        "centers": 4977,
        "metrics": metrics,
        "comparisons": comparisons,
        "historical_existing720_global_mean_probability_max_difference": replay_error,
        "elapsed_seconds": time.perf_counter() - started,
        "oof_sha256": sha256_file(result_dir / "oof_probabilities.npz"),
        "source_or_head_selected_on_outer_labels": False,
        "learned_oof_fusion": False,
        "gpu_models_loaded": 0,
        "interpretation": "Fixed source-resolution comparison, not a new architecture claim; report every predeclared arm",
    }
    if (result_dir / "summary.json").exists():
        previous = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        if (
            previous["protocol_sha256"] != result["protocol_sha256"]
            or previous["oof_sha256"] != result["oof_sha256"]
            or previous["metrics"] != result["metrics"]
        ):
            raise RuntimeError("Completed source-screen summary differs from exact replay")
        return previous
    write_json(result_dir / "summary.json", result)
    return result


def run(output: Path, selection_path: Path | None, mode: str = "pilot"):
    output.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name("audit_okutama_source_views.py")):
        (output / path.name).write_bytes(path.read_bytes())
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock, all_clips, frames, allowed = load_allowed_manifest()
    if mode == "pilot":
        if selection_path is None:
            raise ValueError("Pilot requires a fixed selection")
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        if selected != select_pilot(all_clips):
            raise RuntimeError("Dense pilot selection differs from fixed source audit")
    else:
        selected = all_clips
    fallback = np.asarray([r["all_frames_valid"].lower() not in {"1", "true"} for r in selected])
    old_short = None
    short_receipt = None
    if fallback.any():
        p0_path = ROOT / lock["p0_extraction_lock"]["path"]
        if sha256_file(p0_path) != lock["p0_extraction_lock"]["sha256"]:
            raise RuntimeError("Historical short input extraction lock changed")
        p0 = json.loads(p0_path.read_text(encoding="utf-8"))
        for name in ("clip_index", "frame_manifest", "image_allowlist"):
            receipt = p0["manifest"]["artifacts"][name]
            if sha256_file(ROOT / receipt["path"]) != receipt["sha256"]:
                raise RuntimeError(f"Short fallback manifest changed: {name}")
        short_clips = read_csv(ROOT / p0["manifest"]["artifacts"]["clip_index"]["path"])
        if [r["sample_id"] for r in short_clips] != [r["sample_id"] for r in all_clips]:
            raise RuntimeError("Short fallback identity order differs")
        wanted = {
            r["sample_id"] for r, use_short in zip(selected, fallback, strict=True) if use_short
        }
        short_frames = {}
        for row in read_csv(ROOT / p0["manifest"]["artifacts"]["frame_manifest"]["path"]):
            if row["sample_id"] in wanted:
                short_frames.setdefault(row["sample_id"], []).append(row)
        for sample_id, rows in short_frames.items():
            rows.sort(key=lambda row: int(row["time_index"]))
            if len(rows) != 16 or not all(r["valid_frame"].lower() in {"1", "true"} for r in rows):
                raise RuntimeError("Exact short fallback is incomplete")
            frames[sample_id] = rows
        if set(short_frames) != wanted:
            raise RuntimeError("Missing short fallback identity")
        allowed |= {
            r["image_member"]
            for r in read_csv(ROOT / p0["manifest"]["artifacts"]["image_allowlist"]["path"])
        }
        short_path = p0_path.parent / "vjepa_full/vjepa21_real_clip.npy"
        old_short = np.load(short_path, mmap_mode="r", allow_pickle=False)
        short_receipt = {
            "lock_sha256": sha256_file(p0_path),
            "cache_sha256": sha256_file(short_path),
        }
    old_index = {r["sample_id"]: i for i, r in enumerate(all_clips)}
    old = np.load(P3 / "vjepa_full/vjepa21_long16_real_clip.npy", mmap_mode="r", allow_pickle=False)
    request = {
        "sample_ids": [r["sample_id"] for r in selected],
        "mode": mode,
        "selection_sha256": sha256_file(selection_path) if selection_path else None,
        "short_fallback_rows": int(fallback.sum()),
        "short_lineage": short_receipt,
        "spatial_grids": [3, 12],
        "temporal_tubelets": 8,
        "precision": "bfloat16",
        "cache_dtype": "float16",
        "batch_size": 1,
        "models_fitted": 0,
        "action_values_used": 0,
        "extraction_lock_sha256": sha256_file(P3 / "extraction_lock.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "source_audit_helper_sha256": sha256_file(
            Path(__file__).with_name("audit_okutama_source_views.py")
        ),
        "dataset_helper_sha256": sha256_file(
            Path(__file__).with_name("cache_okutama_long_features.py")
        ),
    }
    write_json(output / "request.json", request)
    if (
        mode == "full"
        and sha256_file(Path(lock["inputs"]["archive"]["path"]))
        != lock["inputs"]["archive"]["sha256"]
    ):
        raise RuntimeError("Source JPEG archive changed")
    np.save(
        output / "sample_ids.npy",
        np.asarray([r["sample_id"] for r in selected]),
        allow_pickle=False,
    )
    np.save(output / "short_fallback.npy", fallback, allow_pickle=False)
    np.save(
        output / "source_frames.npy",
        np.asarray(
            [[int(r["source_frame"]) for r in frames[c["sample_id"]]] for c in selected],
            dtype=np.int32,
        ),
        allow_pickle=False,
    )
    completed = np.zeros(len(selected), dtype=bool)
    np.save(output / "completed.npy", completed, allow_pickle=False)
    dataset = LongClipDataset(Path(lock["inputs"]["archive"]["path"]), selected, frames, allowed)
    arrays = {
        g: np.lib.format.open_memmap(
            output / f"tokens_grid{g}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(selected), 8, g * g, 768),
        )
        for g in (3, 12)
    }
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    encoder, provenance = load_vjepa21_encoder(
        Path(lock["upstream"]["path"]), Path(lock["inputs"]["checkpoint"]["path"])
    )
    write_json(output / "model_receipt.json", provenance)
    timing, comparisons = [], []
    for i, clip in enumerate(selected):
        t0 = time.perf_counter()
        item = dataset[i]
        if not item["valid"]:
            raise RuntimeError("Complete-clip pilot selected incomplete input")
        pixels = item["pixels"].unsqueeze(0).to("cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            dense = encoder(pixels)
        values = {g: pool_spatiotemporal_tokens(dense, frames=16, spatial_grid=g) for g in (3, 12)}
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        maps = values[12].reshape(8, 12, 12, 768).permute(0, 3, 1, 2)
        recovered = F.avg_pool2d(maps, 4).permute(0, 2, 3, 1).reshape(1, 8, 9, 768)
        pool_delta = float((recovered - values[3]).abs().max())
        for g in (3, 12):
            arrays[g][i] = values[g][0].cpu().numpy().astype(np.float16)
        reference = old_short if fallback[i] else old
        existing = np.asarray(reference[old_index[clip["sample_id"]]])
        old_delta = float(
            np.max(np.abs(arrays[3][i].astype(np.float32) - existing.astype(np.float32)))
        )
        fine = values[12].float().reshape(8, 12, 12, 768)
        coarse_up = (
            values[3]
            .float()
            .reshape(8, 3, 3, 768)
            .repeat_interleave(4, dim=1)
            .repeat_interleave(4, dim=2)
        )
        detail_rms = float((fine - coarse_up).square().mean().sqrt())
        comparisons.append(
            {
                "sample_id": clip["sample_id"],
                "short_fallback": bool(fallback[i]),
                "cached3_max_abs_difference": old_delta,
                "cached3_bit_exact": bool(np.array_equal(existing, arrays[3][i])),
                "grid12_to3_max_abs_difference_float32": pool_delta,
                "fine_within_coarse_cell_rms": detail_rms,
            }
        )
        timing.append(
            {
                "decode_preprocess_upload_seconds": t1 - t0,
                "forward_both_pool_seconds": t2 - t1,
                "total_clip_seconds": time.perf_counter() - t0,
            }
        )
        del pixels, dense, values, maps, recovered, fine, coarse_up
        completed[i] = True
        if (i + 1) % (16 if mode == "pilot" else 64) == 0:
            for arr in arrays.values():
                arr.flush()
            with (output / "completed.tmp").open("wb") as stream:
                np.save(stream, completed, allow_pickle=False)
            (output / "completed.tmp").replace(output / "completed.npy")
            print(
                json.dumps(
                    {"dense_completed": i + 1, "elapsed_seconds": time.perf_counter() - started}
                ),
                flush=True,
            )
    for arr in arrays.values():
        arr.flush()
    with (output / "completed.tmp").open("wb") as stream:
        np.save(stream, completed, allow_pickle=False)
    (output / "completed.tmp").replace(output / "completed.npy")
    result = {
        "status": "DENSE_GRID_RESOURCE_PILOT_COMPLETE"
        if mode == "pilot"
        else "DENSE_GRID_ALL_CENTER_CACHE_COMPLETE",
        "clips": len(selected),
        "short_fallback_rows": int(fallback.sum()),
        "gpu": torch.cuda.get_device_name(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds_including_model_load": time.perf_counter() - started,
        "median_forward_both_pool_seconds": float(
            np.median([r["forward_both_pool_seconds"] for r in timing[1:]])
        ),
        "median_total_clip_seconds": float(
            np.median([r["total_clip_seconds"] for r in timing[1:]])
        ),
        "cached3_bit_exact_rows": sum(r["cached3_bit_exact"] for r in comparisons),
        "cached3_max_abs_difference": max(r["cached3_max_abs_difference"] for r in comparisons),
        "grid12_to3_max_abs_difference_float32": max(
            r["grid12_to3_max_abs_difference_float32"] for r in comparisons
        ),
        "median_fine_within_coarse_cell_rms": float(
            np.median([r["fine_within_coarse_cell_rms"] for r in comparisons])
        ),
        "artifacts": {
            f"tokens_grid{g}.npy": {
                "bytes": (output / f"tokens_grid{g}.npy").stat().st_size,
                "sha256": sha256_file(output / f"tokens_grid{g}.npy"),
            }
            for g in (3, 12)
        },
        "rows": comparisons,
        "timing": timing,
        "action_values_used": 0,
        "classifier_fits": 0,
        "protected_data_payloads_read": 0,
        "image_decodes": 16 * len(selected),
        "interpretation": "Finer-grid variation is representational detail, not a measured classification gain",
    }
    write_json(output / "summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"rows", "timing"}}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "pilot",
            "full",
            "verify",
            "native_prepare",
            "native_encode",
            "source_protocol",
            "native_full_encode",
            "source_classify",
        ),
        default="pilot",
    )
    parser.add_argument("--alignment-summary", type=Path)
    parser.add_argument("--fidelity-summary", type=Path)
    parser.add_argument("--native-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        print(json.dumps(verify_cache(args.output_dir), indent=2), flush=True)
    elif args.mode == "source_protocol":
        print(json.dumps(declare_source_effect(args.output_dir), indent=2), flush=True)
    elif args.mode == "native_full_encode":
        result = native_full_encode(args.output_dir)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in {"pilot_checks", "timing"}}, indent=2
            ),
            flush=True,
        )
    elif args.mode == "source_classify":
        if args.native_dir is None:
            parser.error("source_classify requires --native-dir")
        result = run_source_effect(args.output_dir, args.native_dir)
        print(json.dumps(result, indent=2), flush=True)
    elif args.mode == "native_prepare":
        if args.alignment_summary is None or args.fidelity_summary is None:
            parser.error("native_prepare requires --alignment-summary and --fidelity-summary")
        result = native4k_prepare(args.output_dir, args.alignment_summary, args.fidelity_summary)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)
    elif args.mode == "native_encode":
        result = native4k_encode(args.output_dir)
        print(
            json.dumps({k: v for k, v in result.items() if k not in {"rows", "timings"}}, indent=2),
            flush=True,
        )
    else:
        run(args.output_dir, args.selection, args.mode)
