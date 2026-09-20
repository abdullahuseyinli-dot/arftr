"""Long-video-only source swap with exact original feature/fallback reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.actor_memory_base import file_sha256
from hac.video_multiscale import arm_features, derive_multiscale_features
from hac.video_token_moments import arm_inputs, derive_token_moments


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Immutable source-swap artifact changed: {path}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)


def derive_features(short_video, long_video, short_dino, long_dino, valid):
    multiscale = derive_multiscale_features(
        short_video, long_video, short_dino, long_dino, valid, valid
    )
    moments = derive_token_moments(multiscale, short_video, long_video, short_dino, long_dino)
    factorized = arm_inputs(moments, "orthogonal_moments_factorized")
    base = {
        "long_vjepa_mean": multiscale.long_v,
        "dual_scale_vjepa_dino": arm_features(multiscale, "dual_scale_vjepa_dino")[0],
        "orthogonal_moments_posture": factorized.posture_or_direct,
        "orthogonal_moments_motion": factorized.motion_references[0],
    }
    memory = np.concatenate(
        (multiscale.short_v, multiscale.long_v, multiscale.short_d, multiscale.long_d),
        axis=1,
    )
    return memory, base


def masked_long_source(regenerated_tokens, original_long, valid, source_fallback):
    if (
        regenerated_tokens.shape != original_long.shape
        or regenerated_tokens.dtype != np.float16
        or original_long.dtype != np.float16
        or valid.dtype != bool
        or source_fallback.dtype != bool
        or valid.shape != source_fallback.shape
        or len(valid) != len(original_long)
    ):
        raise ValueError("Source swap shape/dtype/row contract changed")
    if not np.array_equal(
        regenerated_tokens[valid & source_fallback], original_long[valid & source_fallback]
    ):
        raise RuntimeError("A valid-long source fallback differs from original long tokens")
    replacement = np.array(regenerated_tokens, copy=True)
    # The paired cache re-encoded short fallbacks from source pixels. Stage1 is
    # deliberately LONG-ONLY: restore zero sentinels so unchanged original short
    # features are selected by the original multiscale and token-moment helpers.
    replacement[~valid] = 0
    return replacement


def materialize(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    old = root / ".runs/research_20260908/evidence_memory"
    paired = root / ".runs/research_20260908/native4k_paired_full/features"
    summary_path = output / "summary.json"
    if summary_path.exists():
        result = read_json(summary_path)
        for name, expected in result["output_sha256"].items():
            if file_sha256(output / name) != expected:
                raise RuntimeError("Completed source-swap data changed")
        for path, expected in result["input_sha256"].items():
            if file_sha256(root / path) != expected:
                raise RuntimeError("Source-swap input/code changed")
        return result
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Partial source-swap data retained; use a fresh version")
    provenance = {}

    def checked(path, expected=None):
        path = Path(path)
        actual = file_sha256(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Source identity changed: {path}")
        provenance[str(path.relative_to(root))] = actual
        return path

    old_lock = read_json(checked(old / "execution_lock.json"))
    for name, expected in old_lock["files"].items():
        checked(root / name, expected)
    data_lock = read_json(checked(old / "data/data_lock.json"))
    sources = data_lock["source_receipts"]

    def cached(name):
        receipt = sources[name]
        return np.load(checked(Path(receipt["path"]), receipt["sha256"]), mmap_mode="r")

    short_video = cached("vjepa21_real_clip")
    short_dino = cached("dinov2_native_frames")
    long_video = cached("vjepa21_long16_real_clip_vjepa21_long16_real_clip.npy")
    long_dino = cached("dinov2_long16_native_frames_dinov2_long16_native_frames.npy")
    with np.load(old / "data/memory_data.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    valid = data["long_valid"]
    verified = read_json(checked(paired / "independent_post_extraction_verification.json"))
    source_summary = read_json(checked(paired / "summary.json", verified["summary_sha256"]))
    binding = read_json(
        checked(
            paired / "execution_source_binding.json", verified["execution_source_binding_sha256"]
        )
    )
    checked(paired / binding["snapshot_path"], binding["snapshot_sha256"])
    checked(paired / "request.json", binding["request_sha256"])
    checked(paired / "model_receipt.json", binding["model_receipt_sha256"])
    if source_summary["status"] != "FULL_PAIRED_NATIVE4K_FEATURE_EXTRACTION_COMPLETE":
        raise RuntimeError("Source extraction is not complete")

    def paired_array(name):
        return np.load(
            checked(paired / name, source_summary["artifacts"][name]["sha256"]), mmap_mode="r"
        )

    regenerated = paired_array("exact4k_downsample720_grid3.npy")
    source_fallback = paired_array("native_source_fallback.npy")
    historical_short = paired_array("historical_short_fallback.npy")
    ids, frames = paired_array("sample_ids.npy"), paired_array("source_frames.npy")
    if (
        not np.array_equal(ids, data["sample_ids"])
        or not np.array_equal(historical_short, ~valid)
        or not paired_array("completed.npy").all()
        or len(ids) != 4977
        or int(historical_short.sum()) != 467
        or int(source_fallback.sum()) != 29
        or int((historical_short & source_fallback).sum()) != 27
    ):
        raise RuntimeError("Source fallback/cohort identity changed")
    selected_frames = []
    for key in ("short_frame_manifest", "long_frame_manifest"):
        receipt = sources[key]
        table = pd.read_csv(
            checked(Path(receipt["path"]), receipt["sha256"]),
            usecols=["sample_id", "time_index", "source_frame"],
        )
        table = table.set_index(["sample_id", "time_index"]).loc[
            pd.MultiIndex.from_product([ids, range(16)])
        ]
        selected_frames.append(table.source_frame.to_numpy().reshape(4977, 16))
    expected_frames = np.where(valid[:, None], selected_frames[1], selected_frames[0])
    if not np.array_equal(frames, expected_frames):
        raise RuntimeError("Regenerated source does not preserve original selected timestamps")
    original_memory, original_base = derive_features(
        short_video, long_video, short_dino, long_dino, valid
    )
    if not np.array_equal(original_memory, data["features"]):
        raise RuntimeError("Original 3072-dimensional memory features fail exact replay")
    with np.load(old / "data/base_features.npz", allow_pickle=False) as reference:
        for name, values in original_base.items():
            if not np.array_equal(values, reference[name]):
                raise RuntimeError(f"Original base feature derivation is not exact: {name}")
    replacement = masked_long_source(regenerated, long_video, valid, source_fallback)
    new_memory, new_base = derive_features(short_video, replacement, short_dino, long_dino, valid)
    unchanged = historical_short | source_fallback
    changed = valid & ~source_fallback
    for name, before, after in [("memory", original_memory, new_memory)] + [
        (name, original_base[name], new_base[name]) for name in original_base
    ]:
        if not np.array_equal(before[unchanged], after[unchanged]):
            raise RuntimeError(f"Fallback features changed in {name}")
        if not np.array_equal(np.any(before != after, axis=1), changed):
            raise RuntimeError(f"Unexpected changed-row dependency in {name}")
    unchanged_columns = np.r_[0:768, 1536:3072]
    if not np.array_equal(new_memory[:, unchanged_columns], original_memory[:, unchanged_columns]):
        raise RuntimeError("Source swap altered non-long-video memory features")
    data["features"] = new_memory
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "memory_data.npz", **data)
    np.savez_compressed(output / "base_features.npz", sample_ids=ids, **new_base)
    np.savez_compressed(
        output / "source_masks.npz",
        sample_ids=ids,
        historical_short_fallback=historical_short,
        native_source_fallback=source_fallback,
        changed_input=changed,
    )
    for path in (
        Path(__file__),
        root / "src/hac/video_multiscale.py",
        root / "src/hac/video_token_moments.py",
    ):
        checked(path)
    result = {
        "status": "SOURCE_SWAP_LONG_ONLY_MATERIALIZED_NO_FITS",
        "rows": 4977,
        "changed_input_rows": int(changed.sum()),
        "unchanged_input_rows": int(unchanged.sum()),
        "historical_short_rows": 467,
        "source_fallback_rows": 29,
        "fallback_overlap": 27,
        "valid_long_source_fallback_rows": 2,
        "old_feature_derivation_bit_exact": True,
        "nonfeature_metadata_unchanged": True,
        "original_short_and_all_dino_streams_unchanged": True,
        "dependency_map": {
            "long_vjepa_mean": "long_video mean",
            "dual_scale_vjepa_dino": "long_video mean and absolute long-short difference",
            "orthogonal_moments_posture": "long_video mean and time-mean spatial contrasts",
            "orthogonal_moments_motion": "long_video mean, scale difference, temporal differences and DCT magnitudes",
            "memory_features": "only float32 columns768:1536; other2304columns unchanged",
        },
        "base_estimators_affected_per_population": 4,
        "expected_nested_populations": 58,
        "new_estimator_fits_required": 3016,
        "input_sha256": provenance,
        "output_sha256": {
            name: file_sha256(output / name)
            for name in ("memory_data.npz", "base_features.npz", "source_masks.npz")
        },
        "new_model_fits": 0,
        "new_pixels_decoded": 0,
        "new_annotation_payload_reads": 0,
        "fallback_prediction_caution": "Features are unchanged on469rows, but newly fitted global weights may change their predictions.",
    }
    immutable_json(summary_path, result)
    return result
