"""Identity-safe fixed-context actor memory data, with separate supervision.

Graph construction and source support never inspect action labels. Edge targets
inspect every annotation frame between adjacent observed centers; unknown frames
invalidate supervision rather than inventing continuity across an annotation gap.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hac.okutama_native_video import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPECTED_RECORDINGS,
    LABEL_INDEX,
    TARGET_LABEL,
    Annotation,
    read_stable_annotations,
)
from hac.video_multiscale import arm_features, derive_multiscale_features
from hac.video_token_moments import arm_inputs, derive_token_moments

OFFSETS_SECONDS = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)
FPS = 30
CENTER_SLOT = 2
QUALITY_NAMES = (
    "native_height_over_720",
    "native_width_over_1280",
    "native_area_fraction",
    "native_width_over_height",
    "log1p_native_height_pixels",
    "long_feature_valid",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_neighbor_graph(
    recordings: np.ndarray,
    tracks: np.ndarray,
    frames: np.ndarray,
    folds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Preserve five real-time slots and connect adjacent *observed* slots."""
    n = len(frames)
    if n == 0 or any(np.asarray(v).shape != (n,) for v in (recordings, tracks, frames, folds)):
        raise ValueError("Graph identities must be nonempty equal-length vectors")
    if not np.issubdtype(np.asarray(frames).dtype, np.integer):
        raise ValueError("Source frames must be integers")
    keys = [(str(r), str(t), int(f)) for r, t, f in zip(recordings, tracks, frames, strict=True)]
    if len(set(keys)) != n:
        raise ValueError("Duplicate recording/track/frame identity")
    lookup = {key: i for i, key in enumerate(keys)}
    track_folds: dict[tuple[str, str], int] = {}
    for i, (r, t, _) in enumerate(keys):
        if (r, t) in track_folds and track_folds[(r, t)] != int(folds[i]):
            raise ValueError("A provider track crosses folds")
        track_folds[(r, t)] = int(folds[i])
    neighbors = np.full((n, 5), -1, np.int64)
    previous = np.full((n, 5), -1, np.int64)
    elapsed = np.zeros((n, 5), np.float32)
    gap = np.zeros((n, 5), bool)
    for i, (r, t, frame) in enumerate(keys):
        for slot, offset in enumerate(OFFSETS_SECONDS):
            neighbors[i, slot] = lookup.get((r, t, frame + int(offset * FPS)), -1)
        observed = np.flatnonzero(neighbors[i] >= 0)
        for source, dest in zip(observed[:-1], observed[1:], strict=True):
            previous[i, dest] = source
            elapsed[i, dest] = OFFSETS_SECONDS[dest] - OFFSETS_SECONDS[source]
            gap[i, dest] = dest - source > 1
    if not np.array_equal(neighbors[:, CENTER_SLOT], np.arange(n)):
        raise RuntimeError("Graph does not preserve every original center")
    return {
        "neighbor_indices": neighbors,
        "valid": neighbors >= 0,
        "times": np.broadcast_to(OFFSETS_SECONDS, (n, 5)).copy(),
        "edge_source_slot": previous,
        "edge_dt_seconds": elapsed,
        "edge_bridges_missing_slot": gap,
    }


def pad_support(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    width = max(map(len, rows), default=0)
    frames = np.full((len(rows), width), -1, np.int32)
    valid = np.zeros_like(frames, bool)
    for i, row in enumerate(rows):
        if len(row) and (np.any(row < 0) or len(np.unique(row)) != len(row)):
            raise ValueError("Actual frame support must be unique and nonnegative")
        frames[i, : len(row)] = row
        valid[i, : len(row)] = True
    return frames, valid


def actual_support_union(
    short_frames: np.ndarray,
    long_frames: np.ndarray,
    long_valid: np.ndarray,
    graph: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Exact cached frame unions, including short fallback and missing centers."""
    n = len(long_valid)
    if short_frames.shape != (n, 16) or long_frames.shape != (n, 16):
        raise ValueError("Expected sixteen source timestamps in each cached clip")
    node = [
        np.unique(np.concatenate((short_frames[i], long_frames[i])))
        if long_valid[i]
        else np.unique(short_frames[i])
        for i in range(n)
    ]
    memory = [
        np.unique(np.concatenate([node[j] for j in neighbors if j >= 0]))
        for neighbors in graph["neighbor_indices"]
    ]
    node_frames, node_mask = pad_support(node)
    memory_frames, memory_mask = pad_support(memory)
    return {
        "node_support_frames": node_frames,
        "node_support_valid": node_mask,
        "memory_support_frames": memory_frames,
        "memory_support_valid": memory_mask,
        "memory_support_start_frame": np.asarray([x.min() for x in memory], np.int32),
        "memory_support_end_frame": np.asarray([x.max() for x in memory], np.int32),
        "memory_support_span_seconds": np.asarray(
            [(x.max() - x.min()) / FPS for x in memory], np.float32
        ),
    }


def annotation_target(annotation: Annotation | None) -> int:
    """Return target 0..2, missing -1, or ambiguous/unsupported -2."""
    if annotation is None or annotation.lost:
        return -1
    labels = {LABEL_INDEX[TARGET_LABEL[a]] for a in annotation.actions if a in TARGET_LABEL}
    return next(iter(labels)) if len(labels) == 1 else -2


def target_interval(
    stable: dict[tuple[int, int], Annotation], track: int, start: int, end: int
) -> dict[str, Any]:
    """Inspect ALL inclusive frames, including a return to the initial state."""
    if start > end:
        raise ValueError("Target interval endpoints are reversed")
    labels = np.asarray([annotation_target(stable.get((track, f))) for f in range(start, end + 1)])
    known = labels >= 0
    complete = bool(known.all())
    variation = len(np.unique(labels[known])) > 1
    return {
        "complete": complete,
        "boundary": bool(complete and variation),
        "observed_variation": variation,
        "missing_count": int((labels == -1).sum()),
        "ambiguous_count": int((labels == -2).sum()),
        "frames": len(labels),
        "labels": labels,
    }


def derive_annotation_supervision(
    index: pd.DataFrame,
    graph: dict[str, np.ndarray],
    support: dict[str, np.ndarray],
    archive_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = len(index)
    targets = {
        "boundary_targets": np.zeros((n, 5), np.float32),
        "boundary_valid": np.zeros((n, 5), bool),
        "boundary_missing_frames": np.zeros((n, 5), np.int16),
        "boundary_ambiguous_frames": np.zeros((n, 5), np.int16),
    }
    for prefix in ("node_support", "memory_support", "node_interval", "memory_interval"):
        targets[f"{prefix}_complete"] = np.zeros(n, bool)
        targets[f"{prefix}_boundary"] = np.zeros(n, bool)
        targets[f"{prefix}_observed_variation"] = np.zeros(n, bool)
    targets["node_purity_target"] = np.zeros(n, np.float32)
    targets["node_purity_valid"] = np.zeros(n, bool)
    accessed: list[str] = []
    annotation_hashes: dict[str, str] = {}
    center_checks = 0
    with zipfile.ZipFile(archive_path) as archive:
        for recording, row_ids in index.groupby("provider_recording_id").groups.items():
            stable = read_stable_annotations(archive, str(recording), accessed)
            # Hash exactly the same permitted payloads; no archive-wide decompression.
            for member in accessed[-2:]:
                annotation_hashes[member] = hashlib.sha256(archive.read(member)).hexdigest()
            interval_cache: dict[tuple[int, int, int], dict] = {}

            def interval(
                track: int,
                start: int,
                end: int,
                *,
                cache: dict = interval_cache,
                annotations: dict = stable,
            ) -> dict:
                key = (track, start, end)
                if key not in cache:
                    cache[key] = target_interval(annotations, track, start, end)
                return cache[key]

            for i in row_ids:
                row = index.loc[i]
                track, frame = int(row.provider_track_id), int(row.center_frame)
                if annotation_target(stable.get((track, frame))) != int(row.label_index):
                    raise RuntimeError("An immutable center label disagrees with annotations")
                center_checks += 1
                for dest, source in enumerate(graph["edge_source_slot"][i]):
                    if source < 0:
                        continue
                    left = graph["neighbor_indices"][i, source]
                    right = graph["neighbor_indices"][i, dest]
                    info = interval(
                        track,
                        int(index.loc[left, "center_frame"]),
                        int(index.loc[right, "center_frame"]),
                    )
                    targets["boundary_targets"][i, dest] = float(info["boundary"])
                    targets["boundary_valid"][i, dest] = info["complete"]
                    targets["boundary_missing_frames"][i, dest] = info["missing_count"]
                    targets["boundary_ambiguous_frames"][i, dest] = info["ambiguous_count"]
                for kind in ("node", "memory"):
                    values = support[f"{kind}_support_frames"][
                        i, support[f"{kind}_support_valid"][i]
                    ]
                    labels = np.asarray(
                        [annotation_target(stable.get((track, int(f)))) for f in values]
                    )
                    complete = bool((labels >= 0).all())
                    variation = len(np.unique(labels[labels >= 0])) > 1
                    targets[f"{kind}_support_complete"][i] = complete
                    targets[f"{kind}_support_boundary"][i] = complete and variation
                    targets[f"{kind}_support_observed_variation"][i] = variation
                    info = interval(track, int(values.min()), int(values.max()))
                    targets[f"{kind}_interval_complete"][i] = info["complete"]
                    targets[f"{kind}_interval_boundary"][i] = info["boundary"]
                    targets[f"{kind}_interval_observed_variation"][i] = info["observed_variation"]
                    if kind == "node":
                        # Purity describes the actual input frames, not a predicted label.
                        targets["node_purity_valid"][i] = complete
                        targets["node_purity_target"][i] = (
                            float(np.mean(labels == int(row.label_index))) if complete else 0.0
                        )
    if center_checks != n or len(accessed) != 42 or len(set(accessed)) != 42:
        raise RuntimeError("Expected exactly 42 allowed annotation members and all centers")
    targets["edge_boundary"] = targets["boundary_targets"].copy()
    targets["edge_label_valid"] = targets["boundary_valid"].copy()
    return targets, {
        "annotation_members": accessed,
        "annotation_member_sha256": annotation_hashes,
        "unique_annotation_members_read": len(accessed),
        "annotation_payload_reads": 2 * len(accessed),
        "immutable_center_labels_verified": center_checks,
        "edge_target_definition": "Any target-class transition across every inclusive frame between previous observed center and destination. Valid only when all frames have exactly one mapped target class.",
        "unknown_policy": "Missing, lost, unsupported or ambiguous annotations invalidate targets; masks never change inference graph or frame support.",
    }


def materialize_memory_data(root: Path, output: Path, archive_path: Path) -> dict[str, Any]:
    """Materialize the fixed 4977-row feature/data package without fitting models."""
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Refusing to overwrite a nonempty memory data directory")
    root, output, archive_path = root.resolve(), output.resolve(), archive_path.resolve()
    runs = root / ".runs/research_20260907"
    source_receipts: dict[str, dict] = {}

    def checked(name: str, path: Path, expected: str | None = None) -> Path:
        actual = file_sha256(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Source digest changed: {name}")
        source_receipts[name] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
        return path

    p5_path = checked(
        "p5_lock",
        runs / "okutama_native_video_p5/execution_lock.json",
        "464b086cb4a3deece080cb1455346dd8faa1b429acbb4fe9d46e08a3c22252ff",
    )
    p5 = json.loads(p5_path.read_text(encoding="utf-8"))
    p2_receipt = p5["p3_reference"]["p2_execution_lock"]
    p2 = json.loads(
        checked("p2_lock", root / p2_receipt["path"], p2_receipt["sha256"]).read_text(
            encoding="utf-8"
        )
    )
    index_path = checked(
        "eligible_index", runs / "cptr_replay_r0/eligible_index.csv", EXPECTED_INDEX_SHA256
    )
    index = pd.read_csv(
        index_path,
        dtype={
            "recording_id": str,
            "provider_recording_id": str,
            "provider_track_id": str,
            "track_id": str,
        },
    )
    index = index[
        index.scope.eq("grouped_crossfit_oof") & index.development_role.eq("train")
    ].reset_index(drop=True)
    if (
        len(index) != 4977
        or index.sample_id.duplicated().any()
        or set(index.provider_recording_id) != EXPECTED_RECORDINGS
    ):
        raise RuntimeError("Primary cohort changed")
    ids = index.sample_id.to_numpy(dtype=str)
    cached: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    for short_key, name in (("video", "vjepa21_real_clip"), ("dino", "dinov2_native_frames")):
        cache = p2["caches"][short_key]
        receipt = cache["features"][name]
        summary = json.loads(
            checked(
                f"{name}_summary", root / cache["summary"]["path"], cache["summary"]["sha256"]
            ).read_text(encoding="utf-8")
        )
        if summary["sample_ids"] != ids.tolist():
            raise RuntimeError("Short cache row identity mismatch")
        cached[name] = np.load(
            checked(name, root / receipt["path"], receipt["sha256"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        v = np.load(
            checked(
                f"{name}_valid", root / receipt["validity"]["path"], receipt["validity"]["sha256"]
            ),
            allow_pickle=False,
        )
        validity[name] = v[:, receipt["validity_column"]] if v.ndim == 2 else v
        if not validity[name].all():
            raise RuntimeError("The short anchor is incomplete")
    for cache in p5["p3_long_caches"].values():
        name = cache["arm"]
        artifacts = cache["artifacts"]
        summary = json.loads(
            checked(
                f"{name}_summary",
                root / artifacts["summary.json"]["path"],
                artifacts["summary.json"]["sha256"],
            ).read_text(encoding="utf-8")
        )
        if summary["sample_ids"] != ids.tolist():
            raise RuntimeError("Long cache row identity mismatch")
        for filename in (f"{name}.npy", "validity.npy", "completed.npy"):
            receipt = artifacts[filename]
            path = checked(f"{name}_{filename}", root / receipt["path"], receipt["sha256"])
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            if filename == f"{name}.npy":
                cached[name] = value
            elif filename == "validity.npy":
                validity[name] = value
            elif not value.all():
                raise RuntimeError("Long cache extraction did not complete")
    manifests = {}
    for key, phase, expected in (
        (
            "short",
            "okutama_native_video_p0_r1",
            "b69fe9e00057ff535698afb02cb3e9031ce3f53246c7e7ba64f061d1d557427f",
        ),
        (
            "long",
            "okutama_native_video_p3",
            "68a6f62dd0c033fd05421272345587f369da8d5ab04668d1f368477866c9cb9c",
        ),
    ):
        frame_path = checked(
            f"{key}_frame_manifest", runs / phase / "manifest/frame_manifest.csv", expected
        )
        frame_table = pd.read_csv(frame_path, dtype={"provider_recording_id": str})
        if set(frame_table.sample_id) != set(ids) or len(frame_table) != 16 * len(index):
            raise RuntimeError("Frame manifest is not the fixed center cohort")
        manifests[key] = (
            frame_table.set_index(["sample_id", "time_index"])
            .loc[pd.MultiIndex.from_product([ids, range(16)])]
            .reset_index()
        )
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("Training archive size changed")
    checked("training_archive", archive_path, EXPECTED_ARCHIVE_SHA256)
    checked("data_module", Path(__file__).resolve())
    builder = root / "experiments/build_okutama_memory_data.py"
    if builder.exists():
        checked("data_builder", builder)
    output.mkdir(parents=True, exist_ok=True)
    lock = {
        "status": "MEMORY_DATA_LOCKED_BEFORE_ANNOTATION_SUPERVISION_OR_FEATURE_DERIVATION",
        "source_receipts": source_receipts,
        "center_sample_ids": ids.tolist(),
        "offsets_seconds": OFFSETS_SECONDS.tolist(),
        "labels_used_in_graph_or_feature_derivation": False,
        "historical_oof_predictions_read": False,
        "model_fits": 0,
    }
    (output / "data_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    sv, lv = cached["vjepa21_real_clip"], cached["vjepa21_long16_real_clip"]
    sd, ld = cached["dinov2_native_frames"], cached["dinov2_long16_native_frames"]
    multiscale = derive_multiscale_features(
        sv,
        lv,
        sd,
        ld,
        validity["vjepa21_long16_real_clip"],
        validity["dinov2_long16_native_frames"],
    )
    moments = derive_token_moments(multiscale, sv, lv, sd, ld)
    ostm = arm_inputs(moments, "orthogonal_moments_factorized")
    dual, _ = arm_features(multiscale, "dual_scale_vjepa_dino")
    base = {
        "sample_ids": ids,
        "long_vjepa_mean": multiscale.long_v,
        "dual_scale_vjepa_dino": dual,
        "orthogonal_moments_posture": ostm.posture_or_direct,
        "orthogonal_moments_motion": ostm.motion_references[0],
    }
    center = manifests["short"].loc[manifests["short"].time_index.eq(8)]
    height = (center.bbox_ymax - center.bbox_ymin).to_numpy(np.float32)
    width = (center.bbox_xmax - center.bbox_xmin).to_numpy(np.float32)
    quality = np.column_stack(
        (
            height / 720,
            width / 1280,
            height * width / (720 * 1280),
            width / height,
            np.log1p(height),
            multiscale.long_valid,
        )
    ).astype(np.float32)
    folds = index.fold.str.removeprefix("fold-").to_numpy(np.int64)
    graph = build_neighbor_graph(
        index.provider_recording_id.to_numpy(),
        index.provider_track_id.to_numpy(),
        index.center_frame.to_numpy(),
        folds,
    )
    support = actual_support_union(
        manifests["short"].source_frame.to_numpy().reshape(-1, 16),
        manifests["long"].source_frame.to_numpy().reshape(-1, 16),
        multiscale.long_valid,
        graph,
    )
    supervision, label_audit = derive_annotation_supervision(index, graph, support, archive_path)
    arrays = {
        "features": np.concatenate(
            (multiscale.short_v, multiscale.long_v, multiscale.short_d, multiscale.long_d), axis=1
        ),
        "quality": quality,
        "labels": index.label_index.to_numpy(np.int64),
        "scenarios": index.recording_id.to_numpy(dtype=str),
        "folds": folds,
        "sample_ids": ids,
        "recordings": index.provider_recording_id.to_numpy(dtype=str),
        "tracks": index.provider_track_id.to_numpy(dtype=str),
        "frames": index.center_frame.to_numpy(np.int32),
        "long_valid": multiscale.long_valid,
        **graph,
        **support,
        **supervision,
    }
    if (
        arrays["features"].shape != (4977, 3072)
        or not np.isfinite(arrays["features"]).all()
        or not np.isfinite(quality).all()
    ):
        raise RuntimeError("Memory feature contract failed")
    np.savez(output / "memory_data.npz", **arrays)
    np.savez(output / "base_features.npz", **base)
    summary = {
        "status": "MEMORY_DATA_MATERIALIZED_NO_MODEL_TRAINING",
        "rows": len(index),
        "scenarios": int(index.recording_id.nunique()),
        "tracks": int(index.track_id.nunique()),
        "feature_recipe": "float32 short video mean, long video mean, short DINO mean, long DINO mean; exact short fallback for invalid long inputs",
        "quality_names": QUALITY_NAMES,
        "inference_inputs": [
            "features",
            "quality",
            "neighbor_indices",
            "times",
            "valid",
            "edge_source_slot",
            "edge_dt_seconds",
        ],
        "label_derived_arrays_are_supervision_or_diagnostics_only": True,
        "model_fits": 0,
        "raw_image_reads": 0,
        "protected_rows_read": 0,
        "historical_oof_predictions_read": False,
        "annotations": label_audit,
        "graph": {
            "observed_slots": int(graph["valid"].sum()),
            "observed_edges": int((graph["edge_source_slot"] >= 0).sum()),
            "edges_bridging_missing_slot": int(graph["edge_bridges_missing_slot"].sum()),
            "all_original_centers_preserved": True,
        },
        "supervision": {
            "valid_edges": int(supervision["boundary_valid"].sum()),
            "positive_edges": int(supervision["boundary_targets"].sum()),
            "node_purity_valid": int(supervision["node_purity_valid"].sum()),
        },
        "diagnostics": {
            key: int(value.sum())
            for key, value in supervision.items()
            if value.shape == (len(index),) and value.dtype == bool
        },
        "support": {
            "actual_frame_union_min": int(support["memory_support_valid"].sum(1).min()),
            "actual_frame_union_max": int(support["memory_support_valid"].sum(1).max()),
            "span_seconds_min": float(support["memory_support_span_seconds"].min()),
            "span_seconds_max": float(support["memory_support_span_seconds"].max()),
            "timing": "Nominal source frame/30; memory uses future and past observations and includes actual cached feature receptive-field support.",
        },
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "base_arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in base.items()
        },
        "artifacts": {
            name: {
                "sha256": file_sha256(output / name),
                "size_bytes": (output / name).stat().st_size,
            }
            for name in ("data_lock.json", "memory_data.npz", "base_features.npz")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
