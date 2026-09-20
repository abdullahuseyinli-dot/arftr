"""Retrospective, no-fit audit of temporal evidence in the fixed P6 development rows.

The oracle uses labels ONLY to measure unavailable selection headroom. The single
fixed +/-2-second mean uses track identity, nominal timestamps and probabilities,
never labels. It expands inference context and is not a promoted model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss

from hac.okutama_native_video import (
    EXPECTED_RECORDINGS,
    LABEL_INDEX,
    TARGET_LABEL,
    read_stable_annotations,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".runs/research_20260907"
EXPECTED = {
    "p8": "e649f35a22ccd1c7deb7c3bee62cc762a9beb22e3190672118d44e03a8dfdbe4",
    "p6": "5a2741dcf1617d0f584d3291c0d40050a380405ab265761697f938cf8bdd8acc",
    "index": "5bf6d0cc11a18d3e986b714f8a3de71cadd9fef3bd233eb9b5f212f701aadbbb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "rows": int(len(y)),
        "macro_f1": float(
            f1_score(y, p.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "accuracy": float(np.mean(p.argmax(1) == y)),
        "nll": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
        "confusion": confusion_matrix(y, p.argmax(1), labels=[0, 1, 2]).tolist(),
    }


def run(archive_path: Path) -> tuple[dict, pd.DataFrame]:
    sources = {
        "p8": RUNS / "okutama_native_video_p8/results/oof_probabilities.npz",
        "p6": RUNS / "okutama_native_video_p6/results/oof_probabilities.npz",
        "index": RUNS / "cptr_replay_r0/eligible_index.csv",
        "short_frames": RUNS / "okutama_native_video_p0_r1/manifest/frame_manifest.csv",
        "long_frames": RUNS / "okutama_native_video_p3/manifest/frame_manifest.csv",
        "script": Path(__file__).resolve(),
    }
    hashes = {key: sha256(path) for key, path in sources.items()}
    for key, expected in EXPECTED.items():
        if hashes[key] != expected:
            raise RuntimeError(f"Immutable source changed: {key}")
    with np.load(sources["p8"], allow_pickle=False) as bundle:
        d = {key: bundle[key] for key in bundle.files}
    with np.load(sources["p6"], allow_pickle=False) as original:
        for key in ("sample_ids", "labels", "folds", "recording_ids"):
            if not np.array_equal(d[key], original[key]):
                raise RuntimeError(f"P6/P8 identity mismatch: {key}")
        if not np.array_equal(d["p6_reference"], original["ocvc_uniform_diverse_triad"]):
            raise RuntimeError("P6 probabilities changed")
    ids = d["sample_ids"]
    if len(ids) != 4977 or len(set(ids.tolist())) != len(ids):
        raise RuntimeError("Expected exactly 4977 unique development sample IDs")
    t = pd.read_csv(
        sources["index"], dtype={"recording_id": str, "provider_recording_id": str, "track_id": str}
    )
    t = t.loc[t.scope.eq("grouped_crossfit_oof") & t.development_role.eq("train")].copy()
    if set(t.sample_id) != set(ids) or t.sample_id.duplicated().any():
        raise RuntimeError("Index is not an exact sample-ID bijection")
    t = t.set_index("sample_id").loc[ids].reset_index()
    y, p = d["labels"], d["p6_reference"]
    if not np.array_equal(y, t.label_index) or set(t.provider_recording_id) != EXPECTED_RECORDINGS:
        raise RuntimeError("Labels or authorized recording identities changed")
    if not np.allclose(p.sum(1), 1) or not np.isfinite(p).all():
        raise RuntimeError("Malformed probabilities")
    pred = p.argmax(1)
    err = pred != y
    components = [
        "p3_long_vjepa_mean",
        "p3_dual_scale_vjepa_dino",
        "p5_orthogonal_moments_factorized",
    ]
    shared = np.logical_and.reduce([d[key].argmax(1) != y for key in components])
    if int(err.sum()) != 821 or int(shared.sum()) != 508:
        raise RuntimeError("Reference error partition changed")
    for _, ix in t.groupby("track_id").groups.items():
        if len(set(d["folds"][np.asarray(ix)])) != 1:
            raise RuntimeError("A track crosses outer folds")

    short = pd.read_csv(sources["short_frames"])
    center = short.loc[short.offset_seconds == 0].set_index("sample_id")
    if center.index.duplicated().any() or set(center.index) != set(ids):
        raise RuntimeError("Center-frame geometry identity mismatch")
    center = center.loc[ids]
    t["height"] = (center.bbox_ymax - center.bbox_ymin).to_numpy()
    t["errors"], t["shared_errors"] = err, shared
    track = t.groupby("track_id").agg(
        rows=("errors", "size"), errors=("errors", "sum"), shared_errors=("shared_errors", "sum")
    )
    track = track.sort_values(["shared_errors", "errors", "rows"], ascending=False, kind="stable")

    horizons = (1, 2, 3, 5)
    counts = {
        h: {
            key: 0
            for key in (
                "rows_with_neighbor",
                "shared_rows_with_neighbor",
                "oracle_rescues",
                "oracle_shared_rescues",
                "same_label_neighbor_oracle_rescues",
                "same_label_neighbor_oracle_shared_rescues",
                "two_same_label_correct_neighbors_shared_rescues",
            )
        }
        for h in horizons
    }
    q = np.zeros_like(p)
    neighbor_counts = np.zeros(len(t), dtype=int)
    for _, ix in t.groupby("track_id").groups.items():
        ii = np.asarray(ix)
        frame = t.loc[ii, "center_frame"].to_numpy()
        distance = np.abs(frame[:, None] - frame[None, :])
        for h in horizons:
            mask = (distance <= 30 * h) & (distance > 0)
            # Each ROW is the target center, each COLUMN a possible observation.
            correct_for_center = mask & (pred[ii][None, :] == y[ii][:, None])
            same_label_correct = correct_for_center & (y[ii][None, :] == y[ii][:, None])
            c = counts[h]
            c["rows_with_neighbor"] += int(mask.any(1).sum())
            c["shared_rows_with_neighbor"] += int((mask.any(1) & shared[ii]).sum())
            c["oracle_rescues"] += int((correct_for_center.any(1) & err[ii]).sum())
            c["oracle_shared_rescues"] += int((correct_for_center.any(1) & shared[ii]).sum())
            c["same_label_neighbor_oracle_rescues"] += int(
                (same_label_correct.any(1) & err[ii]).sum()
            )
            c["same_label_neighbor_oracle_shared_rescues"] += int(
                (same_label_correct.any(1) & shared[ii]).sum()
            )
            c["two_same_label_correct_neighbors_shared_rescues"] += int(
                ((same_label_correct.sum(1) >= 2) & shared[ii]).sum()
            )
        averaging_mask = (distance <= 60).astype(float)
        q[ii] = averaging_mask @ p[ii] / averaging_mask.sum(1, keepdims=True)
        neighbor_counts[ii] = averaging_mask.sum(1).astype(int) - 1
    new_pred = q.argmax(1)
    rescued, harmed = err & (new_pred == y), ~err & (new_pred != y)

    def changes(mask: np.ndarray) -> dict:
        return {
            "rows": int(mask.sum()),
            "baseline_errors": int((err & mask).sum()),
            "shared_errors": int((shared & mask).sum()),
            "rescues": int((rescued & mask).sum()),
            "harms": int((harmed & mask).sum()),
            "shared_rescues": int((rescued & shared & mask).sum()),
            "baseline": metrics(y[mask], p[mask]),
            "fixed_mean": metrics(y[mask], q[mask]),
        }

    # Exact target labels at the EXISTING 16 long-clip requested timestamps.
    # No images, protected recordings, new centers, or fitting are accessed.
    long = pd.read_csv(sources["long_frames"], dtype={"provider_recording_id": str})
    if set(long.sample_id) != set(ids) or set(long.provider_recording_id) != EXPECTED_RECORDINGS:
        raise RuntimeError("Long-frame manifest cohort changed")
    frames = {key: group.source_frame.to_numpy() for key, group in long.groupby("sample_id")}
    if any(len(value) != 16 for value in frames.values()):
        raise RuntimeError("Expected 16 requested frame slots per center")
    boundary, unknown = np.zeros(len(t), bool), np.zeros(len(t), bool)
    accessed: list[str] = []
    center_mismatches = 0
    with zipfile.ZipFile(archive_path) as archive:
        for recording, ix in t.groupby("provider_recording_id").groups.items():
            stable = read_stable_annotations(archive, recording, accessed)
            for i in ix:
                row = t.loc[i]
                labels = set()
                for frame in frames[row.sample_id]:
                    annotation = stable.get((int(row.provider_track_id), int(frame)))
                    values = (
                        {
                            LABEL_INDEX[TARGET_LABEL[a]]
                            for a in annotation.actions
                            if a in TARGET_LABEL
                        }
                        if annotation
                        else set()
                    )
                    labels.update(values)
                    unknown[i] |= len(values) != 1
                boundary[i] = len(labels) > 1
                annotation = stable.get((int(row.provider_track_id), int(row.center_frame)))
                values = (
                    {LABEL_INDEX[TARGET_LABEL[a]] for a in annotation.actions if a in TARGET_LABEL}
                    if annotation
                    else set()
                )
                center_mismatches += int(values != {int(y[i])})
    if center_mismatches or len(accessed) != 42:
        raise RuntimeError("Annotation identity audit failed")

    legacy_transition = t.transition_window.to_numpy(dtype=bool)
    legacy_occlusion = t.window_any_occluded.to_numpy(dtype=bool)
    strata = {
        "legacy_clear_stable": ~legacy_transition & ~legacy_occlusion,
        "legacy_transition": legacy_transition,
        "observed_target_boundary": boundary,
        "complete_target_boundary": boundary & ~unknown,
        "complete_stable_target_window": ~boundary & ~unknown,
        "unknown_or_ambiguous_window": unknown,
    }
    scenario_results = {
        str(key): changes(t.recording_id.eq(key).to_numpy())
        for key in sorted(t.recording_id.unique())
    }
    pair_recordings = {
        str(key): sorted(group.provider_recording_id.unique().tolist())
        for key, group in t.groupby("recording_id")
        if group.provider_recording_id.nunique() == 2
    }
    same_time_rows, common_keys = 0, 0
    for recording_pair in pair_recordings.values():
        a, b = [t[t.provider_recording_id.eq(v)] for v in recording_pair]
        common = set(a.center_frame) & set(b.center_frame)
        common_keys += len(common)
        same_time_rows += int(a.center_frame.isin(common).sum() + b.center_frame.isin(common).sum())
    size = t.assign(
        height_band=pd.cut(
            t.height, [0, 32, 64, np.inf], labels=["at_most_32", "32_to_64", "above_64"]
        )
    )
    result = {
        "status": "RETROSPECTIVE_NO_FIT_TEMPORAL_OPPORTUNITY_AUDIT",
        "interpretation": "Development diagnostics; horizons were inspected post hoc. Oracle is not deployable accuracy. The only evaluated label-free inference rule is fixed +/-2s arithmetic mean, including center, with original rows preserved; no tuning or promotion.",
        "inference_context": "Neighbor centers within 60 nominal source frames, same provider track. Neighbor predictions already consume approximately +/-1s clips, so total available visual support can approach +/-3s; future access is used. Timestamps are frame/30, not measured PTS.",
        "sources": {
            key: {"path": str(path), "sha256": hashes[key]} for key, path in sources.items()
        },
        "archive": {
            "path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "payload_members_read": accessed,
            "archive_digest_recomputed": False,
        },
        "model_fits": 0,
        "raw_image_reads": 0,
        "protected_rows_read": 0,
        "rows": len(t),
        "tracks": len(track),
        "scenarios": int(t.recording_id.nunique()),
        "baseline": metrics(y, p),
        "shared_error_count": int(shared.sum()),
        "shared_error_confusion": confusion_matrix(
            y[shared], pred[shared], labels=[0, 1, 2]
        ).tolist(),
        "oracles_by_horizon_seconds": counts,
        "fixed_two_second_mean": {
            **changes(np.ones(len(t), bool)),
            "mean_neighbor_count": float(neighbor_counts.mean()),
            "rows_without_neighbor": int((neighbor_counts == 0).sum()),
        },
        "strata": {key: changes(mask) for key, mask in strata.items()},
        "scenario_results": scenario_results,
        "target_transition_audit": {
            "scope": "Observed target-class variation at the existing 16 long-clip timestamps, not all intervening frames. Observed variation and unknown strata overlap; complete_target_boundary requires all 16 slots to have one unambiguous target class and more than one class across slots.",
            "center_label_mismatches": center_mismatches,
            "legacy_boundary_rows": int(legacy_transition.sum()),
            "target_boundary_rows": int(boundary.sum()),
            "new_boundary_without_legacy_flag": int((boundary & ~legacy_transition).sum()),
            "legacy_flag_without_observed_target_boundary": int(
                (legacy_transition & ~boundary).sum()
            ),
            "boundary_and_unknown_overlap": int((boundary & unknown).sum()),
        },
        "track_concentration_sorted_by_shared_errors": {
            str(k): {
                name: int(track.head(k)[name].sum()) for name in ("rows", "errors", "shared_errors")
            }
            for k in (5, 10, 20, 40, 80)
        },
        "center_actor_height_quantiles_720p": {
            str(k): float(v)
            for k, v in t.height.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).items()
        },
        "height_bands": size.groupby("height_band", observed=True)
        .agg(
            rows=("errors", "size"),
            errors=("errors", "sum"),
            shared_errors=("shared_errors", "sum"),
        )
        .reset_index()
        .to_dict("records"),
        "cross_view_upper_bound": {
            "scenario_pairs": pair_recordings,
            "rows_in_paired_scenarios": int(t.recording_id.isin(pair_recordings).sum()),
            "center_rows_at_selected_timestamp_in_both_views": same_time_rows,
            "common_scenario_timestamp_keys": common_keys,
            "qualification": "Recording-pair/time overlap only. No inter-drone actor identity mapping or image-level synchronization has been established; track IDs must not be equated.",
        },
    }
    return result, track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(r"C:\Users\DELL\hac_external_data\OkutamaAction\TrainSetFrames.zip"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir and any(
        (args.output_dir / name).exists() for name in ("summary.json", "track_aggregates.csv")
    ):
        raise RuntimeError("Refusing to overwrite an existing diagnostic artifact")
    result, tracks = run(args.archive)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        tracks.to_csv(args.output_dir / "track_aggregates.csv")
    print(
        json.dumps(
            result
            if not args.output_dir
            else {
                "output": str(args.output_dir),
                "status": result["status"],
                "fixed_two_second_mean": result["fixed_two_second_mean"],
                "oracles_by_horizon_seconds": result["oracles_by_horizon_seconds"],
                "target_transition_audit": result["target_transition_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
