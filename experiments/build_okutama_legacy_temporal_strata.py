"""Reproduce the old requested-long-window strata for comparison-only reporting.

These masks intentionally describe all 16 REQUESTED P3 timestamps, including
invalid long clips. They differ from the new data package's fallback-corrected
actual observation support and dense interval supervision. No fitting input is
created or changed by this comparison artifact.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hac.actor_memory_data import file_sha256
from hac.okutama_native_video import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_INDEX_SHA256,
    LABEL_INDEX,
    TARGET_LABEL,
    read_stable_annotations,
)


def build(root: Path, output: Path, archive_path: Path) -> dict:
    runs = root / ".runs/research_20260907"
    sources = {
        "index": (runs / "cptr_replay_r0/eligible_index.csv", EXPECTED_INDEX_SHA256),
        "long_manifest": (
            runs / "okutama_native_video_p3/manifest/frame_manifest.csv",
            "68a6f62dd0c033fd05421272345587f369da8d5ab04668d1f368477866c9cb9c",
        ),
        "original_audit_summary": (
            root / ".runs/research_20260908/temporal_opportunity/v2/summary.json",
            "d9bd3841cec9638083bb069e850680585d594af0dce4ec97b7e9b10c1c560947",
        ),
        "p6_evaluation_reference": (
            runs / "okutama_native_video_p6/results/oof_probabilities.npz",
            "5a2741dcf1617d0f584d3291c0d40050a380405ab265761697f938cf8bdd8acc",
        ),
    }
    receipts = {}
    for name, (path, expected) in sources.items():
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Original comparison source changed: {name}")
        receipts[name] = {"path": str(path), "sha256": actual}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "legacy_temporal_strata.npz"
    receipt_path = output / "legacy_temporal_strata_receipt.json"
    if destination.exists() or receipt_path.exists():
        raise RuntimeError("Refusing to overwrite existing comparison-only strata")
    index = pd.read_csv(
        sources["index"][0], dtype={"recording_id": str, "provider_recording_id": str}
    )
    index = index[
        index.scope.eq("grouped_crossfit_oof") & index.development_role.eq("train")
    ].reset_index(drop=True)
    ids = index.sample_id.to_numpy(dtype=str)
    if len(ids) != 4977 or len(set(ids)) != 4977:
        raise RuntimeError("Expected all 4977 unique original centers")
    long = pd.read_csv(sources["long_manifest"][0], dtype={"provider_recording_id": str})
    frames = {key: values.source_frame.to_numpy() for key, values in long.groupby("sample_id")}
    if set(frames) != set(ids) or any(len(values) != 16 for values in frames.values()):
        raise RuntimeError("Original long-window timestamp cohort changed")
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("Training archive size changed")
    observed = np.zeros(len(ids), bool)
    unknown = np.zeros(len(ids), bool)
    accessed: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for recording, positions in index.groupby("provider_recording_id").groups.items():
            stable = read_stable_annotations(archive, recording, accessed)
            for i in positions:
                row = index.loc[i]
                all_labels = set()
                for frame in frames[row.sample_id]:
                    annotation = stable.get((int(row.provider_track_id), int(frame)))
                    labels = (
                        {
                            LABEL_INDEX[TARGET_LABEL[a]]
                            for a in annotation.actions
                            if a in TARGET_LABEL
                        }
                        if annotation
                        else set()
                    )
                    all_labels.update(labels)
                    unknown[i] |= len(labels) != 1
                observed[i] = len(all_labels) > 1
    if len(accessed) != 42 or len(set(accessed)) != 42:
        raise RuntimeError("Unexpected annotation member access")
    boundary = observed & ~unknown
    stable_window = ~observed & ~unknown
    if (int(boundary.sum()), int(stable_window.sum()), int(unknown.sum()), int(observed.sum())) != (
        671,
        3730,
        576,
        772,
    ):
        raise RuntimeError("Requested-window strata do not reproduce the original audit")
    if not np.all(boundary.astype(int) + stable_window.astype(int) + unknown.astype(int) == 1):
        raise RuntimeError("Complete/unknown reporting strata are not a partition")
    with np.load(sources["p6_evaluation_reference"][0], allow_pickle=False) as original:
        if not np.array_equal(original["sample_ids"], ids):
            raise RuntimeError("Reference predictions and reporting masks are not aligned")
        probabilities = original["ocvc_uniform_diverse_triad"]
        labels = original["labels"]
    if not np.array_equal(labels, index.label_index):
        raise RuntimeError("Immutable center label mismatch")
    averaged = np.zeros_like(probabilities)
    for _, positions in index.groupby(
        ["provider_recording_id", "provider_track_id"]
    ).groups.items():
        rows = np.asarray(positions)
        times = index.loc[rows, "center_frame"].to_numpy()
        weights = (np.abs(times[:, None] - times[None, :]) <= 60).astype(float)
        averaged[rows] = weights @ probabilities[rows] / weights.sum(1, keepdims=True)
    old_errors = probabilities.argmax(1) != labels
    rescues = old_errors & (averaged.argmax(1) == labels)
    harms = ~old_errors & (averaged.argmax(1) != labels)
    if int((harms & boundary).sum()) != 73 or int((rescues & boundary).sum()) != 39:
        raise RuntimeError("Original fixed-mean boundary comparator did not reproduce")
    np.savez(
        destination,
        sample_ids=ids,
        complete_target_boundary=boundary,
        complete_stable_target_window=stable_window,
        unknown_target=unknown,
        observed_target_variation=observed,
    )
    receipt = {
        "status": "LEGACY_REQUESTED_WINDOW_COMPARISON_STRATA_REPRODUCED",
        "usage": "Comparison-only final reporting; never feature construction, fitting, selection or routing.",
        "definition": "All 16 requested P3 long timestamps without short-feature fallback. Complete boundary means one unambiguous target per slot and >1 target across slots. Missing/ambiguous slots define unknown_target.",
        "rows": len(ids),
        "complete_target_boundary": int(boundary.sum()),
        "complete_stable_target_window": int(stable_window.sum()),
        "unknown_target": int(unknown.sum()),
        "observed_target_variation": int(observed.sum()),
        "fixed_two_second_mean_boundary_rescues": int((rescues & boundary).sum()),
        "fixed_two_second_mean_boundary_harms": int((harms & boundary).sum()),
        "source_receipts": receipts,
        "script_sha256": file_sha256(Path(__file__)),
        "annotation_members_read": accessed,
        "raw_image_reads": 0,
        "model_fits": 0,
        "protected_rows_read": 0,
        "old_oof_use": "Read only to reproduce previously reported comparator harm counts; no model fitting.",
        "artifact_sha256": file_sha256(destination),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=root / ".runs/research_20260908/evidence_memory/data"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(r"C:\Users\DELL\hac_external_data\OkutamaAction\TrainSetFrames.zip"),
    )
    args = parser.parse_args()
    receipt = build(root, args.output_dir, args.archive)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "rows",
                    "complete_target_boundary",
                    "complete_stable_target_window",
                    "unknown_target",
                    "fixed_two_second_mean_boundary_rescues",
                    "fixed_two_second_mean_boundary_harms",
                    "artifact_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
