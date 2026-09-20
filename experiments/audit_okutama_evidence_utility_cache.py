"""Seal and summarize the full observation-only aligned-evidence cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_okutama_evidence_utility as trial

from hac.actor_memory_base import file_sha256

ROOT = Path(__file__).resolve().parents[1]
DINO_SUMMARY = ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json"


def audit(cache_root: Path, output: Path) -> dict:
    sample_ids = np.asarray(trial.read_json(DINO_SUMMARY)["sample_ids"], dtype=str)
    store = trial.EvidenceStore(cache_root, sample_ids)
    summaries = []
    feature_bytes = 0
    for record in store.receipts:
        directory = ROOT / record["directory"]
        summaries.append(trial.read_json(directory / "summary.json"))
        feature_bytes += (directory / "features.npz").stat().st_size
    rows = sum(value["rows"] for value in summaries)
    receipt = {
        "status": "EVIDENCE_UTILITY_OBSERVATION_CACHE_AUDIT_PASS",
        "rows": rows,
        "shards": len(summaries),
        "source_views_encoded": sum(value["views_encoded"] for value in summaries),
        "unmasked_affine_fits": sum(value["unmasked_affine_fits"] for value in summaries),
        "valid_affine_fits": sum(value["valid_affine_fits"] for value in summaries),
        "rows_with_all_four_valid": sum(value["all_four_valid_rows"] for value in summaries),
        "weighted_mean_coverage": float(
            sum(value["coverage_mean"] * value["rows"] for value in summaries) / rows
        ),
        "compressed_feature_bytes": feature_bytes,
        "assembled_shapes": {
            "context": list(store.context.shape),
            "center_bins": list(store.center_bins.shape),
            "contrasts": list(store.contrasts.shape),
            "coverage": list(store.coverage.shape),
        },
        "B0_contrast_exact_zero": bool(np.all(store.contrasts[:, 0] == 0)),
        "all_values_finite": bool(
            all(
                np.isfinite(value).all()
                for value in (store.context, store.center_bins, store.contrasts, store.coverage)
            )
        ),
        "task_labels_read": 0,
        "ARFTR_outputs_read": 0,
        "annotation_support_fields_used": [],
        "shard_receipts": store.receipts,
        "auditor_sha256": file_sha256(Path(__file__)),
    }
    if rows != 4977 or receipt["source_views_encoded"] != 4977 * 5:
        raise RuntimeError("Observation cache population is incomplete")
    trial.write_json_exclusive(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=trial.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    cache_root.relative_to(ROOT.resolve())
    output = args.output.resolve() if args.output else cache_root / "cache_audit_receipt.json"
    output.relative_to(ROOT.resolve())
    print(json.dumps(audit(cache_root, output), indent=2), flush=True)


if __name__ == "__main__":
    main()
