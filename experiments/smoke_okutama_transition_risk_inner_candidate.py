"""Bounded one-fold startup smoke for the source/posture inner candidate path.

This is not a scientific result. It fits one posture head on one deterministic
inner training split, with a uniform anchor solely to exercise dimensions and
runtime. It never writes predictions or metrics and never indexes inner-held
labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.actor_memory_base import file_sha256
from hac.posture_witness import apply_projection, fit_posture_head, fit_projection
from hac.source_posture_cache import load_source_posture_cache

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz"
CACHE = ROOT / ".runs/research_20260913/source_posture_cache_v1"
PLAN = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/inner_cache_plan.json"
OUTPUT = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/inner_candidate_smoke.json"
CROP_KEYS = ("R_whole", "R_upper", "R_lower")


def run(output: Path) -> dict[str, object]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    with np.load(METADATA, allow_pickle=False) as saved:
        sample_ids = saved["sample_ids"]
        scenarios = saved["scenarios"]
        folds = saved["folds"]
        # Labels are indexed only for the inner-training rows below. The held
        # labels are deliberately never materialized by this smoke.
        labels = saved["labels"]
    outer = plan["outer_plans"][0]
    inner = outer["inner_folds"][0]
    outer_train = np.flatnonzero(folds != 0)
    assignment = outer["scenario_assignment"]
    held = outer_train[np.array([assignment[str(scenarios[row])] == 0 for row in outer_train])]
    train = np.setdiff1d(outer_train, held, assume_unique=True)
    if len(train) != inner["train_rows"] or len(held) != inner["held_rows"]:
        raise RuntimeError("Inner plan is not deterministic")
    if np.intersect1d(held, np.flatnonzero(folds == 0)).size:
        raise RuntimeError("Smoke held rows escape outer training")
    cache = load_source_posture_cache(
        CACHE,
        sample_ids,
        expected_crop_keys=(
            "J_whole", "J_upper", "J_lower", "R_whole", "R_upper", "R_lower",
            "N_whole", "N_upper", "N_lower", "R_whole_1p0", "R_whole_1p5",
        ),
    )
    descriptor, available = cache.descriptor(CROP_KEYS)
    start = time.perf_counter()
    projection = fit_projection(descriptor[train][available[train]])
    train_features = apply_projection(descriptor[train], projection)
    held_features = apply_projection(descriptor[held], projection)
    uniform_train = np.full((len(train), 3), 1.0 / 3.0, dtype=np.float64)
    head = fit_posture_head(
        train_features,
        labels[train],
        uniform_train,
        available=available[train],
    )
    elapsed = time.perf_counter() - start
    receipt: dict[str, object] = {
        "status": "TRANSITION_RISK_INNER_CANDIDATE_SMOKE_PASS",
        "metadata_sha256": file_sha256(METADATA),
        "cache_execution_lock_sha256": file_sha256(CACHE / "execution_lock.json"),
        "outer_fold": 0,
        "inner_fold": 0,
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "available_train_rows": int(available[train].sum()),
        "available_held_rows": int(available[held].sum()),
        "projection_dim": int(projection.output_dim),
        "head_coefficients": int(head.coefficients.size),
        "elapsed_seconds": float(elapsed),
        "uniform_anchor_only_for_smoke": True,
        "inner_held_labels_read": 0,
        "outer_held_labels_read": 0,
        "predictions_written": False,
        "metrics_written": False,
        "scientific_result": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
