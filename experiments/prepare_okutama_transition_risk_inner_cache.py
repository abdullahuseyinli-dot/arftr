"""Prepare deterministic, scenario-grouped inner folds for router training.

This command is planning-only. It does not fit a candidate, read labels, or
select a policy. Its purpose is to remove the remaining ambiguity before any
inner source/posture predictions are materialized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.actor_memory_base import canonical_hash, file_sha256

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz"
ROUTER_LOCK = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/execution_lock.json"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/inner_cache_plan.json"


def prepare(output: Path) -> dict[str, object]:
    with np.load(METADATA, allow_pickle=False) as saved:
        sample_ids = saved["sample_ids"]
        scenarios = saved["scenarios"]
        folds = saved["folds"]
        if len(sample_ids) != 4977 or len(np.unique(sample_ids)) != 4977:
            raise RuntimeError("Locked cohort identity changed")
        if not np.array_equal(np.unique(folds), np.arange(5)):
            raise RuntimeError("Locked outer folds changed")
    router_lock = json.loads(ROUTER_LOCK.read_text(encoding="utf-8"))
    scenario_names = np.unique(scenarios)
    # Deterministic, label-blind scenario assignment. The assignment is made
    # separately inside each outer training population and never uses labels.
    outer_plans: list[dict[str, object]] = []
    for outer_fold in range(5):
        train_rows = np.flatnonzero(folds != outer_fold)
        train_scenarios = np.unique(scenarios[train_rows])
        assignment = {str(name): int(index % 5) for index, name in enumerate(sorted(train_scenarios.tolist()))}
        inner = []
        for inner_fold in range(5):
            held = train_rows[np.array([assignment[str(scenarios[row])] == inner_fold for row in train_rows])]
            inner_train = np.setdiff1d(train_rows, held, assume_unique=True)
            if len(held) == 0 or len(inner_train) == 0:
                raise RuntimeError("Deterministic inner fold is empty")
            if np.intersect1d(held, np.flatnonzero(folds == outer_fold)).size:
                raise RuntimeError("Inner held rows escape outer training")
            inner.append(
                {
                    "inner_fold": inner_fold,
                    "held_scenarios": sorted({str(scenarios[row]) for row in held}),
                    "train_scenarios": sorted({str(scenarios[row]) for row in inner_train}),
                    "held_rows": int(len(held)),
                    "train_rows": int(len(inner_train)),
                    "held_rows_sha256": canonical_hash(sample_ids[held].tolist()),
                    "train_rows_sha256": canonical_hash(sample_ids[inner_train].tolist()),
                }
            )
        outer_plans.append(
            {
                "outer_fold": outer_fold,
                "outer_train_rows": int(len(train_rows)),
                "outer_train_rows_sha256": canonical_hash(sample_ids[train_rows].tolist()),
                "scenario_assignment": assignment,
                "inner_folds": inner,
            }
        )
    receipt: dict[str, object] = {
        "status": "TRANSITION_RISK_INNER_CACHE_PLAN_LOCKED_NO_FITS",
        "router_execution_lock_sha256": file_sha256(ROUTER_LOCK),
        "metadata_sha256": file_sha256(METADATA),
        "rows": 4977,
        "outer_folds": 5,
        "inner_folds_per_outer": 5,
        "scenario_count": int(len(scenario_names)),
        "outer_label_reads": 0,
        "fit_performed": False,
        "threshold_selected": False,
        "candidate_oof_required": True,
        "outer_plans": outer_plans,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
