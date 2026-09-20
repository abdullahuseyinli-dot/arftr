"""Independent, no-fit replay audit for the completed PCAR-v5 screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from hac.pcar import PCARPolicy  # noqa: E402
from experiments import run_okutama_pcar as runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / ".runs/research_20260917/pcar_v5_screen_v2",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    runner._verify_execution_lock()
    cache, locks, packed = runner._load_packed()
    bundle = runner._matched_affine_candidates(cache, packed)
    split_pairs = runner.outer_scenario_splits(locks["cache"]["scenarios"])
    fold_ids = np.empty(runner.CENTERS, dtype=np.int64)
    for fold, (_, held) in enumerate(split_pairs):
        fold_ids[held] = fold
    saved = np.load(args.run / "per_center.npz", allow_pickle=False)
    max_deltas = {"p2": 0.0, "c_affine": 0.0, "pcar": 0.0, "actions": 0}
    per_fold: dict[str, dict[str, float | int]] = {}
    for fold in range(runner.FOLDS):
        held = np.flatnonzero(fold_ids == fold)
        p0, p2 = runner._load_screen_models(fold, args.device)
        p0_dense = runner._predict_dense(p0, packed["P0_center_only"], held, args.device)
        p2_dense = runner._predict_dense(p2, packed["P2_same_grid_pool"], held, args.device)
        policy = PCARPolicy().to(args.device)
        checkpoint = torch.load(args.run / f"policy_fold{fold}.pt", map_location=args.device, weights_only=False)
        if checkpoint.get("outer_fold") != fold or checkpoint.get("updates") != runner.POLICY_UPDATES:
            raise RuntimeError(f"policy checkpoint contract changed at fold {fold}")
        policy.load_state_dict(checkpoint["state_dict"], strict=True)
        policy.eval()
        evaluation = runner._evaluate_policy(
            policy, p0_dense, p2_dense, runner._slice_bundle(bundle, held), device=args.device
        )
        expected_p2 = saved["p2"][held]
        expected_c = saved["c_affine"][held]
        expected_pcar = saved["pcar"][held]
        deltas = {
            "p2": float(np.max(np.abs(evaluation["p2_per_center"] - expected_p2))),
            "c_affine": float(np.max(np.abs(evaluation["candidate_per_center"] - expected_c))),
            "pcar": float(np.max(np.abs(evaluation["per_center"] - expected_pcar))),
            "actions": int(np.max(np.abs(evaluation["actions"] - saved["pcar_actions"][held]))),
        }
        for key, value in deltas.items():
            max_deltas[key] = max(max_deltas[key], value)
        per_fold[str(fold)] = {"held_centers": int(len(held)), **deltas}
        del p0, p2, policy, p0_dense, p2_dense
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    passed = (
        max_deltas["p2"] <= 2e-6
        and max_deltas["c_affine"] <= 2e-6
        and max_deltas["pcar"] <= 2e-6
        and max_deltas["actions"] == 0
    )
    audit = {
        "status": "PCAR_V5_INDEPENDENT_REPLAY_AUDIT_PASS" if passed else "PCAR_V5_INDEPENDENT_REPLAY_AUDIT_FAIL",
        "run": str(args.run),
        "device": args.device,
        "fit_or_optimizer_steps": 0,
        "task_labels_read": 0,
        "ARFTR_probability_arrays_read": 0,
        "outer_held_labels_read": 0,
        "per_fold": per_fold,
        "max_deltas": max_deltas,
        "action_tie_rule_rechecked": True,
        "retained_arftr_changed": False,
    }
    path = args.run / "independent_audit.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": audit["status"], "max_deltas": max_deltas}, sort_keys=True))


if __name__ == "__main__":
    main()

