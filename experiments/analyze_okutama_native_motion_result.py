"""Post-release forensics for the fixed native-motion screen."""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import run_okutama_native_motion_innovation as experiment
from hac.native_motion_innovation import ARMS


def transitions(anchor: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict:
    old, new = anchor.argmax(1), candidate.argmax(1)
    changed = old != new
    rescue = changed & (old != labels) & (new == labels)
    harm = changed & (old == labels) & (new != labels)
    wrong = changed & (old != labels) & (new != labels)
    return {
        "changes": int(changed.sum()),
        "rescues": int(rescue.sum()),
        "harms": int(harm.sum()),
        "wrong_to_other_wrong": int(wrong.sum()),
        "net": int(rescue.sum() - harm.sum()),
    }


def main() -> None:
    run = experiment.DEFAULT_RUN
    summary = experiment.read_json(run / "summary.json")
    d = experiment.data()
    candidates, routed, deltas, eligible = {}, {}, {}, {}
    policy_support = {}
    for arm in ARMS:
        candidates[arm] = np.full_like(d["anchor"], np.nan)
        routed[arm] = np.full_like(d["anchor"], np.nan)
        deltas[arm] = np.full(len(d["labels"]), np.nan)
        eligible[arm] = np.zeros(len(d["labels"]), bool)
        policy_support[arm] = []
        for fold in range(5):
            saved = experiment.load_predictions(run / arm / f"fold-{fold}" / "predictions.npz")
            rows = saved["held_rows"]
            candidates[arm][rows] = saved["candidate_probabilities"]
            routed[arm][rows] = saved["routed_probabilities"]
            deltas[arm][rows] = saved["delta"]
            eligible[arm][rows] = saved["eligible"]
            receipt = experiment.read_json(run / arm / f"fold-{fold}" / "receipt.json")
            policy_support[arm].append(receipt["policy_receipt"])
    result = {
        "status": "NATIVE_MOTION_POST_RELEASE_FORENSICS_COMPLETE",
        "screen_status": summary["status"],
        "promoted": summary["promoted"],
        "raw_transitions": {},
        "routed_transitions": {},
        "pairwise_raw_disagreement": {},
        "delta_distribution": {},
        "policy_training_support": policy_support,
        "camera_diagnostics_posthoc_only": {},
    }
    for arm in ARMS:
        result["raw_transitions"][arm] = {
            **transitions(d["anchor"], candidates[arm], d["labels"]),
            "per_fold": [
                transitions(d["anchor"][d["folds"] == fold], candidates[arm][d["folds"] == fold], d["labels"][d["folds"] == fold])
                for fold in range(5)
            ],
        }
        result["routed_transitions"][arm] = transitions(d["anchor"], routed[arm], d["labels"])
        values = np.abs(deltas[arm][eligible[arm]])
        result["delta_distribution"][arm] = {
            "eligible_rows": int(eligible[arm].sum()),
            "median_abs": float(np.median(values)),
            "p95_abs": float(np.quantile(values, .95)),
            "at_bound_fraction": float((values >= .499).mean()),
        }
    for left, right in combinations(ARMS, 2):
        lp, rp = candidates[left].argmax(1), candidates[right].argmax(1)
        different = lp != rp
        result["pairwise_raw_disagreement"][f"{left}__{right}"] = {
            "different_predictions": int(different.sum()),
            "left_correct_right_wrong": int(((lp == d["labels"]) & (rp != d["labels"])).sum()),
            "left_wrong_right_correct": int(((lp != d["labels"]) & (rp == d["labels"])).sum()),
        }
    camera_magnitude = np.linalg.norm(d["motion_camera"][:, :, :2], axis=2).mean(1)
    quantiles = np.quantile(camera_magnitude, np.linspace(0, 1, 6))
    for arm in ARMS:
        bins = []
        for index in range(5):
            chosen = (camera_magnitude >= quantiles[index]) & (
                camera_magnitude <= quantiles[index + 1] if index == 4 else camera_magnitude < quantiles[index + 1]
            )
            raw = transitions(d["anchor"][chosen], candidates[arm][chosen], d["labels"][chosen])
            bins.append({"low": float(quantiles[index]), "high": float(quantiles[index + 1]), "rows": int(chosen.sum()), **raw})
        result["camera_diagnostics_posthoc_only"][arm] = bins
    experiment.write_json(run / "postmortem.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
