"""Independent replay audit for the saved FDCRG outer predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hac.fine_density_residual_gate import FineDensityResidualGate, build_actions, build_features  # noqa: E402
from experiments.run_okutama_fdcrg import metrics, read_json  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_gate(path: Path) -> FineDensityResidualGate:
    with np.load(path, allow_pickle=False) as saved:
        mean = saved["scaler_mean"].astype(np.float64)
        scale = saved["scaler_scale"].astype(np.float64)
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = scale
        scaler.var_ = scale**2
        scaler.n_features_in_ = len(mean)
        regressors = [None]
        for action in range(1, 4):
            model = Ridge(alpha=2.0, fit_intercept=True)
            model.coef_ = saved[f"coef_{action}"].astype(np.float64)
            model.intercept_ = float(saved[f"intercept_{action}"][0])
            model.n_features_in_ = len(mean)
            regressors.append(model)
        threshold = float(saved["threshold"][0])
        residual_cap = float(saved["residual_cap"][0])
        intervention_budget = float(saved["intervention_budget"][0])
    return FineDensityResidualGate(scaler, tuple(regressors), threshold, residual_cap, intervention_budget)


def audit(run: Path) -> dict:
    summary = read_json(run / "summary.json")
    all_ids = []
    all_rows = []
    anchors = []
    candidates = []
    choices = []
    max_probability_delta = 0.0
    max_feature_delta = 0.0
    fold_results = []
    for fold in range(5):
        folder = run / f"fold-{fold}"
        with np.load(folder / "outer_inputs.npz", allow_pickle=False) as saved:
            ids = saved["sample_ids"].astype(str)
            rows = saved["held_rows"].astype(np.int64)
            anchor = saved["anchor"].astype(np.float64)
            fine = saved["fine"].astype(np.float64)
            coarse = saved["coarse"].astype(np.float64)
            quality = saved["quality"].astype(np.float64)
            times = saved["times"].astype(np.float64)
            expected_choices = saved["choices"].astype(np.int64)
            expected_probabilities = saved["probabilities"].astype(np.float64)
        features = build_features(anchor, fine, coarse, quality=quality, times=times)
        actions = build_actions(anchor, fine, coarse)
        if (folder / "gate.npz").is_file():
            gate = load_gate(folder / "gate.npz")
            replay_probabilities, replay_choices = gate.apply(features, actions)
        else:
            replay_probabilities, replay_choices = anchor.copy(), np.zeros(len(rows), dtype=np.int64)
        max_probability_delta = max(max_probability_delta, float(np.max(np.abs(replay_probabilities - expected_probabilities))))
        max_feature_delta = max(max_feature_delta, float(np.max(np.abs(actions[:, 0] - anchor))))
        if not np.array_equal(replay_choices, expected_choices):
            raise RuntimeError(f"FDCRG choice replay mismatch in fold {fold}")
        if not np.allclose(replay_probabilities, expected_probabilities, atol=1e-12, rtol=0):
            raise RuntimeError(f"FDCRG probability replay mismatch in fold {fold}")
        receipt = read_json(folder / "receipt.json")
        if receipt.get("outer_held_labels_read_during_fit") != 0:
            raise RuntimeError(f"Outer-label embargo receipt failed in fold {fold}")
        all_ids.append(ids)
        all_rows.append(rows)
        anchors.append(anchor)
        candidates.append(replay_probabilities)
        choices.append(replay_choices)
        fold_results.append({"fold": fold, "rows": int(len(rows)), "gate_file": (folder / "gate.npz").is_file(), "choice_replay": True, "probability_replay": True})
    rows = np.concatenate(all_rows)
    ids = np.concatenate(all_ids)
    anchor = np.zeros((len(rows), 3), dtype=np.float64)
    candidate = np.zeros_like(anchor)
    selected = np.zeros(len(rows), dtype=np.int64)
    anchor[rows] = np.concatenate(anchors)
    candidate[rows] = np.concatenate(candidates)
    selected[rows] = np.concatenate(choices)
    with np.load(ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz", allow_pickle=False) as saved:
        canonical_ids = saved["sample_ids"].astype(str)
        labels = saved["labels"].astype(np.int64)
    if not np.array_equal(ids[np.argsort(rows)], canonical_ids) or not np.array_equal(np.sort(rows), np.arange(len(labels))):
        raise RuntimeError("FDCRG replay coverage or sample identity mismatch")
    result = {
        "status": "FDCRG_INDEPENDENT_REPLAY_AUDIT_PASS",
        "run": str(run.relative_to(ROOT)).replace("\\", "/"),
        "folds": fold_results,
        "rows": int(len(labels)),
        "max_probability_delta": max_probability_delta,
        "max_action_anchor_delta": max_feature_delta,
        "metrics_recomputed": {"anchor": metrics(labels, anchor), "fdcrg": metrics(labels, candidate)},
        "interventions": int(np.sum(selected != 0)),
        "optimizer_updates": 0,
        "classifier_fits": 0,
        "outer_held_labels_read_during_fit": 0,
        "labels_read_for_posthoc_metric_check": int(len(labels)),
        "human_review_fields_read": 0,
        "annotation_support_used": False,
        "oracle_results": False,
    }
    write_json(run / "independent_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / ".runs/research_20260917/fdcrg_v1_full")
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), indent=2))


if __name__ == "__main__":
    main()

