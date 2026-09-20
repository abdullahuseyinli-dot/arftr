"""Independently replay the fitted transition-risk router without outer labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_okutama_transition_risk_router as runner

from hac.actor_memory_base import file_sha256
from hac.transition_risk_router import TransitionRiskRouter, router_features


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _router(saved: dict[str, np.ndarray]) -> TransitionRiskRouter:
    names = [str(value) for value in saved["feature_names"].tolist()]
    threshold = float(saved["threshold"][0])
    mean = saved["scaler_mean"]
    scale = saved["scaler_scale"]
    coefficients = saved["coefficients"]
    intercept = saved["intercept"]
    if mean.size == 0:
        if coefficients.size or intercept.size or scale.size:
            raise RuntimeError("constant router contains non-empty model state")
        return TransitionRiskRouter(None, None, names, threshold, 2.0, {})
    if coefficients.shape != (1, len(names)) or intercept.shape != (1,):
        raise RuntimeError("router coefficient schema changed")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = mean.astype(np.float64, copy=True)
    scaler.scale_ = scale.astype(np.float64, copy=True)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(names)
    model = LogisticRegression()
    model.classes_ = np.asarray([0, 1], dtype=np.int64)
    model.coef_ = coefficients.astype(np.float64, copy=True)
    model.intercept_ = intercept.astype(np.float64, copy=True)
    model.n_features_in_ = len(names)
    model.n_iter_ = np.asarray([1], dtype=np.int32)
    return TransitionRiskRouter(scaler, model, names, threshold, 2.0, {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    parser.add_argument("--result-dir", type=Path, default=runner.DEFAULT_RESULT)
    args = parser.parse_args()
    run = args.run.resolve()
    result_dir = args.result_dir.resolve()
    run.relative_to(runner.ROOT.resolve())
    result_dir.relative_to(runner.ROOT.resolve())
    lock = runner.validate_lock(run)
    fit_receipt = _read(result_dir / "fit_receipt.json")
    if fit_receipt.get("status") != "TRANSITION_RISK_ROUTER_OOF_COMPLETE_OUTER_METRICS_EMBARGOED":
        raise RuntimeError("router fit receipt is not complete")
    d = runner.data()
    cache = runner._load_cache(d)
    exact = []
    for outer in range(5):
        held = np.flatnonzero(d["folds"] == outer)
        fold_dir = result_dir / f"fold-{outer}"
        fold_receipt = _read(fold_dir / "receipt.json")
        if fold_receipt.get("outer_held_labels_read") != 0:
            raise RuntimeError("outer-held labels were recorded as read")
        prediction_path = fold_dir / "predictions.npz"
        router_path = fold_dir / "router.npz"
        if fold_receipt.get("predictions_sha256") != file_sha256(prediction_path):
            raise RuntimeError("router prediction receipt hash changed")
        if fold_receipt.get("router_sha256") != file_sha256(router_path):
            raise RuntimeError("router state receipt hash changed")
        with np.load(prediction_path, allow_pickle=False) as saved:
            prediction = {name: saved[name].copy() for name in saved.files}
        with np.load(router_path, allow_pickle=False) as saved:
            state = {name: saved[name].copy() for name in saved.files}
        if not np.array_equal(prediction["held_rows"], held):
            raise RuntimeError("router held-row identity changed")
        if not np.array_equal(prediction["sample_ids"], d["sample_ids"][held]):
            raise RuntimeError("router sample identity changed")
        descriptor, available = cache.descriptor(runner.POSTURE_KEYS)
        quality = np.linalg.norm(
            descriptor[held].reshape(len(held), 3, 768), axis=2
        ) / np.sqrt(768.0)
        features, names = router_features(
            prediction["anchor_probabilities"],
            prediction["candidate_probabilities"],
            availability=available[held],
            quality=quality,
        )
        if names != [str(value) for value in state["feature_names"].tolist()]:
            raise RuntimeError("router feature names changed")
        model = _router(state)
        reproduced, action = model.apply(
            prediction["anchor_probabilities"],
            prediction["candidate_probabilities"],
            features,
        )
        scores = model.score(features)
        checks = {
            "routed_probabilities": np.array_equal(reproduced, prediction["routed_probabilities"]),
            "action": np.array_equal(action, prediction["action"]),
            "router_scores": np.array_equal(scores, prediction["router_scores"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"independent router replay failed on fold {outer}: {checks}")
        exact.append({"outer_fold": outer, "checks": checks, "maximum_score_difference": 0.0})
    result = {
        "status": "TRANSITION_RISK_ROUTER_INDEPENDENT_REPLAY_PASS",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "fit_receipt_sha256": file_sha256(result_dir / "fit_receipt.json"),
        "outer_held_labels_read": 0,
        "all_folds_exact": True,
        "folds": exact,
        "auditor_sha256": file_sha256(Path(__file__)),
        "router_lock_sha256": lock["router_execution_lock_sha256"],
    }
    output = result_dir / "independent_audit.json"
    if output.exists():
        raise RuntimeError("independent audit output already exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "folds": len(exact)}, indent=2))


if __name__ == "__main__":
    main()
