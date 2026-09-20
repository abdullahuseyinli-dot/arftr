"""Run the locked one-seed PTG discovery probe.

Stages are deliberately explicit: ``prepare`` builds a hash lock and the
label-blind feature cache, ``timing-smoke`` measures one outer fold, ``run``
fits the four arms on five scenario-held folds, ``audit`` replays predictions,
and ``summarize`` releases outer metrics.  No integration/router is implemented
here; ARFTR remains a read-only context/default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.persistent_token_gauge import PTGDiscoveryHead, coarse_temporal_features, persistent_token_features


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_persistent_token_gauge_protocol.json"
METADATA = ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz"
DENSE = ROOT / ".runs/research_20260908/dense_tokens_full"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
SOURCE_FRAMES = DENSE / "source_frames.npy"
MODULE = ROOT / "src/hac/persistent_token_gauge.py"
TESTS = ROOT / "tests/test_persistent_token_gauge.py"
DEFAULT_RUN = ROOT / ".runs/research_20260917/ptg_task_probe_v1"
ARMS = ("G0_arftr_context", "G1_coarse_temporal_control", "G2_raw_persistent_paths", "G3_gauge_parity")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"Immutable artifact differs: {path}")
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _data() -> dict[str, np.ndarray]:
    with np.load(METADATA, allow_pickle=False) as z:
        data = {key: z[key] for key in ("sample_ids", "labels", "scenarios", "folds")}
    if data["labels"].shape != (4977,) or len(np.unique(data["sample_ids"])) != 4977:
        raise RuntimeError("Canonical population contract changed")
    if not np.array_equal(np.sort(np.unique(data["folds"])), np.arange(5)):
        raise RuntimeError("Expected five grouped folds")
    return data


def _lock(run: Path) -> dict[str, Any]:
    verification = json.loads((DENSE / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    data = _data()
    lock = {
        "status": "PTG_TASK_DISCOVERY_LOCKED_OUTER_METRICS_EMBARGOED",
        "protocol_sha256": sha256(PROTOCOL),
        "module_sha256": sha256(MODULE),
        "tests_sha256": sha256(TESTS),
        "metadata_sha256": sha256(METADATA),
        "tokens_grid12_sha256": verification["artifacts"]["tokens_grid12.npy"]["sha256"],
        "tokens_grid3_sha256": verification["artifacts"]["tokens_grid3.npy"]["sha256"],
        "rows": 4977,
        "outer_folds": 5,
        "arms": list(ARMS),
        "seed": 20260917,
        "epochs": 2,
        "batch_size": 256,
        "parameter_limit": 1000000,
        "outer_held_labels_read_at_lock": 0,
        "task_training_authorized": False,
        "retained_arftr_changed": False,
        "sample_ids_sha256": hashlib.sha256("\n".join(data["sample_ids"].tolist()).encode()).hexdigest(),
        "protocol_status": protocol["status"],
    }
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "execution_lock.json", lock)
    return lock


@torch.inference_mode()
def _precompute(run: Path, *, device: str) -> dict[str, np.ndarray]:
    path = run / "feature_cache.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    fine = np.load(DENSE / "tokens_grid12.npy", mmap_mode="r", allow_pickle=False)
    coarse = np.load(DENSE / "tokens_grid3.npy", mmap_mode="r", allow_pickle=False)
    source_frames = np.load(SOURCE_FRAMES, mmap_mode="r", allow_pickle=False)
    times = ((source_frames[:, 0::2] + source_frames[:, 1::2]) / 2.0).astype(np.float32)
    with np.load(ARFTR, allow_pickle=False) as z:
        anchor = z["mean_probabilities"][5].astype(np.float32)
    if fine.shape != (4977, 8, 144, 768) or coarse.shape != (4977, 8, 9, 768):
        raise RuntimeError("Dense feature shape changed")
    raw_parts, gauge_parts, coarse_parts = [], [], []
    batch_size = 16
    for start in range(0, len(fine), batch_size):
        stop = min(len(fine), start + batch_size)
        ft = torch.from_numpy(np.asarray(fine[start:stop], dtype=np.float16)).to(device)
        ct = torch.from_numpy(np.asarray(coarse[start:stop], dtype=np.float16)).to(device)
        tt = torch.from_numpy(times[start:stop]).to(device)
        ptg = persistent_token_features(ft, tt)
        coarse_value = coarse_temporal_features(ct, tt)
        raw_parts.append(ptg["raw"].cpu().numpy().astype(np.float32))
        gauge_parts.append(ptg["gauge"].cpu().numpy().astype(np.float32))
        coarse_parts.append(coarse_value.cpu().numpy().astype(np.float32))
    raw = np.concatenate(raw_parts)
    gauge = np.concatenate(gauge_parts)
    coarse_features = np.concatenate(coarse_parts)
    logp = np.log(np.clip(anchor, 1e-7, 1)).astype(np.float32)
    g0 = np.concatenate((anchor, logp, -(anchor * logp).sum(1, keepdims=True)), axis=1)
    arrays = {"g0": g0, "g1": coarse_features, "g2": raw, "g3": gauge, "anchor": anchor, "times": times}
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    return arrays


def _arm_key(arm: str) -> str:
    return {ARMS[0]: "g0", ARMS[1]: "g1", ARMS[2]: "g2", ARMS[3]: "g3"}[arm]


def _fit_one(
    run: Path,
    arm: str,
    fold: int,
    data: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    *,
    device: str,
    save: bool,
) -> dict[str, Any]:
    key = _arm_key(arm)
    held = np.flatnonzero(data["folds"] == fold)
    train = np.flatnonzero(data["folds"] != fold)
    x = np.asarray(features[key], dtype=np.float32)
    if not np.isfinite(x).all():
        raise RuntimeError(f"Nonfinite {arm} features")
    mean = x[train].mean(0)
    scale = np.maximum(x[train].std(0), 1e-5)
    x_train = (x[train] - mean) / scale
    x_held = (x[held] - mean) / scale
    y_train = data["labels"][train]
    class_count = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = len(y_train) / (3.0 * np.maximum(class_count, 1.0))
    torch.manual_seed(20260917 + fold)
    if device == "cuda":
        torch.cuda.manual_seed_all(20260917 + fold)
    model = PTGDiscoveryHead(x.shape[1]).to(device)
    params = sum(v.numel() for v in model.parameters() if v.requires_grad)
    if params > 1_000_000:
        raise RuntimeError(f"{arm} exceeds parameter limit: {params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    order = np.arange(len(train))
    started = time.perf_counter()
    model.train()
    updates = 0
    for epoch in range(2):
        rng = np.random.default_rng(20260917 + fold * 10 + epoch)
        rng.shuffle(order)
        for begin in range(0, len(order), 256):
            idx = order[begin : begin + 256]
            xb = torch.from_numpy(x_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(x_held).to(device)), dim=1).cpu().numpy().astype(np.float32)
    receipt = {
        "status": "PTG_DISCOVERY_FIT_COMPLETE",
        "arm": arm,
        "fold": fold,
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "epochs": 2,
        "optimizer_updates": updates,
        "parameter_count": params,
        "fit_seconds": elapsed,
        "outer_held_labels_read": 0,
        "training_labels_read": int(len(train)),
        "human_review_fields_read": 0,
        "annotation_support_read": 0,
        "device": device,
        "feature_key": key,
    }
    if save:
        directory = run / f"fold-{fold}" / arm
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "predictions.npz").open("wb") as f:
            np.savez_compressed(f, sample_ids=data["sample_ids"][held], held_rows=held, probabilities=probabilities)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "scale": scale}, directory / "checkpoint.pt")
        write_json(directory / "receipt.json", receipt)
    return {**receipt, "held": held, "probabilities": probabilities}


def _metric(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probabilities, dtype=np.float64)
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    onehot = np.eye(3)[y]
    brier = np.mean(np.square(p - onehot))
    return {
        "rows": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
        "accuracy": float((pred == y).mean()),
        "nll": float(nll),
        "brier": float(brier),
        "errors": int((pred != y).sum()),
        "per_class_f1": f1_score(y, pred, average=None, labels=[0, 1, 2]).tolist(),
        "confusion": cm.tolist(),
    }


def _bootstrap(y: np.ndarray, cand: np.ndarray, base: np.ndarray, scenarios: np.ndarray, seed: int = 20260917) -> dict[str, Any]:
    groups = np.unique(scenarios)
    c, b = [], []
    for group in groups:
        rows = scenarios == group
        c.append(np.bincount(3 * y[rows] + cand[rows].argmax(1), minlength=9).reshape(3, 3))
        b.append(np.bincount(3 * y[rows] + base[rows].argmax(1), minlength=9).reshape(3, 3))
    c, b = np.asarray(c), np.asarray(b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(10000)
    for i in range(len(deltas)):
        picks = rng.integers(0, len(groups), len(groups))
        cm = c[picks].sum(0)
        bm = b[picks].sum(0)
        cf = 2 * np.diag(cm) / np.maximum(cm.sum(1) + cm.sum(0), 1)
        bf = 2 * np.diag(bm) / np.maximum(bm.sum(1) + bm.sum(0), 1)
        deltas[i] = (cf.mean() - bf.mean())
    return {"resamples": 10000, "seed": seed, "lower_95": float(np.quantile(deltas, 0.025)), "mean": float(deltas.mean()), "upper_95": float(np.quantile(deltas, 0.975))}


def prepare(run: Path, device: str) -> dict[str, Any]:
    lock = _lock(run)
    data = _data()
    started = time.perf_counter()
    features = _precompute(run, device=device)
    receipt = {"status": "PTG_DISCOVERY_PREPARE_COMPLETE", "feature_shapes": {k: list(v.shape) for k, v in features.items()}, "device": device, "precompute_seconds": time.perf_counter() - started, "outer_held_labels_read": 0, "classifier_fits": 0, "lock": lock}
    write_json(run / "prepare_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


def timing_smoke(run: Path, device: str) -> dict[str, Any]:
    data = _data()
    features = _precompute(run, device=device)
    times = []
    for arm in ARMS:
        receipt = _fit_one(run, arm, 0, data, features, device=device, save=False)
        times.append(receipt["fit_seconds"])
    projected = float(np.mean(times) * 5)
    result = {"status": "PTG_DISCOVERY_TIMING_SMOKE_COMPLETE", "arm_seconds": dict(zip(ARMS, times)), "projected_full_seconds": projected, "projected_full_minutes": projected / 60.0, "training_authorized": projected <= 20 * 60, "device": device}
    write_json(run / "timing_smoke.json", result)
    print(json.dumps(result, indent=2))
    return result


def run_all(run: Path, device: str) -> None:
    lock = json.loads((run / "execution_lock.json").read_text(encoding="utf-8"))
    if not lock.get("status", "").startswith("PTG_TASK_DISCOVERY_LOCKED"):
        raise RuntimeError("Prepare the PTG run before fitting")
    timing = json.loads((run / "timing_smoke.json").read_text(encoding="utf-8"))
    if not timing.get("training_authorized"):
        raise RuntimeError("Timing smoke did not authorize the 20-minute discovery matrix")
    data = _data()
    features = _precompute(run, device=device)
    receipts = []
    for fold in range(5):
        for arm in ARMS:
            receipts.append(_fit_one(run, arm, fold, data, features, device=device, save=True))
    lock["task_training_authorized"] = True
    lock["completed_fits"] = len(receipts)
    (run / "execution_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PTG_DISCOVERY_MATRIX_COMPLETE", "fits": len(receipts), "seconds": float(sum(r["fit_seconds"] for r in receipts))}, indent=2))


def summarize(run: Path) -> dict[str, Any]:
    data = _data()
    probabilities = {arm: np.zeros((len(data["labels"]), 3), dtype=np.float32) for arm in ARMS}
    fold_metrics = {arm: {} for arm in ARMS}
    for fold in range(5):
        held = np.flatnonzero(data["folds"] == fold)
        for arm in ARMS:
            with np.load(run / f"fold-{fold}" / arm / "predictions.npz", allow_pickle=False) as z:
                if not np.array_equal(z["held_rows"], held):
                    raise RuntimeError("Held-row order changed")
                probabilities[arm][held] = z["probabilities"]
            fold_metrics[arm][str(fold)] = _metric(data["labels"][held], probabilities[arm][held])
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise RuntimeError("Incomplete PTG OOF coverage")
    with np.load(ARFTR, allow_pickle=False) as z:
        anchor = z["mean_probabilities"][5].astype(np.float32)
    results = {arm: _metric(data["labels"], probabilities[arm]) for arm in ARMS}
    pairwise = {}
    for arm in ARMS[1:]:
        pairwise[f"{arm}_vs_G0_delta"] = results[arm]["macro_f1"] - results[ARMS[0]]["macro_f1"]
    g3 = probabilities[ARMS[3]]
    g2 = probabilities[ARMS[2]]
    g0 = probabilities[ARMS[0]]
    g3_vs_g0_folds = [fold_metrics[ARMS[3]][str(f)]["macro_f1"] - fold_metrics[ARMS[0]][str(f)]["macro_f1"] for f in range(5)]
    g3_vs_g2_folds = [fold_metrics[ARMS[3]][str(f)]["macro_f1"] - fold_metrics[ARMS[2]][str(f)]["macro_f1"] for f in range(5)]
    bootstrap = _bootstrap(data["labels"], g3, g0, data["scenarios"])
    gates = {
        "g3_vs_g0_delta_min_0_005": pairwise["G3_gauge_parity_vs_G0_delta"] >= 0.005,
        "g3_vs_g2_positive_every_fold": min(g3_vs_g2_folds) > 0,
        "g3_vs_g0_bootstrap_lower_positive": bootstrap["lower_95"] > 0,
        "g3_nll_not_worse_than_g0": results[ARMS[3]]["nll"] <= results[ARMS[0]]["nll"],
        "g3_brier_not_worse_than_g0": results[ARMS[3]]["brier"] <= results[ARMS[0]]["brier"],
    }
    summary = {
        "status": "PTG_DISCOVERY_MATRIX_COMPLETE",
        "scientific": True,
        "outer_folds": 5,
        "arms": list(ARMS),
        "results": results,
        "fold_metrics": fold_metrics,
        "pairwise": pairwise,
        "g3_vs_g0_folds": g3_vs_g0_folds,
        "g3_vs_g2_folds": g3_vs_g2_folds,
        "g3_vs_g0_bootstrap": bootstrap,
        "gates": gates,
        "discovery_gate_pass": bool(all(gates.values())),
        "anchor_arftr": {"macro_f1": _metric(data["labels"], anchor)["macro_f1"], "errors": int((anchor.argmax(1) != data["labels"]).sum())},
        "outer_held_labels_read_during_fit": 0,
        "human_review_fields_read": 0,
        "annotation_support_read": 0,
        "oracle_results": False,
        "retained_arftr_changed": False,
        "task_training_authorized": False,
    }
    write_json(run / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "results": {k: round(v["macro_f1"], 8) for k, v in results.items()}, "gates": gates, "discovery_gate_pass": summary["discovery_gate_pass"]}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("prepare", "timing-smoke", "run", "summarize"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    if args.stage == "prepare":
        prepare(run, args.device)
    elif args.stage == "timing-smoke":
        timing_smoke(run, args.device)
    elif args.stage == "run":
        run_all(run, args.device)
    else:
        summarize(run)


if __name__ == "__main__":
    main()
