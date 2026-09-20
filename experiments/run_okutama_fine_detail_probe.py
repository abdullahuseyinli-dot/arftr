"""Run the locked one-seed fine-density representation probe.

The probe compares genuine 12x12 V-JEPA tokens with a repeated 3x3 control
under the same ``FineLocalMotion`` head.  It is deliberately a standalone
outer-fold experiment: ARFTR probabilities are not read until fitting has
finished, and no result from this run may overwrite the retained ARFTR model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from hac.fine_local_motion import FineLocalMotion, expand_coarse_tokens


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/okutama_fine_detail_probe_protocol.json"
LOCK_PATH = ROOT / "experiments/okutama_fine_detail_execution_lock.json"
DENSE_DIR = ROOT / ".runs/research_20260908/dense_tokens_full"
MEMORY_PATH = ROOT / ".runs/research_20260908/evidence_memory/data/memory_data.npz"
ARFTR_PATH = ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(1)
    one_hot = np.eye(3, dtype=np.float64)[labels]
    return {
        "rows": int(len(labels)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "accuracy": float(np.mean(predictions == labels)),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0)).mean()),
        "brier": float(np.mean((probabilities - one_hot) ** 2)),
        "per_class_f1": f1_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "confusion": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "errors": int(np.sum(predictions != labels)),
    }


def validate_inputs(protocol: dict) -> dict:
    if not LOCK_PATH.is_file():
        raise RuntimeError(f"Execution lock is missing: {LOCK_PATH}")
    lock = read_json(LOCK_PATH)
    expected = {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "module_sha256": sha256_file(ROOT / "src/hac/fine_local_motion.py"),
        "tests_sha256": sha256_file(ROOT / "tests/test_fine_local_motion.py"),
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"Execution lock mismatch for {key}: {lock.get(key)} != {value}")
    verification = read_json(DENSE_DIR / "verification.json")
    if verification.get("status") != "FULL_DENSE_CACHE_INDEPENDENT_REPLAY_VERIFIED":
        raise RuntimeError("Dense token verification is not PASS")
    dense_summary = read_json(DENSE_DIR / "summary.json")
    grid_hashes = dense_summary.get("artifacts", {})
    if grid_hashes.get("tokens_grid12.npy", {}).get("sha256") != lock.get("tokens_grid12_sha256"):
        raise RuntimeError("12x12 token hash differs from execution lock")
    if grid_hashes.get("tokens_grid3.npy", {}).get("sha256") != lock.get("tokens_grid3_sha256"):
        raise RuntimeError("3x3 token hash differs from execution lock")
    fine = np.load(DENSE_DIR / "tokens_grid12.npy", mmap_mode="r", allow_pickle=False)
    coarse = np.load(DENSE_DIR / "tokens_grid3.npy", mmap_mode="r", allow_pickle=False)
    source_frames = np.load(DENSE_DIR / "source_frames.npy", allow_pickle=False)
    dense_ids = np.load(DENSE_DIR / "sample_ids.npy", allow_pickle=False).astype(str)
    with np.load(MEMORY_PATH, allow_pickle=False) as data:
        ids = data["sample_ids"].astype(str)
        if not np.array_equal(ids, dense_ids):
            raise RuntimeError("Dense token and metadata sample IDs differ")
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        frames = data["frames"].astype(np.float32)
        quality = data["quality"].astype(np.float32)
        recordings = data["recordings"].astype(str)
        tracks = data["tracks"].astype(str)
    if fine.shape != (4977, 8, 144, 768) or coarse.shape != (4977, 8, 9, 768):
        raise RuntimeError(f"Unexpected token shapes: fine={fine.shape}, coarse={coarse.shape}")
    if source_frames.shape != (4977, 16) or labels.shape != (4977,) or folds.shape != (4977,):
        raise RuntimeError("Unexpected metadata shapes")
    if not np.isfinite(source_frames).all() or np.any(source_frames[:, 1:] <= source_frames[:, :-1]):
        raise RuntimeError("Source frames are not finite and strictly increasing")
    # Eight tubelet midpoints are the only temporal clock supplied to the head.
    times = (source_frames[:, 0::2] + source_frames[:, 1::2]) / 2.0
    times = (times - frames[:, None]) / 30.0
    if np.any(times[:, 1:] <= times[:, :-1]) or not np.isfinite(times).all():
        raise RuntimeError("Tubelet midpoint times are invalid")
    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Unexpected outer-fold mapping")
    if recordings.shape != tracks.shape:
        raise RuntimeError("Track metadata shape mismatch")
    return {
        "fine": fine,
        "coarse": coarse,
        "times": times.astype(np.float32),
        "quality": quality,
        "labels": labels,
        "folds": folds,
        "sample_ids": ids,
        "recordings": recordings,
        "tracks": tracks,
        "lock": lock,
    }


def training_stats(data: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coarse = np.asarray(data["coarse"][rows], dtype=np.float32)
    mean = coarse.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = coarse.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    q = data["quality"][rows].astype(np.float64)
    qmean = q.mean(axis=0).astype(np.float32)
    qstd = np.maximum(q.std(axis=0), 1e-5).astype(np.float32)
    counts = np.bincount(data["labels"][rows], minlength=3).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights *= 3.0 / weights.sum()
    return mean, std, qmean, qstd, weights.astype(np.float32)


def batch_inputs(data: dict, rows: np.ndarray, mean: np.ndarray, std: np.ndarray, qmean: np.ndarray, qstd: np.ndarray, arm: str, device: str) -> tuple[torch.Tensor, ...]:
    coarse = np.asarray(data["coarse"][rows], dtype=np.float32)
    coarse = (coarse - mean[None, None, None, :]) / std[None, None, None, :]
    if arm == "coarse_control":
        fine = expand_coarse_tokens(torch.from_numpy(coarse)).numpy()
    else:
        fine = np.asarray(data["fine"][rows], dtype=np.float32)
        fine = (fine - mean[None, None, None, :]) / std[None, None, None, :]
    quality = (data["quality"][rows] - qmean[None, :]) / qstd[None, :]
    times = data["times"][rows]
    return (
        torch.from_numpy(np.ascontiguousarray(fine)).to(device),
        torch.from_numpy(np.ascontiguousarray(times)).to(device),
        torch.from_numpy(np.ascontiguousarray(quality)).to(device),
        torch.from_numpy(np.ascontiguousarray(coarse)).to(device),
    )


def make_model(arm: str, device: str) -> FineLocalMotion:
    mode = "both"
    if arm.endswith("spatial_only"):
        mode = "spatial_only"
    elif arm.endswith("motion_only"):
        mode = "motion_only"
    model = FineLocalMotion(
        rank=32,
        width=128,
        layers=2,
        heads=4,
        dropout=0.1,
        match_mode="fixed",
        detail_mode=mode,
        parameter_limit=1_000_000,
    ).to(device)
    if model.trainable_parameters != 584148:
        raise RuntimeError(f"Unexpected FineLocalMotion parameter count: {model.trainable_parameters}")
    return model


def train_fold(data: dict, output: Path, fold: int, arm: str, arm_index: int, protocol: dict, device: str) -> dict:
    fold_dir = output / f"fold-{fold}" / arm
    fold_dir.mkdir(parents=True, exist_ok=False)
    train_rows = np.flatnonzero(data["folds"] != fold)
    held_rows = np.flatnonzero(data["folds"] == fold)
    mean, std, qmean, qstd, weights = training_stats(data, train_rows)
    np.savez(fold_dir / "normalization.npz", mean=mean, std=std, qmean=qmean, qstd=qstd, class_weights=weights)
    seed = int(protocol["training"]["seed"]) + fold * 100 + arm_index
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(arm, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    weight_tensor = torch.from_numpy(weights).to(device)
    batch_size = int(protocol["training"]["batch_size"])
    epochs = int(protocol["training"]["epochs"])
    started = time.perf_counter()
    losses: list[float] = []
    updates = 0
    model.train()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(train_rows), generator=generator).numpy()
        epoch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            rows = train_rows[local]
            fine, times, quality, coarse = batch_inputs(data, rows, mean, std, qmean, qstd, arm, device)
            labels = torch.from_numpy(data["labels"][rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(fine, times, quality=quality, coarse_reference=coarse, center_index=4)
            loss = torch.nn.functional.cross_entropy(result["logits"], labels, weight=weight_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at fold={fold}, arm={arm}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            epoch_losses.append(value)
            updates += 1
        if not epoch_losses:
            raise RuntimeError("No optimizer updates")
    fit_seconds = time.perf_counter() - started
    # Candidate predictions are generated only after all updates for this fold.
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(held_rows), batch_size):
            rows = held_rows[start : start + batch_size]
            fine, times, quality, coarse = batch_inputs(data, rows, mean, std, qmean, qstd, arm, device)
            predictions.append(model(fine, times, quality=quality, coarse_reference=coarse, center_index=4)["probabilities"].cpu().numpy())
    probabilities = np.concatenate(predictions, axis=0).astype(np.float32)
    np.savez(
        fold_dir / "predictions.npz",
        sample_ids=data["sample_ids"][held_rows],
        held_rows=held_rows,
        probabilities=probabilities,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "fold": fold,
            "arm": arm,
            "seed": seed,
            "mean": mean,
            "std": std,
            "qmean": qmean,
            "qstd": qstd,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "module_sha256": sha256_file(ROOT / "src/hac/fine_local_motion.py"),
        },
        fold_dir / "checkpoint.pt",
    )
    result = {
        "status": "FINE_DETAIL_FOLD_COMPLETE",
        "fold": fold,
        "arm": arm,
        "seed": seed,
        "train_rows": int(len(train_rows)),
        "held_rows": int(len(held_rows)),
        "outer_held_labels_read_during_fit": 0,
        "optimizer_updates": updates,
        "epochs": epochs,
        "parameter_count": model.trainable_parameters,
        "fit_seconds": fit_seconds,
        "mean_loss_last_epoch": float(np.mean(losses[-len(epoch_losses) :])),
        "mean_loss_all_updates": float(np.mean(losses)),
        "input_arm": "repeated_coarse" if arm == "coarse_control" else "genuine_fine",
        "checkpoint": str((fold_dir / "checkpoint.pt").relative_to(ROOT)).replace("\\", "/"),
        "predictions": str((fold_dir / "predictions.npz").relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(fold_dir / "receipt.json", result)
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_smoke(data: dict, output: Path, protocol: dict, device: str) -> dict:
    path = output / "timing_smoke.json"
    if path.exists():
        return read_json(path)
    fold = 0
    arm = "fine_both"
    rows = np.flatnonzero(data["folds"] != fold)[:128]
    mean, std, qmean, qstd, weights = training_stats(data, rows)
    seed = int(protocol["training"]["seed"])
    torch.manual_seed(seed)
    model = make_model(arm, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["training"]["learning_rate"]), weight_decay=float(protocol["training"]["weight_decay"]))
    labels = torch.from_numpy(data["labels"][rows]).to(device)
    batch = int(protocol["training"]["batch_size"])
    weight_tensor = torch.from_numpy(weights).to(device)
    started = time.perf_counter()
    updates = 0
    model.train()
    for start in range(0, len(rows), batch):
        subset = rows[start : start + batch]
        fine, times, quality, coarse = batch_inputs(data, subset, mean, std, qmean, qstd, arm, device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(fine, times, quality=quality, coarse_reference=coarse)["logits"], labels[start : start + len(subset)], weight=weight_tensor)
        loss.backward()
        optimizer.step()
        updates += 1
    elapsed = time.perf_counter() - started
    projected = elapsed * (5 * 4 * int(protocol["training"]["epochs"]) / max(updates, 1)) * (4977 / len(rows))
    result = {
        "status": "FINE_DETAIL_TIMING_SMOKE_PASS",
        "scientific": False,
        "rows": int(len(rows)),
        "updates": updates,
        "elapsed_seconds": elapsed,
        "projected_full_seconds": projected,
        "projected_full_minutes": projected / 60.0,
        "device": device,
        "parameter_count": model.trainable_parameters,
        "labels_used": int(len(rows)),
        "outer_held_labels_read": 0,
        "training_authorized": projected <= 1200.0,
    }
    write_json(path, result)
    if projected > 1200.0:
        raise RuntimeError(f"Projected full probe exceeds 20 minutes: {projected / 60.0:.2f} minutes")
    return result


def summarize(output: Path, data: dict) -> dict:
    arm_names = ["coarse_control", "fine_spatial_only", "fine_motion_only", "fine_both"]
    probabilities: dict[str, np.ndarray] = {}
    fold_receipts: dict[str, dict] = {}
    for arm in arm_names:
        values = np.zeros((len(data["labels"]), 3), dtype=np.float32)
        for fold in range(5):
            folder = output / f"fold-{fold}" / arm
            receipt = read_json(folder / "receipt.json")
            saved = np.load(folder / "predictions.npz", allow_pickle=False)
            rows = saved["held_rows"].astype(np.int64)
            ids = saved["sample_ids"].astype(str)
            if not np.array_equal(ids, data["sample_ids"][rows]):
                raise RuntimeError(f"Prediction identity mismatch for {arm}/fold-{fold}")
            if np.any(values[rows] != 0):
                raise RuntimeError("Duplicate held prediction rows")
            values[rows] = saved["probabilities"].astype(np.float32)
            fold_receipts[f"{arm}/fold-{fold}"] = receipt
        if not np.isfinite(values).all() or np.any(values.sum(1) <= 0):
            raise RuntimeError(f"Incomplete probabilities for {arm}")
        probabilities[arm] = values
    labels = data["labels"]
    metrics_all = {arm: metrics(labels, p) for arm, p in probabilities.items()}
    # ARFTR is deliberately opened only after all candidate fitting is done.
    with np.load(ARFTR_PATH, allow_pickle=False) as arftr:
        if not np.array_equal(arftr["sample_ids"].astype(str), data["sample_ids"]):
            raise RuntimeError("ARFTR and fine-probe IDs differ")
        arftr_probabilities = arftr["mean_probabilities"][5].astype(np.float64)
    anchor_metrics = metrics(labels, arftr_probabilities)
    blend_metrics = metrics(labels, 0.5 * probabilities["fine_both"] + 0.5 * arftr_probabilities)
    candidate_pred = probabilities["fine_both"].argmax(1)
    anchor_pred = arftr_probabilities.argmax(1)
    rescues = (anchor_pred != labels) & (candidate_pred == labels)
    harms = (anchor_pred == labels) & (candidate_pred != labels)
    fold_summary = {}
    for fold in range(5):
        mask = data["folds"] == fold
        fold_summary[str(fold)] = {
            arm: metrics(labels[mask], probabilities[arm][mask]) for arm in arm_names
        }
        fold_summary[str(fold)]["fine_both_vs_arftr"] = {
            "rescues": int(np.sum(rescues & mask)),
            "harms": int(np.sum(harms & mask)),
            "net_corrections": int(np.sum(rescues & mask) - np.sum(harms & mask)),
        }
    result = {
        "status": "FINE_DETAIL_SCIENTIFIC_PROBE_COMPLETE",
        "scientific": True,
        "task_training_authorized": False,
        "retained_arftr_changed": False,
        "rows": int(len(labels)),
        "outer_folds": 5,
        "arms": arm_names,
        "metrics": metrics_all,
        "arftr_anchor": anchor_metrics,
        "fixed_half_blend_fine_both_arftr": blend_metrics,
        "fine_both_vs_arftr": {
            "rescues": int(rescues.sum()),
            "harms": int(harms.sum()),
            "net_corrections": int(rescues.sum() - harms.sum()),
            "candidate_unique_error_rescues": int(rescues.sum()),
        },
        "folds": fold_summary,
        "fold_receipts": fold_receipts,
        "arftr_read_phase": "post_fit_summary_only",
        "outer_held_labels_read_during_fit": 0,
        "human_review_fields_read": 0,
        "oracle_results": False,
        "promotion_decision": "not_authorized; primary advance gates are evaluated in the report",
    }
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "timing-smoke", "all", "summarize"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".runs/research_20260917/fine_detail_probe_v1")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL_PATH)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage == "all" and args.output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.stage == "all":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "source_snapshot").mkdir()
        shutil.copy2(Path(__file__), args.output_dir / "source_snapshot" / Path(__file__).name)
        shutil.copy2(PROTOCOL_PATH, args.output_dir / "source_snapshot" / PROTOCOL_PATH.name)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    data = validate_inputs(protocol)
    if args.stage == "validate":
        print(json.dumps({"status": "FINE_DETAIL_VALIDATE_PASS", "rows": len(data["labels"]), "device": args.device}, indent=2))
        return
    if args.stage == "timing-smoke":
        print(json.dumps(run_smoke(data, args.output_dir, protocol, args.device), indent=2))
        return
    if args.stage == "all":
        smoke = run_smoke(data, args.output_dir, protocol, args.device)
        if not smoke["training_authorized"]:
            raise RuntimeError("Timing smoke does not authorize the full probe")
        for fold in range(5):
            for index, arm in enumerate(("coarse_control", "fine_spatial_only", "fine_motion_only", "fine_both")):
                train_fold(data, args.output_dir, fold, arm, index, protocol, args.device)
    if args.stage in ("all", "summarize"):
        result = summarize(args.output_dir, data)
        print(json.dumps({"status": result["status"], "output": str(args.output_dir / "summary.json"), "metrics": result["metrics"], "blend": result["fixed_half_blend_fine_both_arftr"], "fine_both_vs_arftr": result["fine_both_vs_arftr"]}, indent=2))


if __name__ == "__main__":
    main()
