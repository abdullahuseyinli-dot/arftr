"""Low-budget two-arm execution wrapper for the fine-density probe.

The v1 four-arm matrix remains preserved as an over-budget preflight.  This
wrapper reuses its audited data and fit primitives but freezes a two-arm,
two-epoch comparison so the decisive representation test stays within the
runtime boundary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import run_okutama_fine_detail_probe as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/okutama_fine_detail_probe_v2_protocol.json"
LOCK_PATH = ROOT / "experiments/okutama_fine_detail_execution_v2_lock.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_smoke(data: dict, output: Path, protocol: dict, device: str) -> dict:
    path = output / "timing_smoke.json"
    if path.exists():
        return read_json(path)
    fold = 0
    rows = np.flatnonzero(data["folds"] != fold)[:128]
    mean, std, qmean, qstd, weights = base.training_stats(data, rows)
    seed = int(protocol["training"]["seed"])
    torch.manual_seed(seed)
    model = base.make_model("fine_both", device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    weight_tensor = torch.from_numpy(weights).to(device)
    batch_size = int(protocol["training"]["batch_size"])
    started = time.perf_counter()
    model.train()
    updates = 0
    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        fine, times, quality, coarse = base.batch_inputs(data, subset, mean, std, qmean, qstd, "fine_both", device)
        labels = torch.from_numpy(data["labels"][subset]).to(device)
        optimizer.zero_grad(set_to_none=True)
        result = model(fine, times, quality=quality, coarse_reference=coarse)
        loss = torch.nn.functional.cross_entropy(result["logits"], labels, weight=weight_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError("Fine-detail timing smoke produced a nonfinite loss")
        loss.backward()
        optimizer.step()
        updates += 1
    elapsed = time.perf_counter() - started
    projected = elapsed * (5 * 2 * int(protocol["training"]["epochs"]) / max(updates, 1)) * (4977 / len(rows))
    result = {
        "status": "FINE_DETAIL_V2_TIMING_SMOKE_PASS",
        "scientific": False,
        "rows": int(len(rows)),
        "updates": updates,
        "elapsed_seconds": elapsed,
        "projected_full_seconds": projected,
        "projected_full_minutes": projected / 60.0,
        "device": device,
        "parameter_count": model.trainable_parameters,
        "training_authorized": projected <= 1200.0,
        "outer_held_labels_read": 0,
    }
    write_json(path, result)
    if projected > 1200.0:
        raise RuntimeError(f"Projected v2 probe exceeds 20 minutes: {projected / 60.0:.2f} minutes")
    return result


def summarize(output: Path, data: dict) -> dict:
    arms = ("coarse_control", "fine_both")
    values: dict[str, np.ndarray] = {}
    fold_receipts: dict[str, dict] = {}
    for arm in arms:
        probabilities = np.zeros((len(data["labels"]), 3), dtype=np.float32)
        seen = np.zeros(len(data["labels"]), dtype=bool)
        for fold in range(5):
            folder = output / f"fold-{fold}" / arm
            fold_receipts[f"{arm}/fold-{fold}"] = read_json(folder / "receipt.json")
            saved = np.load(folder / "predictions.npz", allow_pickle=False)
            rows = saved["held_rows"].astype(np.int64)
            ids = saved["sample_ids"].astype(str)
            if not np.array_equal(ids, data["sample_ids"][rows]) or np.any(seen[rows]):
                raise RuntimeError(f"Prediction identity mismatch for {arm}/fold-{fold}")
            probabilities[rows] = saved["probabilities"].astype(np.float32)
            seen[rows] = True
        if not np.all(seen) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"Incomplete probabilities for {arm}")
        values[arm] = probabilities
    labels = data["labels"]
    metrics = {arm: base.metrics(labels, p) for arm, p in values.items()}
    with np.load(base.ARFTR_PATH, allow_pickle=False) as arftr:
        if not np.array_equal(arftr["sample_ids"].astype(str), data["sample_ids"]):
            raise RuntimeError("ARFTR and fine-detail IDs differ")
        anchor = arftr["mean_probabilities"][5].astype(np.float64)
    anchor_metrics = base.metrics(labels, anchor)
    blend = base.metrics(labels, 0.5 * anchor + 0.5 * values["fine_both"])
    anchor_pred = anchor.argmax(1)
    fine_pred = values["fine_both"].argmax(1)
    rescues = (anchor_pred != labels) & (fine_pred == labels)
    harms = (anchor_pred == labels) & (fine_pred != labels)
    fold_metrics = {}
    for fold in range(5):
        mask = data["folds"] == fold
        fold_metrics[str(fold)] = {
            arm: base.metrics(labels[mask], values[arm][mask]) for arm in arms
        }
        fold_metrics[str(fold)]["fine_both_vs_arftr"] = {
            "rescues": int(np.sum(rescues & mask)),
            "harms": int(np.sum(harms & mask)),
            "net_corrections": int(np.sum(rescues & mask) - np.sum(harms & mask)),
        }
    result = {
        "status": "FINE_DETAIL_V2_SCIENTIFIC_PROBE_COMPLETE",
        "scientific": True,
        "task_training_authorized": False,
        "retained_arftr_changed": False,
        "rows": int(len(labels)),
        "outer_folds": 5,
        "arms": list(arms),
        "metrics": metrics,
        "arftr_anchor": anchor_metrics,
        "fixed_half_blend_fine_both_arftr": blend,
        "fine_both_vs_arftr": {
            "rescues": int(rescues.sum()),
            "harms": int(harms.sum()),
            "net_corrections": int(rescues.sum() - harms.sum()),
            "candidate_unique_error_rescues": int(rescues.sum()),
        },
        "folds": fold_metrics,
        "fold_receipts": fold_receipts,
        "arftr_read_phase": "post_fit_summary_only",
        "outer_held_labels_read_during_fit": 0,
        "human_review_fields_read": 0,
        "oracle_results": False,
        "promotion_decision": "not authorized; probe only decides whether to design nested ARFTR-preserving integration",
    }
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "timing-smoke", "all", "summarize"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".runs/research_20260917/fine_detail_probe_v2")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    # Rebind audited primitives to the v2 lock/protocol without changing v1.
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.LOCK_PATH = LOCK_PATH
    protocol = read_json(PROTOCOL_PATH)
    os_config = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for key, value in os_config.items():
        import os

        os.environ.setdefault(key, value)
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
    data = base.validate_inputs(protocol)
    if args.stage == "validate":
        print(json.dumps({"status": "FINE_DETAIL_V2_VALIDATE_PASS", "rows": len(data["labels"]), "device": args.device}, indent=2))
        return
    if args.stage == "timing-smoke":
        print(json.dumps(run_smoke(data, args.output_dir, protocol, args.device), indent=2))
        return
    if args.stage == "all":
        smoke = run_smoke(data, args.output_dir, protocol, args.device)
        if not smoke["training_authorized"]:
            raise RuntimeError("Timing smoke did not authorize v2 full probe")
        for fold in range(5):
            for index, arm in enumerate(("coarse_control", "fine_both")):
                base.train_fold(data, args.output_dir, fold, arm, index, protocol, args.device)
    if args.stage in ("all", "summarize"):
        result = summarize(args.output_dir, data)
        print(json.dumps({"status": result["status"], "output": str(args.output_dir / "summary.json"), "metrics": result["metrics"], "blend": result["fixed_half_blend_fine_both_arftr"], "fine_both_vs_arftr": result["fine_both_vs_arftr"]}, indent=2))


if __name__ == "__main__":
    main()
