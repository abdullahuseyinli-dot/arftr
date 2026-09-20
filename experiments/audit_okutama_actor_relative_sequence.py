"""Independently replay the actor-relative timestamped-frame evidence screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_actor_relative_sequence as trial
import torch

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.actor_relative_sequence import ARMS
from hac.cached_frame_supervision import validate_cached_frame_supervision_receipt
from hac.matr_artifacts import load_cached_study_data
from hac.source_swap_data import immutable_json


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    with np.load(path, allow_pickle=False) as saved:
        observed = {name.replace("__", "."): saved[name] for name in saved.files}
    expected = model.state_dict()
    require(set(observed) == set(expected), "Sequence checkpoint inventory changed")
    state = {}
    for name, reference in expected.items():
        value = observed[name]
        require(value.shape == tuple(reference.shape) and np.isfinite(value).all(), f"Sequence checkpoint changed: {name}")
        state[name] = torch.as_tensor(value)
    model.load_state_dict(state, strict=True)


def replay_fit(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    store: trial.SequenceStore,
    lock_hash: str,
    arm: str,
    fold: int,
    seed: int,
) -> np.ndarray:
    held = np.flatnonzero(data["folds"] == fold)
    prediction_path, checkpoint_path, receipt_path = trial.fit_paths(run, arm, fold, seed)
    receipt = trial.read_json(receipt_path)
    require(
        receipt.get("status") == "ACTOR_RELATIVE_SEQUENCE_FIT_COMPLETE_OUTER_METRICS_EMBARGOED"
        and receipt.get("execution_lock_sha256") == lock_hash
        and receipt.get("arm") == arm
        and receipt.get("fold") == fold
        and receipt.get("seed") == seed
        and receipt.get("outer_held_labels_read") == 0
        and receipt.get("held_sample_ids_sha256") == canonical_hash(data["sample_ids"][held].tolist())
        and receipt.get("predictions_sha256") == file_sha256(prediction_path)
        and receipt.get("checkpoint_sha256") == file_sha256(checkpoint_path),
        "Sequence fit receipt identity changed",
    )
    model = trial.make_model(protocol, arm)
    load_checkpoint(model, checkpoint_path)
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for left in range(0, len(held), protocol["training"]["batch_size"]):
            payload = trial.to_device(store.batch(held[left : left + protocol["training"]["batch_size"]]), torch.device("cpu"))
            output = model(payload["features"], payload["timestamps"], payload["streams"], payload["valid"], payload["center_indices"])
            probabilities.append(torch.softmax(output["center_logits"], 1).numpy())
    replay = np.concatenate(probabilities).astype(np.float64)
    with np.load(prediction_path, allow_pickle=False) as saved:
        require(np.array_equal(saved["sample_ids"], data["sample_ids"][held]) and np.array_equal(saved["held_rows"], held), "Sequence held IDs changed")
        published = saved["probabilities"].copy()
    require(np.allclose(replay, published, rtol=1e-4, atol=1e-5), "CPU sequence checkpoint replay drift exceeded tolerance")
    require(int(np.sum(replay.argmax(1) != published.argmax(1))) == 0, "CPU sequence replay changed a class decision")
    return published


def audit(run: Path, output: Path) -> dict[str, Any]:
    protocol = trial.read_json(trial.PROTOCOL)
    trial.validate_protocol(protocol)
    data = load_cached_study_data(trial.SOURCE)
    frames, _ = validate_cached_frame_supervision_receipt(trial.ROOT, trial.FRAME_DATA)
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        require(np.array_equal(data[name], frames[name]), f"Sequence frame identity changed: {name}")
    lock_hash = file_sha256(run / "execution_lock.json")
    trial.validate_lock(run, data, lock_hash)
    store = trial.SequenceStore(frames)
    rows = len(data["labels"])
    replay = np.full((len(ARMS), 3, rows, 3), np.nan)
    fits = 0
    for fold in range(5):
        held = np.flatnonzero(data["folds"] == fold)
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(trial.SEEDS):
                replay[arm_index, seed_index, held] = replay_fit(run, protocol, data, store, lock_hash, arm, fold, seed)
                fits += 1
    require(fits == 75 and np.isfinite(replay).all(), "Sequence replay inventory is incomplete")
    result_dir = run / "results/v0001"
    summary = trial.read_json(result_dir / "summary.json")
    with np.load(result_dir / "oof_probabilities.npz", allow_pickle=False) as saved:
        require(np.array_equal(saved["sample_ids"], data["sample_ids"]) and tuple(saved["arms"].tolist()) == ARMS and tuple(saved["seeds"].tolist()) == trial.SEEDS, "Sequence OOF identity changed")
        require(np.array_equal(saved["seed_probabilities"], replay), "Published sequence OOF changed")
        average = saved["mean_probabilities"].copy()
        reference = saved["arftr_mean_probabilities"].copy()
    require(np.array_equal(average, replay.mean(1)), "Published sequence mean changed")
    metrics = {arm: probability_metrics(data["labels"], average[index]) for index, arm in enumerate(ARMS)}
    require(metrics == summary["metrics"], "Published sequence metrics changed")
    arftr = probability_metrics(data["labels"], reference)
    require(arftr == summary["arftr_metrics"], "Published ARFTR reference metrics changed")
    statistics = {
        arm: trial.fsar._scenario_statistics(data["labels"], average[index], reference, data["scenarios"], resamples=10000, seed=20260912 + index)
        for index, arm in enumerate(ARMS)
    }
    require(statistics == summary["statistics_vs_arftr"], "Published sequence statistics changed")
    audit_summary = {
        "status": "ACTOR_RELATIVE_SEQUENCE_INDEPENDENT_AUDIT_PASS",
        "complete": True,
        "execution_run": str(run),
        "execution_lock_sha256": lock_hash,
        "execution_summary_sha256": file_sha256(result_dir / "summary.json"),
        "fits_verified": fits,
        "checkpoint_forward_passes_replayed": fits,
        "metrics": metrics,
        "arftr_metrics": arftr,
        "screen_gate_passed": bool(summary["screen_gate_passed"]),
        "limitations": "Adaptive repeated-development evidence; audit validates computation, not external generalization.",
    }
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "summary.json", audit_summary)
    print(json.dumps({"status": audit_summary["status"], "output": str(output / "summary.json")}, indent=2), flush=True)
    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=trial.DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=trial.ROOT / ".runs/research_20260912/actor_relative_sequence_v1_audit")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    run.relative_to(trial.ROOT.resolve())
    output.relative_to(trial.ROOT.resolve())
    audit(run, output)


if __name__ == "__main__":
    main()
