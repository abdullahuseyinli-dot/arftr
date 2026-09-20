"""Fixed-epoch memory refits that cannot access prediction-population labels.

The historical memory trainer always calculated a terminal held-set metric,
including for fixed-epoch outer refits.  That statistic did not influence the
model, but it violated the stronger ancestry contract required for F(S).  This
module keeps the historical optimizer/model semantics while replacing all
supervision outside ``train`` with sentinels before any torch batch object is
constructed and emits no held metric.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hac.actor_memory_base import canonical_hash, file_sha256
from hac.actor_memory_training import MemoryBatches, new_model, predict, seed_training
from hac.source_swap_data import immutable_json, read_json
from hac.source_swap_execution import ARM, FIELDS, guard_attempt


def _embargo_supervision(
    data: dict[str, np.ndarray], train: np.ndarray
) -> dict[str, np.ndarray]:
    """Return a shallow inference-data view with supervision only on train."""
    rows = np.asarray(train, dtype=np.int64)
    result = dict(data)
    labels = np.full_like(data["labels"], -1)
    labels[rows] = data["labels"][rows]
    result["labels"] = labels
    boundary_targets = np.zeros_like(data["boundary_targets"])
    boundary_targets[rows] = data["boundary_targets"][rows]
    result["boundary_targets"] = boundary_targets
    boundary_valid = np.zeros_like(data["boundary_valid"], dtype=bool)
    boundary_valid[rows] = data["boundary_valid"][rows]
    result["boundary_valid"] = boundary_valid
    edge_source_slot = np.full_like(data["edge_source_slot"], -1)
    edge_source_slot[rows] = data["edge_source_slot"][rows]
    result["edge_source_slot"] = edge_source_slot
    return result


def _request(
    data: dict[str, np.ndarray],
    protocol: dict[str, Any],
    train: np.ndarray,
    held: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    epochs: int,
    ancestry: list[dict[str, Any]],
    execution_hash: str,
) -> dict[str, Any]:
    return {
        "status": "LABEL_EMBARGOED_FIXED_EPOCH_MEMORY_REQUEST",
        "data_sha256": data.get("_file_sha256", "synthetic_test_data"),
        "arm": ARM,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "epochs": epochs,
        "protocol_sha256": canonical_hash(protocol),
        "training_code_sha256": file_sha256(Path(__file__)),
        "architecture_code_sha256": file_sha256(
            Path(__file__).with_name("actor_evidence_memory.py")
        ),
        "historical_batch_code_sha256": file_sha256(
            Path(__file__).with_name("actor_memory_training.py")
        ),
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "held_labels_sha256": None,
        "selection_labels_sha256": None,
        "base_ancestry_sha256": canonical_hash(ancestry),
        "execution_lock_sha256": execution_hash,
    }


def _load_completed(
    directory: Path,
    request: dict[str, Any],
    held: np.ndarray,
    held_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    receipt = read_json(directory / "receipt.json")
    if (
        receipt.get("status") != "LABEL_EMBARGOED_MEMORY_REFIT_COMPLETE"
        or receipt.get("request") != request
        or receipt.get("outer_prediction_labels_read") != 0
        or receipt.get("outer_prediction_boundary_targets_read") != 0
        or "held_metrics" in receipt
        or receipt.get("held_rows") != held.tolist()
    ):
        raise RuntimeError("Label-embargoed memory receipt changed")
    for name in ("predictions.npz", "checkpoint.pt"):
        if file_sha256(directory / name) != receipt["output_sha256"][name]:
            raise RuntimeError("Label-embargoed memory output changed")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        if (
            not np.array_equal(saved["held_rows"], held)
            or not np.array_equal(saved["sample_ids"], held_ids)
            or set(saved.files) != {"held_rows", "sample_ids", *FIELDS}
        ):
            raise RuntimeError("Label-embargoed prediction identity changed")
        outputs = {name: saved[name] for name in FIELDS}
    return outputs, receipt


def fit_outer_label_embargoed(
    directory: Path,
    data: dict[str, np.ndarray],
    protocol: dict[str, Any],
    train: np.ndarray,
    held: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    epochs: int,
    probabilities: np.ndarray,
    ancestry: list[dict[str, Any]],
    execution_hash: str,
    progress: Callable[..., None] | None,
    *,
    device: str = "cuda",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit a fixed-epoch outer model after removing all held supervision."""
    train = np.asarray(train, dtype=np.int64)
    held = np.asarray(held, dtype=np.int64)
    if (
        not isinstance(epochs, int)
        or epochs < 1
        or np.intersect1d(train, held).size
        or set(data["scenarios"][train]) & set(data["scenarios"][held])
    ):
        raise ValueError("Label-embargoed refit requires fixed epochs and disjoint scenarios")
    request = _request(
        data,
        protocol,
        train,
        held,
        learning_rate,
        weight_decay,
        seed,
        epochs,
        ancestry,
        execution_hash,
    )
    directory = Path(directory)
    receipt_path = directory / "receipt.json"
    if receipt_path.is_file():
        return _load_completed(directory, request, held, data["sample_ids"][held])
    guard = {
        "request": request,
        "train_boundary_targets_sha256": canonical_hash(
            data["boundary_targets"][train].tolist()
        ),
        "train_boundary_valid_sha256": canonical_hash(
            data["boundary_valid"][train].tolist()
        ),
    }
    guard_attempt(
        directory,
        guard,
        complete_files={"receipt.json", "checkpoint.pt", "predictions.npz"},
    )
    blind = _embargo_supervision(data, train)
    if (
        np.any(blind["labels"][held] != -1)
        or np.any(blind["boundary_valid"][held])
        or np.any(blind["edge_source_slot"][held] != -1)
    ):
        raise RuntimeError("Prediction supervision embargo failed")
    seed_training(seed)
    started = time.perf_counter()
    batches = MemoryBatches(blind, probabilities, train, device=device)
    model = new_model(data["features"].shape[1], ARM, protocol, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = np.random.default_rng(seed)
    history = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(train)
        training_loss = 0.0
        for start in range(0, len(order), protocol["batch_size"]):
            rows = order[start : start + protocol["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            outputs, index = batches.forward(model, rows)
            loss = batches.loss(outputs, index, protocol)
            if not torch.isfinite(loss):
                raise RuntimeError("Label-embargoed memory loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                protocol["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            training_loss += float(loss.detach()) * len(rows)
        row = {"epoch": epoch, "training_loss": training_loss / len(train)}
        history.append(row)
        if progress is not None and (epoch == 1 or epoch % 5 == 0):
            progress(
                epoch=epoch,
                maximum=epochs,
                training_loss=row["training_loss"],
                elapsed_seconds=time.perf_counter() - started,
            )
    outputs = predict(model, batches, held, protocol["batch_size"])
    np.savez_compressed(
        directory / "predictions.npz",
        held_rows=held,
        sample_ids=data["sample_ids"][held],
        **outputs,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "scaler_mean": batches.scaler.mean_,
            "scaler_scale": batches.scaler.scale_,
            "request": request,
            "epoch": epochs,
        },
        directory / "checkpoint.pt",
    )
    receipt = {
        "status": "LABEL_EMBARGOED_MEMORY_REFIT_COMPLETE",
        "request": request,
        "train_rows": train.tolist(),
        "held_rows": held.tolist(),
        "base_ancestry": ancestry,
        "selected_epoch": epochs,
        "epochs_run": epochs,
        "history": history,
        "class_weight": batches.class_weight.cpu().tolist(),
        "boundary_positive_weight": float(batches.positive_weight),
        "parameters": model.trainable_parameters,
        "outer_prediction_labels_read": 0,
        "outer_prediction_boundary_targets_read": 0,
        "seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else 0,
        "output_sha256": {
            name: file_sha256(directory / name)
            for name in ("predictions.npz", "checkpoint.pt")
        },
    }
    immutable_json(receipt_path, receipt)
    del optimizer, model, batches
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return _load_completed(directory, request, held, data["sample_ids"][held])
