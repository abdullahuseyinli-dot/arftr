"""Integrity wrappers around unchanged original source-swap fitting semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.actor_memory_training import MemoryBatches, fit_memory, new_model, predict, seed_training
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[2]
ARM = "survival_memory"
FIELDS = ("probabilities", "gate", "attention", "survival", "boundary_probabilities")


def expected_request(data, protocol, train, held, lr, wd, seed, epochs, ancestry):
    if set(data["scenarios"][train]) & set(data["scenarios"][held]):
        raise RuntimeError("Source-swap scenario leakage")
    return {
        "data_sha256": data.get("_file_sha256", "synthetic_test_data"),
        "arm": ARM,
        "learning_rate": lr,
        "weight_decay": wd,
        "seed": seed,
        "epochs": epochs,
        "protocol_sha256": canonical_hash(protocol),
        "training_code_sha256": file_sha256(ROOT / "src/hac/actor_memory_training.py"),
        "architecture_code_sha256": file_sha256(ROOT / "src/hac/actor_evidence_memory.py"),
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": canonical_hash(data["labels"][held].tolist())
        if epochs is None
        else None,
        "base_ancestry_sha256": canonical_hash(ancestry),
    }


def guard_attempt(directory, request, *, complete_files, marker="request_guard.json"):
    """Do not overwrite a partial/foreign attempt; an untouched request may resume."""
    directory = Path(directory)
    if directory.exists():
        inventory = {path.name for path in directory.iterdir()}
        if inventory and inventory not in ({marker}, set(complete_files) | {marker}):
            raise RuntimeError(
                f"Partial or foreign attempt retained without overwrite: {directory}"
            )
        if inventory and marker not in inventory:
            raise RuntimeError("Existing output has no source-swap prefit request")
    immutable_json(directory / marker, request)


def validate_fit(
    directory,
    data,
    protocol,
    train,
    held,
    lr,
    wd,
    seed,
    epochs,
    ancestry,
    *,
    replay=False,
    probabilities=None,
    guard=None,
):
    expected = expected_request(data, protocol, train, held, lr, wd, seed, epochs, ancestry)
    expected_files = {"receipt.json", "checkpoint.pt", "predictions.npz"}
    if guard is not None:
        expected_files.add("request_guard.json")
        if read_json(directory / "request_guard.json") != guard:
            raise RuntimeError("Fit pre-registration changed")
    if {path.name for path in directory.iterdir()} != expected_files:
        raise RuntimeError("Completed source-swap fit inventory differs")
    receipt = read_json(directory / "receipt.json")
    if (
        receipt["request"] != expected
        or receipt["base_ancestry"] != ancestry
        or receipt["train_rows"] != train.tolist()
        or receipt["held_rows"] != held.tolist()
    ):
        raise RuntimeError("Fit request/ancestry/rows changed")
    hashes = {str(directory / "receipt.json"): file_sha256(directory / "receipt.json")}
    if set(receipt["output_sha256"]) != {"checkpoint.pt", "predictions.npz"}:
        raise RuntimeError("Fit checksum inventory changed")
    for name, expected_hash in receipt["output_sha256"].items():
        path = directory / name
        if file_sha256(path) != expected_hash:
            raise RuntimeError("Fit output changed")
        hashes[str(path)] = expected_hash
    if guard is not None:
        hashes[str(directory / "request_guard.json")] = file_sha256(
            directory / "request_guard.json"
        )
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["held_rows"], held) or not np.array_equal(
            saved["sample_ids"], data["sample_ids"][held]
        ):
            raise RuntimeError("Saved prediction identity changed")
        outputs = {key: saved[key] for key in FIELDS}
    measured = probability_metrics(data["labels"][held], outputs["probabilities"])
    if measured != receipt["held_metrics"]:
        raise RuntimeError("Saved fit metrics changed")
    history = receipt["history"]
    if len(history) != receipt["epochs_run"] or [r["epoch"] for r in history] != list(
        range(1, len(history) + 1)
    ):
        raise RuntimeError("Fit epoch coverage changed")
    if epochs is None:
        best = min(
            history,
            key=lambda r: (
                -r["validation_metrics"]["macro_f1"],
                r["validation_metrics"]["nll"],
                r["epoch"],
            ),
        )
        if best["epoch"] != receipt["selected_epoch"] or best["validation_metrics"] != measured:
            raise RuntimeError("Inner checkpoint selection differs from retained evidence")
    elif (
        len(history) != epochs
        or receipt["selected_epoch"] != epochs
        or any("validation_metrics" in r for r in history)
    ):
        raise RuntimeError("Outer fit read held labels for checkpoint selection")
    counts = np.bincount(data["labels"][train], minlength=3)
    weight = (len(train) / (3 * counts)).astype(np.float32)
    boundary = data["boundary_targets"][train][data["boundary_valid"][train]]
    positive = float(boundary.sum())
    pos_weight = float(np.float32((len(boundary) - positive) / positive if positive else 1.0))
    if (
        not np.array_equal(weight, np.asarray(receipt["class_weight"], dtype=np.float32))
        or receipt["boundary_positive_weight"] != pos_weight
    ):
        raise RuntimeError("Original training-only loss weighting changed")
    max_difference = 0.0
    if replay:
        if probabilities is None:
            raise ValueError("Checkpoint replay requires exact nested base probabilities")
        seed_training(seed)
        checkpoint = torch.load(
            directory / "checkpoint.pt", map_location="cuda", weights_only=False
        )
        batches = MemoryBatches(data, probabilities, train)
        if (
            checkpoint["request"] != expected
            or checkpoint["epoch"] != receipt["selected_epoch"]
            or not np.array_equal(checkpoint["scaler_mean"], batches.scaler.mean_)
            or not np.array_equal(checkpoint["scaler_scale"], batches.scaler.scale_)
        ):
            raise RuntimeError("Retained checkpoint request/training-only scaler differs")
        model = new_model(data["features"].shape[1], ARM, protocol)
        model.load_state_dict(checkpoint["model"])
        reproduced = predict(model, batches, held, protocol["batch_size"])
        for key in FIELDS:
            delta = float(np.max(np.abs(reproduced[key] - outputs[key])))
            max_difference = max(max_difference, delta)
            if not np.array_equal(reproduced[key], outputs[key]):
                raise RuntimeError(
                    f"Old-source checkpoint is not bit-exact: {directory}, {key}, {delta}"
                )
        del batches, model, checkpoint
        torch.cuda.empty_cache()
    return outputs, receipt, hashes, max_difference


def selection_from(candidates):
    selected = min(
        candidates,
        key=lambda c: (
            -c["inner_metrics"]["macro_f1"],
            c["inner_metrics"]["nll"],
            c["learning_rate"],
            c["weight_decay"],
        ),
    )
    return {
        "candidates": candidates,
        "selected": selected,
        "refit_epochs": max(1, int(np.rint(np.median(selected["inner_epochs"])))),
        "outer_held_used_for_selection": False,
    }


def fit_guard(data, protocol, train, held, lr, wd, seed, epochs, ancestry, execution_hash):
    return {
        "fit_request": expected_request(
            data, protocol, train, held, lr, wd, seed, epochs, ancestry
        ),
        "source_swap_execution_sha256": execution_hash,
        "train_boundary_targets_sha256": canonical_hash(data["boundary_targets"][train].tolist()),
        "train_boundary_valid_sha256": canonical_hash(data["boundary_valid"][train].tolist()),
    }


def fit_new(
    directory,
    data,
    protocol,
    train,
    held,
    lr,
    wd,
    seed,
    epochs,
    probabilities,
    ancestry,
    execution_hash,
    progress,
):
    guard = fit_guard(data, protocol, train, held, lr, wd, seed, epochs, ancestry, execution_hash)
    guard_attempt(
        directory, guard, complete_files={"receipt.json", "checkpoint.pt", "predictions.npz"}
    )
    fit_memory(
        data,
        probabilities,
        train,
        held,
        arm=ARM,
        learning_rate=lr,
        weight_decay=wd,
        seed=seed,
        epochs=epochs,
        protocol=protocol,
        ancestry=ancestry,
        directory=directory,
        progress=progress,
    )
    return validate_fit(
        directory, data, protocol, train, held, lr, wd, seed, epochs, ancestry, guard=guard
    )
