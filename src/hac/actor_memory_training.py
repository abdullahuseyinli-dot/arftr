"""Train small evidence-memory heads, never their frozen visual encoders."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from hac.actor_evidence_memory import ActorEvidenceMemory
from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics


def seed_training(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def new_model(input_dim, arm, protocol, device="cuda"):
    return ActorEvidenceMemory(
        input_dim,
        arm,
        width=protocol["width"],
        layers=protocol["layers"],
        heads=protocol["heads"],
        dropout=protocol["dropout"],
        parameter_limit=protocol["max_parameters"],
    ).to(device)


class MemoryBatches:
    def __init__(self, data, probabilities, train, *, device="cuda"):
        self.scaler = StandardScaler().fit(data["features"][train])
        self.features = torch.as_tensor(self.scaler.transform(data["features"]), device=device)
        # Double precision input preserves the exact frozen center fallback at inference.
        self.probabilities = torch.as_tensor(probabilities, dtype=torch.float64, device=device)
        self.neighbors = torch.as_tensor(data["neighbor_indices"], device=device).clamp_min(0)
        self.times = torch.as_tensor(data["times"], device=device)
        self.valid = torch.as_tensor(data["valid"], device=device)
        self.labels = torch.as_tensor(data["labels"], device=device)
        self.boundary_targets = torch.as_tensor(data["boundary_targets"], device=device)
        self.boundary_valid = torch.as_tensor(data["boundary_valid"], device=device)
        self.edge_source_slot = torch.as_tensor(data["edge_source_slot"], device=device)
        self.device = device
        counts = np.bincount(data["labels"][train], minlength=3)
        if (counts == 0).any():
            raise RuntimeError("Meta-training partition is missing a class")
        self.class_weight = torch.as_tensor(
            len(train) / (3 * counts), dtype=torch.float32, device=device
        )
        edge_targets = data["boundary_targets"][train][data["boundary_valid"][train]]
        positive = float(edge_targets.sum())
        self.positive_weight = torch.tensor(
            (len(edge_targets) - positive) / positive if positive else 1.0,
            dtype=torch.float32,
            device=device,
        )

    def forward(self, model, rows):
        index = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        neighbors = self.neighbors[index]
        return model(
            self.features[neighbors],
            self.probabilities[neighbors],
            self.times[index],
            self.valid[index],
        ), index

    def loss(self, outputs, index, protocol):
        primary = F.nll_loss(
            outputs["probabilities"].float().clamp_min(1e-12).log(),
            self.labels[index],
            weight=self.class_weight,
        )
        boundary_mask = outputs["boundary_valid"] & self.boundary_valid[index]
        if boundary_mask.any():
            if not torch.equal(
                outputs["boundary_left_index"][boundary_mask],
                self.edge_source_slot[index][boundary_mask],
            ):
                raise RuntimeError("Model and annotation edge ancestry differ")
            boundary = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"][boundary_mask],
                self.boundary_targets[index][boundary_mask],
                pos_weight=self.positive_weight,
            )
        else:
            boundary = primary.new_zeros(())
        return (
            primary
            + protocol["boundary_loss_weight"] * boundary
            + protocol["gate_square_penalty"] * outputs["gate"].square().mean()
        )


@torch.no_grad()
def predict(model, batches, rows, batch_size):
    model.eval()
    outputs = {
        name: []
        for name in ("probabilities", "gate", "attention", "survival", "boundary_probabilities")
    }
    for start in range(0, len(rows), batch_size):
        result, _ = batches.forward(model, rows[start : start + batch_size])
        for name in outputs:
            outputs[name].append(result[name].detach().cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in outputs.items()}


def fit_memory(
    data,
    probabilities,
    train,
    held,
    *,
    arm,
    learning_rate,
    weight_decay,
    seed,
    protocol,
    ancestry,
    directory: Path,
    epochs: int | None = None,
    device="cuda",
    progress=None,
):
    """Inner fits select a checkpoint; outer refits receive already selected epochs."""
    if set(data["scenarios"][train]) & set(data["scenarios"][held]):
        raise RuntimeError("Memory fit/held scenario leakage")
    request = {
        "data_sha256": data.get("_file_sha256", "synthetic_test_data"),
        "arm": arm,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "epochs": epochs,
        "protocol_sha256": canonical_hash(protocol),
        "training_code_sha256": file_sha256(Path(__file__)),
        "architecture_code_sha256": file_sha256(
            Path(__file__).with_name("actor_evidence_memory.py")
        ),
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": canonical_hash(data["labels"][held].tolist())
        if epochs is None
        else None,
        "base_ancestry_sha256": canonical_hash(ancestry),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["request"] != request:
            raise RuntimeError(f"Cannot resume a different memory fit in {directory}")
        for name in ("predictions.npz", "checkpoint.pt"):
            if file_sha256(directory / name) != receipt["output_sha256"][name]:
                raise RuntimeError("Memory fit output changed after completion")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            return saved["probabilities"], receipt
    seed_training(seed)
    started = time.perf_counter()
    batches = MemoryBatches(data, probabilities, train, device=device)
    model = new_model(data["features"].shape[1], arm, protocol, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    maximum = epochs if epochs is not None else protocol["max_epochs"]
    generator = np.random.default_rng(seed)
    best_key, best_state, best_epoch, stale = None, None, 0, 0
    history = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, maximum + 1):
        model.train()
        order = generator.permutation(train)
        training_loss = 0.0
        for start in range(0, len(order), protocol["batch_size"]):
            rows = order[start : start + protocol["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            outputs, index = batches.forward(model, rows)
            loss = batches.loss(outputs, index, protocol)
            if not torch.isfinite(loss):
                raise RuntimeError("Memory loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            training_loss += float(loss.detach()) * len(rows)
        row = {"epoch": epoch, "training_loss": training_loss / len(train)}
        if epochs is None:
            predictions = predict(model, batches, held, protocol["batch_size"])["probabilities"]
            measured = probability_metrics(data["labels"][held], predictions)
            row["validation_metrics"] = measured
            key = (-measured["macro_f1"], measured["nll"], epoch)
            if best_key is None or key < best_key:
                best_key, best_epoch, stale = key, epoch, 0
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            else:
                stale += 1
        history.append(row)
        if progress is not None and (epoch == 1 or epoch % 5 == 0):
            progress(
                epoch=epoch,
                maximum=maximum,
                training_loss=row["training_loss"],
                elapsed_seconds=time.perf_counter() - started,
            )
        if epochs is None and stale >= protocol["early_stopping_patience"]:
            break
    if epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum
    predictions = predict(model, batches, held, protocol["batch_size"])
    np.savez_compressed(
        directory / "predictions.npz",
        held_rows=held,
        sample_ids=data["sample_ids"][held],
        **predictions,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "scaler_mean": batches.scaler.mean_,
            "scaler_scale": batches.scaler.scale_,
            "request": request,
            "epoch": best_epoch,
        },
        directory / "checkpoint.pt",
    )
    receipt = {
        "request": request,
        "train_rows": train.tolist(),
        "held_rows": held.tolist(),
        "base_ancestry": ancestry,
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "class_weight": batches.class_weight.cpu().tolist(),
        "boundary_positive_weight": float(batches.positive_weight),
        "parameters": model.trainable_parameters,
        "held_metrics": probability_metrics(data["labels"][held], predictions["probabilities"]),
        "seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else 0,
        "output_sha256": {
            name: file_sha256(directory / name) for name in ("predictions.npz", "checkpoint.pt")
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    del optimizer, model, batches
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return predictions["probabilities"], receipt


def resource_pilot(protocol, *, input_dim=3072, device="cuda"):
    """Synthetic labels only: choose feasibility, never architecture or accuracy."""
    results = {}
    torch.set_num_threads(4)
    for arm in protocol["arms"]:
        seed_training(42)
        model = new_model(input_dim, arm, protocol, device)
        batch = protocol["batch_size"]
        features = torch.randn(batch, 5, input_dim, device=device)
        probabilities = torch.full((batch, 5, 3), 1 / 3, dtype=torch.float64, device=device)
        times = torch.arange(-2, 3, dtype=torch.float32, device=device).expand(batch, -1)
        valid = torch.ones(batch, 5, dtype=torch.bool, device=device)
        labels = torch.arange(batch, device=device) % 3
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            model.zero_grad(set_to_none=True)
            output = model(features, probabilities, times, valid)
            loss = F.nll_loss(output["probabilities"].float().log(), labels)
            mask = output["boundary_valid"]
            if mask.any():
                loss = loss + 0.2 * F.binary_cross_entropy_with_logits(
                    output["boundary_logits"][mask],
                    torch.zeros_like(output["boundary_logits"][mask]),
                )
            loss.backward()
        torch.cuda.synchronize()
        results[arm] = {
            "parameters": model.trainable_parameters,
            "batch_size": batch,
            "mean_forward_backward_seconds": (time.perf_counter() - start) / 10,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "finite_loss": bool(torch.isfinite(loss)),
        }
        del model, features, probabilities, output, loss
        torch.cuda.empty_cache()
    return {
        "synthetic_only": True,
        "results": results,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "batch_size": protocol["batch_size"],
    }
