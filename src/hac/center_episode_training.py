"""Separate, resumable training for the matched center-episode experiment.

The immutable memory loader/scaler is reused, but its positive-weighted boundary
loss is never called. Hazard and utility gradients are clipped independently.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.actor_memory_training import MemoryBatches, seed_training
from hac.center_episode_memory import CenterEpisodeMemory, unweighted_boundary_loss


class EpisodeBatches(MemoryBatches):
    def __init__(self, data, probabilities, train, *, device="cuda"):
        super().__init__(data, probabilities, train, device=device)
        # The inherited constructor computes this for the old experiment only.
        # Remove it so a future accidental reference fails rather than weighting BCE.
        del self.positive_weight

    def loss(self, outputs, index, protocol):
        primary = F.nll_loss(
            outputs["probabilities"].float().clamp_min(1e-12).log(),
            self.labels[index],
            weight=self.class_weight,
        )
        mask = outputs["boundary_valid"] & self.boundary_valid[index]
        if not torch.equal(
            outputs["boundary_left_index"][mask], self.edge_source_slot[index][mask]
        ):
            raise RuntimeError("Episode model and annotation edge ancestry differ")
        boundary = unweighted_boundary_loss(
            outputs, self.boundary_targets[index], self.boundary_valid[index]
        )
        return (
            primary
            + protocol["boundary_loss_weight"] * boundary
            + protocol["gate_square_penalty"] * outputs["gate"].square().mean()
        )


def new_episode_model(input_dim, arm, protocol, device="cuda"):
    settings = protocol["matched_capacity"]
    model = CenterEpisodeMemory(
        input_dim,
        arm,
        width=settings["width"],
        hazard_width=settings["hazard_width"],
        layers=settings["layers"],
        heads=settings["heads"],
        dropout=settings["dropout"],
        parameter_limit=settings["parameter_limit"],
    ).to(device)
    expected = settings.get("parameters_per_arm")
    if expected is not None and model.trainable_parameters != expected:
        raise RuntimeError("Matched episode parameter count changed")
    return model


def clip_episode_gradients(model, maximum_norm=1.0):
    groups = {
        "hazard": list(model.hazard_parameters()),
        "utility": list(model.utility_parameters()),
    }
    identities = [{id(parameter) for parameter in values} for values in groups.values()]
    if identities[0] & identities[1] or identities[0] | identities[1] != {
        id(parameter) for parameter in model.parameters()
    }:
        raise RuntimeError("Hazard and utility parameter partition is not disjoint and exhaustive")
    return {
        name: float(torch.nn.utils.clip_grad_norm_(values, maximum_norm, error_if_nonfinite=True))
        for name, values in groups.items()
    }


@torch.no_grad()
def predict_episode(model, batches, rows, batch_size, tolerance):
    model.eval()
    fields = (
        "probabilities",
        "gate",
        "attention",
        "survival",
        "boundary_probabilities",
        "boundary_logits",
        "boundary_rates",
        "boundary_dt",
        "boundary_valid",
        "boundary_left_index",
        "boundary_mass",
        "utility",
        "segment_probabilities",
        "segment_valid",
        "segment_left_index",
        "segment_right_index",
    )
    outputs = {name: [] for name in fields}
    bound = {
        "maximum_attention_minus_survival": 0.0,
        "maximum_final_neighbor_coefficient_minus_survival": 0.0,
    }
    for start in range(0, len(rows), batch_size):
        result, index = batches.forward(model, rows[start : start + batch_size])
        neighbors = batches.valid[index].clone()
        neighbors[:, 2] = False
        if neighbors.any():
            bound["maximum_attention_minus_survival"] = max(
                bound["maximum_attention_minus_survival"],
                float((result["attention"] - result["survival"])[neighbors].max()),
            )
            bound["maximum_final_neighbor_coefficient_minus_survival"] = max(
                bound["maximum_final_neighbor_coefficient_minus_survival"],
                float(
                    (result["gate"][:, None] * result["attention"] - result["survival"])[
                        neighbors
                    ].max()
                ),
            )
        for name in fields:
            values = result[name].detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite episode prediction: {name}")
            outputs[name].append(values)
    if model.variant == "expected_episode" and max(bound.values()) > tolerance:
        raise RuntimeError("Expected-episode coefficient bound failed")
    result = {name: np.concatenate(values) for name, values in outputs.items()}
    if not np.allclose(result["probabilities"].sum(1), 1, atol=1e-12, rtol=0):
        raise RuntimeError("Episode probabilities are not normalized")
    return result, bound


def fit_episode(
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
    epochs=None,
    device="cuda",
    progress=None,
):
    train, held = np.asarray(train, dtype=np.int64), np.asarray(held, dtype=np.int64)
    if (
        not len(train)
        or not len(held)
        or len(np.unique(train)) != len(train)
        or len(np.unique(held)) != len(held)
    ):
        raise RuntimeError("Episode partitions must be nonempty and unique")
    if min(train.min(), held.min()) < 0 or max(train.max(), held.max()) >= len(data["labels"]):
        raise RuntimeError("Episode partition is outside row population")
    if set(data["scenarios"][train]) & set(data["scenarios"][held]):
        raise RuntimeError("Episode fit/held scenario leakage")
    train_set = set(train.tolist())
    for row in train:
        if not set(data["neighbor_indices"][row, data["valid"][row]].tolist()).issubset(train_set):
            raise RuntimeError("Episode training neighborhood escapes training partition")
    if (
        arm not in protocol["arms"]
        or epochs is not None
        and not 1 <= epochs <= protocol["max_epochs"]
    ):
        raise RuntimeError("Undeclared episode arm or epoch budget")
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
            Path(__file__).with_name("center_episode_memory.py")
        ),
        "batches_seed_code_sha256": file_sha256(
            Path(__file__).with_name("actor_memory_training.py")
        ),
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "train_boundary_targets_sha256": canonical_hash(data["boundary_targets"][train].tolist()),
        "train_boundary_valid_sha256": canonical_hash(data["boundary_valid"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": canonical_hash(data["labels"][held].tolist())
        if epochs is None
        else None,
        "base_ancestry_sha256": canonical_hash(ancestry),
        "boundary_loss": "unweighted_masked_BCE_with_logits",
        "gradient_clipping": "separate_hazard_and_utility_norm_1",
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    receipt_path = directory / "receipt.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Cannot resume a different episode fit request")
    else:
        if any(directory.iterdir()):
            raise RuntimeError(
                "Foreign/partial episode artifacts lack an immutable pre-fit request"
            )
        with request_path.open("x", encoding="utf-8") as handle:
            json.dump(request, handle, indent=2, allow_nan=False)
    names = {path.name for path in directory.iterdir()}
    if receipt_path.exists():
        if names != {"request.json", "receipt.json", "predictions.npz", "checkpoint.pt"}:
            raise RuntimeError("Completed episode artifact inventory changed")
        receipt = json.loads(receipt_path.read_text())
        if receipt["request"] != request:
            raise RuntimeError("Cannot resume a different episode fit")
        if set(receipt["output_sha256"]) != {"predictions.npz", "checkpoint.pt"}:
            raise RuntimeError("Episode output checksum inventory changed")
        for name, expected in receipt["output_sha256"].items():
            if file_sha256(directory / name) != expected:
                raise RuntimeError("Completed episode output changed")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["held_rows"], held) or not np.array_equal(
                saved["sample_ids"], data["sample_ids"][held]
            ):
                raise RuntimeError("Episode resume prediction identities changed")
            return saved["probabilities"], receipt
    if names != {"request.json"}:
        raise RuntimeError(
            "Incomplete episode outputs retained; refusing to overwrite partial artifacts"
        )
    seed_training(seed)
    started = time.perf_counter()
    batches = EpisodeBatches(data, probabilities, train, device=device)
    model = new_episode_model(data["features"].shape[1], arm, protocol, device)
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
        gradient_norms = {"hazard": 0.0, "utility": 0.0}
        for start in range(0, len(order), protocol["batch_size"]):
            rows = order[start : start + protocol["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            outputs, index = batches.forward(model, rows)
            loss = batches.loss(outputs, index, protocol)
            if not torch.isfinite(loss):
                raise RuntimeError("Episode loss became nonfinite")
            loss.backward()
            norms = clip_episode_gradients(model, protocol["gradient_clip_norm"])
            for name, value in norms.items():
                gradient_norms[name] = max(gradient_norms[name], value)
            optimizer.step()
            training_loss += float(loss.detach()) * len(rows)
        row = {
            "epoch": epoch,
            "training_loss": training_loss / len(train),
            "maximum_preclip_norm": gradient_norms,
        }
        if epochs is None:
            predictions, _ = predict_episode(
                model,
                batches,
                held,
                protocol["batch_size"],
                protocol["coefficient_bound_tolerance"],
            )
            measured = probability_metrics(data["labels"][held], predictions["probabilities"])
            row["validation_metrics"] = measured
            key = (-measured["macro_f1"], measured["nll"], epoch)
            if best_key is None or key < best_key:
                best_key, best_epoch, stale = key, epoch, 0
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
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
    predictions, bounds = predict_episode(
        model, batches, held, protocol["batch_size"], protocol["coefficient_bound_tolerance"]
    )
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
        "boundary_positive_weight": None,
        "boundary_loss": "unweighted_masked_BCE_with_logits",
        "coefficient_bound_audit": bounds,
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
