"""One authorized diagnostic replay of the failed v1 inner fit; never a candidate."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import run_okutama_center_episode as original
import torch

from hac.actor_memory_base import canonical_hash, file_sha256, group_splits, probability_metrics
from hac.actor_memory_training import seed_training
from hac.center_episode_training import EpisodeBatches, clip_episode_gradients, new_episode_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260908/center_episode_forensics_v1"
FAILED = original.RUN / "models/expected_episode/fold-1/config-2/inner-1"


@torch.no_grad()
def inspect_validation(model, batches, held, protocol, epoch, data, optimizer, history):
    model.eval()
    predicted = []
    for start in range(0, len(held), protocol["batch_size"]):
        rows = held[start : start + protocol["batch_size"]]
        output, index = batches.forward(model, rows)
        neighbors = batches.valid[index].clone()
        neighbors[:, 2] = False
        attention_excess = output["attention"] - output["survival"]
        final_excess = output["gate"][:, None] * output["attention"] - output["survival"]
        bound = {
            "maximum_attention_minus_survival": float(attention_excess[neighbors].max())
            if neighbors.any()
            else 0.0,
            "maximum_final_neighbor_coefficient_minus_survival": float(
                final_excess[neighbors].max()
            )
            if neighbors.any()
            else 0.0,
        }
        if max(bound.values()) > protocol["coefficient_bound_tolerance"]:
            values = {name: value.detach().cpu().numpy() for name, value in output.items()}
            neighbor_rows = batches.neighbors[index]
            values.update(
                rows=rows,
                sample_ids=data["sample_ids"][rows],
                input_features=batches.features[neighbor_rows].cpu().numpy(),
                input_probabilities=batches.probabilities[neighbor_rows].cpu().numpy(),
                input_times=batches.times[index].cpu().numpy(),
                input_valid=batches.valid[index].cpu().numpy(),
                attention_excess=attention_excess.cpu().numpy(),
                final_excess=final_excess.cpu().numpy(),
            )
            np.savez_compressed(RUN / "first_failure_batch.npz", **values)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "scaler_mean": batches.scaler.mean_,
                    "scaler_scale": batches.scaler.scale_,
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all(),
                },
                RUN / "first_failure_state.pt",
            )
            report = {
                "diagnostic_only": True,
                "candidate_result": False,
                "original_fit": str(FAILED.relative_to(ROOT)),
                "epoch": epoch,
                "held_batch_start": start,
                "rows": rows.tolist(),
                "bound": bound,
                "unchanged_tolerance": protocol["coefficient_bound_tolerance"],
                "segment_sum_minimum": float(output["segment_probabilities"].sum(1).min()),
                "segment_sum_maximum": float(output["segment_probabilities"].sum(1).max()),
                "maximum_boundary_mass": float(output["boundary_mass"].max()),
                "completed_epoch_history": history,
                "artifact_sha256": {
                    name: file_sha256(RUN / name)
                    for name in (
                        "first_failure_batch.npz",
                        "first_failure_state.pt",
                        "replay_lock.json",
                    )
                },
                "stop_reason": "First original coefficient-bound failure captured; no subsequent epochs or selection",
            }
            original.write_immutable(RUN / "failure_receipt.json", report)
            original.event(
                "forensic_bound_captured",
                **{k: report[k] for k in ("epoch", "held_batch_start", "bound")},
            )
            return None
        predicted.append(output["probabilities"].cpu().numpy())
    return np.concatenate(predicted)


def main():
    torch.set_num_threads(2)
    if RUN.exists():
        raise RuntimeError("One forensic replay only; existing directory is preserved")
    protocol = original.audit.read_json(original.PROTOCOL)
    failed_request = original.audit.read_json(FAILED / "request.json")
    if {path.name for path in FAILED.iterdir()} != {"request.json"}:
        raise RuntimeError("Original failed-fit inventory changed")
    if len(list((original.RUN / "models").rglob("receipt.json"))) != 97:
        raise RuntimeError("Original v1 fit census changed")
    old_lock = original.audit.read_json(original.RUN / "execution_lock.json")
    for path, expected in old_lock["source_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Locked v1 source/input changed: {path}")
    data, cache, preflight = original.preflight(RUN, protocol)
    outer_train = np.flatnonzero(data["folds"] != 1)
    train, held = group_splits(
        data["labels"],
        data["scenarios"],
        outer_train,
        n_splits=protocol["inner_folds"],
        seed=protocol["split_seed"],
    )[1]
    probabilities, ancestry = cache.meta_probabilities(train)
    expected = {
        "arm": "expected_episode",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
        "epochs": None,
        "protocol_sha256": canonical_hash(protocol),
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "train_boundary_targets_sha256": canonical_hash(data["boundary_targets"][train].tolist()),
        "train_boundary_valid_sha256": canonical_hash(data["boundary_valid"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": canonical_hash(data["labels"][held].tolist()),
        "base_ancestry_sha256": canonical_hash(ancestry),
    }
    if any(failed_request[key] != value for key, value in expected.items()):
        raise RuntimeError("Forensic replay does not exactly match the original failed request")
    original.write_immutable(
        RUN / "replay_lock.json",
        {
            "diagnostic_only": True,
            "maximum_replayed_fits": 1,
            "authorization": "Parent-authorized exact failed inner-fit forensic replay only",
            "original_completed_fits": 97,
            "original_failed_requests": 1,
            "original_execution_lock_sha256": file_sha256(original.RUN / "execution_lock.json"),
            "original_failed_request_sha256": file_sha256(FAILED / "request.json"),
            "original_request": failed_request,
            "preflight": preflight,
            "instrumented_replay_source_sha256": file_sha256(Path(__file__)),
            "train_rows": train.tolist(),
            "held_rows": held.tolist(),
            "base_ancestry": ancestry,
            "no_new_base_fits": True,
            "change": "Only captures and stops at the first original bound-failing validation batch; no model/loss/optimizer/seed changes",
        },
    )
    seed_training(42)
    started = time.perf_counter()
    batches = EpisodeBatches(data, probabilities, train)
    model = new_episode_model(3072, "expected_episode", protocol)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = np.random.default_rng(42)
    best_key, stale, history = None, 0, []
    for epoch in range(1, protocol["max_epochs"] + 1):
        model.train()
        order = generator.permutation(train)
        training_loss = 0.0
        gradient_norms = {"hazard": 0.0, "utility": 0.0}
        for start in range(0, len(order), protocol["batch_size"]):
            rows = order[start : start + protocol["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            output, index = batches.forward(model, rows)
            loss = batches.loss(output, index, protocol)
            if not torch.isfinite(loss):
                raise RuntimeError("Forensic loss became nonfinite")
            loss.backward()
            for name, value in clip_episode_gradients(
                model, protocol["gradient_clip_norm"]
            ).items():
                gradient_norms[name] = max(gradient_norms[name], value)
            optimizer.step()
            training_loss += float(loss.detach()) * len(rows)
        predicted = inspect_validation(
            model, batches, held, protocol, epoch, data, optimizer, history
        )
        if predicted is None:
            return
        measured = probability_metrics(data["labels"][held], predicted)
        key = (-measured["macro_f1"], measured["nll"], epoch)
        if best_key is None or key < best_key:
            best_key, stale = key, 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "training_loss": training_loss / len(train),
            "maximum_preclip_norm": gradient_norms,
            "validation_metrics": measured,
        }
        history.append(row)
        original.event(
            "forensic_epoch",
            epoch=epoch,
            training_loss=row["training_loss"],
            elapsed_seconds=time.perf_counter() - started,
        )
        if stale >= protocol["early_stopping_patience"]:
            break
    original.write_immutable(
        RUN / "not_reproduced.json", {"history": history, "diagnostic_only": True}
    )
    raise RuntimeError("Original failure not reproduced; no further replay authorized")


if __name__ == "__main__":
    main()
