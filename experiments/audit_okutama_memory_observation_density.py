"""Fixed inference-only observation-density stress tests of completed M2-M5.

No optimizer, fitting, checkpoint selection, new pixels, or changed timestamps.
Duplication is deliberately an out-of-training-distribution stress, not a claim
that duplicate evidence is a new observation or an accuracy improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_okutama_memory_results import load_arm, load_inputs, read_json  # noqa: E402

from hac.actor_memory_base import file_sha256, probability_metrics  # noqa: E402
from hac.actor_memory_training import new_model, seed_training  # noqa: E402

CONDITIONS = ("original", "drop_inner_neighbors", "duplicate_all_observations")


def transform_observations(features, probabilities, times, valid, condition):
    """Transform only the observation axis, preserving original source identities."""
    if features.shape[1] != 5 or valid.shape[1] != 5:
        raise ValueError("Stress inputs must be the original five-slot windows")
    if condition == "original":
        return features, probabilities, times, valid, 2
    if condition == "drop_inner_neighbors":
        kept = valid.clone()
        kept[:, (1, 3)] = False
        return features, probabilities, times, kept, 2
    if condition == "duplicate_all_observations":
        return (
            features.repeat_interleave(2, dim=1),
            probabilities.repeat_interleave(2, dim=1),
            times.repeat_interleave(2, dim=1),
            valid.repeat_interleave(2, dim=1),
            4,
        )
    raise ValueError(f"Unknown fixed stress condition: {condition}")


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def inference(model, feature_tensor, base, data, held, batch_size, condition, device):
    collected = {name: [] for name in ("probabilities", "survival", "attention", "gate")}
    for start in range(0, len(held), batch_size):
        rows = held[start : start + batch_size]
        neighbors = torch.as_tensor(data["neighbor_indices"][rows], device=device).clamp_min(0)
        valid = torch.as_tensor(data["valid"][rows], device=device)
        times = torch.as_tensor(data["times"][rows], device=device)
        inputs = transform_observations(
            feature_tensor[neighbors], base[neighbors], times, valid, condition
        )
        result = model(*inputs)
        for name in collected:
            value = result[name].detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise RuntimeError(f"Nonfinite {condition} output: {name}")
            collected[name].append(value)
    return {name: np.concatenate(values) for name, values in collected.items()}


def comparison(labels, values, reference, mask):
    if not mask.any():
        return {"rows": 0}
    correct, previous = values.argmax(1) == labels, reference.argmax(1) == labels
    current_metrics = probability_metrics(labels[mask], values[mask])
    reference_metrics = probability_metrics(labels[mask], reference[mask])
    return {
        "rows": int(mask.sum()),
        "metrics": current_metrics,
        "macro_f1_delta": current_metrics["macro_f1"] - reference_metrics["macro_f1"],
        "rescues_from_unperturbed_arm": int((correct & ~previous & mask).sum()),
        "harms_to_unperturbed_arm": int((~correct & previous & mask).sum()),
        "prediction_flips": int(((values.argmax(1) != reference.argmax(1)) & mask).sum()),
        "mean_probability_total_variation": float(
            np.abs(values[mask] - reference[mask]).sum(1).mean() / 2
        ),
        "maximum_probability_absolute_change": float(np.abs(values[mask] - reference[mask]).max()),
    }


def run(run_directory, output, device):
    if output.exists():
        raise RuntimeError("Choose a fresh diagnostic output; existing results are immutable")
    started = time.perf_counter()
    provenance = {str(Path(__file__)): file_sha256(Path(__file__))}
    helper = ROOT / "experiments/analyze_okutama_memory_results.py"
    provenance[str(helper)] = file_sha256(helper)
    data, baseline, legacy, lock = load_inputs(run_directory, provenance)
    protocol = lock["protocol"]
    with np.load(run_directory / "data/memory_data.npz", allow_pickle=False) as saved:
        features = saved["features"]
    declaration = {
        "declared_utc": datetime.now(UTC).isoformat(),
        "conditions": {
            "original": "Original five slots, same batch128 and saved scaler; replay must pass",
            "drop_inner_neighbors": "Mask slots1 and3; retain center and available endpoints; bridge real time gaps",
            "duplicate_all_observations": "Repeat every slot twice with identical feature/probability/time/mask; first center copy is slot4; no additional physical exposure",
        },
        "arms": protocol["arms"],
        "outer_seeds": protocol["outer_seeds"],
        "replay_maximum_absolute_difference": 1e-10,
        "training_steps": 0,
        "checkpoint_or_hyperparameter_selection": False,
        "purpose": "Previously planned density sensitivity audit; not a new candidate model or score search",
        "stress_caveat": "Ten repeated slots are outside the trained five-slot distribution; M2 changes convolution topology and M5 includes a repeated center among its support witnesses",
        "scope": "All4977 reused development rows; no protected data; drop changes evidence availability while duplication adds no information",
        "provenance": provenance,
    }
    output.mkdir(parents=True)
    (output / "declaration.json").write_text(json.dumps(declaration, indent=2), encoding="utf-8")
    torch.set_num_threads(2)
    seed_training(20260908)
    base = torch.as_tensor(baseline["probabilities"], dtype=torch.float64, device=device)
    results, saved_arrays = {}, {"sample_ids": data["sample_ids"]}
    audits = []
    masks = {"all": np.ones(len(features), dtype=bool), **legacy}
    masks["at_least_one_inner_neighbor_removed"] = data["valid"][:, (1, 3)].any(1)
    for arm in protocol["arms"]:
        original, coverage = load_arm(run_directory, data, lock, arm, provenance)
        if not coverage["complete_primary_population"]:
            raise RuntimeError("Density audit requires all original outer refits complete")
        seed_values = {
            condition: np.zeros((len(protocol["outer_seeds"]), len(features), 3), dtype=np.float64)
            for condition in CONDITIONS
        }
        for fold in protocol["outer_folds"]:
            held = np.flatnonzero(data["folds"] == fold)
            adjacent = data["neighbor_indices"][held][data["valid"][held]]
            if not np.all(data["folds"][adjacent] == fold):
                raise RuntimeError("Historical outer probabilities are not held-fold-local")
            for seed_index, seed in enumerate(protocol["outer_seeds"]):
                directory = run_directory / "models" / arm / f"fold-{fold}" / f"refit-seed-{seed}"
                receipt = read_json(directory / "receipt.json")
                checkpoint = torch.load(
                    directory / "checkpoint.pt", map_location="cpu", weights_only=False
                )
                if checkpoint["request"] != receipt["request"]:
                    raise RuntimeError("Checkpoint and verified receipt requests differ")
                model = new_model(features.shape[1], arm, protocol, device)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                initial_digest = parameter_digest(model)
                scaler = StandardScaler()
                scaler.mean_, scaler.scale_ = checkpoint["scaler_mean"], checkpoint["scaler_scale"]
                scaler.n_features_in_ = features.shape[1]
                values = torch.as_tensor(scaler.transform(features), device=device)
                predictions = {}
                for condition in CONDITIONS:
                    predictions[condition] = inference(
                        model, values, base, data, held, protocol["batch_size"], condition, device
                    )
                    seed_values[condition][seed_index, held] = predictions[condition][
                        "probabilities"
                    ]
                replay_delta = float(
                    np.abs(
                        predictions["original"]["probabilities"]
                        - original["probabilities"][seed_index, held]
                    ).max()
                )
                if replay_delta > declaration["replay_maximum_absolute_difference"]:
                    raise RuntimeError(
                        f"Original model replay failed: {arm}/{fold}/{seed}, {replay_delta}"
                    )
                if initial_digest != parameter_digest(model):
                    raise RuntimeError("Model parameters changed during inference-only audit")
                duplication = predictions["duplicate_all_observations"]
                audits.append(
                    {
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "original_probability_replay_max_difference": replay_delta,
                        "duplicate_survival_max_absolute_change": float(
                            np.abs(
                                duplication["survival"][:, ::2]
                                - predictions["original"]["survival"]
                            ).max()
                        ),
                        "duplicate_merged_attention_max_absolute_change": float(
                            np.abs(
                                duplication["attention"].reshape(len(held), 5, 2).sum(2)
                                - predictions["original"]["attention"]
                            ).max()
                        ),
                        "parameter_digest_unchanged": initial_digest,
                    }
                )
                del model, values, checkpoint
        reference = seed_values["original"].mean(0)
        results[arm] = {}
        for condition, values in seed_values.items():
            mean = values.mean(0)
            saved_arrays[f"{arm}__{condition}__seeds"] = values
            results[arm][condition] = {
                str(name): comparison(data["labels"], mean, reference, mask)
                for name, mask in masks.items()
            }
        print(
            json.dumps(
                {
                    "event": "density_arm_complete",
                    "arm": arm,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    np.savez_compressed(output / "predictions.npz", **saved_arrays)
    summary = {
        "complete": True,
        "evaluation_only": True,
        "new_classifier_fits": 0,
        "rows": len(features),
        "checkpoint_replays": len(audits),
        "declaration_sha256": file_sha256(output / "declaration.json"),
        "results": results,
        "checkpoint_audits": audits,
        "verified_provenance": provenance,
        "predictions_sha256": file_sha256(output / "predictions.npz"),
        "seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"complete": True, "seconds": summary["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path, default=ROOT / ".runs/research_20260908/evidence_memory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run(arguments.run.resolve(), arguments.output.resolve(), arguments.device)
