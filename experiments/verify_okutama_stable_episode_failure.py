"""Label-blind, frozen-state validation of numerical v2 on the captured v1 failure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pilot_okutama_stable_center_episode import state_digest

from hac.actor_memory_base import file_sha256
from hac.actor_memory_training import seed_training
from hac.center_episode_memory import CenterEpisodeMemory
from hac.stable_center_episode_memory import StableCenterEpisodeMemory


@torch.no_grad()
def main():
    root = Path(__file__).resolve().parents[1]
    forensic = root / ".runs/research_20260908/center_episode_forensics_v1"
    output = root / ".runs/research_20260908/center_episode_v2/actual_failure_validation.json"
    if output.exists():
        raise RuntimeError("Existing actual-case numerical validation is immutable")
    failure_path = forensic / "failure_receipt.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    for name, expected in failure["artifact_sha256"].items():
        if file_sha256(forensic / name) != expected:
            raise RuntimeError("Forensic failure artifact changed")
    with np.load(forensic / "first_failure_batch.npz", allow_pickle=False) as archive:
        # No class/boundary labels or epoch-selection metrics are used here.
        inputs = tuple(
            torch.as_tensor(archive[name], device="cuda")
            for name in ("input_features", "input_probabilities", "input_times", "input_valid")
        )
        saved_probability = archive["probabilities"].copy()
        saved_attention = archive["attention"].copy()
        saved_survival = archive["survival"].copy()
    state = torch.load(forensic / "first_failure_state.pt", weights_only=False, map_location="cpu")
    seed_training(42)
    torch.set_num_threads(2)
    models = {}
    for name, constructor in (("v1", CenterEpisodeMemory), ("v2", StableCenterEpisodeMemory)):
        model = constructor(3072, "expected_episode").cuda().eval()
        model.load_state_dict(state["model"], strict=True)
        before = state_digest(model)
        result = model(*inputs)
        if state_digest(model) != before:
            raise RuntimeError("Numerical validation changed model state")
        models[name] = {key: value.detach().cpu() for key, value in result.items()}
        del model
    old, new = models["v1"], models["v2"]
    for key, reference in (
        ("probabilities", saved_probability),
        ("attention", saved_attention),
        ("survival", saved_survival),
    ):
        if not np.array_equal(old[key].numpy(), reference):
            raise RuntimeError(f"Frozen original failure replay is not exact: {key}")
    for key in ("boundary_mass", "boundary_logits", "boundary_rates", "utility"):
        if not torch.equal(old[key], new[key]):
            raise RuntimeError(f"A neural module changed in numerical-only replay: {key}")
    valid = inputs[3].cpu()
    neighbors = valid.clone()
    neighbors[:, 2] = False
    old_excess = old["attention"] - old["survival"]
    old_max = float(old_excess[neighbors].max())
    new_max = float((new["attention"] - new["survival"])[neighbors].max())
    final_max = float((new["gate"][:, None] * new["attention"] - new["survival"])[neighbors].max())
    if old_max <= 1e-6 or max(new_max, final_max) > 1e-6:
        raise RuntimeError("Original failure/stable unchanged-bound diagnostic did not pass")
    if (
        not torch.isfinite(new["probabilities"]).all()
        or (new["probabilities"].sum(1) - 1).abs().max() > 1e-12
    ):
        raise RuntimeError("Stable actual-case probability simplex failed")
    location = int(old_excess.masked_fill(~neighbors, -torch.inf).reshape(-1).argmax())
    row, slot = divmod(location, 5)
    receipt = {
        "status": "FROZEN_ACTUAL_FAILURE_NUMERICAL_V2_VALIDATION_PASSED",
        "forensic_only": True,
        "new_optimizer_steps": 0,
        "new_classifier_fits": 0,
        "classification_labels_used": 0,
        "classification_scores_calculated": False,
        "original_failure_epoch": failure["epoch"],
        "original_model_and_batch_replay_bit_exact": True,
        "original_and_stable_neural_outputs_bit_exact": [
            "boundary_mass",
            "boundary_logits",
            "boundary_rates",
            "utility",
        ],
        "identical_model_state_hash": before,
        "parameter_states_unchanged": True,
        "unchanged_tolerance": 1e-6,
        "original_maximum_attention_minus_survival": old_max,
        "stable_maximum_attention_minus_survival": new_max,
        "stable_maximum_final_neighbor_coefficient_minus_survival": final_max,
        "original_maximum_segment_mass_error": float(
            (old["segment_probabilities"].sum(1) - 1).abs().max()
        ),
        "stable_maximum_segment_mass_error": float(
            (new["segment_probabilities"].sum(1) - 1).abs().max()
        ),
        "maximum_probability_change_without_retraining": float(
            (new["probabilities"] - old["probabilities"]).abs().max()
        ),
        "offending_case": {
            "batch_row": row,
            "slot": slot,
            "edge_masses": old["boundary_mass"][row].tolist(),
            "utility": old["utility"][row].tolist(),
            "old_survival": old["survival"][row].tolist(),
            "stable_survival": new["survival"][row].tolist(),
            "old_attention": old["attention"][row].tolist(),
            "stable_attention": new["attention"][row].tolist(),
        },
        "failure_receipt_sha256": file_sha256(failure_path),
        "forensic_artifact_sha256": failure["artifact_sha256"],
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in (
                Path(__file__),
                root / "src/hac/stable_center_episode_memory.py",
                root / "src/hac/center_episode_memory.py",
                root / "experiments/pilot_okutama_stable_center_episode.py",
            )
        },
        "interpretation": "Numerical invariant repaired on the same frozen state and observations, not a classifier improvement or a v2 training checkpoint",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
