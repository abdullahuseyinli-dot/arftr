"""Independently replay every completed PDI producer and fixed verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import run_okutama_paired_detail_innovation as runner
from hac.actor_memory_base import file_sha256
from hac.intervention_utility import verifier_features
from hac.paired_detail_innovation import PairedDetailInnovation


def replay_producer(data: dict, directory: Path, device: str) -> dict:
    receipt = runner.read_json(directory / "receipt.json")
    saved = runner._load_predictions(directory / "predictions.npz")
    if receipt["predictions_sha256"] != file_sha256(directory / "predictions.npz"):
        raise RuntimeError("PDI prediction hash changed")
    with np.load(directory / "normalization.npz", allow_pickle=False) as norm:
        stats = tuple(norm[name] for name in ("mean", "std", "qmean", "qstd", "class_weights"))
    checkpoint = torch.load(directory / "checkpoint.pt", map_location=device, weights_only=True)
    model = PairedDetailInnovation().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rows = saved["held_rows"].astype(np.int64)
    replay = {name: [] for name in ("probabilities", "actions", "correction", "scaffold_norms")}
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            subset = rows[start : start + 128]
            anchor = saved["anchor_probabilities"][start : start + len(subset)]
            batch = runner._batch(data, subset, stats, receipt["arm"], anchor, device)
            output = model(*batch, arm=receipt["arm"])
            replay["probabilities"].append(output["probabilities"].cpu().numpy())
            replay["actions"].append(output["actions"].cpu().numpy())
            replay["correction"].append(output["correction"].cpu().numpy())
            replay["scaffold_norms"].append(output["coarse_scaffold_norms"].cpu().numpy())
    maxima = {}
    for name in replay:
        value = np.concatenate(replay[name]).astype(np.float64)
        maxima[name] = float(np.max(np.abs(value - saved[name])))
        if maxima[name] > 1e-7:
            raise RuntimeError(f"PDI producer replay mismatch {directory}/{name}: {maxima[name]}")
    if not np.array_equal(saved["actions"][:, 0], saved["anchor_probabilities"]):
        raise RuntimeError("PDI retain action is not bit-exact")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"directory": str(directory), "rows": len(rows), "maximum_deltas": maxima}


def replay_verifier(fold_dir: Path) -> dict:
    predictions = runner._load_predictions(fold_dir / "predictions.npz")
    with np.load(fold_dir / "verifier.npz", allow_pickle=False) as saved:
        mean, scale = saved["scaler_mean"], saved["scaler_scale"]
        features = predictions["verifier_features"]
        transformed = (features - mean[None, None, :]) / scale[None, None, :]
        class_scores = np.column_stack([
            transformed[:, index] @ saved[f"class_coef_{index}"] + saved[f"class_intercept_{index}"][0]
            for index in range(3)
        ])
        nll_scores = np.column_stack([
            transformed[:, index] @ saved[f"nll_coef_{index}"] + saved[f"nll_intercept_{index}"][0]
            for index in range(3)
        ])
    eligible = (class_scores > 0) & (nll_scores > 0)
    best = np.where(eligible, class_scores, -np.inf).argmax(1)
    active = eligible.any(1)
    choices = np.zeros(len(features), dtype=np.int64)
    choices[active] = best[active] + 1
    routed = predictions["anchor_probabilities"].copy()
    routed[active] = predictions["actions"][np.flatnonzero(active), choices[active]]
    if not np.array_equal(choices, predictions["choices"]) or not np.array_equal(routed, predictions["routed_probabilities"]):
        raise RuntimeError("PDI verifier exact replay failed")
    names = np.load(fold_dir / "verifier.npz", allow_pickle=False)["feature_names"].astype(str).tolist()
    regenerated, regenerated_names = verifier_features(
        predictions["anchor_probabilities"], predictions["actions"], predictions["correction"],
        predictions["scaffold_norms"], runner.data()["quality"][predictions["held_rows"]],
    )
    if names != regenerated_names or not np.array_equal(regenerated, features):
        raise RuntimeError("PDI verifier feature regeneration failed")
    return {"rows": len(features), "interventions": int(active.sum()), "exact": True}


def audit(run: Path, device: str) -> dict:
    runner.validate_lock(run)
    data = runner.data()
    producer_results, verifier_results, stopped = [], [], []
    for arm in runner.ARMS:
        for outer in range(5):
            fold_dir = run / arm / f"fold-{outer}"
            receipt = runner.read_json(fold_dir / "receipt.json")
            for inner in range(5):
                producer_results.append(replay_producer(data, fold_dir / f"inner-{inner}", device))
            if receipt["status"] == "PDI_ARM_FOLD_STOPPED_INNER_CAPACITY_NO_GO":
                stopped.append({"arm": arm, "outer": outer})
                continue
            if receipt["status"] != "PDI_ARM_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED":
                raise RuntimeError("unexpected PDI arm/fold state")
            producer_results.append(replay_producer(data, fold_dir / "final", device))
            verifier_results.append({"arm": arm, "outer": outer, **replay_verifier(fold_dir)})
    result = {
        "status": "PDI_INDEPENDENT_FULL_PRODUCER_AND_VERIFIER_REPLAY_PASS",
        "producer_replays": len(producer_results), "verifier_replays": len(verifier_results),
        "stopped_arm_folds": stopped, "outer_labels_read_during_replay_fit": 0,
        "maximum_probability_delta": max((item["maximum_deltas"]["probabilities"] for item in producer_results), default=0.0),
        "details": producer_results, "verifiers": verifier_results,
    }
    runner.write_json(run / "independent_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("details", "verifiers")}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    audit(args.run.resolve(), args.device)


if __name__ == "__main__":
    main()
