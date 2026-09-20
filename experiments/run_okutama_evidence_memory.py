"""Execute the locked actor-evidence-memory trial with strictly nested bases."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from hac.actor_memory_base import (
    NestedBaseCache,
    canonical_hash,
    file_sha256,
    probability_metrics,
    training_populations,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".runs/research_20260908/evidence_memory"
PROTOCOL = ROOT / "experiments/okutama_evidence_memory_protocol.json"


def event(kind: str, **values):
    print(
        json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, **values}),
        flush=True,
    )


def write_json(path: Path, values: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


def load_data(run: Path) -> dict[str, np.ndarray]:
    with np.load(run / "data/memory_data.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data["labels"]) != 4977 or not np.array_equal(np.unique(data["folds"]), np.arange(5)):
        raise RuntimeError("Primary population changed")
    neighbors, valid = data["neighbor_indices"], data["valid"]
    for row in range(len(neighbors)):
        selected = neighbors[row, valid[row]]
        if not (data["scenarios"][selected] == data["scenarios"][row]).all():
            raise RuntimeError("Memory crosses scenarios")
    data["_file_sha256"] = file_sha256(run / "data/memory_data.npz")
    return data


def base_cache(run: Path, data, protocol):
    mapping = {
        "base_long_vjepa": "long_vjepa_mean",
        "base_dual_scale": "dual_scale_vjepa_dino",
        "base_ostm_posture": "orthogonal_moments_posture",
        "base_ostm_motion": "orthogonal_moments_motion",
    }
    path = run / "data/base_features.npz"
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_ids"], data["sample_ids"]):
            raise RuntimeError("Base feature IDs changed")
        features = {name: archive[key] for name, key in mapping.items()}
    return NestedBaseCache(
        run / "base_cache",
        features,
        data["labels"],
        data["scenarios"],
        data["sample_ids"],
        protocol["base_probe"],
        file_sha256(path),
    )


def replay_reference(run, data, cache, protocol):
    oof = np.full((len(data["labels"]), 3), np.nan)
    components = np.full((len(data["labels"]), 3, 3), np.nan)
    for fold in protocol["outer_folds"]:
        train = np.flatnonzero(data["folds"] != fold)
        held = np.flatnonzero(data["folds"] == fold)
        oof[held] = cache.held_predictions(train, held)
        with np.load(cache.directory(train) / "predictions.npz", allow_pickle=False) as archive:
            components[held] = archive["components"][held]
    path = ROOT / ".runs/research_20260907/okutama_native_video_p6/results/oof_probabilities.npz"
    if file_sha256(path) != "5a2741dcf1617d0f584d3291c0d40050a380405ab265761697f938cf8bdd8acc":
        raise RuntimeError("Historical P6 reference bytes changed")
    with np.load(path, allow_pickle=False) as reference:
        for key in ("sample_ids", "labels", "folds"):
            if not np.array_equal(reference[key], data[key]):
                raise RuntimeError(f"P6 replay ID alignment failed: {key}")
        difference = float(np.max(np.abs(oof - reference["ocvc_uniform_diverse_triad"])))
    receipt = {
        "maximum_absolute_difference": difference,
        "metrics": probability_metrics(data["labels"], oof),
        "passed": difference <= protocol["reference_replay_absolute_tolerance"],
    }
    write_json(run / "base_replay.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError(f"P6 replay failed: maximum error {difference}")
    np.savez_compressed(
        run / "base_replay.npz",
        probabilities=oof,
        components=components,
        sample_ids=data["sample_ids"],
    )
    event("p6_replay_passed", **receipt)


def prepare_bases(run, data, protocol, workers=4):
    cache = base_cache(run, data, protocol)
    populations = training_populations(data["labels"], data["scenarios"], data["folds"])
    # Validate every deep regularization split before the first fit.
    from hac.actor_memory_base import group_splits

    for rows in populations.values():
        group_splits(data["labels"], data["scenarios"], rows)
    outer = [np.flatnonzero(data["folds"] != fold) for fold in protocol["outer_folds"]]
    outer_keys = {canonical_hash(rows.tolist()) for rows in outer}
    remaining = [rows for key, rows in populations.items() if key not in outer_keys]
    event("base_preparation_started", populations=len(populations), workers=workers)
    completed = 0

    def fit_batch(batch):
        nonlocal completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(cache.prepare, rows): rows for rows in batch}
            for future in as_completed(pending):
                receipt = future.result()
                completed += 1
                event(
                    "base_population_complete",
                    completed=completed,
                    total=len(populations),
                    train_rows=len(pending[future]),
                    scenarios=receipt["request"]["train_scenarios"],
                    seconds=receipt["seconds"],
                    estimator_fits=receipt["estimator_fits"],
                )

    # BLAS threads are process-wide: set once around all concurrent tasks.
    with threadpool_limits(limits=1):
        fit_batch(outer)
        replay_reference(run, data, cache, protocol)
        fit_batch(remaining)
    receipt = {
        "populations": len(populations),
        "estimator_fits": sum(
            cache.load(rows)[1]["estimator_fits"] for rows in populations.values()
        ),
        "base_identity": cache.identity,
        "population_cache_keys": [cache.directory(rows).name for rows in populations.values()],
    }
    write_json(run / "base_preparation.json", receipt)
    event(
        "base_preparation_complete",
        populations=receipt["populations"],
        estimator_fits=receipt["estimator_fits"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("base", "pilot", "train", "summarize"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    data = load_data(args.run)
    if args.stage == "base":
        prepare_bases(args.run, data, protocol, args.workers)
    elif args.stage == "pilot":
        from hac.actor_memory_training import resource_pilot

        receipt = resource_pilot(protocol)
        write_json(args.run / "resource_pilot.json", receipt)
        event("memory_resource_pilot", **receipt)
    elif args.stage == "train":
        from okutama_memory_execution import train_heads

        train_heads(args.run, data, protocol, arms=args.arms, folds=args.folds)
    else:
        from okutama_memory_execution import summarize

        summarize(args.run, data, protocol)


if __name__ == "__main__":
    main()
