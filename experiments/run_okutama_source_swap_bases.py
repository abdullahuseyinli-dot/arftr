"""Replay retained old-source P6/M4, then refit all new-source nested bases."""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import run_okutama_evidence_memory as original
import torch
from threadpoolctl import threadpool_limits

from hac.actor_memory_base import (
    canonical_hash,
    file_sha256,
    group_splits,
    probability_metrics,
    training_populations,
)
from hac.source_swap_data import immutable_json, materialize, read_json
from hac.source_swap_execution import ARM, guard_attempt, selection_from, validate_fit

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / ".runs/research_20260908/evidence_memory"
RUN = ROOT / ".runs/research_20260908/source_swap_v1"
PROTOCOL = ROOT / "experiments/okutama_source_swap_protocol.json"


def p6_oof(data, cache, protocol):
    p = np.full((len(data["labels"]), 3), np.nan)
    components = np.full((len(p), 3, 3), np.nan)
    for fold in protocol["outer_folds"]:
        train, held = np.flatnonzero(data["folds"] != fold), np.flatnonzero(data["folds"] == fold)
        p[held] = cache.held_predictions(train, held)
        with np.load(cache.directory(train) / "predictions.npz", allow_pickle=False) as saved:
            components[held] = saved["components"][held]
    probability_metrics(data["labels"], p)
    return p, components


def preflight(run):
    protocol = read_json(PROTOCOL)
    old_lock_path = ROOT / protocol["original_execution_lock"]
    if file_sha256(old_lock_path) != protocol["original_execution_lock_sha256"]:
        raise RuntimeError("Original execution lock changed")
    old_lock = read_json(old_lock_path)
    summary = materialize(ROOT, run / "data")
    data, old_data = original.load_data(run), original.load_data(OLD)
    for key in old_data:
        if key not in ("features", "_file_sha256") and not np.array_equal(data[key], old_data[key]):
            raise RuntimeError(f"Source swap altered metadata: {key}")
    old_cache = original.base_cache(OLD, old_data, old_lock["protocol"])
    preparation = read_json(OLD / "base_preparation.json")
    if old_cache.identity != preparation["base_identity"] or preparation["estimator_fits"] != 3016:
        raise RuntimeError("Original base population identity changed")
    populations = training_populations(data["labels"], data["scenarios"], data["folds"])
    old_hashes = {}
    for rows in populations.values():
        group_splits(data["labels"], data["scenarios"], rows)
        old_cache.load(rows)
        directory = old_cache.directory(rows)
        for name in ("receipt.json", "checkpoint.npz", "predictions.npz"):
            old_hashes[str(directory / name)] = file_sha256(directory / name)
    if len(populations) != 58:
        raise RuntimeError("Nested base population census changed")
    cache = original.base_cache(run, data, old_lock["protocol"])
    if cache.identity == old_cache.identity or set(
        cache.directory(r).name for r in populations.values()
    ) & set(preparation["population_cache_keys"]):
        raise RuntimeError("New source would reuse an old fitted base cache")
    files = {str(ROOT / name): value for name, value in summary["input_sha256"].items()}
    files.update(old_hashes)
    paths = [
        Path(__file__),
        PROTOCOL,
        ROOT / "src/hac/source_swap_execution.py",
        ROOT / "tests/test_source_swap_data.py",
        ROOT / "tests/test_source_swap_execution.py",
        run / "data/summary.json",
        OLD / "base_preparation.json",
    ]
    paths += [run / "data" / name for name in summary["output_sha256"]]
    files.update({str(path): file_sha256(path) for path in paths})
    lock = {
        "protocol": protocol,
        "original_learning_protocol": old_lock["protocol"],
        "new_base_identity": cache.identity,
        "old_base_identity": old_cache.identity,
        "population_count": 58,
        "new_base_estimator_fits": 3016,
        "new_memory_fits": 75,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "metadata_sha256": canonical_hash(
            {
                key: data[key].tolist()
                for key in ("labels", "scenarios", "folds", "recordings", "tracks", "frames")
            }
        ),
        "source_sha256": files,
        "old_fit_replay_required_before_new_fitting": True,
    }
    immutable_json(run / "base_execution_lock.json", lock)
    snapshot = run / "base_source_snapshot"
    for path in (
        Path(__file__),
        PROTOCOL,
        ROOT / "src/hac/source_swap_execution.py",
        ROOT / "src/hac/source_swap_data.py",
    ):
        destination = snapshot / path.relative_to(ROOT)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        if file_sha256(destination) != file_sha256(path):
            raise RuntimeError("Base source snapshot changed")
    return data, cache, old_data, old_cache, lock


def replay_old(run, data, cache, protocol):
    receipt_path = run / "old_source_replay.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for path, expected in receipt["source_sha256"].items():
            if file_sha256(Path(path)) != expected:
                raise RuntimeError("Replayed original source/checkpoint changed")
        if file_sha256(run / "old_source_replay.npz") != receipt["output_sha256"]:
            raise RuntimeError("Replayed output changed")
        return receipt
    p6, components = p6_oof(data, cache, protocol)
    source_hashes = {}
    reference = (
        ROOT / ".runs/research_20260907/okutama_native_video_p6/results/oof_probabilities.npz"
    )
    if file_sha256(reference) != "5a2741dcf1617d0f584d3291c0d40050a380405ab265761697f938cf8bdd8acc":
        raise RuntimeError("Original P6 reference changed")
    source_hashes[str(reference)] = file_sha256(reference)
    with np.load(reference, allow_pickle=False) as saved:
        for key in ("sample_ids", "labels", "folds", "scenarios"):
            if key in saved and not np.array_equal(saved[key], data[key]):
                raise RuntimeError("Original P6 row/fold/scenario identity changed")
        if not np.array_equal(p6, saved["ocvc_uniform_diverse_triad"]):
            raise RuntimeError("Original P6 does not replay exactly")
    source_hashes[str(OLD / "base_replay.npz")] = file_sha256(OLD / "base_replay.npz")
    with np.load(OLD / "base_replay.npz", allow_pickle=False) as saved:
        if not np.array_equal(p6, saved["probabilities"]) or not np.array_equal(
            components, saved["components"]
        ):
            raise RuntimeError("Original P6 component replay differs")
    seed_outputs = np.full((3, len(p6), 3), np.nan)
    checked = 0
    for fold in protocol["outer_folds"]:
        outer_train, outer_held = (
            np.flatnonzero(data["folds"] != fold),
            np.flatnonzero(data["folds"] == fold),
        )
        splits = group_splits(data["labels"], data["scenarios"], outer_train)
        contexts = [cache.meta_probabilities(train) for train, _ in splits]
        directory = OLD / "models" / ARM / f"fold-{fold}"
        expected_dirs = [
            directory / f"config-{c}" / f"inner-{i}" for c in range(4) for i in range(3)
        ] + [directory / f"refit-seed-{s}" for s in protocol["outer_seeds"]]
        if set(directory.rglob("receipt.json")) != {d / "receipt.json" for d in expected_dirs}:
            raise RuntimeError("Original M4 fold does not have exactly15fit receipts")
        candidates = []
        configs = [
            (lr, wd) for lr in protocol["learning_rates"] for wd in protocol["weight_decays"]
        ]
        for config, (lr, wd) in enumerate(configs):
            inner_p = np.full_like(p6, np.nan)
            epochs = []
            for inner, ((train, held), (probabilities, ancestry)) in enumerate(
                zip(splits, contexts, strict=True)
            ):
                outputs, fit, hashes, _ = validate_fit(
                    directory / f"config-{config}" / f"inner-{inner}",
                    data,
                    protocol,
                    train,
                    held,
                    lr,
                    wd,
                    protocol["inner_seed"],
                    None,
                    ancestry,
                    replay=True,
                    probabilities=probabilities,
                )
                source_hashes.update(hashes)
                inner_p[held] = outputs["probabilities"]
                epochs.append(fit["selected_epoch"])
                checked += 1
                original.event(
                    "old_source_checkpoint_replayed",
                    completed=checked,
                    total=75,
                    fold=fold,
                    config=config,
                    inner=inner,
                    max_difference=0,
                )
            candidates.append(
                {
                    "config_index": config,
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "inner_epochs": epochs,
                    "inner_metrics": probability_metrics(
                        data["labels"][outer_train], inner_p[outer_train]
                    ),
                }
            )
        selection = selection_from(candidates)
        if selection != read_json(directory / "selection.json"):
            raise RuntimeError("Original M4 selection is not reconstructed from12innerfits")
        source_hashes[str(directory / "selection.json")] = file_sha256(directory / "selection.json")
        selected = selection["selected"]
        probabilities, ancestry = cache.meta_probabilities(outer_train)
        for index, seed in enumerate(protocol["outer_seeds"]):
            outputs, _, hashes, _ = validate_fit(
                directory / f"refit-seed-{seed}",
                data,
                protocol,
                outer_train,
                outer_held,
                selected["learning_rate"],
                selected["weight_decay"],
                seed,
                selection["refit_epochs"],
                ancestry,
                replay=True,
                probabilities=probabilities,
            )
            source_hashes.update(hashes)
            seed_outputs[index, outer_held] = outputs["probabilities"]
            checked += 1
            original.event(
                "old_source_checkpoint_replayed",
                completed=checked,
                total=75,
                fold=fold,
                seed=seed,
                max_difference=0,
            )
        outer_path = directory / "outer_predictions.npz"
        with np.load(outer_path, allow_pickle=False) as saved:
            if (
                not np.array_equal(saved["held_rows"], outer_held)
                or not np.array_equal(saved["sample_ids"], data["sample_ids"][outer_held])
                or not np.array_equal(saved["seed_probabilities"], seed_outputs[:, outer_held])
                or not np.array_equal(saved["probabilities"], seed_outputs[:, outer_held].mean(0))
            ):
                raise RuntimeError("Retained original outer seed aggregation differs")
        source_hashes[str(outer_path)] = file_sha256(outer_path)
    output = run / "old_source_replay.npz"
    if output.exists():
        raise RuntimeError("Partial replay output retained; do not overwrite")
    np.savez_compressed(
        output,
        sample_ids=data["sample_ids"],
        labels=data["labels"],
        folds=data["folds"],
        scenarios=data["scenarios"],
        old_source_p6=p6,
        old_source_components=components,
        old_source_m4=seed_outputs.mean(0),
        old_source_m4_seeds=seed_outputs,
    )
    receipt = {
        "status": "OLD_SOURCE_P6_AND_ALL75_M4_CHECKPOINTS_BIT_EXACT",
        "rows": 4977,
        "verified_m4_fits": checked,
        "maximum_absolute_difference": 0.0,
        "new_fits": 0,
        "source_sha256": source_hashes,
        "output_sha256": file_sha256(output),
        "base_execution_lock_sha256": file_sha256(run / "base_execution_lock.json"),
    }
    immutable_json(receipt_path, receipt)
    original.event("old_source_replay_complete", fits=checked, maximum_absolute_difference=0)
    return receipt


def prepare_bases(run, data, cache, protocol, workers):
    replay = read_json(run / "old_source_replay.json")
    if replay["status"] != "OLD_SOURCE_P6_AND_ALL75_M4_CHECKPOINTS_BIT_EXACT" or replay[
        "base_execution_lock_sha256"
    ] != file_sha256(run / "base_execution_lock.json"):
        raise RuntimeError("Exact old-source replay gate has not passed")
    populations = training_populations(data["labels"], data["scenarios"], data["folds"])
    execution_hash = file_sha256(run / "base_execution_lock.json")

    def prepare(rows):
        directory = cache.directory(rows)
        guard_attempt(
            directory,
            {"request": cache.request(rows), "source_swap_base_lock_sha256": execution_hash},
            complete_files={"receipt.json", "checkpoint.npz", "predictions.npz"},
        )
        return cache.prepare(rows)

    outer = [np.flatnonzero(data["folds"] != fold) for fold in protocol["outer_folds"]]
    outer_keys = {canonical_hash(rows.tolist()) for rows in outer}
    remaining = [rows for key, rows in populations.items() if key not in outer_keys]
    completed = 0
    with threadpool_limits(limits=1):
        for batch in (outer, remaining):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {pool.submit(prepare, rows): rows for rows in batch}
                for future in as_completed(pending):
                    receipt = future.result()
                    completed += 1
                    original.event(
                        "new_source_base_population_complete",
                        completed=completed,
                        total=58,
                        train_rows=len(pending[future]),
                        seconds=receipt["seconds"],
                        estimator_fits=receipt["estimator_fits"],
                    )
            if completed == 5:
                p6, components = p6_oof(data, cache, protocol)
                path = run / "new_source_p6_oof.npz"
                if not path.exists():
                    np.savez_compressed(
                        path,
                        probabilities=p6,
                        components=components,
                        sample_ids=data["sample_ids"],
                        labels=data["labels"],
                        folds=data["folds"],
                        scenarios=data["scenarios"],
                    )
                else:
                    with np.load(path, allow_pickle=False) as saved:
                        if not np.array_equal(saved["probabilities"], p6) or not np.array_equal(
                            saved["components"], components
                        ):
                            raise RuntimeError("New-source P6 completed predictions changed")
                immutable_json(
                    run / "new_source_p6_oof.json",
                    {
                        "scope": "Exploratory repeated-development source intervention; not a replay of old P6",
                        "metrics": probability_metrics(data["labels"], p6),
                        "source_swap_base_lock_sha256": execution_hash,
                        "output_sha256": file_sha256(path),
                    },
                )
                original.event(
                    "new_source_p6_complete", metrics=probability_metrics(data["labels"], p6)
                )
    files = {}
    receipts = []
    for rows in populations.values():
        receipts.append(cache.load(rows)[1])
        directory = cache.directory(rows)
        if {p.name for p in directory.iterdir()} != {
            "request_guard.json",
            "receipt.json",
            "checkpoint.npz",
            "predictions.npz",
        }:
            raise RuntimeError("Completed nested base inventory changed")
        for path in directory.iterdir():
            files[str(path)] = file_sha256(path)
    immutable_json(
        run / "base_preparation.json",
        {
            "populations": len(receipts),
            "estimator_fits": sum(r["estimator_fits"] for r in receipts),
            "base_identity": cache.identity,
            "population_cache_keys": [cache.directory(rows).name for rows in populations.values()],
            "files_sha256": files,
            "source_swap_base_lock_sha256": execution_hash,
        },
    )
    original.event("new_source_all_bases_complete", populations=58, estimator_fits=3016)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "replay", "base"), required=True)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers not in (1, 2, 3, 4):
        raise ValueError("Locked maximum of4baseworkers")
    torch.set_num_threads(2)
    data, cache, old_data, old_cache, lock = preflight(args.run)
    original.event("source_swap_preflight_passed", rows=4977, changed_inputs=4508, populations=58)
    if args.stage == "preflight":
        return
    with threadpool_limits(limits=1):
        replay_old(args.run, old_data, old_cache, lock["original_learning_protocol"])
    del old_data, old_cache
    if args.stage == "base":
        prepare_bases(args.run, data, cache, lock["original_learning_protocol"], args.workers)


if __name__ == "__main__":
    main()
