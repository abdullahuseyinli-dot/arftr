# ruff: noqa: E402

"""Prepare and fit the missing population-specific ARFTR ancestors.

Stages are immutable and restart-safe. ``prepare`` writes no fitted artifact;
``bases`` prepares only missing P6 populations; ``fit-one`` runs the exact
historical 15-fit M4 and 15-fit A3 recipes for one noncanonical F(S), then
composes its label-embargoed predictions outside S.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import run_okutama_evidence_memory as evidence
import run_okutama_sear_matrix as sear_driver
import run_okutama_source_swap_memory as memory_driver
from threadpoolctl import threadpool_limits

from hac.actor_memory_base import (
    NestedBaseCache,
    canonical_hash,
    file_sha256,
    group_splits,
    probability_metrics,
)
from hac.generalized_arftr import PopulationInputs, execute_population
from hac.layered_base_cache import LayeredNestedBaseCache
from hac.matr_artifacts import load_cached_study_data
from hac.nested_arftr_plan import fixed_recipe_plan
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1"
PARITY = ROOT / ".runs/research_20260913/generalized_arftr_parity_v1/audit.json"
DESIGN_PROTOCOL = ROOT / "experiments/okutama_source_posture_protocol.json"
ARFTR_PROTOCOL = ROOT / "experiments/okutama_arftr_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v1"
SEEDS = (42, 43, 44)
A3_ARM = "a3_unrestricted_templates"


def event(name: str, **values: Any) -> None:
    print(json.dumps({"time": time.time(), "event": name, **values}), flush=True)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return np.array_equal(left, right, equal_nan=True)


def _immutable_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not _arrays_equal(saved[name], value) for name, value in arrays.items()
            ):
                raise RuntimeError(f"Immutable generalized ancestor changed: {path}")
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _source_data() -> dict[str, np.ndarray]:
    data = evidence.load_data(SOURCE)
    canonical = load_cached_study_data(SOURCE)
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(data[name], canonical[name]):
            raise RuntimeError("Generalized ancestor cohort changed")
    return data


def _plans(data: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    plan = fixed_recipe_plan(
        data["labels"],
        data["scenarios"],
        data["folds"],
        data["sample_ids"],
        enforce_canonical_counts=True,
    )
    populations = {
        receipt["id"]: np.flatnonzero(np.isin(data["scenarios"], receipt["scenarios"]))
        for receipt in plan["F_populations"]
    }
    if len(populations) != 19:
        raise RuntimeError("Generalized F population count changed")
    return plan, populations


def _required_base_populations(
    data: dict[str, np.ndarray], populations: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}

    def add(rows: np.ndarray) -> None:
        result[canonical_hash(rows.tolist())] = rows

    for population in populations.values():
        add(population)
        for inner_train, _ in group_splits(data["labels"], data["scenarios"], population):
            add(inner_train)
            for base_train, _ in group_splits(
                data["labels"], data["scenarios"], inner_train
            ):
                add(base_train)
    if len(result) != 183:
        raise RuntimeError("Generalized base population count changed")
    return result


def _historical_base(data: dict[str, np.ndarray]):
    base_lock = read_json(SOURCE / "base_execution_lock.json")
    return evidence.base_cache(SOURCE, data, base_lock["original_learning_protocol"])


def _layered_base(run: Path, data: dict[str, np.ndarray]) -> LayeredNestedBaseCache:
    historical = _historical_base(data)
    current = NestedBaseCache(
        run / "base_cache",
        historical.features,
        historical.labels,
        historical.scenarios,
        historical.sample_ids,
        historical.config,
        historical.identity["feature_sha256"],
    )
    return LayeredNestedBaseCache(historical, current)


def _canonical_population_ids(data: dict[str, np.ndarray]) -> set[str]:
    return {
        canonical_hash(sorted(set(data["scenarios"][data["folds"] != fold].tolist())))
        for fold in range(5)
    }


def _locked_inputs() -> tuple[Path, ...]:
    paths = {
        PARITY,
        DESIGN_PROTOCOL,
        ARFTR_PROTOCOL,
        SOURCE / "data/memory_data.npz",
        SOURCE / "data/base_features.npz",
        SOURCE / "base_execution_lock.json",
        SOURCE / "base_preparation.json",
        SOURCE / "memory_execution_lock.json",
        SEAR / "execution_lock.json",
        ARFTR / "execution_lock.json",
        ARFTR / "results/v0001/oof_probabilities.npz",
        Path(__file__),
        ROOT / "src/hac/generalized_arftr.py",
        ROOT / "src/hac/layered_base_cache.py",
        ROOT / "src/hac/actor_memory_base.py",
        ROOT / "src/hac/actor_memory_training.py",
        ROOT / "src/hac/actor_evidence_memory.py",
        ROOT / "src/hac/sear.py",
        ROOT / "src/hac/sear_controls.py",
        ROOT / "src/hac/sear_training.py",
        ROOT / "experiments/run_okutama_evidence_memory.py",
        ROOT / "experiments/run_okutama_source_swap_memory.py",
        ROOT / "experiments/run_okutama_sear_matrix.py",
    }
    for fold in range(5):
        for name in ("receipt.json", "checkpoint.npz", "predictions.npz"):
            paths.add(ARFTR / f"fold-{fold}" / name)
    if any(not path.is_file() for path in paths):
        raise RuntimeError("A generalized ancestor lock input is missing")
    return tuple(sorted(paths, key=lambda path: str(path.resolve())))


def prepare(run: Path) -> dict[str, Any]:
    parity = read_json(PARITY)
    if (
        parity.get("status")
        != "GENERALIZED_ARFTR_FT_PARITY_EXACT_AND_REUSE_AUDIT_PASS"
        or not parity.get("final_mean_probabilities_exact")
        or parity.get("final_maximum_absolute_difference") != 0.0
        or parity.get("reuse", {}).get("canonical_F_populations_certified") != 5
        or parity.get("reuse", {}).get("base_populations_certified_reusable") != 58
    ):
        raise RuntimeError("Exact generalized F(T) parity has not passed")
    data = _source_data()
    plan, populations = _plans(data)
    required = _required_base_populations(data, populations)
    sear_lock, _patches, _center, sear_data = sear_driver.load_locked(SEAR)
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(sear_data[name], data[name]):
            raise RuntimeError("M4 and A3 generalized cohorts differ")
    canonical_ids = _canonical_population_ids(data)
    if len(canonical_ids) != 5 or not canonical_ids <= set(populations):
        raise RuntimeError("Canonical reusable F identities changed")
    inputs = [_record(path) for path in _locked_inputs()]
    lock = {
        "status": "GENERALIZED_ARFTR_ANCESTORS_LOCKED_BEFORE_NEW_FITS",
        "rows": len(data["labels"]),
        "protocol": sear_lock["protocol"],
        "parameter_counts": sear_lock["parameter_counts"],
        "device": sear_lock["device"],
        "memory_protocol": read_json(SOURCE / "base_execution_lock.json")[
            "original_learning_protocol"
        ],
        "F_populations": {
            key: {
                "rows": len(rows),
                "scenarios": sorted(set(data["scenarios"][rows].tolist())),
                "sample_ids_sha256": canonical_hash(data["sample_ids"][rows].tolist()),
                "canonical_reuse": key in canonical_ids,
            }
            for key, rows in sorted(populations.items())
        },
        "required_base_population_keys": sorted(required),
        "counts": {
            **plan["counts"],
            "canonical_F_reuse_certified": 5,
            "new_F_to_fit": 14,
            "new_M4_A3_neural_fits": 420,
            "historical_base_reuse_certified": 58,
            "new_base_populations": 125,
            "new_base_estimator_fits": 6068,
        },
        "parity_audit_sha256": file_sha256(PARITY),
        "inputs": inputs,
        "task_outer_metrics_embargoed": True,
        "posture_head_fits": 0,
        "new_fits_at_lock": 0,
    }
    run.mkdir(parents=True, exist_ok=True)
    immutable_json(run / "execution_lock.json", lock)
    validate_lock(run)
    return lock


def validate_lock(run: Path) -> dict[str, Any]:
    lock = read_json(run / "execution_lock.json")
    if lock.get("status") != "GENERALIZED_ARFTR_ANCESTORS_LOCKED_BEFORE_NEW_FITS":
        raise RuntimeError("Generalized ancestor execution lock is invalid")
    for record in lock["inputs"]:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"Generalized ancestor input changed: {record['path']}")
    if file_sha256(PARITY) != lock["parity_audit_sha256"]:
        raise RuntimeError("Generalized F(T) parity receipt changed")
    return lock


def prepare_bases(run: Path, *, workers: int, limit: int | None) -> dict[str, Any]:
    lock = validate_lock(run)
    data = _source_data()
    _plan, populations = _plans(data)
    required = _required_base_populations(data, populations)
    cache = _layered_base(run, data)
    pending = [
        rows
        for _, rows in sorted(required.items())
        if not (cache.directory(rows) / "receipt.json").is_file()
    ]
    selected = pending if limit is None else pending[:limit]
    event(
        "generalized_base_preparation_start",
        required=len(required),
        pending=len(pending),
        selected=len(selected),
        workers=workers,
    )
    started = time.perf_counter()
    completed = 0

    def fit(rows: np.ndarray):
        return rows, cache.prepare(rows)

    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit, rows): rows for rows in selected}
            for future in as_completed(futures):
                rows, receipt = future.result()
                completed += 1
                event(
                    "generalized_base_population_complete",
                    completed=completed,
                    selected=len(selected),
                    rows=len(rows),
                    seconds=receipt["seconds"],
                )
    still_missing = [
        rows
        for rows in required.values()
        if not (cache.directory(rows) / "receipt.json").is_file()
    ]
    result = {
        "status": "GENERALIZED_BASES_PARTIAL" if still_missing else "GENERALIZED_BASES_COMPLETE",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "required": len(required),
        "historical_reused": sum(
            (cache.historical.directory(rows) / "receipt.json").is_file()
            for rows in required.values()
        ),
        "new_complete": sum(
            (cache.current.directory(rows) / "receipt.json").is_file()
            for rows in required.values()
        ),
        "remaining": len(still_missing),
        "completed_this_invocation": completed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not still_missing:
        receipts = []
        for key, rows in sorted(required.items()):
            _, receipt = cache.load(rows)
            path = cache.directory(rows) / "receipt.json"
            receipts.append(
                {
                    "population_key": key,
                    "layer": "historical"
                    if (cache.historical.directory(rows) / "receipt.json").is_file()
                    else "current",
                    "receipt_sha256": file_sha256(path),
                    "estimator_fits": receipt["estimator_fits"],
                }
            )
        if sum(row["estimator_fits"] for row in receipts) != lock["counts"][
            "base_estimators_before_reuse"
        ]:
            raise RuntimeError("Completed generalized base estimator count changed")
        completion = {**result, "receipts": receipts}
        immutable_json(run / "base_completion.json", completion)
    print(json.dumps(result, indent=2))
    return result


def _require_bases(run: Path) -> LayeredNestedBaseCache:
    completion = read_json(run / "base_completion.json")
    if (
        completion.get("status") != "GENERALIZED_BASES_COMPLETE"
        or completion.get("execution_lock_sha256") != file_sha256(
            run / "execution_lock.json"
        )
        or completion.get("required") != 183
        or completion.get("remaining") != 0
    ):
        raise RuntimeError("All generalized base populations are not certified")
    data = _source_data()
    cache = _layered_base(run, data)
    _plan, populations = _plans(data)
    required = _required_base_populations(data, populations)
    for record in completion["receipts"]:
        rows = required[record["population_key"]]
        if file_sha256(cache.directory(rows) / "receipt.json") != record["receipt_sha256"]:
            raise RuntimeError("A completed generalized base receipt changed")
    return cache


def _fit_m4(
    run: Path,
    lock: dict[str, Any],
    data: dict[str, np.ndarray],
    cache: LayeredNestedBaseCache,
    population_id: str,
    population: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    protocol = lock["memory_protocol"]
    directory = run / "populations" / population_id / "m4"
    execution_hash = file_sha256(run / "execution_lock.json")
    splits = group_splits(data["labels"], data["scenarios"], population)
    contexts = [cache.meta_probabilities(rows) for rows, _ in splits]
    configs = [
        (lr, wd)
        for lr in protocol["learning_rates"]
        for wd in protocol["weight_decays"]
    ]
    candidates, candidate_oof, fit_receipts = [], {}, []
    for config, (learning_rate, weight_decay) in enumerate(configs):
        oof = np.full((len(data["labels"]), 3), np.nan)
        epochs = []
        for inner, ((train, held), (probabilities, ancestry)) in enumerate(
            zip(splits, contexts, strict=True)
        ):
            fit_dir = directory / f"config-{config}" / f"inner-{inner}"
            outputs, receipt, _hashes, _difference = memory_driver.fit_new(
                fit_dir,
                data,
                protocol,
                train,
                held,
                learning_rate,
                weight_decay,
                protocol["inner_seed"],
                None,
                probabilities,
                ancestry,
                execution_hash,
                lambda config=config, inner=inner, **values: event(
                    "generalized_m4_epoch",
                    population_id=population_id,
                    config=config,
                    inner=inner,
                    **values,
                ),
            )
            oof[held] = outputs["probabilities"]
            epochs.append(receipt["selected_epoch"])
            fit_receipts.append(_record(fit_dir / "receipt.json"))
        candidate_oof[config] = oof[population].copy()
        candidates.append(
            {
                "config_index": config,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "inner_epochs": epochs,
                "inner_metrics": probability_metrics(
                    data["labels"][population], oof[population]
                ),
            }
        )
    selection = memory_driver.selection_from(candidates)
    immutable_json(directory / "selection.json", selection)
    selected = selection["selected"]
    probabilities, ancestry = cache.meta_probabilities(population)
    outer = []
    for seed in protocol["outer_seeds"]:
        fit_dir = directory / f"refit-seed-{seed}"
        outputs, _receipt, _hashes, _difference = memory_driver.fit_new(
            fit_dir,
            data,
            protocol,
            population,
            prediction,
            selected["learning_rate"],
            selected["weight_decay"],
            seed,
            selection["refit_epochs"],
            probabilities,
            ancestry,
            execution_hash,
            lambda seed=seed, **values: event(
                "generalized_m4_epoch",
                population_id=population_id,
                seed=seed,
                **values,
            ),
        )
        outer.append(outputs["probabilities"])
        fit_receipts.append(_record(fit_dir / "receipt.json"))
    if len(fit_receipts) != 15:
        raise RuntimeError("Generalized M4 did not produce exactly15 fits")
    return candidate_oof[selected["config_index"]], np.stack(outer), fit_receipts


def _fit_a3(
    run: Path,
    lock: dict[str, Any],
    patches: np.ndarray,
    center: np.ndarray,
    data: dict[str, np.ndarray],
    population_id: str,
    context_index: int,
    population: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    protocol = lock["protocol"]
    splits = group_splits(data["labels"], data["scenarios"], population)
    candidates, candidate_oof, fit_receipts = [], {}, []
    configs = [
        (lr, wd)
        for lr in protocol["training"]["learning_rates"]
        for wd in protocol["training"]["weight_decays"]
    ]
    for config, (learning_rate, weight_decay) in enumerate(configs):
        oof = np.full((len(data["labels"]), 3), np.nan)
        epochs = []
        for inner, (train, held) in enumerate(splits):
            predicted, receipt = sear_driver.fit_model(
                run,
                lock,
                patches,
                center,
                data,
                A3_ARM,
                train,
                held,
                learning_rate,
                weight_decay,
                protocol["training"]["inner_seed"],
                lock["device"],
                outer_fold=context_index,
                inner_fold=inner,
            )
            oof[held] = predicted["probabilities"]
            epochs.append(receipt["selected_epoch"])
            fit_receipts.append(
                _record(run / "fits" / canonical_hash(receipt["request"]) / "receipt.json")
            )
        candidate_oof[config] = oof[population].copy()
        candidates.append(
            {
                "config_index": config,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "inner_best_epochs": epochs,
                "inner_metrics": probability_metrics(
                    data["labels"][population], oof[population]
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            -item["inner_metrics"]["macro_f1"],
            item["inner_metrics"]["nll"],
            item["learning_rate"],
            item["weight_decay"],
        ),
    )
    epochs = max(1, int(np.rint(np.median(selected["inner_best_epochs"]))))
    outer = []
    for seed in protocol["training"]["outer_seeds"]:
        predicted, receipt = sear_driver.fit_model(
            run,
            lock,
            patches,
            center,
            data,
            A3_ARM,
            population,
            prediction,
            selected["learning_rate"],
            selected["weight_decay"],
            seed,
            lock["device"],
            outer_fold=context_index,
            epochs=epochs,
        )
        outer.append(predicted["probabilities"])
        fit_receipts.append(
            _record(run / "fits" / canonical_hash(receipt["request"]) / "receipt.json")
        )
    if len(fit_receipts) != 15:
        raise RuntimeError("Generalized A3 did not produce exactly15 fits")
    directory = run / "populations" / population_id / "a3"
    selection = {
        "candidates": candidates,
        "selected": selected,
        "outer_refit_epochs": epochs,
    }
    immutable_json(directory / "selection.json", selection)
    return candidate_oof[selected["config_index"]], np.stack(outer), fit_receipts


def fit_population(run: Path, population_id: str) -> dict[str, Any]:
    lock = validate_lock(run)
    cache = _require_bases(run)
    data = _source_data()
    _plan, populations = _plans(data)
    if population_id not in populations:
        raise ValueError("Unknown fixed F population ID")
    if lock["F_populations"][population_id]["canonical_reuse"]:
        raise ValueError("Canonical F(T) is certified for reuse and must not be refit")
    population = populations[population_id]
    prediction = np.setdiff1d(np.arange(len(data["labels"])), population)
    directory = run / "populations" / population_id
    receipt_path = directory / "receipt.json"
    output_path = directory / "predictions.npz"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") != "GENERALIZED_F_POPULATION_COMPLETE"
            or receipt.get("execution_lock_sha256")
            != file_sha256(run / "execution_lock.json")
            or file_sha256(output_path) != receipt.get("predictions_sha256")
        ):
            raise RuntimeError("Completed generalized F population changed")
        return receipt
    sear_lock, patches, center, sear_data = sear_driver.load_locked(SEAR)
    del sear_lock
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(sear_data[name], data[name]):
            raise RuntimeError("A3 and M4 population identities differ")
    context_index = sorted(populations).index(population_id)
    started = time.perf_counter()
    m4_inner, m4_outer, m4_receipts = _fit_m4(
        run, lock, data, cache, population_id, population, prediction
    )
    a3_inner, a3_outer, a3_receipts = _fit_a3(
        run,
        lock,
        patches,
        center,
        sear_data,
        population_id,
        context_index,
        population,
        prediction,
    )
    p6_all, p6_ancestry = cache.meta_probabilities(population)
    composed = execute_population(
        data,
        population,
        prediction,
        PopulationInputs(
            m4_inner=m4_inner,
            p6_inner=p6_all[population],
            a3_inner=a3_inner,
            m4_outer=m4_outer,
            p6_outer=cache.held_predictions(population, prediction),
            a3_outer=a3_outer,
        ),
        read_json(ARFTR_PROTOCOL),
    )
    _immutable_npz(
        output_path,
        population_rows=population,
        prediction_rows=prediction,
        sample_ids=data["sample_ids"][prediction],
        parameters=composed.parameters.as_array(),
        seed_probabilities=composed.seed_probabilities,
        mean_probabilities=composed.mean_probabilities,
        m4_inner=m4_inner,
        a3_inner=a3_inner,
        p6_inner=p6_all[population],
        m4_outer=m4_outer,
        a3_outer=a3_outer,
        p6_outer=cache.held_predictions(population, prediction),
        inner_neighbor_map=composed.inner_neighbor_map,
        outer_neighbor_map=composed.outer_neighbor_map,
    )
    receipt = {
        "status": "GENERALIZED_F_POPULATION_COMPLETE",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "population_id": population_id,
        "training_scenarios": list(composed.training_scenarios),
        "prediction_scenarios": list(composed.prediction_scenarios),
        "training_rows": len(population),
        "prediction_rows": len(prediction),
        "selected_parameters": composed.parameters.as_array().tolist(),
        "training_metrics": composed.training_metrics,
        "outer_prediction_labels_read": 0,
        "m4_fit_receipts": m4_receipts,
        "a3_fit_receipts": a3_receipts,
        "p6_ancestry": p6_ancestry,
        "neural_fits": 30,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions_sha256": file_sha256(output_path),
    }
    immutable_json(receipt_path, receipt)
    event(
        "generalized_F_population_complete",
        population_id=population_id,
        seconds=receipt["elapsed_seconds"],
    )
    return receipt


def status(run: Path) -> dict[str, Any]:
    lock = validate_lock(run)
    complete = []
    for population_id, spec in lock["F_populations"].items():
        if spec["canonical_reuse"]:
            complete.append({"population_id": population_id, "status": "canonical_reuse"})
        elif (run / "populations" / population_id / "receipt.json").is_file():
            complete.append({"population_id": population_id, "status": "new_complete"})
    result = {
        "status": "GENERALIZED_ANCESTOR_STATUS",
        "F_complete_or_reused": len(complete),
        "F_required": 19,
        "F_remaining": 19 - len(complete),
        "populations": complete,
        "base_completion": read_json(run / "base_completion.json")
        if (run / "base_completion.json").is_file()
        else None,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--stage", choices=("prepare", "bases", "fit-one", "status"), required=True
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--population-id")
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    if args.stage == "prepare":
        lock = prepare(run)
        print(
            json.dumps(
                {
                    "status": lock["status"],
                    "new_base_populations": lock["counts"]["new_base_populations"],
                    "new_F_to_fit": lock["counts"]["new_F_to_fit"],
                },
                indent=2,
            )
        )
    elif args.stage == "bases":
        if not 1 <= args.workers <= 8 or args.limit is not None and args.limit < 1:
            raise ValueError("workers must be1..8 and limit must be positive")
        prepare_bases(run, workers=args.workers, limit=args.limit)
    elif args.stage == "fit-one":
        if not args.population_id:
            raise ValueError("fit-one requires --population-id")
        receipt = fit_population(run, args.population_id)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "population_id": receipt["population_id"],
                    "elapsed_seconds": receipt["elapsed_seconds"],
                },
                indent=2,
            )
        )
    else:
        status(run)


if __name__ == "__main__":
    main()
