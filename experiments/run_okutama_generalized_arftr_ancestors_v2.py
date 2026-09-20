# ruff: noqa: E402

"""Run generalized ARFTR ancestors with a strict outer-label embargo.

Version 1 safely completed the 125 label-blind P6 base populations, but its M4
outer-refit helper calculated a terminal prediction-set metric.  No generalized
F population was started.  This corrected runner reuses only the immutable v1
base cache and replaces the M4 outer refit with a helper that removes all
prediction-population supervision before constructing training batches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import run_okutama_generalized_arftr_ancestors as legacy
import run_okutama_source_swap_memory as memory_driver

from hac.actor_memory_base import (
    NestedBaseCache,
    canonical_hash,
    file_sha256,
    group_splits,
    probability_metrics,
)
from hac.label_embargoed_memory import fit_outer_label_embargoed
from hac.layered_base_cache import LayeredNestedBaseCache
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[1]
V1_BASE_RUN = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v1"
DEFAULT_RUN = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v2"
LOCK_STATUS = "GENERALIZED_ARFTR_ANCESTORS_V2_LABEL_EMBARGO_LOCK"


def _layered_base(_run: Path, data: dict[str, np.ndarray]) -> LayeredNestedBaseCache:
    historical = legacy._historical_base(data)
    prepared_v1 = NestedBaseCache(
        V1_BASE_RUN / "base_cache",
        historical.features,
        historical.labels,
        historical.scenarios,
        historical.sample_ids,
        historical.config,
        historical.identity["feature_sha256"],
    )
    return LayeredNestedBaseCache(historical, prepared_v1)


def _v1_base_gate() -> dict[str, Any]:
    lock_path = V1_BASE_RUN / "execution_lock.json"
    completion_path = V1_BASE_RUN / "base_completion.json"
    lock = read_json(lock_path)
    completion = read_json(completion_path)
    population_receipts = list((V1_BASE_RUN / "populations").glob("*/receipt.json"))
    if (
        lock.get("status") != "GENERALIZED_ARFTR_ANCESTORS_LOCKED_BEFORE_NEW_FITS"
        or completion.get("status") != "GENERALIZED_BASES_COMPLETE"
        or completion.get("execution_lock_sha256") != file_sha256(lock_path)
        or completion.get("required") != 183
        or completion.get("historical_reused") != 58
        or completion.get("new_complete") != 125
        or completion.get("remaining") != 0
        or population_receipts
    ):
        raise RuntimeError(
            "V1 may be reused only as a complete base cache with zero generalized F fits"
        )
    return completion


def prepare(run: Path) -> dict[str, Any]:
    completion = _v1_base_gate()
    parity = read_json(legacy.PARITY)
    if (
        parity.get("status")
        != "GENERALIZED_ARFTR_FT_PARITY_EXACT_AND_REUSE_AUDIT_PASS"
        or not parity.get("final_mean_probabilities_exact")
        or parity.get("final_maximum_absolute_difference") != 0.0
    ):
        raise RuntimeError("Exact generalized F(T) parity has not passed")
    data = legacy._source_data()
    plan, populations = legacy._plans(data)
    required = legacy._required_base_populations(data, populations)
    sear_lock, _patches, _center, sear_data = legacy.sear_driver.load_locked(
        legacy.SEAR
    )
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(sear_data[name], data[name]):
            raise RuntimeError("M4 and A3 generalized cohorts differ")
    canonical_ids = legacy._canonical_population_ids(data)
    if len(canonical_ids) != 5 or not canonical_ids <= set(populations):
        raise RuntimeError("Canonical reusable F identities changed")
    extra_inputs = {
        Path(__file__),
        ROOT / "experiments/audit_okutama_generalized_arftr_ancestors_v2.py",
        ROOT / "src/hac/label_embargoed_memory.py",
        ROOT / "src/hac/source_swap_execution.py",
        V1_BASE_RUN / "execution_lock.json",
        V1_BASE_RUN / "base_completion.json",
    }
    inputs = [_record(path) for path in sorted({*legacy._locked_inputs(), *extra_inputs}, key=str)]
    lock = {
        "status": LOCK_STATUS,
        "correction": "M4 outer refits remove prediction labels and boundary targets",
        "rejected_v1_F_populations": 0,
        "base_reuse_source": _record(V1_BASE_RUN / "base_completion.json"),
        "rows": len(data["labels"]),
        "protocol": sear_lock["protocol"],
        "parameter_counts": sear_lock["parameter_counts"],
        "device": sear_lock["device"],
        "memory_protocol": read_json(legacy.SOURCE / "base_execution_lock.json")[
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
            "v1_prepared_base_reuse_certified": 125,
            "new_base_populations": 0,
            "new_base_estimator_fits": 0,
            "base_estimators_before_reuse": sum(
                item["estimator_fits"] for item in completion["receipts"]
            ),
        },
        "parity_audit_sha256": file_sha256(legacy.PARITY),
        "inputs": inputs,
        "task_outer_metrics_embargoed": True,
        "prediction_population_labels_available_to_outer_refits": False,
        "posture_head_fits": 0,
        "new_fits_at_lock": 0,
    }
    run.mkdir(parents=True, exist_ok=True)
    immutable_json(run / "execution_lock.json", lock)
    validate_lock(run)
    return lock


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def validate_lock(run: Path) -> dict[str, Any]:
    lock = read_json(run / "execution_lock.json")
    if (
        lock.get("status") != LOCK_STATUS
        or lock.get("prediction_population_labels_available_to_outer_refits") is not False
        or lock.get("rejected_v1_F_populations") != 0
    ):
        raise RuntimeError("Corrected generalized ancestor execution lock is invalid")
    _v1_base_gate()
    for record in lock["inputs"]:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked v2 ancestor input changed: {record['path']}")
    return lock


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
        (learning_rate, weight_decay)
        for learning_rate in protocol["learning_rates"]
        for weight_decay in protocol["weight_decays"]
    ]
    candidates: list[dict[str, Any]] = []
    candidate_oof: dict[int, np.ndarray] = {}
    fit_receipts: list[dict[str, Any]] = []
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
                lambda config=config, inner=inner, **values: legacy.event(
                    "generalized_m4_inner_epoch",
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
        outputs, receipt = fit_outer_label_embargoed(
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
            lambda seed=seed, **values: legacy.event(
                "generalized_m4_embargoed_outer_epoch",
                population_id=population_id,
                seed=seed,
                **values,
            ),
            device=lock["device"],
        )
        if (
            receipt.get("outer_prediction_labels_read") != 0
            or "held_metrics" in receipt
        ):
            raise RuntimeError("M4 outer prediction-label embargo failed")
        outer.append(outputs["probabilities"])
        fit_receipts.append(_record(fit_dir / "receipt.json"))
    if len(fit_receipts) != 15:
        raise RuntimeError("Corrected generalized M4 did not produce exactly 15 fits")
    return candidate_oof[selected["config_index"]], np.stack(outer), fit_receipts


legacy.validate_lock = validate_lock
legacy._layered_base = _layered_base
legacy._fit_m4 = _fit_m4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--stage", choices=("prepare", "bases", "fit-one", "status"), required=True
    )
    parser.add_argument("--workers", type=int, default=1)
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
                    "new_F_to_fit": lock["counts"]["new_F_to_fit"],
                    "new_base_populations": lock["counts"]["new_base_populations"],
                },
                indent=2,
            )
        )
    elif args.stage == "bases":
        legacy.prepare_bases(run, workers=args.workers, limit=None)
    elif args.stage == "fit-one":
        if not args.population_id:
            raise ValueError("fit-one requires --population-id")
        receipt = legacy.fit_population(run, args.population_id)
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
        legacy.status(run)


if __name__ == "__main__":
    main()
