# ruff: noqa: E402

"""Independently replay and audit corrected generalized ARFTR ancestors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import run_okutama_generalized_arftr_ancestors_v2 as runner

from hac.actor_memory_base import canonical_hash, file_sha256, group_splits
from hac.generalized_arftr import PopulationInputs, execute_population
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[1]


def _checked_record(record: dict[str, Any]) -> Path:
    path = ROOT / record["path"]
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or file_sha256(path) != record["sha256"]
    ):
        raise RuntimeError(f"Generalized fit record changed: {record.get('path')}")
    return path


def _probabilities(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if (
        value.shape != shape
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or not np.allclose(value.sum(-1), 1, atol=1e-6, rtol=0)
    ):
        raise RuntimeError(f"Malformed audited {name} probabilities")


def _audit_m4_receipts(
    records: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    population: np.ndarray,
    prediction: np.ndarray,
    execution_hash: str,
) -> dict[str, Any]:
    if len(records) != 15:
        raise RuntimeError("M4 fit receipt inventory is not exactly 15")
    splits = group_splits(data["labels"], data["scenarios"], population)
    inner_expected = [pair for _config in range(4) for pair in splits]
    inner_label_reads = 0
    outer_label_reads = 0
    paths = []
    for index, record in enumerate(records):
        path = _checked_record(record)
        receipt = read_json(path)
        paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        if index < 12:
            train, held = inner_expected[index]
            if (
                receipt.get("train_rows") != train.tolist()
                or receipt.get("held_rows") != held.tolist()
                or receipt.get("request", {}).get("epochs") is not None
                or receipt.get("request", {}).get("selection_labels_sha256")
                != canonical_hash(data["labels"][held].tolist())
            ):
                raise RuntimeError("M4 inner fit ancestry changed")
            inner_label_reads += len(held)
            guard = read_json(path.with_name("request_guard.json"))
            if guard.get("source_swap_execution_sha256") != execution_hash:
                raise RuntimeError("M4 inner request is not bound to this execution")
        else:
            if (
                receipt.get("status") != "LABEL_EMBARGOED_MEMORY_REFIT_COMPLETE"
                or receipt.get("train_rows") != population.tolist()
                or receipt.get("held_rows") != prediction.tolist()
                or receipt.get("outer_prediction_labels_read") != 0
                or receipt.get("outer_prediction_boundary_targets_read") != 0
                or receipt.get("request", {}).get("held_labels_sha256") is not None
                or receipt.get("request", {}).get("selection_labels_sha256") is not None
                or receipt.get("request", {}).get("execution_lock_sha256")
                != execution_hash
                or "held_metrics" in receipt
            ):
                raise RuntimeError("M4 outer refit violated the prediction-label embargo")
            outer_label_reads += receipt["outer_prediction_labels_read"]
        for name, digest in receipt.get("output_sha256", {}).items():
            if file_sha256(path.with_name(name)) != digest:
                raise RuntimeError("M4 fit output hash changed")
    return {
        "fits": len(records),
        "inner_selection_label_rows_read": inner_label_reads,
        "outer_prediction_label_rows_read": outer_label_reads,
        "paths_sha256": canonical_hash(paths),
    }


def _audit_a3_receipts(
    records: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    population: np.ndarray,
    prediction: np.ndarray,
    execution_hash: str,
) -> dict[str, Any]:
    if len(records) != 15:
        raise RuntimeError("A3 fit receipt inventory is not exactly 15")
    splits = group_splits(data["labels"], data["scenarios"], population)
    inner_expected = [pair for _config in range(4) for pair in splits]
    inner_label_reads = 0
    paths = []
    for index, record in enumerate(records):
        path = _checked_record(record)
        receipt = read_json(path)
        request = receipt.get("request", {})
        paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        if receipt.get("status") != "SEAR_FIT_COMPLETE" or request.get(
            "execution_lock_sha256"
        ) != execution_hash:
            raise RuntimeError("A3 fit is not bound to the corrected execution")
        if index < 12:
            train, held = inner_expected[index]
            if (
                request.get("stage") != "inner"
                or request.get("train_ids_sha256")
                != canonical_hash(data["sample_ids"][train].tolist())
                or request.get("held_ids_sha256")
                != canonical_hash(data["sample_ids"][held].tolist())
                or request.get("selection_labels_sha256")
                != canonical_hash(data["labels"][held].tolist())
                or "held_metrics" not in receipt
            ):
                raise RuntimeError("A3 inner fit ancestry changed")
            inner_label_reads += len(held)
        elif (
            request.get("stage") != "outer_refit"
            or request.get("train_ids_sha256")
            != canonical_hash(data["sample_ids"][population].tolist())
            or request.get("held_ids_sha256")
            != canonical_hash(data["sample_ids"][prediction].tolist())
            or request.get("selection_labels_sha256") is not None
            or receipt.get("outer_held_metrics_embargoed") is not True
            or "held_metrics" in receipt
        ):
            raise RuntimeError("A3 outer fit violated its prediction-label embargo")
        for artifact in receipt.get("artifacts", {}).values():
            artifact_path = Path(artifact["path"])
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != artifact["size_bytes"]
                or file_sha256(artifact_path) != artifact["sha256"]
            ):
                raise RuntimeError("A3 fit artifact changed")
    return {
        "fits": len(records),
        "inner_selection_label_rows_read": inner_label_reads,
        "outer_prediction_label_rows_read": 0,
        "paths_sha256": canonical_hash(paths),
    }


def audit_population(run: Path, population_id: str) -> dict[str, Any]:
    lock = runner.validate_lock(run)
    data = runner.legacy._source_data()
    _plan, populations = runner.legacy._plans(data)
    if population_id not in populations or lock["F_populations"][population_id][
        "canonical_reuse"
    ]:
        raise ValueError("Audit requires one of the 14 noncanonical F populations")
    population = populations[population_id]
    prediction = np.setdiff1d(np.arange(len(data["labels"])), population)
    directory = run / "populations" / population_id
    receipt_path = directory / "receipt.json"
    prediction_path = directory / "predictions.npz"
    receipt = read_json(receipt_path)
    execution_hash = file_sha256(run / "execution_lock.json")
    if (
        receipt.get("status") != "GENERALIZED_F_POPULATION_COMPLETE"
        or receipt.get("execution_lock_sha256") != execution_hash
        or receipt.get("population_id") != population_id
        or receipt.get("outer_prediction_labels_read") != 0
        or receipt.get("neural_fits") != 30
        or receipt.get("predictions_sha256") != file_sha256(prediction_path)
    ):
        raise RuntimeError("Generalized F population receipt changed")
    with np.load(prediction_path, allow_pickle=False) as saved_file:
        saved = {name: saved_file[name] for name in saved_file.files}
    required = {
        "population_rows",
        "prediction_rows",
        "sample_ids",
        "parameters",
        "seed_probabilities",
        "mean_probabilities",
        "m4_inner",
        "a3_inner",
        "p6_inner",
        "m4_outer",
        "a3_outer",
        "p6_outer",
        "inner_neighbor_map",
        "outer_neighbor_map",
    }
    if (
        set(saved) != required
        or not np.array_equal(saved["population_rows"], population)
        or not np.array_equal(saved["prediction_rows"], prediction)
        or not np.array_equal(saved["sample_ids"], data["sample_ids"][prediction])
        or receipt.get("training_rows") != len(population)
        or receipt.get("prediction_rows") != len(prediction)
        or lock["F_populations"][population_id]["sample_ids_sha256"]
        != canonical_hash(data["sample_ids"][population].tolist())
    ):
        raise RuntimeError("Generalized F population identity changed")
    n_train, n_prediction = len(population), len(prediction)
    _probabilities(saved["m4_inner"], (n_train, 3), "M4 inner")
    _probabilities(saved["a3_inner"], (n_train, 3), "A3 inner")
    _probabilities(saved["p6_inner"], (n_train, 3), "P6 inner")
    _probabilities(saved["m4_outer"], (3, n_prediction, 3), "M4 outer")
    _probabilities(saved["a3_outer"], (3, n_prediction, 3), "A3 outer")
    _probabilities(saved["p6_outer"], (n_prediction, 3), "P6 outer")
    _probabilities(saved["seed_probabilities"], (3, n_prediction, 3), "F seeds")
    _probabilities(saved["mean_probabilities"], (n_prediction, 3), "F mean")
    cache = runner.legacy._require_bases(run)
    p6_all, _ancestry = cache.meta_probabilities(population)
    p6_inner_exact = np.array_equal(saved["p6_inner"], p6_all[population])
    p6_outer_exact = np.array_equal(
        saved["p6_outer"], cache.held_predictions(population, prediction)
    )
    inputs = PopulationInputs(
        m4_inner=saved["m4_inner"],
        p6_inner=saved["p6_inner"],
        a3_inner=saved["a3_inner"],
        m4_outer=saved["m4_outer"],
        p6_outer=saved["p6_outer"],
        a3_outer=saved["a3_outer"],
    )
    masked = dict(data)
    masked_labels = np.full_like(data["labels"], -1)
    masked_labels[population] = data["labels"][population]
    masked["labels"] = masked_labels
    replay = execute_population(
        masked,
        population,
        prediction,
        inputs,
        read_json(runner.legacy.ARFTR_PROTOCOL),
    )
    exact = {
        "parameters": np.array_equal(saved["parameters"], replay.parameters.as_array()),
        "seed_probabilities": np.array_equal(
            saved["seed_probabilities"], replay.seed_probabilities
        ),
        "mean_probabilities": np.array_equal(
            saved["mean_probabilities"], replay.mean_probabilities
        ),
        "inner_neighbor_map": np.array_equal(
            saved["inner_neighbor_map"], replay.inner_neighbor_map
        ),
        "outer_neighbor_map": np.array_equal(
            saved["outer_neighbor_map"], replay.outer_neighbor_map
        ),
        "p6_inner": p6_inner_exact,
        "p6_outer": p6_outer_exact,
    }
    if (
        not all(exact.values())
        or receipt.get("selected_parameters") != replay.parameters.as_array().tolist()
        or receipt.get("training_metrics") != replay.training_metrics
        or receipt.get("training_scenarios") != list(replay.training_scenarios)
        or receipt.get("prediction_scenarios") != list(replay.prediction_scenarios)
        or population_id != replay.population_id
    ):
        raise RuntimeError("Independent generalized F recomposition is not exact")
    m4 = _audit_m4_receipts(
        receipt["m4_fit_receipts"], data, population, prediction, execution_hash
    )
    a3 = _audit_a3_receipts(
        receipt["a3_fit_receipts"], data, population, prediction, execution_hash
    )
    result = {
        "status": "GENERALIZED_F_POPULATION_V2_INDEPENDENT_AUDIT_PASS",
        "population_id": population_id,
        "execution_lock_sha256": execution_hash,
        "prediction_artifact_sha256": file_sha256(prediction_path),
        "auditor_sha256": file_sha256(Path(__file__)),
        "training_rows": n_train,
        "prediction_rows": n_prediction,
        "prediction_scenario_overlap": 0,
        "prediction_label_counterfactual": "all_prediction_labels_replaced_by_minus_one",
        "prediction_label_counterfactual_invariance": True,
        "outer_prediction_label_rows_read_during_fits": 0,
        "exact_recomposition": exact,
        "maximum_probability_difference": 0.0,
        "m4": m4,
        "a3": a3,
    }
    immutable_json(directory / "independent_audit.json", result)
    return result


def audit_all(run: Path) -> dict[str, Any]:
    lock = runner.validate_lock(run)
    population_ids = [
        population_id
        for population_id, spec in lock["F_populations"].items()
        if not spec["canonical_reuse"]
    ]
    results = [audit_population(run, population_id) for population_id in population_ids]
    if len(results) != 14:
        raise RuntimeError("Corrected generalized F audit did not cover 14 populations")
    result = {
        "status": "GENERALIZED_ARFTR_ANCESTORS_V2_INDEPENDENT_AUDIT_PASS",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "auditor_sha256": file_sha256(Path(__file__)),
        "canonical_F_reused_after_exact_parity": 5,
        "new_F_populations_replayed": len(results),
        "new_neural_fits_audited": sum(
            item["m4"]["fits"] + item["a3"]["fits"] for item in results
        ),
        "outer_prediction_label_rows_read_during_fits": 0,
        "all_recompositions_exact": all(
            all(item["exact_recomposition"].values()) for item in results
        ),
        "all_prediction_label_counterfactual_invariant": all(
            item["prediction_label_counterfactual_invariance"] for item in results
        ),
        "population_audits": [
            {
                "population_id": item["population_id"],
                "path": str(
                    (
                        run
                        / "populations"
                        / item["population_id"]
                        / "independent_audit.json"
                    ).relative_to(ROOT)
                ).replace("\\", "/"),
                "sha256": file_sha256(
                    run
                    / "populations"
                    / item["population_id"]
                    / "independent_audit.json"
                ),
            }
            for item in results
        ],
    }
    immutable_json(run / "independent_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    parser.add_argument("--population-id")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    if args.all == bool(args.population_id):
        raise ValueError("Choose exactly one of --population-id or --all")
    result = audit_all(run) if args.all else audit_population(run, args.population_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
