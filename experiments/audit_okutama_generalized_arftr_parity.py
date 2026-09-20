"""Certify canonical F(T) parity and quantify reusable F(S) dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_okutama_evidence_memory as evidence

from hac.actor_memory_base import canonical_hash, file_sha256, group_splits
from hac.generalized_arftr import PopulationInputs, execute_population
from hac.matr_artifacts import load_cached_study_data, selected_fold_artifacts
from hac.nested_arftr_plan import fixed_recipe_plan

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1"
ARFTR_PROTOCOL = ROOT / "experiments/okutama_arftr_protocol.json"
DESIGN_PROTOCOL = ROOT / "experiments/okutama_source_posture_protocol.json"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/generalized_arftr_parity_v1/audit.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_cache(data: dict[str, np.ndarray]):
    base_lock = _read_json(SOURCE / "base_execution_lock.json")
    raw = evidence.load_data(SOURCE)
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(raw[name], data[name]):
            raise RuntimeError("Generalized P6 cohort differs from retained ARFTR")
    return evidence.base_cache(SOURCE, raw, base_lock["original_learning_protocol"])


def _required_base_populations(data: dict[str, np.ndarray], plan: dict[str, Any]):
    populations: dict[str, np.ndarray] = {}

    def add(rows: np.ndarray) -> None:
        populations[canonical_hash(rows.tolist())] = rows

    for receipt in plan["F_populations"]:
        population = np.flatnonzero(np.isin(data["scenarios"], receipt["scenarios"]))
        add(population)
        for inner_train, _ in group_splits(data["labels"], data["scenarios"], population):
            add(inner_train)
            for base_train, _ in group_splits(
                data["labels"], data["scenarios"], inner_train
            ):
                add(base_train)
    return populations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT.resolve())
    if output.exists():
        raise RuntimeError("Parity audit output already exists")
    protocol = _read_json(ARFTR_PROTOCOL)
    design = _read_json(DESIGN_PROTOCOL)
    data = load_cached_study_data(SOURCE)
    cache = _base_cache(data)
    fold_receipts = []
    reconstructed = np.full((len(data["labels"]), 3), np.nan)
    for fold in range(5):
        cached = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold)
        train, held = cached.train_rows, cached.held_rows
        p6_all, p6_ancestry = cache.meta_probabilities(train)
        result = execute_population(
            data,
            train,
            held,
            PopulationInputs(
                m4_inner=cached.m4_inner["probabilities"],
                p6_inner=p6_all[train],
                a3_inner=cached.a3_inner["probabilities"],
                m4_outer=cached.m4_outer["probabilities"],
                p6_outer=cache.held_predictions(train, held),
                a3_outer=cached.a3_outer["probabilities"],
            ),
            protocol,
        )
        retained_receipt = _read_json(ARFTR / f"fold-{fold}/receipt.json")
        with np.load(ARFTR / f"fold-{fold}/checkpoint.npz", allow_pickle=False) as saved:
            checkpoint = {name: saved[name] for name in saved.files}
        with np.load(ARFTR / f"fold-{fold}/predictions.npz", allow_pickle=False) as saved:
            retained_seed = saved["seed_probabilities"][5]
            if not np.array_equal(saved["held_rows"], held):
                raise RuntimeError("Retained ARFTR held rows changed")
        parameter_exact = np.array_equal(
            result.parameters.as_array(), checkpoint["parameters"]
        )
        seed_exact = np.array_equal(result.seed_probabilities, retained_seed)
        inner_neighbor_exact = np.array_equal(
            result.inner_neighbor_map, checkpoint["inner_neighbor_map"]
        )
        outer_neighbor_exact = np.array_equal(
            result.outer_neighbor_map, checkpoint["outer_neighbor_map"]
        )
        metrics_exact = result.training_metrics == retained_receipt["training_metrics"]
        if not all(
            (
                parameter_exact,
                seed_exact,
                inner_neighbor_exact,
                outer_neighbor_exact,
                metrics_exact,
            )
        ):
            raise RuntimeError(f"Generalized F(T) does not replay retained fold {fold}")
        reconstructed[held] = result.mean_probabilities
        fold_receipts.append(
            {
                "fold": fold,
                "population_id": result.population_id,
                "train_rows": len(train),
                "held_rows": len(held),
                "selected_parameters": result.parameters.as_array().tolist(),
                "parameter_exact": parameter_exact,
                "training_metrics_exact": metrics_exact,
                "inner_neighbor_exact": inner_neighbor_exact,
                "outer_neighbor_exact": outer_neighbor_exact,
                "seed_probabilities_exact": seed_exact,
                "maximum_seed_absolute_difference": float(
                    np.max(np.abs(result.seed_probabilities - retained_seed))
                ),
                "p6_ancestry_entries": len(p6_ancestry),
            }
        )
    with np.load(ARFTR / "results/v0001/oof_probabilities.npz", allow_pickle=False) as saved:
        retained = saved["mean_probabilities"][5]
        final_exact = np.array_equal(reconstructed, retained)
        final_difference = float(np.max(np.abs(reconstructed - retained)))
    if not final_exact:
        raise RuntimeError("Generalized F(T) reconstruction differs from retained ARFTR")
    plan = fixed_recipe_plan(
        data["labels"],
        data["scenarios"],
        data["folds"],
        data["sample_ids"],
        enforce_canonical_counts=True,
    )
    required = _required_base_populations(data, plan)
    if len(required) != plan["counts"]["unique_base_training_populations"]:
        raise RuntimeError("Required base population enumeration changed")
    reused, missing = [], []
    for key, rows in sorted(required.items()):
        directory = cache.directory(rows)
        if (directory / "receipt.json").is_file():
            _, receipt = cache.load(rows)
            reused.append(
                {
                    "population_key": key,
                    "rows": len(rows),
                    "scenarios": sorted(set(data["scenarios"][rows].tolist())),
                    "estimator_fits": receipt["estimator_fits"],
                    "receipt_sha256": file_sha256(directory / "receipt.json"),
                }
            )
        else:
            groups = len(set(data["scenarios"][rows].tolist()))
            missing.append(
                {
                    "population_key": key,
                    "rows": len(rows),
                    "scenarios": sorted(set(data["scenarios"][rows].tolist())),
                    "estimator_fits": 4 * (4 * min(3, groups) + 1),
                }
            )
    canonical_f_ids = {row["population_id"] for row in fold_receipts}
    all_f_ids = {receipt["id"] for receipt in plan["F_populations"]}
    if not canonical_f_ids <= all_f_ids or len(canonical_f_ids) != 5:
        raise RuntimeError("Canonical reusable F population identity changed")
    result = {
        "status": "GENERALIZED_ARFTR_FT_PARITY_EXACT_AND_REUSE_AUDIT_PASS",
        "rows": len(data["labels"]),
        "folds": fold_receipts,
        "final_mean_probabilities_exact": final_exact,
        "final_maximum_absolute_difference": final_difference,
        "retained_arftr_sha256": _sha256(
            ARFTR / "results/v0001/oof_probabilities.npz"
        ),
        "design_authority_sha256": design["design_authority"]["sha256"],
        "protocol_sha256": _sha256(ARFTR_PROTOCOL),
        "generalized_code_sha256": _sha256(
            ROOT / "src/hac/generalized_arftr.py"
        ),
        "reuse": {
            "F_populations_required": len(all_f_ids),
            "canonical_F_populations_certified": len(canonical_f_ids),
            "new_F_populations_requiring_M4_A3_fits": len(all_f_ids - canonical_f_ids),
            "new_M4_A3_neural_fits": 30 * len(all_f_ids - canonical_f_ids),
            "base_populations_required": len(required),
            "base_populations_certified_reusable": len(reused),
            "base_populations_missing": len(missing),
            "missing_base_estimator_fits": sum(row["estimator_fits"] for row in missing),
        },
        "reused_base_populations": reused,
        "missing_base_populations": missing,
        "outer_held_labels_used_for_selection": 0,
        "new_model_fits": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "final_exact": final_exact,
                "reuse": result["reuse"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
