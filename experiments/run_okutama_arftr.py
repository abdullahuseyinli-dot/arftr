"""Run the immutable adaptive cached ARFTR trial with outer metrics embargoed."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_evidence_memory as evidence

from hac.actor_memory_base import canonical_hash, file_sha256, group_splits, probability_metrics
from hac.arftr import ARFTRParameters, apply_arftr, exact_track_neighbors, shuffled_neighbors
from hac.matr_artifacts import load_cached_study_data, selected_fold_artifacts, selected_input_paths
from hac.source_swap_data import immutable_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
PROTOCOL = ROOT / "experiments/okutama_arftr_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/arftr_v1"
ARMS = (
    "r0_exact_m4",
    "r1_p6_restoration_only",
    "r2_a3_motion_only",
    "r3_temporal_only",
    "r4_residual_no_temporal",
    "r5_arftr_full",
    "r6_shuffled_neighbor_control",
)
SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _validate_protocol(protocol: dict[str, Any]) -> None:
    selection = protocol["selection"]
    expected = {
        "posture_restoration": [0.0, 0.1, 0.2, 0.3, 0.4],
        "motion_restoration": [0.1, 0.2, 0.3, 0.4, 0.5],
        "a3_motion_residual": [0.0, 0.1, 0.2, 0.3],
        "temporal_strength": [0.1, 0.2, 0.3],
    }
    if (
        protocol["status"] != "adaptive_internal_development_not_independent_confirmation"
        or tuple(protocol["arms"]) != ARMS
        or protocol["primary_arm"] != ARMS[5]
        or tuple(protocol["population"]["outer_folds"]) != tuple(range(5))
        or tuple(protocol["population"]["prediction_seeds"]) != SEEDS
        or any(selection[name] != values for name, values in expected.items())
        or selection["candidates_per_outer_fold"] != 300
        or protocol["factorization"]["epsilon"] != 1e-9
        or protocol["control_contract"]["shuffle_seed"] != 20260912
        or protocol["statistics"]["paired_scenario_bootstrap_resamples"] != 10000
        or protocol["statistics"]["exact_scenario_swap_assignments"] != 2048
        or protocol["statistics"]["one_sided_significance_alpha"] != 0.05
    ):
        raise RuntimeError("Frozen ARFTR protocol contract changed")


def _p6_cache(data: dict[str, np.ndarray]):
    base_lock = _read_json(SOURCE / "base_execution_lock.json")
    raw = evidence.load_data(SOURCE)
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(raw[name], data[name]):
            raise RuntimeError(f"P6/source-swap identity changed: {name}")
    cache = evidence.base_cache(SOURCE, raw, base_lock["original_learning_protocol"])
    preparation = _read_json(SOURCE / "base_preparation.json")
    if (
        preparation.get("populations") != 58
        or preparation.get("estimator_fits") != 3016
        or preparation.get("base_identity") != cache.identity
        or preparation.get("source_swap_base_lock_sha256") != file_sha256(
            SOURCE / "base_execution_lock.json"
        )
    ):
        raise RuntimeError("Nested new-source P6 preparation ancestry changed")
    return cache


def _p6_input_paths(data: dict[str, np.ndarray], cache) -> tuple[Path, ...]:
    paths: set[Path] = {
        SOURCE / "base_execution_lock.json",
        SOURCE / "base_preparation.json",
        SOURCE / "data/base_features.npz",
        ROOT / "experiments/run_okutama_evidence_memory.py",
        ROOT / "src/hac/actor_memory_base.py",
        ROOT / "src/hac/video_fusion.py",
    }
    populations: dict[str, np.ndarray] = {}
    for fold in range(5):
        outer_train = np.flatnonzero(data["folds"] != fold)
        populations[canonical_hash(outer_train.tolist())] = outer_train
        for inner_train, _ in group_splits(data["labels"], data["scenarios"], outer_train):
            populations[canonical_hash(inner_train.tolist())] = inner_train
    for population in populations.values():
        cache.load(population)
        directory = cache.directory(population)
        paths.update(
            (directory / "receipt.json", directory / "predictions.npz", directory / "checkpoint.npz")
        )
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing nested P6 input: {path}")
        path.resolve().relative_to(ROOT.resolve())
    return tuple(sorted(paths, key=lambda value: str(value.resolve())))


def prepare(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    protocol = _read_json(PROTOCOL)
    _validate_protocol(protocol)
    data = load_cached_study_data(SOURCE)
    # These calls validate every selected M4/A3 receipt, population and byte hash.
    for fold in range(5):
        selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    cache = _p6_cache(data)
    inputs = set(selected_input_paths(ROOT, SOURCE, SEAR, data, seeds=SEEDS))
    inputs.update(_p6_input_paths(data, cache))
    inputs.update(
        {
            PROTOCOL,
            Path(__file__),
            ROOT / "experiments/audit_okutama_arftr.py",
            ROOT / "src/hac/arftr.py",
            ROOT / "src/hac/matr_artifacts.py",
        }
    )
    lock = {
        "status": "ARFTR_EXECUTION_LOCKED_ADAPTIVE_OUTER_METRICS_EMBARGOED",
        "study_id": protocol["study_id"],
        "primary_arm": ARMS[5],
        "rows": 4977,
        "outer_folds": 5,
        "prediction_seeds": list(SEEDS),
        "cached_parameter_candidates": 1500,
        "new_neural_fits": 0,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "protocol_sha256": file_sha256(PROTOCOL),
        "input_files": [_file_record(path) for path in sorted(inputs, key=str)],
        "outer_metric_embargo": protocol["selection"]["outer_metric_embargo"],
    }
    run.mkdir(parents=True, exist_ok=True)
    immutable_json(run / "execution_lock.json", lock)
    return protocol, data, file_sha256(run / "execution_lock.json")


def validate_lock(run: Path, data: dict[str, np.ndarray], expected_hash: str) -> dict[str, Any]:
    path = run / "execution_lock.json"
    if file_sha256(path) != expected_hash:
        raise RuntimeError("ARFTR execution-lock bytes changed")
    lock = _read_json(path)
    if (
        lock.get("status") != "ARFTR_EXECUTION_LOCKED_ADAPTIVE_OUTER_METRICS_EMBARGOED"
        or lock.get("sample_ids_sha256") != canonical_hash(data["sample_ids"].tolist())
        or lock.get("cached_parameter_candidates") != 1500
    ):
        raise RuntimeError("ARFTR execution-lock identity changed")
    for record in lock["input_files"]:
        input_path = (ROOT / record["path"]).resolve()
        input_path.relative_to(ROOT.resolve())
        if (
            not input_path.is_file()
            or input_path.stat().st_size != record["size_bytes"]
            or file_sha256(input_path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked ARFTR input changed: {record['path']}")
    return lock


def parameter_grid(protocol: dict[str, Any]) -> tuple[ARFTRParameters, ...]:
    selection = protocol["selection"]
    result = tuple(
        ARFTRParameters(*values)
        for values in itertools.product(
            selection["posture_restoration"],
            selection["motion_restoration"],
            selection["a3_motion_residual"],
            selection["temporal_strength"],
        )
    )
    if len(result) != 300:
        raise RuntimeError("ARFTR parameter-grid cardinality changed")
    return result


def _selection_key(metrics: dict[str, Any], parameters: ARFTRParameters) -> tuple[Any, ...]:
    values = tuple(float(value) for value in parameters.as_array())
    return (-metrics["macro_f1"], metrics["nll"], sum(values), values)


def select_parameters(
    labels: np.ndarray,
    m4: np.ndarray,
    p6: np.ndarray,
    a3: np.ndarray,
    neighbors: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[ARFTRParameters, dict[str, Any]]:
    candidates = []
    for parameters in parameter_grid(protocol):
        probabilities = apply_arftr(
            m4, p6, a3, neighbors, parameters, epsilon=protocol["factorization"]["epsilon"]
        )
        metrics = probability_metrics(labels, probabilities)
        candidates.append((parameters, metrics))
    parameters, metrics = min(candidates, key=lambda item: _selection_key(item[1], item[0]))
    return parameters, metrics


def _arm_parameters(parameters: ARFTRParameters) -> tuple[ARFTRParameters, ...]:
    p, m, a, t = parameters.as_array().tolist()
    zero = ARFTRParameters(0.0, 0.0, 0.0, 0.0)
    return (
        zero,
        ARFTRParameters(p, m, 0.0, 0.0),
        ARFTRParameters(0.0, 0.0, a, 0.0),
        ARFTRParameters(0.0, 0.0, 0.0, t),
        ARFTRParameters(p, m, a, 0.0),
        parameters,
        parameters,
    )


def _fold_paths(run: Path, fold: int) -> tuple[Path, Path, Path]:
    directory = run / f"fold-{fold}"
    return directory / "predictions.npz", directory / "checkpoint.npz", directory / "receipt.json"


def fit_fold(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    lock_hash: str,
    fold: int,
) -> dict[str, Any]:
    validate_lock(run, data, lock_hash)
    prediction_path, checkpoint_path, receipt_path = _fold_paths(run, fold)
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if (
            receipt.get("status") != "ARFTR_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED"
            or receipt.get("execution_lock_sha256") != lock_hash
            or not prediction_path.is_file()
            or not checkpoint_path.is_file()
            or file_sha256(prediction_path) != receipt.get("predictions_sha256")
            or file_sha256(checkpoint_path) != receipt.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"Completed ARFTR fold {fold} is stale or changed")
        return receipt
    if receipt_path.parent.exists() and any(receipt_path.parent.iterdir()):
        raise RuntimeError(f"Partial ARFTR fold {fold} retained; use a fresh run path")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    cached = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    cache = _p6_cache(data)
    train, held = cached.train_rows, cached.held_rows
    p6_all, p6_ancestry = cache.meta_probabilities(train)
    p6_inner = p6_all[train]
    p6_outer = cache.held_predictions(train, held)
    inner_neighbors = exact_track_neighbors(data, train)
    outer_neighbors = exact_track_neighbors(data, held)
    shuffled = shuffled_neighbors(
        outer_neighbors,
        data["scenarios"][held],
        seed=int(protocol["control_contract"]["shuffle_seed"]) + fold,
    )
    parameters, training_metrics = select_parameters(
        data["labels"][train],
        cached.m4_inner["probabilities"],
        p6_inner,
        cached.a3_inner["probabilities"],
        inner_neighbors,
        protocol,
    )
    np.savez_compressed(
        checkpoint_path,
        parameters=parameters.as_array(),
        training_macro_f1=np.asarray([training_metrics["macro_f1"]]),
        training_nll=np.asarray([training_metrics["nll"]]),
        inner_neighbor_map=inner_neighbors,
        outer_neighbor_map=outer_neighbors,
        shuffled_outer_neighbor_map=shuffled,
    )
    seed_probabilities = np.empty((len(ARMS), len(SEEDS), len(held), 3), dtype=np.float64)
    arm_parameters = _arm_parameters(parameters)
    epsilon = protocol["factorization"]["epsilon"]
    for seed_index in range(len(SEEDS)):
        m4 = cached.m4_outer["probabilities"][seed_index]
        a3 = cached.a3_outer["probabilities"][seed_index]
        for arm_index, arm_parameter in enumerate(arm_parameters):
            neighbor_map = shuffled if arm_index == 6 else outer_neighbors
            seed_probabilities[arm_index, seed_index] = apply_arftr(
                m4, p6_outer, a3, neighbor_map, arm_parameter, epsilon=epsilon
            )
        if not np.array_equal(seed_probabilities[0, seed_index], m4):
            raise RuntimeError("Exact M4 control did not preserve anchor bytes")
    np.savez_compressed(
        prediction_path,
        sample_ids=data["sample_ids"][held],
        held_rows=held,
        arms=np.asarray(ARMS),
        seeds=np.asarray(SEEDS),
        seed_probabilities=seed_probabilities,
        p6_probabilities=p6_outer,
        a3_seed_probabilities=cached.a3_outer["probabilities"],
    )
    receipt = {
        "status": "ARFTR_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": lock_hash,
        "fold": fold,
        "outer_train_rows": int(len(train)),
        "outer_held_rows": int(len(held)),
        "outer_held_labels_read": 0,
        "cached_parameter_candidates": 300,
        "selected_parameters": parameters.as_array().tolist(),
        "training_metrics": training_metrics,
        "inner_exact_neighbor_counts": np.bincount(
            np.sum(inner_neighbors >= 0, axis=1), minlength=3
        ).tolist(),
        "outer_exact_neighbor_counts": np.bincount(
            np.sum(outer_neighbors >= 0, axis=1), minlength=3
        ).tolist(),
        "p6_inner_ancestry": p6_ancestry,
        "train_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "predictions_sha256": file_sha256(prediction_path),
    }
    immutable_json(receipt_path, receipt)
    print(json.dumps({"event": "arftr_fold_complete", "fold": fold, "held_rows": len(held)}))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--fold", default="all", choices=("all", "0", "1", "2", "3", "4"))
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    run.relative_to(ROOT.resolve())
    protocol, data, lock_hash = prepare(run)
    folds = range(5) if arguments.fold == "all" else (int(arguments.fold),)
    for fold in folds:
        fit_fold(run, protocol, data, lock_hash, fold)
    print(
        json.dumps(
            {
                "status": "ARFTR_REQUESTED_FOLDS_COMPLETE_OUTER_METRICS_STILL_EMBARGOED",
                "run": _relative(run),
                "folds": list(folds),
            }
        )
    )


if __name__ == "__main__":
    main()
