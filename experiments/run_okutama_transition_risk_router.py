"""Run the fully nested ARFTR-preserving source/posture router.

The runner has three explicit stages:

* ``prepare`` writes a lock for 25 inner F(S) populations and their base-cache
  dependencies.
* ``bases`` prepares only missing label-blind P6 base populations.
* ``fit-one`` fits one inner F(S) population using the corrected generalized
  ancestor executor; ``queue`` completes all pending populations.
* ``fit-router`` builds source/posture inner OOF rows and applies the router to
  untouched outer rows without reading outer labels.
* ``summarize`` releases metrics only after all fold receipts exist.

The source/posture candidate is deliberately used as a residual expert, never
as a replacement for ARFTR. Every outer row has the explicit retain action.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_okutama_generalized_arftr_ancestors as legacy
import run_okutama_generalized_arftr_ancestors_v2 as generalized_v2

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.actor_memory_base import NestedBaseCache, group_splits
from hac.arftr import ARFTRParameters, apply_arftr, exact_track_neighbors
from hac.generalized_arftr import PopulationInputs, execute_population
from hac.posture_witness import apply_posture_policy, apply_projection, fit_posture_head, fit_projection
from hac.source_posture_cache import load_source_posture_cache
from hac.transition_risk_router import router_features, fit_transition_risk_router

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
METADATA = SOURCE / "data/memory_data.npz"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1"
SOURCE_POSTURE = ROOT / ".runs/research_20260913/source_posture_fixed_primary_v1"
SOURCE_CACHE = ROOT / ".runs/research_20260913/source_posture_cache_v1"
SOURCE_CACHE_AUDIT = ROOT / ".runs/research_20260913/source_posture_cache_v1_audit.json"
GENERALIZED_V2 = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v2"
ROUTER_ROOT = ROOT / ".runs/research_20260916/source_posture_failure_router_v1"
DEFAULT_RUN = ROUTER_ROOT / "inner_ancestors"
DEFAULT_RESULT = ROUTER_ROOT / "router_results"
PLAN = ROUTER_ROOT / "inner_cache_plan.json"
LOCK = DEFAULT_RUN / "execution_lock.json"
CROP_KEYS = (
    "J_whole", "J_upper", "J_lower", "R_whole", "R_upper", "R_lower",
    "N_whole", "N_upper", "N_lower", "R_whole_1p0", "R_whole_1p5",
)
POSTURE_KEYS = ("R_whole", "R_upper", "R_lower")
V1_BASE = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v1"
V1_BASE_COMPLETION = V1_BASE / "base_completion.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Immutable router artifact changed: {path}")
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[name], value, equal_nan=True)
                for name, value in arrays.items()
            ):
                raise RuntimeError(f"Immutable router array changed: {path}")
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


class ReadThroughBaseCache:
    """Read historical/v1 bases read-only and write new bases under this run."""

    def __init__(self, run: Path, d: dict[str, np.ndarray]):
        historical = legacy._historical_base(d)
        self.layers = [
            ("historical", historical),
            (
                "generalized_v1",
                NestedBaseCache(
                    V1_BASE / "base_cache",
                    historical.features,
                    historical.labels,
                    historical.scenarios,
                    historical.sample_ids,
                    historical.config,
                    historical.identity["feature_sha256"],
                ),
            ),
            (
                "router_current",
                NestedBaseCache(
                    run / "base_cache",
                    historical.features,
                    historical.labels,
                    historical.scenarios,
                    historical.sample_ids,
                    historical.config,
                    historical.identity["feature_sha256"],
                ),
            ),
        ]
        completion = read_json(V1_BASE_COMPLETION)
        if completion.get("status") != "GENERALIZED_BASES_COMPLETE":
            raise RuntimeError("generalized v1 base completion is not valid")
        expected = {
            entry["receipt_sha256"]
            for entry in completion["receipts"]
            if entry.get("layer") == "current"
        }
        actual = {
            file_sha256(path / "receipt.json")
            for path in (V1_BASE / "base_cache").iterdir()
            if path.is_dir() and (path / "receipt.json").is_file()
        }
        if actual != expected:
            raise RuntimeError("generalized v1 base cache inventory changed")

    @staticmethod
    def _exists(cache: NestedBaseCache, train: np.ndarray) -> bool:
        return (cache.directory(train) / "receipt.json").is_file()

    def _chosen(self, train: np.ndarray) -> tuple[str, NestedBaseCache]:
        for name, cache in self.layers:
            if self._exists(cache, train):
                return name, cache
        return self.layers[-1]

    def directory(self, train: np.ndarray) -> Path:
        return self._chosen(train)[1].directory(train)

    def prepare(self, train: np.ndarray) -> dict[str, Any]:
        name, cache = self._chosen(train)
        if name != "router_current":
            return cache.load(train)[1]
        return cache.prepare(train)

    def load(self, train: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return self._chosen(train)[1].load(train)

    def held_predictions(self, train: np.ndarray, held: np.ndarray) -> np.ndarray:
        if set(self.layers[0][1].scenarios[train]) & set(self.layers[0][1].scenarios[held]):
            raise RuntimeError("Refusing in-scenario read-through base predictions")
        return self.load(train)[0][held]

    def meta_probabilities(self, population: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        result = np.full((len(self.layers[0][1].labels), 3), np.nan)
        ancestry = []
        for train, held in group_splits(
            self.layers[0][1].labels, self.layers[0][1].scenarios, population
        ):
            result[held] = self.held_predictions(train, held)
            name, cache = self._chosen(train)
            ancestry.append({"predicted_rows": held.tolist(), "base_cache": cache.directory(train).name, "cache_layer": name})
        outside = np.setdiff1d(np.arange(len(result)), population)
        result[outside] = self.held_predictions(population, outside)
        name, cache = self._chosen(population)
        ancestry.append({"predicted_rows": outside.tolist(), "base_cache": cache.directory(population).name, "cache_layer": name})
        if not np.isfinite(result).all():
            raise RuntimeError("Read-through meta-prediction coverage failed")
        return result, ancestry


def data() -> dict[str, np.ndarray]:
    with np.load(METADATA, allow_pickle=False) as saved:
        result = {name: saved[name] for name in saved.files}
    if len(result["labels"]) != 4977 or len(np.unique(result["sample_ids"])) != 4977:
        raise RuntimeError("Router cohort identity changed")
    if not np.array_equal(np.unique(result["folds"]), np.arange(5)):
        raise RuntimeError("Router outer folds changed")
    return result


def _inner_rows(d: dict[str, np.ndarray], outer: int, inner: int) -> tuple[np.ndarray, np.ndarray]:
    plan = read_json(PLAN)["outer_plans"][outer]
    assignment = plan["scenario_assignment"]
    outer_train = np.flatnonzero(d["folds"] != outer)
    held = outer_train[
        np.array([assignment[str(d["scenarios"][row])] == inner for row in outer_train])
    ]
    train = np.setdiff1d(outer_train, held, assume_unique=True)
    if canonical_hash(d["sample_ids"][train].tolist()) != plan["inner_folds"][inner]["train_rows_sha256"]:
        raise RuntimeError("Inner training identity changed")
    if canonical_hash(d["sample_ids"][held].tolist()) != plan["inner_folds"][inner]["held_rows_sha256"]:
        raise RuntimeError("Inner held identity changed")
    return train, held


def population_id(outer: int, inner: int, d: dict[str, np.ndarray]) -> str:
    train, _held = _inner_rows(d, outer, inner)
    return f"o{outer}_i{inner}_{canonical_hash(d['sample_ids'][train].tolist())[:20]}"


def prepare(run: Path) -> dict[str, Any]:
    d = data()
    router_lock = read_json(ROUTER_ROOT / "execution_lock.json")
    if router_lock.get("status") != "TRANSITION_RISK_ROUTER_PREFLIGHT_PASS_SMOKE_AUTHORIZED":
        raise RuntimeError("Router preflight lock has not passed")
    base_lock = read_json(GENERALIZED_V2 / "execution_lock.json")
    populations: dict[str, dict[str, Any]] = {}
    for outer in range(5):
        for inner in range(5):
            train, held = _inner_rows(d, outer, inner)
            pid = population_id(outer, inner, d)
            populations[pid] = {
                "outer_fold": outer,
                "inner_fold": inner,
                "train_rows": int(len(train)),
                "prediction_rows": int(len(d["labels"]) - len(train)),
                "train_sample_ids_sha256": canonical_hash(d["sample_ids"][train].tolist()),
                "held_rows_sha256": canonical_hash(d["sample_ids"][held].tolist()),
                "training_scenarios": sorted(set(d["scenarios"][train].tolist())),
                "prediction_scenarios": sorted(set(d["scenarios"][np.setdiff1d(np.arange(len(d["labels"])), train)].tolist())),
            }
    lock = {
        "status": "TRANSITION_RISK_INNER_ANCESTORS_LOCKED",
        "router_execution_lock_sha256": file_sha256(ROUTER_ROOT / "execution_lock.json"),
        "generalized_v2_execution_lock_sha256": file_sha256(GENERALIZED_V2 / "execution_lock.json"),
        "metadata_sha256": file_sha256(METADATA),
        "source_posture_cache_audit_sha256": file_sha256(SOURCE_CACHE_AUDIT),
        "rows": 4977,
        "outer_folds": 5,
        "inner_folds_per_outer": 5,
        "F_populations": populations,
        "protocol": base_lock["protocol"],
        "memory_protocol": base_lock["memory_protocol"],
        "parameter_counts": base_lock["parameter_counts"],
        "device": base_lock["device"],
        "task_outer_metrics_embargoed": True,
        "outer_prediction_labels_available_to_fits": False,
        "new_fits_at_lock": 0,
    }
    write_json(run / "execution_lock.json", lock)
    return lock


def validate_lock(run: Path) -> dict[str, Any]:
    lock = read_json(run / "execution_lock.json")
    if lock.get("status") != "TRANSITION_RISK_INNER_ANCESTORS_LOCKED":
        raise RuntimeError("Inner ancestor lock is invalid")
    if lock.get("outer_prediction_labels_available_to_fits") is not False:
        raise RuntimeError("Outer prediction labels are not embargoed")
    if file_sha256(ROUTER_ROOT / "execution_lock.json") != lock["router_execution_lock_sha256"]:
        raise RuntimeError("Router lock changed")
    authorization = read_json(ROUTER_ROOT / "full_fit_authorization.json")
    if authorization.get("status") != "TRANSITION_RISK_ROUTER_FULL_FIT_AUTHORIZED":
        raise RuntimeError("Full router fit has not been authorized")
    router_lock = read_json(ROUTER_ROOT / "execution_lock.json")
    if authorization.get("router_preflight_lock_sha256") != file_sha256(ROUTER_ROOT / "execution_lock.json"):
        raise RuntimeError("Full-fit authorization is detached from the router preflight lock")
    if authorization.get("router_preflight_sha256") != router_lock.get("preflight_sha256"):
        raise RuntimeError("Full-fit authorization preflight hash changed")
    if authorization.get("outer_labels_available_to_fits") is not False:
        raise RuntimeError("Full-fit authorization permits outer labels")
    if file_sha256(METADATA) != lock["metadata_sha256"]:
        raise RuntimeError("Metadata changed")
    return lock


def _prepare_population_bases(run: Path, population: np.ndarray, d: dict[str, np.ndarray], cache: Any) -> int:
    needed: dict[str, np.ndarray] = {}

    def add(rows: np.ndarray) -> None:
        needed[canonical_hash(rows.tolist())] = rows

    add(population)
    for inner_train, _held in legacy.group_splits(d["labels"], d["scenarios"], population):
        add(inner_train)
        for base_train, _base_held in legacy.group_splits(d["labels"], d["scenarios"], inner_train):
            add(base_train)
    missing = [rows for rows in needed.values() if not (cache.directory(rows) / "receipt.json").is_file()]
    for rows in missing:
        cache.prepare(rows)
    return len(missing)


def fit_one(run: Path, pid: str) -> dict[str, Any]:
    lock = validate_lock(run)
    d = legacy._source_data()
    spec = lock["F_populations"].get(pid)
    if spec is None:
        raise ValueError(f"Unknown router population: {pid}")
    train, held_inner = _inner_rows(d, int(spec["outer_fold"]), int(spec["inner_fold"]))
    prediction = np.setdiff1d(np.arange(len(d["labels"])), train)
    if canonical_hash(d["sample_ids"][train].tolist()) != spec["train_sample_ids_sha256"]:
        raise RuntimeError("Population train identity changed")
    cache = ReadThroughBaseCache(run, d)
    started = time.perf_counter()
    _prepare_population_bases(run, train, d, cache)
    sear_lock, patches, center, sear_data = legacy.sear_driver.load_locked(SEAR)
    del sear_lock
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(sear_data[name], d[name]):
            raise RuntimeError("A3 and M4 cohorts differ")
    m4_inner, m4_outer, m4_receipts = generalized_v2._fit_m4(
        run, lock, d, cache, pid, train, prediction
    )
    a3_inner, a3_outer, a3_receipts = legacy._fit_a3(
        run, lock, patches, center, sear_data, pid, int(spec["outer_fold"] * 5 + spec["inner_fold"]), train, prediction
    )
    p6_all, p6_ancestry = cache.meta_probabilities(train)
    composed = execute_population(
        d,
        train,
        prediction,
        PopulationInputs(
            m4_inner=m4_inner,
            p6_inner=p6_all[train],
            a3_inner=a3_inner,
            m4_outer=m4_outer,
            p6_outer=cache.held_predictions(train, prediction),
            a3_outer=a3_outer,
        ),
        read_json(ROOT / "experiments/okutama_arftr_protocol.json"),
    )
    out_dir = run / "populations" / pid
    output_path = out_dir / "predictions.npz"
    write_npz(
        output_path,
        population_rows=train,
        prediction_rows=prediction,
        sample_ids=d["sample_ids"][prediction],
        parameters=composed.parameters.as_array(),
        seed_probabilities=composed.seed_probabilities,
        mean_probabilities=composed.mean_probabilities,
        m4_inner=m4_inner,
        a3_inner=a3_inner,
        p6_inner=p6_all[train],
        m4_outer=m4_outer,
        a3_outer=a3_outer,
        p6_outer=cache.held_predictions(train, prediction),
        inner_neighbor_map=composed.inner_neighbor_map,
        outer_neighbor_map=composed.outer_neighbor_map,
    )
    receipt = {
        "status": "TRANSITION_RISK_INNER_F_COMPLETE",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "population_id": pid,
        "outer_fold": int(spec["outer_fold"]),
        "inner_fold": int(spec["inner_fold"]),
        "training_rows": int(len(train)),
        "prediction_rows": int(len(prediction)),
        "router_inner_held_rows": int(len(held_inner)),
        "selected_parameters": composed.parameters.as_array().tolist(),
        "training_metrics": composed.training_metrics,
        "m4_fit_receipts": m4_receipts,
        "a3_fit_receipts": a3_receipts,
        "p6_ancestry": p6_ancestry,
        "neural_fits": 30,
        "elapsed_seconds": time.perf_counter() - started,
        "outer_prediction_labels_read": 0,
        "predictions_sha256": file_sha256(output_path),
    }
    write_json(out_dir / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "population_id": pid, "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))
    return receipt


def queue(run: Path) -> None:
    lock = validate_lock(run)
    for pid in lock["F_populations"]:
        receipt_path = run / "populations" / pid / "receipt.json"
        if receipt_path.is_file():
            continue
        fit_one(run, pid)
    print(json.dumps({"status": "TRANSITION_RISK_INNER_ANCESTOR_QUEUE_COMPLETE", "populations": len(lock["F_populations"])}, indent=2))


def _load_cache(d: dict[str, np.ndarray]):
    audit = read_json(SOURCE_CACHE_AUDIT)
    if audit.get("status") != "SOURCE_POSTURE_CACHE_AUDIT_PASS":
        raise RuntimeError("Source/posture cache audit has not passed")
    return load_source_posture_cache(SOURCE_CACHE, d["sample_ids"], expected_crop_keys=CROP_KEYS)


def _f_artifact(run: Path, pid: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = run / "populations" / pid
    receipt = read_json(directory / "receipt.json")
    if receipt.get("status") != "TRANSITION_RISK_INNER_F_COMPLETE" or receipt.get("predictions_sha256") != file_sha256(directory / "predictions.npz"):
        raise RuntimeError(f"Incomplete inner F population: {pid}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        return receipt, {name: saved[name] for name in saved.files}


def fit_router(run: Path, result_dir: Path) -> dict[str, Any]:
    lock = validate_lock(run)
    d = data()
    cache = _load_cache(d)
    result_dir.mkdir(parents=True, exist_ok=True)
    fold_receipts = []
    for outer in range(5):
        outer_train = np.flatnonzero(d["folds"] != outer)
        outer_held = np.flatnonzero(d["folds"] == outer)
        inner_anchor = np.full((len(d["labels"]), 3), np.nan, dtype=np.float64)
        inner_candidate = np.full((len(d["labels"]), 3), np.nan, dtype=np.float64)
        inner_available = np.zeros(len(d["labels"]), dtype=bool)
        inner_quality = np.zeros((len(d["labels"]), 3), dtype=np.float64)
        for inner in range(5):
            train, held = _inner_rows(d, outer, inner)
            pid = population_id(outer, inner, d)
            _receipt, f = _f_artifact(run, pid)
            prediction_rows = f["prediction_rows"]
            locations = np.searchsorted(prediction_rows, held)
            if np.any(locations >= len(prediction_rows)) or not np.array_equal(prediction_rows[locations], held):
                raise RuntimeError("Inner F prediction coverage changed")
            anchor_held = f["mean_probabilities"][locations]
            parameters = ARFTRParameters(*f["parameters"].tolist())
            neighbors = exact_track_neighbors(d, train)
            anchor_train = apply_arftr(
                f["m4_inner"], f["p6_inner"], f["a3_inner"], neighbors, parameters,
                epsilon=read_json(ROOT / "experiments/okutama_arftr_protocol.json")["factorization"]["epsilon"],
            )
            descriptor, available = cache.descriptor(POSTURE_KEYS)
            projection = fit_projection(descriptor[train][available[train]])
            projected_train = apply_projection(descriptor[train], projection)
            projected_held = apply_projection(descriptor[held], projection)
            head = fit_posture_head(
                projected_train,
                d["labels"][train],
                anchor_train,
                available=available[train],
            )
            candidate_held, _intervention, _delta = apply_posture_policy(
                anchor_held,
                projected_held,
                head,
                available=available[held],
            )
            inner_anchor[held] = anchor_held
            inner_candidate[held] = candidate_held
            inner_available[held] = available[held]
            inner_quality[held] = np.linalg.norm(descriptor[held].reshape(len(held), 3, 768), axis=2) / np.sqrt(768.0)
        if not np.isfinite(inner_anchor[outer_train]).all() or not np.isfinite(inner_candidate[outer_train]).all():
            raise RuntimeError("Inner OOF coverage is incomplete")
        router_x, names = router_features(
            inner_anchor[outer_train],
            inner_candidate[outer_train],
            availability=inner_available[outer_train],
            quality=inner_quality[outer_train],
        )
        router = fit_transition_risk_router(
            router_x,
            names,
            inner_anchor[outer_train],
            inner_candidate[outer_train],
            d["labels"][outer_train],
            harm_cost=2.0,
            random_state=42,
        )
        anchor_path = SOURCE_POSTURE / f"fold-{outer}" / "A0_retain" / "predictions.npz"
        candidate_path = SOURCE_POSTURE / f"fold-{outer}" / "R_parts" / "predictions.npz"
        with np.load(anchor_path, allow_pickle=False) as saved:
            anchor_outer = saved["probabilities"]
            held_rows = saved["held_rows"]
        with np.load(candidate_path, allow_pickle=False) as saved:
            candidate_outer = saved["probabilities"]
        if not np.array_equal(held_rows, outer_held):
            raise RuntimeError("Source/posture outer held identity changed")
        descriptor_outer, available_outer = cache.descriptor(POSTURE_KEYS)
        quality_outer = np.linalg.norm(descriptor_outer[outer_held].reshape(len(outer_held), 3, 768), axis=2) / np.sqrt(768.0)
        outer_x, _outer_names = router_features(
            anchor_outer,
            candidate_outer,
            availability=available_outer[outer_held],
            quality=quality_outer,
        )
        routed, action = router.apply(anchor_outer, candidate_outer, outer_x)
        fold_dir = result_dir / f"fold-{outer}"
        write_npz(
            fold_dir / "predictions.npz",
            sample_ids=d["sample_ids"][outer_held],
            held_rows=outer_held,
            anchor_probabilities=anchor_outer,
            candidate_probabilities=candidate_outer,
            routed_probabilities=routed,
            action=action,
            router_scores=router.score(outer_x),
        )
        model_arrays = {
            "scaler_mean": router.scaler.mean_ if router.scaler is not None else np.empty(0),
            "scaler_scale": router.scaler.scale_ if router.scaler is not None else np.empty(0),
            "coefficients": router.model.coef_ if router.model is not None else np.empty((0, 0)),
            "intercept": router.model.intercept_ if router.model is not None else np.empty(0),
            "feature_names": np.asarray(names),
            "threshold": np.asarray([router.threshold]),
        }
        write_npz(fold_dir / "router.npz", **model_arrays)
        fold_receipts.append(
            {
                "status": "TRANSITION_RISK_ROUTER_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED",
                "outer_fold": outer,
                "outer_train_rows": int(len(outer_train)),
                "outer_held_rows": int(len(outer_held)),
                "outer_held_labels_read": 0,
                "inner_populations": [population_id(outer, i, d) for i in range(5)],
                "router_training": router.training_summary,
                "action_rows": int(action.sum()),
                "predictions_sha256": file_sha256(fold_dir / "predictions.npz"),
                "router_sha256": file_sha256(fold_dir / "router.npz"),
            }
        )
        write_json(fold_dir / "receipt.json", fold_receipts[-1])
    receipt = {
        "status": "TRANSITION_RISK_ROUTER_OOF_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "rows": 4977,
        "outer_folds": 5,
        "outer_held_labels_read": 0,
        "fold_receipts": fold_receipts,
    }
    write_json(result_dir / "fit_receipt.json", receipt)
    return receipt


def summarize(result_dir: Path) -> dict[str, Any]:
    d = data()
    fit_receipt = read_json(result_dir / "fit_receipt.json")
    if fit_receipt.get("status") != "TRANSITION_RISK_ROUTER_OOF_COMPLETE_OUTER_METRICS_EMBARGOED":
        raise RuntimeError("Router OOF fit is incomplete")
    rows, anchor, candidate, routed, actions = [], [], [], [], []
    for outer in range(5):
        fold_dir = result_dir / f"fold-{outer}"
        fold_receipt = read_json(fold_dir / "receipt.json")
        if fold_receipt.get("outer_held_labels_read") != 0:
            raise RuntimeError("Router fold consumed outer labels during fit")
        with np.load(fold_dir / "predictions.npz", allow_pickle=False) as saved:
            rows.append(saved["held_rows"])
            anchor.append(saved["anchor_probabilities"])
            candidate.append(saved["candidate_probabilities"])
            routed.append(saved["routed_probabilities"])
            actions.append(saved["action"])
    held_rows = np.concatenate(rows)
    if not np.array_equal(np.sort(held_rows), np.arange(len(d["labels"]))):
        raise RuntimeError("Router OOF rows do not cover the cohort exactly once")
    order = np.argsort(held_rows)
    anchor_p = np.concatenate(anchor)[order]
    candidate_p = np.concatenate(candidate)[order]
    routed_p = np.concatenate(routed)[order]
    action = np.concatenate(actions)[order]
    audit_path = result_dir / "independent_audit.json"
    if not audit_path.is_file() or read_json(audit_path).get("status") != "TRANSITION_RISK_ROUTER_INDEPENDENT_REPLAY_PASS":
        raise RuntimeError("Independent router replay must pass before metrics release")
    metrics = {
        "anchor": probability_metrics(d["labels"], anchor_p),
        "candidate": probability_metrics(d["labels"], candidate_p),
        "routed": probability_metrics(d["labels"], routed_p),
    }
    anchor_class, routed_class = anchor_p.argmax(1), routed_p.argmax(1)
    true = d["labels"]
    rescue = (routed_class == true) & (anchor_class != true)
    harm = (routed_class != true) & (anchor_class == true)
    net = int(rescue.sum() - harm.sum())
    per_fold = []
    for outer in range(5):
        held = d["folds"] == outer
        per_fold.append(int(rescue[held].sum() - harm[held].sum()))
    scenario_ids = np.asarray(d["scenarios"])
    unique_scenarios = np.unique(scenario_ids)
    scenario_rows = [np.flatnonzero(scenario_ids == scenario) for scenario in unique_scenarios]
    bootstrap_rng = np.random.default_rng(20260916)
    bootstrap_deltas = np.empty(10000, dtype=np.float64)
    for index in range(len(bootstrap_deltas)):
        selected = bootstrap_rng.integers(0, len(scenario_rows), size=len(scenario_rows))
        sample_rows = np.concatenate([scenario_rows[value] for value in selected])
        bootstrap_deltas[index] = f1_score(
            true[sample_rows], routed_class[sample_rows], average="macro"
        ) - f1_score(true[sample_rows], anchor_class[sample_rows], average="macro")
    bootstrap_interval = np.quantile(bootstrap_deltas, [0.025, 0.975]).tolist()
    summary = {
        "status": "TRANSITION_RISK_ROUTER_SCIENTIFIC_SUMMARY_COMPLETE",
        "anchor_macro_f1": metrics["anchor"]["macro_f1"],
        "candidate_macro_f1": metrics["candidate"]["macro_f1"],
        "routed_macro_f1": metrics["routed"]["macro_f1"],
        "metrics": metrics,
        "rescues": int(rescue.sum()),
        "harms": int(harm.sum()),
        "net_corrections": net,
        "per_outer_fold_net_corrections": per_fold,
        "action_rows": int(action.sum()),
        "outer_label_reads_during_fit": 0,
        "independent_replay": True,
        "scenario_bootstrap": {
            "resamples": int(len(bootstrap_deltas)),
            "seed": 20260916,
            "delta_routed_minus_anchor_macro_f1_95_interval": bootstrap_interval,
        },
        "gates": {
            "macro_f1": bool(metrics["routed"]["macro_f1"] >= 0.8588364807373623),
            "net_corrections": bool(net >= 25),
            "positive_each_outer_fold": bool(all(value > 0 for value in per_fold)),
            "class_f1_loss_cap": bool(np.min(np.asarray(metrics["routed"]["per_class_f1"]) - np.asarray(metrics["anchor"]["per_class_f1"])) >= -0.005),
            "nll_not_worse": bool(metrics["routed"]["nll"] <= metrics["anchor"]["nll"]),
            "brier_not_worse": bool(metrics["routed"]["brier"] <= metrics["anchor"]["brier"]),
            "independent_replay": True,
            "scenario_bootstrap_lower_positive": bool(bootstrap_interval[0] > 0),
        },
    }
    write_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--stage", choices=("prepare", "bases", "fit-one", "queue", "fit-router", "summarize"), required=True)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--population-id")
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    if args.stage == "prepare":
        lock = prepare(run)
        print(json.dumps({"status": lock["status"], "populations": len(lock["F_populations"])}, indent=2))
    elif args.stage == "bases":
        raise RuntimeError("Use fit-one: each custom population prepares only its exact read-through bases before fitting")
    elif args.stage == "fit-one":
        if args.population_id is None:
            if args.outer_fold is None or args.inner_fold is None:
                raise ValueError("fit-one requires --population-id or --outer-fold and --inner-fold")
            args.population_id = population_id(args.outer_fold, args.inner_fold, data())
        fit_one(run, args.population_id)
    elif args.stage == "queue":
        queue(run)
    elif args.stage == "fit-router":
        fit_router(run, args.result_dir.resolve())
    else:
        summarize(args.result_dir.resolve())


if __name__ == "__main__":
    main()
