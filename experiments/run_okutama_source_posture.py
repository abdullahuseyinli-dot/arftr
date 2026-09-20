"""Fit the fixed source-resolved posture witness after every ancestry gate.

No outer-held label is indexed by ``prepare`` or ``fit-fold``. Metrics are
released only by ``summarize`` after the independent replay audit exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.nested_arftr_plan import fixed_recipe_plan
from hac.posture_witness import (
    apply_posture_policy,
    apply_projection,
    fit_posture_head,
    fit_projection,
)
from hac.source_posture_cache import SourcePostureCache, load_source_posture_cache

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_source_posture_protocol.json"
METADATA = ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz"
CACHE = ROOT / ".runs/research_20260913/source_posture_cache_v1"
CACHE_AUDIT = ROOT / ".runs/research_20260913/source_posture_cache_v1_audit.json"
ANCESTORS = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v2"
ANCESTOR_AUDIT = ANCESTORS / "independent_audit.json"
PARITY = ROOT / ".runs/research_20260913/generalized_arftr_parity_v1/audit.json"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1"
DEFAULT_RUN = ROOT / ".runs/research_20260913/source_posture_fixed_primary_v1"
CROP_KEYS = (
    "J_whole",
    "J_upper",
    "J_lower",
    "R_whole",
    "R_upper",
    "R_lower",
    "N_whole",
    "N_upper",
    "N_lower",
    "R_whole_1p0",
    "R_whole_1p5",
)
ARM_KEYS: dict[str, tuple[str, ...] | None] = {
    "A0_retain": None,
    "C0_anchor_calibration": (),
    "J_whole": ("J_whole",),
    "J_parts": ("J_whole", "J_upper", "J_lower"),
    "R_whole": ("R_whole",),
    "R_parts": ("R_whole", "R_upper", "R_lower"),
    "N_whole": ("N_whole",),
    "N_parts": ("N_whole", "N_upper", "N_lower"),
    "R_multiscale_whole": ("R_whole_1p0", "R_whole", "R_whole_1p5"),
}
PRIMARY = "R_parts"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != value:
            raise RuntimeError(f"Immutable source/posture artifact changed: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _immutable_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not (
                    np.array_equal(saved[name], value, equal_nan=True)
                    if np.issubdtype(np.asarray(value).dtype, np.inexact)
                    else np.array_equal(saved[name], value)
                )
                for name, value in arrays.items()
            ):
                raise RuntimeError(f"Immutable source/posture array changed: {path}")
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _data() -> dict[str, np.ndarray]:
    with np.load(METADATA, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    if (
        len(data["labels"]) != 4977
        or len(np.unique(data["sample_ids"])) != 4977
        or not np.array_equal(np.unique(data["labels"]), np.arange(3))
        or not np.array_equal(np.unique(data["folds"]), np.arange(5))
    ):
        raise RuntimeError("Source/posture task cohort changed")
    return data


def _feature_cache(data: dict[str, np.ndarray]) -> SourcePostureCache:
    return load_source_posture_cache(
        CACHE,
        data["sample_ids"],
        expected_crop_keys=CROP_KEYS,
    )


def _ancestor_population_receipts(
    plan: dict[str, Any],
) -> tuple[list[Path], set[str]]:
    parity = _read_json(PARITY)
    if (
        parity.get("status")
        != "GENERALIZED_ARFTR_FT_PARITY_EXACT_AND_REUSE_AUDIT_PASS"
        or not parity.get("final_mean_probabilities_exact")
        or parity.get("final_maximum_absolute_difference") != 0.0
    ):
        raise RuntimeError("Generalized canonical F(T) parity is not exact")
    lock = _read_json(ANCESTORS / "execution_lock.json")
    completion = _read_json(ANCESTORS / "base_completion.json")
    if (
        lock.get("status") != "GENERALIZED_ARFTR_ANCESTORS_V2_LABEL_EMBARGO_LOCK"
        or completion.get("status") != "GENERALIZED_BASES_COMPLETE"
        or completion.get("remaining") != 0
        or completion.get("execution_lock_sha256")
        != file_sha256(ANCESTORS / "execution_lock.json")
    ):
        raise RuntimeError("Generalized ancestor base dependency is incomplete")
    audit = _read_json(ANCESTOR_AUDIT)
    if (
        audit.get("status")
        != "GENERALIZED_ARFTR_ANCESTORS_V2_INDEPENDENT_AUDIT_PASS"
        or audit.get("execution_lock_sha256")
        != file_sha256(ANCESTORS / "execution_lock.json")
        or audit.get("new_F_populations_replayed") != 14
        or audit.get("outer_prediction_label_rows_read_during_fits") != 0
        or not audit.get("all_recompositions_exact")
        or not audit.get("all_prediction_label_counterfactual_invariant")
    ):
        raise RuntimeError("Generalized ancestor independent audit has not passed")
    canonical = {
        key for key, value in lock["F_populations"].items() if value["canonical_reuse"]
    }
    required = {receipt["id"] for receipt in plan["F_populations"]}
    paths = [
        PARITY,
        ANCESTORS / "execution_lock.json",
        ANCESTORS / "base_completion.json",
        ANCESTOR_AUDIT,
    ]
    for population_id in sorted(required - canonical):
        receipt_path = ANCESTORS / "populations" / population_id / "receipt.json"
        prediction_path = receipt_path.with_name("predictions.npz")
        receipt = _read_json(receipt_path)
        if (
            receipt.get("status") != "GENERALIZED_F_POPULATION_COMPLETE"
            or receipt.get("population_id") != population_id
            or receipt.get("outer_prediction_labels_read") != 0
            or receipt.get("neural_fits") != 30
            or receipt.get("predictions_sha256") != file_sha256(prediction_path)
        ):
            raise RuntimeError(f"Generalized F population is incomplete: {population_id}")
        paths.extend((receipt_path, prediction_path))
    if len(canonical) != 5 or len(required - canonical) != 14:
        raise RuntimeError("Generalized F reuse/new population inventory changed")
    return paths, canonical


def _fold_priors(
    data: dict[str, np.ndarray], plan: dict[str, Any], fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    outer = plan["outer_plans"][fold]
    if outer["outer_fold"] != fold:
        raise RuntimeError("Fixed ancestry plan fold ordering changed")
    train = np.flatnonzero(data["folds"] != fold)
    held = np.flatnonzero(data["folds"] == fold)
    training_prior = np.full((len(data["labels"]), 3), np.nan)
    ancestry = []
    for block in outer["blocks"]:
        target = np.flatnonzero(np.isin(data["scenarios"], block["target_scenarios"]))
        if not np.all(np.isin(target, train)):
            raise RuntimeError("Correction-training block escapes outer training")
        path = ANCESTORS / "populations" / block["prior_F_id"] / "predictions.npz"
        receipt_path = path.with_name("receipt.json")
        receipt = _read_json(receipt_path)
        with np.load(path, allow_pickle=False) as saved:
            prediction_rows = saved["prediction_rows"]
            indices = np.searchsorted(prediction_rows, target)
            if (
                np.any(indices >= len(prediction_rows))
                or not np.array_equal(prediction_rows[indices], target)
                or not np.array_equal(saved["sample_ids"][indices], data["sample_ids"][target])
            ):
                raise RuntimeError("F(T-minus-B) does not cover its target block")
            training_prior[target] = saved["mean_probabilities"][indices]
        ancestry.append(
            {
                "block": block["block"],
                "target_sample_ids_sha256": canonical_hash(data["sample_ids"][target].tolist()),
                "prior_F_id": block["prior_F_id"],
                "prior_receipt_sha256": file_sha256(receipt_path),
                "prior_predictions_sha256": receipt["predictions_sha256"],
            }
        )
    if not np.isfinite(training_prior[train]).all() or np.isfinite(training_prior[held]).any():
        raise RuntimeError("Training priors are incomplete or leak onto outer-held rows")
    with np.load(ARFTR / f"fold-{fold}/predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["held_rows"], held):
            raise RuntimeError("Retained outer F(T) identity changed")
        outer_prior = saved["seed_probabilities"][5].mean(0)
    if not np.isfinite(outer_prior).all():
        raise RuntimeError("Retained outer F(T) probabilities are invalid")
    return train, held, training_prior[train], ancestry + [
        {
            "outer_F_id": outer["outer_prior_F_id"],
            "outer_fold_predictions_sha256": file_sha256(
                ARFTR / f"fold-{fold}/predictions.npz"
            ),
        }
    ]


def prepare(run: Path) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL)
    if tuple(protocol["arms"]) != tuple(ARM_KEYS) or protocol["observations"]["primary"] != PRIMARY:
        raise RuntimeError("Fixed source/posture arm inventory changed")
    cache_audit = _read_json(CACHE_AUDIT)
    if (
        cache_audit.get("status") != "SOURCE_POSTURE_CACHE_AUDIT_PASS"
        or cache_audit.get("rows") != 4977
        or cache_audit.get("reference_preflight_parity", {}).get("rows") != 16
        or not cache_audit["reference_preflight_parity"].get("feature_exact")
    ):
        raise RuntimeError("Complete source/posture feature-cache audit has not passed")
    data = _data()
    cache = _feature_cache(data)
    plan = fixed_recipe_plan(
        data["labels"],
        data["scenarios"],
        data["folds"],
        data["sample_ids"],
        enforce_canonical_counts=True,
    )
    ancestor_paths, _canonical = _ancestor_population_receipts(plan)
    inputs = {
        PROTOCOL,
        METADATA,
        CACHE_AUDIT,
        CACHE / "execution_lock.json",
        ARFTR / "results/v0001/oof_probabilities.npz",
        Path(__file__),
        ROOT / "src/hac/source_posture_cache.py",
        ROOT / "src/hac/posture_witness.py",
        ROOT / "src/hac/nested_arftr_plan.py",
        ROOT / "experiments/audit_okutama_source_posture_cache.py",
        ROOT / "experiments/audit_okutama_source_posture.py",
        *cache.receipt_paths,
        *(path.with_suffix(".npz") for path in cache.receipt_paths),
        *ancestor_paths,
    }
    for fold in range(5):
        inputs.update(
            {
                ARFTR / f"fold-{fold}/receipt.json",
                ARFTR / f"fold-{fold}/checkpoint.npz",
                ARFTR / f"fold-{fold}/predictions.npz",
            }
        )
    if any(not path.is_file() for path in inputs):
        raise RuntimeError("A source/posture task-fit input is missing")
    lock = {
        "status": "SOURCE_POSTURE_TASK_FIT_LOCKED_OUTER_METRICS_EMBARGOED",
        "study_id": protocol["study_id"],
        "rows": len(data["labels"]),
        "arms": list(ARM_KEYS),
        "primary": PRIMARY,
        "outer_folds": 5,
        "head_fits": 40,
        "task_fits_at_lock": 0,
        "outer_held_labels_read_at_lock": 0,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "protocol_sha256": file_sha256(PROTOCOL),
        "inputs": [_record(path) for path in sorted(inputs, key=str)],
        "cache_shape": list(cache.features.shape),
        "ancestry_counts": plan["counts"],
    }
    run.mkdir(parents=True, exist_ok=True)
    _immutable_json(run / "execution_lock.json", lock)
    validate_lock(run)
    return lock


def validate_lock(run: Path) -> dict[str, Any]:
    lock = _read_json(run / "execution_lock.json")
    if lock.get("status") != "SOURCE_POSTURE_TASK_FIT_LOCKED_OUTER_METRICS_EMBARGOED":
        raise RuntimeError("Source/posture task execution lock is invalid")
    for record in lock["inputs"]:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked source/posture input changed: {record['path']}")
    return lock


def _fit_arm(
    run: Path,
    lock: dict[str, Any],
    data: dict[str, np.ndarray],
    cache: SourcePostureCache,
    fold: int,
    arm: str,
    train: np.ndarray,
    held: np.ndarray,
    training_prior: np.ndarray,
    outer_prior: np.ndarray,
) -> dict[str, np.ndarray]:
    directory = run / f"fold-{fold}" / arm
    receipt_path = directory / "receipt.json"
    prediction_path = directory / "predictions.npz"
    checkpoint_path = directory / "checkpoint.npz"
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if (
            receipt.get("execution_lock_sha256") != file_sha256(
                run / "execution_lock.json"
            )
            or receipt.get("arm") != arm
            or receipt.get("fold") != fold
            or receipt.get("predictions_sha256") != file_sha256(prediction_path)
            or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        ):
            raise RuntimeError("Completed source/posture arm changed")
        with np.load(prediction_path, allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("Partial source/posture arm retained; use a fresh run")
    keys = ARM_KEYS[arm]
    if keys is None:
        probabilities = outer_prior.copy()
        intervention = np.zeros(len(held), dtype=bool)
        delta = np.zeros(len(held), dtype=np.float64)
        train_probabilities = training_prior.copy()
        projection_receipt = None
        optimizer_receipt = None
        checkpoint = {
            "coefficients": np.empty(0),
            "intercept": np.asarray([0.0]),
            "projection_components": np.empty((0, 0)),
        }
        available_train = np.ones(len(train), dtype=bool)
        available_held = np.ones(len(held), dtype=bool)
    else:
        if not keys:
            train_features = np.empty((len(train), 0), dtype=np.float64)
            held_features = np.empty((len(held), 0), dtype=np.float64)
            available_train = np.ones(len(train), dtype=bool)
            available_held = np.ones(len(held), dtype=bool)
            projection = None
        else:
            descriptor, available = cache.descriptor(keys)
            train_features, held_features = descriptor[train], descriptor[held]
            available_train, available_held = available[train], available[held]
            projection = fit_projection(train_features[available_train])
            train_features = apply_projection(train_features, projection)
            held_features = apply_projection(held_features, projection)
        head = fit_posture_head(
            train_features,
            data["labels"][train],
            training_prior,
            available=available_train,
        )
        train_probabilities, _, _ = apply_posture_policy(
            training_prior,
            train_features,
            head,
            available=available_train,
        )
        probabilities, intervention, delta = apply_posture_policy(
            outer_prior,
            held_features,
            head,
            available=available_held,
        )
        projection_receipt = projection.receipt() if projection is not None else None
        optimizer_receipt = head.optimizer_receipt
        checkpoint = {
            "coefficients": head.coefficients,
            "intercept": np.asarray([head.intercept]),
            "cap": np.asarray([head.cap]),
            "dead_zone": np.asarray([head.dead_zone]),
            "l2_weight": np.asarray([head.l2_weight]),
            "projection_feature_mean": projection.feature_mean
            if projection is not None
            else np.empty(0),
            "projection_components": projection.components
            if projection is not None
            else np.empty((0, 0)),
            "projection_projected_mean": projection.projected_mean
            if projection is not None
            else np.empty(0),
            "projection_projected_scale": projection.projected_scale
            if projection is not None
            else np.empty(0),
        }
    _immutable_npz(checkpoint_path, **checkpoint)
    _immutable_npz(
        prediction_path,
        sample_ids=data["sample_ids"][held],
        held_rows=held,
        probabilities=probabilities,
        intervention=intervention,
        continuous_delta=delta,
        available=available_held,
    )
    receipt = {
        "status": "SOURCE_POSTURE_ARM_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "fold": fold,
        "arm": arm,
        "crop_keys": list(keys) if keys is not None else None,
        "train_rows": len(train),
        "held_rows": len(held),
        "available_train_rows": int(available_train.sum()),
        "available_held_rows": int(available_held.sum()),
        "intervention_rows": int(intervention.sum()),
        "training_metrics": probability_metrics(data["labels"][train], train_probabilities),
        "projection": projection_receipt,
        "optimizer": optimizer_receipt,
        "outer_held_labels_read": 0,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "predictions_sha256": file_sha256(prediction_path),
    }
    _immutable_json(receipt_path, receipt)
    return {
        "sample_ids": data["sample_ids"][held],
        "held_rows": held,
        "probabilities": probabilities,
        "intervention": intervention,
        "continuous_delta": delta,
        "available": available_held,
    }


def fit_fold(run: Path, fold: int) -> dict[str, Any]:
    lock = validate_lock(run)
    data = _data()
    cache = _feature_cache(data)
    plan = fixed_recipe_plan(
        data["labels"], data["scenarios"], data["folds"], data["sample_ids"],
        enforce_canonical_counts=True,
    )
    train, held, training_prior, ancestry = _fold_priors(data, plan, fold)
    with np.load(ARFTR / f"fold-{fold}/predictions.npz", allow_pickle=False) as saved:
        outer_prior = saved["seed_probabilities"][5].mean(0)
    outputs = [
        _fit_arm(
            run,
            lock,
            data,
            cache,
            fold,
            arm,
            train,
            held,
            training_prior,
            outer_prior,
        )
        for arm in ARM_KEYS
    ]
    path = run / f"fold-{fold}/predictions.npz"
    _immutable_npz(
        path,
        sample_ids=data["sample_ids"][held],
        held_rows=held,
        arms=np.asarray(tuple(ARM_KEYS)),
        probabilities=np.stack([item["probabilities"] for item in outputs]),
        interventions=np.stack([item["intervention"] for item in outputs]),
        continuous_deltas=np.stack([item["continuous_delta"] for item in outputs]),
        available=np.stack([item["available"] for item in outputs]),
    )
    receipt = {
        "status": "SOURCE_POSTURE_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "fold": fold,
        "train_rows": len(train),
        "held_rows": len(held),
        "outer_held_labels_read": 0,
        "ancestry": ancestry,
        "arm_receipts": {
            arm: file_sha256(run / f"fold-{fold}" / arm / "receipt.json")
            for arm in ARM_KEYS
        },
        "predictions_sha256": file_sha256(path),
    }
    _immutable_json(run / f"fold-{fold}/receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "fold": fold}, indent=2))
    return receipt


def _scenario_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    scenarios: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    candidate_matrices, anchor_matrices = [], []
    for group in groups:
        rows = scenarios == group
        candidate_matrices.append(
            np.bincount(3 * labels[rows] + candidate[rows].argmax(1), minlength=9).reshape(3, 3)
        )
        anchor_matrices.append(
            np.bincount(3 * labels[rows] + anchor[rows].argmax(1), minlength=9).reshape(3, 3)
        )
    candidate_matrices = np.asarray(candidate_matrices)
    anchor_matrices = np.asarray(anchor_matrices)
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples)
    for start in range(0, resamples, 4096):
        count = min(4096, resamples - start)
        sampled = rng.integers(0, len(groups), size=(count, len(groups)))
        c = candidate_matrices[sampled].sum(1)
        a = anchor_matrices[sampled].sum(1)
        cden = c.sum(1) + c.sum(2)
        aden = a.sum(1) + a.sum(2)
        cf1 = np.divide(
            2 * np.diagonal(c, axis1=1, axis2=2),
            cden,
            out=np.zeros_like(cden, dtype=np.float64),
            where=cden > 0,
        )
        af1 = np.divide(
            2 * np.diagonal(a, axis1=1, axis2=2),
            aden,
            out=np.zeros_like(aden, dtype=np.float64),
            where=aden > 0,
        )
        deltas[start : start + count] = cf1.mean(1) - af1.mean(1)
    return {
        "resamples": resamples,
        "seed": seed,
        "delta_mean": float(deltas.mean()),
        "delta_95_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "one_sided_nonpositive_fraction": float(np.mean(deltas <= 0)),
    }


def _exact_scenario_swap(
    labels: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    scenarios: np.ndarray,
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    candidate_matrices, anchor_matrices = [], []
    for group in groups:
        rows = scenarios == group
        candidate_matrices.append(
            np.bincount(
                3 * labels[rows] + candidate[rows].argmax(1), minlength=9
            ).reshape(3, 3)
        )
        anchor_matrices.append(
            np.bincount(
                3 * labels[rows] + anchor[rows].argmax(1), minlength=9
            ).reshape(3, 3)
        )
    candidate_matrices = np.asarray(candidate_matrices)
    anchor_matrices = np.asarray(anchor_matrices)
    assignments = np.arange(2 ** len(groups), dtype=np.uint16)
    choose = ((assignments[:, None] >> np.arange(len(groups))) & 1).astype(bool)
    first = np.where(
        choose[:, :, None, None], candidate_matrices, anchor_matrices
    ).sum(1)
    second = np.where(
        choose[:, :, None, None], anchor_matrices, candidate_matrices
    ).sum(1)

    def macro(matrix: np.ndarray) -> np.ndarray:
        denominator = matrix.sum(1) + matrix.sum(2)
        f1 = np.divide(
            2 * np.diagonal(matrix, axis1=1, axis2=2),
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 0,
        )
        return f1.mean(1)

    deltas = macro(first) - macro(second)
    observed = probability_metrics(labels, candidate)["macro_f1"] - probability_metrics(
        labels, anchor
    )["macro_f1"]
    return {
        "assignments": len(assignments),
        "observed_delta": observed,
        "one_sided_pvalue": float(np.mean(deltas >= observed - 1e-15)),
        "two_sided_pvalue": float(np.mean(np.abs(deltas) >= abs(observed) - 1e-15)),
    }


def summarize(run: Path) -> dict[str, Any]:
    validate_lock(run)
    audit = _read_json(run / "independent_audit.json")
    if (
        audit.get("status") != "SOURCE_POSTURE_INDEPENDENT_REPLAY_PASS"
        or audit.get("execution_lock_sha256") != file_sha256(run / "execution_lock.json")
        or not audit.get("all_probabilities_exact")
    ):
        raise RuntimeError("Independent source/posture replay has not passed")
    data = _data()
    probabilities = np.full((len(ARM_KEYS), len(data["labels"]), 3), np.nan)
    interventions = np.zeros((len(ARM_KEYS), len(data["labels"])), dtype=bool)
    for fold in range(5):
        receipt = _read_json(run / f"fold-{fold}/receipt.json")
        path = run / f"fold-{fold}/predictions.npz"
        if (
            receipt.get("status") != "SOURCE_POSTURE_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED"
            or receipt.get("predictions_sha256") != file_sha256(path)
        ):
            raise RuntimeError("Source/posture fold is incomplete")
        with np.load(path, allow_pickle=False) as saved:
            held = saved["held_rows"]
            probabilities[:, held] = saved["probabilities"]
            interventions[:, held] = saved["interventions"]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Source/posture OOF prediction coverage is incomplete")
    with np.load(ARFTR / "results/v0001/oof_probabilities.npz", allow_pickle=False) as saved:
        anchor = saved["mean_probabilities"][5]
    if not np.array_equal(probabilities[0], anchor):
        raise RuntimeError("A0 exact-retain arm differs from retained ARFTR")
    labels = data["labels"]
    anchor_correct = anchor.argmax(1) == labels
    results = {}
    for index, arm in enumerate(ARM_KEYS):
        candidate = probabilities[index]
        correct = candidate.argmax(1) == labels
        per_fold_net = []
        for fold in range(5):
            rows = data["folds"] == fold
            rescues = int((~anchor_correct[rows] & correct[rows]).sum())
            harms = int((anchor_correct[rows] & ~correct[rows]).sum())
            per_fold_net.append(rescues - harms)
        results[arm] = {
            "metrics": probability_metrics(labels, candidate),
            "rescues": int((~anchor_correct & correct).sum()),
            "harms": int((anchor_correct & ~correct).sum()),
            "net_corrections": int((~anchor_correct & correct).sum())
            - int((anchor_correct & ~correct).sum()),
            "interventions": int(interventions[index].sum()),
            "per_fold_net_corrections": per_fold_net,
        }
    protocol = _read_json(PROTOCOL)
    primary = results[PRIMARY]
    anchor_metrics = results["A0_retain"]["metrics"]
    bootstrap = _scenario_bootstrap(
        labels,
        probabilities[list(ARM_KEYS).index(PRIMARY)],
        anchor,
        data["scenarios"],
        resamples=100000,
        seed=20260913,
    )
    exact_swap = _exact_scenario_swap(
        labels,
        probabilities[list(ARM_KEYS).index(PRIMARY)],
        anchor,
        data["scenarios"],
    )
    class_delta = np.asarray(primary["metrics"]["per_class_f1"]) - np.asarray(
        anchor_metrics["per_class_f1"]
    )
    gates = {
        "macro_f1": primary["metrics"]["macro_f1"] >= protocol["go_gate"]["macro_f1_min"],
        "net_corrections": primary["net_corrections"] >= protocol["go_gate"]["net_corrections_min"],
        "positive_each_outer_fold": min(primary["per_fold_net_corrections"]) > 0,
        "scenario_bootstrap_lower_positive": bootstrap["delta_95_interval"][0] > 0,
        "class_f1_loss_cap": float(class_delta.min()) >= -protocol["go_gate"]["class_f1_max_loss"],
        "nll_not_worse": primary["metrics"]["nll"] <= anchor_metrics["nll"],
        "brier_not_worse": primary["metrics"]["brier"] <= anchor_metrics["brier"],
        "independent_replay": True,
    }
    result_dir = run / "results/v0001"
    _immutable_npz(
        result_dir / "oof_probabilities.npz",
        sample_ids=data["sample_ids"],
        labels=labels,
        scenarios=data["scenarios"],
        folds=data["folds"],
        arms=np.asarray(tuple(ARM_KEYS)),
        probabilities=probabilities,
        interventions=interventions,
    )
    summary = {
        "status": "SOURCE_POSTURE_FIXED_PRIMARY_COMPLETE",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "independent_audit_sha256": file_sha256(run / "independent_audit.json"),
        "anchor": anchor_metrics,
        "results": results,
        "primary": PRIMARY,
        "primary_bootstrap": bootstrap,
        "primary_exact_scenario_swap": exact_swap,
        "primary_class_f1_delta": class_delta.tolist(),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "post_hoc_arm_selection": False,
        "stop_rule": protocol["stopping"],
        "artifacts": {"oof_probabilities.npz": _record(result_dir / "oof_probabilities.npz")},
    }
    _immutable_json(result_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "primary_macro_f1": primary["metrics"]["macro_f1"],
                "gates": gates,
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("prepare", "fit-fold", "summarize"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    if args.stage == "prepare":
        lock = prepare(run)
        print(json.dumps({"status": lock["status"], "head_fits": lock["head_fits"]}, indent=2))
    elif args.stage == "fit-fold":
        if args.fold is None:
            raise ValueError("fit-fold requires --fold")
        fit_fold(run, args.fold)
    else:
        summarize(run)


if __name__ == "__main__":
    main()
