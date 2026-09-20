"""Strictly nested, provenance-keyed P6 base predictions for learned actor memory.

No historical OOF probability is a fitting input. A cache is a fitted pipeline for
one exact population, including regularization selection inside that population.
Only predictions outside its training scenarios may enter a meta learner.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from hac.video_fusion import decode_factorized_probabilities


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def probability_metrics(labels: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if p.shape != (len(labels), 3) or not np.isfinite(p).all():
        raise ValueError("Probabilities must be aligned, finite three-class vectors")
    if np.min(p) < 0 or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError("Invalid probability simplex")
    matrix = np.bincount(3 * labels + p.argmax(1), minlength=9).reshape(3, 3)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    return {
        "rows": len(labels),
        "macro_f1": float(f1.mean()),
        "accuracy": float(matrix.trace() / len(labels)),
        "nll": float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier": float(np.square(p - np.eye(3)[labels]).sum(1).mean()),
        "per_class_f1": f1.tolist(),
        "confusion": matrix.tolist(),
    }


def group_splits(
    labels: np.ndarray,
    scenarios: np.ndarray,
    population: np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    population = np.asarray(population, dtype=np.int64)
    if not np.array_equal(population, np.unique(population)):
        raise ValueError("Training populations must be unique and sorted")
    groups = len(np.unique(scenarios[population]))
    if groups < 2:
        raise ValueError("Nested selection requires at least two scenario groups")
    splitter = StratifiedGroupKFold(n_splits=min(n_splits, groups), shuffle=True, random_state=seed)
    result = []
    for fit, held in splitter.split(
        np.zeros(len(population)), labels[population], scenarios[population]
    ):
        train, validation = population[fit], population[held]
        if set(scenarios[train]) & set(scenarios[validation]):
            raise RuntimeError("Scenario leakage in nested splitting")
        if not np.array_equal(np.unique(labels[train]), np.arange(3)):
            raise RuntimeError("Nested training partition is missing a target class")
        result.append((train, validation))
    if not np.array_equal(np.sort(np.concatenate([v for _, v in result])), population):
        raise RuntimeError("Cross-fit coverage is incomplete or duplicated")
    return result


def training_populations(labels, scenarios, folds) -> dict[str, np.ndarray]:
    """Enumerate every base training set needed by outer and inner meta fitting."""
    result = {}

    def add(rows):
        result[canonical_hash(rows.tolist())] = rows

    for fold in np.unique(folds):
        outer_train = np.flatnonzero(folds != fold)
        add(outer_train)
        for inner_train, _ in group_splits(labels, scenarios, outer_train):
            add(inner_train)
            for base_train, _ in group_splits(labels, scenarios, inner_train):
                add(base_train)
    return result


def _fit(x: np.ndarray, y: np.ndarray, config: dict, c: float, *, binary: bool = False):
    expected = np.arange(2 if binary else 3)
    if not np.array_equal(np.unique(y), expected):
        raise RuntimeError("Base fit is missing classes")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(x)
    model = LogisticRegression(
        C=c,
        solver=config["solver"],
        class_weight=config["class_weight"],
        max_iter=config["max_iter"],
        tol=config["tolerance"],
        random_state=42,
    )
    model.fit(standardized, y)
    if int(model.n_iter_.max()) >= config["max_iter"]:
        raise RuntimeError("Base model failed to converge within the locked budget")
    return scaler, model


def _predict(fit, features):
    scaler, model = fit
    return model.predict_proba(scaler.transform(features))


def _decode(posture, motion):
    # Historical P5 subtracts in the predictor's dtype, then promotes both
    # probabilities to float64 before factorized products. Reordering that
    # subtraction or multiplying in float32 changes the locked reference.
    return decode_factorized_probabilities(1 - posture, motion)


def _checkpoint(fit, prefix):
    scaler, model = fit
    return {
        f"{prefix}_{key}": value
        for key, value in {
            "coefficients": model.coef_,
            "intercept": model.intercept_,
            "classes": model.classes_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "scaler_variance": scaler.var_,
        }.items()
    }


class NestedBaseCache:
    def __init__(
        self,
        cache_dir: Path,
        features: dict[str, np.ndarray],
        labels: np.ndarray,
        scenarios: np.ndarray,
        sample_ids: np.ndarray,
        config: dict,
        feature_sha256: str,
    ):
        self.cache_dir = Path(cache_dir)
        self.features, self.labels, self.scenarios = features, labels, scenarios
        self.sample_ids, self.config = sample_ids, config
        rows = len(labels)
        if labels.shape != (rows,) or scenarios.shape != (rows,) or sample_ids.shape != (rows,):
            raise ValueError("Base metadata must have aligned one-dimensional rows")
        if len(np.unique(sample_ids)) != rows or not np.array_equal(
            np.unique(labels), np.arange(3)
        ):
            raise ValueError("Base metadata needs unique IDs and all three classes")
        for name in ("base_long_vjepa", "base_dual_scale", "base_ostm_posture", "base_ostm_motion"):
            values = features[name]
            if values.ndim != 2 or len(values) != rows or not np.isfinite(values).all():
                raise ValueError(f"Invalid aligned finite base features: {name}")
        self.identity = {
            "feature_sha256": feature_sha256,
            "code_sha256": file_sha256(Path(__file__)),
            "decoder_code_sha256": file_sha256(Path(__file__).with_name("video_fusion.py")),
            "probe": config,
            "all_sample_ids_sha256": canonical_hash(sample_ids.tolist()),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def request(self, train: np.ndarray) -> dict:
        train = np.asarray(train, dtype=np.int64)
        if not np.array_equal(train, np.unique(train)):
            raise ValueError("Unsorted or duplicate base training IDs")
        if not len(train) or train.min() < 0 or train.max() >= len(self.labels):
            raise ValueError("Base training rows are empty or outside the population")
        # No label outside train enters this request or any fitting operation.
        return {
            **self.identity,
            "train_sample_ids_sha256": canonical_hash(self.sample_ids[train].tolist()),
            "train_labels_sha256": canonical_hash(self.labels[train].tolist()),
            "train_scenarios_sha256": canonical_hash(self.scenarios[train].tolist()),
            "train_rows": train.tolist(),
            "train_scenarios": sorted(set(self.scenarios[train].tolist())),
        }

    def directory(self, train):
        return self.cache_dir / canonical_hash(self.request(train))

    def load(self, train) -> tuple[np.ndarray, dict]:
        directory = self.directory(train)
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["request"] != self.request(train):
            raise RuntimeError("Base cache ancestry mismatch")
        path = directory / "predictions.npz"
        if file_sha256(path) != receipt["predictions_sha256"]:
            raise RuntimeError("Base prediction cache bytes changed")
        if file_sha256(directory / "checkpoint.npz") != receipt["checkpoint_sha256"]:
            raise RuntimeError("Base checkpoint bytes changed")
        with np.load(path, allow_pickle=False) as saved:
            result = saved["probabilities"]
        if result.shape != (len(self.labels), 3) or not np.isfinite(result).all():
            raise RuntimeError("Malformed base probability cache")
        return result, receipt

    def _direct(self, x, train, splits):
        candidates = []
        for c in self.config["C_values"]:
            oof = np.full((len(self.labels), 3), np.nan)
            for fit_rows, held in splits:
                fit = _fit(x[fit_rows], self.labels[fit_rows], self.config, c)
                oof[held] = _predict(fit, x[held])
            candidates.append(
                {"C": c, "metrics": probability_metrics(self.labels[train], oof[train])}
            )
        selected = min(
            candidates, key=lambda v: (-v["metrics"]["macro_f1"], v["metrics"]["nll"], v["C"])
        )
        fit = _fit(x[train], self.labels[train], self.config, selected["C"])
        probabilities = _predict(fit, x)
        outside = np.setdiff1d(np.arange(len(self.labels)), train)
        # Binary BLAS reductions can depend on matrix row count. Match the
        # historical outer-held inference batch exactly, not all-row inference.
        if len(outside):
            probabilities[outside] = _predict(fit, x[outside])
        return probabilities, {"selected": selected, "candidates": candidates}, fit

    def _factorized(self, posture_x, motion_x, train, splits):
        sitting_predictions, motion_predictions = {}, {}
        for c in self.config["C_values"]:
            sitting_predictions[c] = np.full(len(self.labels), np.nan)
            motion_predictions[c] = np.full(len(self.labels), np.nan)
            for fit_rows, held in splits:
                upright = fit_rows[self.labels[fit_rows] != 0]
                posture = _fit(
                    posture_x[fit_rows],
                    (self.labels[fit_rows] != 0).astype(int),
                    self.config,
                    c,
                    binary=True,
                )
                motion = _fit(
                    motion_x[upright],
                    (self.labels[upright] == 2).astype(int),
                    self.config,
                    c,
                    binary=True,
                )
                # Compute the complement BEFORE assignment to the float64 OOF
                # buffer, exactly as in original P5 inner candidate selection.
                sitting_predictions[c][held] = 1 - _predict(posture, posture_x[held])[:, 1]
                motion_predictions[c][held] = _predict(motion, motion_x[held])[:, 1]
        candidates = []
        for pc in self.config["C_values"]:
            for mc in self.config["C_values"]:
                decoded = decode_factorized_probabilities(
                    sitting_predictions[pc][train], motion_predictions[mc][train]
                )
                candidates.append(
                    {
                        "posture_C": pc,
                        "motion_C": mc,
                        "metrics": probability_metrics(self.labels[train], decoded),
                    }
                )
        selected = min(
            candidates,
            key=lambda v: (
                -v["metrics"]["macro_f1"],
                v["metrics"]["nll"],
                v["posture_C"],
                v["motion_C"],
            ),
        )
        upright = train[self.labels[train] != 0]
        posture = _fit(
            posture_x[train],
            (self.labels[train] != 0).astype(int),
            self.config,
            selected["posture_C"],
            binary=True,
        )
        motion = _fit(
            motion_x[upright],
            (self.labels[upright] == 2).astype(int),
            self.config,
            selected["motion_C"],
            binary=True,
        )
        probabilities = _decode(
            _predict(posture, posture_x)[:, 1], _predict(motion, motion_x)[:, 1]
        )
        outside = np.setdiff1d(np.arange(len(self.labels)), train)
        if len(outside):
            probabilities[outside] = _decode(
                _predict(posture, posture_x[outside])[:, 1],
                _predict(motion, motion_x[outside])[:, 1],
            )
        return probabilities, {"selected": selected, "candidates": candidates}, (posture, motion)

    def prepare(self, train) -> dict:
        directory = self.directory(train)
        if (directory / "receipt.json").exists():
            return self.load(train)[1]
        start = time.perf_counter()
        splits = group_splits(
            self.labels,
            self.scenarios,
            train,
            n_splits=self.config["inner_folds"],
            seed=self.config["inner_seed"],
        )
        directory.mkdir(parents=True, exist_ok=True)
        components, checkpoints, selections = [], {}, {}
        for name in ("base_long_vjepa", "base_dual_scale"):
            probabilities, selection, fit = self._direct(self.features[name], train, splits)
            components.append(probabilities)
            checkpoints.update(_checkpoint(fit, name))
            selections[name] = selection
        probabilities, selection, fits = self._factorized(
            self.features["base_ostm_posture"], self.features["base_ostm_motion"], train, splits
        )
        components.append(probabilities)
        checkpoints.update(_checkpoint(fits[0], "base_ostm_posture"))
        checkpoints.update(_checkpoint(fits[1], "base_ostm_motion"))
        selections["base_ostm_factorized"] = selection
        combined = np.sum(np.asarray(components, dtype=np.float64), axis=0) / 3
        combined /= combined.sum(1, keepdims=True)
        np.savez_compressed(
            directory / "predictions.npz",
            probabilities=combined,
            components=np.stack(components, axis=1),
        )
        np.savez_compressed(directory / "checkpoint.npz", **checkpoints)
        receipt = {
            "request": self.request(train),
            "selection": selections,
            "selection_splits": [{"train": t.tolist(), "held": h.tolist()} for t, h in splits],
            "estimator_fits": 4 * (len(self.config["C_values"]) * len(splits) + 1),
            "seconds": time.perf_counter() - start,
            "prediction_batch_policy": "Final complement-of-train rows are predicted in one sorted held batch, matching original P6 outer inference. Training-row predictions never enter meta fitting.",
            "factorized_precision_policy": "Complement posture prediction in its native dtype before float64 OOF assignment; use original float64 P5 decoder for inner selection and final products.",
            "predictions_sha256": file_sha256(directory / "predictions.npz"),
            "checkpoint_sha256": file_sha256(directory / "checkpoint.npz"),
        }
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return receipt

    def held_predictions(self, train, held):
        if set(self.scenarios[train]) & set(self.scenarios[held]):
            raise RuntimeError("Refusing in-scenario base predictions for a meta learner")
        return self.load(train)[0][held]

    def meta_probabilities(self, population) -> tuple[np.ndarray, list[dict]]:
        """Cross-fit population and predict its complement, with row-level ancestry."""
        result = np.full((len(self.labels), 3), np.nan)
        ancestry = []
        for train, held in group_splits(self.labels, self.scenarios, population):
            result[held] = self.held_predictions(train, held)
            ancestry.append(
                {"predicted_rows": held.tolist(), "base_cache": self.directory(train).name}
            )
        outside = np.setdiff1d(np.arange(len(self.labels)), population)
        result[outside] = self.held_predictions(population, outside)
        ancestry.append(
            {"predicted_rows": outside.tolist(), "base_cache": self.directory(population).name}
        )
        if not np.isfinite(result).all():
            raise RuntimeError("Meta-prediction coverage failed")
        return result, ancestry
