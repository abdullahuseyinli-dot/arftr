"""Independent, no-fitting audit of the fixed crossing-event verifier matrix.

Feature, outcome, weight, objective and decision algebra is deliberately repeated
here instead of imported from the producer.  This checks the fitted solution and
its provenance; it does not optimize, refit or choose a scientific setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
READERS = ("coarse_residual", "paired_spatial", "paired_spatial_temporal")
VARIANTS = ("V1_all_event_ridge", "V2_crossing_ridge", "V3_crossing_multinomial")
PDI = ROOT / ".runs/research_20260919/paired_detail_innovation_v1_full"
ANCHOR = ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
METADATA = ROOT / ".runs/research_20260908/evidence_memory/data/memory_data.npz"
ANCESTORS = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/inner_ancestors"
PAIR_ORDER = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
FEATURE_NAMES = (
    [f"log_anchor_{i}" for i in range(3)]
    + ["anchor_margin", "anchor_entropy"]
    + [f"coarse_scaffold_norm_{i}" for i in range(4)]
    + [f"quality_{i}" for i in range(6)]
    + ["available_0", "posture_delta", "motion_delta"]
    + [f"logit_delta_{i}" for i in range(3)]
    + ["candidate_margin", "crosses_class_boundary"]
    + [f"predicted_pair_{a}_to_{b}" for a, b in PAIR_ORDER]
    + ["action_posture", "action_motion", "action_both"]
    + ["rival_log_margin_before", "rival_log_margin_after", "crossing_action_count"]
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, explanation: str) -> None:
    if not condition:
        raise RuntimeError(explanation)


def _same(actual: np.ndarray, expected: np.ndarray, name: str, *, atol=1e-10) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(actual.shape == expected.shape, f"{name}: shape differs")
    _require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{name}: nonfinite")
    difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    _require(np.allclose(actual, expected, atol=atol, rtol=1e-10), f"{name}: delta {difference}")
    return difference


def reconstruct_features(
    anchor: np.ndarray,
    actions: np.ndarray,
    correction: np.ndarray,
    scaffold: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    """Recreate all 35 fixed features from primitive saved PDI quantities."""
    anchor, actions, correction, scaffold, quality = [
        np.asarray(v, dtype=np.float64) for v in (anchor, actions, correction, scaffold, quality)
    ]
    n = len(anchor)
    _require(anchor.shape == (n, 3) and actions.shape == (n, 4, 3), "bad probability shape")
    _require(correction.shape == (n, 2) and scaffold.shape == (n, 4), "bad producer feature shape")
    _require(quality.shape == (n, 6), "bad quality shape")
    logp = np.log(np.clip(anchor, 1e-12, 1.0))
    ordered = np.sort(anchor, axis=1)
    old = anchor.argmax(1)
    new = actions[:, 1:].argmax(2)
    crossing = new != old[:, None]
    common = np.column_stack((
        logp, ordered[:, -1] - ordered[:, -2], -(anchor * logp).sum(1),
        scaffold, quality, np.ones(n),
    ))
    result = []
    for a, mask in enumerate(((1, 0), (0, 1), (1, 1))):
        candidate = actions[:, a + 1]
        logq = np.log(np.clip(candidate, 1e-12, 1.0))
        sorted_q = np.sort(candidate, axis=1)
        original = np.column_stack((
            common, correction * np.asarray(mask), logq - logp,
            sorted_q[:, -1] - sorted_q[:, -2], crossing[:, a].astype(float),
        ))
        pair = np.column_stack([(old == left) & (new[:, a] == right) for left, right in PAIR_ORDER])
        action_id = np.broadcast_to(np.eye(3)[a], (n, 3))
        row = np.arange(n)
        margins = np.column_stack((
            logp[row, new[:, a]] - logp[row, old],
            logq[row, new[:, a]] - logq[row, old], crossing.sum(1),
        ))
        result.append(np.column_stack((original, pair, action_id, margins)))
    assembled = np.stack(result, axis=1)
    _require(assembled.shape == (n, 3, 35), "feature assembly shape")
    return assembled


def reconstruct_targets(anchor: np.ndarray, actions: np.ndarray, labels: np.ndarray) -> dict:
    old = anchor.argmax(1)
    new = actions[:, 1:].argmax(2)
    y = np.asarray(labels, dtype=np.int64)
    crossing = new != old[:, None]
    rescue = (old != y)[:, None] & (new == y[:, None])
    harm = (old == y)[:, None] & (new != y[:, None])
    both_wrong = crossing & (old != y)[:, None] & (new != y[:, None])
    _require(np.array_equal(rescue | harm | both_wrong, crossing), "crossing outcomes not exhaustive")
    event = np.full(crossing.shape, -1, dtype=np.int64)
    event[rescue], event[harm], event[both_wrong] = 0, 1, 2
    utility = rescue.astype(np.float64) - 2.0 * harm
    crossing_weights = crossing / np.maximum(crossing.sum(1), 1)[:, None]
    return {"crossing": crossing, "events": event, "utility": utility,
            "all_weights": np.full(crossing.shape, 1 / 3), "crossing_weights": crossing_weights}


def reconstruct_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(features, dtype=np.float64).reshape(-1, 35)
    _require(np.isfinite(flat).all(), "nonfinite training features")
    mean = flat.mean(0)
    scale = flat.std(0)
    scale[scale == 0] = 1.0
    return mean, scale


def solution_checks(model: dict, features: np.ndarray, targets: dict) -> dict:
    """Evaluate frozen normal equations / gradient; never solve or optimize."""
    variant = str(np.asarray(model["variant"]).item())
    mean, scale = reconstruct_scaler(features)
    _same(model["scaler_mean"], mean, "scaler mean")
    _same(model["scaler_scale"], scale, "scaler scale")
    x = ((features - mean) / scale).reshape(-1, 35)
    coef = np.asarray(model["coefficients"], dtype=np.float64)
    intercept = np.asarray(model["intercepts"], dtype=np.float64)
    crossing = targets["crossing"].reshape(-1)
    if variant in VARIANTS[:2]:
        _require(coef.shape == (1, 35) and intercept.shape == (1,), "ridge parameter shape")
        selected = np.ones(len(x), dtype=bool) if variant == VARIANTS[0] else crossing
        weight = targets["all_weights" if variant == VARIANTS[0] else "crossing_weights"].reshape(-1)[selected]
        xx = x[selected]
        yy = targets["utility"].reshape(-1)[selected]
        residual = xx @ coef[0] + intercept[0] - yy
        gradient = np.r_[xx.T @ (weight * residual) + 10.0 * coef[0], np.sum(weight * residual)]
        reference = max(1.0, float(np.max(np.abs(xx.T @ (weight * yy)))))
        relative = float(np.max(np.abs(gradient))) / reference
        _require(relative <= 1e-8, f"ridge normal equation residual {relative}")
        objective = float(np.dot(weight, residual**2) + 10.0 * np.sum(coef**2))
        return {"objective": objective, "gradient_linf": float(2 * np.max(np.abs(gradient))),
                "normal_equation_relative_residual": relative, "selected_events": int(selected.sum()),
                "weight_sum": float(weight.sum())}
    _require(variant == VARIANTS[2], f"unknown variant {variant}")
    _require(coef.shape == (3, 35) and intercept.shape == (3,), "softmax parameter shape")
    xx = x[crossing]
    event = targets["events"].reshape(-1)[crossing]
    weights = targets["crossing_weights"].reshape(-1)[crossing]
    logits = xx @ coef.T + intercept
    shifted = logits - logits.max(1, keepdims=True)
    logprob = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    prob = np.exp(logprob)
    error = prob - np.eye(3)[event]
    weighted_error = error * (weights / weights.sum())[:, None]
    gradient_w = weighted_error.T @ xx + 0.05 * coef
    gradient_b = weighted_error.sum(0) + 0.0001 * intercept
    gradient = np.r_[gradient_w.ravel(), gradient_b]
    maximum = float(np.max(np.abs(gradient)))
    _require(maximum <= 1.001e-7, f"softmax gradient did not converge: {maximum}")
    objective = float(-np.dot(weights, logprob[np.arange(len(event)), event]) / weights.sum()
                      + 0.025 * np.sum(coef**2) + 0.00005 * np.sum(intercept**2))
    return {"objective": objective, "gradient_linf": maximum, "selected_events": int(crossing.sum()),
            "weight_sum": float(weights.sum())}


def reconstruct_scores(model: dict, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = (features - model["scaler_mean"]) / model["scaler_scale"]
    values = transformed @ model["coefficients"].T + model["intercepts"]
    if str(np.asarray(model["variant"]).item()) == VARIANTS[2]:
        shifted = values - values.max(2, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(2, keepdims=True)
        return probabilities[:, :, 0] - 2 * probabilities[:, :, 1], probabilities
    return values[:, :, 0], np.empty(0, dtype=np.float64)


def reconstruct_nll(saved_verifier: dict, original_features: np.ndarray) -> np.ndarray:
    transformed = (original_features - saved_verifier["scaler_mean"]) / saved_verifier["scaler_scale"]
    return np.column_stack([
        transformed[:, a] @ saved_verifier[f"nll_coef_{a}"] + saved_verifier[f"nll_intercept_{a}"][0]
        for a in range(3)
    ])


def reconstruct_decision(anchor, actions, features, utility, nll_scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = np.asarray(anchor, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    _require(np.isfinite(anchor).all() and np.all(anchor >= 0)
             and np.allclose(anchor.sum(1), 1, atol=1e-6, rtol=0), "invalid anchor")
    invalid = (~np.isfinite(features).all((1, 2)) | ~np.isfinite(actions).all((1, 2))
               | (actions < 0).any((1, 2)) | ~np.isclose(actions.sum(2), 1, atol=1e-6, rtol=0).all(1)
               | ~np.isfinite(utility).all(1) | ~np.isfinite(nll_scores).all(1)
               | (features[:, :, 15] <= 0).any(1))
    crossing = actions[:, 1:].argmax(2) != anchor.argmax(1)[:, None]
    eligible = crossing & (utility > 0) & (nll_scores > 0) & ~invalid[:, None]
    scores = np.where(eligible, utility, -np.inf)
    chosen = np.zeros(len(anchor), dtype=np.int64)
    active = eligible.any(1)
    chosen[active] = 1 + scores[active].argmax(1)
    output = anchor.copy()
    output[active] = actions[np.flatnonzero(active), chosen[active]]
    return output, chosen, invalid


def _check_fold_sources(reader: str, fold: int, canonical: dict, quality: np.ndarray) -> dict:
    folder = PDI / reader / f"fold-{fold}"
    inner = _npz(folder / "inner_oof_verifier_inputs.npz")
    outer = _npz(folder / "predictions.npz")
    train = np.flatnonzero(canonical["folds"] != fold)
    held = np.flatnonzero(canonical["folds"] == fold)
    _require(np.array_equal(inner["rows"], train), "inner rows differ from outer training population")
    _require(np.array_equal(outer["held_rows"], held), "outer row identity mismatch")
    _require(not np.intersect1d(train, held).size, "train/held row overlap")
    scenarios = canonical["scenarios"].astype(str)
    _require(not set(scenarios[train]) & set(scenarios[held]), "outer scenario overlap")
    _require(np.array_equal(inner["sample_ids"].astype(str), canonical["sample_ids"][train].astype(str)), "inner sample IDs differ")
    _require(np.array_equal(outer["sample_ids"].astype(str), canonical["sample_ids"][held].astype(str)), "outer sample IDs differ")
    _require(np.array_equal(outer["anchor_probabilities"], canonical["anchor"][held]), "anchor differs from retained ARFTR")
    _same(inner["quality"], quality[train], "inner quality")
    for group in range(5):
        directory = folder / f"inner-{group}"
        receipt = _json(directory / "receipt.json")
        _require(_hash(directory / "predictions.npz") == receipt["predictions_sha256"], "inner producer hash changed")
        saved = _npz(directory / "predictions.npz")
        mask = inner["inner_group"] == group
        _require(np.array_equal(saved["held_rows"], train[mask]), "inner producer held rows mismatch")
        held_scenarios = set(scenarios[train[mask]])
        fitting_scenarios = set(scenarios[train[~mask]])
        ancestry = receipt["ancestry"]
        _require(set(ancestry["held_scenarios"]) == held_scenarios, "inner held scenarios differ")
        _require(set(ancestry["train_scenarios"]) == fitting_scenarios, "inner fitting scenarios differ")
        _require(not fitting_scenarios & (held_scenarios | set(scenarios[held])), "inner ancestry contamination")
        population = ANCESTORS / "populations" / ancestry["population_id"] / "receipt.json"
        _require(_hash(population) == ancestry["population_receipt_sha256"], "base population receipt changed")
        for name in ("actions", "correction", "scaffold_norms", "anchor_probabilities"):
            _same(inner[name][mask], saved[name], f"held producer input {name}")
    train_features = reconstruct_features(inner["anchor_probabilities"], inner["actions"], inner["correction"], inner["scaffold_norms"], quality[train])
    outer_features = reconstruct_features(outer["anchor_probabilities"], outer["actions"], outer["correction"], outer["scaffold_norms"], quality[held])
    _same(train_features[:, :, :23], inner["verifier_features"], "original inner features")
    _same(outer_features[:, :, :23], outer["verifier_features"], "original outer features")
    targets = reconstruct_targets(inner["anchor_probabilities"], inner["actions"], canonical["labels"][train])
    return {"folder": folder, "inner": inner, "outer": outer, "train": train, "held": held,
            "train_features": train_features, "outer_features": outer_features, "targets": targets}


def audit(run: Path) -> dict:
    """Check the complete 45-fit matrix without any estimator fitting."""
    run = Path(run).resolve()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Only the producer's hash validator is reused. All scientific algebra is local.
    from experiments.run_okutama_crossing_event_verifier import validate_lock

    validate_lock(run)
    for name in ("preflight.json", "ancestry_audit.json"):
        receipt = _json(run / name)
        _require("PASS" in str(receipt.get("status", "")), f"{name} is not PASS")
        _require(not receipt.get("failures"), f"{name} reports failures")
    canonical = _npz(ANCHOR)
    names = canonical["arms"].astype(str).tolist()
    canonical["anchor"] = canonical["mean_probabilities"][names.index("r5_arftr_full")]
    metadata = _npz(METADATA)
    _require(np.array_equal(metadata["sample_ids"].astype(str), canonical["sample_ids"].astype(str)), "quality identity mismatch")
    quality = metadata["quality"].astype(np.float32)
    reports = []
    audited_artifacts = []
    maxima = {"features": 0.0, "utility": 0.0, "nll_scores": 0.0, "probabilities": 0.0}
    for reader in READERS:
        for fold in range(5):
            source = _check_fold_sources(reader, fold, canonical, quality)
            original_verifier = _npz(source["folder"] / "verifier.npz")
            nll = reconstruct_nll(original_verifier, source["outer_features"][:, :, :23])
            for variant in VARIANTS:
                directory = run / reader / f"fold-{fold}" / variant
                receipt = _json(directory / "receipt.json")
                model = _npz(directory / "model.npz")
                saved = _npz(directory / "predictions.npz")
                _require(receipt["reader"] == reader and receipt["variant"] == variant
                         and receipt["outer_fold"] == fold, "gate identity mismatch")
                _require(receipt["train_rows"] == len(source["train"])
                         and receipt["held_rows"] == len(source["held"]), "gate population count mismatch")
                for name in ("model", "predictions"):
                    _require(_hash(directory / f"{name}.npz") == receipt[f"{name}_sha256"], f"gate {name} hash changed")
                for filename in ("receipt.json", "model.npz", "predictions.npz"):
                    path = (directory / filename).resolve()
                    audited_artifacts.append({"path": path.relative_to(ROOT).as_posix(),
                                              "bytes": path.stat().st_size, "sha256": _hash(path)})
                for name, filename in (("inner", "inner_oof_verifier_inputs.npz"), ("outer", "predictions.npz"), ("verifier", "verifier.npz")):
                    _require(_hash(source["folder"] / filename) == receipt[f"source_{name}_sha256"], f"source {name} hash changed")
                _require(int(np.asarray(model["format_version"]).item()) == 1, "model schema version")
                _require(str(np.asarray(model["variant"]).item()) == variant, "model variant differs")
                _require(model["feature_names"].astype(str).tolist() == FEATURE_NAMES, "model feature schema differs")
                _require(receipt["feature_names"] == FEATURE_NAMES, "receipt feature schema differs")
                _require(np.array_equal(saved["held_rows"], source["held"]), "saved prediction row identity")
                _require(np.array_equal(saved["sample_ids"].astype(str), canonical["sample_ids"][source["held"]].astype(str)), "prediction sample identity")
                solution = solution_checks(model, source["train_features"], source["targets"])
                fit = receipt["fit_receipt"]
                _require(bool(fit.get("converged")), "fit receipt is not converged")
                if variant == VARIANTS[2]:
                    _require(int(fit["optimizer_iterations"]) <= 1000, "softmax iteration budget exceeded")
                _same(np.asarray(fit["objective"]), np.asarray(solution["objective"]), "fit objective", atol=1e-8)
                utility, probabilities = reconstruct_scores(model, source["outer_features"])
                output, choices, invalid = reconstruct_decision(source["outer"]["anchor_probabilities"], source["outer"]["actions"], source["outer_features"], utility, nll)
                for name, expected in (("features", source["outer_features"]), ("utility", utility), ("nll_scores", nll), ("probabilities", output)):
                    maxima[name] = max(maxima[name], _same(saved[name], expected, name))
                _same(saved["event_probabilities"], probabilities, "event probabilities")
                _require(np.array_equal(saved["choices"], choices), "decision replay changed")
                _require(np.array_equal(saved["invalid_rows"], invalid), "invalid-row replay changed")
                _require(np.array_equal(saved["probabilities"][choices == 0], source["outer"]["anchor_probabilities"][choices == 0]), "retain output is not bit-exact")
                reports.append({"reader": reader, "outer_fold": fold, "variant": variant,
                                "solution": solution, "rows": len(output),
                                "interventions": int(np.count_nonzero(choices)), "exact_choices": True,
                                "exact_retention": True, "invalid_rows": int(invalid.sum())})
    _require(len(reports) == 45, "incomplete gate matrix")
    result = {"status": "CROSSING_EVENT_INDEPENDENT_NO_FIT_AUDIT_PASS", "gate_replays": len(reports),
              "scientific_fits": 0, "optimizer_updates": 0, "maximum_deltas": maxima,
              "execution_lock_sha256": _hash(run / "execution_lock.json"),
              "ancestry_audit_sha256": _hash(run / "ancestry_audit.json"),
              "preflight_sha256": _hash(run / "preflight.json"), "checks": reports,
              "audited_artifacts": audited_artifacts,
              "audit_scope": "Independent features, outcomes, weights, scaler, fitted-solution stationarity, frozen NLL, decisions, exact retention, source identities and receipt hashes. Recursive base ancestry is inherited from the pinned ancestry audit; no base model is retrained."}
    destination = run / "independent_audit.json"
    if destination.exists():
        _require(_json(destination) == result, "refusing to overwrite a different independent audit")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
