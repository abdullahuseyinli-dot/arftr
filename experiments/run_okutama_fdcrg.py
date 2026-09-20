"""Run the one-shot Fine-Density Counterfactual Residual Gate (FDCRG).

This is intentionally a small, nested experiment.  Fine/coarse readers are
cross-fitted inside every outer training population; a utility gate is then
cross-fitted over those rows and is allowed to intervene only through the
fixed four-action family in :mod:`hac.fine_density_residual_gate`.  ARFTR is
read-only and remains the explicit zero-action fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hac.actor_memory_base import group_splits
from hac.arftr import apply_arftr, exact_track_neighbors
from hac.fine_density_residual_gate import (
    ACTION_NAMES,
    FineDensityResidualGate,
    build_actions,
    build_features,
)

from experiments import run_okutama_arftr as arftr_base
from experiments import run_okutama_fine_detail_probe_v2 as fine_base


PROTOCOL_PATH = ROOT / "experiments/okutama_fdcrg_protocol.json"
FINE_RUN = ROOT / ".runs/research_20260917/fine_detail_probe_v2_retry"
ARFTR_RUN = ROOT / ".runs/research_20260912/arftr_v1/results/v0001"
ARFTR_PROBABILITIES = ARFTR_RUN / "oof_probabilities.npz"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260917/fdcrg_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(1)
    one_hot = np.eye(3, dtype=np.float64)[labels]
    return {
        "rows": int(len(labels)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "accuracy": float(np.mean(predictions == labels)),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0)).mean()),
        "brier": float(np.mean((probabilities - one_hot) ** 2)),
        "per_class_f1": f1_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "confusion": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "errors": int(np.sum(predictions != labels)),
    }


def validate_inputs(protocol: dict) -> dict:
    if protocol.get("protocol_name") != "okutama_fine_density_counterfactual_residual_gate_v1":
        raise RuntimeError("FDCRG protocol name changed")
    if protocol.get("status") not in {"NEXT_PRIMARY_NOT_EXECUTED", "EXECUTED_FAILED_PROMOTION_GATES"}:
        raise RuntimeError("FDCRG protocol status is not recognized")
    if protocol.get("training_and_leakage", {}).get("outer_folds") != 5:
        raise RuntimeError("FDCRG outer fold contract changed")
    if not PROTOCOL_PATH.is_file() or not FINE_RUN.is_dir() or not ARFTR_PROBABILITIES.is_file():
        raise RuntimeError("FDCRG inputs are incomplete")
    fine_protocol = fine_base.read_json(fine_base.PROTOCOL_PATH)
    fine_data = fine_base.base.validate_inputs(fine_protocol)
    with np.load(ARFTR_PROBABILITIES, allow_pickle=False) as saved:
        anchor_ids = saved["sample_ids"].astype(str)
        anchor_labels = saved["labels"].astype(np.int64)
        anchor_folds = saved["folds"].astype(np.int64)
        anchor = saved["mean_probabilities"][5].astype(np.float64)
    if not np.array_equal(anchor_ids, fine_data["sample_ids"]):
        raise RuntimeError("ARFTR and fine-reader sample IDs differ")
    if not np.array_equal(anchor_labels, fine_data["labels"]) or not np.array_equal(anchor_folds, fine_data["folds"]):
        raise RuntimeError("ARFTR and fine-reader labels/folds differ")
    if anchor.shape != (4977, 3) or not np.isfinite(anchor).all():
        raise RuntimeError("Malformed ARFTR anchor")
    fine_values: dict[str, np.ndarray] = {}
    for arm in ("coarse_control", "fine_both"):
        values = np.zeros((len(anchor), 3), dtype=np.float64)
        seen = np.zeros(len(anchor), dtype=bool)
        for fold in range(5):
            path = FINE_RUN / f"fold-{fold}" / arm / "predictions.npz"
            if not path.is_file():
                raise RuntimeError(f"Missing audited fine-reader prediction: {path}")
            with np.load(path, allow_pickle=False) as saved:
                rows = saved["held_rows"].astype(np.int64)
                ids = saved["sample_ids"].astype(str)
                if not np.array_equal(ids, fine_data["sample_ids"][rows]) or np.any(seen[rows]):
                    raise RuntimeError(f"Fine-reader identity mismatch in {path}")
                values[rows] = saved["probabilities"].astype(np.float64)
                seen[rows] = True
        if not seen.all() or (values < 0).any() or not np.allclose(values.sum(1), 1.0, atol=1e-6):
            raise RuntimeError(f"Incomplete fine-reader arm {arm}")
        fine_values[arm] = values
    source_data = arftr_base.load_cached_study_data(arftr_base.SOURCE)
    for key in ("sample_ids", "labels", "folds"):
        if key == "sample_ids":
            if not np.array_equal(source_data[key].astype(str), fine_data[key]):
                raise RuntimeError(f"Source and fine data differ for {key}")
        elif not np.array_equal(source_data[key], fine_data[key]):
            raise RuntimeError(f"Source and fine data differ for {key}")
    return {
        "fine": fine_data,
        "source": source_data,
        "anchor": anchor,
        "outer_fine": fine_values["fine_both"],
        "outer_coarse": fine_values["coarse_control"],
    }


def _reader_fit_predict(data: dict, fit_rows: np.ndarray, held_rows: np.ndarray, arm: str, seed: int, device: str) -> np.ndarray:
    """Fit one fresh reader on ``fit_rows`` and predict only ``held_rows``."""

    mean, std, qmean, qstd, weights = fine_base.base.training_stats(data, fit_rows)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = fine_base.base.make_model("fine_both" if arm == "fine_both" else "coarse_control", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weight_tensor = torch.from_numpy(weights).to(device)
    batch_size, epochs = 128, 2
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(len(fit_rows), generator=generator).numpy()
        for start in range(0, len(order), batch_size):
            rows = fit_rows[order[start : start + batch_size]]
            fine, times, quality, coarse = fine_base.base.batch_inputs(
                data, rows, mean, std, qmean, qstd,
                "fine_both" if arm == "fine_both" else "coarse_control", device,
            )
            labels = torch.from_numpy(data["labels"][rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(fine, times, quality=quality, coarse_reference=coarse, center_index=4)
            loss = torch.nn.functional.cross_entropy(output["logits"], labels, weight=weight_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError("FDCRG reader produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(held_rows), batch_size):
            rows = held_rows[start : start + batch_size]
            fine, times, quality, coarse = fine_base.base.batch_inputs(
                data, rows, mean, std, qmean, qstd,
                "fine_both" if arm == "fine_both" else "coarse_control", device,
            )
            predictions.append(model(fine, times, quality=quality, coarse_reference=coarse, center_index=4)["probabilities"].cpu().numpy())
    result = np.concatenate(predictions, axis=0).astype(np.float64)
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _fresh_inner_readers(data: dict, outer_train: np.ndarray, inner_splits: list[tuple[np.ndarray, np.ndarray]], outer_fold: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    fine = np.full((len(data["labels"]), 3), np.nan, dtype=np.float64)
    coarse = np.full((len(data["labels"]), 3), np.nan, dtype=np.float64)
    for inner_index, (fit_rows, held_rows) in enumerate(inner_splits):
        fine[held_rows] = _reader_fit_predict(data, fit_rows, held_rows, "fine_both", 4100 + outer_fold * 100 + inner_index, device)
        coarse[held_rows] = _reader_fit_predict(data, fit_rows, held_rows, "coarse_control", 5100 + outer_fold * 100 + inner_index, device)
    if not np.isfinite(fine[outer_train]).all() or not np.isfinite(coarse[outer_train]).all():
        raise RuntimeError("Fresh inner reader coverage is incomplete")
    return fine, coarse


def _fresh_inner_arftr(data: dict, outer_fold: int, outer_train: np.ndarray, inner_splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Create nested ARFTR anchor predictions for outer-training rows.

    The cached M4/A3/P6 ancestry is already cross-fitted at the base layer. For
    each gate split, ARFTR coefficients are selected using only that split's
    fit rows, then applied without labels to its held rows.
    """

    protocol = arftr_base._read_json(arftr_base.PROTOCOL)
    cached = arftr_base.selected_fold_artifacts(ROOT, arftr_base.SOURCE, arftr_base.SEAR, data, outer_fold, seeds=(42, 43, 44))
    p6_cache = arftr_base._p6_cache(data)
    p6_all, _ancestry = p6_cache.meta_probabilities(outer_train)
    m4 = np.full((len(data["labels"]), 3), np.nan, dtype=np.float64)
    a3 = np.full((len(data["labels"]), 3), np.nan, dtype=np.float64)
    p6 = np.full((len(data["labels"]), 3), np.nan, dtype=np.float64)
    m4[outer_train] = cached.m4_inner["probabilities"]
    a3[outer_train] = cached.a3_inner["probabilities"]
    p6[outer_train] = p6_all[outer_train]
    neighbors_all = exact_track_neighbors(data, outer_train)
    result = np.full((len(data["labels"]), 3), np.nan, dtype=np.float64)
    local_index = {int(row): index for index, row in enumerate(outer_train.tolist())}
    for fit_rows, held_rows in inner_splits:
        neighbors_fit = exact_track_neighbors(data, fit_rows)
        parameters, _ = arftr_base.select_parameters(
            data["labels"][fit_rows], m4[fit_rows], p6[fit_rows], a3[fit_rows], neighbors_fit, protocol
        )
        applied = apply_arftr(m4[outer_train], p6[outer_train], a3[outer_train], neighbors_all, parameters, epsilon=protocol["factorization"]["epsilon"])
        result[held_rows] = applied[[local_index[int(row)] for row in held_rows.tolist()]]
    if not np.isfinite(result[outer_train]).all():
        raise RuntimeError("Fresh nested ARFTR coverage is incomplete")
    return result


def _utility(labels: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> float:
    rows = np.arange(len(labels))
    before = -np.log(np.clip(anchor[rows, labels], 1e-8, 1.0))
    after = -np.log(np.clip(candidate[rows, labels], 1e-8, 1.0))
    return float(np.sum(before - after))


def _fold_gate(data: dict, outer_fold: int, outer_train: np.ndarray, outer_held: np.ndarray, inner_splits: list[tuple[np.ndarray, np.ndarray]], anchor_inner: np.ndarray, fine_inner: np.ndarray, coarse_inner: np.ndarray, device: str) -> dict:
    quality = data["fine"]["quality"]
    times = data["fine"]["times"]
    features = build_features(anchor_inner[outer_train], fine_inner[outer_train], coarse_inner[outer_train], quality=quality[outer_train], times=times[outer_train])
    actions = build_actions(anchor_inner[outer_train], fine_inner[outer_train], coarse_inner[outer_train])
    local = {int(row): index for index, row in enumerate(outer_train.tolist())}
    inner_receipts = []
    for inner_index, (fit_rows, held_rows) in enumerate(inner_splits):
        fit_local = np.asarray([local[int(row)] for row in fit_rows], dtype=np.int64)
        held_local = np.asarray([local[int(row)] for row in held_rows], dtype=np.int64)
        gate = FineDensityResidualGate.fit(features[fit_local], anchor_inner[fit_rows], actions[fit_local], data["source"]["labels"][fit_rows])
        candidate, choices = gate.apply(features[held_local], actions[held_local])
        utility = _utility(data["source"]["labels"][held_rows], anchor_inner[held_rows], candidate)
        inner_receipts.append(
            {
                "inner": inner_index,
                "rows": int(len(held_rows)),
                "utility_nll_sum": utility,
                "positive_utility": bool(utility > 0),
                "retain_fraction": float(np.mean(choices == 0)),
                "intervention_rate": float(np.mean(choices != 0)),
                "actions": np.bincount(choices, minlength=4).tolist(),
                "threshold": gate.threshold,
            }
        )
    positive_count = int(sum(value["positive_utility"] for value in inner_receipts))
    retain_ok = bool(all(value["retain_fraction"] >= 0.70 for value in inner_receipts))
    utility_ok = bool(positive_count >= 4 and retain_ok)
    final_gate = FineDensityResidualGate.fit(features, anchor_inner[outer_train], actions, data["source"]["labels"][outer_train]) if utility_ok else None
    outer_features = build_features(
        data["anchor"][outer_held],
        data["outer_fine"][outer_held],
        data["outer_coarse"][outer_held],
        quality=quality[outer_held],
        times=times[outer_held],
    )
    outer_actions = build_actions(data["anchor"][outer_held], data["outer_fine"][outer_held], data["outer_coarse"][outer_held])
    if final_gate is None:
        probabilities = data["anchor"][outer_held].copy()
        choices = np.zeros(len(outer_held), dtype=np.int64)
        threshold = None
    else:
        probabilities, choices = final_gate.apply(outer_features, outer_actions)
        threshold = final_gate.threshold
    return {
        "outer_fold": outer_fold,
        "inner": inner_receipts,
        "positive_inner_count": positive_count,
        "retain_gate_ok": retain_ok,
        "utility_gate_ok": utility_ok,
        "outer_rows": outer_held,
        "outer_probabilities": probabilities,
        "outer_choices": choices,
        "outer_actions": outer_actions,
        "outer_features": outer_features,
        "outer_threshold": threshold,
        "final_gate": final_gate,
    }


def _save_gate(path: Path, gate: FineDensityResidualGate) -> None:
    arrays = {"scaler_mean": gate.scaler.mean_, "scaler_scale": gate.scaler.scale_, "threshold": np.asarray([gate.threshold]), "residual_cap": np.asarray([gate.residual_cap]), "intervention_budget": np.asarray([gate.intervention_budget])}
    for action in range(1, 4):
        model = gate.regressors[action]
        arrays[f"coef_{action}"] = model.coef_
        arrays[f"intercept_{action}"] = np.asarray([model.intercept_])
    np.savez_compressed(path, **arrays)


def _scenario_bootstrap(labels: np.ndarray, scenarios: np.ndarray, anchor: np.ndarray, candidate: np.ndarray, *, seed: int = 20260917, resamples: int = 10000) -> dict:
    unique = np.unique(scenarios)
    deltas = []
    for scenario in unique:
        mask = scenarios == scenario
        deltas.append(metrics(labels[mask], candidate[mask])["macro_f1"] - metrics(labels[mask], anchor[mask])["macro_f1"])
    deltas = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = deltas[rng.integers(0, len(deltas), size=(resamples, len(deltas)))].mean(1)
    return {"scenarios": unique.tolist(), "scenario_deltas": deltas.tolist(), "resamples": resamples, "seed": seed, "lower": float(np.quantile(sampled, 0.025)), "mean": float(sampled.mean()), "upper": float(np.quantile(sampled, 0.975))}


def run_timing_smoke(data: dict, output: Path, device: str) -> dict:
    path = output / "timing_smoke.json"
    if path.is_file():
        return read_json(path)
    outer_fold = 0
    outer_train = np.flatnonzero(data["source"]["folds"] != outer_fold)
    inner_splits = group_splits(data["source"]["labels"], data["source"]["scenarios"], outer_train, n_splits=5, seed=20260917)
    fit_rows, held_rows = inner_splits[0]
    started = time.perf_counter()
    _reader_fit_predict(data["fine"], fit_rows[:128], held_rows[:32], "fine_both", 6200, device)
    reader_seconds = time.perf_counter() - started
    rng = np.random.default_rng(7)
    toy_features = rng.normal(size=(128, 32))
    toy_anchor = np.full((128, 3), 1 / 3, dtype=np.float64)
    toy_actions = np.repeat(toy_anchor[:, None, :], 4, axis=1)
    toy_labels = np.arange(128) % 3
    started_gate = time.perf_counter()
    FineDensityResidualGate.fit(toy_features, toy_anchor, toy_actions, toy_labels)
    gate_seconds = time.perf_counter() - started_gate
    # Two reader arms x 25 inner fits x five outer folds, plus a small gate fit.
    projected = reader_seconds * 2 * 25 * 5 + gate_seconds * 30
    result = {
        "status": "FDCRG_TIMING_SMOKE_PASS",
        "device": device,
        "reader_seconds_one_partial_fit": reader_seconds,
        "gate_seconds_one_fit": gate_seconds,
        "projected_full_seconds": projected,
        "projected_full_minutes": projected / 60.0,
        "training_authorized": bool(projected <= 1200.0),
        "outer_held_labels_read": 0,
        "classifier_fits_in_smoke": 0,
    }
    write_json(path, result)
    if not result["training_authorized"]:
        raise RuntimeError(f"FDCRG projected budget exceeds 20 minutes: {projected / 60.0:.2f}")
    return result


def run_all(data: dict, protocol: dict, output: Path, device: str) -> dict:
    smoke = run_timing_smoke(data, output, device)
    if not smoke["training_authorized"]:
        raise RuntimeError("FDCRG timing smoke did not authorize the run")
    output.mkdir(parents=True, exist_ok=True)
    all_anchor = data["anchor"]
    all_candidate = np.zeros_like(all_anchor)
    all_choices = np.zeros(len(all_anchor), dtype=np.int64)
    fold_receipts = []
    total_inner_fits = 0
    started = time.perf_counter()
    for outer_fold in range(5):
        outer_train = np.flatnonzero(data["source"]["folds"] != outer_fold)
        outer_held = np.flatnonzero(data["source"]["folds"] == outer_fold)
        inner_splits = group_splits(data["source"]["labels"], data["source"]["scenarios"], outer_train, n_splits=5, seed=20260917)
        inner_fine, inner_coarse = _fresh_inner_readers(data["fine"], outer_train, inner_splits, outer_fold, device)
        inner_anchor = _fresh_inner_arftr(data["source"], outer_fold, outer_train, inner_splits)
        gate_result = _fold_gate(
            data,
            outer_fold,
            outer_train,
            outer_held,
            inner_splits,
            inner_anchor,
            inner_fine,
            inner_coarse,
            device,
        )
        all_candidate[outer_held] = gate_result["outer_probabilities"]
        all_choices[outer_held] = gate_result["outer_choices"]
        total_inner_fits += len(inner_splits) * 2
        fold_dir = output / f"fold-{outer_fold}"
        fold_dir.mkdir(parents=False, exist_ok=False)
        np.savez_compressed(
            fold_dir / "outer_inputs.npz",
            sample_ids=data["fine"]["sample_ids"][outer_held],
            held_rows=outer_held,
            anchor=data["anchor"][outer_held],
            fine=data["outer_fine"][outer_held],
            coarse=data["outer_coarse"][outer_held],
            quality=data["fine"]["quality"][outer_held],
            times=data["fine"]["times"][outer_held],
            choices=gate_result["outer_choices"],
            probabilities=gate_result["outer_probabilities"],
        )
        if gate_result["final_gate"] is not None:
            _save_gate(fold_dir / "gate.npz", gate_result["final_gate"])
        receipt = {
            "status": "FDCRG_OUTER_FOLD_COMPLETE",
            "outer_fold": outer_fold,
            "held_rows": int(len(outer_held)),
            "sample_ids_sha256": hashlib.sha256("|".join(data["fine"]["sample_ids"][outer_held].tolist()).encode()).hexdigest(),
            "positive_inner_count": gate_result["positive_inner_count"],
            "retain_gate_ok": gate_result["retain_gate_ok"],
            "utility_gate_ok": gate_result["utility_gate_ok"],
            "inner": gate_result["inner"],
            "outer_intervention_rate": float(np.mean(gate_result["outer_choices"] != 0)),
            "outer_choices": np.bincount(gate_result["outer_choices"], minlength=4).tolist(),
            "outer_held_labels_read_during_fit": 0,
            "reader_fits": len(inner_splits) * 2,
            "outer_gate_fit": bool(gate_result["final_gate"] is not None),
            "outer_inputs": str((fold_dir / "outer_inputs.npz").relative_to(ROOT)).replace("\\", "/"),
        }
        write_json(fold_dir / "receipt.json", receipt)
        fold_receipts.append(receipt)
    if not np.isfinite(all_candidate).all() or not np.allclose(all_candidate.sum(1), 1.0, atol=1e-6):
        raise RuntimeError("FDCRG outer coverage is incomplete")
    labels = data["source"]["labels"]
    anchor_metrics = metrics(labels, all_anchor)
    candidate_metrics = metrics(labels, all_candidate)
    rescues = (all_anchor.argmax(1) != labels) & (all_candidate.argmax(1) == labels)
    harms = (all_anchor.argmax(1) == labels) & (all_candidate.argmax(1) != labels)
    folds = {}
    for fold in range(5):
        mask = data["source"]["folds"] == fold
        folds[str(fold)] = {
            "anchor": metrics(labels[mask], all_anchor[mask]),
            "fdcrg": metrics(labels[mask], all_candidate[mask]),
            "rescues": int(np.sum(rescues[mask])),
            "harms": int(np.sum(harms[mask])),
            "net_corrections": int(np.sum(rescues[mask]) - np.sum(harms[mask])),
            "intervention_rate": float(np.mean(all_choices[mask] != 0)),
        }
    bootstrap = _scenario_bootstrap(labels, data["source"]["scenarios"], all_anchor, all_candidate)
    promotion = {
        "macro_f1_gate": candidate_metrics["macro_f1"] >= anchor_metrics["macro_f1"] + 0.005,
        "net_corrections_gate": int(rescues.sum() - harms.sum()) >= 25,
        "positive_net_every_outer_fold": all(value["net_corrections"] > 0 for value in folds.values()),
        "scenario_bootstrap_lower_gate": bootstrap["lower"] > 0,
        "class_f1_gate": all(candidate_metrics["per_class_f1"][i] >= anchor_metrics["per_class_f1"][i] - 0.005 for i in range(3)),
        "proper_losses_gate": candidate_metrics["nll"] <= anchor_metrics["nll"] and candidate_metrics["brier"] <= anchor_metrics["brier"],
        "intervention_gate": float(np.mean(all_choices != 0)) <= 0.15,
    }
    promotion["all_gates"] = all(promotion.values())
    result = {
        "status": "FDCRG_SCIENTIFIC_PROBE_COMPLETE",
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": int(len(labels)),
        "outer_folds": 5,
        "actions": list(ACTION_NAMES),
        "anchor": anchor_metrics,
        "fdcrg": candidate_metrics,
        "rescues": int(rescues.sum()),
        "harms": int(harms.sum()),
        "net_corrections": int(rescues.sum() - harms.sum()),
        "intervention_rate": float(np.mean(all_choices != 0)),
        "folds": folds,
        "fold_receipts": fold_receipts,
        "scenario_bootstrap": bootstrap,
        "promotion_gates": promotion,
        "reader_fits": total_inner_fits,
        "outer_held_labels_read_during_fit": 0,
        "human_review_fields_read": 0,
        "annotation_support_used": False,
        "oracle_results": False,
        "retained_arftr_changed": False,
        "independent_audit": "pending",
    }
    write_json(output / "summary.json", result)
    np.savez_compressed(output / "oof_predictions.npz", sample_ids=data["fine"]["sample_ids"], anchor=all_anchor, fdcrg=all_candidate, choices=all_choices)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "timing-smoke", "all", "summarize"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    protocol = read_json(PROTOCOL_PATH)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage == "all" and args.output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.stage == "all":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "source_snapshot").mkdir()
        shutil.copy2(Path(__file__), args.output_dir / "source_snapshot" / Path(__file__).name)
        shutil.copy2(PROTOCOL_PATH, args.output_dir / "source_snapshot" / PROTOCOL_PATH.name)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    data = validate_inputs(protocol)
    if args.stage == "validate":
        print(json.dumps({"status": "FDCRG_VALIDATE_PASS", "rows": 4977, "device": args.device}, indent=2))
        return
    if args.stage == "timing-smoke":
        print(json.dumps(run_timing_smoke(data, args.output_dir, args.device), indent=2))
        return
    if args.stage == "all":
        result = run_all(data, protocol, args.output_dir, args.device)
    else:
        result = read_json(args.output_dir / "summary.json")
    print(json.dumps({"status": result["status"], "output": str(args.output_dir / "summary.json"), "anchor": result.get("anchor"), "fdcrg": result.get("fdcrg"), "gates": result.get("promotion_gates")}, indent=2))


if __name__ == "__main__":
    main()
