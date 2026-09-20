"""Run the leakage-safe cached screen for M4-Anchored Transition-Gated Residuals."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.matr import (
    boundary_context_features,
    fit_correction_gate,
    geometric_residual,
    predicted_boundary_score,
    select_boundary_threshold,
    smooth_gate,
    uncertainty_features,
)
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
PROTOCOL = ROOT / "experiments/okutama_matr_cached_gate_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/matr_cached_gate_v1"
ARMS = (
    "r0_exact_m4",
    "r1_fixed_10pct_a3",
    "r2_uncertainty_risk_gate",
    "r3_boundary_risk_gate",
    "r4_direct_boundary_switch",
    "r5_unrestricted_gate",
)
SEEDS = (42, 43, 44)


def _checked_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _input_inventory(data: dict[str, np.ndarray]) -> dict[str, str]:
    paths = {
        PROTOCOL,
        Path(__file__),
        ROOT / "src/hac/matr.py",
        SOURCE / "data/memory_data.npz",
        SOURCE / "results/v0001/oof_probabilities.npz",
        SEAR / "results/v0001/oof_probabilities.npz",
    }
    for fold in range(5):
        source_fold = SOURCE / f"models/survival_memory/fold-{fold}"
        selection_path = source_fold / "selection.json"
        selection = read_json(selection_path)
        paths.update((selection_path, source_fold / "completion_manifest.json"))
        selected_config = selection["selected"]["config_index"]
        for inner in range(3):
            directory = source_fold / f"config-{selected_config}/inner-{inner}"
            paths.update((directory / "receipt.json", directory / "predictions.npz"))
        for seed in SEEDS:
            directory = source_fold / f"refit-seed-{seed}"
            paths.update((directory / "receipt.json", directory / "predictions.npz"))
        workload = SEAR / f"workloads/a3_unrestricted_templates/fold-{fold}"
        workload_receipt = workload / "receipt.json"
        paths.update((workload_receipt, workload / "predictions.npz"))
        receipt = read_json(workload_receipt)
        selected = receipt["selected"]
        selected_inner_count = 0
        for reference in receipt["fit_receipts"]:
            fit_receipt = Path(reference["path"])
            if file_sha256(fit_receipt) != reference["sha256"]:
                raise RuntimeError("A3 workload fit-receipt ancestry changed")
            request = read_json(fit_receipt)["request"]
            if (
                request["stage"] == "inner"
                and request["learning_rate"] == selected["learning_rate"]
                and request["weight_decay"] == selected["weight_decay"]
            ):
                paths.update((fit_receipt, fit_receipt.parent / "predictions.npz"))
                selected_inner_count += 1
        if selected_inner_count != 3:
            raise RuntimeError("A3 selected inner OOF inventory is not exactly three fits")
    inventory = {_relative(path): file_sha256(path) for path in sorted(paths, key=str)}
    if canonical_hash(data["sample_ids"].tolist()) == "":
        raise AssertionError("Unreachable identity guard")
    return inventory


def prepare(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    protocol = read_json(PROTOCOL)
    if tuple(protocol["splitting"]["outer_folds"]) != tuple(range(5)):
        raise RuntimeError("MATR outer-fold contract changed")
    if protocol["inputs"]["cached_router_fits"] != 15:
        raise RuntimeError("MATR router fit count changed")
    data = _checked_npz(SOURCE / "data/memory_data.npz")
    if len(data["labels"]) != 4977 or not np.array_equal(np.unique(data["folds"]), np.arange(5)):
        raise RuntimeError("MATR requires the exact 4977-row five-fold source dataset")
    inventory = _input_inventory(data)
    lock = {
        "status": "MATR_CACHED_GATE_EXECUTION_LOCKED",
        "study_id": protocol["study_id"],
        "primary_arm": protocol["primary_arm"],
        "rows": 4977,
        "new_backbone_fits": 0,
        "new_neural_checkpoint_fits": 0,
        "cached_router_fits": 15,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "input_sha256": inventory,
    }
    run.mkdir(parents=True, exist_ok=True)
    lock_path = run / "execution_lock.json"
    immutable_json(lock_path, lock)
    return protocol, data, file_sha256(lock_path)


def _validate_lock(run: Path, data: dict[str, np.ndarray], lock_hash: str) -> None:
    lock_path = run / "execution_lock.json"
    if file_sha256(lock_path) != lock_hash:
        raise RuntimeError("MATR execution lock changed")
    lock = read_json(lock_path)
    if lock["sample_ids_sha256"] != canonical_hash(data["sample_ids"].tolist()):
        raise RuntimeError("MATR sample identity changed")
    for name, expected in lock["input_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Locked MATR input changed: {name}")


def _selected_m4(fold: int, data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = SOURCE / f"models/survival_memory/fold-{fold}"
    selection = read_json(directory / "selection.json")
    selected = selection["selected"]
    config = selected["config_index"]
    train = np.flatnonzero(data["folds"] != fold)
    held = np.flatnonzero(data["folds"] == fold)
    inner = {
        "probabilities": np.full((len(data["labels"]), 3), np.nan),
        "gate": np.full(len(data["labels"]), np.nan),
        "attention": np.full((len(data["labels"]), 5), np.nan),
        "survival": np.full((len(data["labels"]), 5), np.nan),
        "boundary_probabilities": np.full((len(data["labels"]), 5), np.nan),
    }
    covered = []
    for inner_fold in range(3):
        fit_dir = directory / f"config-{config}/inner-{inner_fold}"
        receipt = read_json(fit_dir / "receipt.json")
        request = receipt["request"]
        if (
            request["learning_rate"] != selected["learning_rate"]
            or request["weight_decay"] != selected["weight_decay"]
            or request["selection_labels_sha256"] is None
        ):
            raise RuntimeError("Selected M4 inner ancestry changed")
        output = _checked_npz(fit_dir / "predictions.npz")
        rows = output["held_rows"].astype(np.int64)
        if not np.array_equal(output["sample_ids"], data["sample_ids"][rows]):
            raise RuntimeError("Selected M4 inner sample identity changed")
        for key in inner:
            inner[key][rows] = output[key]
        covered.append(rows)
    if not np.array_equal(np.sort(np.concatenate(covered)), train):
        raise RuntimeError("Selected M4 inner predictions do not exactly cover outer training")
    inner = {key: values[train] for key, values in inner.items()}
    outer = {
        "probabilities": [],
        "gate": [],
        "attention": [],
        "survival": [],
        "boundary_probabilities": [],
    }
    for seed in SEEDS:
        fit_dir = directory / f"refit-seed-{seed}"
        receipt = read_json(fit_dir / "receipt.json")
        if (
            receipt["request"]["selection_labels_sha256"] is not None
            or receipt["request"]["seed"] != seed
            or receipt["request"]["learning_rate"] != selected["learning_rate"]
            or receipt["request"]["weight_decay"] != selected["weight_decay"]
        ):
            raise RuntimeError("Selected M4 outer ancestry changed")
        output = _checked_npz(fit_dir / "predictions.npz")
        if not np.array_equal(output["held_rows"], held) or not np.array_equal(
            output["sample_ids"], data["sample_ids"][held]
        ):
            raise RuntimeError("Selected M4 outer sample identity changed")
        for key in outer:
            outer[key].append(output[key])
    return inner, {key: np.stack(values) for key, values in outer.items()}


def _selected_a3(fold: int, data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = SEAR / f"workloads/a3_unrestricted_templates/fold-{fold}"
    workload = read_json(directory / "receipt.json")
    if workload["status"] != "SEAR_WORKLOAD_COMPLETE" or workload["request"]["fold"] != fold:
        raise RuntimeError("A3 workload is not complete for the requested fold")
    selected = workload["selected"]
    train = np.flatnonzero(data["folds"] != fold)
    held = np.flatnonzero(data["folds"] == fold)
    index = {sample_id: row for row, sample_id in enumerate(data["sample_ids"].tolist())}
    inner_probability = np.full((len(data["labels"]), 3), np.nan)
    inner_slots = np.full((len(data["labels"]), 6), np.nan)
    covered = []
    for reference in workload["fit_receipts"]:
        fit_receipt = Path(reference["path"])
        receipt = read_json(fit_receipt)
        request = receipt["request"]
        if not (
            request["stage"] == "inner"
            and request["learning_rate"] == selected["learning_rate"]
            and request["weight_decay"] == selected["weight_decay"]
        ):
            continue
        output = _checked_npz(fit_receipt.parent / "predictions.npz")
        rows = np.asarray([index[value] for value in output["sample_ids"].tolist()], dtype=np.int64)
        if np.any(data["folds"][rows] == fold):
            raise RuntimeError("Selected A3 inner prediction reaches the outer-held fold")
        inner_probability[rows] = output["probabilities"]
        inner_slots[rows] = output["slot_masses"]
        covered.append(rows)
    if len(covered) != 3 or not np.array_equal(np.sort(np.concatenate(covered)), train):
        raise RuntimeError("Selected A3 inner predictions do not exactly cover outer training")
    output = _checked_npz(directory / "predictions.npz")
    if not np.array_equal(output["sample_ids"], data["sample_ids"][held]):
        raise RuntimeError("Selected A3 outer sample identity changed")
    if output["seed_probabilities"].shape != (3, len(held), 3) or output[
        "seed_slot_masses"
    ].shape != (3, len(held), 6):
        raise RuntimeError("Selected A3 outer shape contract changed")
    return (
        {"probabilities": inner_probability[train], "slot_masses": inner_slots[train]},
        {
            "probabilities": output["seed_probabilities"].astype(np.float64),
            "slot_masses": output["seed_slot_masses"].astype(np.float64),
        },
    )


def _suppressed_transition(m4: np.ndarray, a3: np.ndarray, pair: list[int]) -> np.ndarray:
    return (m4.argmax(1) == pair[0]) & (a3.argmax(1) == pair[1])


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
    prediction_path, checkpoint_path, receipt_path = _fold_paths(run, fold)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt["execution_lock_sha256"] != lock_hash
            or file_sha256(prediction_path) != receipt["predictions_sha256"]
            or file_sha256(checkpoint_path) != receipt["checkpoint_sha256"]
        ):
            raise RuntimeError(f"Completed MATR fold {fold} changed")
        return receipt
    directory = receipt_path.parent
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Partial MATR fold {fold} retained; use a fresh run directory")
    directory.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(data["folds"] != fold)
    held = np.flatnonzero(data["folds"] == fold)
    m4_inner, m4_outer = _selected_m4(fold, data)
    a3_inner, a3_outer = _selected_a3(fold, data)
    uncertain_inner, uncertain_names = uncertainty_features(
        m4_inner["probabilities"], a3_inner["probabilities"]
    )
    context_inner, context_names = boundary_context_features(
        m4_inner["probabilities"],
        a3_inner["probabilities"],
        m4_gate=m4_inner["gate"],
        attention=m4_inner["attention"],
        survival=m4_inner["survival"],
        boundary_probabilities=m4_inner["boundary_probabilities"],
        slot_masses=a3_inner["slot_masses"],
        quality=data["quality"][train],
        memory_features=data["features"][train],
    )
    config = protocol["router"]
    common = {
        "maximum_iterations": config["maximum_iterations"],
        "tolerance": config["tolerance"],
        "random_state": config["random_state"],
    }
    uncertainty_gate = fit_correction_gate(
        uncertain_inner,
        uncertain_names,
        m4_inner["probabilities"],
        a3_inner["probabilities"],
        data["labels"][train],
        c=config["uncertainty_C"],
        harm_weight=config["risk_harm_weight"],
        **common,
    )
    boundary_gate = fit_correction_gate(
        context_inner,
        context_names,
        m4_inner["probabilities"],
        a3_inner["probabilities"],
        data["labels"][train],
        c=config["boundary_C"],
        harm_weight=config["risk_harm_weight"],
        **common,
    )
    unrestricted_gate = fit_correction_gate(
        context_inner,
        context_names,
        m4_inner["probabilities"],
        a3_inner["probabilities"],
        data["labels"][train],
        c=config["unrestricted_C"],
        harm_weight=1.0,
        **common,
    )
    inner_boundary_score = predicted_boundary_score(
        m4_inner["attention"], m4_inner["boundary_probabilities"]
    )
    boundary_threshold, threshold_candidates = select_boundary_threshold(
        inner_boundary_score,
        data["node_support_boundary"][train],
        data["node_support_complete"][train],
        config["direct_boundary_thresholds"],
    )
    checkpoint = {
        **uncertainty_gate.checkpoint("r2"),
        **boundary_gate.checkpoint("r3"),
        **unrestricted_gate.checkpoint("r5"),
        "r4_boundary_threshold": np.asarray([boundary_threshold]),
    }
    np.savez_compressed(checkpoint_path, **checkpoint)
    seed_probabilities = np.full((len(ARMS), 3, len(held), 3), np.nan, dtype=np.float64)
    seed_gates = np.full((len(ARMS), 3, len(held)), np.nan, dtype=np.float64)
    activation = config["activation_probability"]
    full = config["full_gate_probability"]
    pair = config["protected_prediction_transition"]
    for seed_index in range(3):
        m4 = m4_outer["probabilities"][seed_index]
        a3 = a3_outer["probabilities"][seed_index]
        uncertain, _ = uncertainty_features(m4, a3)
        context, _ = boundary_context_features(
            m4,
            a3,
            m4_gate=m4_outer["gate"][seed_index],
            attention=m4_outer["attention"][seed_index],
            survival=m4_outer["survival"][seed_index],
            boundary_probabilities=m4_outer["boundary_probabilities"][seed_index],
            slot_masses=a3_outer["slot_masses"][seed_index],
            quality=data["quality"][held],
            memory_features=data["features"][held],
        )
        disagreement = m4.argmax(1) != a3.argmax(1)
        suppress = _suppressed_transition(m4, a3, pair)
        gates = {
            "r0_exact_m4": np.zeros(len(held)),
            "r1_fixed_10pct_a3": np.full(len(held), 0.1),
            "r2_uncertainty_risk_gate": smooth_gate(
                uncertainty_gate.predict(uncertain),
                disagreement,
                activation_probability=activation,
                full_gate_probability=full,
            ),
            "r3_boundary_risk_gate": smooth_gate(
                boundary_gate.predict(context),
                disagreement,
                activation_probability=activation,
                full_gate_probability=full,
                suppress=suppress,
            ),
            "r4_direct_boundary_switch": (
                disagreement
                & ~suppress
                & (
                    predicted_boundary_score(
                        m4_outer["attention"][seed_index],
                        m4_outer["boundary_probabilities"][seed_index],
                    )
                    >= boundary_threshold
                )
            ).astype(np.float64),
            "r5_unrestricted_gate": unrestricted_gate.predict(context),
        }
        for arm_index, arm in enumerate(ARMS):
            gate = gates[arm]
            seed_gates[arm_index, seed_index] = gate
            seed_probabilities[arm_index, seed_index] = (
                m4.copy()
                if arm == "r0_exact_m4"
                else geometric_residual(m4, a3, gate, epsilon=config["residual_epsilon"])
            )
    np.savez_compressed(
        prediction_path,
        sample_ids=data["sample_ids"][held],
        held_rows=held,
        arms=np.asarray(ARMS),
        seeds=np.asarray(SEEDS),
        seed_probabilities=seed_probabilities,
        seed_gates=seed_gates,
        a3_seed_probabilities=a3_outer["probabilities"],
    )
    receipt = {
        "status": "MATR_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": lock_hash,
        "fold": fold,
        "outer_train_rows": int(len(train)),
        "outer_held_rows": int(len(held)),
        "outer_held_labels_read": 0,
        "router_fits": 3,
        "router_training": {
            "r2_uncertainty_risk_gate": uncertainty_gate.training_summary,
            "r3_boundary_risk_gate": boundary_gate.training_summary,
            "r5_unrestricted_gate": unrestricted_gate.training_summary,
        },
        "r4_boundary_threshold": boundary_threshold,
        "r4_training_threshold_candidates": threshold_candidates,
        "sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "predictions_sha256": file_sha256(prediction_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    immutable_json(receipt_path, receipt)
    print(json.dumps({"event": "matr_fold_complete", "fold": fold, "held_rows": len(held)}))
    return receipt


def _transitions(labels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> dict[str, int]:
    anchor_correct = anchor.argmax(1) == labels
    candidate_correct = candidate.argmax(1) == labels
    rescues = int((~anchor_correct & candidate_correct).sum())
    harms = int((anchor_correct & ~candidate_correct).sum())
    return {
        "rescues": rescues,
        "harms": harms,
        "net_corrections": rescues - harms,
        "prediction_changes": int((candidate.argmax(1) != anchor.argmax(1)).sum()),
    }


def _masked_metrics(labels: np.ndarray, probabilities: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"rows": 0}
    return probability_metrics(labels[mask], probabilities[mask])


def _scenario_resampling(
    labels: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    scenarios: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    observed = probability_metrics(labels, candidate)["macro_f1"] - probability_metrics(
        labels, anchor
    )["macro_f1"]
    rng = np.random.default_rng(seed)
    delta_f1 = np.empty(resamples)
    delta_nll = np.empty(resamples)
    delta_brier = np.empty(resamples)
    group_rows = {group: np.flatnonzero(scenarios == group) for group in groups}
    for index in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        cand = probability_metrics(labels[rows], candidate[rows])
        base = probability_metrics(labels[rows], anchor[rows])
        delta_f1[index] = cand["macro_f1"] - base["macro_f1"]
        delta_nll[index] = cand["nll"] - base["nll"]
        delta_brier[index] = cand["brier"] - base["brier"]
    null = []
    for assignment in itertools.product((False, True), repeat=len(groups)):
        mixed_candidate, mixed_anchor = candidate.copy(), anchor.copy()
        for keep_orientation, group in zip(assignment, groups, strict=True):
            if not keep_orientation:
                mixed_candidate[group_rows[group]] = anchor[group_rows[group]]
                mixed_anchor[group_rows[group]] = candidate[group_rows[group]]
        null.append(
            probability_metrics(labels, mixed_candidate)["macro_f1"]
            - probability_metrics(labels, mixed_anchor)["macro_f1"]
        )
    return {
        "groups": int(len(groups)),
        "bootstrap_resamples": resamples,
        "macro_f1_delta": float(observed),
        "macro_f1_delta_ci95": np.quantile(delta_f1, [0.025, 0.975]).tolist(),
        "nll_delta_one_sided_upper95": float(np.quantile(delta_nll, 0.95)),
        "brier_delta_one_sided_upper95": float(np.quantile(delta_brier, 0.95)),
        "exact_scenario_swap_assignments": int(len(null)),
        "scenario_swap_one_sided_p": float(np.mean(np.asarray(null) >= observed - 1e-15)),
    }


def summarize(
    run: Path, protocol: dict[str, Any], data: dict[str, np.ndarray], lock_hash: str
) -> dict[str, Any]:
    _validate_lock(run, data, lock_hash)
    rows = len(data["labels"])
    seed_probabilities = np.full((len(ARMS), 3, rows, 3), np.nan)
    seed_gates = np.full((len(ARMS), 3, rows), np.nan)
    a3_seed_probabilities = np.full((3, rows, 3), np.nan)
    receipts = {}
    for fold in range(5):
        prediction_path, checkpoint_path, receipt_path = _fold_paths(run, fold)
        receipt = read_json(receipt_path)
        if (
            receipt["status"] != "MATR_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED"
            or receipt["outer_held_labels_read"] != 0
            or file_sha256(prediction_path) != receipt["predictions_sha256"]
            or file_sha256(checkpoint_path) != receipt["checkpoint_sha256"]
        ):
            raise RuntimeError("MATR fold cannot pass the metric-release embargo")
        output = _checked_npz(prediction_path)
        held = np.flatnonzero(data["folds"] == fold)
        if (
            not np.array_equal(output["held_rows"], held)
            or not np.array_equal(output["sample_ids"], data["sample_ids"][held])
            or tuple(output["arms"].tolist()) != ARMS
            or tuple(output["seeds"].tolist()) != SEEDS
        ):
            raise RuntimeError("MATR fold identity changed")
        seed_probabilities[:, :, held] = output["seed_probabilities"]
        seed_gates[:, :, held] = output["seed_gates"]
        a3_seed_probabilities[:, held] = output["a3_seed_probabilities"]
        receipts[_relative(receipt_path)] = file_sha256(receipt_path)
    if not np.isfinite(seed_probabilities).all() or not np.isfinite(seed_gates).all():
        raise RuntimeError("MATR outer coverage is incomplete")
    source_reference = _checked_npz(SOURCE / "results/v0001/oof_probabilities.npz")
    sear_reference = _checked_npz(SEAR / "results/v0001/oof_probabilities.npz")
    for archive in (source_reference, sear_reference):
        if not np.array_equal(archive["sample_ids"], data["sample_ids"]):
            raise RuntimeError("MATR reference sample identity changed")
    if not np.array_equal(seed_probabilities[0], source_reference["new_source_m4_seeds"]):
        raise RuntimeError("MATR R0 is not the exact official new-source M4 seed output")
    if not np.array_equal(
        a3_seed_probabilities, sear_reference["a3_unrestricted_templates_seeds"]
    ):
        raise RuntimeError("MATR residual expert is not the exact official A3 seed output")
    averaged = seed_probabilities.mean(1)
    labels = data["labels"]
    metrics = {arm: probability_metrics(labels, averaged[index]) for index, arm in enumerate(ARMS)}
    seed_metrics = {
        arm: [probability_metrics(labels, seed_probabilities[index, seed]) for seed in range(3)]
        for index, arm in enumerate(ARMS)
    }
    transitions = {
        arm: _transitions(labels, averaged[index], averaged[0])
        for index, arm in enumerate(ARMS)
        if index
    }
    masks = {
        "node_boundary": data["node_support_complete"] & data["node_support_boundary"],
        "node_stable": data["node_support_complete"] & ~data["node_support_boundary"],
        "node_unknown": ~data["node_support_complete"],
        "native_height_le32": data["quality"][:, 0] * 720 <= 32 + 1e-4,
    }
    strata = {
        name: {
            arm: _masked_metrics(labels, averaged[index], mask)
            for index, arm in enumerate(ARMS)
        }
        for name, mask in masks.items()
    }
    scenarios = {
        str(scenario): {
            arm: _masked_metrics(labels, averaged[index], data["scenarios"] == scenario)
            for index, arm in enumerate(ARMS)
        }
        for scenario in np.unique(data["scenarios"])
    }
    primary_index = ARMS.index(protocol["primary_arm"])
    statistics = _scenario_resampling(
        labels,
        averaged[primary_index],
        averaged[0],
        data["scenarios"],
        resamples=protocol["statistics"]["bootstrap_resamples"],
        seed=protocol["statistics"]["seed"],
    )
    promotion = protocol["promotion_gates"]
    primary = metrics[protocol["primary_arm"]]
    anchor = metrics["r0_exact_m4"]
    class_delta = np.asarray(primary["per_class_f1"]) - np.asarray(anchor["per_class_f1"])
    seed_sd = float(
        np.std([value["macro_f1"] for value in seed_metrics[protocol["primary_arm"]]], ddof=0)
    )
    stable_delta = (
        strata["node_stable"][protocol["primary_arm"]]["accuracy"]
        - strata["node_stable"]["r0_exact_m4"]["accuracy"]
    )
    small_delta = (
        strata["native_height_le32"][protocol["primary_arm"]]["accuracy"]
        - strata["native_height_le32"]["r0_exact_m4"]["accuracy"]
    )
    boundary_delta = (
        strata["node_boundary"][protocol["primary_arm"]]["accuracy"]
        - strata["node_boundary"]["r0_exact_m4"]["accuracy"]
    )
    gain = primary["macro_f1"] - anchor["macro_f1"]
    checks = {
        "minimum_macro_f1": primary["macro_f1"] >= promotion["minimum_macro_f1"],
        "minimum_gain_macro_f1_points": 100 * gain
        >= promotion["minimum_gain_macro_f1_points"],
        "minimum_net_corrections": transitions[protocol["primary_arm"]]["net_corrections"]
        >= promotion["minimum_net_corrections"],
        "scenario_bootstrap_lower_gain_points": 100 * statistics["macro_f1_delta_ci95"][0]
        > promotion["scenario_bootstrap_lower_gain_points"],
        "maximum_one_sided_nll_delta": statistics["nll_delta_one_sided_upper95"]
        <= promotion["maximum_one_sided_nll_delta"],
        "maximum_one_sided_brier_delta": statistics["brier_delta_one_sided_upper95"]
        <= promotion["maximum_one_sided_brier_delta"],
        "maximum_class_f1_loss_points": 100 * class_delta.min()
        >= -promotion["maximum_class_f1_loss_points"],
        "maximum_stable_accuracy_loss_points": 100 * stable_delta
        >= -promotion["maximum_stable_accuracy_loss_points"],
        "maximum_small_actor_accuracy_loss_points": 100 * small_delta
        >= -promotion["maximum_small_actor_accuracy_loss_points"],
        "minimum_boundary_accuracy_gain_points": 100 * boundary_delta
        >= promotion["minimum_boundary_accuracy_gain_points"],
        "maximum_seed_macro_f1_sd_points": 100 * seed_sd
        <= promotion["maximum_seed_macro_f1_sd_points"],
        "exact_zero_gate_anchor_replay": bool(
            np.array_equal(seed_probabilities[0], source_reference["new_source_m4_seeds"])
            and np.array_equal(averaged[0], source_reference["new_source_m4"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result_dir = run / "results/v0001"
    result_dir.mkdir(parents=True, exist_ok=True)
    oof_path = result_dir / "oof_predictions.npz"
    if not oof_path.exists():
        np.savez_compressed(
            oof_path,
            sample_ids=data["sample_ids"],
            labels=labels,
            scenarios=data["scenarios"],
            folds=data["folds"],
            arms=np.asarray(ARMS),
            seeds=np.asarray(SEEDS),
            seed_probabilities=seed_probabilities,
            seed_gates=seed_gates,
            mean_probabilities=averaged,
        )
    receipt = {
        "status": "MATR_CACHED_GATE_SCREEN_COMPLETE",
        "complete": True,
        "study_id": protocol["study_id"],
        "development_status": protocol["interpretation"]["development_status"],
        "rows": rows,
        "outer_folds": 5,
        "base_prediction_seeds": 3,
        "new_backbone_fits": 0,
        "new_neural_checkpoint_fits": 0,
        "cached_router_fits": 15,
        "primary_arm": protocol["primary_arm"],
        "execution_lock_sha256": lock_hash,
        "fold_receipt_sha256": receipts,
        "oof_predictions_sha256": file_sha256(oof_path),
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "transitions_vs_exact_m4": transitions,
        "gate_activity": {
            arm: {
                "mean_gate": float(seed_gates[index].mean()),
                "active_rows_per_seed": (seed_gates[index] > 0).sum(1).tolist(),
                "full_rows_per_seed": (seed_gates[index] >= 1 - 1e-12).sum(1).tolist(),
            }
            for index, arm in enumerate(ARMS)
        },
        "strata": strata,
        "scenario_metrics": scenarios,
        "primary_statistics": statistics,
        "primary_effects_points": {
            "macro_f1": 100 * gain,
            "per_class_f1": (100 * class_delta).tolist(),
            "node_boundary_accuracy": 100 * boundary_delta,
            "node_stable_accuracy": 100 * stable_delta,
            "native_height_le32_accuracy": 100 * small_delta,
            "seed_macro_f1_sd": 100 * seed_sd,
        },
        "promotion_checks": checks,
        "promotion_passed": bool(all(checks.values())),
        "a3_reference_bit_exact": True,
        "interpretation": "Primary is fixed before metric release. Descriptive controls cannot replace it post hoc. Stage2 transported-part training is justified only by safe anchor improvement and positive net corrections.",
    }
    immutable_json(result_dir / "summary.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(result_dir / "summary.json"),
                "primary_macro_f1": primary["macro_f1"],
                "anchor_macro_f1": anchor["macro_f1"],
                "primary_gain_points": 100 * gain,
                "promotion_passed": receipt["promotion_passed"],
            },
            indent=2,
        )
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    protocol, data, lock_hash = prepare(run)
    _validate_lock(run, data, lock_hash)
    for fold in protocol["splitting"]["outer_folds"]:
        fit_fold(run, protocol, data, lock_hash, fold)
    summarize(run, protocol, data, lock_hash)


if __name__ == "__main__":
    main()
