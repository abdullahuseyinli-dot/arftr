"""Locked neural fitting and comparison stages, separate from base generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import run_okutama_evidence_memory as driver

from hac.actor_memory_base import canonical_hash, file_sha256, group_splits, probability_metrics


def execution_lock(run, data, protocol, cache):
    preparation = json.loads((run / "base_preparation.json").read_text())
    if preparation["base_identity"] != cache.identity:
        raise RuntimeError("Base generation code or inputs changed")
    files = [
        Path(__file__),
        Path(driver.__file__),
        driver.PROTOCOL,
        driver.ROOT / "src/hac/actor_evidence_memory.py",
        driver.ROOT / "src/hac/actor_memory_training.py",
        driver.ROOT / "src/hac/actor_memory_base.py",
        run / "data/memory_data.npz",
        run / "data/base_features.npz",
        run / "data/data_lock.json",
        run / "base_preparation.json",
        run / "base_replay.json",
        run / "resource_pilot.json",
    ]
    request = {
        "protocol": protocol,
        "files": {str(path.relative_to(driver.ROOT)): file_sha256(path) for path in files},
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "protected_rows_read": 0,
        "frozen_encoders": True,
    }
    path = run / "execution_lock.json"
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise RuntimeError("Memory execution lock changed; do not mix trials")
    else:
        driver.write_json(path, request)
        for source in files[:6]:
            destination = run / "source_snapshot" / source.relative_to(driver.ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return request


def train_heads(run, data, protocol, *, arms=None, folds=None):
    import torch

    from hac.actor_memory_training import fit_memory

    torch.set_num_threads(2)
    cache = driver.base_cache(run, data, protocol)
    execution_lock(run, data, protocol, cache)
    configurations = [
        (lr, wd) for lr in protocol["learning_rates"] for wd in protocol["weight_decays"]
    ]
    order = ("survival_memory", "query_attention", "temporal_conv", "corroborated_memory")
    for arm in arms or order:
        if arm not in protocol["arms"]:
            raise ValueError("Undeclared arm")
        for outer_fold in folds or protocol["outer_folds"]:
            outer_train = np.flatnonzero(data["folds"] != outer_fold)
            outer_held = np.flatnonzero(data["folds"] == outer_fold)
            splits = group_splits(data["labels"], data["scenarios"], outer_train)
            candidates = []
            fold_dir = run / "models" / arm / f"fold-{outer_fold}"
            for config_index, (lr, wd) in enumerate(configurations):
                inner_oof = np.full((len(data["labels"]), 3), np.nan)
                epochs = []
                for inner, (train, held) in enumerate(splits):
                    probabilities, ancestry = cache.meta_probabilities(train)
                    fit_name = f"{arm}/fold-{outer_fold}/config-{config_index}/inner-{inner}"
                    driver.event("memory_fit_started", fit=fit_name)
                    predicted, receipt = fit_memory(
                        data,
                        probabilities,
                        train,
                        held,
                        arm=arm,
                        learning_rate=lr,
                        weight_decay=wd,
                        seed=protocol["inner_seed"],
                        protocol=protocol,
                        ancestry=ancestry,
                        directory=fold_dir / f"config-{config_index}" / f"inner-{inner}",
                        progress=lambda fit=fit_name, **value: driver.event(
                            "memory_epoch", fit=fit, **value
                        ),
                    )
                    inner_oof[held] = predicted
                    epochs.append(receipt["selected_epoch"])
                    driver.event(
                        "memory_fit_complete",
                        fit=fit_name,
                        selected_epoch=receipt["selected_epoch"],
                        seconds=receipt["seconds"],
                    )
                candidates.append(
                    {
                        "config_index": config_index,
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "inner_epochs": epochs,
                        "inner_metrics": probability_metrics(
                            data["labels"][outer_train], inner_oof[outer_train]
                        ),
                    }
                )
            selected = min(
                candidates,
                key=lambda c: (
                    -c["inner_metrics"]["macro_f1"],
                    c["inner_metrics"]["nll"],
                    c["learning_rate"],
                    c["weight_decay"],
                ),
            )
            epochs = max(1, int(np.rint(np.median(selected["inner_epochs"]))))
            driver.write_json(
                fold_dir / "selection.json",
                {
                    "candidates": candidates,
                    "selected": selected,
                    "refit_epochs": epochs,
                    "outer_held_used_for_selection": False,
                },
            )
            probabilities, ancestry = cache.meta_probabilities(outer_train)
            outputs = []
            for seed in protocol["outer_seeds"]:
                fit_name = f"{arm}/fold-{outer_fold}/refit-seed-{seed}"
                driver.event("memory_fit_started", fit=fit_name, epochs=epochs)
                predicted, receipt = fit_memory(
                    data,
                    probabilities,
                    outer_train,
                    outer_held,
                    arm=arm,
                    learning_rate=selected["learning_rate"],
                    weight_decay=selected["weight_decay"],
                    seed=seed,
                    protocol=protocol,
                    ancestry=ancestry,
                    directory=fold_dir / f"refit-seed-{seed}",
                    epochs=epochs,
                    progress=lambda fit=fit_name, **value: driver.event(
                        "memory_epoch", fit=fit, **value
                    ),
                )
                outputs.append(predicted)
                driver.event("memory_fit_complete", fit=fit_name, seconds=receipt["seconds"])
            ensemble = np.mean(outputs, axis=0)
            np.savez_compressed(
                fold_dir / "outer_predictions.npz",
                sample_ids=data["sample_ids"][outer_held],
                held_rows=outer_held,
                probabilities=ensemble,
                seed_probabilities=np.stack(outputs),
            )
            driver.write_json(
                fold_dir / "outer_complete.json",
                {
                    "probabilities_sha256": file_sha256(fold_dir / "outer_predictions.npz"),
                    "sample_ids_sha256": canonical_hash(data["sample_ids"][outer_held].tolist()),
                },
            )
            driver.event(
                "memory_outer_fold_complete",
                arm=arm,
                outer_fold=outer_fold,
                metrics=probability_metrics(data["labels"][outer_held], ensemble),
            )
        driver.event("memory_arm_requested_folds_complete", arm=arm)
    summarize(run, data, protocol, require_all=False)


def summarize(run, data, protocol, *, require_all=True):
    import run_okutama_video_probe as p1

    with np.load(run / "base_replay.npz", allow_pickle=False) as saved:
        reference, components = saved["probabilities"], saved["components"]
    if not json.loads((run / "base_replay.json").read_text())["passed"]:
        raise RuntimeError("P6 replay has not passed")
    indices = data["neighbor_indices"].clip(min=0)
    temporal = (
        np.where(data["valid"][..., None], reference[indices], 0).sum(1)
        / data["valid"].sum(1)[:, None]
    )
    all_probabilities = {"p6": reference, "temporal_mean": temporal}
    seed_outputs, missing = {}, []
    for arm in protocol["arms"]:
        ensemble = np.full_like(reference, np.nan)
        seeds = np.full((3, *reference.shape), np.nan)
        for fold in protocol["outer_folds"]:
            path = run / "models" / arm / f"fold-{fold}" / "outer_predictions.npz"
            marker = path.with_name("outer_complete.json")
            if not path.exists() or not marker.exists():
                missing.append(f"{arm}/fold-{fold}")
                continue
            if file_sha256(path) != json.loads(marker.read_text())["probabilities_sha256"]:
                raise RuntimeError("Completed outer predictions changed")
            with np.load(path, allow_pickle=False) as saved:
                held = np.flatnonzero(data["folds"] == fold)
                if not np.array_equal(saved["sample_ids"], data["sample_ids"][held]):
                    raise RuntimeError("Memory OOF identity mismatch")
                ensemble[held], seeds[:, held] = saved["probabilities"], saved["seed_probabilities"]
        if np.isfinite(ensemble).all():
            all_probabilities[arm], seed_outputs[arm] = ensemble, seeds
    if missing and require_all:
        raise RuntimeError(f"Memory matrix incomplete: {missing}")
    y = data["labels"]
    shared = (components.argmax(2) != y[:, None]).all(1)
    strata = {"all": np.ones(len(y), bool)}
    legacy = run / "data/legacy_temporal_strata.npz"
    if legacy.exists():
        with np.load(legacy, allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], data["sample_ids"]):
                raise RuntimeError("Legacy diagnostic alignment failed")
            for name in (
                "complete_target_boundary",
                "complete_stable_target_window",
                "unknown_target",
            ):
                strata[name] = saved[name]
    for name in ("node_interval", "memory_interval"):
        complete, boundary = data[name + "_complete"], data[name + "_boundary"]
        strata[name + "_complete_boundary"] = complete & boundary
        strata[name + "_complete_stable"] = complete & ~boundary
    results = {}
    for arm, p in all_probabilities.items():
        correct, base_correct = p.argmax(1) == y, reference.argmax(1) == y
        scenario_results = {}
        for scenario in np.unique(data["scenarios"]):
            mask = data["scenarios"] == scenario
            scenario_results[str(scenario)] = {
                "metrics": probability_metrics(y[mask], p[mask]),
                "accuracy_delta": float(correct[mask].mean() - base_correct[mask].mean()),
            }
        results[arm] = {
            "metrics": probability_metrics(y, p),
            "rescues": int((correct & ~base_correct).sum()),
            "harms": int((~correct & base_correct).sum()),
            "shared_failure_repairs": int((correct & shared).sum()),
            "per_scenario": scenario_results,
            "strata": {
                name: {
                    "metrics": probability_metrics(y[mask], p[mask]),
                    "rescues": int((correct & ~base_correct & mask).sum()),
                    "harms": int((~correct & base_correct & mask).sum()),
                    "shared_failure_repairs": int((correct & shared & mask).sum()),
                }
                for name, mask in strata.items()
                if mask.any()
            },
        }
        if arm in seed_outputs:
            results[arm]["seed_metrics"] = {
                str(seed): probability_metrics(y, values)
                for seed, values in zip(protocol["outer_seeds"], seed_outputs[arm], strict=True)
            }
    comparisons = {}
    for candidate, baseline in protocol["statistics"]["comparisons"]:
        if candidate in all_probabilities and baseline in all_probabilities:
            comparisons[candidate + "_vs_" + baseline] = p1.paired_statistics(
                y,
                all_probabilities[candidate],
                all_probabilities[baseline],
                data["scenarios"],
                bootstrap_resamples=protocol["statistics"]["bootstrap_resamples"],
                bootstrap_seed=protocol["statistics"]["seed"],
            )
    if not missing:
        adjusted = p1.holm_adjust(
            {name: result["one_sided_exact_swap_pvalue"] for name, result in comparisons.items()}
        )
        for name in comparisons:
            comparisons[name]["holm_adjusted_pvalue"] = adjusted[name]
    for arm in protocol["arms"]:
        if arm not in results:
            continue
        row = results[arm]
        limits = protocol["engineering_gate"]
        scenario_delta = [value["accuracy_delta"] for value in row["per_scenario"].values()]
        boundary = row["strata"].get("complete_target_boundary")
        stable = row["strata"].get("complete_stable_target_window")
        gates = {
            "target_84_macro_f1": row["metrics"]["macro_f1"] >= limits["target_macro_f1"],
            "positive_net_corrections": row["rescues"] > row["harms"],
            "shared_repairs": row["shared_failure_repairs"]
            >= limits["shared_failure_repairs_minimum"],
            "scenario_breadth": sum(value > 0 for value in scenario_delta)
            >= limits["improved_scenarios_minimum"],
            "worst_scenario": min(scenario_delta)
            >= limits["worst_scenario_accuracy_delta_minimum"],
            "nll_no_worse": row["metrics"]["nll"] <= results["p6"]["metrics"]["nll"],
            "brier_no_worse": row["metrics"]["brier"] <= results["p6"]["metrics"]["brier"],
            "boundary_harm_reduced": boundary is not None
            and boundary["harms"] < limits["boundary_harms_less_than"],
            "stable_net_positive": stable is not None and stable["rescues"] > stable["harms"],
        }
        if arm == protocol["primary_arm"]:
            for reference_arm in ("temporal_mean", "query_attention"):
                gates["beats_" + reference_arm] = (
                    reference_arm in results
                    and row["metrics"]["macro_f1"] > results[reference_arm]["metrics"]["macro_f1"]
                )
        row["engineering_checks"] = gates
        row["all_engineering_checks_passed"] = all(gates.values())
    receipt = {
        "complete": not missing,
        "missing": missing,
        "scope": protocol["scope"],
        "rows": len(y),
        "scenarios": len(np.unique(data["scenarios"])),
        "shared_p6_errors": int(shared.sum()),
        "results": results,
        "comparisons": comparisons,
        "note": "Outer seed-mean probability ensemble; repeated development population, not external confirmation.",
    }
    output = run / "results"
    driver.write_json(output / "summary.json", receipt)
    np.savez_compressed(
        output / "oof_probabilities.npz",
        sample_ids=data["sample_ids"],
        labels=y,
        scenarios=data["scenarios"],
        folds=data["folds"],
        **all_probabilities,
    )
    driver.event(
        "memory_summary",
        complete=not missing,
        results={arm: row["metrics"] for arm, row in results.items()},
    )
    return receipt
