"""Fresh 75-fit M4 matrix on regenerated-source features and nested bases."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import analyze_okutama_memory_results as audit
import numpy as np
import run_okutama_evidence_memory as original
import run_okutama_source_swap_bases as base_driver
import run_okutama_video_probe as statistics
import torch
from threadpoolctl import threadpool_limits

from hac.actor_memory_base import (
    canonical_hash,
    file_sha256,
    group_splits,
    probability_metrics,
    training_populations,
)
from hac.source_swap_data import immutable_json, read_json
from hac.source_swap_execution import ARM, FIELDS, fit_guard, fit_new, selection_from, validate_fit

ROOT, RUN, OLD = base_driver.ROOT, base_driver.RUN, base_driver.OLD


def preflight(run):
    lock_path = run / "base_execution_lock.json"
    base_lock = read_json(lock_path)
    if file_sha256(base_driver.PROTOCOL) != base_lock["source_sha256"][str(base_driver.PROTOCOL)]:
        raise RuntimeError("Source intervention protocol changed")
    files = {str(lock_path): file_sha256(lock_path), **base_lock["source_sha256"]}
    replay = read_json(run / "old_source_replay.json")
    if replay["status"] != "OLD_SOURCE_P6_AND_ALL75_M4_CHECKPOINTS_BIT_EXACT" or replay[
        "base_execution_lock_sha256"
    ] != file_sha256(lock_path):
        raise RuntimeError("Required old-source replay did not pass")
    files.update(replay["source_sha256"])
    files[str(run / "old_source_replay.npz")] = replay["output_sha256"]
    preparation = read_json(run / "base_preparation.json")
    if (
        preparation["populations"] != 58
        or preparation["estimator_fits"] != 3016
        or preparation["source_swap_base_lock_sha256"] != file_sha256(lock_path)
    ):
        raise RuntimeError("New-source nested preparation is incomplete")
    files.update(preparation["files_sha256"])
    p6_receipt = read_json(run / "new_source_p6_oof.json")
    files[str(run / "new_source_p6_oof.npz")] = p6_receipt["output_sha256"]
    for path, expected in files.items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Frozen source, input or fitted base changed: {path}")
    protocol = base_lock["original_learning_protocol"]
    data = original.load_data(run)
    cache = original.base_cache(run, data, protocol)
    if cache.identity != preparation["base_identity"]:
        raise RuntimeError("New-source nested base identity changed")
    for rows in training_populations(data["labels"], data["scenarios"], data["folds"]).values():
        cache.load(rows)
    for name in ("scenarios", "recordings", "tracks", "folds"):
        center, slot = np.nonzero(data["valid"])
        if not np.array_equal(
            data[name][center], data[name][data["neighbor_indices"][center, slot]]
        ):
            raise RuntimeError("New-source neighborhoods cross an identity boundary")
    paths = [
        Path(__file__),
        ROOT / "tests/test_source_swap_memory.py",
        Path(audit.__file__),
        Path(statistics.__file__),
        run / "base_preparation.json",
        run / "old_source_replay.json",
        run / "new_source_p6_oof.json",
        OLD / "data/legacy_temporal_strata.npz",
        OLD / "data/legacy_temporal_strata_receipt.json",
    ]
    files.update({str(path): file_sha256(path) for path in paths})
    request = {
        "stage": "fresh_new_source_M4_75fits",
        "source_swap_protocol": base_lock["protocol"],
        "unchanged_learning_protocol": protocol,
        "source_sha256": files,
        "new_base_fits_in_this_stage": 0,
        "new_memory_fits": 75,
        "old_neural_checkpoint_reuse_for_new_source": False,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
    }
    immutable_json(run / "memory_execution_lock.json", request)
    snapshot = run / "memory_source_snapshot" / Path(__file__).name
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), snapshot)
    if file_sha256(snapshot) != file_sha256(Path(__file__)):
        raise RuntimeError("Memory source snapshot changed")
    return data, cache, protocol, file_sha256(run / "memory_execution_lock.json")


def fit_directories(directory, protocol):
    return [directory / f"config-{c}" / f"inner-{i}" for c in range(4) for i in range(3)] + [
        directory / f"refit-seed-{s}" for s in protocol["outer_seeds"]
    ]


def require_fold_inventory(directory, protocol):
    expected = {path / "receipt.json" for path in fit_directories(directory, protocol)}
    if set(directory.rglob("receipt.json")) != expected:
        raise RuntimeError("Source-swap completed fold requires exactly15 declared receipts")


def fit_fold(run, data, cache, protocol, fold, execution_hash, *, train):
    outer_train, outer_held = (
        np.flatnonzero(data["folds"] != fold),
        np.flatnonzero(data["folds"] == fold),
    )
    splits = group_splits(data["labels"], data["scenarios"], outer_train)
    contexts = [cache.meta_probabilities(rows) for rows, _ in splits]
    directory = run / "models" / ARM / f"fold-{fold}"
    hashes, candidates = {}, []

    def fit_or_validate(fit_dir, rows, held, lr, wd, seed, epochs, probabilities, ancestry):
        if train:
            original.event(
                "source_swap_memory_fit_started", fit=str(fit_dir.relative_to(run)), epochs=epochs
            )
            result = fit_new(
                fit_dir,
                data,
                protocol,
                rows,
                held,
                lr,
                wd,
                seed,
                epochs,
                probabilities,
                ancestry,
                execution_hash,
                lambda **values: original.event(
                    "source_swap_memory_epoch", fit=str(fit_dir.relative_to(run)), **values
                ),
            )
        else:
            result = validate_fit(
                fit_dir,
                data,
                protocol,
                rows,
                held,
                lr,
                wd,
                seed,
                epochs,
                ancestry,
                guard=fit_guard(
                    data, protocol, rows, held, lr, wd, seed, epochs, ancestry, execution_hash
                ),
            )
        outputs, receipt, fit_hashes, _ = result
        hashes.update(fit_hashes)
        if train:
            original.event(
                "source_swap_memory_fit_complete",
                fit=str(fit_dir.relative_to(run)),
                selected_epoch=receipt["selected_epoch"],
                seconds=receipt["seconds"],
            )
        return outputs, receipt

    configs = [(lr, wd) for lr in protocol["learning_rates"] for wd in protocol["weight_decays"]]
    for config, (lr, wd) in enumerate(configs):
        p = np.full((len(data["labels"]), 3), np.nan)
        epochs = []
        for inner, ((rows, held), (probabilities, ancestry)) in enumerate(
            zip(splits, contexts, strict=True)
        ):
            output, receipt = fit_or_validate(
                directory / f"config-{config}" / f"inner-{inner}",
                rows,
                held,
                lr,
                wd,
                protocol["inner_seed"],
                None,
                probabilities,
                ancestry,
            )
            p[held] = output["probabilities"]
            epochs.append(receipt["selected_epoch"])
        candidates.append(
            {
                "config_index": config,
                "learning_rate": lr,
                "weight_decay": wd,
                "inner_epochs": epochs,
                "inner_metrics": probability_metrics(data["labels"][outer_train], p[outer_train]),
            }
        )
    selection = selection_from(candidates)
    if train:
        immutable_json(directory / "selection.json", selection)
    elif selection != read_json(directory / "selection.json"):
        raise RuntimeError("Selection not reproduced from all12innerfit evidence")
    hashes[str(directory / "selection.json")] = file_sha256(directory / "selection.json")
    selected, outputs = selection["selected"], {name: [] for name in FIELDS}
    probabilities, ancestry = cache.meta_probabilities(outer_train)
    for seed in protocol["outer_seeds"]:
        predicted, _ = fit_or_validate(
            directory / f"refit-seed-{seed}",
            outer_train,
            outer_held,
            selected["learning_rate"],
            selected["weight_decay"],
            seed,
            selection["refit_epochs"],
            probabilities,
            ancestry,
        )
        for name in FIELDS:
            outputs[name].append(predicted[name])
    outputs = {name: np.stack(values) for name, values in outputs.items()}
    require_fold_inventory(directory, protocol)
    completion = {
        "source_swap_memory_execution_sha256": execution_hash,
        "fold": fold,
        "fit_receipts": 15,
        "files_sha256": hashes,
        "sample_ids_sha256": canonical_hash(data["sample_ids"][outer_held].tolist()),
    }
    if train:
        immutable_json(directory / "completion_manifest.json", completion)
    elif completion != read_json(directory / "completion_manifest.json"):
        raise RuntimeError("Completed fold evidence changed")
    return outer_held, outputs, completion


def comparison_strata(data, legacy, masks):
    result = {"all": np.ones(len(data["labels"]), bool), **legacy, **masks}
    for prefix in ("node_support", "memory_support", "node_interval", "memory_interval"):
        complete, boundary = data[prefix + "_complete"], data[prefix + "_boundary"]
        result[prefix + "_complete_boundary"] = complete & boundary
        result[prefix + "_complete_stable"] = complete & ~boundary
        result[prefix + "_unknown"] = ~complete
    height = data["quality"][:, 0].astype(float) * 720
    result.update(
        {
            "native_height_le32": height <= 32 + 1e-4,
            "native_height_32to64": (height > 32 + 1e-4) & (height <= 64 + 1e-4),
            "native_height_gt64": height > 64 + 1e-4,
        }
    )
    for value in np.unique(data["scenarios"]):
        result["scenario_" + str(value)] = data["scenarios"] == value
    for value in range(3):
        result[f"class_{value}"] = data["labels"] == value
    return result


def compare_source(labels, candidate, reference, shared, mask):
    result = audit.compare(labels, candidate, reference, shared, mask)
    if result["rows"]:
        # The original helper names its reference P6; here it can also be M4.
        result["reference_metrics"] = result.pop("p6_metrics")
        result["reference_errors"] = result.pop("p6_errors")
        result["historical_shared_P6_failures"] = result.pop("shared_p6_failures")
    return result


def summarize(run, data, cache, protocol, execution_hash):
    all_outputs, manifest_hashes = {}, {}
    for fold in protocol["outer_folds"]:
        rows, outputs, _ = fit_fold(run, data, cache, protocol, fold, execution_hash, train=False)
        for key, value in outputs.items():
            if key not in all_outputs:
                all_outputs[key] = np.full(
                    (3, len(data["labels"]), *value.shape[2:]), np.nan, value.dtype
                )
            all_outputs[key][:, rows] = value
        manifest_path = run / "models" / ARM / f"fold-{fold}" / "completion_manifest.json"
        manifest_hashes[str(manifest_path)] = file_sha256(manifest_path)
    if len(list((run / "models").rglob("receipt.json"))) != 75 or any(
        not np.isfinite(v).all() for v in all_outputs.values()
    ):
        raise RuntimeError("Source-swap M4 needs exactly75fits and all4977rows")
    with np.load(run / "old_source_replay.npz", allow_pickle=False) as saved:
        old = {key: saved[key] for key in saved.files}
    with np.load(run / "new_source_p6_oof.npz", allow_pickle=False) as saved:
        new = {key: saved[key] for key in saved.files}
    for archive in (old, new):
        for key in ("sample_ids", "labels", "folds", "scenarios"):
            if not np.array_equal(archive[key], data[key]):
                raise RuntimeError("Final source comparison identity mismatch")
    with np.load(OLD / "data/legacy_temporal_strata.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], data["sample_ids"]):
            raise RuntimeError("Legacy diagnostic identity mismatch")
        legacy = {key: saved[key] for key in saved.files if key != "sample_ids"}
    with np.load(run / "data/source_masks.npz", allow_pickle=False) as saved:
        masks = {key: saved[key] for key in saved.files if key != "sample_ids"}
    # These are pure outer-held base inputs; no fitting is done here.
    neighbor_p = np.where(
        data["valid"][..., None], new["probabilities"][data["neighbor_indices"].clip(min=0)], 0
    )
    memory = np.einsum("sbn,bnc->sbc", all_outputs["attention"], neighbor_p)
    gate = all_outputs["gate"].astype(np.float64)[..., None]
    replay = (1 - gate) * new["probabilities"] + gate * memory
    mixture_difference = float(np.max(np.abs(replay - all_outputs["probabilities"])))
    if mixture_difference > 1e-12:
        raise RuntimeError("New-source M4 mixture does not use its own nested held P6")
    probabilities = {
        "old_source_p6": old["old_source_p6"],
        "old_source_m4": old["old_source_m4"],
        "new_source_p6": new["probabilities"],
        "new_source_m4": all_outputs["probabilities"].mean(0),
    }
    labels = data["labels"]
    shared = (old["old_source_components"].argmax(2) != labels[:, None]).all(1)
    strata = comparison_strata(data, legacy, masks)
    contrasts = [("new_source_m4", "old_source_m4"), ("new_source_p6", "old_source_p6")]
    comparisons = {}
    for candidate, reference in contrasts + [
        ("new_source_m4", "new_source_p6"),
        ("new_source_m4", "old_source_p6"),
    ]:
        comparisons[candidate + "_vs_" + reference] = {
            "strata": {
                name: compare_source(
                    labels, probabilities[candidate], probabilities[reference], shared, mask
                )
                for name, mask in strata.items()
            }
        }
    intervention_protocol = read_json(base_driver.PROTOCOL)
    for candidate, reference in contrasts:
        comparisons[candidate + "_vs_" + reference]["paired_statistics"] = (
            statistics.paired_statistics(
                labels,
                probabilities[candidate],
                probabilities[reference],
                data["scenarios"],
                bootstrap_resamples=intervention_protocol["statistics"]["bootstrap_resamples"],
                bootstrap_seed=intervention_protocol["statistics"]["seed"],
            )
        )
    adjusted = statistics.holm_adjust(
        {
            name: value["paired_statistics"]["one_sided_exact_swap_pvalue"]
            for name, value in comparisons.items()
            if "paired_statistics" in value
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["paired_statistics"]["holm_adjusted_pvalue"] = value
    boundary, edges = audit.boundary_audit(
        "new_source_m4", data, np.arange(len(labels)), all_outputs["boundary_probabilities"]
    )
    result = {
        "scope": intervention_protocol["scope"],
        "complete": True,
        "rows": 4977,
        "fresh_memory_fits": 75,
        "fresh_base_estimator_fits": 3016,
        "new_source_changed_input_rows": 4508,
        "source_swap_memory_execution_sha256": execution_hash,
        "completion_manifest_sha256": manifest_hashes,
        "results": {name: probability_metrics(labels, p) for name, p in probabilities.items()},
        "seed_metrics": {
            "new_source_m4": [probability_metrics(labels, p) for p in all_outputs["probabilities"]],
            "old_source_m4": [probability_metrics(labels, p) for p in old["old_source_m4_seeds"]],
        },
        "comparisons": comparisons,
        "comparison_definition": "Rescues/harms use the named reference, while shared-failure repairs always use the original508 shared P6 component failures.",
        "historical_shared_P6_failures": int(shared.sum()),
        "boundary_calibration": boundary,
        "boundary_probability_caution": "Original M4 BCE is cost-weighted by training-negative/positive ratio. Sigmoid is not automatically an event-calibrated posterior; no calibration fit was added.",
        "new_source_mixture_replay_max_absolute_difference": mixture_difference,
        "gate_by_stratum": {
            name: audit.distribution(all_outputs["gate"].mean(0)[mask])
            for name, mask in {
                **strata,
                "new_source_P6_correct": probabilities["new_source_p6"].argmax(1) == labels,
                "new_source_P6_wrong": probabilities["new_source_p6"].argmax(1) != labels,
            }.items()
        },
        "no_new_fusion_or_calibration_or_weight_search": True,
        "caution": "Source regeneration changes a processing chain, not isolated pixel resolution; unchanged469inputrows may change predictions through newly fitted global weights. Gains must not be added across separate probes.",
    }
    version = 1
    while (run / "results" / f"v{version:04d}").exists():
        version += 1
    output = run / "results" / f"v{version:04d}"
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "oof_probabilities.npz",
        sample_ids=data["sample_ids"],
        labels=labels,
        folds=data["folds"],
        scenarios=data["scenarios"],
        new_source_m4_seeds=all_outputs["probabilities"],
        **probabilities,
    )
    audit.write_csv(output / "physical_edges.csv", edges)
    result["artifacts_sha256"] = {path.name: file_sha256(path) for path in output.iterdir()}
    immutable_json(output / "summary.json", result)
    original.event(
        "source_swap_complete",
        output=str(output),
        results=result["results"],
        summary_sha256=file_sha256(output / "summary.json"),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "summarize"), required=True)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):
        data, cache, protocol, execution_hash = preflight(args.run)
        original.event("source_swap_memory_preflight_passed", new_memory_fits=75, new_base_fits=0)
        if args.stage == "preflight":
            return
        if args.stage == "train":
            for fold in protocol["outer_folds"]:
                rows, outputs, _ = fit_fold(
                    args.run, data, cache, protocol, fold, execution_hash, train=True
                )
                original.event(
                    "source_swap_memory_fold_complete",
                    fold=fold,
                    rows=len(rows),
                    metrics=probability_metrics(
                        data["labels"][rows], outputs["probabilities"].mean(0)
                    ),
                )
        summarize(args.run, data, cache, protocol, execution_hash)


if __name__ == "__main__":
    main()
