"""Execute the separate matched center-episode trial using verified frozen bases."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import analyze_okutama_memory_results as audit
import numpy as np
import run_okutama_evidence_memory as memory_driver
import run_okutama_video_probe as statistics
import torch

from hac.actor_memory_base import (
    canonical_hash,
    file_sha256,
    group_splits,
    probability_metrics,
    training_populations,
)
from hac.center_episode_training import fit_episode

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260908/center_episode_v1"
MEMORY_RUN = ROOT / ".runs/research_20260908/evidence_memory"
PROTOCOL = ROOT / "experiments/okutama_center_episode_protocol.json"


def event(kind, **values):
    print(
        json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, **values}),
        flush=True,
    )


def write_immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if audit.read_json(path) != value:
            raise RuntimeError(f"Immutable episode receipt changed: {path}")
    else:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)


def selection_from(candidates):
    selected = min(
        candidates,
        key=lambda value: (
            -value["inner_metrics"]["macro_f1"],
            value["inner_metrics"]["nll"],
            value["learning_rate"],
            value["weight_decay"],
        ),
    )
    epochs = max(1, int(np.rint(np.median(selected["inner_epochs"]))))
    return {
        "candidates": candidates,
        "selected": selected,
        "refit_epochs": epochs,
        "outer_held_used_for_selection": False,
    }


def fit_directories(directory, protocol):
    configs = len(protocol["learning_rates"]) * len(protocol["weight_decays"])
    return [
        directory / f"config-{config}" / f"inner-{inner}"
        for config in range(configs)
        for inner in range(protocol["inner_folds"])
    ] + [directory / f"refit-seed-{seed}" for seed in protocol["outer_seeds"]]


def completion_manifest(directory, protocol, arm, fold, execution_hash):
    expected = fit_directories(directory, protocol)
    if set(directory.rglob("receipt.json")) != {path / "receipt.json" for path in expected}:
        raise RuntimeError("A completed episode fold must contain exactly 15 expected fit receipts")
    hashes = {"selection.json": file_sha256(directory / "selection.json")}
    for fit_dir in expected:
        if {path.name for path in fit_dir.iterdir()} != {
            "request.json",
            "receipt.json",
            "checkpoint.pt",
            "predictions.npz",
        }:
            raise RuntimeError("Completed episode fit inventory differs")
        receipt = audit.read_json(fit_dir / "receipt.json")
        if audit.read_json(fit_dir / "request.json") != receipt["request"]:
            raise RuntimeError("Episode pre-fit request differs from completed receipt")
        if set(receipt["output_sha256"]) != {"checkpoint.pt", "predictions.npz"}:
            raise RuntimeError("Completed episode checksum inventory differs")
        for filename in ("request.json", "receipt.json", "checkpoint.pt", "predictions.npz"):
            path = fit_dir / filename
            actual = file_sha256(path)
            if (
                filename in receipt["output_sha256"]
                and actual != receipt["output_sha256"][filename]
            ):
                raise RuntimeError("Completed episode fit bytes changed")
            hashes[str(path.relative_to(directory))] = actual
    return {
        "arm": arm,
        "fold": fold,
        "execution_lock_sha256": execution_hash,
        "fit_receipts": len(expected),
        "files_sha256": hashes,
    }


def validate_fit_evidence(
    directory, data, protocol, arm, train, held, lr, wd, seed, epochs, ancestry
):
    receipt = audit.read_json(directory / "receipt.json")
    request = receipt["request"]
    expected = {
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": wd,
        "data_sha256": data.get("_file_sha256", "synthetic_test_data"),
        "protocol_sha256": canonical_hash(protocol),
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "train_boundary_targets_sha256": canonical_hash(data["boundary_targets"][train].tolist()),
        "train_boundary_valid_sha256": canonical_hash(data["boundary_valid"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": canonical_hash(data["labels"][held].tolist())
        if epochs is None
        else None,
        "base_ancestry_sha256": canonical_hash(ancestry),
        "training_code_sha256": file_sha256(ROOT / "src/hac/center_episode_training.py"),
        "architecture_code_sha256": file_sha256(ROOT / "src/hac/center_episode_memory.py"),
        "batches_seed_code_sha256": file_sha256(ROOT / "src/hac/actor_memory_training.py"),
        "boundary_loss": "unweighted_masked_BCE_with_logits",
        "gradient_clipping": "separate_hazard_and_utility_norm_1",
    }
    if (
        request != expected
        or audit.read_json(directory / "request.json") != expected
        or receipt["base_ancestry"] != ancestry
    ):
        raise RuntimeError("Episode fit evidence has a foreign request or base ancestry")
    if receipt["train_rows"] != train.tolist() or receipt["held_rows"] != held.tolist():
        raise RuntimeError("Episode fit evidence has foreign train/held rows")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["held_rows"], held) or not np.array_equal(
            saved["sample_ids"], data["sample_ids"][held]
        ):
            raise RuntimeError("Episode fit evidence prediction IDs changed")
        probabilities = saved["probabilities"]
    measured = probability_metrics(data["labels"][held], probabilities)
    if measured != receipt["held_metrics"] or receipt["boundary_positive_weight"] is not None:
        raise RuntimeError("Episode saved metrics or boundary weighting differ")
    history = receipt["history"]
    if len(history) != receipt["epochs_run"] or [row["epoch"] for row in history] != list(
        range(1, len(history) + 1)
    ):
        raise RuntimeError("Episode history epoch coverage changed")
    if epochs is None:
        best = min(
            history,
            key=lambda row: (
                -row["validation_metrics"]["macro_f1"],
                row["validation_metrics"]["nll"],
                row["epoch"],
            ),
        )
        if best["epoch"] != receipt["selected_epoch"] or best["validation_metrics"] != measured:
            raise RuntimeError("Episode selected inner checkpoint does not match saved evidence")
    elif (
        receipt["selected_epoch"] != epochs
        or len(history) != epochs
        or any("validation_metrics" in row for row in history)
    ):
        raise RuntimeError("Episode outer fit performed unexpected checkpoint selection")
    return probabilities, receipt


def validate_selection_evidence(
    directory, data, protocol, arm, fold, cache, *, validator=validate_fit_evidence
):
    train = np.flatnonzero(data["folds"] != fold)
    splits = group_splits(
        data["labels"],
        data["scenarios"],
        train,
        n_splits=protocol["inner_folds"],
        seed=protocol["split_seed"],
    )
    ancestries = [cache.meta_probabilities(inner_train)[1] for inner_train, _ in splits]
    candidates = []
    configs = [(lr, wd) for lr in protocol["learning_rates"] for wd in protocol["weight_decays"]]
    for config, (lr, wd) in enumerate(configs):
        probabilities = np.full((len(data["labels"]), 3), np.nan)
        epochs = []
        for inner, ((inner_train, inner_held), ancestry) in enumerate(
            zip(splits, ancestries, strict=True)
        ):
            predicted, receipt = validator(
                directory / f"config-{config}" / f"inner-{inner}",
                data,
                protocol,
                arm,
                inner_train,
                inner_held,
                lr,
                wd,
                protocol["inner_seed"],
                None,
                ancestry,
            )
            probabilities[inner_held] = predicted
            epochs.append(receipt["selected_epoch"])
        candidates.append(
            {
                "config_index": config,
                "learning_rate": lr,
                "weight_decay": wd,
                "inner_epochs": epochs,
                "inner_metrics": probability_metrics(data["labels"][train], probabilities[train]),
            }
        )
    reconstructed = selection_from(candidates)
    if reconstructed != audit.read_json(directory / "selection.json"):
        raise RuntimeError("Episode selection cannot be reconstructed from all 12 inner fits")
    return reconstructed


def preflight(run, protocol):
    if not protocol["classifier_fitting_authorized"]:
        raise RuntimeError("Episode protocol does not authorize fitting")
    if (
        file_sha256(MEMORY_RUN / "execution_lock.json")
        != protocol["original_memory_execution_lock_sha256"]
    ):
        raise RuntimeError("Original memory execution lock changed")
    original_lock = audit.read_json(MEMORY_RUN / "execution_lock.json")
    source_files = {
        str(ROOT / relative): expected for relative, expected in original_lock["files"].items()
    }
    for path, expected in source_files.items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Frozen original source changed: {path}")
    proposal_path = ROOT / protocol["proposal_snapshot"]
    if file_sha256(proposal_path) != protocol["proposal_snapshot_sha256"]:
        raise RuntimeError("Original episode proposal snapshot changed")
    pilot_path = ROOT / protocol["resource_pilot"]
    pilot = audit.read_json(pilot_path)
    if (
        pilot["model_source_sha256"] != file_sha256(ROOT / "src/hac/center_episode_memory.py")
        or pilot["protocol_at_pilot_sha256"] != protocol["proposal_snapshot_sha256"]
    ):
        raise RuntimeError("Episode pilot does not match prototype/proposal")
    if (
        not pilot["synthetic_only"]
        or pilot["optimizer_steps"] != 0
        or pilot["dataset_rows_read"] != 0
    ):
        raise RuntimeError("Resource pilot was not synthetic-only")
    for arm in protocol["arms"]:
        result = pilot["results"][arm]
        if (
            not result["finite_loss"]
            or result["parameters"] != protocol["matched_capacity"]["parameters_per_arm"]
            or result["batch_size"] != protocol["batch_size"]
        ):
            raise RuntimeError("Episode GPU resource gate failed")
    data = memory_driver.load_data(MEMORY_RUN)
    if data["features"].shape != (4977, 3072):
        raise RuntimeError("Episode feature population or dimensionality changed")
    for row in range(len(data["labels"])):
        neighbors = data["neighbor_indices"][row, data["valid"][row]]
        for name in ("scenarios", "recordings", "tracks", "folds"):
            if not (data[name][neighbors] == data[name][row]).all():
                raise RuntimeError(f"Episode neighborhood crosses {name}")
    cache = memory_driver.base_cache(MEMORY_RUN, data, original_lock["protocol"])
    preparation = audit.read_json(MEMORY_RUN / "base_preparation.json")
    if cache.identity != preparation["base_identity"]:
        raise RuntimeError("Cached base identity changed; new base fitting is forbidden")
    populations = training_populations(data["labels"], data["scenarios"], data["folds"])
    keys = []
    for rows in populations.values():
        _, receipt = cache.load(rows)
        directory = cache.directory(rows)
        keys.append(directory.name)
        for filename, expected in (
            ("predictions.npz", receipt["predictions_sha256"]),
            ("checkpoint.npz", receipt["checkpoint_sha256"]),
        ):
            source_files[str(directory / filename)] = expected
        source_files[str(directory / "receipt.json")] = file_sha256(directory / "receipt.json")
    if sorted(keys) != sorted(preparation["population_cache_keys"]) or len(keys) != 58:
        raise RuntimeError("Nested base population coverage changed")
    # Independently verify exact P6 replay without calling a fitting/preparation method.
    with np.load(MEMORY_RUN / "base_replay.npz", allow_pickle=False) as saved:
        base_reference = saved["probabilities"]
        if not np.array_equal(saved["sample_ids"], data["sample_ids"]):
            raise RuntimeError("Base reference identity mismatch")
    for fold in protocol["outer_folds"]:
        train = np.flatnonzero(data["folds"] != fold)
        held = np.flatnonzero(data["folds"] == fold)
        if not np.array_equal(cache.held_predictions(train, held), base_reference[held]):
            raise RuntimeError("Frozen P6 outer replay is no longer exact")
    if not audit.read_json(MEMORY_RUN / "base_replay.json")["passed"]:
        raise RuntimeError("Prior P6 replay did not pass")
    files = [
        Path(__file__),
        PROTOCOL,
        proposal_path,
        pilot_path,
        ROOT / "experiments/pilot_okutama_center_episode.py",
        ROOT / "src/hac/center_episode_training.py",
        ROOT / "src/hac/center_episode_memory.py",
        ROOT / "src/hac/actor_memory_training.py",
        ROOT / "src/hac/actor_memory_base.py",
        ROOT / "src/hac/video_fusion.py",
        ROOT / "experiments/run_okutama_evidence_memory.py",
        ROOT / "experiments/run_okutama_video_probe.py",
        ROOT / "experiments/analyze_okutama_memory_results.py",
        ROOT / "tests/test_center_episode_memory.py",
        ROOT / "tests/test_center_episode_training.py",
        ROOT / "tests/test_center_episode_runner.py",
        MEMORY_RUN / "execution_lock.json",
        MEMORY_RUN / "base_replay.npz",
        MEMORY_RUN / "data/legacy_temporal_strata.npz",
        MEMORY_RUN / "data/legacy_temporal_strata_receipt.json",
        MEMORY_RUN / "results/summary.json",
        MEMORY_RUN / "results/oof_probabilities.npz",
        MEMORY_RUN / "diagnostics/access_accounting_clarification_v1.json",
    ]
    source_files.update({str(path): file_sha256(path) for path in files})
    receipt = {
        "protocol": protocol,
        "source_sha256": source_files,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "base_identity": cache.identity,
        "base_populations_verified": len(keys),
        "new_base_fits": 0,
        "p6_replay_maximum_absolute_difference": 0.0,
        "rows": len(data["labels"]),
        "features": 3072,
        "new_validation_or_protected_inputs": 0,
        "access_note": "Existing memory data only; original mixed descriptive metadata parsing is explained by the locked access-accounting sidecar",
    }
    write_immutable(run / ("preflight-" + canonical_hash(receipt) + ".json"), receipt)
    return data, cache, receipt


def lock_execution(run, preflight_receipt):
    path = run / "execution_lock.json"
    write_immutable(path, preflight_receipt)
    snapshot_root = run / "source_snapshot"
    for source, expected in preflight_receipt["source_sha256"].items():
        source = Path(source)
        if source.suffix not in (".py", ".json") or ".runs" in source.parts:
            continue
        destination = snapshot_root / source.relative_to(ROOT)
        if destination.exists():
            if file_sha256(destination) != expected:
                raise RuntimeError("Episode source snapshot changed")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return file_sha256(path)


def fit_fold(run, data, protocol, cache, arm, fold, execution_hash, *, fitter=fit_episode):
    train, held = np.flatnonzero(data["folds"] != fold), np.flatnonzero(data["folds"] == fold)
    directory = run / "models" / arm / f"fold-{fold}"
    splits = group_splits(
        data["labels"],
        data["scenarios"],
        train,
        n_splits=protocol["inner_folds"],
        seed=protocol["split_seed"],
    )
    # All meta packages are assembled strictly inside each allowed population.
    packages = [cache.meta_probabilities(inner_train) for inner_train, _ in splits]
    candidates = []
    configs = [(lr, wd) for lr in protocol["learning_rates"] for wd in protocol["weight_decays"]]
    for config_index, (lr, wd) in enumerate(configs):
        inner_oof = np.full((len(data["labels"]), 3), np.nan)
        epochs = []
        for inner, ((inner_train, inner_held), (probabilities, ancestry)) in enumerate(
            zip(splits, packages, strict=True)
        ):
            name = f"{arm}/fold-{fold}/config-{config_index}/inner-{inner}"
            event("episode_fit_started", fit=name)
            predicted, receipt = fitter(
                data,
                probabilities,
                inner_train,
                inner_held,
                arm=arm,
                learning_rate=lr,
                weight_decay=wd,
                seed=protocol["inner_seed"],
                protocol=protocol,
                ancestry=ancestry,
                directory=directory / f"config-{config_index}" / f"inner-{inner}",
                progress=lambda fit=name, **value: event("episode_epoch", fit=fit, **value),
            )
            inner_oof[inner_held] = predicted
            epochs.append(receipt["selected_epoch"])
            event(
                "episode_fit_complete",
                fit=name,
                seconds=receipt["seconds"],
                selected_epoch=receipt["selected_epoch"],
            )
        if not np.isfinite(inner_oof[train]).all() or np.isfinite(inner_oof[held]).any():
            raise RuntimeError("Episode inner selection coverage leaks outer rows")
        candidates.append(
            {
                "config_index": config_index,
                "learning_rate": lr,
                "weight_decay": wd,
                "inner_epochs": epochs,
                "inner_metrics": probability_metrics(data["labels"][train], inner_oof[train]),
            }
        )
    selection = selection_from(candidates)
    write_immutable(directory / "selection.json", selection)
    selected, epochs = selection["selected"], selection["refit_epochs"]
    probabilities, ancestry = cache.meta_probabilities(train)
    outputs = []
    for seed in protocol["outer_seeds"]:
        name = f"{arm}/fold-{fold}/refit-seed-{seed}"
        event("episode_fit_started", fit=name, epochs=epochs)
        predicted, receipt = fitter(
            data,
            probabilities,
            train,
            held,
            arm=arm,
            learning_rate=selected["learning_rate"],
            weight_decay=selected["weight_decay"],
            seed=seed,
            protocol=protocol,
            ancestry=ancestry,
            directory=directory / f"refit-seed-{seed}",
            epochs=epochs,
            progress=lambda fit=name, **value: event("episode_epoch", fit=fit, **value),
        )
        outputs.append(predicted)
        event("episode_fit_complete", fit=name, seconds=receipt["seconds"])
    path, marker = directory / "outer_predictions.npz", directory / "outer_complete.json"
    ensemble = np.mean(outputs, axis=0)
    manifest_path = directory / "completion_manifest.json"
    write_immutable(
        manifest_path, completion_manifest(directory, protocol, arm, fold, execution_hash)
    )
    manifest_hash = file_sha256(manifest_path)
    if marker.exists():
        completed = audit.read_json(marker)
        if (
            completed["execution_lock_sha256"] != execution_hash
            or completed["probabilities_sha256"] != file_sha256(path)
            or completed["completion_manifest_sha256"] != manifest_hash
        ):
            raise RuntimeError("Completed episode outer predictions changed")
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["probabilities"], ensemble) or not np.array_equal(
                saved["sample_ids"], data["sample_ids"][held]
            ):
                raise RuntimeError("Episode outer resume differs")
    else:
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (
                    not np.array_equal(saved["probabilities"], ensemble)
                    or not np.array_equal(saved["sample_ids"], data["sample_ids"][held])
                    or not np.array_equal(saved["seed_probabilities"], np.stack(outputs))
                ):
                    raise RuntimeError(
                        "Retained partial outer aggregation differs; refusing overwrite"
                    )
        else:
            np.savez_compressed(
                path,
                sample_ids=data["sample_ids"][held],
                held_rows=held,
                probabilities=ensemble,
                seed_probabilities=np.stack(outputs),
            )
        write_immutable(
            marker,
            {
                "arm": arm,
                "fold": fold,
                "execution_lock_sha256": execution_hash,
                "probabilities_sha256": file_sha256(path),
                "sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
                "completion_manifest_sha256": manifest_hash,
            },
        )
    event(
        "episode_outer_fold_complete",
        arm=arm,
        fold=fold,
        metrics=probability_metrics(data["labels"][held], ensemble),
    )


def completed_arm(run, data, protocol, arm, execution_hash, cache=None):
    fields = {name: [] for name in audit.SAVED_FIELDS}
    bounds, weights = [], []
    for fold in protocol["outer_folds"]:
        directory = run / "models" / arm / f"fold-{fold}"
        marker = directory / "outer_complete.json"
        if not marker.exists():
            return None
        completed = audit.read_json(marker)
        path = directory / "outer_predictions.npz"
        if (
            completed["arm"] != arm
            or completed["fold"] != fold
            or completed["execution_lock_sha256"] != execution_hash
            or completed["probabilities_sha256"] != file_sha256(path)
        ):
            raise RuntimeError("Episode completed marker identity/hash mismatch")
        held, train = np.flatnonzero(data["folds"] == fold), np.flatnonzero(data["folds"] != fold)
        manifest_path = directory / "completion_manifest.json"
        if completed["completion_manifest_sha256"] != file_sha256(manifest_path) or audit.read_json(
            manifest_path
        ) != completion_manifest(directory, protocol, arm, fold, execution_hash):
            raise RuntimeError("Episode immutable completion evidence changed")
        if cache is None:
            raise RuntimeError(
                "Read-only nested cache is required to audit completed episode ancestry"
            )
        selection = validate_selection_evidence(directory, data, protocol, arm, fold, cache)
        outer_ancestry = cache.meta_probabilities(train)[1]
        seed_values = {name: [] for name in fields}
        for seed in protocol["outer_seeds"]:
            seed_dir = directory / f"refit-seed-{seed}"
            receipt = audit.read_json(seed_dir / "receipt.json")
            request = receipt["request"]
            validate_fit_evidence(
                seed_dir,
                data,
                protocol,
                arm,
                train,
                held,
                selection["selected"]["learning_rate"],
                selection["selected"]["weight_decay"],
                seed,
                selection["refit_epochs"],
                outer_ancestry,
            )
            expected = {
                "arm": arm,
                "seed": seed,
                "epochs": selection["refit_epochs"],
                "protocol_sha256": canonical_hash(protocol),
                "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
                "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
                "train_boundary_targets_sha256": canonical_hash(
                    data["boundary_targets"][train].tolist()
                ),
                "train_boundary_valid_sha256": canonical_hash(
                    data["boundary_valid"][train].tolist()
                ),
                "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
                "selection_labels_sha256": None,
                "data_sha256": data["_file_sha256"],
                "learning_rate": selection["selected"]["learning_rate"],
                "weight_decay": selection["selected"]["weight_decay"],
                "training_code_sha256": file_sha256(ROOT / "src/hac/center_episode_training.py"),
                "architecture_code_sha256": file_sha256(ROOT / "src/hac/center_episode_memory.py"),
                "boundary_loss": "unweighted_masked_BCE_with_logits",
                "base_ancestry_sha256": canonical_hash(receipt["base_ancestry"]),
            }
            if (
                any(request[key] != value for key, value in expected.items())
                or receipt["boundary_positive_weight"] is not None
            ):
                raise RuntimeError("Episode outer receipt contract mismatch")
            if (
                receipt["train_rows"] != train.tolist()
                or receipt["held_rows"] != held.tolist()
                or receipt["selected_epoch"] != selection["refit_epochs"]
                or receipt["epochs_run"] != selection["refit_epochs"]
                or any("validation_metrics" in row for row in receipt["history"])
            ):
                raise RuntimeError(
                    "Episode outer refit used unexpected rows or validation/epoch selection"
                )
            for filename, expected_hash in receipt["output_sha256"].items():
                if file_sha256(seed_dir / filename) != expected_hash:
                    raise RuntimeError("Episode seed output changed")
            with np.load(seed_dir / "predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(
                    saved["sample_ids"], data["sample_ids"][held]
                ) or not np.array_equal(saved["held_rows"], held):
                    raise RuntimeError("Episode seed identity mismatch")
                for name in fields:
                    seed_values[name].append(saved[name])
            bounds.append({"fold": fold, "seed": seed, **receipt["coefficient_bound_audit"]})
            weights.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "boundary_positive_weight": None,
                    "class_weight": receipt["class_weight"],
                }
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                not np.array_equal(saved["sample_ids"], data["sample_ids"][held])
                or not np.array_equal(
                    saved["seed_probabilities"], np.stack(seed_values["probabilities"])
                )
                or not np.array_equal(
                    saved["probabilities"], np.mean(seed_values["probabilities"], axis=0)
                )
            ):
                raise RuntimeError("Episode outer ensemble differs from exact three-seed mean")
        for name in fields:
            fields[name].append((held, np.stack(seed_values[name])))
    result = {}
    for name, pieces in fields.items():
        shape = pieces[0][1].shape
        combined = np.empty((shape[0], len(data["labels"]), *shape[2:]), dtype=pieces[0][1].dtype)
        for held, values in pieces:
            combined[:, held] = values
        result[name] = combined
    result["rows"] = np.arange(len(data["labels"]))
    expected_receipts = {
        path / "receipt.json"
        for fold in protocol["outer_folds"]
        for path in fit_directories(run / "models" / arm / f"fold-{fold}", protocol)
    }
    if (
        set((run / "models" / arm).rglob("receipt.json")) != expected_receipts
        or len(expected_receipts) != 75
    ):
        raise RuntimeError("A complete episode arm must have exactly 75 verified fit receipts")
    result["bound_audits"], result["training_weights"] = bounds, weights
    return result


def summarize(run, data, protocol, execution_hash, cache):
    with np.load(MEMORY_RUN / "base_replay.npz", allow_pickle=False) as saved:
        baseline = {name: saved[name] for name in ("probabilities", "components")}
    with np.load(MEMORY_RUN / "data/legacy_temporal_strata.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], data["sample_ids"]):
            raise RuntimeError("Legacy episode comparison strata are misaligned")
        legacy = {name: saved[name] for name in saved.files if name != "sample_ids"}
    with np.load(MEMORY_RUN / "results/oof_probabilities.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], data["sample_ids"]):
            raise RuntimeError("Existing descriptive reference identities changed")
        descriptive_references = {
            name: saved[name] for name in protocol["descriptive_references_only"]
        }
    summaries, completed, tables = (
        {},
        {},
        {name: [] for name in ("rows", "physical_edges", "scenario", "recording", "track")},
    )
    for arm in protocol["arms"]:
        values = completed_arm(run, data, protocol, arm, execution_hash, cache)
        if values is None:
            continue
        completed[arm] = values["probabilities"].mean(0)
        result, rows, edges, groups = audit.analyze_arm(arm, values, data, baseline, legacy)
        result["boundary"]["calibration_caution"] = (
            "Uniform unweighted proper BCE with classification-detached hazard; finite-sample calibration is not guaranteed"
        )
        result["coefficient_bound_audits"] = values["bound_audits"]
        result["training_weights"] = values["training_weights"]
        result["seeds"] = protocol["outer_seeds"]
        occurrence_probabilities = values["boundary_probabilities"].mean(0)
        result["boundary"]["by_physical_duration_seconds"] = {}
        for dt in np.unique(data["edge_dt_seconds"][data["boundary_valid"]]):
            mask = data["boundary_valid"] & (data["edge_dt_seconds"] == dt)
            selected_edges = [edge for edge in edges if edge["dt_seconds"] == float(dt)]
            result["boundary"]["by_physical_duration_seconds"][str(float(dt))] = {
                "occurrence_weighted": audit.binary_calibration(
                    data["boundary_targets"][mask], occurrence_probabilities[mask]
                ),
                "physical_edge_deduplicated": audit.binary_calibration(
                    [edge["boundary_target"] for edge in selected_edges],
                    [edge["mean_boundary_probability"] for edge in selected_edges],
                ),
            }
        summaries[arm] = result
        tables["rows"].extend(rows)
        tables["physical_edges"].extend(edges)
        for name, values in groups.items():
            tables[name].extend(values)
    contrast = None
    if all(arm in completed for arm in protocol["primary_contrast"]):
        candidate, reference = (completed[arm] for arm in protocol["primary_contrast"])
        contrast = statistics.paired_statistics(
            data["labels"],
            candidate,
            reference,
            data["scenarios"],
            bootstrap_resamples=protocol["statistics"]["bootstrap_resamples"],
            bootstrap_seed=protocol["statistics"]["seed"],
        )
    shared = (baseline["components"].argmax(2) != data["labels"][:, None]).all(1)
    masks = {"all": np.ones(len(data["labels"]), bool), **legacy}
    for prefix in ("node_interval", "memory_interval"):
        complete, boundary = data[prefix + "_complete"], data[prefix + "_boundary"]
        masks[prefix + "_complete_boundary"] = complete & boundary
        masks[prefix + "_complete_stable"] = complete & ~boundary
    for arm, result in summaries.items():
        references = {
            **descriptive_references,
            **{name: p for name, p in completed.items() if name != arm},
        }
        result["descriptive_comparisons"] = {}
        for name, reference in references.items():
            rows = {}
            for stratum, mask in masks.items():
                compared = audit.compare(data["labels"], completed[arm], reference, shared, mask)
                if "p6_metrics" in compared:
                    compared["reference_metrics"] = compared.pop("p6_metrics")
                    compared["reference_errors"] = compared.pop("p6_errors")
                rows[stratum] = compared
            result["descriptive_comparisons"][name] = rows
        limits = protocol["engineering_checks"]
        scenario_rows = [row for row in tables["scenario"] if row["arm"] == arm]
        scenario_delta = [row["accuracy_delta"] for row in scenario_rows]
        legacy_boundary = result["strata"]["legacy_complete_target_boundary"]
        result["engineering_checks"] = {
            "target_84_macro_f1": result["metrics"]["macro_f1"] >= limits["target_macro_f1"],
            "stretch_85_macro_f1": result["metrics"]["macro_f1"] >= limits["stretch_macro_f1"],
            "positive_net_corrections": result["rescues"] > result["harms"],
            "shared_repairs": result["shared_failure_repairs"]
            >= limits["shared_failure_repairs_minimum"],
            "scenario_breadth": sum(delta > 0 for delta in scenario_delta)
            >= limits["improved_scenarios_minimum"],
            "worst_scenario": min(scenario_delta)
            >= limits["worst_scenario_accuracy_delta_minimum"],
            "boundary_harm_reduced": legacy_boundary["harms"]
            < limits["legacy_boundary_harms_less_than"],
            "stable_net_positive": result["strata"]["legacy_complete_stable_target_window"][
                "net_corrections"
            ]
            > 0,
            "nll_no_worse": result["metrics"]["nll"] <= result["p6_metrics"]["nll"],
            "brier_no_worse": result["metrics"]["brier"] <= result["p6_metrics"]["brier"],
        }
        if arm == "expected_episode":
            result["engineering_checks"]["beats_matched_control_macro_f1"] = (
                "log_survival_control" in summaries
                and result["metrics"]["macro_f1"]
                > summaries["log_survival_control"]["metrics"]["macro_f1"]
            )
    # New versioned results only, including partial-arm completion explicitly.
    sequence = 1
    while (run / "results" / f"v{sequence:04d}").exists():
        sequence += 1
    output = run / "results" / f"v{sequence:04d}"
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        audit.write_csv(output / (name + ".csv"), rows)
    np.savez_compressed(
        output / "oof_probabilities.npz",
        sample_ids=data["sample_ids"],
        labels=data["labels"],
        scenarios=data["scenarios"],
        folds=data["folds"],
        **completed,
    )
    result = {
        "scope": protocol["status_note"],
        "complete": len(completed) == len(protocol["arms"]),
        "rows_per_complete_arm": len(data["labels"]),
        "missing_arms": [arm for arm in protocol["arms"] if arm not in completed],
        "verified_fit_receipts": 75 * len(completed),
        "execution_lock_sha256": execution_hash,
        "results": summaries,
        "primary_contrast": contrast,
        "artifacts_sha256": {
            path.name: file_sha256(path) for path in output.iterdir() if path.is_file()
        },
        "note": "All-row three-seed mean; no partial-fold projection, global-OOF fitting or new base fits",
    }
    write_immutable(output / "summary.json", result)
    event(
        "episode_summary",
        output_dir=str(output),
        complete=result["complete"],
        metrics={arm: value["metrics"] for arm, value in summaries.items()},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "summarize"), required=True)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int)
    args = parser.parse_args()
    protocol = audit.read_json(PROTOCOL)
    arms = protocol["arms"] if args.arms is None else args.arms
    folds = protocol["outer_folds"] if args.folds is None else args.folds
    if (
        len(arms) != len(set(arms))
        or not set(arms).issubset(protocol["arms"])
        or len(folds) != len(set(folds))
        or not set(folds).issubset(protocol["outer_folds"])
    ):
        raise RuntimeError("Undeclared or duplicate arm/fold requested")
    torch.set_num_threads(2)
    data, cache, receipt = preflight(args.run, protocol)
    event("episode_preflight_passed", rows=4977, base_populations=58, new_base_fits=0)
    if args.stage == "preflight":
        return
    execution_hash = lock_execution(args.run, receipt)
    if args.stage == "train":
        for arm in arms:
            for fold in folds:
                fit_fold(args.run, data, protocol, cache, arm, fold, execution_hash)
            event("episode_requested_arm_folds_complete", arm=arm)
    summarize(args.run, data, protocol, execution_hash, cache)


if __name__ == "__main__":
    main()
