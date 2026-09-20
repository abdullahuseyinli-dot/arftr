"""Strict loaders for nested M4 and A3 cached prediction ancestry.

The loaders never expose labels from a held outer fold as training inputs.  They
bind selected inner OOF predictions and the three corresponding outer refits to
the canonical source-swap row identity.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hac.actor_memory_base import canonical_hash, file_sha256


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _require_under(path: Path, root: Path, description: str) -> Path:
    resolved, base = path.resolve(), root.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise RuntimeError(f"{description} escapes its locked artifact root: {resolved}") from error
    return resolved


def _validate_probability(values: np.ndarray, rows: int, description: str) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.shape != (rows, 3)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(1), 1.0, atol=1e-6, rtol=0)
    ):
        raise RuntimeError(f"Malformed probability simplex: {description}")
    return values


def load_cached_study_data(source_root: Path) -> dict[str, np.ndarray]:
    """Load and validate the canonical 4,977-row source-swap memory dataset."""

    source_root = Path(source_root).resolve()
    path = source_root / "data/memory_data.npz"
    data = _read_npz(path)
    required = {"sample_ids", "labels", "scenarios", "folds"}
    if not required.issubset(data):
        raise RuntimeError("Cached study data is missing identity arrays")
    rows = len(data["labels"])
    if (
        rows != 4977
        or any(data[name].shape != (rows,) for name in required)
        or len(np.unique(data["sample_ids"])) != rows
        or not np.array_equal(np.unique(data["labels"]), np.arange(3))
        or not np.array_equal(np.unique(data["folds"]), np.arange(5))
    ):
        raise RuntimeError("Cached study identity is not the locked 4,977-row five-fold cohort")
    for fold in range(5):
        train = data["folds"] != fold
        held = ~train
        if set(data["scenarios"][train].tolist()) & set(data["scenarios"][held].tolist()):
            raise RuntimeError("Outer folds are not scenario-disjoint")
    return data


@dataclass(frozen=True)
class SelectedFoldArtifacts:
    """Selected nested caches for one outer fold, in canonical row order."""

    fold: int
    train_rows: np.ndarray
    held_rows: np.ndarray
    m4_inner: dict[str, np.ndarray]
    a3_inner: dict[str, np.ndarray]
    m4_outer: dict[str, np.ndarray]
    a3_outer: dict[str, np.ndarray]
    input_paths: tuple[Path, ...]


def _selected_a3_receipts(
    sear_root: Path, fold: int
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    workload_path = sear_root / f"workloads/a3_unrestricted_templates/fold-{fold}/receipt.json"
    workload = _read_json(workload_path)
    if (
        workload.get("status") != "SEAR_WORKLOAD_COMPLETE"
        or workload.get("request", {}).get("arm") != "a3_unrestricted_templates"
        or workload.get("request", {}).get("fold") != fold
        or workload.get("unique_fits") != 15
        or len(workload.get("fit_receipts", ())) != 15
    ):
        raise RuntimeError(f"A3 workload {fold} is incomplete or has changed")
    selected = workload["selected"]
    result = []
    for reference in workload["fit_receipts"]:
        receipt_path = _require_under(Path(reference["path"]), sear_root / "fits", "A3 fit receipt")
        if (
            not receipt_path.is_file()
            or receipt_path.name != "receipt.json"
            or receipt_path.stat().st_size != reference["size_bytes"]
            or file_sha256(receipt_path) != reference["sha256"]
        ):
            raise RuntimeError("A3 workload fit-receipt ancestry changed")
        receipt = _read_json(receipt_path)
        request = receipt["request"]
        if (
            request.get("stage") == "inner"
            and request.get("outer_fold") == fold
            and request.get("learning_rate") == selected["learning_rate"]
            and request.get("weight_decay") == selected["weight_decay"]
        ):
            result.append((receipt_path, receipt))
    result.sort(key=lambda item: item[1]["request"]["inner_fold"])
    if [item[1]["request"]["inner_fold"] for item in result] != [0, 1, 2]:
        raise RuntimeError("A3 selected inner ancestry is not exactly three folds")
    return workload, result


def selected_input_paths(
    root: Path,
    source_root: Path,
    sear_root: Path,
    data: dict[str, np.ndarray],
    *,
    seeds: Iterable[int] = (42, 43, 44),
) -> tuple[Path, ...]:
    """Enumerate every selected cache/receipt used by all outer folds."""

    root, source_root, sear_root = map(lambda value: Path(value).resolve(), (root, source_root, sear_root))
    paths: set[Path] = {
        source_root / "data/memory_data.npz",
        source_root / "results/v0001/oof_probabilities.npz",
        sear_root / "results/v0001/oof_probabilities.npz",
        sear_root / "execution_lock.json",
    }
    seed_tuple = tuple(int(seed) for seed in seeds)
    if seed_tuple != (42, 43, 44):
        raise RuntimeError("Selected outer-seed contract changed")
    sear_lock = _read_json(sear_root / "execution_lock.json")
    if tuple(sear_lock["protocol"]["training"]["outer_seeds"]) != seed_tuple:
        raise RuntimeError("A3 workload seed order changed")
    for fold in range(5):
        source_fold = source_root / f"models/survival_memory/fold-{fold}"
        selection_path = source_fold / "selection.json"
        completion_path = source_fold / "completion_manifest.json"
        paths.update((selection_path, completion_path))
        selection = _read_json(selection_path)["selected"]
        config = selection["config_index"]
        for inner in range(3):
            directory = source_fold / f"config-{config}/inner-{inner}"
            paths.update((directory / "receipt.json", directory / "predictions.npz"))
        for seed in seed_tuple:
            directory = source_fold / f"refit-seed-{seed}"
            paths.update((directory / "receipt.json", directory / "predictions.npz"))
        workload_dir = sear_root / f"workloads/a3_unrestricted_templates/fold-{fold}"
        workload_path = workload_dir / "receipt.json"
        paths.update((workload_path, workload_dir / "predictions.npz"))
        _, receipts = _selected_a3_receipts(sear_root, fold)
        for receipt_path, receipt in receipts:
            prediction = _require_under(
                Path(receipt["artifacts"]["predictions.npz"]["path"]),
                sear_root / "fits",
                "A3 selected prediction",
            )
            paths.update((receipt_path, prediction))
    for path in paths:
        _require_under(path, root, "Selected cached input")
        if not path.is_file():
            raise RuntimeError(f"Missing selected cached input: {path}")
    if canonical_hash(data["sample_ids"].tolist()) == "":
        raise AssertionError("Unreachable identity guard")
    return tuple(sorted((path.resolve() for path in paths), key=str))


def selected_fold_artifacts(
    root: Path,
    source_root: Path,
    sear_root: Path,
    data: dict[str, np.ndarray],
    fold: int,
    *,
    seeds: Iterable[int] = (42, 43, 44),
) -> SelectedFoldArtifacts:
    """Load one fold with hash, population, identity, and seed-order checks."""

    root, source_root, sear_root = map(lambda value: Path(value).resolve(), (root, source_root, sear_root))
    if fold not in range(5):
        raise ValueError("fold must be in [0, 4]")
    seed_tuple = tuple(int(seed) for seed in seeds)
    if seed_tuple != (42, 43, 44):
        raise RuntimeError("Selected outer-seed contract changed")
    train = np.flatnonzero(data["folds"] != fold)
    held = np.flatnonzero(data["folds"] == fold)
    id_index = {sample_id: row for row, sample_id in enumerate(data["sample_ids"].tolist())}
    if len(id_index) != len(data["sample_ids"]):
        raise RuntimeError("Sample IDs are not unique")

    source_fold = source_root / f"models/survival_memory/fold-{fold}"
    selection_path = source_fold / "selection.json"
    completion_path = source_fold / "completion_manifest.json"
    selection_document = _read_json(selection_path)
    selection = selection_document["selected"]
    config = selection["config_index"]
    completion = _read_json(completion_path)
    if (
        completion.get("fold") != fold
        or completion.get("fit_receipts") != 15
        or completion.get("sample_ids_sha256") != canonical_hash(data["sample_ids"][held].tolist())
    ):
        raise RuntimeError("M4 fold completion identity changed")
    completion_files = {str(Path(name).resolve()): value for name, value in completion["files_sha256"].items()}

    m4_fields = ("probabilities", "gate", "attention", "survival", "boundary_probabilities")
    m4_inner_full = {
        "probabilities": np.full((len(data["labels"]), 3), np.nan),
        "gate": np.full(len(data["labels"]), np.nan),
        "attention": np.full((len(data["labels"]), 5), np.nan),
        "survival": np.full((len(data["labels"]), 5), np.nan),
        "boundary_probabilities": np.full((len(data["labels"]), 5), np.nan),
    }
    source_inner_paths: list[Path] = []
    covered: list[np.ndarray] = []
    for inner in range(3):
        directory = source_fold / f"config-{config}/inner-{inner}"
        receipt_path, prediction_path = directory / "receipt.json", directory / "predictions.npz"
        for path in (receipt_path, prediction_path):
            if completion_files.get(str(path.resolve())) != file_sha256(path):
                raise RuntimeError("M4 selected inner file is not bound by completion manifest")
        receipt = _read_json(receipt_path)
        request = receipt["request"]
        rows = np.asarray(receipt["held_rows"], dtype=np.int64)
        train_rows = np.asarray(receipt["train_rows"], dtype=np.int64)
        if (
            request.get("arm") != "survival_memory"
            or request.get("learning_rate") != selection["learning_rate"]
            or request.get("weight_decay") != selection["weight_decay"]
            or request.get("seed") != 42
            or request.get("selection_labels_sha256") is None
            or request.get("train_ids_sha256") != canonical_hash(data["sample_ids"][train_rows].tolist())
            or request.get("train_labels_sha256") != canonical_hash(data["labels"][train_rows].tolist())
            or request.get("held_ids_sha256") != canonical_hash(data["sample_ids"][rows].tolist())
            or request.get("selection_labels_sha256") != canonical_hash(data["labels"][rows].tolist())
            or np.any(data["folds"][rows] == fold)
        ):
            raise RuntimeError("M4 selected inner receipt population changed")
        output = _read_npz(prediction_path)
        if (
            receipt["output_sha256"]["predictions.npz"] != file_sha256(prediction_path)
            or not np.array_equal(output["held_rows"], rows)
            or not np.array_equal(output["sample_ids"], data["sample_ids"][rows])
        ):
            raise RuntimeError("M4 selected inner output identity changed")
        _validate_probability(output["probabilities"], len(rows), "M4 inner")
        for name in m4_fields:
            m4_inner_full[name][rows] = output[name]
        covered.append(rows)
        source_inner_paths.extend((receipt_path, prediction_path))
    if not np.array_equal(np.sort(np.concatenate(covered)), train):
        raise RuntimeError("M4 selected inner OOF predictions do not exactly cover outer training")
    m4_inner = {name: values[train] for name, values in m4_inner_full.items()}
    if any(not np.isfinite(values).all() for values in m4_inner.values()):
        raise RuntimeError("M4 selected inner OOF output is incomplete")

    m4_outer_lists: dict[str, list[np.ndarray]] = {name: [] for name in m4_fields}
    source_outer_paths: list[Path] = []
    for seed in seed_tuple:
        directory = source_fold / f"refit-seed-{seed}"
        receipt_path, prediction_path = directory / "receipt.json", directory / "predictions.npz"
        for path in (receipt_path, prediction_path):
            if completion_files.get(str(path.resolve())) != file_sha256(path):
                raise RuntimeError("M4 outer refit is not bound by completion manifest")
        receipt, output = _read_json(receipt_path), _read_npz(prediction_path)
        request = receipt["request"]
        receipt_train = np.asarray(receipt["train_rows"], dtype=np.int64)
        receipt_held = np.asarray(receipt["held_rows"], dtype=np.int64)
        if (
            request.get("seed") != seed
            or request.get("learning_rate") != selection["learning_rate"]
            or request.get("weight_decay") != selection["weight_decay"]
            or request.get("selection_labels_sha256") is not None
            or not np.array_equal(receipt_train, train)
            or not np.array_equal(receipt_held, held)
            or request.get("train_ids_sha256") != canonical_hash(data["sample_ids"][train].tolist())
            or request.get("train_labels_sha256") != canonical_hash(data["labels"][train].tolist())
            or request.get("held_ids_sha256") != canonical_hash(data["sample_ids"][held].tolist())
            or receipt["output_sha256"]["predictions.npz"] != file_sha256(prediction_path)
            or not np.array_equal(output["held_rows"], held)
            or not np.array_equal(output["sample_ids"], data["sample_ids"][held])
        ):
            raise RuntimeError("M4 selected outer refit identity changed")
        _validate_probability(output["probabilities"], len(held), "M4 outer")
        for name in m4_fields:
            m4_outer_lists[name].append(output[name])
        source_outer_paths.extend((receipt_path, prediction_path))
    m4_outer = {name: np.stack(values) for name, values in m4_outer_lists.items()}

    workload, a3_receipts = _selected_a3_receipts(sear_root, fold)
    workload_path = sear_root / f"workloads/a3_unrestricted_templates/fold-{fold}/receipt.json"
    workload_prediction = workload_path.parent / "predictions.npz"
    artifact = workload["artifact"]
    if (
        _require_under(Path(artifact["path"]), workload_path.parent, "A3 workload output")
        != workload_prediction.resolve()
        or workload_prediction.stat().st_size != artifact["size_bytes"]
        or file_sha256(workload_prediction) != artifact["sha256"]
    ):
        raise RuntimeError("A3 outer workload artifact changed")
    a3_probability = np.full((len(data["labels"]), 3), np.nan)
    a3_slots = np.full((len(data["labels"]), 6), np.nan)
    a3_inner_paths: list[Path] = []
    covered = []
    for receipt_path, receipt in a3_receipts:
        request = receipt["request"]
        prediction_record = receipt["artifacts"]["predictions.npz"]
        prediction_path = _require_under(
            Path(prediction_record["path"]), sear_root / "fits", "A3 selected inner output"
        )
        if (
            prediction_path.stat().st_size != prediction_record["size_bytes"]
            or file_sha256(prediction_path) != prediction_record["sha256"]
            or request.get("train_ids_sha256") is None
            or request.get("train_labels_sha256") is None
            or request.get("held_ids_sha256") is None
            or request.get("selection_labels_sha256") is None
        ):
            raise RuntimeError("A3 selected inner artifact ancestry changed")
        output = _read_npz(prediction_path)
        try:
            rows = np.asarray([id_index[value] for value in output["sample_ids"].tolist()])
        except KeyError as error:
            raise RuntimeError("A3 selected output contains an unknown sample ID") from error
        if (
            len(np.unique(rows)) != len(rows)
            or np.any(data["folds"][rows] == fold)
            or request["held_ids_sha256"] != canonical_hash(data["sample_ids"][rows].tolist())
            or request["selection_labels_sha256"] != canonical_hash(data["labels"][rows].tolist())
            or int(receipt["held_rows"]) != len(rows)
        ):
            raise RuntimeError("A3 selected inner population changed")
        _validate_probability(output["probabilities"], len(rows), "A3 inner")
        if output["slot_masses"].shape != (len(rows), 6) or not np.isfinite(output["slot_masses"]).all():
            raise RuntimeError("A3 selected inner slot evidence changed")
        a3_probability[rows] = output["probabilities"]
        a3_slots[rows] = output["slot_masses"]
        covered.append(rows)
        a3_inner_paths.extend((receipt_path, prediction_path))
    if not np.array_equal(np.sort(np.concatenate(covered)), train):
        raise RuntimeError("A3 selected inner OOF predictions do not exactly cover outer training")
    a3_inner = {"probabilities": a3_probability[train], "slot_masses": a3_slots[train]}
    outer = _read_npz(workload_prediction)
    if (
        not np.array_equal(outer["sample_ids"], data["sample_ids"][held])
        or outer["seed_probabilities"].shape != (3, len(held), 3)
        or outer["seed_slot_masses"].shape != (3, len(held), 6)
    ):
        raise RuntimeError("A3 selected outer output identity or seed shape changed")
    for seed_index in range(3):
        _validate_probability(outer["seed_probabilities"][seed_index], len(held), "A3 outer")
    a3_outer = {
        "probabilities": outer["seed_probabilities"],
        "slot_masses": outer["seed_slot_masses"],
    }
    fold_paths = (
        selection_path,
        completion_path,
        *source_inner_paths,
        *source_outer_paths,
        workload_path,
        workload_prediction,
        *a3_inner_paths,
    )
    for path in fold_paths:
        _require_under(path, root, "Selected fold artifact")
    return SelectedFoldArtifacts(
        fold=fold,
        train_rows=train,
        held_rows=held,
        m4_inner=m4_inner,
        a3_inner=a3_inner,
        m4_outer=m4_outer,
        a3_outer=a3_outer,
        input_paths=tuple(sorted({path.resolve() for path in fold_paths}, key=str)),
    )
