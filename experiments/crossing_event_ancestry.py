"""Read-only ancestry validation for reuse of the completed PDI producers.

No model is fitted or replayed here.  Explicit populations, scenario exclusions,
selection-label scopes and request hashes are checked recursively.  Historical
checkpoint bytes are pinned, not executed; this distinction is in the result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PDI = ROOT / ".runs/research_20260919/paired_detail_innovation_v1_full"
INNER = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/inner_ancestors"
GENERAL = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v2"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1"
READERS = ("coarse_residual", "paired_spatial", "paired_spatial_temporal")
BASE_ROOTS = (
    INNER / "base_cache",
    ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v1/base_cache",
    ROOT / ".runs/research_20260908/source_swap_v1/base_cache",
    GENERAL / "base_cache",
)


def _require(value, message):
    if not value:
        raise RuntimeError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rows(values, size, label):
    a = np.asarray(values)
    _require(a.ndim == 1 and a.dtype.kind in "iu", f"{label}: invalid row representation")
    a = a.astype(np.int64)
    _require(len(a) > 0 and np.array_equal(a, np.unique(a)), f"{label}: empty/duplicate/unsorted rows")
    _require(a.min() >= 0 and a.max() < size, f"{label}: rows outside cohort")
    return a


def check_scoped_fit(train, held, allowed, scenarios, selection_labels):
    """Validate a nested fit without assuming outside predictions were selected on."""
    train = _rows(train, len(scenarios), "train")
    held = _rows(held, len(scenarios), "held")
    allowed = _rows(allowed, len(scenarios), "allowed")
    _require(np.isin(train, allowed).all(), "training rows escape allowed population")
    _require(not np.intersect1d(train, held).size, "train/held row overlap")
    _require(not set(scenarios[train]) & set(scenarios[held]), "train/held scenario overlap")
    if selection_labels is not None:
        _require(np.isin(held, allowed).all(), "selection labels escape allowed population")
        _require(np.array_equal(np.union1d(train, held), allowed), "selection split does not partition allowed population")
    else:
        _require(np.array_equal(train, allowed), "final no-selection fit does not use entire allowed population")
    return train, held


def check_base_scope(request, selection_splits, predicted_rows, allowed, ids, labels, scenarios):
    """Validate P6's final fit and every model-selection fold in that fit."""
    train = _rows(request["train_rows"], len(ids), "base train")
    pred = _rows(predicted_rows, len(ids), "base predicted")
    _require(np.isin(train, allowed).all(), "base training rows escape producer population")
    _require(not set(scenarios[train]) & set(scenarios[pred]), "base predicted scenario occurs in training")
    expected = {
        "all_sample_ids_sha256": _canonical(ids.tolist()),
        "train_sample_ids_sha256": _canonical(ids[train].tolist()),
        "train_labels_sha256": _canonical(labels[train].tolist()),
        "train_scenarios_sha256": _canonical(scenarios[train].tolist()),
        "train_scenarios": sorted(set(scenarios[train].tolist())),
    }
    for key, value in expected.items():
        _require(request[key] == value, f"base {key} mismatch")
    held_parts = []
    for split in selection_splits:
        _, held = check_scoped_fit(split["train"], split["held"], train, scenarios, "selection")
        held_parts.append(held)
    _require(bool(held_parts), "missing base selection folds")
    _require(np.array_equal(np.sort(np.concatenate(held_parts)), train), "base selection held folds lack exact-once coverage")
    return train


class _Recorder:
    def __init__(self):
        self.records = {}

    def add(self, path, sha=None, size=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        _require(path.is_relative_to(ROOT.resolve()), f"dependency outside repository: {path}")
        key = path.relative_to(ROOT).as_posix()
        if key not in self.records:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.records[key] = {"path": key, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        record = self.records[key]
        _require(sha is None or record["sha256"] == sha, f"dependency checksum mismatch: {key}")
        _require(size is None or record["bytes"] == size, f"dependency size mismatch: {key}")
        return record

    def reference(self, record):
        return self.add(record["path"], record.get("sha256"), record.get("bytes", record.get("size_bytes")))


def audit_source_ancestry() -> dict:
    recorder = _Recorder()
    pdi_lock = _json(PDI / "execution_lock.json")
    pdi_lock_hash = recorder.add(PDI / "execution_lock.json")["sha256"]
    _require("LOCKED_BEFORE" in pdi_lock["status"], "PDI lacks pre-fit lock")
    for dep in pdi_lock["dependencies"]:
        recorder.reference(dep)
    anchor_path = ARFTR / "results/v0001/oof_probabilities.npz"
    recorder.add(anchor_path)
    with np.load(anchor_path, allow_pickle=False) as f:
        ids, labels, scenarios, folds = (f[k].copy() for k in ("sample_ids", "labels", "scenarios", "folds"))
    _require(scenarios.dtype.kind in "US", "scenario identity must remain strings")
    _require(len(ids) == 4977 and len(set(ids)) == 4977, "canonical cohort changed")
    _require(pdi_lock["sample_ids_sha256"] == _canonical(ids.tolist()), "PDI cohort hash mismatch")
    all_rows = np.arange(len(ids))
    counters = {"pdi_producers": 0, "inner_arftr_populations": 0, "outer_training_prior_populations": 0,
                "m4_receipts": 0, "a3_receipts": 0, "base_ancestry_edges": 0}
    bases_seen, populations_seen = set(), set()

    def base_edges(ancestry, allowed):
        coverage = []
        for edge in ancestry:
            key = edge["base_cache"]
            candidates = [p / key for p in BASE_ROOTS if (p / key / "receipt.json").is_file()]
            _require(bool(candidates), f"missing referenced base cache: {key}")
            directory = candidates[0]
            receipt = _json(directory / "receipt.json")
            _require(_canonical(receipt["request"]) == key, "base directory does not hash its request")
            check_base_scope(receipt["request"], receipt["selection_splits"], edge["predicted_rows"],
                             allowed, ids, labels, scenarios)
            recorder.add(directory / "receipt.json")
            recorder.add(directory / "predictions.npz", receipt["predictions_sha256"])
            recorder.add(directory / "checkpoint.npz", receipt["checkpoint_sha256"])
            for other in candidates[1:]:
                duplicate = _json(other / "receipt.json")
                _require(duplicate["request"] == receipt["request"], "duplicate cache key has different request")
                _require(duplicate["predictions_sha256"] == receipt["predictions_sha256"], "duplicate base cache predictions diverge")
            coverage.append(np.asarray(edge["predicted_rows"], dtype=np.int64))
            bases_seen.add(key)
            counters["base_ancestry_edges"] += 1
        _require(np.array_equal(np.sort(np.concatenate(coverage)), all_rows), "base ancestry does not cover cohort exactly once")

    def request_rows(request, train, held):
        expected = {"train_ids_sha256": _canonical(ids[train].tolist()),
                    "train_labels_sha256": _canonical(labels[train].tolist()),
                    "held_ids_sha256": _canonical(ids[held].tolist())}
        if request["selection_labels_sha256"] is not None:
            expected["selection_labels_sha256"] = _canonical(labels[held].tolist())
        for key, value in expected.items():
            _require(request[key] == value, f"fit request {key} mismatch")

    def population(directory, allowed, expected_hash):
        recorder.add(directory / "receipt.json", expected_hash)
        receipt = _json(directory / "receipt.json")
        parent_lock = directory.parents[1] / "execution_lock.json"
        recorder.add(parent_lock, receipt["execution_lock_sha256"])
        recorder.add(directory / "predictions.npz", receipt["predictions_sha256"])
        _require(receipt["outer_prediction_labels_read"] == 0, "parent admits reading outside labels")
        with np.load(directory / "predictions.npz", allow_pickle=False) as f:
            _require(np.array_equal(f["population_rows"], allowed), "parent population differs from PDI allowed population")
            complement = np.setdiff1d(all_rows, allowed)
            _require(np.array_equal(f["prediction_rows"], complement), "parent predictions do not cover complement")
            _require(np.array_equal(f["sample_ids"], ids[complement]), "parent prediction IDs differ")
        cachekey = str(directory)
        if cachekey in populations_seen:
            return
        populations_seen.add(cachekey)
        _require(len(receipt["m4_fit_receipts"]) == 15 and len(receipt["a3_fit_receipts"]) == 15,
                 "ARFTR producer requires 15 M4 and 15 A3 receipts")
        row_lookup = {}
        for ref in receipt["m4_fit_receipts"]:
            recorder.reference(ref)
            path = ROOT / ref["path"]
            fit = _json(path)
            request = fit["request"]
            train, held = check_scoped_fit(fit["train_rows"], fit["held_rows"], allowed, scenarios,
                                          request["selection_labels_sha256"])
            request_rows(request, train, held)
            _require(request["base_ancestry_sha256"] == _canonical(fit["base_ancestry"]), "M4 base ancestry hash mismatch")
            for name, sha in fit["output_sha256"].items():
                recorder.add(path.with_name(name), sha)
            with np.load(path.with_name("predictions.npz"), allow_pickle=False) as f:
                _require(np.array_equal(f["held_rows"], held), "M4 prediction rows mismatch")
                _require(np.array_equal(f["sample_ids"], ids[held]), "M4 prediction IDs mismatch")
            row_lookup[(request["train_ids_sha256"], request["held_ids_sha256"])] = (train, held)
            base_edges(fit["base_ancestry"], train)
            counters["m4_receipts"] += 1
        for ref in receipt["a3_fit_receipts"]:
            recorder.reference(ref)
            path = ROOT / ref["path"]
            fit = _json(path)
            request = fit["request"]
            key = (request["train_ids_sha256"], request["held_ids_sha256"])
            _require(key in row_lookup, "A3 train/held ID hashes do not match a verified scoped population")
            train, held = row_lookup[key]
            check_scoped_fit(train, held, allowed, scenarios, request["selection_labels_sha256"])
            request_rows(request, train, held)
            _require(fit["train_rows"] == len(train) and fit["held_rows"] == len(held), "A3 row counts mismatch")
            for artifact in fit["artifacts"].values():
                recorder.reference(artifact)
            with np.load(path.with_name("predictions.npz"), allow_pickle=False) as f:
                _require(np.array_equal(f["sample_ids"], ids[held]), "A3 prediction IDs mismatch")
            counters["a3_receipts"] += 1
        base_edges(receipt["p6_ancestry"], allowed)

    def verify_producer(directory, reader, train, held):
        receipt = _json(directory / "receipt.json")
        recorder.add(directory / "receipt.json")
        _require(receipt["execution_lock_sha256"] == pdi_lock_hash, "producer lock hash mismatch")
        _require(receipt["arm"] == reader and receipt["updates"] == 256, "producer arm/training budget mismatch")
        _require(receipt["outer_held_labels_read"] == 0, "producer admits reading outer labels")
        _require(receipt["fit_rows"] == len(train) and receipt["held_rows"] == len(held), "producer row count mismatch")
        _require(receipt["fit_ids_sha256"] == _canonical(ids[train].tolist()), "producer training ID mismatch")
        _require(receipt["held_ids_sha256"] == _canonical(ids[held].tolist()), "producer held ID mismatch")
        for name, key in (("checkpoint.pt", "checkpoint_sha256"), ("training_checkpoints.pt", "training_checkpoints_sha256"),
                          ("normalization.npz", "normalization_sha256"), ("predictions.npz", "predictions_sha256")):
            recorder.add(directory / name, receipt[key])
        with np.load(directory / "predictions.npz", allow_pickle=False) as f:
            _require(np.array_equal(f["sample_ids"], ids[held]), "producer saved prediction identity mismatch")
        counters["pdi_producers"] += 1
        return receipt

    for reader in READERS:
        for outer in range(5):
            train = np.flatnonzero(folds != outer)
            held = np.flatnonzero(folds == outer)
            group_names = sorted(set(scenarios[train].tolist()))
            inner_parts = []
            fold_dir = PDI / reader / f"fold-{outer}"
            fold_receipt = _json(fold_dir / "receipt.json")
            recorder.add(fold_dir / "receipt.json")
            recorder.add(fold_dir / "inner_oof_verifier_inputs.npz", fold_receipt["inner_oof_verifier_inputs_sha256"])
            for inner in range(5):
                inner_held = train[np.isin(scenarios[train], group_names[inner::5])]
                fit_rows = np.setdiff1d(train, inner_held)
                inner_parts.append(inner_held)
                producer = verify_producer(fold_dir / f"inner-{inner}", reader, fit_rows, inner_held)
                ancestry = producer["ancestry"]
                _require(ancestry["train_scenarios"] == sorted(set(scenarios[fit_rows].tolist())), "PDI inner training scenarios mismatch")
                _require(ancestry["held_scenarios"] == sorted(set(scenarios[inner_held].tolist())), "PDI inner held scenarios mismatch")
                _require(ancestry["outer_prediction_labels_read"] == 0, "inner ancestry admits outside-label access")
                expected_id = f"o{outer}_i{inner}_{_canonical(ids[fit_rows].tolist())[:20]}"
                _require(ancestry["population_id"] == expected_id, "PDI inner ARFTR population ID mismatch")
                population(INNER / "populations" / expected_id, fit_rows, ancestry["population_receipt_sha256"])
                if reader == READERS[0]:
                    counters["inner_arftr_populations"] += 1
            _require(np.array_equal(np.sort(np.concatenate(inner_parts)), train), "PDI inner held coverage mismatch")
            producer = verify_producer(fold_dir / "final", reader, train, held)
            prior_parts = []
            for edge in producer["ancestry"]["ancestry"]:
                if "prior_F_id" in edge:
                    directory = GENERAL / "populations" / edge["prior_F_id"]
                    recorder.add(directory / "predictions.npz", edge["prior_predictions_sha256"])
                    with np.load(directory / "predictions.npz", allow_pickle=False) as f:
                        allowed = f["population_rows"].copy()
                    _require(np.isin(allowed, train).all(), "outer training prior contains outer-held rows")
                    target = np.setdiff1d(train, allowed)
                    _require(_canonical(ids[target].tolist()) == edge["target_sample_ids_sha256"], "outer prior target hash mismatch")
                    _require(not set(scenarios[allowed]) & set(scenarios[target]), "outer prior target scene occurs in training")
                    population(directory, allowed, edge["prior_receipt_sha256"])
                    prior_parts.append(target)
                    if reader == READERS[0]:
                        counters["outer_training_prior_populations"] += 1
                else:
                    recorder.add(ARFTR / f"fold-{outer}/predictions.npz", edge["outer_fold_predictions_sha256"])
            _require(np.array_equal(np.sort(np.concatenate(prior_parts)), train), "outer training priors lack exact-once block coverage")
    _require(counters["pdi_producers"] == 90 and counters["inner_arftr_populations"] == 25,
             "incomplete source producer matrix")
    return {
        "status": "CROSSING_EVENT_ANCESTRY_PASS", "source_run": PDI.relative_to(ROOT).as_posix(),
        "counts": {**counters, "unique_arftr_populations": len(populations_seen), "unique_base_caches": len(bases_seen)},
        "checks": ["PDI pre-fit dependency bytes", "90 PDI checkpoints, normalizations, saved predictions and train/held identity hashes",
                   "25 inner ARFTR parents and outer training-prior parents recursively scoped",
                   "M4 explicit row/scenario exclusions and selection-label scopes", "A3 request hashes matched to the same verified scoped train/held sets",
                   "nested P6 base model-selection splits and predicted-scenario exclusions", "exact-once inner and outer-prior coverage"],
        "limitations": ["Receipt and checkpoint provenance audit, not replay of historical neural training or proof against fabricated receipts.",
                        "Retained final outer ARFTR predictions and their historical parity/independent audits are hash-pinned; its original training is not repeated.",
                        "Frozen visual feature extraction and externally pretrained backbone provenance are inherited from pinned source evidence, not re-extracted here.",
                        "Dataset is an adaptively reused research benchmark; this does not create an untouched confirmatory test set.",
                        "Inner OOF rows may train one another's producers; this permits one fixed final outer gate, not a second cross-validation on those saved rows."],
        "dependency_receipts": sorted(recorder.records.values(), key=lambda row: row["path"]),
    }
