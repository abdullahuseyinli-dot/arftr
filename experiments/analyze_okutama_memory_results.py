"""Evaluation-only, versioned audits of completed locked memory outer refits.

No model is fitted or loaded. Incomplete folds are excluded from three-seed
ensembles. Labels are used only for retrospective diagnostics; these outputs
must not change the locked trial. Repeated physical boundary edges are reported
both as context occurrences and after averaging over their observed contexts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hac.actor_memory_base import (  # noqa: E402
    canonical_hash,
    file_sha256,
    group_splits,
    probability_metrics,
)

DEFAULT_RUN = ROOT / ".runs/research_20260908/evidence_memory"
SAVED_FIELDS = ("probabilities", "gate", "attention", "survival", "boundary_probabilities")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"rows": 0}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values")
    return {
        "rows": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def binary_calibration(target, probabilities):
    target, p = np.asarray(target, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)
    if len(target) == 0:
        return {"edges": 0}
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid boundary probabilities")
    clipped = p.clip(1e-12, 1 - 1e-12)
    predicted = p >= 0.5
    bins, ece = [], 0.0
    bin_indices = np.minimum((p * 10).astype(int), 9)
    for bin_index in range(10):
        lower, upper = bin_index / 10, (bin_index + 1) / 10
        mask = bin_indices == bin_index
        if mask.any():
            observed, expected = float(target[mask].mean()), float(p[mask].mean())
            ece += mask.mean() * abs(observed - expected)
            bins.append(
                {
                    "lower_inclusive": float(lower),
                    "upper_exclusive_except_final": float(upper),
                    "edges": int(mask.sum()),
                    "mean_probability": expected,
                    "observed_frequency": observed,
                }
            )
    return {
        "edges": len(target),
        "positives": int(target.sum()),
        "positive_frequency": float(target.mean()),
        "mean_probability": float(p.mean()),
        "brier": float(np.mean((p - target) ** 2)),
        "nll": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
        "ece_10_equal_width_bins": float(ece),
        "roc_auc": float(roc_auc_score(target, p)) if len(np.unique(target)) == 2 else None,
        "fixed_threshold": 0.5,
        "true_positive": int((predicted & (target == 1)).sum()),
        "false_positive": int((predicted & (target == 0)).sum()),
        "true_negative": int((~predicted & (target == 0)).sum()),
        "false_negative": int((~predicted & (target == 1)).sum()),
        "reliability_bins": bins,
    }


def safe_spearman(left, right):
    if len(left) < 3 or len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return None
    return float(spearmanr(left, right).statistic)


def load_inputs(run, provenance):
    lock_path = run / "execution_lock.json"
    lock = read_json(lock_path)
    provenance[str(lock_path)] = file_sha256(lock_path)
    for relative, expected in lock["files"].items():
        # Large base feature bytes have already been replay-verified by training;
        # inspect all other immutable files, including memory metadata and code.
        if relative.endswith("base_features.npz"):
            continue
        path = ROOT / relative
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Execution source changed: {path}")
        provenance[str(path)] = actual
    with np.load(run / "data/memory_data.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files if key != "features"}
    if canonical_hash(data["sample_ids"].tolist()) != lock["sample_ids_sha256"]:
        raise RuntimeError("Execution/data identities differ")
    if len(data["labels"]) != lock["protocol"]["primary_rows"]:
        raise RuntimeError("Unexpected primary row population")
    for filename in ("base_replay.npz", "data/legacy_temporal_strata.npz"):
        path = run / filename
        provenance[str(path)] = file_sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(data["sample_ids"], saved["sample_ids"]):
                raise RuntimeError(f"Row alignment failed: {path}")
            if filename == "base_replay.npz":
                baseline = {key: saved[key] for key in ("probabilities", "components")}
            else:
                legacy = {key: saved[key] for key in saved.files if key != "sample_ids"}
    if not read_json(run / "base_replay.json")["passed"]:
        raise RuntimeError("Baseline replay did not pass")
    legacy_receipt_path = run / "data/legacy_temporal_strata_receipt.json"
    legacy_receipt = read_json(legacy_receipt_path)
    provenance[str(legacy_receipt_path)] = file_sha256(legacy_receipt_path)
    if (
        provenance[str(run / "data/legacy_temporal_strata.npz")]
        != legacy_receipt["artifact_sha256"]
    ):
        raise RuntimeError("Legacy diagnostic strata changed")
    original = legacy_receipt["source_receipts"]["p6_evaluation_reference"]
    original_path = Path(original["path"])
    provenance[str(original_path)] = file_sha256(original_path)
    if provenance[str(original_path)] != original["sha256"]:
        raise RuntimeError("Original P6 reference changed")
    with np.load(original_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["ocvc_uniform_diverse_triad"], baseline["probabilities"]):
            raise RuntimeError("Memory baseline no longer exactly reproduces original P6")
    neighbors, valid = data["neighbor_indices"], data["valid"]
    if not np.array_equal(neighbors >= 0, valid):
        raise RuntimeError("Invalid neighbor mask")
    row, slot = np.nonzero(valid)
    n = neighbors[row, slot]
    for name in ("scenarios", "recordings", "tracks", "folds"):
        if not np.array_equal(data[name][row], data[name][n]):
            raise RuntimeError(f"Neighborhood crosses {name}")
    return data, baseline, legacy, lock


def verify_receipt(run, receipt, data, lock, arm, train, held, seed, epochs, provenance):
    request = receipt["request"]
    expected = {
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "protocol_sha256": canonical_hash(lock["protocol"]),
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": None,
        "data_sha256": lock["files"][str((run / "data/memory_data.npz").relative_to(ROOT))],
        "training_code_sha256": lock["files"][str(Path("src/hac/actor_memory_training.py"))],
        "architecture_code_sha256": lock["files"][str(Path("src/hac/actor_evidence_memory.py"))],
    }
    if any(request[key] != value for key, value in expected.items()):
        raise RuntimeError("Outer refit request differs from locked partition")
    if receipt["train_rows"] != train.tolist() or receipt["held_rows"] != held.tolist():
        raise RuntimeError("Receipt row identity mismatch")
    if set(data["scenarios"][train]) & set(data["scenarios"][held]):
        raise RuntimeError("Outer scenario leakage")
    if receipt["selected_epoch"] != epochs or receipt["epochs_run"] != epochs:
        raise RuntimeError("Outer refit did not use locked epochs")
    known_targets = data["boundary_targets"][train][data["boundary_valid"][train]]
    positives = float(known_targets.sum())
    expected_weight = np.float32((len(known_targets) - positives) / positives if positives else 1)
    if receipt["boundary_positive_weight"] != float(expected_weight):
        raise RuntimeError("Boundary class weight differs from meta-training labels")
    if any("validation_metrics" in item for item in receipt["history"]):
        raise RuntimeError("Outer refit evaluated held labels during optimization")
    ancestry = receipt["base_ancestry"]
    if canonical_hash(ancestry) != request["base_ancestry_sha256"]:
        raise RuntimeError("Base ancestry hash mismatch")
    coverage = np.zeros(len(data["labels"]), dtype=int)
    train_set = set(train.tolist())
    for item in ancestry:
        predicted_rows = np.asarray(item["predicted_rows"], dtype=int)
        base_path = run / "base_cache" / item["base_cache"] / "receipt.json"
        base_receipt = read_json(base_path)
        provenance[str(base_path)] = file_sha256(base_path)
        base_rows = np.asarray(base_receipt["request"]["train_rows"], dtype=int)
        if not set(base_rows.tolist()).issubset(train_set):
            raise RuntimeError("Base fit used rows outside meta training")
        if set(data["scenarios"][base_rows]) & set(data["scenarios"][predicted_rows]):
            raise RuntimeError("Base ancestor predicts its training scenario")
        coverage[predicted_rows] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Base ancestry does not cover every row exactly once")


def load_arm(run, data, lock, arm, provenance):
    protocol = lock["protocol"]
    fields = {name: [] for name in SAVED_FIELDS}
    rows, completed, pending, audits = [], [], [], []
    for fold in protocol["outer_folds"]:
        directory = run / "models" / arm / f"fold-{fold}"
        marker = directory / "outer_complete.json"
        if not marker.exists():
            pending.append(fold)
            continue
        path = directory / "outer_predictions.npz"
        marker_value = read_json(marker)
        if file_sha256(path) != marker_value["probabilities_sha256"]:
            raise RuntimeError("Completed outer predictions changed")
        for source in (marker, path, directory / "selection.json"):
            provenance[str(source)] = file_sha256(source)
        held = np.flatnonzero(data["folds"] == fold)
        train = np.flatnonzero(data["folds"] != fold)
        if marker_value["sample_ids_sha256"] != canonical_hash(data["sample_ids"][held].tolist()):
            raise RuntimeError("Outer completion marker identity mismatch")
        selection = read_json(directory / "selection.json")
        selected = min(
            selection["candidates"],
            key=lambda c: (
                -c["inner_metrics"]["macro_f1"],
                c["inner_metrics"]["nll"],
                c["learning_rate"],
                c["weight_decay"],
            ),
        )
        epochs = max(1, int(np.rint(np.median(selected["inner_epochs"]))))
        if selection["selected"] != selected or selection["refit_epochs"] != epochs:
            raise RuntimeError("Selection tie-break or median epoch mismatch")
        if selection["outer_held_used_for_selection"] is not False:
            raise RuntimeError("Held selection declared")
        # Cross-check the selected inner receipt partitions and epochs, not its
        # metrics for a new choice. The completed locked selection is immutable.
        for inner, (inner_train, inner_held) in enumerate(
            group_splits(data["labels"], data["scenarios"], train)
        ):
            inner_path = (
                directory / f"config-{selected['config_index']}" / f"inner-{inner}" / "receipt.json"
            )
            inner_receipt = read_json(inner_path)
            provenance[str(inner_path)] = file_sha256(inner_path)
            if (
                inner_receipt["train_rows"] != inner_train.tolist()
                or inner_receipt["held_rows"] != inner_held.tolist()
            ):
                raise RuntimeError("Selected inner partition mismatch")
            if inner_receipt["selected_epoch"] != selected["inner_epochs"][inner]:
                raise RuntimeError("Selected inner epoch mismatch")
        seed_fields = {name: [] for name in SAVED_FIELDS}
        for seed in protocol["outer_seeds"]:
            seed_dir = directory / f"refit-seed-{seed}"
            receipt_path = seed_dir / "receipt.json"
            receipt = read_json(receipt_path)
            verify_receipt(run, receipt, data, lock, arm, train, held, seed, epochs, provenance)
            if (
                receipt["request"]["learning_rate"] != selected["learning_rate"]
                or receipt["request"]["weight_decay"] != selected["weight_decay"]
            ):
                raise RuntimeError("Outer refit configuration mismatch")
            provenance[str(receipt_path)] = file_sha256(receipt_path)
            for filename, expected_hash in receipt["output_sha256"].items():
                source = seed_dir / filename
                actual = file_sha256(source)
                if actual != expected_hash:
                    raise RuntimeError(f"Completed seed output changed: {source}")
                provenance[str(source)] = actual
            with np.load(seed_dir / "predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["held_rows"], held) or not np.array_equal(
                    saved["sample_ids"], data["sample_ids"][held]
                ):
                    raise RuntimeError("Seed prediction identity mismatch")
                for name in SAVED_FIELDS:
                    values = saved[name]
                    if not np.isfinite(values).all():
                        raise RuntimeError("Nonfinite completed seed output")
                    seed_fields[name].append(values)
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["held_rows"], held) or not np.array_equal(
                saved["sample_ids"], data["sample_ids"][held]
            ):
                raise RuntimeError("Outer prediction identity mismatch")
            if not np.array_equal(
                saved["seed_probabilities"], np.stack(seed_fields["probabilities"])
            ):
                raise RuntimeError("Outer seed ensemble contents mismatch")
            if not np.array_equal(
                saved["probabilities"], np.mean(seed_fields["probabilities"], axis=0)
            ):
                raise RuntimeError("Outer ensemble differs from exact three-seed mean")
        for name in SAVED_FIELDS:
            fields[name].append(np.stack(seed_fields[name]))
        rows.append(held)
        completed.append(fold)
        audits.append(
            {
                "fold": fold,
                "rows": len(held),
                "epochs": epochs,
                "seeds": protocol["outer_seeds"],
                "boundary_training_positive_weight": receipt["boundary_positive_weight"],
                "training_class_weight": receipt["class_weight"],
                "receipt_checks_passed": True,
            }
        )
    if not rows:
        return None, {"complete_folds": completed, "pending_folds": pending, "rows": 0}
    rows = np.concatenate(rows)
    order = np.argsort(rows)
    result = {key: np.concatenate(values, axis=1)[:, order] for key, values in fields.items()}
    result["rows"] = rows[order]
    return result, {
        "complete_folds": completed,
        "pending_folds": pending,
        "rows": len(rows),
        "complete_primary_population": not pending,
        "scope": "FULL 4977 DEVELOPMENT ROWS"
        if not pending
        else "PARTIAL COMPLETED FOLDS ONLY; NOT AN OVERALL RESULT",
        "fold_audits": audits,
    }


def compare(labels, probabilities, reference, shared, mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"rows": 0}
    correct, base_correct = probabilities.argmax(1) == labels, reference.argmax(1) == labels
    return {
        "rows": int(mask.sum()),
        "metrics": probability_metrics(labels[mask], probabilities[mask]),
        "p6_metrics": probability_metrics(labels[mask], reference[mask]),
        "p6_errors": int((~base_correct & mask).sum()),
        "errors": int((~correct & mask).sum()),
        "rescues": int((correct & ~base_correct & mask).sum()),
        "harms": int((~correct & base_correct & mask).sum()),
        "net_corrections": int((correct & mask).sum() - (base_correct & mask).sum()),
        "shared_p6_failures": int((shared & mask).sum()),
        "shared_failure_repairs": int((correct & shared & mask).sum()),
    }


def boundary_audit(arm, data, rows, seed_probabilities):
    if arm == "temporal_conv":
        return {"applicable": False, "reason": "Temporal convolution has no boundary head"}, []
    probabilities = np.mean(seed_probabilities, axis=0)
    mask = data["boundary_valid"][rows]
    local, destination = np.nonzero(mask)
    centers = rows[local]
    source_slot = data["edge_source_slot"][centers, destination]
    if (source_slot < 0).any():
        raise RuntimeError("Known edge has no source")
    source_row = data["neighbor_indices"][centers, source_slot]
    destination_row = data["neighbor_indices"][centers, destination]
    target = data["boundary_targets"][centers, destination].astype(int)
    p = probabilities[local, destination]
    grouped = {}
    for index, center in enumerate(centers):
        key = (
            str(data["recordings"][center]),
            str(data["tracks"][center]),
            int(data["frames"][source_row[index]]),
            int(data["frames"][destination_row[index]]),
        )
        entry = grouped.setdefault(
            key,
            {
                "target": int(target[index]),
                "probabilities": [],
                "bridged": bool(data["edge_bridges_missing_slot"][center, destination[index]]),
            },
        )
        if entry["target"] != target[index]:
            raise RuntimeError("Same physical edge has conflicting complete labels")
        entry["probabilities"].append(float(p[index]))
    table = []
    for key, value in sorted(grouped.items()):
        recording, track, left, right = key
        table.append(
            {
                "arm": arm,
                "recording": recording,
                "track": track,
                "source_frame": left,
                "destination_frame": right,
                "dt_seconds": (right - left) / 30,
                "boundary_target": value["target"],
                "mean_boundary_probability": float(np.mean(value["probabilities"])),
                "context_occurrences": len(value["probabilities"]),
                "bridges_missing_slot": value["bridged"],
            }
        )
    return {
        "applicable": True,
        "labels": "All inclusive annotation frames known; any target-state change over edge",
        "probability_aggregation": "Mean of three outer seeds, then mean of contexts for each physical edge; diagnostic only",
        "calibration_caution": "Boundary BCE was positive-class weighted; raw scores need not be calibrated probabilities",
        "occurrence_weighted": binary_calibration(target, p),
        "physical_edge_deduplicated": binary_calibration(
            [entry["boundary_target"] for entry in table],
            [entry["mean_boundary_probability"] for entry in table],
        ),
        "per_seed_occurrence_weighted": [
            binary_calibration(target, values[local, destination]) for values in seed_probabilities
        ],
    }, table


def path_strata(data, rows):
    """Retrospective all-frame knowledge along each neighbor-to-center path."""
    shape = data["valid"][rows].shape
    stable, changed, unknown = (np.zeros(shape, bool) for _ in range(3))
    for local, row in enumerate(rows):
        available = np.flatnonzero(data["valid"][row])
        for slot in available:
            if slot == 2:
                continue
            lower, upper = sorted((int(slot), 2))
            edges = available[(available > lower) & (available <= upper)]
            if not data["boundary_valid"][row, edges].all():
                unknown[local, slot] = True
            elif data["boundary_targets"][row, edges].any():
                changed[local, slot] = True
            else:
                stable[local, slot] = True
    return {
        "verified_same_state_path": stable,
        "verified_boundary_crossing_path": changed,
        "unknown_annotation_path": unknown,
    }


def analyze_arm(arm, outputs, data, baseline, legacy):
    rows = outputs["rows"]
    labels, reference = data["labels"][rows], baseline["probabilities"][rows]
    probabilities = outputs["probabilities"].mean(0)
    correct, base_correct = probabilities.argmax(1) == labels, reference.argmax(1) == labels
    shared = (baseline["components"][rows].argmax(2) != labels[:, None]).all(1)
    gate = outputs["gate"].mean(0)
    attention = outputs["attention"].mean(0)
    effective = (outputs["gate"][..., None] * outputs["attention"]).mean(0)
    valid = data["valid"][rows]
    neighbors = data["neighbor_indices"][rows].clip(min=0)
    neighbor_probabilities = np.where(valid[..., None], baseline["probabilities"][neighbors], 0)
    seed_memory = np.einsum("sbn,bnc->sbc", outputs["attention"], neighbor_probabilities)
    seed_gate = outputs["gate"].astype(np.float64)[..., None]
    replay = (1 - seed_gate) * reference + seed_gate * seed_memory
    replay_difference = float(np.max(np.abs(replay - outputs["probabilities"])))
    if replay_difference > 1e-12:
        raise RuntimeError(
            "Saved probabilities cannot be reconstructed from frozen P6 and saved mixture weights"
        )
    neighbor_pred = baseline["probabilities"][neighbors].argmax(2)
    helpful = valid & (neighbor_pred == labels[:, None])
    helpful[:, 2] = False
    witnesses = valid.copy()
    witnesses[:, 2] = False
    if not np.allclose(attention.sum(1), 1, atol=1e-12, rtol=0):
        raise RuntimeError("Saved attention does not sum to one")
    if np.any(attention[~valid] != 0):
        raise RuntimeError("Invalid observation has attention")
    height = data["quality"][rows, 0].astype(float) * 720
    masks = {
        "all": np.ones(len(rows), bool),
        "p6_correct": base_correct,
        "p6_error": ~base_correct,
        "rescue": correct & ~base_correct,
        "harm": ~correct & base_correct,
        "shared_failure": shared,
        "shared_failure_repaired": shared & correct,
        "shared_failure_unrepaired": shared & ~correct,
        "long_fallback": ~data["long_valid"][rows],
        "valid_long": data["long_valid"][rows],
        "native_height_le32": height <= 32 + 1e-4,
        "native_height_32to64": (height > 32 + 1e-4) & (height <= 64 + 1e-4),
        "native_height_gt64": height > 64 + 1e-4,
        "no_neighbor": ~witnesses.any(1),
        "helpful_neighbor_available": helpful.any(1),
        "no_helpful_neighbor": ~helpful.any(1),
    }
    for key, values in legacy.items():
        masks["legacy_" + key] = values[rows]
    for class_index, name in enumerate(("sitting", "standing", "walking_running")):
        masks["true_class_" + name] = labels == class_index
    for prefix in ("node_support", "memory_support", "node_interval", "memory_interval"):
        complete = data[prefix + "_complete"][rows]
        boundary = data[prefix + "_boundary"][rows]
        masks[prefix + "_complete_boundary"] = complete & boundary
        masks[prefix + "_complete_stable"] = complete & ~boundary
        masks[prefix + "_unknown"] = ~complete
    attention_values = {
        "center_attention": attention[:, 2],
        "helpful_neighbor_attention": (attention * helpful).sum(1),
        "wrong_for_center_neighbor_attention": (attention * (witnesses & ~helpful)).sum(1),
        "expected_absolute_distance_seconds": (attention * np.abs(data["times"][rows])).sum(1),
        "effective_helpful_neighbor_weight": (effective * helpful).sum(1),
        "effective_wrong_neighbor_weight": (effective * (witnesses & ~helpful)).sum(1),
        "effective_center_weight": 1 - gate + effective[:, 2],
    }
    for name, path_mask in path_strata(data, rows).items():
        attention_values[name + "_attention"] = (attention * path_mask).sum(1)
        attention_values[name + "_effective_weight"] = (effective * path_mask).sum(1)
    temporal_mean = neighbor_probabilities.sum(1) / valid.sum(1)[:, None]
    strata = {}
    for name, mask in masks.items():
        strata[name] = {
            **compare(labels, probabilities, reference, shared, mask),
            "fixed_two_second_mean": compare(labels, temporal_mean, reference, shared, mask),
            "gate": distribution(gate[mask]),
            "attention": {
                key: distribution(value[mask]) for key, value in attention_values.items()
            },
        }
    boundary, edge_table = boundary_audit(arm, data, rows, outputs["boundary_probabilities"])
    grouping_tables = {}
    for group_name, values in {
        "scenario": data["scenarios"][rows],
        "recording": data["recordings"][rows],
        "track": np.char.add(np.char.add(data["recordings"][rows], "::"), data["tracks"][rows]),
    }.items():
        grouping_tables[group_name] = []
        for value in np.unique(values):
            mask = values == value
            compared = compare(labels, probabilities, reference, shared, mask)
            grouping_tables[group_name].append(
                {
                    "arm": arm,
                    group_name: str(value),
                    **{key: item for key, item in compared.items() if not isinstance(item, dict)},
                    "accuracy_delta": compared["metrics"]["accuracy"]
                    - compared["p6_metrics"]["accuracy"],
                    "macro_f1": compared["metrics"]["macro_f1"],
                    "p6_macro_f1": compared["p6_metrics"]["macro_f1"],
                    "mean_gate": float(gate[mask].mean()),
                    "mean_native_height": float(height[mask].mean()),
                    "long_fallback_rows": int((~data["long_valid"][rows] & mask).sum()),
                }
            )
    row_table = []
    for i, row in enumerate(rows):
        row_table.append(
            {
                "arm": arm,
                "row": int(row),
                "sample_id": str(data["sample_ids"][row]),
                "scenario": str(data["scenarios"][row]),
                "fold": int(data["folds"][row]),
                "recording": str(data["recordings"][row]),
                "track": str(data["tracks"][row]),
                "center_frame": int(data["frames"][row]),
                "label": int(labels[i]),
                "p6_prediction": int(reference[i].argmax()),
                "memory_prediction": int(probabilities[i].argmax()),
                "p6_correct": bool(base_correct[i]),
                "memory_correct": bool(correct[i]),
                "shared_failure": bool(shared[i]),
                "rescue": bool(correct[i] and not base_correct[i]),
                "harm": bool(not correct[i] and base_correct[i]),
                "mean_gate": float(gate[i]),
                "native_height": float(height[i]),
                "long_fallback": bool(not data["long_valid"][row]),
                "available_neighbors": int(witnesses[i].sum()),
                "helpful_neighbors": int(helpful[i].sum()),
                **{key: float(value[i]) for key, value in attention_values.items()},
                **{
                    key: bool(value[i]) for key, value in masks.items() if key.startswith("legacy_")
                },
            }
        )
    summary = {
        **compare(labels, probabilities, reference, shared, masks["all"]),
        "seed_metrics": [probability_metrics(labels, p) for p in outputs["probabilities"]],
        "fixed_two_second_mean": compare(labels, temporal_mean, reference, shared, masks["all"]),
        "gate_error_detection_auc": float(roc_auc_score(~base_correct, gate))
        if len(np.unique(base_correct)) == 2
        else None,
        "gate_per_seed": [distribution(value) for value in outputs["gate"]],
        "probability_mixture_replay_maximum_absolute_difference": replay_difference,
        "native_height_definition": "720 times stored float32 normalized center bbox height; bands use 0.0001-pixel tolerance at 32 and 64 to avoid float32 threshold artifacts, so exact-threshold counts can differ from earlier float64 manifest bands",
        "retrospective_spearman_correlations": {
            "gate_vs_native_height": safe_spearman(gate, height),
            "gate_vs_p6_error": safe_spearman(gate, ~base_correct),
            "gate_vs_helpful_neighbor_count": safe_spearman(gate, helpful.sum(1)),
            "gate_vs_neighbor_count": safe_spearman(gate, witnesses.sum(1)),
            "memory_error_vs_native_height": safe_spearman(~correct, height),
        },
        "strata": strata,
        "boundary": boundary,
        "exact_center_fallback": {
            "rows": int(masks["no_neighbor"].sum()),
            "bit_exact_all_seeds": bool(
                all(
                    np.array_equal(p[masks["no_neighbor"]], reference[masks["no_neighbor"]])
                    for p in outputs["probabilities"]
                )
            ),
        },
        "interpretation": "Associations are descriptive, not causal. Helpful attention uses center ground truth only in this audit. Effective mixture weights average seed-specific gate times attention, not the product of their averages.",
    }
    return summary, row_table, edge_table, grouping_tables


def write_csv(path, rows):
    if not rows:
        return False
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return True


def completed_primary_comparison(available, data, legacy):
    """No partial-scope extrapolation when comparing survival to plain retrieval."""
    required = ("survival_memory", "query_attention")
    if any(name not in available for name in required):
        return {"status": "WAITING FOR BOTH COMPLETE 4977-ROW ARMS"}
    if any(len(available[name]["rows"]) != len(data["labels"]) for name in required):
        return {"status": "WAITING FOR BOTH COMPLETE 4977-ROW ARMS"}
    candidate, reference = (available[name]["probabilities"].mean(0) for name in required)
    labels = data["labels"]
    candidate_correct = candidate.argmax(1) == labels
    reference_correct = reference.argmax(1) == labels
    strata = {"all": np.ones(len(labels), bool), **legacy}
    results = {}
    for name, mask in strata.items():
        candidate_metrics = probability_metrics(labels[mask], candidate[mask])
        reference_metrics = probability_metrics(labels[mask], reference[mask])
        results[name] = {
            "rows": int(mask.sum()),
            "survival_correct_query_wrong": int(
                (candidate_correct & ~reference_correct & mask).sum()
            ),
            "survival_wrong_query_correct": int(
                (~candidate_correct & reference_correct & mask).sum()
            ),
            "macro_f1_delta": candidate_metrics["macro_f1"] - reference_metrics["macro_f1"],
            "accuracy_delta": candidate_metrics["accuracy"] - reference_metrics["accuracy"],
            "nll_delta": candidate_metrics["nll"] - reference_metrics["nll"],
            "brier_delta": candidate_metrics["brier"] - reference_metrics["brier"],
        }
    return {
        "status": "BOTH ARMS COMPLETE; DESCRIPTIVE PAIRED COMPARISON ONLY",
        "candidate": required[0],
        "reference": required[1],
        "strata": results,
        "note": "Same outer rows and three-seed means. No post-hoc parameter changes or significance claims from this diagnostic.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    provenance = {str(Path(__file__)): file_sha256(Path(__file__))}
    data, baseline, legacy, lock = load_inputs(run, provenance)
    arms = args.arms or lock["protocol"]["arms"]
    if not set(arms).issubset(lock["protocol"]["arms"]):
        raise ValueError("Undeclared diagnostic arm")
    output_root = (run / "diagnostics").resolve()
    if args.output_dir is None:
        sequence = 1
        while (output_root / f"v{sequence:04d}").exists():
            sequence += 1
        output = output_root / f"v{sequence:04d}"
    else:
        output = args.output_dir.resolve()
    if output_root not in output.parents:
        raise ValueError("Diagnostic output must be a new version inside run/diagnostics")
    if output.exists():
        raise FileExistsError("Never overwrite a diagnostic version")
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluation_only": True,
        "training_or_checkpoint_inference_performed": False,
        "changes_to_locked_trial_allowed": False,
        "primary_population_rows": len(data["labels"]),
        "class_order": lock["protocol"]["class_order"],
        "outer_seed_order": lock["protocol"]["outer_seeds"],
        "partial_policy": "Only complete outer-fold markers with all declared seeds; missing folds are not imputed or projected",
        "statistical_scope": "Retrospective development diagnostics; no tuning, significance testing, or external confirmation",
        "arms": {},
    }
    tables = {name: [] for name in ("rows", "physical_edges", "scenario", "recording", "track")}
    available = {}
    for arm in arms:
        outputs, coverage = load_arm(run, data, lock, arm, provenance)
        if outputs is None:
            summary["arms"][arm] = {"coverage": coverage, "status": "NO COMPLETE OUTER FOLD"}
            continue
        available[arm] = outputs
        analyzed, rows, edges, groups = analyze_arm(arm, outputs, data, baseline, legacy)
        summary["arms"][arm] = {"coverage": coverage, **analyzed}
        tables["rows"].extend(rows)
        tables["physical_edges"].extend(edges)
        for name, values in groups.items():
            tables[name].extend(values)
    summary["complete_requested_arms"] = all(
        item["coverage"].get("complete_primary_population", False)
        for item in summary["arms"].values()
    )
    summary["survival_vs_query_attention"] = completed_primary_comparison(available, data, legacy)
    summary["provenance_sha256"] = provenance
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    summary["tables_sha256"] = {
        path.name: file_sha256(path) for path in sorted(output.glob("*.csv"))
    }
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "summary_sha256": file_sha256(output / "summary.json"),
                "arms": {
                    name: {
                        "coverage": value["coverage"],
                        "metrics": value.get("metrics"),
                        "rescues": value.get("rescues"),
                        "harms": value.get("harms"),
                    }
                    for name, value in summary["arms"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
