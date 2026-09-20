"""Independent, read-only reconstruction of a *completed* SEAR matrix.

This intentionally imports no SEAR runner, model, selection or scoring helpers.
It checks stored checkpoint ancestry, not a fresh forward pass of all 450 models.
No partial matrix scores are loaded. The only writes are a new audit directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".runs/research_20260908/sear_matrix_v1"
ARMS = (
    "a0_center_cls",
    "a1_dense_cnn",
    "a2_dense_transformer",
    "a3_unrestricted_templates",
    "a4_shared_no_geometry",
    "a5_sear",
)
TEMPLATE_ARMS = ARMS[3:]
CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0])) + tuple((ARMS[5], arm) for arm in ARMS[1:5])
CLASS_NAMES = ("sitting", "standing", "walking_running")
CONDITIONS = ("coordinate_shuffle", "outer_border_mask", "video_removed")
REFERENCES = ("old_source_p6", "old_source_m4", "new_source_p6", "new_source_m4")
RESULT_VERSION = "results/v0001"
EXPECTED_PARAMETERS = dict(zip(ARMS, (563462, 551266, 548066, 552508, 549171, 551558), strict=True))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal_array(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return np.array_equal(left, right, equal_nan=bool(np.issubdtype(left.dtype, np.inexact)))


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


class Evidence:
    """Hash each unique immutable input once; recheck size/mtime before publication."""

    def __init__(self):
        self.records: dict[str, dict] = {}
        self.stats: dict[str, tuple[int, int]] = {}

    def check(self, entry: dict, expected_path: Path | None = None) -> Path:
        path = Path(entry["path"]).resolve()
        if expected_path is not None:
            require(path == expected_path.resolve(), f"Artifact path substitution: {path}")
        key = str(path)
        require(path.is_file(), f"Missing artifact: {path}")
        observed = path.stat()
        signature = (observed.st_size, observed.st_mtime_ns)
        require(observed.st_size == entry["size_bytes"], f"Artifact size changed: {path}")
        if key not in self.records:
            require(sha256(path) == entry["sha256"], f"Artifact hash changed: {path}")
            self.records[key] = dict(entry)
            self.stats[key] = signature
        else:
            require(self.records[key]["sha256"] == entry["sha256"], f"Conflicting receipts: {path}")
            require(self.stats[key] == signature, f"Artifact changed during audit: {path}")
        return path

    def bind(self, path: Path) -> dict:
        entry = file_record(path)
        self.check(entry, path)
        return entry

    def finish(self) -> None:
        for key, expected in self.stats.items():
            actual = Path(key).stat()
            require((actual.st_size, actual.st_mtime_ns) == expected, f"Input changed: {key}")


def same(expected, actual, name: str, *, tolerance: float = 1e-10) -> None:
    """Compare all expected fields; tolerate additive reporting-only fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Not a mapping: {name}")
        for key, value in expected.items():
            require(key in actual, f"Missing field: {name}.{key}")
            same(value, actual[key], f"{name}.{key}", tolerance=tolerance)
    elif isinstance(expected, (tuple, list)):
        require(
            isinstance(actual, (tuple, list)) and len(actual) == len(expected), f"Length: {name}"
        )
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            same(left, right, f"{name}[{index}]", tolerance=tolerance)
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and np.isfinite(expected)
            and np.isfinite(actual)
            and abs(expected - actual) <= tolerance,
            f"Numeric mismatch: {name}",
        )
    else:
        require(expected == actual, f"Value mismatch: {name}")


def f1(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    return np.divide(
        2 * diagonal,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def confusion(labels, predictions):
    result = np.zeros((3, 3), dtype=np.int64)
    np.add.at(result, (labels, predictions), 1)
    return result


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    require(probabilities.shape == (len(labels), 3), "Probability shape mismatch")
    require(np.isfinite(probabilities).all(), "Nonfinite probabilities")
    require((probabilities >= 0).all(), "Negative probabilities")
    require(np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0), "Probabilities not normalized")
    if not len(labels):
        return {
            "rows": 0,
            "macro_f1": None,
            "accuracy": None,
            "nll": None,
            "brier": None,
            "per_class_f1": [None] * 3,
            "confusion": [[0] * 3 for _ in range(3)],
        }
    matrix = confusion(labels, probabilities.argmax(1))
    scores = f1(matrix)
    return {
        "rows": len(labels),
        "macro_f1": float(scores.mean()),
        "accuracy": float(np.trace(matrix) / len(labels)),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        ),
        "brier": float(((probabilities - np.eye(3)[labels]) ** 2).sum(1).mean()),
        "per_class_f1": scores.tolist(),
        "confusion": matrix.tolist(),
    }


def contrast(labels, candidate, reference, scenarios, *, resamples, seed) -> dict:
    """Scenario paired bootstrap and exhaustive swaps, independently implemented."""
    groups = np.unique(scenarios)
    cand_pred, ref_pred = candidate.argmax(1), reference.argmax(1)
    cand_cm = np.array(
        [confusion(labels[scenarios == g], cand_pred[scenarios == g]) for g in groups]
    )
    ref_cm = np.array([confusion(labels[scenarios == g], ref_pred[scenarios == g]) for g in groups])
    observed = float(f1(cand_cm.sum(0)).mean() - f1(ref_cm.sum(0)).mean())
    draws = np.random.default_rng(seed).integers(len(groups), size=(resamples, len(groups)))
    boot_cand = cand_cm[draws].sum(1)
    deltas = f1(boot_cand).mean(1) - f1(ref_cm[draws].sum(1)).mean(1)
    # Arithmetic group swaps instead of the runner's np.where formulation.
    bits = ((np.arange(2 ** len(groups))[:, None] >> np.arange(len(groups))) % 2).astype(np.int64)
    change = np.einsum("sg,gij->sij", bits, ref_cm - cand_cm)
    swapped = f1(cand_cm.sum(0) + change).mean(1) - f1(ref_cm.sum(0) - change).mean(1)
    rescued = (cand_pred == labels) & (ref_pred != labels)
    harmed = (cand_pred != labels) & (ref_pred == labels)
    counts = np.array([np.count_nonzero(scenarios == g) for g in groups])
    target = np.eye(3)[labels]
    losses = {
        "nll": -np.log(np.clip(candidate[np.arange(len(labels)), labels], 1e-12, 1))
        + np.log(np.clip(reference[np.arange(len(labels)), labels], 1e-12, 1)),
        "brier": ((candidate - target) ** 2).sum(1) - ((reference - target) ** 2).sum(1),
    }
    result = {}
    for name, difference in losses.items():
        sums = np.array([difference[scenarios == g].sum() for g in groups])
        sampled = sums[draws].sum(1) / counts[draws].sum(1)
        result.update(
            {
                name + "_delta": float(difference.mean()),
                name + "_delta_two_sided_95pct": np.quantile(sampled, [0.025, 0.975]).tolist(),
                name + "_delta_one_sided_95pct_upper": float(np.quantile(sampled, 0.95)),
            }
        )
    result.update(
        {
            "macro_f1_delta": observed,
            "macro_f1_delta_two_sided_95pct": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "macro_f1_delta_one_sided_95pct_lower": float(np.quantile(deltas, 0.05)),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "bootstrap_valid_fraction": float((boot_cand.sum(2) > 0).all(1).mean()),
            "bootstrap_draws_used": len(deltas),
            "bootstrap_missing_class_policy": "retain_fixed_three_classes_zero_division_0",
            "scenario_order": groups.tolist(),
            "exact_swap_assignments": len(bits),
            "one_sided_exact_swap_pvalue": float((swapped >= observed - 1e-12).mean()),
            "two_sided_exact_swap_pvalue": float((np.abs(swapped) >= abs(observed) - 1e-12).mean()),
            "rescued_errors": int(rescued.sum()),
            "new_errors": int(harmed.sum()),
            "shared_errors": int(((cand_pred != labels) & (ref_pred != labels)).sum()),
            "both_correct": int(((cand_pred == labels) & (ref_pred == labels)).sum()),
            "net_correct_change": int(rescued.sum() - harmed.sum()),
            "rescue_harm_per_class": {
                name: {
                    "rescued": int(rescued[labels == i].sum()),
                    "harmed": int(harmed[labels == i].sum()),
                }
                for i, name in enumerate(CLASS_NAMES)
            },
            "per_scenario": {
                str(g): {
                    "macro_f1_delta": float(f1(cand_cm[i]).mean() - f1(ref_cm[i]).mean()),
                    "rescued": int(rescued[scenarios == g].sum()),
                    "harmed": int(harmed[scenarios == g].sum()),
                }
                for i, g in enumerate(groups)
            },
        }
    )
    return result


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvalues, key=lambda key: (pvalues[key], key))
    raw = np.array([pvalues[key] * (len(keys) - i) for i, key in enumerate(keys)])
    return dict(zip(keys, np.minimum(np.maximum.accumulate(raw), 1).tolist(), strict=True))


def stratum_pair(labels, candidate, reference, mask, shared):
    correct = candidate.argmax(1) == labels
    ref_correct = reference.argmax(1) == labels
    rescued, harmed = correct & ~ref_correct & mask, ~correct & ref_correct & mask
    changed = candidate.argmax(1) != reference.argmax(1)
    both_wrong = ~correct & ~ref_correct & mask

    def count(value):
        return int(np.count_nonzero(value))

    return {
        "rows": count(mask),
        "candidate_metrics": metrics(labels[mask], candidate[mask]),
        "reference_metrics": metrics(labels[mask], reference[mask]),
        "candidate_errors": count(~correct & mask),
        "reference_errors": count(~ref_correct & mask),
        "rescues": count(rescued),
        "harms": count(harmed),
        "net_corrections": count(rescued) - count(harmed),
        "both_correct": count(correct & ref_correct & mask),
        "both_wrong": count(both_wrong),
        "correctness_flips": count(rescued | harmed),
        "prediction_flips": count(changed & mask),
        "wrong_to_different_wrong": count(both_wrong & changed),
        "original_shared_rows": count(shared & mask),
        "original_shared_repairs": count(shared & correct & mask),
        "reference_original_shared_repairs": count(shared & ref_correct & mask),
        "original_shared_rescues_vs_reference": count(shared & rescued),
        "original_shared_harms_vs_reference": count(shared & harmed),
    }


def distribution(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    require(len(x) > 0, "Expected nonempty slot statistic")
    return {
        "count": len(x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.9)),
        "max": float(x.max()),
    }


def slot_mass_core(values, threshold):
    """Per-seed mass core only; never conflate unnamed slots between seeds."""
    reports = {}
    for seed, raw in zip((42, 43, 44), values, strict=True):
        mass = raw.astype(np.float64)
        require(np.isfinite(mass).all() and (mass >= 0).all(), "Invalid slot mass")
        total = mass.sum(1)
        positive = total > 1e-12
        shares = np.divide(mass, total[:, None], out=np.zeros_like(mass), where=positive[:, None])
        active = (shares >= threshold) & (mass > 0)
        reports[str(seed)] = {
            "valid_local_rows": len(mass),
            "masked_local_rows": 0,
            "positive_evidence_rows": int(positive.sum()),
            "zero_or_negligible_total_mass_rows": int((~positive).sum()),
            "total_matched_mass": distribution(total),
            "active_slot_count": distribution(active.sum(1)),
            "positive_rows_with_zero_active_slots": int((positive & (active.sum(1) == 0)).sum()),
            "positive_rows_with_one_active_slot": int((positive & (active.sum(1) == 1)).sum()),
            "batch_utilization": {
                "mean_raw_mass_by_slot": mass.mean(0).tolist(),
                "slots_with_zero_aggregate_mass": int((mass.sum(0) == 0).sum()),
            },
        }
    return {
        "evaluation_only": True,
        "action_labels_used": 0,
        "rows": values.shape[1],
        "seeds": 3,
        "slots": 6,
        "per_seed": reports,
    }


def evaluation_strata(data, evaluation):
    mappings = {
        "historical_short_fallback": "historical_short_fallback",
        "native_source_fallback": "native_source_fallback",
        "changed_source_input": "changed_input",
        "legacy_boundary": "complete_target_boundary",
        "legacy_stable": "complete_stable_target_window",
        "legacy_unknown": "unknown_target",
        **{
            name: name
            for name in (
                "sampled_node_pure",
                "sampled_node_mixed",
                "sampled_node_unknown",
                "dense_interval_stable",
                "dense_interval_boundary",
                "dense_interval_unknown",
                "pure_persistent_error",
                "medium_pure_persistent_error",
            )
        },
    }
    masks = {name: evaluation[source] for name, source in mappings.items()}
    masks["all"] = np.ones(len(data["labels"]), dtype=bool)
    masks["unchanged_source_input"] = ~evaluation["changed_input"]
    height = data["quality"][:, 0].astype(np.float64) * 720
    masks.update(
        {
            "native_height_le32": height <= 32 + 1e-4,
            "native_height_32to64": (height > 32 + 1e-4) & (height <= 64 + 1e-4),
            "native_height_gt64": height > 64 + 1e-4,
        }
    )
    masks.update(
        {
            "scenario_" + str(group): data["scenarios"] == group
            for group in np.unique(data["scenarios"])
        }
    )
    masks.update(
        {"class_" + name: data["labels"] == index for index, name in enumerate(CLASS_NAMES)}
    )
    require(
        all(
            value.shape == data["labels"].shape and value.dtype == np.bool_
            for value in masks.values()
        ),
        "Diagnostic strata not aligned booleans",
    )
    return masks


def decision_gates(results, reference, scenario_deltas, old_reference):
    primary = results[ARMS[5]]
    return {
        "macro_f1_at_least_0_84": primary["macro_f1"] >= 0.84,
        "macro_f1_at_least_0_85": primary["macro_f1"] >= 0.85,
        "beats_a1_dense_cnn_by_at_least_0_5_points": primary["macro_f1"]
        - results[ARMS[1]]["macro_f1"]
        >= 0.005,
        "beats_a2_dense_transformer_by_at_least_0_5_points": primary["macro_f1"]
        - results[ARMS[2]]["macro_f1"]
        >= 0.005,
        "no_nll_regression_vs_best_ordinary_dense": primary["nll"]
        <= min(results[ARMS[1]]["nll"], results[ARMS[2]]["nll"]),
        "no_brier_regression_vs_best_ordinary_dense": primary["brier"]
        <= min(results[ARMS[1]]["brier"], results[ARMS[2]]["brier"]),
        "positive_net_corrections_vs_new_source_m4": reference["all"]["net_corrections"] > 0,
        "scenario_accuracy_improves_at_least_7_of_11": sum(
            value > 0 for value in scenario_deltas.values()
        )
        >= 7,
        "no_scenario_accuracy_loss_over_2_points": min(scenario_deltas.values()) >= -0.02,
        "legacy_boundary_harms_fewer_than_historical_73_vs_old_source_p6": old_reference[
            "legacy_boundary"
        ]["harms"]
        < 73,
        "repairs_at_least_one_pure_persistent_error": reference["pure_persistent_error"][
            "original_shared_repairs"
        ]
        > 0,
    }


def completed_inventory(run: Path) -> tuple[dict, list[Path]]:
    """This gate reads no prediction arrays or model scores."""
    lock = read_json(run / "execution_lock.json")
    require(
        lock.get("status") == "SEAR_SIX_ARM_450_FIT_LOCKED_BEFORE_CLASSIFIER_FITTING",
        "Invalid execution lock",
    )
    protocol = lock["protocol"]
    require(tuple(protocol["arms"]) == ARMS, "Arm order changed")
    require(protocol["splitting"]["outer_folds"] == list(range(5)), "Outer fold grid changed")
    require(protocol["splitting"]["inner_folds"] == 3, "Inner fold grid changed")
    require(protocol["training"]["learning_rates"] == [0.0003, 0.001], "Learning-rate grid changed")
    require(protocol["training"]["weight_decays"] == [0.0001, 0.01], "Weight-decay grid changed")
    require(protocol["training"]["outer_seeds"] == [42, 43, 44], "Outer seeds changed")
    require(protocol["training"]["maximum_classifier_fits"] == 450, "Fit budget changed")
    require(
        protocol["training"]["max_epochs"] == 30
        and protocol["training"]["early_stopping_patience"] == 7,
        "Epoch/early-stop contract changed",
    )
    require(
        protocol["training"]["inner_seed"] == 42 and protocol["splitting"]["split_seed"] == 42,
        "Inner seed contract changed",
    )
    require(
        protocol["training"]["batch_size"] == 32 and protocol["class_order"] == list(CLASS_NAMES),
        "Batch/class contract changed",
    )
    require(
        tuple(map(tuple, protocol["statistics"]["directional_contrasts"])) == CONTRASTS,
        "Contrasts changed",
    )
    require(protocol["primary_rows"] == 4977, "Cohort size changed")
    require(
        lock["classifier_fits_completed_at_lock"] == 0 and lock["protected_rows_read"] == 0,
        "Lock chronology or protected-row declaration invalid",
    )
    expected_workloads = {
        run / "workloads" / arm / f"fold-{fold}" / "receipt.json"
        for arm in ARMS
        for fold in range(5)
    }
    require(
        set((run / "workloads").glob("*/*/receipt.json")) == expected_workloads,
        "Incomplete or foreign workload inventory; scores not opened",
    )
    fits = sorted((run / "fits").glob("*/receipt.json"))
    require(len(fits) == 450, "Incomplete or excessive fit inventory; scores not opened")
    directories = {path.parent for path in fits}
    require(
        set(path for path in (run / "fits").iterdir() if path.is_dir()) == directories,
        "Uncommitted or foreign fit directory",
    )
    for path in fits:
        require(
            all(
                (path.parent / name).is_file()
                for name in ("request.json", "checkpoint.pt", "predictions.npz", "progress.pt")
            ),
            "Incomplete fit payload; scores not opened",
        )
        require(
            {item.name for item in path.parent.iterdir()}
            == {"request.json", "checkpoint.pt", "predictions.npz", "progress.pt", "receipt.json"},
            "Foreign or unfinished fit payload remains",
        )
    for path in (*fits, *expected_workloads):
        required_status = "SEAR_FIT_COMPLETE" if path in fits else "SEAR_WORKLOAD_COMPLETE"
        require(
            read_json(path).get("status") == required_status,
            "An incomplete receipt remains; predictions not opened",
        )
    require(
        (run / RESULT_VERSION / "summary.json").is_file()
        and (run / RESULT_VERSION / "oof_probabilities.npz").is_file()
        and (run / RESULT_VERSION / "slot_evidence.npz").is_file(),
        "Final matrix results missing; scores not opened",
    )
    return lock, fits


def independent_request(
    data, arm, train, held, lr, wd, seed, device, lock_sha, fold, *, inner=None, epochs=None
):
    return {
        "execution_lock_sha256": lock_sha,
        "arm": arm,
        "outer_fold": int(fold),
        "inner_fold": inner,
        "stage": "inner" if epochs is None else "outer_refit",
        "learning_rate": lr,
        "weight_decay": wd,
        "seed": seed,
        "epochs": epochs,
        "train_ids_sha256": canonical(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical(data["labels"][train].tolist()),
        "held_ids_sha256": canonical(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": canonical(data["labels"][held].tolist())
        if epochs is None
        else None,
        "device": device,
    }


def audit_fit(run, evidence, expected, data, train, held, protocol, count, expected_schema=None):
    directory = run / "fits" / canonical(expected)
    require(not set(data["scenarios"][train]) & set(data["scenarios"][held]), "Scenario leakage")
    receipt_record = evidence.bind(directory / "receipt.json")
    receipt = read_json(directory / "receipt.json")
    require(receipt.get("status") == "SEAR_FIT_COMPLETE", "Incomplete fit receipt")
    require(receipt.get("request") == expected, "Fit request mismatch")
    require(
        set(receipt["artifacts"])
        == {"request.json", "checkpoint.pt", "predictions.npz", "progress.pt"},
        "Fit artifact inventory changed",
    )
    for name, entry in receipt["artifacts"].items():
        evidence.check(entry, directory / name)
    require(read_json(directory / "request.json") == expected, "Serialized fit request mismatch")
    require(
        receipt["train_rows"] == len(train) and receipt["held_rows"] == len(held),
        "Fit row counts changed",
    )
    require(receipt["parameters"] == count, "Fit parameter count changed")
    history = receipt["history"]
    require(
        len(history) == receipt["epochs_run"] and len(history) > 0, "Training history incomplete"
    )
    require(
        [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
        "Epoch history not contiguous",
    )
    require(
        all(
            np.isfinite(row[key])
            for row in history
            for key in ("training_loss", "classification_loss")
        ),
        "Nonfinite training history",
    )
    if expected["stage"] == "inner":
        require(
            all(
                row["validation_metrics"]["rows"] == len(held)
                and all(
                    np.isfinite(row["validation_metrics"][key])
                    for key in ("macro_f1", "accuracy", "nll", "brier")
                )
                for row in history
            ),
            "Invalid inner validation history",
        )
        require(len(history) <= protocol["training"]["max_epochs"], "Inner epoch limit exceeded")
        best = min(
            history,
            key=lambda row: (
                -row["validation_metrics"]["macro_f1"],
                row["validation_metrics"]["nll"],
                row["epoch"],
            ),
        )
        require(
            receipt["selected_epoch"] == best["epoch"], "Inner selected epoch not history optimum"
        )
        patience = protocol["training"]["early_stopping_patience"]
        running = None
        first_stop = None
        for row in history:
            key = (
                -row["validation_metrics"]["macro_f1"],
                row["validation_metrics"]["nll"],
                row["epoch"],
            )
            if running is None or key < running:
                running = key
            if row["epoch"] - running[2] >= patience:
                first_stop = row["epoch"]
                break
        require(
            first_stop is None or first_stop == len(history),
            "Inner fit continued past early-stop rule",
        )
        require(
            len(history) == protocol["training"]["max_epochs"] or first_stop == len(history),
            "Inner fit stopped early without declared patience",
        )
    else:
        require(
            len(history) == expected["epochs"] == receipt["selected_epoch"],
            "Outer refit epoch count differs from inner selection",
        )
        require(
            all("validation_metrics" not in row for row in history),
            "Outer labels entered epoch-selection history",
        )
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    require(
        checkpoint["request"] == expected and checkpoint["epoch"] == receipt["selected_epoch"],
        "Checkpoint ancestry mismatch",
    )
    state = checkpoint["state_dict"]
    require(
        state
        and all(
            isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all()
            for tensor in state.values()
        ),
        "Invalid checkpoint tensors",
    )
    excluded = {
        name
        for name in state
        if expected["arm"] in (ARMS[3], ARMS[4]) and name.startswith("extractor.nuisance.")
    }
    if expected["arm"] == ARMS[4]:
        excluded.add("template_positions")
    require(
        sum(tensor.numel() for name, tensor in state.items() if name not in excluded) == count,
        "Checkpoint active-parameter inventory differs from receipt",
    )
    progress = torch.load(directory / "progress.pt", map_location="cpu", weights_only=True)
    maximum = expected["epochs"] or protocol["training"]["max_epochs"]
    require(
        progress["request"] == expected and progress["maximum_epochs"] == maximum,
        "Progress checkpoint ancestry mismatch",
    )
    require(
        progress["history"] == history and progress["next_epoch"] == len(history) + 1,
        "Progress history mismatch",
    )
    selected_state = (
        progress["best_state"] if expected["stage"] == "inner" else progress["model_state"]
    )
    require(
        set(selected_state) == set(state)
        and all(torch.equal(selected_state[name], tensor) for name, tensor in state.items()),
        "Selected checkpoint is not exact declared progress state",
    )
    if expected["stage"] == "inner":
        require(
            progress["best_epoch"] == receipt["selected_epoch"], "Progress selected epoch mismatch"
        )
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        require(
            np.array_equal(saved["sample_ids"], data["sample_ids"][held]),
            "Fit held IDs/order changed",
        )
        arrays = {key: saved[key].copy() for key in saved.files if key != "sample_ids"}
    names = {"probabilities", "local_logits", "video_logits"}
    if expected["arm"] in TEMPLATE_ARMS:
        names.update({"slot_masses", "slot_positions", "slot_descriptors"})
    if expected["stage"] == "outer_refit":
        names.update(
            {
                "coordinate_shuffle_probabilities",
                "outer_border_mask_probabilities",
                "video_removed_probabilities",
            }
        )
    require(set(arrays) == names, "Fit prediction fields changed")
    for key, value in arrays.items():
        shape = {
            "slot_masses": (len(held), 6),
            "slot_positions": (len(held), 6, 2),
            "slot_descriptors": (len(held), 6, protocol["architecture"]["rank"]),
        }.get(key, (len(held), 3))
        require(
            value.shape == shape and np.isfinite(value).all(),
            "Invalid fit prediction payload",
        )
        if key.endswith("probabilities"):
            metrics(data["labels"][held], value)
    measured = metrics(data["labels"][held], arrays["probabilities"])
    if expected["stage"] == "inner":
        same(measured, receipt["held_metrics"], "fit held metrics")
        same(
            measured,
            history[receipt["selected_epoch"] - 1]["validation_metrics"],
            "selected checkpoint validation metrics",
        )
    else:
        require(
            "held_metrics" not in receipt and receipt.get("outer_held_metrics_embargoed") is True,
            "Outer metric embargo violated",
        )
        logits = arrays["local_logits"] - arrays["local_logits"].max(1, keepdims=True)
        exponentials = np.exp(logits)
        local_only = exponentials / exponentials.sum(1, keepdims=True)
        require(
            np.array_equal(local_only, arrays["video_removed_probabilities"]),
            "Video-removed diagnostic not exact local-logit softmax",
        )
    schema = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in state.items()
    }
    if expected_schema is not None:
        require(
            schema == expected_schema, "Checkpoint schema differs from label-blind resource model"
        )
    return arrays, receipt, receipt_record, canonical(schema)


def audit_workload(run, evidence, lock, data, arm, fold, device, state_schema=None):
    protocol = lock["protocol"]
    lock_sha = sha256(run / "execution_lock.json")
    outer_train = np.flatnonzero(data["folds"] != fold)
    outer_held = np.flatnonzero(data["folds"] == fold)
    splitter = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=protocol["splitting"]["split_seed"]
    )
    splits = [
        (outer_train[fit], outer_train[held])
        for fit, held in splitter.split(
            np.zeros(len(outer_train)), data["labels"][outer_train], data["scenarios"][outer_train]
        )
    ]
    candidates, fit_records, prediction_records, schemas = [], [], [], set()
    expected_fit_paths = set()

    def collect(train, held, lr, wd, seed, *, inner=None, epochs=None):
        request = independent_request(
            data, arm, train, held, lr, wd, seed, device, lock_sha, fold, inner=inner, epochs=epochs
        )
        arrays, receipt, record, schema = audit_fit(
            run,
            evidence,
            request,
            data,
            train,
            held,
            protocol,
            lock["parameter_counts"][arm],
            state_schema,
        )
        expected_fit_paths.add(Path(record["path"]))
        fit_records.append(record)
        prediction_records.append(receipt["artifacts"]["predictions.npz"])
        schemas.add(schema)
        return arrays, receipt

    for lr in protocol["training"]["learning_rates"]:
        for wd in protocol["training"]["weight_decays"]:
            oof = np.full((len(data["labels"]), 3), np.nan)
            epochs, coverage = [], np.zeros(len(data["labels"]), dtype=np.int64)
            for inner, (train, held) in enumerate(splits):
                arrays, receipt = collect(
                    train, held, lr, wd, protocol["training"]["inner_seed"], inner=inner
                )
                oof[held] = arrays["probabilities"]
                coverage[held] += 1
                epochs.append(receipt["selected_epoch"])
            require(
                (coverage[outer_train] == 1).all() and not coverage[outer_held].any(),
                "Inner OOF coverage mismatch",
            )
            candidates.append(
                {
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "inner_metrics": metrics(data["labels"][outer_train], oof[outer_train]),
                    "inner_best_epochs": epochs,
                }
            )
    selected = min(
        candidates,
        key=lambda value: (
            -value["inner_metrics"]["macro_f1"],
            value["inner_metrics"]["nll"],
            value["learning_rate"],
            value["weight_decay"],
        ),
    )
    epochs = max(1, int(np.rint(np.median(selected["inner_best_epochs"]))))
    outputs = [
        collect(
            outer_train,
            outer_held,
            selected["learning_rate"],
            selected["weight_decay"],
            seed,
            epochs=epochs,
        )[0]
        for seed in protocol["training"]["outer_seeds"]
    ]
    require(
        len(expected_fit_paths) == 15 and len(schemas) == 1,
        "Nonunique fits or changing model schema",
    )
    directory = run / "workloads" / arm / f"fold-{fold}"
    evidence.bind(directory / "receipt.json")
    receipt = read_json(directory / "receipt.json")
    require(
        receipt.get("status") == "SEAR_WORKLOAD_COMPLETE" and receipt["unique_fits"] == 15,
        "Workload incomplete",
    )
    require(
        receipt["request"] == {"execution_lock_sha256": lock_sha, "arm": arm, "fold": fold},
        "Workload lock ancestry mismatch",
    )
    require(receipt.get("fit_device") == device, "Workload device differs from resource gate")
    same(candidates, receipt["candidates"], "workload candidates")
    same(selected, receipt["selected"], "workload selection")
    require(epochs == receipt["outer_refit_epochs"], "Outer epoch rule mismatch")
    require(
        receipt.get("fit_receipts") == fit_records, "Workload full-fit receipt ancestry mismatch"
    )
    if "fit_prediction_receipts" in receipt:
        require(
            receipt["fit_prediction_receipts"] == prediction_records,
            "Workload prediction ancestry mismatch",
        )
    evidence.check(receipt["artifact"], directory / "predictions.npz")
    arrays = {"seed_" + name: np.stack([value[name] for value in outputs]) for name in outputs[0]}
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        require(
            set(saved.files) == {"sample_ids", *arrays}, "Workload prediction inventory mismatch"
        )
        require(
            np.array_equal(saved["sample_ids"], data["sample_ids"][outer_held]),
            "Workload held IDs mismatch",
        )
        for name, value in arrays.items():
            require(
                np.array_equal(saved[name], value),
                f"Workload not exact individual-fit predictions: {name}",
            )
    return (
        arrays,
        expected_fit_paths,
        {
            "arm": arm,
            "fold": fold,
            "selected": selected,
            "outer_refit_epochs": epochs,
            "fit_receipts": fit_records,
            "checkpoint_schema_sha256": next(iter(schemas)),
        },
    )


def audit(run: Path) -> dict:
    run = run.resolve()
    lock, actual_fit_paths = completed_inventory(run)
    require(
        lock["parameter_counts"] == EXPECTED_PARAMETERS, "Locked active-capacity design changed"
    )
    evidence = Evidence()
    evidence.bind(run / "execution_lock.json")
    evidence.bind(Path(__file__))
    for entry in (*lock["source_artifacts"].values(), *lock["dense_artifacts"].values()):
        evidence.check(entry)
    required_sources = (
        "src/hac/sear.py",
        "src/hac/sear_controls.py",
        "src/hac/sear_training.py",
        "src/hac/sear_evaluation.py",
        "src/hac/actor_memory_base.py",
        "src/hac/source_swap_data.py",
        "experiments/run_okutama_video_probe.py",
        "experiments/run_okutama_sear_matrix.py",
        "experiments/okutama_sear_matrix_protocol.json",
    )
    require(
        all(str((ROOT / name).resolve()) in evidence.records for name in required_sources),
        "Executable semantics helpers omitted from lock",
    )
    protocol_path = ROOT / "experiments/okutama_sear_matrix_protocol.json"
    require(
        read_json(protocol_path) == lock["protocol"],
        "Locked protocol differs from hashed protocol artifact",
    )
    require(
        str((run / "population.npz").resolve()) in evidence.records, "Population not bound by lock"
    )
    with np.load(run / "population.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    n = lock["protocol"]["primary_rows"]
    require(
        len(data["sample_ids"]) == n and len(np.unique(data["sample_ids"])) == n,
        "Population IDs nonunique/wrong length",
    )
    require(
        canonical(data["sample_ids"].tolist()) == lock["sample_ids_sha256"],
        "Locked ID hash mismatch",
    )
    require(
        canonical(data["labels"].tolist()) == lock["labels_sha256"], "Locked label hash mismatch"
    )
    for name in ("folds", "scenarios"):
        require(
            canonical(data[name].tolist()) == lock[name + "_sha256"], f"Locked {name} hash mismatch"
        )
    require(np.array_equal(np.unique(data["labels"]), np.arange(3)), "Class inventory mismatch")
    require(np.array_equal(np.unique(data["folds"]), np.arange(5)), "Fold inventory mismatch")
    require(len(np.unique(data["scenarios"])) == 11, "Scenario count mismatch")
    for group in np.unique(data["scenarios"]):
        require(
            len(np.unique(data["folds"][data["scenarios"] == group])) == 1,
            "Scenario crosses outer folds",
        )
    dense_ids = np.load(lock["dense_artifacts"]["sample_ids.npy"]["path"], allow_pickle=False)
    require(np.array_equal(dense_ids, data["sample_ids"]), "Dense IDs mismatch")
    valid = np.load(lock["dense_artifacts"]["local_valid.npy"]["path"], allow_pickle=False)
    complete = np.load(lock["dense_artifacts"]["completed.npy"]["path"], allow_pickle=False)
    require(
        valid.dtype == np.bool_
        and valid.shape == (n,)
        and np.array_equal(valid, data["local_valid"]),
        "Local-evidence mask mismatch",
    )
    require(
        complete.dtype == np.bool_ and complete.shape == (n,) and complete.all(),
        "Dense cache incomplete",
    )
    require(valid.all(), "Unexpected unavailable center in fully valid locked cohort")
    for filename, shape in (("patch_tokens.npy", (n, 729, 768)), ("center_cls.npy", (n, 768))):
        array = np.load(
            lock["dense_artifacts"][filename]["path"], mmap_mode="r", allow_pickle=False
        )
        require(
            array.shape == shape and array.dtype == np.float16,
            f"Dense shape/dtype mismatch: {filename}",
        )
    frames = np.load(lock["dense_artifacts"]["source_frames.npy"]["path"], allow_pickle=False)
    fallback = np.load(
        lock["dense_artifacts"]["historical_short_fallback.npy"]["path"], allow_pickle=False
    )
    require(np.array_equal(frames, data["frames"]), "Dense center-frame identity mismatch")
    require(
        fallback.dtype == np.bool_
        and fallback.shape == (n,)
        and fallback.sum() == 467
        and np.array_equal(fallback, ~data["long_valid"]),
        "Historical fallback identity mismatch",
    )
    require(
        np.array_equal(np.bincount(data["labels"], minlength=3), [734, 2118, 2125]),
        "Class support changed",
    )
    require(
        data["video"].shape == (n, 1536)
        and data["video"].dtype == np.float32
        and np.isfinite(data["video"]).all(),
        "Frozen video shape/dtype invalid",
    )
    # Re-derive the exact video and metadata population from the locked source.
    source_paths = [
        Path(entry["path"])
        for entry in lock["source_artifacts"].values()
        if Path(entry["path"]).name == "memory_data.npz"
    ]
    require(len(source_paths) == 1, "Ambiguous source population")
    with np.load(source_paths[0], allow_pickle=False) as saved:
        for name, values in data.items():
            if name == "local_valid":
                continue
            expected = saved["features"][:, :1536] if name == "video" else saved[name]
            require(
                equal_array(values, expected),
                f"Population not derived from locked source: {name}",
            )
    resource_path = run / "resource_pilot.json"
    require(str(resource_path) in evidence.records, "Resource gate not locked")
    resource = read_json(resource_path)
    require(resource["protocol_sha256"] == sha256(protocol_path), "Resource protocol hash mismatch")
    require(
        resource.get("synthetic_only") is True
        and resource.get("action_labels_read") == 0
        and resource.get("optimizer_steps") == 0,
        "Resource pilot exceeded label-blind no-fit scope",
    )
    require(
        resource.get("passed") and resource.get("status") == "SEAR_SIX_ARM_RESOURCE_GATE_PASS",
        "Resource gate failed",
    )
    device = resource["device"]
    require(
        device.startswith("cuda") and resource["batch_size"] == 32, "Resource device/batch mismatch"
    )
    require(tuple(resource["results"]) == ARMS, "Resource arm inventory mismatch")
    require(
        resource["runtime"] == lock["runtime"] and resource["gpu"] == lock["gpu"],
        "Runtime or GPU resource-lock mismatch",
    )
    runtime = resource["runtime"]
    require(
        runtime["cublas_workspace_config"] == ":4096:8"
        and runtime["deterministic_algorithms"]
        and not runtime["matmul_allow_tf32"]
        and not runtime["cudnn_allow_tf32"]
        and runtime["float32_matmul_precision"] == "highest",
        "Execution determinism settings differ from contract",
    )
    for name, digest in resource["code_sha256"].items():
        require(
            str(Path(name).resolve()) in evidence.records
            and evidence.records[str(Path(name).resolve())]["sha256"] == digest,
            "Resource code not execution-locked",
        )
    require(
        set(resource["code_sha256"])
        == {
            str((ROOT / name).resolve()) for name in required_sources if not name.endswith(".json")
        },
        "Resource executable-code inventory incomplete or foreign",
    )
    require(
        resource["runtime"]["sklearn"] == sklearn.__version__,
        "Audit split-library version differs from locked execution",
    )
    for arm, result in resource["results"].items():
        require(
            result["parameters"] == lock["parameter_counts"][arm]
            and result["finite_active_gradients"]
            and result["exact_video_fallback"]
            and 0 < result["peak_cuda_allocated_bytes"] < 4 * 1024**3,
            "Invalid per-arm resource gate",
        )
        require(
            isinstance(result.get("state_dict_schema"), dict) and result["state_dict_schema"],
            "Resource checkpoint schema not bound",
        )
        state_schema = result["state_dict_schema"]
        excluded = {
            name
            for name in state_schema
            if arm in (ARMS[3], ARMS[4]) and name.startswith("extractor.nuisance.")
        }
        if arm == ARMS[4]:
            excluded.add("template_positions")
        require(
            len(result["trainable_parameter_names"])
            == len(set(result["trainable_parameter_names"]))
            and set(result["trainable_parameter_names"]) == set(state_schema) - excluded,
            "Resource active-parameter name inventory differs from fixed architecture",
        )
        prefix = "video_head." if arm in ARMS[:3] else "video."
        common = {
            name.removeprefix(prefix): value
            for name, value in state_schema.items()
            if name.startswith(prefix)
        }
        require(
            common and common == resource["common_video_state_dict_schema"],
            "Resource common-video schema differs between arms",
        )
    labels, scenarios = data["labels"], data["scenarios"]
    seeds, slots, all_expected, workloads = {}, {}, set(), []
    diagnostic_seeds = {condition: {} for condition in CONDITIONS}
    for arm in ARMS:
        seeds[arm] = np.full((3, n, 3), np.nan)
        for condition in CONDITIONS:
            diagnostic_seeds[condition][arm] = np.full((3, n, 3), np.nan)
        if arm in TEMPLATE_ARMS:
            slots[arm] = {
                "masses": np.full((3, n, 6), np.nan, dtype=np.float32),
                "positions": np.full((3, n, 6, 2), np.nan, dtype=np.float32),
                "descriptors": np.full((3, n, 6, 32), np.nan, dtype=np.float32),
            }
        for fold in range(5):
            arrays, paths, receipt = audit_workload(
                run,
                evidence,
                lock,
                data,
                arm,
                fold,
                device,
                resource["results"][arm]["state_dict_schema"],
            )
            all_expected.update(paths)
            workloads.append(receipt)
            held = np.flatnonzero(data["folds"] == fold)
            seeds[arm][:, held] = arrays["seed_probabilities"]
            for condition in CONDITIONS:
                diagnostic_seeds[condition][arm][:, held] = arrays[
                    "seed_" + condition + "_probabilities"
                ]
            if arm in TEMPLATE_ARMS:
                for name in slots[arm]:
                    slots[arm][name][:, held] = arrays["seed_slot_" + name]
    require(
        all_expected == {path.resolve() for path in actual_fit_paths} and len(all_expected) == 450,
        "Exact reconstructed fit inventory differs",
    )
    probabilities = {arm: values.mean(0) for arm, values in seeds.items()}
    diagnostics = {
        arm + "_" + condition: diagnostic_seeds[condition][arm].mean(0)
        for condition in CONDITIONS
        for arm in ARMS
    }
    evaluation_path = run / "evaluation_inputs.npz"
    require(str(evaluation_path) in evidence.records, "Evaluation inputs not execution-locked")
    with np.load(evaluation_path, allow_pickle=False) as saved:
        require(
            np.array_equal(saved["sample_ids"], data["sample_ids"]),
            "Evaluation input identity mismatch",
        )
        evaluation = {name: saved[name].copy() for name in saved.files if name != "sample_ids"}
    references = {name: evaluation[name] for name in REFERENCES}
    summary_path = run / RESULT_VERSION / "summary.json"
    evidence.bind(summary_path)
    summary = read_json(summary_path)
    require(
        summary.get("status") == "SEAR_SIX_ARM_450_FIT_COMPLETE"
        and summary["classifier_fits"] == 450,
        "Final summary incomplete",
    )
    require(
        summary["rows"] == n
        and summary["no_outer_selected_fusion"]
        and summary["protected_rows_read"] == 0,
        "Final scope declaration mismatch",
    )
    require(
        summary["execution_lock_sha256"] == sha256(run / "execution_lock.json"),
        "Summary execution lock ancestry mismatch",
    )
    expected_workload_hashes = {
        str((run / "workloads" / arm / f"fold-{fold}" / "receipt.json").resolve()): sha256(
            run / "workloads" / arm / f"fold-{fold}" / "receipt.json"
        )
        for arm in ARMS
        for fold in range(5)
    }
    require(
        summary["workload_receipt_sha256"] == expected_workload_hashes,
        "Summary workload ancestry mismatch",
    )
    oof_path = run / RESULT_VERSION / "oof_probabilities.npz"
    evidence.check(summary["artifacts"]["oof_probabilities.npz"], oof_path)
    with np.load(oof_path, allow_pickle=False) as saved:
        require(
            set(saved.files)
            == {
                "sample_ids",
                "labels",
                "scenarios",
                "folds",
                *ARMS,
                *(arm + "_seeds" for arm in ARMS),
                *REFERENCES,
                *diagnostics,
            },
            "Final OOF inventory changed",
        )
        for name in ("sample_ids", "labels", "scenarios", "folds"):
            require(np.array_equal(saved[name], data[name]), f"Final OOF identity mismatch: {name}")
        for arm in ARMS:
            require(
                np.array_equal(saved[arm], probabilities[arm])
                and np.array_equal(saved[arm + "_seeds"], seeds[arm]),
                "Final OOF is not exact outer three-seed mean",
            )
        for name, values in {**references, **diagnostics}.items():
            require(
                np.array_equal(saved[name], values),
                f"Final reference or diagnostic not exact: {name}",
            )
    slot_path = run / RESULT_VERSION / "slot_evidence.npz"
    evidence.check(summary["artifacts"]["slot_evidence.npz"], slot_path)
    expected_slots = {
        arm + "_seed_slot_" + name: values
        for arm, fields in slots.items()
        for name, values in fields.items()
    }
    with np.load(slot_path, allow_pickle=False) as saved:
        require(
            set(saved.files) == {"sample_ids", "outer_seeds", *expected_slots},
            "Slot-evidence artifact inventory mismatch",
        )
        require(
            np.array_equal(saved["sample_ids"], data["sample_ids"])
            and np.array_equal(saved["outer_seeds"], [42, 43, 44]),
            "Slot-evidence identity mismatch",
        )
        for name, values in expected_slots.items():
            require(
                saved[name].dtype == values.dtype and np.array_equal(saved[name], values),
                "Final slot evidence differs from original fit outputs",
            )
    results = {arm: metrics(labels, value) for arm, value in probabilities.items()}
    for name in (
        "results",
        "seed_metrics",
        "scenario_metrics",
        "reference_diagnostics",
        "fixed_inference_diagnostics",
    ):
        require(set(summary[name]) == set(ARMS), f"Summary arm inventory mismatch: {name}")
    same(results, summary["results"], "outer metrics")
    same(
        {arm: [metrics(labels, p) for p in values] for arm, values in seeds.items()},
        summary["seed_metrics"],
        "seed metrics",
    )
    same(
        {
            arm: {
                str(group): metrics(labels[scenarios == group], value[scenarios == group])
                for group in np.unique(scenarios)
            }
            for arm, value in probabilities.items()
        },
        summary["scenario_metrics"],
        "scenario metrics",
    )
    core_slots = {
        arm: slot_mass_core(
            value["masses"], lock["protocol"]["fixed_diagnostics"]["slot_active_share_threshold"]
        )
        for arm, value in slots.items()
    }
    require(
        set(summary["slot_diagnostics"]) == set(TEMPLATE_ARMS), "Slot report arm inventory changed"
    )
    same(core_slots, summary["slot_diagnostics"], "per-seed slot mass core")
    same(
        {name: metrics(labels, values) for name, values in references.items()},
        summary["source_reference_results"],
        "source reference metrics",
    )
    strata = evaluation_strata(data, evaluation)
    shared = evaluation["original_shared_p6_failures"]
    reference_reports = {}
    for arm in ARMS:
        reference_reports[arm] = {
            name: {
                key: stratum_pair(labels, probabilities[arm], reference, mask, shared)
                for key, mask in strata.items()
            }
            for name, reference in references.items()
        }
        declared = summary["reference_diagnostics"][arm]
        require(
            set(declared["references"]) == set(REFERENCES), "Reference report inventory changed"
        )
        for name, values in reference_reports[arm].items():
            require(
                set(declared["references"][name]["strata"]) == set(strata),
                "Reference stratum inventory changed",
            )
            same(
                values,
                declared["references"][name]["strata"],
                "reference stratum metrics and corrections",
            )
        require(
            set(summary["fixed_inference_diagnostics"][arm]) == set(CONDITIONS),
            "Fixed diagnostic inventory changed",
        )
        for condition in CONDITIONS:
            value = diagnostics[arm + "_" + condition]
            expected = {
                "metrics": metrics(labels, value),
                "vs_unperturbed": stratum_pair(
                    labels, value, probabilities[arm], strata["all"], shared
                ),
            }
            same(
                expected,
                summary["fixed_inference_diagnostics"][arm][condition],
                "fixed inference diagnostic",
            )
    scenario_deltas = {
        str(group): metrics(labels[scenarios == group], probabilities[ARMS[5]][scenarios == group])[
            "accuracy"
        ]
        - metrics(labels[scenarios == group], references["new_source_m4"][scenarios == group])[
            "accuracy"
        ]
        for group in np.unique(scenarios)
    }
    same(
        scenario_deltas,
        summary["scenario_accuracy_delta_vs_new_source_m4"],
        "scenario accuracy deltas",
    )
    same(
        decision_gates(
            results,
            reference_reports[ARMS[5]]["new_source_m4"],
            scenario_deltas,
            reference_reports[ARMS[5]]["old_source_p6"],
        ),
        summary["decision_gates"],
        "engineering decision gates",
    )
    statistics = lock["protocol"]["statistics"]
    require(
        statistics["bootstrap_resamples"] == 10000 and statistics["scenario_swaps"] == 2048,
        "Randomization grid changed",
    )
    comparisons = {
        candidate + "_vs_" + reference: contrast(
            labels,
            probabilities[candidate],
            probabilities[reference],
            scenarios,
            resamples=10000,
            seed=statistics["seed"],
        )
        for candidate, reference in CONTRASTS
    }
    adjusted = holm(
        {name: value["one_sided_exact_swap_pvalue"] for name, value in comparisons.items()}
    )
    for name, value in comparisons.items():
        value["holm_adjusted_pvalue"] = adjusted[name]
    require(set(summary["comparisons"]) == set(comparisons), "Contrast family changed")
    same(comparisons, summary["comparisons"], "six scenario contrasts")
    require(summary["primary_arm"] == ARMS[5], "Primary arm changed")
    same(results[ARMS[5]]["macro_f1"], summary["primary_macro_f1"], "primary metric")
    evidence.finish()
    return {
        "status": "SEAR_INDEPENDENT_450_FIT_AUDIT_PASS",
        "scope": "Completed adaptive-development matrix; no new fits, labels, selection or model changes",
        "run": str(run),
        "rows": n,
        "fit_count": len(all_expected),
        "workload_count": len(workloads),
        "classifier_fits_executed_by_auditor": 0,
        "protected_rows_read": 0,
        "exact_checkpoint_ancestry_verified": True,
        "all_checkpoint_schemas_match_label_blind_resource_gate": True,
        "all_checkpoint_forward_passes_replayed": False,
        "exact_outer_seed_means_verified": True,
        "inner_selection_reconstructed": True,
        "metric_absolute_tolerance": 1e-10,
        "limitations": [
            "Receipt and checkpoint ancestry checks do not independently rerun every training step or prediction forward pass.",
            "No OS-level adversarial writer protection; file hashes are checked on read and size/mtime again before output.",
            "Per-seed slot mass core is independently recomputed; detailed descriptor cosine/position redundancy summaries are not. Their saved per-fit evidence arrays are checked exactly.",
        ],
        "audit_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "results": results,
        "comparisons": comparisons,
        "workloads": workloads,
        "verified_artifacts": evidence.records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output", type=Path, required=True, help="New, separate immutable audit directory"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    run = args.run.resolve()
    require(not output.exists(), "Audit output already exists; will not overwrite")
    require(
        output != run and run not in output.parents,
        "Audit output must be separate from the experiment",
    )
    result = audit(run)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "fits": result["fit_count"],
                "output": str(output),
                "summary_sha256": sha256(output / "summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
