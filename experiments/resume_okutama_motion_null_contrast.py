"""Append-only resumption of nine fits after the user-approved fold-0 exception.

The locked runner, protocol, checkpoint and predictions are not modified.
Only the first plain fit is exempted from its historical residual tolerance.
All remaining fits use the original runner and original checks without retry.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_okutama_motion_null_contrast as trial


def schedule_digest(train):
    """Reconstruct the fixed row schedule, without optimizing or reading labels."""
    rng = np.random.default_rng(42)
    order = np.empty(0, np.int64)
    cursor = 0
    digest = hashlib.sha256()
    for _ in range(256):
        if cursor + 128 > len(order):
            order = rng.permutation(len(train))
            cursor = 0
        local = order[cursor:cursor + 128]
        cursor += len(local)
        digest.update(np.asarray(train[local], dtype="<i8").tobytes())
    return digest.hexdigest()


def resume():
    run = trial.RUN
    trial.validate(run)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if trial.read(run / "preflight.json")["status"] != "MOTION_NULL_PREFLIGHT_PASS":
        raise RuntimeError("Original preflight must have passed")
    first = run / "plain/fold-0"
    if (run / "numerical_parity_exception.json").exists():
        raise RuntimeError("This authorization adapter is one-shot; no automatic retries")
    if {p.name for p in first.iterdir()} != {"checkpoint.pt", "predictions.npz"}:
        raise RuntimeError("Unexpected first-control artifacts")
    for fold in range(5):
        for arm in trial.ARMS:
            if (arm, fold) != ("plain", 0) and (run / arm / f"fold-{fold}").exists():
                raise RuntimeError("Unexpected additional fit: refuse selection or overwrite")
    cohort = trial.arrays(run / "inputs/cohort.npz")
    inputs = trial.arrays(run / "inputs/fold-0.npz")
    saved = trial.arrays(first / "predictions.npz")
    old = trial.arrays(trial.legacy.DEFAULT_RUN / "signed_motion/fold-0/final/predictions.npz")
    stop = trial.read(run / "PARITY_STOP.json")
    if not np.array_equal(saved["held_rows"], inputs["held_rows"]):
        raise RuntimeError("First-control held rows changed")
    if not np.array_equal(saved["sample_ids"], cohort["sample_ids"][inputs["held_rows"]]):
        raise RuntimeError("First-control identities changed")
    delta_difference = float(np.max(np.abs(saved["delta"] - old["delta"])))
    probability_difference = float(np.max(np.abs(saved["candidate_probabilities"] - old["candidate_probabilities"])))
    disagreements = int(np.count_nonzero(saved["candidate_probabilities"].argmax(1) != old["candidate_probabilities"].argmax(1)))
    if delta_difference != stop["observed_delta_max_abs"] or probability_difference != stop["observed_probability_max_abs"] or disagreements:
        raise RuntimeError("The approved exception does not match the saved first control")
    hashes = {name: trial.file_sha256(first / name) for name in ("checkpoint.pt", "predictions.npz")}
    amendment = {
        "status": "USER_APPROVED_FIRST_CONTROL_NUMERICAL_EXCEPTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user_authorization": "Complete the remaining nine fits with the documented numerical exception",
        "scope": "plain fold-0 only; use its existing checkpoint and predictions without retry or selection",
        "execution_lock_sha256": trial.file_sha256(run / "execution_lock.json"),
        "protocol_sha256": trial.file_sha256(trial.PROTOCOL),
        "locked_runner_sha256": trial.file_sha256(Path(trial.__file__)),
        "resume_source_sha256": trial.file_sha256(Path(__file__)),
        "technical_stop_sha256": trial.file_sha256(run / "PARITY_STOP.json"),
        "first_control_hashes": hashes,
        "original_delta_atol": 1e-6,
        "observed_delta_max_abs": delta_difference,
        "observed_probability_max_abs": probability_difference,
        "class_prediction_disagreements": disagreements,
        "original_numeric_parity_pass": False,
        "performance_and_safety_gates_changed": False,
        "remaining_control_numerical_checks_changed": False,
        "new_held_scores_computed": False,
        "retraining_or_selection_allowed": False,
        "preserved_files_verified": trial.check_preservation(),
        "missing_telemetry": "Original first-fit losses and elapsed time were not persisted before the parity exception; recorded as null, not reconstructed."
    }
    trial.write(run / "numerical_parity_exception.json", amendment)
    amendment_sha = trial.file_sha256(run / "numerical_parity_exception.json")
    # The locked runner saves its checkpoint only after all 256 updates complete.
    receipt = {
        "status": "MOTION_NULL_FIT_COMPLETE_OUTCOMES_EMBARGOED",
        "arm": "plain", "fold": 0, "seed": 42, "updates": 256,
        "active_parameters": 120129,
        "execution_lock_sha256": amendment["execution_lock_sha256"],
        "numerical_exception_sha256": amendment_sha,
        "receipt_recovered_without_training": True,
        "updates_evidence": "Locked fit_one writes checkpoint only after the complete 256-update loop",
        "elapsed_training_seconds": None, "loss_curve": None,
        "missing_telemetry_reason": amendment["missing_telemetry"],
        "training_schedule_sha256": schedule_digest(inputs["train_rows"]),
        "training_schedule_provenance": "Deterministically reconstructed from locked seed, rows, batch size and update count; not an independently observed log",
        "training_label_rows": len(inputs["train_rows"]),
        "fit_ids_sha256": str(inputs["fit_ids_sha256"].item()),
        "held_ids_sha256": str(inputs["held_ids_sha256"].item()),
        "held_labels_passed_to_optimizer": False,
        "new_held_scores_computed": False,
        "historical_plain_parity": {"max_abs_delta": delta_difference, "same_class_predictions": True,
                                    "original_numeric_parity_pass": False, "accepted_by_user_exception": True},
        "checkpoint_sha256": hashes["checkpoint.pt"],
        "predictions_sha256": hashes["predictions.npz"]
    }
    trial.write(first / "receipt.json", receipt)
    cache = trial.load_native_motion_cache(trial.legacy.CACHE, cohort["sample_ids"])
    geometry = trial.motion_geometry(cache.geometry, cache.camera)
    started = time.perf_counter()
    completed = 0
    print(json.dumps({"event": "approved_resume", "retained_fits": 1, "remaining_fits": 9,
                      "gpu": torch.cuda.get_device_name(), "amendment_sha256": amendment_sha}), flush=True)
    for fold in range(5):
        for arm in trial.ARMS:
            if (arm, fold) == ("plain", 0):
                continue
            trial.fit_one(run, arm, fold, cohort, cache, geometry, "cuda", started)
            completed += 1
    for name, digest in hashes.items():
        if trial.file_sha256(first / name) != digest:
            raise RuntimeError("First control changed during resumption")
    result = {"status": "MOTION_NULL_TEN_FITS_COMPLETE", "head_fits": completed + 1,
              "router_fits": 0, "preexisting_fits_retained": 1, "new_fits_this_resume": completed,
              "new_held_scores_computed": False, "numerical_exception_sha256": amendment_sha,
              "elapsed_resumed_queue_seconds": time.perf_counter() - started,
              "elapsed_original_first_fit_seconds": None,
              "first_control_hashes_preserved": True, "performance_and_safety_gates_changed": False}
    trial.write(run / "queue_receipt.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(resume(), indent=2), flush=True)
