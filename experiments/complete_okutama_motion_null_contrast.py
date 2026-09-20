"""Finish seven first-attempt fits under the approved numerical-only amendment."""
from __future__ import annotations

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
from experiments.resume_okutama_motion_null_contrast import schedule_digest


def parity_measurements(saved, historical, tolerance):
    for name in ("sample_ids", "anchor_probabilities"):
        if not np.array_equal(saved[name], historical[name]):
            raise RuntimeError(f"Numerical exception cannot waive {name} mismatch")
    delta, p = saved["delta"], saved["candidate_probabilities"]
    if not np.isfinite(delta).all() or not np.isfinite(p).all() or np.max(np.abs(delta)) > 0.5:
        raise RuntimeError("Numerical exception cannot waive finiteness or residual bounds")
    if np.any(p < 0) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12):
        raise RuntimeError("Invalid candidate probabilities")
    disagreements = int(np.count_nonzero(p.argmax(1) != historical["candidate_probabilities"].argmax(1)))
    if disagreements:
        raise RuntimeError("Numerical exception cannot waive class-prediction mismatch")
    difference = float(np.max(np.abs(delta - historical["delta"])))
    return {"max_abs_delta": difference,
            "max_abs_probability": float(np.max(np.abs(p - historical["candidate_probabilities"]))),
            "same_class_predictions": True, "class_prediction_disagreements": disagreements,
            "original_numeric_parity_pass": difference <= tolerance,
            "accepted_by_user_exception": difference > tolerance}


def check_control(run, fold, cohort, lock):
    inputs = trial.arrays(run / "inputs" / f"fold-{fold}.npz")
    folder = run / "plain" / f"fold-{fold}"
    historical = trial.legacy.DEFAULT_RUN / "signed_motion" / f"fold-{fold}" / "final"
    old_receipt = trial.read(historical / "receipt.json")
    saved = trial.arrays(folder / "predictions.npz")
    rows = inputs["held_rows"]
    if not np.array_equal(saved["held_rows"], rows) or not np.array_equal(saved["sample_ids"], cohort["sample_ids"][rows]):
        raise RuntimeError("Held population mismatch")
    for key in ("fit_ids_sha256", "held_ids_sha256"):
        if str(inputs[key].item()) != old_receipt[key] or old_receipt[key] != lock["populations"][str(fold)][key]:
            raise RuntimeError("Training/prediction population provenance changed")
    if old_receipt["ancestry"] != lock["populations"][str(fold)]["ancestry"]:
        raise RuntimeError("Ancestor provenance changed")
    current_transform = trial.arrays(run / "transforms" / f"outer-{fold}_final.npz")
    original_transform = trial.arrays(ROOT / old_receipt["transform_path"])
    if any(not np.array_equal(current_transform[k], original_transform[k]) for k in ("mean", "scale")):
        raise RuntimeError("Geometry transform changed")
    report = parity_measurements(saved, trial.arrays(historical / "predictions.npz"),
                                 lock["protocol"]["historical_plain_parity"]["delta_atol"])
    report.update({"fold": fold, "input_provenance_matches": True,
                   "training_inputs_hash_verified_by_execution_lock": True})
    return report


def exception_telemetry(error):
    trace = error.__traceback__
    while trace:
        if trace.tb_frame.f_code is trial.fit_one.__code__:
            state = trace.tb_frame.f_locals
            if state.get("update") != 256:
                raise RuntimeError("Cannot accept incomplete optimizer run")
            return {"loss_curve": state["losses"], "elapsed_training_seconds": state["elapsed"],
                    "training_schedule_sha256": state["schedule_hash"].hexdigest()}
        trace = trace.tb_next
    raise RuntimeError("Parity failure did not originate from the locked training function")


def recover_receipt(run, fold, parity, amendment_sha, telemetry=None):
    inputs = trial.arrays(run / "inputs" / f"fold-{fold}.npz")
    folder = run / "plain" / f"fold-{fold}"
    schedule = schedule_digest(inputs["train_rows"])
    if telemetry and telemetry["training_schedule_sha256"] != schedule:
        raise RuntimeError("Observed minibatch schedule differs from locked recipe")
    result = {"status": "MOTION_NULL_FIT_COMPLETE_OUTCOMES_EMBARGOED", "arm": "plain", "fold": fold,
              "seed": 42, "updates": 256, "active_parameters": 120129,
              "execution_lock_sha256": trial.file_sha256(run / "execution_lock.json"),
              "numerical_exception_sha256": amendment_sha,
              "receipt_recovered_without_training": True,
              "training_schedule_sha256": schedule,
              "training_schedule_provenance": "observed and independently reconstructed" if telemetry else "reconstructed from locked recipe; observed trace unavailable",
              "training_label_rows": len(inputs["train_rows"]),
              "fit_ids_sha256": str(inputs["fit_ids_sha256"].item()),
              "held_ids_sha256": str(inputs["held_ids_sha256"].item()),
              "held_labels_passed_to_optimizer": False, "new_held_scores_computed": False,
              "historical_plain_parity": parity,
              "loss_curve": telemetry["loss_curve"] if telemetry else None,
              "elapsed_training_seconds": telemetry["elapsed_training_seconds"] if telemetry else None,
              "telemetry_note": "Captured from the original exception frame; no optimizer rerun" if telemetry else "Original stopped process did not persist telemetry; no rerun to recover it",
              "checkpoint_sha256": trial.file_sha256(folder / "checkpoint.pt"),
              "predictions_sha256": trial.file_sha256(folder / "predictions.npz")}
    trial.write(folder / "receipt.json", result)


def complete():
    run = trial.RUN
    lock = trial.validate(run)
    if not torch.cuda.is_available() or trial.read(run / "preflight.json")["status"] != "MOTION_NULL_PREFLIGHT_PASS":
        raise RuntimeError("Original preflight and CUDA required")
    amendment_path = run / "numerical_parity_exception_v2.json"
    if amendment_path.exists() or (run / "queue_receipt.json").exists():
        raise RuntimeError("One-shot completion only; no automatic retries")
    existing = {("plain", 0), ("paired_null", 0), ("plain", 1)}
    for fold in range(5):
        for arm in trial.ARMS:
            if (run / arm / f"fold-{fold}").exists() != ((arm, fold) in existing):
                raise RuntimeError("Unexpected existing fits; refuse selection")
    prior = trial.read(run / "numerical_parity_exception.json")
    for name, digest in prior["first_control_hashes"].items():
        if trial.file_sha256(run / "plain/fold-0" / name) != digest:
            raise RuntimeError("First control changed")
    old_adapter = ROOT / "experiments/resume_okutama_motion_null_contrast.py"
    if trial.file_sha256(old_adapter) != prior["resume_source_sha256"]:
        raise RuntimeError("Prior resumption code changed")
    preexisting = [trial.record(p) for arm, fold in sorted(existing)
                   for p in sorted((run / arm / f"fold-{fold}").iterdir())]
    cohort = trial.arrays(run / "inputs/cohort.npz")
    controls = [check_control(run, f, cohort, lock) for f in (0, 1)]
    stop = trial.read(run / "RESUME_PARITY_STOP.json")
    if controls[1]["max_abs_delta"] != stop["observed_delta_max_abs"]:
        raise RuntimeError("Second control differs from the recorded technical stop")
    amendment = {"status": "USER_APPROVED_ALL_CONTROLS_NUMERICAL_ONLY_EXCEPTION",
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "user_authorization": "yes",
                 "approved_question": "Extend the numerical-only exception to subsequent controls provided class predictions and input/provenance checks match, keeping every performance/safety gate unchanged and never retrying fits",
                 "scope": "Accept numerical residual differences in first-attempt plain controls only; never waive class predictions, input provenance, bounds, replay, or performance gates",
                 "execution_lock_sha256": trial.file_sha256(run / "execution_lock.json"),
                 "protocol_sha256": trial.file_sha256(trial.PROTOCOL),
                 "previous_amendment_sha256": trial.file_sha256(run / "numerical_parity_exception.json"),
                 "resume_stop_sha256": trial.file_sha256(run / "RESUME_PARITY_STOP.json"),
                 "completion_source_sha256": trial.file_sha256(Path(__file__)),
                 "guard_tests_sha256": trial.file_sha256(ROOT / "tests/test_motion_null_completion.py"),
                 "preexisting_artifacts": preexisting, "preexisting_control_checks": controls,
                 "performance_and_safety_gates_changed": False, "new_held_scores_computed": False,
                 "new_fits_authorized": 7, "fits_retrained_or_selected": 0,
                 "preserved_files_verified": trial.check_preservation()}
    trial.write(amendment_path, amendment)
    amendment_sha = trial.file_sha256(amendment_path)
    recover_receipt(run, 1, controls[1], amendment_sha)
    cache = trial.load_native_motion_cache(trial.legacy.CACHE, cohort["sample_ids"])
    geometry = trial.motion_geometry(cache.geometry, cache.camera)
    started = time.perf_counter()
    completed = 0
    print(json.dumps({"event": "seven_fit_completion_started", "gpu": torch.cuda.get_device_name(),
                      "amendment_sha256": amendment_sha}), flush=True)
    for fold in range(5):
        for arm in trial.ARMS:
            if (arm, fold) in existing:
                continue
            try:
                trial.fit_one(run, arm, fold, cohort, cache, geometry, "cuda", started)
            except RuntimeError as error:
                if arm != "plain" or not str(error).startswith("Historical plain recipe parity failed: delta="):
                    raise
                parity = check_control(run, fold, cohort, lock)
                recover_receipt(run, fold, parity, amendment_sha, exception_telemetry(error))
                print(json.dumps({"event": "control_complete_numeric_exception", **parity}), flush=True)
            torch.cuda.empty_cache()
            completed += 1
    trial.validate(run)
    for item in preexisting:
        trial.verify_record(item)
    controls = [check_control(run, f, cohort, lock) for f in range(5)]
    result = {"status": "MOTION_NULL_TEN_FITS_COMPLETE", "head_fits": 10, "router_fits": 0,
              "preexisting_fits_retained": 3, "new_fits_this_completion": completed,
              "fits_retrained_or_selected": 0, "new_held_scores_computed": False,
              "numerical_exception_sha256": amendment_sha, "historical_control_checks": controls,
              "elapsed_completion_queue_seconds": time.perf_counter() - started,
              "preexisting_artifact_hashes_preserved": True,
              "performance_and_safety_gates_changed": False,
              "preserved_files_verified": trial.check_preservation()}
    trial.write(run / "queue_receipt.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(complete(), indent=2), flush=True)
