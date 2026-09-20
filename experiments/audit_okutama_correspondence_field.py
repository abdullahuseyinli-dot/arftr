"""Independently replay Stage-1 field checkpoints and label-blind structure tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_correspondence_field as runner
import torch

from hac.actor_memory_base import file_sha256
from hac.correspondence_field_reader import ARMS
from hac.source_swap_data import immutable_json

ROOT = Path(__file__).resolve().parents[1]


def _checked_probability(values: np.ndarray, expected_rows: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if (
        values.shape != (expected_rows,)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise RuntimeError("Audited motion probability is malformed or nonfinite")
    return values


def _permuted_points(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {name: value.copy() for name, value in batch.items()}
    rows, pairs, slots = result["point_valid"].shape
    for row in range(rows):
        for pair in range(pairs):
            shift = 1 + (row * pairs + pair) % slots
            order = np.roll(np.arange(slots), shift)
            result["points"][row, pair] = result["points"][row, pair, order]
            result["point_valid"][row, pair] = result["point_valid"][row, pair, order]
    return result


def _spatial_reassignment(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {name: value.copy() for name, value in batch.items()}
    for row in range(len(result["points"])):
        for pair in range(result["points"].shape[1]):
            valid = np.flatnonzero(result["point_valid"][row, pair])
            if len(valid) > 1:
                result["points"][row, pair, valid, :2] = result["points"][
                    row, pair, valid[::-1], :2
                ]
    return result


def _pair_reassignment(batch: dict[str, np.ndarray], *, keep_times: bool) -> dict[str, np.ndarray]:
    result = {name: value.copy() for name, value in batch.items()}
    order = np.roll(np.arange(15), 3)
    for name in ("points", "point_valid", "pair_features", "pair_valid"):
        result[name] = result[name][:, order]
    if not keep_times:
        result["midpoint_times"] = result["midpoint_times"][:, order]
    return result


def _predict(
    model: torch.nn.Module,
    store: runner.FieldStore,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    probability, available = [], []
    model.eval()
    with torch.inference_mode():
        for left in range(0, len(rows), batch_size):
            output = model(**runner.to_device(store.batch(rows[left : left + batch_size]), device))
            probability.append(output["motion_probability"].cpu().numpy())
            available.append(output["observation_available"].cpu().numpy())
    probability = _checked_probability(np.concatenate(probability), len(rows))
    available = np.concatenate(available).astype(bool)
    if available.shape != (len(rows),):
        raise RuntimeError("Audited observation availability is malformed")
    return probability, available


def _predict_transformed(
    model: torch.nn.Module,
    store: runner.FieldStore,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
    transform,
) -> np.ndarray:
    probability = []
    model.eval()
    with torch.inference_mode():
        for left in range(0, len(rows), batch_size):
            batch = transform(store.batch(rows[left : left + batch_size]))
            output = model(**runner.to_device(batch, device))
            probability.append(output["motion_probability"].cpu().numpy())
    return _checked_probability(np.concatenate(probability), len(rows))


def _load_checkpoint(
    path: Path, protocol: dict[str, Any], arm: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    saved = torch.load(path, map_location=device, weights_only=True)
    if not all(
        torch.is_tensor(value) and bool(torch.isfinite(value).all())
        for value in saved["model"].values()
    ):
        raise RuntimeError("Checkpoint contains a nonfinite or malformed model state")
    model = runner.model_for(protocol, arm).to(device)
    model.load_state_dict(saved["model"], strict=True)
    if saved["capacity_report"] != model.capacity_report:
        raise RuntimeError("Checkpoint capacity report changed")
    return model, saved["request"]


def audit(run: Path, phase: str) -> dict[str, Any]:
    protocol = runner.read_json(runner.PROTOCOL)
    runner.validate_protocol(protocol)
    store = runner.FieldStore(ROOT / protocol["inputs"]["field_cache"])
    lock_hash = runner.prepare_lock(run, protocol, store.sample_ids)
    runner.validate_lock(run, protocol, store.sample_ids, lock_hash)
    seeds = runner.INITIAL_SEEDS if phase == "initial" else runner.CONFIRMATION_SEEDS
    output = run / "audit" / phase
    summary_path = output / "summary.json"
    diagnostic_path = output / "diagnostic_probabilities.npz"
    if summary_path.is_file():
        result = runner.read_json(summary_path)
        current_receipts = runner.validated_fit_receipt_hashes(run, seeds)
        if (
            result.get("execution_lock_sha256") != lock_hash
            or result.get("audit_code_sha256") != file_sha256(Path(__file__))
            or result.get("fit_receipt_sha256") != current_receipts
            or result.get("diagnostic_probabilities_sha256") != file_sha256(diagnostic_path)
        ):
            raise RuntimeError("Existing independent audit or its ancestors changed")
        return result
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Partial field audit exists; use a fresh versioned run")
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint replay requires the locked CUDA endpoint")
    device = torch.device("cuda")
    exact_replays = 0
    maximum_replay_difference = 0.0
    invariance = {arm: {"point_order_max_abs": 0.0} for arm in ARMS}
    invariance["c1"]["pair_order_max_abs"] = 0.0
    invariance["c2"].update(
        {
            "joint_pair_time_order_max_abs": 0.0,
            "spatial_reassignment_mean_abs": [],
            "timestamp_reassignment_mean_abs": [],
        }
    )
    diagnostic_q = np.full((2, len(seeds), len(store.sample_ids)), np.nan, dtype=np.float64)
    receipt_hashes = {}
    for seed in seeds:
        for arm in ARMS:
            for fold in range(5):
                prediction_path, checkpoint_path, receipt_path = runner.fit_paths(
                    run, arm, fold, seed
                )
                if not all(
                    path.is_file() for path in (prediction_path, checkpoint_path, receipt_path)
                ):
                    raise RuntimeError("Independent audit requires the complete fit inventory")
                receipt = runner.read_json(receipt_path)
                if (
                    receipt.get("status")
                    != "CORRESPONDENCE_FIELD_FIT_COMPLETE_OUTER_METRICS_EMBARGOED"
                    or receipt.get("outer_held_labels_read") != 0
                    or receipt.get("outer_metrics_computed") != 0
                    or receipt.get("predictions_sha256") != file_sha256(prediction_path)
                    or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
                ):
                    raise RuntimeError("Fit receipt failed independent audit")
                with np.load(prediction_path, allow_pickle=False) as saved_prediction:
                    rows = saved_prediction["held_rows"]
                    expected_probability = saved_prediction["motion_probability"]
                    expected_available = saved_prediction["observation_available"]
                    expected_ids = saved_prediction["sample_ids"]
                if (
                    not np.array_equal(store.sample_ids[rows], expected_ids)
                    or canonical_rows(rows, fold, protocol) is False
                ):
                    raise RuntimeError("Prediction row identity or fold assignment changed")
                model, checkpoint_request = _load_checkpoint(checkpoint_path, protocol, arm, device)
                if (
                    checkpoint_request != receipt["request"]
                    or receipt["request"].get("held_labels_sha256") is not None
                ):
                    raise RuntimeError("Checkpoint/request label embargo changed")
                actual_probability, actual_available = _predict(
                    model, store, rows, device, protocol["training"]["batch_size"]
                )
                difference = float(np.max(np.abs(actual_probability - expected_probability)))
                maximum_replay_difference = max(maximum_replay_difference, difference)
                if difference > 1e-7 or not np.array_equal(actual_available, expected_available):
                    raise RuntimeError("Checkpoint prediction replay differs from saved output")
                exact_replays += int(np.array_equal(actual_probability, expected_probability))
                diagnostic_rows = rows[: min(64, len(rows))]
                base = store.batch(diagnostic_rows)
                with torch.inference_mode():
                    reference = (
                        model(**runner.to_device(base, device))["motion_probability"].cpu().numpy()
                    )
                    permuted = (
                        model(**runner.to_device(_permuted_points(base), device))[
                            "motion_probability"
                        ]
                        .cpu()
                        .numpy()
                    )
                reference = _checked_probability(reference, len(diagnostic_rows))
                permuted = _checked_probability(permuted, len(diagnostic_rows))
                point_difference = float(np.max(np.abs(reference - permuted)))
                invariance[arm]["point_order_max_abs"] = max(
                    invariance[arm]["point_order_max_abs"], point_difference
                )
                if point_difference > 1e-6:
                    raise RuntimeError("Reader changed under a point-order permutation")
                if arm == "c1":
                    with torch.inference_mode():
                        reordered = (
                            model(
                                **runner.to_device(
                                    _pair_reassignment(base, keep_times=True), device
                                )
                            )["motion_probability"]
                            .cpu()
                            .numpy()
                        )
                    reordered = _checked_probability(reordered, len(diagnostic_rows))
                    delta = float(np.max(np.abs(reference - reordered)))
                    invariance[arm]["pair_order_max_abs"] = max(
                        invariance[arm]["pair_order_max_abs"], delta
                    )
                    if delta > 1e-6:
                        raise RuntimeError("C1 changed under an unordered pair permutation")
                if arm == "c2":
                    with torch.inference_mode():
                        joint = (
                            model(
                                **runner.to_device(
                                    _pair_reassignment(base, keep_times=False), device
                                )
                            )["motion_probability"]
                            .cpu()
                            .numpy()
                        )
                        spatial = (
                            model(**runner.to_device(_spatial_reassignment(base), device))[
                                "motion_probability"
                            ]
                            .cpu()
                            .numpy()
                        )
                        temporal = (
                            model(
                                **runner.to_device(
                                    _pair_reassignment(base, keep_times=True), device
                                )
                            )["motion_probability"]
                            .cpu()
                            .numpy()
                        )
                    joint = _checked_probability(joint, len(diagnostic_rows))
                    spatial = _checked_probability(spatial, len(diagnostic_rows))
                    temporal = _checked_probability(temporal, len(diagnostic_rows))
                    joint_delta = float(np.max(np.abs(reference - joint)))
                    invariance[arm]["joint_pair_time_order_max_abs"] = max(
                        invariance[arm]["joint_pair_time_order_max_abs"], joint_delta
                    )
                    if joint_delta > 1e-6:
                        raise RuntimeError("C2 changed under a joint pair/time token permutation")
                    invariance[arm]["spatial_reassignment_mean_abs"].append(
                        float(np.mean(np.abs(reference - spatial)))
                    )
                    invariance[arm]["timestamp_reassignment_mean_abs"].append(
                        float(np.mean(np.abs(reference - temporal)))
                    )
                    diagnostic_q[0, seeds.index(seed), rows] = _predict_transformed(
                        model,
                        store,
                        rows,
                        device,
                        protocol["training"]["batch_size"],
                        _spatial_reassignment,
                    )
                    diagnostic_q[1, seeds.index(seed), rows] = _predict_transformed(
                        model,
                        store,
                        rows,
                        device,
                        protocol["training"]["batch_size"],
                        lambda batch: _pair_reassignment(batch, keep_times=True),
                    )
                receipt_hashes[f"{arm}/{fold}/{seed}"] = file_sha256(receipt_path)
                del model
                torch.cuda.empty_cache()
    preflight = runner.read_json(run / "preflight.json")
    gradients = preflight.get("c2_full_window_gradient_by_pair")
    if (
        not isinstance(gradients, list)
        or len(gradients) != 15
        or not all(value > 0 for value in gradients)
    ):
        raise RuntimeError("Pre-fit full-window gradient receipt is absent")
    invariance["c2"]["spatial_reassignment_mean_abs"] = float(
        np.mean(invariance["c2"]["spatial_reassignment_mean_abs"])
    )
    invariance["c2"]["timestamp_reassignment_mean_abs"] = float(
        np.mean(invariance["c2"]["timestamp_reassignment_mean_abs"])
    )
    if not np.isfinite(diagnostic_q).all():
        raise RuntimeError("Fixed structure-diagnostic prediction coverage is incomplete")
    output.mkdir(parents=True)
    with diagnostic_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=store.sample_ids,
            seeds=np.asarray(seeds),
            diagnostics=np.asarray(["spatial_reassignment", "timestamp_reassignment"]),
            motion_probabilities=diagnostic_q,
        )
    result = {
        "status": "CORRESPONDENCE_FIELD_INDEPENDENT_AUDIT_PASSED",
        "execution_lock_sha256": lock_hash,
        "phase": phase,
        "seeds": list(seeds),
        "fits_audited": len(seeds) * len(ARMS) * 5,
        "bit_exact_checkpoint_replays": exact_replays,
        "maximum_checkpoint_replay_abs_difference": maximum_replay_difference,
        "fit_receipt_sha256": receipt_hashes,
        "diagnostic_probabilities_sha256": file_sha256(diagnostic_path),
        "fixed_structure_diagnostics": invariance,
        "c2_prefit_full_window_gradient_by_pair": gradients,
        "annotation_fields_used_for_inference": [],
        "outer_metrics_computed": 0,
        "audit_code_sha256": file_sha256(Path(__file__)),
    }
    immutable_json(summary_path, result)
    return result


def canonical_rows(rows: np.ndarray, fold: int, protocol: dict[str, Any]) -> bool:
    """Verify held rows from fold metadata without reading task labels."""

    memory = ROOT / protocol["inputs"]["video_context"]
    with np.load(memory, allow_pickle=False) as saved:
        expected = np.flatnonzero(saved["folds"] == fold)
    return np.array_equal(rows, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("initial", "confirmation"), default="initial")
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    print(json.dumps(audit(run, args.phase), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
