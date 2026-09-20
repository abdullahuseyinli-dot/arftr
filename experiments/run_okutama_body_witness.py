"""Plan and bounded-preflight the frozen body-witness screen.

The real 20-fit screen remains unavailable until the human/pose measurement and
target-sensitivity gates pass.  ``preflight`` performs exactly twenty synthetic
optimizer updates per arm to test shapes, gradients, capacity and runtime; its
labels are generated constants and are not experimental outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("Body-witness preflight requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.body_witness import (  # noqa: E402
    ConditionalWitnessDecoder,
    ModalityPostureReader,
    body_witness_training_loss,
    energy_evidence,
    normalized_predictive_energy,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("okutama_body_witness_protocol.json")
DEFAULT_RUN = ROOT / ".runs/research_20260913/body_witness_screen_v6"
ARMS = ("V0", "V1", "V2", "V3")
MODALITY_DIMS = {
    "context": 2304,
    # Identical six-slot RGB bank for V1/V2/V3: the two unmasked extent
    # encodings and the four extent-by-visible-mask encodings.  The V3
    # decoder consumes the latter four, but they also reach every final
    # reader directly so V3-V2 cannot be explained by extra frozen features.
    "body": 6 * 768,
    # Eight common crop-quality scalars plus the exact 6D crop transform and
    # 1D acquisition-valid bit for each of the four masked observations.
    # These are direct inputs to every arm from V1 onward because V3's decoder
    # receives them too.
    "quality": 8 + 4 * 6 + 4,
    "pose": 2 * 8 * 193 + 16,
    "energy": 4 * 3,
}
ARM_MODALITIES = {
    "V0": ("context",),
    "V1": ("context", "body", "quality"),
    "V2": ("context", "body", "quality", "pose"),
    "V3": ("context", "body", "quality", "pose", "energy"),
}
ARM_EXTRA_BLOCKS = {"V0": 0, "V1": 5, "V2": 2, "V3": 0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training = value["training"]
    if (
        value.get("study_id") != "okutama_body_witness_v1"
        or tuple(training.get("initial_seeds", ())) != (42,)
        or training.get("updates") != 800
        or training.get("batch_size") != 128
        or training.get("learning_rate") != 3e-4
        or training.get("weight_decay") != 1e-4
        or training.get("gradient_norm_cap") != 1.0
        or training.get("held_label_early_stopping") is not False
    ):
        raise RuntimeError("Body-witness training protocol differs from its locked screen")
    architecture = value["reader_architecture"]
    expected_blocks = {
        "V1": architecture["V1_active_direct_residual_blocks"],
        "V2": architecture["V2_active_direct_residual_blocks"],
        "V3": architecture["V3_active_direct_residual_blocks"],
    }
    if architecture["modality_dims"] != MODALITY_DIMS or expected_blocks != {
        arm: ARM_EXTRA_BLOCKS[arm] for arm in ("V1", "V2", "V3")
    }:
        raise RuntimeError("Implemented reader architecture differs from the prospective contract")
    receipt = architecture_receipt()
    matched_counts = {
        arm: receipt["active_trainable_parameters"][arm] for arm in ("V1", "V2", "V3")
    }
    if (
        matched_counts != architecture["frozen_expected_active_counts"]
        or not np.isclose(
            receipt["V1_V2_V3_active_relative_parameter_range"],
            architecture["frozen_expected_relative_range"],
            atol=1e-15,
            rtol=0,
        )
    ):
        raise RuntimeError("Active capacity receipt differs from the prospective contract")
    return value


def make_reader(arm: str, *, dropout: float) -> ModalityPostureReader:
    if arm not in ARMS:
        raise ValueError("Unknown body-witness arm")
    dims = {name: MODALITY_DIMS[name] for name in ARM_MODALITIES[arm]}
    return ModalityPostureReader(
        dims,
        extra_residual_blocks=ARM_EXTRA_BLOCKS[arm],
        dropout=dropout,
        parameter_limit=1_000_000,
    )


def architecture_receipt() -> dict[str, Any]:
    readers = {arm: make_reader(arm, dropout=0.1) for arm in ARMS}
    reader_counts = {arm: readers[arm].trainable_parameters for arm in ARMS}
    decoder = ConditionalWitnessDecoder(768, dropout=0.1)
    decoder_counts = {arm: decoder.trainable_parameters if arm == "V3" else 0 for arm in ARMS}
    active_counts = {arm: reader_counts[arm] + decoder_counts[arm] for arm in ARMS}
    matched = np.asarray([active_counts[arm] for arm in ("V1", "V2", "V3")], np.float64)
    relative_range = float(matched.max() / matched.min() - 1)
    if relative_range > 0.10 or max(active_counts.values()) >= 1_000_000:
        raise RuntimeError("Proposed arms violate full active-capacity gates")
    return {
        "modality_dimensions": MODALITY_DIMS,
        "arm_modalities": {key: list(value) for key, value in ARM_MODALITIES.items()},
        "useful_expansion_residual_blocks": ARM_EXTRA_BLOCKS,
        "reader_parameters": reader_counts,
        "decoder_parameters": decoder_counts,
        "active_trainable_parameters": active_counts,
        "V1_V2_V3_active_relative_parameter_range": relative_range,
        "parameter_limit_strict": 1_000_000,
        "capacity_includes_every_task_trained_module": True,
        "identical_direct_RGB_bank_V1_V2_V3": True,
        "note": "V3's task-trained conditional decoder is included in its active count",
    }


def _gate_receipt(path: Path, passing_statuses: set[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"receipt_exists": False, "status": None, "passed": False}
    value = json.loads(path.read_text(encoding="utf-8"))
    status = value.get("status")
    return {
        "receipt_exists": True,
        "status": status,
        "passed": status in passing_statuses,
        "sha256": sha256_file(path),
    }


def plan(run: Path) -> dict[str, Any]:
    value = protocol()
    run.mkdir(parents=True, exist_ok=True)
    pilot = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
    dependency = ROOT / ".runs/research_20260913/body_witness_nested_plan_v3"
    gates = {
        "native_artifact_revalidation": _gate_receipt(
            pilot / "pilot_artifact_revalidation_v6.json",
            {"PASS_NATIVE_ARTIFACTS_REUSABLE_POSE_AND_TASK_GATES_BLOCKED"},
        ),
        "pose_computational_parity": _gate_receipt(
            ROOT
            / ".runs/research_20260913/vitpose_parity_v1/vitpose_computational_parity_receipt.json",
            {
                "PASS_COMPUTATIONAL_MISMATCH_CORRECTED_OFFICIAL_BYTE_IDENTITY_UNRESOLVED"
            },
        ),
        # A random-weight runtime smoke explicitly does not satisfy this gate.
        "pretrained_pose_runtime_and_model_lock": _gate_receipt(
            pilot / "pose_checkpoint_receipt.json", {"PASS"}
        ),
        "availability": _gate_receipt(pilot / "availability_receipt.json", {"PASS"}),
        "target_sensitivity_for_V3": _gate_receipt(
            pilot / "target_sensitivity_receipt.json", {"PASS"}
        ),
        "measurement_cache": _gate_receipt(
            pilot / "body_witness_cache_receipt.json", {"PASS"}
        ),
        "nested_dependency_feasibility": _gate_receipt(
            dependency / "summary.json", {"PASS"}
        ),
    }
    result = {
        "status": "SCREEN_PREPARED_TASK_FITS_GATED",
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(PROTOCOL)},
        "architecture": architecture_receipt(),
        "required_gates": gates,
        "historical_evidence_not_a_current_gate": {
            "pilot_plan": _gate_receipt(
                pilot / "execution_plan.json", {"OBSERVATION_PILOT_PLANNED"}
            ),
            "native_extraction": _gate_receipt(
                pilot / "extraction_receipt.json", {"NATIVE_REVIEW_CROPS_COMPLETE"}
            ),
            "random_weight_pose_runtime": _gate_receipt(
                pilot / "pose_runtime_preflight.json",
                {"RUNTIME_SMOKE_PASS_PRETRAINED_CHECKPOINT_BLOCKED"},
            ),
        },
        "initial_fit_count_if_V3_allowed": 20,
        "initial_fit_count_if_V3_sensitivity_fails": 15,
        "full_fit_launched": False,
        "task_labels_read": 0,
        "fit_command_after_gates": (
            f'& "{ROOT / ".venv/Scripts/python.exe"}" experiments/run_okutama_body_witness.py '
            f'--stage fit --run "{run}"'
        ),
        "stopping_rule": value["compute"]["above_boundary"],
    }
    write_new_json(run / "screen_execution_plan.json", result)
    return result


def _synthetic_modalities(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260913)
    return {
        name: torch.randn(batch, dimension, generator=generator, device=device)
        for name, dimension in MODALITY_DIMS.items()
        if name != "energy"
    }


def preflight(run: Path) -> dict[str, Any]:
    """Twenty synthetic updates per arm; no empirical action data are loaded."""

    value = protocol()
    if not (run / "screen_execution_plan.json").is_file():
        raise RuntimeError("Run the plan stage before synthetic preflight")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = int(value["pilot"]["synthetic_reader_updates"] and value["training"]["batch_size"])
    updates = int(value["pilot"]["synthetic_reader_updates"])
    labels = (torch.arange(batch, device=device) % 2).long()
    weights = torch.ones(batch, device=device)
    base_modalities = _synthetic_modalities(batch, device)
    observed = torch.rand(batch, 4, 8, 193, device=device)
    observed /= observed.sum(-1, keepdim=True)
    landmark_weights = torch.ones(batch, 4, 8, device=device)
    visible = torch.randn(batch, 4, 768, device=device)
    transforms = torch.randn(batch, 4, 6, device=device)
    validity = torch.ones(batch, 4, 1, device=device)
    mask_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    outputs: list[dict[str, Any]] = []
    for arm in ARMS:
        torch.manual_seed(42)
        reader = make_reader(arm, dropout=float(value["training"]["dropout"])).to(device).train()
        decoder = (
            ConditionalWitnessDecoder(768, dropout=float(value["training"]["dropout"])).to(device).train()
            if arm == "V3"
            else None
        )
        parameters = list(reader.parameters()) + (list(decoder.parameters()) if decoder else [])
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(value["training"]["learning_rate"]),
            weight_decay=float(value["training"]["weight_decay"]),
        )
        start = time.perf_counter()
        losses, maximum_gradient = [], 0.0
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            modalities = {
                name: base_modalities[name] for name in ARM_MODALITIES[arm] if name != "energy"
            }
            if decoder is not None:
                predicted = decoder(visible, mask_ids, transforms, validity)
                energy, available = normalized_predictive_energy(
                    observed, predicted, landmark_weights
                )
                modalities["energy"] = energy_evidence(energy)
            logits = reader(modalities)
            if decoder is None:
                target = (labels == 0).to(logits.dtype)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            else:
                loss = body_witness_training_loss(
                    logits, labels, weights, energy, available
                )["loss"]
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                parameters, float(value["training"]["gradient_norm_cap"])
            )
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise RuntimeError(f"Synthetic {arm} preflight produced nonfinite optimization state")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            maximum_gradient = max(maximum_gradient, float(gradient.detach().cpu()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        outputs.append(
            {
                "arm": arm,
                "updates": updates,
                "batch_size": batch,
                "reader_parameters": reader.trainable_parameters,
                "decoder_parameters": decoder.trainable_parameters if decoder else 0,
                "active_trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "all_losses_finite": bool(np.isfinite(losses).all()),
                "maximum_preclip_gradient_norm": maximum_gradient,
                "elapsed_seconds": elapsed,
                "seconds_per_update": elapsed / updates,
                "optimizer_steps_on_task_data": 0,
            }
        )
        del reader, decoder, optimizer, parameters, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "status": "SYNTHETIC_20_UPDATE_PREFLIGHT_PASS",
        "claim_scope": "shape/gradient/resource startup only; not performance, convergence or pose quality",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "architecture": architecture_receipt(),
        "arms": outputs,
        "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in outputs),
        "task_rows_read": 0,
        "task_labels_read": 0,
        "task_optimizer_updates": 0,
        "full_fit_launched": False,
    }
    write_new_json(run / "synthetic_preflight_receipt.json", result)
    return result


def _passing_receipt(path: Path, expected_status: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required gate receipt is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != expected_status:
        raise RuntimeError(f"Required gate did not pass: {path}")
    return value


def fit(run: Path) -> dict[str, Any]:
    """Fail closed at the handoff boundary until observation gates exist."""

    protocol()
    pilot = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
    _passing_receipt(pilot / "availability_receipt.json", "PASS")
    sensitivity = json.loads((pilot / "target_sensitivity_receipt.json").read_text())
    if sensitivity.get("status") not in {"PASS", "FAIL"}:
        raise RuntimeError("Target-sensitivity receipt is missing a locked decision")
    _passing_receipt(pilot / "body_witness_cache_receipt.json", "PASS")
    raise RuntimeError(
        "Observation gates passed, but full fit execution remains a >20-minute handoff: "
        "freeze the measured runtime projection and invoke per-fold bounded jobs explicitly"
    )


def summarize(run: Path) -> dict[str, Any]:
    if not (run / "results/embargo_release_receipt.json").is_file():
        raise RuntimeError("No jointly released five-fold results exist; held outputs remain unavailable")
    raise RuntimeError("Independent replay must pass before any body-witness result summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "preflight", "fit", "summarize"), required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = globals()[args.stage](args.run.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
