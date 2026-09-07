"""Replay frozen CPTR checkpoints under the original, F1, F2, and F3 equations.

This runner accepts only the role-safe bundle produced by the companion R0 builder and
requires a source lock created before execution.  It has no mixed-manifest, calibration,
confirmation, or test input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hac.cptr import PartTrajectoryResidualNetwork, probability_diagnostics
from hac.cptr_training import build_cptr_model, classification_summary
from hac.vcoco_v3_neural import decode_factorized_logits
from hac.vcoco_v3_temporal import StaticIdentifiabilityStudent, TemporalFactorizedTeacher

LOCK_STATUS = "OKUTAMA_CPTR_FROZEN_REPLAY_LOCKED_BEFORE_EXECUTION"
RUN_STATUS = "OKUTAMA_CPTR_FROZEN_INTERVENTION_REPLAY_COMPLETE"
SUMMARY_STATUS = "OKUTAMA_CPTR_FROZEN_INTERVENTION_AGGREGATE_COMPLETE"
INVENTORY_STATUS = "OKUTAMA_CPTR_REPLAY_INPUT_INVENTORY_COMPLETE"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_FOLDS = tuple(range(5))
MODES = ("original", "f1", "f2", "f3")
SHORT_STEPS = 8
CLASS_COUNT = 3
CLASS_NAMES = ("sitting", "standing", "walking_running")
REPRODUCTION_MAX_ABS_TOLERANCE = 5e-4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260906


@dataclass(frozen=True)
class ReplayInputs:
    inventory: Path
    eligible_index: Path
    feature_bundle: Path
    bundle_summary: Path
    protocol_lock: Path


@dataclass(frozen=True)
class BatchResult:
    probabilities: dict[str, torch.Tensor]
    posture_logits: dict[str, torch.Tensor]
    motion_logits: dict[str, torch.Tensor]
    static_posture_logits: torch.Tensor
    static_motion_logits: torch.Tensor
    legacy_posture_logits: torch.Tensor
    legacy_motion_logits: torch.Tensor
    learned_posture_delta: torch.Tensor
    learned_motion_delta: torch.Tensor
    posture_legacy_gate: torch.Tensor
    motion_legacy_gate: torch.Tensor
    posture_gates: torch.Tensor
    motion_gates: torch.Tensor
    expert_reliability: torch.Tensor
    q: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0/input_inventory.json"),
    )
    parser.add_argument(
        "--eligible-index",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0/eligible_index.csv"),
    )
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0/eligible_feature_bundle.npz"),
    )
    parser.add_argument(
        "--bundle-summary",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0/bundle_summary.json"),
    )
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(EXPECTED_FOLDS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def collect_environment() -> dict[str, Any]:
    distributions = sorted(
        (
            {
                "name": distribution.metadata.get("Name", "").lower().replace("_", "-"),
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=lambda item: (item["name"], item["version"]),
    )
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": distributions,
    }
    cuda_available = bool(torch.cuda.is_available())
    environment["torch"] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else None,
        "compute_capability": (
            ".".join(map(str, torch.cuda.get_device_capability(0)))
            if cuda_available
            else None
        ),
    }
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    environment["canonical_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return environment


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def validate_protocol_lock(
    inputs: ReplayInputs,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory = load_json(inputs.inventory)
    lock = load_json(inputs.protocol_lock)
    if (
        inventory.get("status") != INVENTORY_STATUS
        or int(inventory.get("eligible_rows", -1)) != 6360
        or int(inventory.get("eligible_scenarios", -1)) != 14
    ):
        raise RuntimeError("The role-safe replay inventory is incomplete")
    if lock.get("status") != LOCK_STATUS or lock.get("dataset") != "okutama":
        raise RuntimeError("The Okutama replay source lock is absent or has the wrong status")
    source_hashes = lock.get("source_sha256", {})
    repository_root = Path(__file__).resolve().parents[1]
    if (
        git_value(repository_root, "rev-parse", "HEAD") != lock.get("repository_commit")
        or git_value(repository_root, "rev-parse", "HEAD^{tree}")
        != lock.get("repository_tree")
    ):
        raise RuntimeError("Current repository HEAD/tree differs from the replay lock")
    if git_value(
        repository_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise RuntimeError("Working-tree changes invalidate frozen replay execution")
    if lock.get("environment") != collect_environment():
        raise RuntimeError("Current environment differs from the frozen replay lock")
    required = {
        "input_inventory": inputs.inventory,
        "eligible_index": inputs.eligible_index,
        "eligible_feature_bundle": inputs.feature_bundle,
        "bundle_summary": inputs.bundle_summary,
        "bundle_builder": repository_root / "experiments/build_okutama_cptr_replay_bundle.py",
        "replay_runner": Path(__file__),
        "cptr_model_module": repository_root / "src/hac/cptr.py",
        "cptr_feature_module": repository_root / "src/hac/cptr_features.py",
        "cptr_training_module": repository_root / "src/hac/cptr_training.py",
        "temporal_model_module": repository_root / "src/hac/vcoco_v3_temporal.py",
        "neural_module": repository_root / "src/hac/vcoco_v3_neural.py",
    }
    for name, path in required.items():
        expected = source_hashes.get(name)
        observed = sha256_file(path.resolve())
        if expected != observed:
            raise RuntimeError(f"Replay lock mismatch for {name}: {observed}")
    if any(
        int(lock.get("protected_access", {}).get(name, -1)) != 0
        for name in (
            "mixed_manifest_rows_read",
            "mixed_development_metadata_rows_read",
            "calibration_rows_or_arrays_read",
            "confirmation_rows_or_arrays_read",
            "test_rows_or_arrays_read",
        )
    ):
        raise RuntimeError("Replay lock does not explicitly prohibit every protected input")
    authorization = lock.get("authorization", {})
    if (
        authorization.get("R1a_execution") is not True
        or authorization.get("fitting") is not False
        or int(authorization.get("model_updates", -1)) != 0
    ):
        raise RuntimeError("Execution lock does not authorize frozen R1 replay")
    bundle_summary = load_json(inputs.bundle_summary)
    if (
        bundle_summary.get("status") != "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE"
        or int(bundle_summary.get("rows", -1)) != 6360
        or int(bundle_summary.get("scenarios", -1)) != 14
    ):
        raise RuntimeError("The role-safe feature bundle is incomplete")
    if bundle_summary.get("source_sha256", {}).get("protocol_lock") != lock.get(
        "source_sha256", {}
    ).get("source_lock"):
        raise RuntimeError("The feature bundle belongs to another materialization source lock")
    if bundle_summary.get("artifact_sha256", {}).get(inputs.feature_bundle.name) != sha256_file(
        inputs.feature_bundle
    ):
        raise RuntimeError("The role-safe feature bundle hash changed")
    return inventory, bundle_summary, lock


def build_model(
    *,
    state_dict: dict[str, torch.Tensor],
    candidate: dict[str, Any],
    protocol: dict[str, Any],
    temporal_grid: dict[str, Any],
    input_dim: int,
    part_dim: int,
    device: torch.device,
) -> PartTrajectoryResidualNetwork:
    architecture = temporal_grid["architecture"]
    static = StaticIdentifiabilityStudent(
        input_dim,
        hidden_dim=int(architecture["static_hidden_dim"]),
        dropout=float(architecture["dropout"]),
    )
    teacher = TemporalFactorizedTeacher(
        input_dim,
        model_dim=int(architecture["temporal_model_dim"]),
        layers=int(architecture["temporal_layers"]),
        attention_heads=int(architecture["attention_heads"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]),
        maximum_length=16,
    )
    model = build_cptr_model(
        candidate,
        protocol,
        input_dim=input_dim,
        part_input_dim=part_dim,
        pose_joint_count=1,
        siglip_dim=1,
        static=static,
        teacher=teacher,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval().requires_grad_(False)


def validate_zero_epoch_state(model: PartTrajectoryResidualNetwork) -> None:
    for name, head in model.residual_heads.items():
        for output_name in ("posture", "motion"):
            output = getattr(head, output_name)
            if torch.count_nonzero(output.weight).item() or torch.count_nonzero(output.bias).item():
                raise RuntimeError(f"Zero-epoch residual output is nonzero: {name}.{output_name}")
    for name, gate in (("posture", model.posture_gate), ("motion", model.motion_gate)):
        if torch.count_nonzero(gate.weight).item():
            raise RuntimeError(f"Zero-epoch {name} gate weights are nonzero")
        expected = torch.full_like(gate.bias, -2.0)
        expected[0] = 5.0
        if not torch.equal(gate.bias, expected):
            raise RuntimeError(f"Zero-epoch {name} gate bias changed")


def recover_zero_epoch_q(
    static_posture: torch.Tensor,
    static_motion: torch.Tensor,
    legacy_posture: torch.Tensor,
    legacy_motion: torch.Tensor,
    historical_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the discrete short-window valid fraction under locked zero-epoch gates."""

    q_grid = torch.arange(SHORT_STEPS + 1, device=static_posture.device, dtype=torch.float32)
    q_grid = q_grid / SHORT_STEPS
    gate = torch.sigmoid(torch.tensor(5.0, device=static_posture.device))
    candidates = []
    for q in q_grid:
        posture = static_posture + gate * q * (legacy_posture - static_posture)
        motion = static_motion + gate * q * (legacy_motion - static_motion)
        candidates.append(decode_factorized_logits(posture, motion))
    stack = torch.stack(candidates, dim=1)
    errors = torch.max(torch.abs(stack - historical_probabilities[:, None]), dim=2).values
    best = torch.argmin(errors, dim=1)
    row = torch.arange(len(best), device=best.device)
    return q_grid[best], errors[row, best]


def raw_legacy_gates(
    model: PartTrajectoryResidualNetwork,
    static: Any,
    quality_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    diagnostics = probability_diagnostics(static.probabilities)
    gate_features = model.gate_trunk(
        torch.cat((static.features, diagnostics, quality_features), dim=1)
    )
    posture = model.posture_gate(gate_features).sigmoid()[:, 0]
    motion = model.motion_gate(gate_features).sigmoid()[:, 0]
    return posture, motion


def compute_interventions(
    model: PartTrajectoryResidualNetwork,
    kwargs: dict[str, torch.Tensor | int],
) -> BatchResult:
    captured: dict[str, Any] = {}

    def save_static(_module, _inputs, output):
        captured["static"] = output

    def save_legacy(_module, _inputs, output):
        captured["legacy"] = output

    static_handle = model.static_fallback.register_forward_hook(save_static)
    legacy_handle = model.legacy_temporal.register_forward_hook(save_legacy)
    try:
        original = model(**kwargs)
    finally:
        static_handle.remove()
        legacy_handle.remove()
    static = captured["static"]
    legacy = captured["legacy"]
    q = kwargs["short_valid_mask"].float().mean(dim=1)  # type: ignore[union-attr]
    posture_gate, motion_gate = raw_legacy_gates(
        model,
        static,
        kwargs["quality_features"],  # type: ignore[arg-type]
    )
    learned_posture = original.learned_temporal_residual[:, :2]
    learned_motion = original.learned_temporal_residual[:, 2:]
    legacy_posture_residual = legacy.posture_logits - static.posture_logits
    legacy_motion_residual = legacy.motion_logits - static.motion_logits
    posture_logits = {
        "original": original.posture_logits,
        "f1": static.posture_logits
        + posture_gate[:, None] * legacy_posture_residual
        + learned_posture,
        "f2": legacy.posture_logits + learned_posture,
        "f3": legacy.posture_logits,
    }
    motion_logits = {
        "original": original.motion_logits,
        "f1": static.motion_logits + motion_gate[:, None] * legacy_motion_residual + learned_motion,
        "f2": legacy.motion_logits + learned_motion,
        "f3": legacy.motion_logits,
    }
    probabilities = {
        mode: decode_factorized_logits(posture_logits[mode], motion_logits[mode])
        for mode in MODES[:-1]
    }
    probabilities["f3"] = legacy.probabilities
    if not torch.equal(probabilities["f3"], legacy.probabilities):
        raise RuntimeError("F3 is not a direct teacher return")
    return BatchResult(
        probabilities=probabilities,
        posture_logits=posture_logits,
        motion_logits=motion_logits,
        static_posture_logits=static.posture_logits,
        static_motion_logits=static.motion_logits,
        legacy_posture_logits=legacy.posture_logits,
        legacy_motion_logits=legacy.motion_logits,
        learned_posture_delta=learned_posture,
        learned_motion_delta=learned_motion,
        posture_legacy_gate=posture_gate,
        motion_legacy_gate=motion_gate,
        posture_gates=original.posture_gates,
        motion_gates=original.motion_gates,
        expert_reliability=original.expert_reliability,
        q=q,
    )


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def canonicalize_f3_persistence(
    arrays: dict[str, np.ndarray], retained_teacher: np.ndarray
) -> None:
    """Make the retained F3 probabilities unambiguous in persisted replay output."""

    arrays["f3_current_direct_probabilities"] = arrays["f3_probabilities"].copy()
    arrays["f3_current_direct_posture_logits"] = arrays.pop("f3_posture_logits")
    arrays["f3_current_direct_motion_logits"] = arrays.pop("f3_motion_logits")
    arrays["f3_probabilities"] = retained_teacher.copy()


def load_weights_only(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "request_sha256",
        "fixed_epochs",
        "model_state_dict",
    }:
        raise RuntimeError("Unexpected tensor-only CPTR checkpoint schema")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise RuntimeError("CPTR checkpoint state_dict is not tensor-only")
    return checkpoint


def load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def candidate_from_grid(grid: dict[str, Any]) -> dict[str, Any]:
    matches = [item for item in grid["candidates"] if item["candidate_id"] == "centre_short_parts"]
    if len(matches) != 1:
        raise RuntimeError("The centre_short_parts candidate is absent or ambiguous")
    return matches[0]


def run_checkpoint(
    *,
    entry: dict[str, Any],
    bundle: dict[str, np.ndarray],
    protocol: dict[str, Any],
    temporal_grid: dict[str, Any],
    candidate: dict[str, Any],
    repository_root: Path,
    device: torch.device,
    batch_size: int,
    output_path: Path,
    replay_source_sha256: dict[str, str],
    execution_lock_sha256: str,
) -> dict[str, Any]:
    fold = int(entry["fold"])
    seed = int(entry["seed"])
    fixed_epochs = int(entry["fixed_epochs"])
    files = entry["files"]
    for evidence in files.values():
        path = repository_root / evidence["path"]
        if path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != evidence["sha256"]:
            raise RuntimeError(f"Frozen replay input changed: {evidence['path']}")
    checkpoint_path = repository_root / files["candidate_checkpoint"]["path"]
    checkpoint = load_weights_only(checkpoint_path)
    if (
        int(checkpoint["fixed_epochs"]) != fixed_epochs
        or checkpoint["request_sha256"] != entry.get("request_sha256")
    ):
        raise RuntimeError("Checkpoint request/fixed-epoch values differ from the inventory")

    model = build_model(
        state_dict=checkpoint["model_state_dict"],
        candidate=candidate,
        protocol=protocol,
        temporal_grid=temporal_grid,
        input_dim=int(bundle["static_features"].shape[1]),
        part_dim=int(bundle.get("part_tokens", np.empty((0, 0, 0, 768))).shape[-1]),
        device=device,
    )
    if fixed_epochs == 0:
        validate_zero_epoch_state(model)

    fold_name = f"fold-{fold}"
    all_ids = bundle["sample_ids"].astype(str)
    selected = np.flatnonzero(
        (bundle["scope"].astype(str) == "grouped_crossfit_oof")
        & (bundle["fold"].astype(str) == fold_name)
    )
    historical = load_npz(repository_root / files["candidate_predictions"]["path"])
    historical_position = {
        value: index for index, value in enumerate(historical["sample_ids"].astype(str))
    }
    selected_ids = all_ids[selected]
    order = np.asarray([historical_position[value] for value in selected_ids], dtype=np.int64)
    historical_probabilities = historical["probabilities"][order].astype(np.float32)
    candidate_retained_teacher = historical["baseline_probabilities"][order].astype(np.float32)
    primary_teacher_ids = bundle["primary_retained_teacher_sample_ids"].astype(str)
    primary_teacher_position = {
        value: index for index, value in enumerate(primary_teacher_ids)
    }
    retained_seed_positions = {
        int(value): index for index, value in enumerate(bundle["retained_teacher_seeds"])
    }
    if seed not in retained_seed_positions or any(
        value not in primary_teacher_position for value in selected_ids
    ):
        raise RuntimeError("Role-safe retained-teacher bundle coverage changed")
    primary_order = np.asarray(
        [primary_teacher_position[value] for value in selected_ids], dtype=np.int64
    )
    historical_teacher = bundle["primary_retained_teacher_probabilities"][
        primary_order, retained_seed_positions[seed]
    ].astype(np.float32, copy=False)
    if not np.array_equal(historical_teacher, candidate_retained_teacher):
        raise RuntimeError(
            "Role-safe bundled teacher differs from the inventory-bound candidate anchor"
        )

    collected: dict[str, list[np.ndarray]] = {
        **{f"{mode}_probabilities": [] for mode in MODES},
        **{f"{mode}_posture_logits": [] for mode in MODES},
        **{f"{mode}_motion_logits": [] for mode in MODES},
        "static_posture_logits": [],
        "static_motion_logits": [],
        "legacy_posture_logits": [],
        "legacy_motion_logits": [],
        "learned_posture_delta": [],
        "learned_motion_delta": [],
        "posture_legacy_gate": [],
        "motion_legacy_gate": [],
        "posture_gates": [],
        "motion_gates": [],
        "expert_reliability": [],
        "q": [],
        "q_recovery_error": [],
    }
    started = time.perf_counter()
    for start in range(0, len(selected), batch_size):
        local = selected[start : start + batch_size]
        static_features = torch.from_numpy(bundle["static_features"][local]).to(device)
        short_features = torch.from_numpy(bundle["short_features"][local]).to(device)
        quality = (
            torch.from_numpy(bundle["quality_features"][local]).to(device)
            if "quality_features" in bundle
            else torch.zeros((len(local), 8), dtype=torch.float32, device=device)
        )
        short_mask = torch.from_numpy(bundle["short_valid_mask"][local]).to(device)
        part_tokens = torch.from_numpy(bundle["part_tokens"][local]).to(device)
        part_confidence = torch.from_numpy(bundle["part_confidence"][local]).to(device)
        part_mask = torch.from_numpy(bundle["part_valid_mask"][local]).to(device)
        recovery_error = torch.zeros(len(local), device=device)
        if fixed_epochs == 0:
            with torch.inference_mode(), torch.autocast(device_type=device.type):
                static_probe = model.static_fallback(static_features)
                legacy_probe = model.legacy_temporal(
                    short_features,
                    torch.ones(short_features.shape[:2], dtype=torch.bool, device=device),
                )
                q, recovery_error = recover_zero_epoch_q(
                    static_probe.posture_logits,
                    static_probe.motion_logits,
                    legacy_probe.posture_logits,
                    legacy_probe.motion_logits,
                    torch.from_numpy(historical_probabilities[start : start + len(local)]).to(
                        device
                    ),
                )
            actual_q = short_mask.float().mean(dim=1)
            if not torch.equal(q, actual_q):
                maximum = float(torch.max(torch.abs(q - actual_q)))
                raise RuntimeError(f"Recovered zero-epoch q differs from exact masks: {maximum}")
        kwargs: dict[str, torch.Tensor | int] = {
            "static_features": static_features,
            "short_features": short_features,
            "short_valid_mask": short_mask,
            "short_centre_index": int(bundle["short_centre_index"]),
            "quality_features": quality,
            "part_tokens": part_tokens,
            "part_confidence": part_confidence,
            "part_valid_mask": part_mask,
            "part_centre_index": int(bundle["part_centre_index"]),
        }
        with torch.inference_mode(), torch.autocast(device_type=device.type):
            result = compute_interventions(model, kwargs)
        for mode in MODES:
            collected[f"{mode}_probabilities"].append(_numpy(result.probabilities[mode]))
            collected[f"{mode}_posture_logits"].append(_numpy(result.posture_logits[mode]))
            collected[f"{mode}_motion_logits"].append(_numpy(result.motion_logits[mode]))
        for name in (
            "static_posture_logits",
            "static_motion_logits",
            "legacy_posture_logits",
            "legacy_motion_logits",
            "learned_posture_delta",
            "learned_motion_delta",
            "posture_legacy_gate",
            "motion_legacy_gate",
            "posture_gates",
            "motion_gates",
            "expert_reliability",
            "q",
        ):
            collected[name].append(_numpy(getattr(result, name)))
        collected["q_recovery_error"].append(_numpy(recovery_error))

    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    # F3's decision input is the retained, role-safe historical teacher tensor.  Keep
    # the current-device replay only as an explicitly diagnostic triplet; otherwise
    # the persisted ``f3_*_logits`` would decode to different probabilities from the
    # authoritative persisted ``f3_probabilities``.
    canonicalize_f3_persistence(arrays, historical_teacher)
    original_error = float(
        np.max(np.abs(arrays["original_probabilities"] - historical_probabilities))
    )
    teacher_error = float(
        np.max(np.abs(arrays["f3_current_direct_probabilities"] - historical_teacher))
    )
    if original_error > REPRODUCTION_MAX_ABS_TOLERANCE:
        raise RuntimeError(
            f"Original replay exceeds the fixed reproduction tolerance: {original_error}"
        )
    if teacher_error > REPRODUCTION_MAX_ABS_TOLERANCE:
        raise RuntimeError(
            f"Current direct-teacher replay exceeds the fixed tolerance: {teacher_error}"
        )
    if not np.array_equal(arrays["f3_probabilities"], historical_teacher):
        raise RuntimeError("F3 statistical probabilities are not bit-exact retained teacher values")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise RuntimeError("Replay produced non-finite arrays")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Replay output already exists: {output_path}")
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_ids=selected_ids,
        recording_ids=bundle["recording_ids"][selected],
        track_ids=bundle["track_ids"][selected],
        labels=bundle["labels"][selected],
        transition_targets=bundle["transition_targets"][selected],
        occlusion_targets=bundle["occlusion_targets"][selected],
        historical_original_probabilities=historical_probabilities,
        historical_teacher_probabilities=historical_teacher,
        **arrays,
    )
    temporary.replace(output_path)
    q_values, q_counts = np.unique(arrays["q"], return_counts=True)
    expert_count = int(arrays["expert_reliability"].shape[1])
    summary = {
        "status": RUN_STATUS,
        "fold": fold,
        "seed": seed,
        "fixed_epochs": fixed_epochs,
        "rows": len(selected),
        "q_source": (
            "role_safe_exact_window_mask_with_zero_epoch_discrete_recovery_cross_check"
            if fixed_epochs == 0
            else "role_safe_exact_window_mask"
        ),
        "original_reproduction_max_abs": original_error,
        "teacher_reproduction_max_abs": teacher_error,
        "q_recovery_max_abs": float(np.max(arrays["q_recovery_error"])),
        "reproduction_max_abs_tolerance": REPRODUCTION_MAX_ABS_TOLERANCE,
        "reproduction_gate_passed": True,
        "f3_current_direct_teacher_identity_within_replay": True,
        "f3_statistical_identity_with_retained_teacher": True,
        "mechanism_diagnostics": {
            "q_distribution": {
                format(float(value), ".3f"): int(count)
                for value, count in zip(q_values, q_counts, strict=True)
            },
            "posture_legacy_gate": {
                "minimum": float(arrays["posture_legacy_gate"].min()),
                "mean": float(arrays["posture_legacy_gate"].mean()),
                "maximum": float(arrays["posture_legacy_gate"].max()),
            },
            "motion_legacy_gate": {
                "minimum": float(arrays["motion_legacy_gate"].min()),
                "mean": float(arrays["motion_legacy_gate"].mean()),
                "maximum": float(arrays["motion_legacy_gate"].max()),
            },
            "posture_expert_gate_means": arrays["posture_gates"].mean(axis=0).tolist(),
            "motion_expert_gate_means": arrays["motion_gates"].mean(axis=0).tolist(),
            "expert_reliability_means": arrays["expert_reliability"].mean(axis=0).tolist(),
            "expert_count": expert_count,
            "learned_posture_delta_abs_max": float(
                np.abs(arrays["learned_posture_delta"]).max()
            ),
            "learned_motion_delta_abs_max": float(
                np.abs(arrays["learned_motion_delta"]).max()
            ),
        },
        "metrics": {
            mode: classification_summary(
                bundle["labels"][selected], arrays[f"{mode}_probabilities"]
            )
            for mode in MODES
        },
        "window_occluded_metrics": {
            mode: classification_summary(
                bundle["labels"][selected][bundle["occlusion_targets"][selected]],
                arrays[f"{mode}_probabilities"][bundle["occlusion_targets"][selected]],
            )
            for mode in MODES
        },
        "runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": {output_path.name: sha256_file(output_path)},
        "source_sha256": replay_source_sha256,
        "execution_lock_sha256_at_creation": execution_lock_sha256,
        "frozen_input_sha256": {
            name: str(evidence["sha256"]) for name, evidence in sorted(files.items())
        },
        "protected_rows_read": 0,
    }
    return summary


def validate_existing_checkpoint_output(
    *,
    output_path: Path,
    entry: dict[str, Any],
    replay_source_sha256: dict[str, str],
    execution_lock_sha256: str | None = None,
) -> dict[str, Any]:
    summary_path = output_path.with_name("summary.json")
    if not output_path.exists() or not summary_path.exists():
        raise RuntimeError(
            f"Partial existing replay output cannot be resumed safely: {output_path.parent}"
        )
    summary = load_json(summary_path)
    expected_inputs = {
        name: str(evidence["sha256"])
        for name, evidence in sorted(entry["files"].items())
    }
    if (
        summary.get("status") != RUN_STATUS
        or int(summary.get("fold", -1)) != int(entry["fold"])
        or int(summary.get("seed", -1)) != int(entry["seed"])
        or int(summary.get("fixed_epochs", -1)) != int(entry["fixed_epochs"])
        or summary.get("source_sha256") != replay_source_sha256
        or (
            execution_lock_sha256 is not None
            and summary.get("execution_lock_sha256_at_creation") != execution_lock_sha256
        )
        or summary.get("frozen_input_sha256") != expected_inputs
        or summary.get("artifact_sha256", {}).get(output_path.name)
        != sha256_file(output_path)
        or summary.get("reproduction_gate_passed") is not True
        or summary.get("f3_statistical_identity_with_retained_teacher") is not True
    ):
        raise RuntimeError(f"Existing replay output failed closed resume validation: {output_path}")
    return summary


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _group_statistics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    group_order: np.ndarray,
    selector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    confusion = np.zeros((len(group_order), CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    loss_sum = np.zeros(len(group_order), dtype=np.float64)
    rows = np.zeros(len(group_order), dtype=np.int64)
    true_counts = np.zeros((len(group_order), CLASS_COUNT), dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    true_loss = -np.log(
        np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    for group_index, group in enumerate(group_order):
        selected = selector & (groups == group)
        rows[group_index] = int(selected.sum())
        if rows[group_index] == 0:
            continue
        np.add.at(confusion[group_index], (labels[selected], predictions[selected]), 1)
        true_counts[group_index] = np.bincount(labels[selected], minlength=CLASS_COUNT)
        loss_sum[group_index] = float(true_loss[selected].sum())
    return confusion, loss_sum, rows, true_counts


def _macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1).astype(np.float64)
    denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
    scores = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator != 0,
    )
    return scores.mean(axis=-1)


def _bootstrap_contrast(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    group_order: np.ndarray,
    selector: np.ndarray,
    draws: np.ndarray,
) -> dict[str, Any]:
    ref_conf, ref_loss, ref_rows, true_counts = _group_statistics(
        labels, reference, groups, group_order, selector
    )
    cand_conf, cand_loss, cand_rows, candidate_true = _group_statistics(
        labels, candidate, groups, group_order, selector
    )
    if not np.array_equal(ref_rows, cand_rows) or not np.array_equal(true_counts, candidate_true):
        raise RuntimeError("Paired bootstrap row support differs between models")
    ref_confusion = ref_conf[draws].sum(axis=1)
    cand_confusion = cand_conf[draws].sum(axis=1)
    sampled_rows = ref_rows[draws].sum(axis=1)
    sampled_true = true_counts[draws].sum(axis=1)
    valid = (sampled_rows > 0) & np.all(sampled_true > 0, axis=1)
    f1_delta = _macro_f1_from_confusion(cand_confusion) - _macro_f1_from_confusion(
        ref_confusion
    )
    nll_delta = np.divide(
        (cand_loss[draws] - ref_loss[draws]).sum(axis=1),
        sampled_rows,
        out=np.full(len(draws), np.nan, dtype=np.float64),
        where=sampled_rows > 0,
    )
    selected_f1 = f1_delta[valid]
    selected_nll = nll_delta[valid]
    observed_rows = int(ref_rows.sum())
    observed_f1 = float(
        _macro_f1_from_confusion(cand_conf.sum(axis=0))
        - _macro_f1_from_confusion(ref_conf.sum(axis=0))
    )
    observed_nll = float((cand_loss.sum() - ref_loss.sum()) / observed_rows)
    if not len(selected_f1):
        raise RuntimeError("No scenario bootstrap resample supports all three fixed classes")
    return {
        "observed_rows": observed_rows,
        "observed_macro_f1_delta": observed_f1,
        "observed_nll_delta": observed_nll,
        "valid_resamples": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "macro_f1_delta_one_sided_95pct_lower": float(np.quantile(selected_f1, 0.05)),
        "macro_f1_delta_two_sided_95pct": [
            float(np.quantile(selected_f1, 0.025)),
            float(np.quantile(selected_f1, 0.975)),
        ],
        "nll_delta_one_sided_95pct_upper": float(np.quantile(selected_nll, 0.95)),
        "nll_delta_two_sided_95pct": [
            float(np.quantile(selected_nll, 0.025)),
            float(np.quantile(selected_nll, 0.975)),
        ],
    }


def _exact_group_swap_pvalue(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    group_order: np.ndarray,
    assignments: np.ndarray | None = None,
) -> dict[str, Any]:
    selector = np.ones(len(labels), dtype=bool)
    ref_conf, _, _, _ = _group_statistics(labels, reference, groups, group_order, selector)
    cand_conf, _, _, _ = _group_statistics(labels, candidate, groups, group_order, selector)
    if assignments is None:
        integers = np.arange(1 << len(group_order), dtype=np.uint32)[:, None]
        assignments = (
            (integers >> np.arange(len(group_order), dtype=np.uint32)) & 1
        ).astype(bool)
    if assignments.shape != (1 << len(group_order), len(group_order)):
        raise RuntimeError("Exact swap assignment matrix has the wrong shape")
    bits = assignments.astype(bool, copy=False)
    candidate_null = np.where(bits[:, :, None, None], ref_conf[None], cand_conf[None]).sum(
        axis=1
    )
    reference_null = np.where(bits[:, :, None, None], cand_conf[None], ref_conf[None]).sum(
        axis=1
    )
    null_delta = _macro_f1_from_confusion(candidate_null) - _macro_f1_from_confusion(
        reference_null
    )
    observed = float(
        _macro_f1_from_confusion(cand_conf.sum(axis=0))
        - _macro_f1_from_confusion(ref_conf.sum(axis=0))
    )
    pvalue = float(np.mean(null_delta >= observed - 1e-15))
    return {
        "observed_macro_f1_delta": observed,
        "one_sided_pvalue": pvalue,
        "assignments": int(len(bits)),
    }


def _holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, (total - index) * values[name])
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in values}


def _brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    targets = np.eye(CLASS_COUNT, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def _confusion_and_recall(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[list[list[int]], dict[str, float | None]]:
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (labels, probabilities.argmax(axis=1)), 1)
    support = confusion.sum(axis=1)
    recall = {
        name: (float(confusion[index, index] / support[index]) if support[index] else None)
        for index, name in enumerate(CLASS_NAMES)
    }
    return confusion.tolist(), recall


def _error_flow(
    labels: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, int | float | None]:
    reference_correct = reference.argmax(axis=1) == labels
    candidate_correct = candidate.argmax(axis=1) == labels
    rescued = int((~reference_correct & candidate_correct).sum())
    harmed = int((reference_correct & ~candidate_correct).sum())
    baseline_errors = int((~reference_correct).sum())
    baseline_correct = int(reference_correct.sum())
    return {
        "both_correct": int((reference_correct & candidate_correct).sum()),
        "rescued": rescued,
        "harmed": harmed,
        "both_wrong": int((~reference_correct & ~candidate_correct).sum()),
        "rescue_fraction_of_original_errors": (
            float(rescued / baseline_errors) if baseline_errors else None
        ),
        "harm_fraction_of_original_correct": (
            float(harmed / baseline_correct) if baseline_correct else None
        ),
    }


def _slice_report(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    selector: np.ndarray,
) -> dict[str, Any]:
    selected_labels = labels[selector]
    if not len(selected_labels):
        return {"rows": 0, "supported_classes": 0, "status": "EMPTY"}
    class_counts = np.bincount(selected_labels, minlength=CLASS_COUNT)
    metrics: dict[str, Any] = {}
    for mode in MODES:
        selected_probabilities = probabilities[mode][selector]
        confusion, recall = _confusion_and_recall(selected_labels, selected_probabilities)
        metrics[mode] = {
            **classification_summary(selected_labels, selected_probabilities),
            "brier": _brier_score(selected_labels, selected_probabilities),
            "confusion": confusion,
            "true_class_recall": recall,
        }
    return {
        "rows": int(len(selected_labels)),
        "class_counts": {
            name: int(class_counts[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "supported_classes": int(np.count_nonzero(class_counts)),
        "metric_scope": (
            "fixed_three_class_macro_f1"
            if np.all(class_counts > 0)
            else "fixed_three_class_metrics_with_true_class_recall_for_sparse_slice"
        ),
        "metrics": metrics,
        "f2_minus_original": {
            key: float(metrics["f2"][key] - metrics["original"][key])
            for key in (
                "macro_f1",
                "accuracy",
                "log_loss",
                "brier",
                "sitting_f1",
                "standing_f1",
                "walking_running_f1",
            )
        },
        "f2_error_flow_from_original": _error_flow(
            selected_labels,
            probabilities["original"][selector],
            probabilities["f2"][selector],
        ),
    }


def decision_statistics(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    occluded: np.ndarray,
    transition: np.ndarray,
    probabilities: dict[str, np.ndarray],
    output_dir: Path,
    stage_name: str,
) -> tuple[dict[str, Any], Path]:
    group_order = np.asarray(sorted(np.unique(groups.astype(str))), dtype=str)
    if len(group_order) != 11:
        raise RuntimeError("Primary CPTR replay must contain the fixed 11 scenarios")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    initial_rng_state = json.dumps(generator.bit_generator.state, sort_keys=True)
    draws = generator.integers(
        0, len(group_order), size=(BOOTSTRAP_RESAMPLES, len(group_order)), dtype=np.int16
    )
    final_rng_state = json.dumps(generator.bit_generator.state, sort_keys=True)
    integers = np.arange(1 << len(group_order), dtype=np.uint32)[:, None]
    swap_assignments = (
        (integers >> np.arange(len(group_order), dtype=np.uint32)) & 1
    ).astype(bool)

    f2_occluded = _bootstrap_contrast(
        labels,
        probabilities["original"],
        probabilities["f2"],
        groups,
        group_order,
        occluded,
        draws,
    )
    f2_teacher = _bootstrap_contrast(
        labels,
        probabilities["f3"],
        probabilities["f2"],
        groups,
        group_order,
        np.ones(len(labels), dtype=bool),
        draws,
    )
    declared_selectors = {
        "aggregate": np.ones(len(labels), dtype=bool),
        "window_occluded": occluded,
        "window_clear": ~occluded,
        "transition": transition,
        "non_transition": ~transition,
    }
    bootstrap_strata: dict[str, Any] = {}
    for name, selector in declared_selectors.items():
        class_counts = np.bincount(labels[selector], minlength=CLASS_COUNT)
        if not np.all(class_counts > 0):
            bootstrap_strata[name] = {
                "status": "INSUFFICIENT_FIXED_CLASS_SUPPORT",
                "class_counts": class_counts.tolist(),
            }
            continue
        bootstrap_strata[name] = _bootstrap_contrast(
            labels,
            probabilities["original"],
            probabilities["f2"],
            groups,
            group_order,
            selector,
            draws,
        )
    swaps = {
        mode: _exact_group_swap_pvalue(
            labels,
            probabilities["original"],
            probabilities[mode],
            groups,
            group_order,
            swap_assignments,
        )
        for mode in ("f1", "f2")
    }
    adjusted = _holm_adjust(
        {mode: float(result["one_sided_pvalue"]) for mode, result in swaps.items()}
    )
    for mode in swaps:
        swaps[mode]["holm_adjusted_one_sided_pvalue"] = adjusted[mode]
        swaps[mode]["holm_adjusted_significant_at_0_05"] = adjusted[mode] <= 0.05
    point_original = classification_summary(labels[occluded], probabilities["original"][occluded])
    point_f2 = classification_summary(labels[occluded], probabilities["f2"][occluded])
    point_delta = float(point_f2["macro_f1"] - point_original["macro_f1"])
    checks = {
        "occluded_macro_f1_point_at_least_0_010": point_delta >= 0.010,
        "occluded_macro_f1_lower_above_zero": f2_occluded[
            "macro_f1_delta_one_sided_95pct_lower"
        ]
        > 0.0,
        "occluded_nll_upper_at_most_zero": f2_occluded[
            "nll_delta_one_sided_95pct_upper"
        ]
        <= 0.0,
        "aggregate_f2_minus_teacher_lower_above_minus_0_005": f2_teacher[
            "macro_f1_delta_one_sided_95pct_lower"
        ]
        > -0.005,
    }

    stratum_reports = {
        name: _slice_report(labels, probabilities, selector)
        for name, selector in declared_selectors.items()
    }
    scenario_reports = {
        str(group): _slice_report(labels, probabilities, groups == group) for group in group_order
    }
    full_deltas = stratum_reports["aggregate"]["f2_minus_original"]
    per_class_deltas = {
        name: float(full_deltas[f"{name}_f1"]) for name in CLASS_NAMES
    }
    adequate_scenario_deltas = {
        name: float(report["f2_minus_original"]["macro_f1"])
        for name, report in scenario_reports.items()
        if int(report["supported_classes"]) == CLASS_COUNT
    }
    adequate_stratum_deltas = {
        name: float(report["f2_minus_original"]["macro_f1"])
        for name, report in stratum_reports.items()
        if name != "aggregate" and int(report["supported_classes"]) == CLASS_COUNT
    }
    supported_bootstrap = {
        name: value
        for name, value in bootstrap_strata.items()
        if "valid_fraction" in value
    }
    guardrail_checks = {
        "no_full_cohort_class_f1_regression_below_minus_0_010": all(
            delta >= -0.010 for delta in per_class_deltas.values()
        ),
        "occluded_macro_f1_lower_at_or_above_minus_0_010": f2_occluded[
            "macro_f1_delta_one_sided_95pct_lower"
        ]
        >= -0.010,
        "all_supported_subgroup_bootstraps_at_least_95pct_valid": all(
            float(value["valid_fraction"]) >= 0.95 for value in supported_bootstrap.values()
        ),
        "no_adequately_supported_scenario_below_minus_0_010": all(
            delta >= -0.010 for delta in adequate_scenario_deltas.values()
        ),
        "no_adequately_supported_stratum_below_minus_0_010": all(
            delta >= -0.010 for delta in adequate_stratum_deltas.values()
        ),
    }

    resample_path = output_dir / f"{stage_name}_scenario_resample_indices.npz"
    expected_resampling = {
        "group_order": group_order,
        "group_draw_indices": draws,
        "exact_swap_assignments": swap_assignments,
        "bootstrap_seed": np.asarray(BOOTSTRAP_SEED, dtype=np.int64),
        "numpy_version": np.asarray(np.__version__),
        "bit_generator": np.asarray(type(generator.bit_generator).__name__),
        "initial_rng_state_json": np.asarray(initial_rng_state),
        "final_rng_state_json": np.asarray(final_rng_state),
    }
    if resample_path.exists():
        existing = load_npz(resample_path)
        if set(existing) != set(expected_resampling) or any(
            not np.array_equal(existing[name], value)
            for name, value in expected_resampling.items()
        ):
            raise RuntimeError(
                f"Existing scenario resampling artifact failed closed validation: {resample_path}"
            )
    else:
        temporary = resample_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **expected_resampling)
        temporary.replace(resample_path)
    return (
        {
            "bootstrap": {
                "unit": "recording_id_scenario",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "group_order": group_order.tolist(),
                "f2_minus_original_window_occluded": f2_occluded,
                "f2_minus_f3_aggregate": f2_teacher,
                "f2_minus_original_declared_strata": bootstrap_strata,
            },
            "exact_directional_group_swaps": swaps,
            "mechanistic_gate": {
                "f2_minus_original_occluded_macro_f1_point": point_delta,
                "checks": checks,
                "passed": all(checks.values()),
                "decision_authority": stage_name == "r1b",
                "authorizes_R2": stage_name == "r1b" and all(checks.values()),
                "interpretation": (
                    "R1b_full_five_seed_stage_opening_decision"
                    if stage_name == "r1b"
                    else "R1a_seed43_diagnostic_only_cannot_open_R2"
                ),
            },
            "promotion_guardrails": {
                "full_cohort_f2_minus_original_class_f1": per_class_deltas,
                "adequately_supported_scenario_macro_f1_deltas": adequate_scenario_deltas,
                "adequately_supported_stratum_macro_f1_deltas": adequate_stratum_deltas,
                "checks": guardrail_checks,
                "passed": all(guardrail_checks.values()),
            },
            "declared_strata": stratum_reports,
            "scenarios": scenario_reports,
            "rescue_harm_orientation": "F2 relative to Original",
            "resampling_artifact": {
                "file": resample_path.name,
                "sha256": sha256_file(resample_path),
                "bootstrap_index_shape": list(draws.shape),
                "exact_swap_assignment_shape": list(swap_assignments.shape),
                "numpy_version": np.__version__,
                "bit_generator": type(generator.bit_generator).__name__,
            },
        },
        resample_path,
    )


def aggregate_outputs(
    output_paths: list[Path],
    *,
    seeds: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    stage_name = "r1a" if seeds == [43] else "r1b"
    by_seed: dict[int, list[dict[str, np.ndarray]]] = {seed: [] for seed in seeds}
    for path in output_paths:
        seed = int(path.parent.name.removeprefix("seed-"))
        by_seed[seed].append(load_npz(path))
    if any(len(values) != len(EXPECTED_FOLDS) for values in by_seed.values()):
        raise RuntimeError("Aggregate replay requires exactly one output from every fold per seed")
    seed_arrays: dict[int, dict[str, np.ndarray]] = {}
    for seed, groups in by_seed.items():
        merged: dict[str, np.ndarray] = {}
        for key in groups[0]:
            merged[key] = np.concatenate([group[key] for group in groups])
        order = np.argsort(merged["sample_ids"].astype(str), kind="stable")
        seed_arrays[seed] = {key: value[order] for key, value in merged.items()}
    reference_ids = seed_arrays[seeds[0]]["sample_ids"].astype(str)
    if len(reference_ids) != 4977 or len(np.unique(reference_ids)) != len(reference_ids):
        raise RuntimeError("Primary OOF replay row cardinality or uniqueness changed")
    for values in seed_arrays.values():
        if not np.array_equal(reference_ids, values["sample_ids"].astype(str)):
            raise RuntimeError("Replay row identity/order differs across seeds")
        for name in (
            "recording_ids",
            "track_ids",
            "labels",
            "transition_targets",
            "occlusion_targets",
        ):
            if not np.array_equal(seed_arrays[seeds[0]][name], values[name]):
                raise RuntimeError(f"Replay {name} differs across seeds")
        if not np.array_equal(
            values["f3_probabilities"], values["historical_teacher_probabilities"]
        ):
            raise RuntimeError("A checkpoint output lost bit-exact retained-teacher F3 identity")
    aggregate = {
        mode: np.mean(
            np.stack([seed_arrays[seed][f"{mode}_probabilities"] for seed in seeds], axis=0),
            axis=0,
        )
        for mode in MODES
    }
    for mode, probabilities in aggregate.items():
        if (
            not np.isfinite(probabilities).all()
            or np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 1e-5
        ):
            raise RuntimeError(f"Aggregate {mode} probabilities are invalid")
    labels = seed_arrays[seeds[0]]["labels"].astype(np.int64)
    occluded = seed_arrays[seeds[0]]["occlusion_targets"].astype(bool)
    transition = seed_arrays[seeds[0]]["transition_targets"].astype(bool)
    recording_ids = seed_arrays[seeds[0]]["recording_ids"].astype(str)
    aggregate_path = output_dir / f"{stage_name}_aggregate_predictions.npz"
    resample_path = output_dir / f"{stage_name}_scenario_resample_indices.npz"
    expected_aggregate = {
        "sample_ids": reference_ids,
        "recording_ids": seed_arrays[seeds[0]]["recording_ids"],
        "track_ids": seed_arrays[seeds[0]]["track_ids"],
        "labels": labels,
        "transition_targets": seed_arrays[seeds[0]]["transition_targets"],
        "occlusion_targets": seed_arrays[seeds[0]]["occlusion_targets"],
        **{f"{mode}_probabilities": values for mode, values in aggregate.items()},
    }
    if aggregate_path.exists():
        existing = load_npz(aggregate_path)
        if set(existing) != set(expected_aggregate) or any(
            not np.array_equal(existing[name], value)
            for name, value in expected_aggregate.items()
        ):
            raise RuntimeError(
                f"Existing aggregate replay artifact failed closed validation: {aggregate_path}"
            )
    else:
        temporary = aggregate_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            **expected_aggregate,
        )
        temporary.replace(aggregate_path)
    statistics, emitted_resample_path = decision_statistics(
        labels=labels,
        groups=recording_ids,
        occluded=occluded,
        transition=transition,
        probabilities=aggregate,
        output_dir=output_dir,
        stage_name=stage_name,
    )
    if emitted_resample_path != resample_path:
        raise RuntimeError("Decision-statistics artifact path changed unexpectedly")
    return {
        "status": SUMMARY_STATUS,
        "stage": stage_name,
        "seeds": seeds,
        "folds": list(EXPECTED_FOLDS),
        "rows": len(reference_ids),
        "aggregation": "checkpoint_probabilities_then_arithmetic_mean_across_seeds",
        "engineering_identity_gate": {
            "f3_bit_exact_retained_teacher_for_every_checkpoint_row": True,
            "passed": True,
        },
        "metrics": {mode: classification_summary(labels, aggregate[mode]) for mode in MODES},
        "window_occluded_metrics": {
            mode: classification_summary(labels[occluded], aggregate[mode][occluded])
            for mode in MODES
        },
        "primary_deltas": {
            "f2_minus_original_macro_f1": float(
                classification_summary(labels, aggregate["f2"])["macro_f1"]
                - classification_summary(labels, aggregate["original"])["macro_f1"]
            ),
            "f2_minus_original_occluded_macro_f1": float(
                classification_summary(labels[occluded], aggregate["f2"][occluded])["macro_f1"]
                - classification_summary(labels[occluded], aggregate["original"][occluded])[
                    "macro_f1"
                ]
            ),
            "f2_minus_f3_macro_f1": float(
                classification_summary(labels, aggregate["f2"])["macro_f1"]
                - classification_summary(labels, aggregate["f3"])["macro_f1"]
            ),
        },
        "decision_statistics": statistics,
        "artifact_sha256": {
            aggregate_path.name: sha256_file(aggregate_path),
            resample_path.name: sha256_file(resample_path),
        },
        "protected_rows_read": 0,
    }


def validate_role_safe_bundle(bundle: dict[str, np.ndarray]) -> None:
    required = {
        "sample_ids",
        "recording_ids",
        "track_ids",
        "scope",
        "fold",
        "labels",
        "transition_targets",
        "occlusion_targets",
        "source_feature_indices",
        "static_features",
        "short_features",
        "short_valid_mask",
        "short_centre_index",
        "distinct_short_features",
        "distinct_short_valid_mask",
        "distinct_short_indices",
        "distinct_short_centre_index",
        "part_tokens",
        "part_confidence",
        "part_valid_mask",
        "part_centre_index",
        "quality_features",
        "window_occluded",
        "primary_retained_teacher_sample_ids",
        "retained_teacher_seeds",
        "primary_retained_teacher_probabilities",
    }
    if set(bundle) != required:
        raise RuntimeError(
            "Role-safe bundle arrays changed: "
            f"missing={sorted(required - set(bundle))}, extra={sorted(set(bundle) - required)}"
        )
    rows = 6360
    input_dim = int(bundle["static_features"].shape[-1])
    part_dim = int(bundle["part_tokens"].shape[-1])
    expected_shapes = {
        "sample_ids": (rows,),
        "recording_ids": (rows,),
        "track_ids": (rows,),
        "scope": (rows,),
        "fold": (rows,),
        "labels": (rows,),
        "transition_targets": (rows,),
        "occlusion_targets": (rows,),
        "source_feature_indices": (rows,),
        "static_features": (rows, input_dim),
        "short_features": (rows, 8, input_dim),
        "short_valid_mask": (rows, 8),
        "short_centre_index": (),
        "distinct_short_features": (rows, 8, input_dim),
        "distinct_short_valid_mask": (rows, 8),
        "distinct_short_indices": (8,),
        "distinct_short_centre_index": (),
        "part_tokens": (rows, 8, 7, part_dim),
        "part_confidence": (rows, 8, 7),
        "part_valid_mask": (rows, 8),
        "part_centre_index": (),
        "quality_features": (rows, 8),
        "window_occluded": (rows, 17),
        "primary_retained_teacher_sample_ids": (4977,),
        "retained_teacher_seeds": (5,),
        "primary_retained_teacher_probabilities": (4977, 5, 3),
    }
    for name, shape in expected_shapes.items():
        if bundle[name].shape != shape:
            raise RuntimeError(f"Role-safe bundle {name} shape changed: {bundle[name].shape}")
    for name in (
        "static_features",
        "short_features",
        "distinct_short_features",
        "part_tokens",
        "part_confidence",
        "quality_features",
    ):
        value = bundle[name]
        finite = all(
            np.isfinite(value[start : start + 128]).all() for start in range(0, rows, 128)
        )
        if value.dtype != np.dtype("float32") or not finite:
            raise RuntimeError(f"Role-safe bundle {name} dtype or finiteness changed")
    for name in (
        "short_valid_mask",
        "distinct_short_valid_mask",
        "part_valid_mask",
        "window_occluded",
        "transition_targets",
        "occlusion_targets",
    ):
        if bundle[name].dtype != np.dtype("bool"):
            raise RuntimeError(f"Role-safe bundle {name} is not Boolean")
    sample_ids = bundle["sample_ids"].astype(str)
    if len(np.unique(sample_ids)) != rows:
        raise RuntimeError("Role-safe bundle row identity is not unique")
    feature_indices = bundle["source_feature_indices"].astype(np.int64)
    if (
        len(np.unique(feature_indices)) != rows
        or feature_indices.min() < 0
        or feature_indices.max() >= 8339
    ):
        raise RuntimeError("Role-safe source feature positions changed")
    labels = bundle["labels"].astype(np.int64)
    if not np.array_equal(bundle["labels"], labels) or set(labels.tolist()) != {0, 1, 2}:
        raise RuntimeError("Role-safe labels changed")
    if not np.array_equal(bundle["occlusion_targets"], bundle["window_occluded"].any(axis=1)):
        raise RuntimeError("Role-safe occlusion targets differ from exact masks")
    primary_ids = sample_ids[bundle["scope"].astype(str) == "grouped_crossfit_oof"]
    if not np.array_equal(
        bundle["primary_retained_teacher_sample_ids"].astype(str), primary_ids
    ):
        raise RuntimeError("Retained-teacher row order differs from primary bundle order")
    if not np.array_equal(
        bundle["retained_teacher_seeds"], np.asarray(EXPECTED_SEEDS, dtype=np.int64)
    ):
        raise RuntimeError("Retained-teacher seed order changed")
    retained = bundle["primary_retained_teacher_probabilities"]
    if (
        retained.dtype != np.dtype("float32")
        or not np.isfinite(retained).all()
        or np.max(np.abs(retained.sum(axis=2) - 1.0)) > 1e-5
    ):
        raise RuntimeError("Retained-teacher probabilities are invalid")
    if (
        int(bundle["short_centre_index"].item()) != 4
        or int(bundle["part_centre_index"].item()) != 3
        or int(bundle["distinct_short_centre_index"].item()) != 4
        or bundle["distinct_short_indices"].tolist() != [4, 5, 6, 7, 8, 10, 11, 12]
    ):
        raise RuntimeError("Role-safe bundle temporal preprocessing changed")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers != 0:
        raise ValueError("Replay requires positive batch size and workers=0")
    seeds = sorted(set(args.seeds))
    folds = sorted(set(args.folds))
    if not set(seeds) <= set(EXPECTED_SEEDS) or not set(folds) <= set(EXPECTED_FOLDS):
        raise ValueError("Replay seed or fold is outside the frozen 5x5 design")
    if folds != list(EXPECTED_FOLDS):
        raise ValueError("Aggregate replay requires all five folds")
    if seeds not in ([43], list(EXPECTED_SEEDS)):
        raise ValueError("Frozen replay permits only R1a seed 43 or the complete R1b five-seed set")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen CPTR replay requires CUDA")
    inputs = ReplayInputs(
        inventory=args.inventory.resolve(),
        eligible_index=args.eligible_index.resolve(),
        feature_bundle=args.feature_bundle.resolve(),
        bundle_summary=args.bundle_summary.resolve(),
        protocol_lock=args.protocol_lock.resolve(),
    )
    inventory, _bundle_summary, execution_lock = validate_protocol_lock(inputs)
    if any(seed != 43 for seed in seeds) and execution_lock.get("authorization", {}).get(
        "R1b_execution"
    ) is not True:
        raise RuntimeError("Execution lock authorizes R1a only; trained seeds require an R1b lock")
    bundle = load_npz(inputs.feature_bundle)
    validate_role_safe_bundle(bundle)
    repository_root = Path(__file__).resolve().parents[1]
    source_files = inventory["source_files"]
    for name in ("cptr_protocol", "temporal_grid", "candidate_grid"):
        evidence = source_files[name]
        path = repository_root / evidence["path"]
        if path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != evidence["sha256"]:
            raise RuntimeError(f"Frozen replay configuration changed: {evidence['path']}")
    protocol = load_json(repository_root / source_files["cptr_protocol"]["path"])
    temporal_grid = load_json(repository_root / source_files["temporal_grid"]["path"])
    grid = load_json(repository_root / source_files["candidate_grid"]["path"])
    candidate = candidate_from_grid(grid)
    entries = {(int(item["fold"]), int(item["seed"])): item for item in inventory["checkpoints"]}
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_lock_sha256 = sha256_file(inputs.protocol_lock)
    replay_source_sha256 = {
        "materialization_source_lock": str(execution_lock["source_sha256"]["source_lock"]),
        "input_inventory": sha256_file(inputs.inventory),
        "eligible_index": sha256_file(inputs.eligible_index),
        "eligible_feature_bundle": sha256_file(inputs.feature_bundle),
        "bundle_summary": sha256_file(inputs.bundle_summary),
        "runner": sha256_file(Path(__file__)),
        "cptr_model_module": sha256_file(repository_root / "src/hac/cptr.py"),
        "cptr_feature_module": sha256_file(repository_root / "src/hac/cptr_features.py"),
        "cptr_training_module": sha256_file(repository_root / "src/hac/cptr_training.py"),
        "temporal_model_module": sha256_file(
            repository_root / "src/hac/vcoco_v3_temporal.py"
        ),
        "neural_module": sha256_file(repository_root / "src/hac/vcoco_v3_neural.py"),
    }
    outputs: list[Path] = []
    run_summaries: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            path = output_dir / f"fold-{fold}" / f"seed-{seed}" / "replay.npz"
            entry = entries[(fold, seed)]
            if path.exists() or path.with_name("summary.json").exists():
                summary = validate_existing_checkpoint_output(
                    output_path=path,
                    entry=entry,
                    replay_source_sha256=replay_source_sha256,
                    execution_lock_sha256=execution_lock_sha256,
                )
                summary = {**summary, "resume_action": "verified_existing_output_reused"}
            else:
                summary = run_checkpoint(
                    entry=entry,
                    bundle=bundle,
                    protocol=protocol,
                    temporal_grid=temporal_grid,
                    candidate=candidate,
                    repository_root=repository_root,
                    device=device,
                    batch_size=args.batch_size,
                    output_path=path,
                    replay_source_sha256=replay_source_sha256,
                    execution_lock_sha256=execution_lock_sha256,
                )
                write_json_atomic(path.with_name("summary.json"), summary)
            outputs.append(path)
            run_summaries.append(summary)
            print(json.dumps({key: summary[key] for key in ("fold", "seed", "rows")}), flush=True)
    aggregate = aggregate_outputs(outputs, seeds=seeds, output_dir=output_dir)
    aggregate["checkpoint_replays"] = run_summaries
    aggregate["source_sha256"] = {
        **replay_source_sha256,
        "execution_lock": execution_lock_sha256,
    }
    aggregate["execution_device"] = torch.cuda.get_device_name(0)
    aggregate["torch_version"] = torch.__version__
    write_json_atomic(output_dir / f"{aggregate['stage']}_summary.json", aggregate)
    print(json.dumps(aggregate["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
