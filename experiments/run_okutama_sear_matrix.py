# ruff: noqa: E402

"""Resource-gate, cross-fit and summarize the six-arm SEAR experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", EXPECTED_CUBLAS_WORKSPACE_CONFIG)

import numpy as np
import sklearn
import torch
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_okutama_video_probe import holm_adjust, paired_statistics
from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.sear_evaluation import reference_strata_summary, slot_diagnostics
from hac.sear_training import (
    ARMS,
    TEMPLATE_ARMS,
    forward_arm,
    make_model,
    parameter_counts,
    seed_training,
    training_loss,
    video_module,
)
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_sear_matrix_protocol.json"
DENSE = ROOT / ".runs/research_20260908/sear_dense_dino_full_v1"
SOURCE_DATA = ROOT / ".runs/research_20260908/source_swap_v1/data"
SOURCE_RUN = ROOT / ".runs/research_20260908/source_swap_v1"
SOURCE_RESULT = SOURCE_RUN / "results/v0001"
SOURCE_RESULT_SUMMARY = SOURCE_RESULT / "summary.json"
SOURCE_RESULT_OOF = SOURCE_RESULT / "oof_probabilities.npz"
SOURCE_FINAL_AUDIT = SOURCE_RUN / "diagnostics/final_audit_v0001/summary.json"
SOURCE_EXECUTION_LOCK = SOURCE_RUN / "memory_execution_lock.json"
OLD_SOURCE_REPLAY = SOURCE_RUN / "old_source_replay.npz"
OLD_SOURCE_REPLAY_RECEIPT = SOURCE_RUN / "old_source_replay.json"
SOURCE_MASKS = SOURCE_DATA / "source_masks.npz"
EVIDENCE_RUN = ROOT / ".runs/research_20260908/evidence_memory"
LEGACY_STRATA = EVIDENCE_RUN / "data/legacy_temporal_strata.npz"
LEGACY_STRATA_RECEIPT = EVIDENCE_RUN / "data/legacy_temporal_strata_receipt.json"
EVIDENCE_OOF = EVIDENCE_RUN / "results/oof_probabilities.npz"
EVIDENCE_SUMMARY = EVIDENCE_RUN / "results/summary.json"
RUN = ROOT / ".runs/research_20260908/sear_matrix_v1"

CODE_PATHS = (
    ROOT / "src/hac/sear.py",
    ROOT / "src/hac/sear_controls.py",
    ROOT / "src/hac/sear_training.py",
    ROOT / "src/hac/sear_evaluation.py",
    ROOT / "src/hac/actor_memory_base.py",
    ROOT / "src/hac/source_swap_data.py",
    ROOT / "experiments/run_okutama_video_probe.py",
    Path(__file__),
)


def event(name: str, **values) -> None:
    print(
        json.dumps({"time": datetime.now(UTC).isoformat(), "event": name, **values}),
        flush=True,
    )


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(entry: dict) -> Path:
    path = Path(entry["path"])
    if (
        not path.is_file()
        or path.stat().st_size != entry["size_bytes"]
        or file_sha256(path) != entry["sha256"]
    ):
        raise RuntimeError(f"SEAR input or output changed: {path}")
    return path


def code_sha256() -> dict[str, str]:
    missing = [path for path in CODE_PATHS if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required SEAR source is missing: {missing}")
    return {str(path.resolve()): file_sha256(path) for path in CODE_PATHS}


def configure_runtime() -> dict:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != EXPECTED_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must equal {EXPECTED_CUBLAS_WORKSPACE_CONFIG}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def arrays_equal(left, right) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def immutable_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Create an NPZ once, or prove that every retained array is identical."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not arrays_equal(saved[name], value) for name, value in arrays.items()
            ):
                raise RuntimeError(f"Immutable SEAR array artifact changed: {path}")
        return
    np.savez_compressed(path, **arrays)


def validate_resource_receipt(saved: dict, protocol: dict) -> None:
    counts = validate_protocol(protocol)
    required_runtime = configure_runtime()
    if (
        not saved.get("passed")
        or saved.get("status") != "SEAR_SIX_ARM_RESOURCE_GATE_PASS"
        or saved.get("protocol_sha256") != file_sha256(PROTOCOL)
        or saved.get("code_sha256") != code_sha256()
        or saved.get("runtime") != required_runtime
        or not str(saved.get("device", "")).startswith("cuda")
        or saved.get("batch_size") != protocol["resource_gate"]["batch_size"]
        or tuple(saved.get("results", {})) != ARMS
    ):
        raise RuntimeError("Retained SEAR resource receipt does not match current CUDA code")
    maximum = protocol["resource_gate"]["maximum_peak_cuda_allocated_bytes"]
    expected_common = None
    for arm in ARMS:
        result = saved["results"][arm]
        model = make_model(protocol, arm)
        expected_schema = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()
        }
        expected_trainable = [
            name for name, value in model.named_parameters() if value.requires_grad
        ]
        common = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in video_module(model, arm).state_dict().items()
        }
        expected_common = common if expected_common is None else expected_common
        if (
            set(result)
            != {
                "parameters",
                "median_forward_backward_seconds",
                "peak_cuda_allocated_bytes",
                "finite_active_gradients",
                "exact_video_fallback",
                "state_dict_schema",
                "trainable_parameter_names",
            }
            or result.get("parameters") != counts[arm]
            or not result.get("finite_active_gradients")
            or not result.get("exact_video_fallback")
            or not 0 < result.get("peak_cuda_allocated_bytes", 0) < maximum
            or not np.isfinite(result.get("median_forward_backward_seconds", np.nan))
            or result.get("median_forward_backward_seconds", 0) <= 0
            or not result.get("state_dict_schema")
            or not result.get("trainable_parameter_names")
            or result["state_dict_schema"] != expected_schema
            or result["trainable_parameter_names"] != expected_trainable
            or common != expected_common
        ):
            raise RuntimeError(f"Retained SEAR resource evidence failed for {arm}")
        del model
    if saved.get("common_video_state_dict_schema") != expected_common:
        raise RuntimeError("Retained common SEAR video-head schema changed")


def validate_or_write_checkpoint(path: Path, value: dict) -> None:
    """Keep completed checkpoints immutable across interrupted-fit recovery."""
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (
            saved.get("request") != value["request"]
            or saved.get("epoch") != value["epoch"]
            or set(saved.get("state_dict", {})) != set(value["state_dict"])
            or any(
                not torch.equal(saved["state_dict"][name], tensor)
                for name, tensor in value["state_dict"].items()
            )
        ):
            raise RuntimeError(f"Retained SEAR checkpoint changed: {path}")
    else:
        torch.save(value, path)


def _dense_receipt(summary: dict, directory: Path, name: str) -> dict:
    declared = summary.get("artifacts", {}).get(name)
    if not isinstance(declared, dict) or "sha256" not in declared:
        raise RuntimeError(f"Dense cache does not bind {name}")
    size = declared.get("size_bytes", declared.get("bytes"))
    if size is None:
        raise RuntimeError(f"Dense cache does not bind the size of {name}")
    return {
        "path": str((directory / name).resolve()),
        "sha256": declared["sha256"],
        "size_bytes": size,
    }


def validate_protocol(protocol: dict) -> dict[str, int]:
    training = protocol["training"]
    expected = (
        len(ARMS)
        * 5
        * (
            len(training["learning_rates"])
            * len(training["weight_decays"])
            * protocol["splitting"]["inner_folds"]
            + len(training["outer_seeds"])
        )
    )
    if (
        tuple(protocol["arms"]) != ARMS
        or protocol["primary_arm"] != "a5_sear"
        or training["maximum_classifier_fits"] != expected
        or expected != 450
        or protocol["inputs"]["additional_frame_labels"]
        or protocol["inputs"]["image_or_video_backbone_fitting"]
        or len(protocol["statistics"]["directional_contrasts"]) != 6
        or protocol["resource_gate"].get("required_device") != "cuda"
        or protocol["resource_gate"].get("cublas_workspace_config")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
        or protocol["resource_gate"].get("tf32") is not False
        or not protocol.get("fixed_diagnostics", {}).get("evaluation_only")
        or protocol["fixed_diagnostics"].get("score_selected_inference_changes") is not False
    ):
        raise RuntimeError("SEAR protocol or exact workload budget changed")
    return parameter_counts(protocol)


def resource_pilot(protocol: dict, output: Path, device: str) -> dict:
    """Synthetic forward/backward only; no labels, images or optimizer updates."""
    path = output / "resource_pilot.json"
    if path.exists():
        saved = read_json(path)
        validate_resource_receipt(saved, protocol)
        return saved
    runtime = configure_runtime()
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The locked resource gate requires an available CUDA device")
    counts = validate_protocol(protocol)
    batch = protocol["resource_gate"]["batch_size"]
    results = {}
    for arm in ARMS:
        seed_training(20260908)
        model = make_model(protocol, arm, device=device)
        patches = torch.randn(batch, 729, 768, dtype=torch.float16, device=device)
        center = torch.randn(batch, 768, dtype=torch.float16, device=device)
        video = torch.randn(batch, 1536, device=device)
        valid = torch.ones(batch, dtype=torch.bool, device=device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        durations = []
        finite_gradients = True
        for iteration in range(protocol["resource_gate"]["iterations"]):
            model.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            values = forward_arm(model, arm, patches, center, video, valid)
            loss = values["logits"].square().mean()
            if arm in TEMPLATE_ARMS:
                loss = (
                    loss
                    + training_loss(
                        values,
                        torch.arange(batch, device=device) % 3,
                        torch.ones(3, device=device),
                        arm,
                        protocol,
                    )[1]["weighted"]
                )
            loss.backward()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if iteration >= protocol["resource_gate"]["warmup_iterations"]:
                durations.append(time.perf_counter() - started)
            finite_gradients &= bool(torch.isfinite(loss)) and all(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        missing = forward_arm(
            model,
            arm,
            torch.full_like(patches[:2], torch.nan),
            torch.full_like(center[:2], torch.nan),
            video[:2],
            torch.zeros(2, dtype=torch.bool, device=device),
        )
        exact_fallback = bool(torch.equal(missing["logits"], missing["video_logits"]))
        peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        results[arm] = {
            "parameters": counts[arm],
            "median_forward_backward_seconds": float(np.median(durations)),
            "peak_cuda_allocated_bytes": peak,
            "finite_active_gradients": finite_gradients,
            "exact_video_fallback": exact_fallback,
            "state_dict_schema": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in model.state_dict().items()
            },
            "trainable_parameter_names": [
                name for name, value in model.named_parameters() if value.requires_grad
            ],
        }
        del model, patches, center, video, values, missing, loss
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    common_schemas = []
    for arm in ARMS:
        model = make_model(protocol, arm)
        common_schemas.append(
            {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in video_module(model, arm).state_dict().items()
            }
        )
    passed = (
        all(v["finite_active_gradients"] and v["exact_video_fallback"] for v in results.values())
        and max(v["peak_cuda_allocated_bytes"] for v in results.values())
        < protocol["resource_gate"]["maximum_peak_cuda_allocated_bytes"]
        and all(schema == common_schemas[0] for schema in common_schemas)
    )
    saved = {
        "status": "SEAR_SIX_ARM_RESOURCE_GATE_PASS" if passed else "SEAR_RESOURCE_GATE_FAIL",
        "passed": passed,
        "synthetic_only": True,
        "action_labels_read": 0,
        "optimizer_steps": 0,
        "batch_size": batch,
        "protocol_sha256": file_sha256(PROTOCOL),
        "code_sha256": code_sha256(),
        "runtime": runtime,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "common_video_state_dict_schema": common_schemas[0],
        "results": results,
    }
    immutable_json(path, saved)
    if not passed:
        raise RuntimeError("SEAR resource gate failed; classifier fitting is forbidden")
    validate_resource_receipt(saved, protocol)
    return saved


def prepare(output: Path, dense_dir: Path, source_data: Path) -> dict:
    protocol = read_json(PROTOCOL)
    counts = validate_protocol(protocol)
    if any((output / name).exists() for name in ("fits", "workloads", "results")):
        raise RuntimeError("SEAR prepare must precede every classifier fit and result artifact")
    if source_data != SOURCE_DATA.resolve():
        raise RuntimeError("SEAR v1 requires the frozen source_swap_v1 data directory")
    resource = read_json(output / "resource_pilot.json")
    validate_resource_receipt(resource, protocol)
    dense_summary_path = dense_dir / "summary.json"
    dense_summary = read_json(dense_summary_path)
    names = (
        "patch_tokens.npy",
        "center_cls.npy",
        "sample_ids.npy",
        "completed.npy",
        "local_valid.npy",
        "source_frames.npy",
        "historical_short_fallback.npy",
        "execution_lock.json",
        "model_receipt.json",
    )
    dense_receipts = {name: _dense_receipt(dense_summary, dense_dir, name) for name in names}
    for entry in dense_receipts.values():
        checked(entry)
    if (
        dense_summary.get("status") != "FULL_DENSE_DINO_CENTER_CACHE_COMPLETE"
        or dense_summary.get("rows") != protocol["primary_rows"]
        or dense_summary.get("completed_rows") != protocol["primary_rows"]
        or dense_summary.get("valid_local_rows") != protocol["primary_rows"]
        or not dense_summary.get("all_primary_rows")
        or not dense_summary.get("historical_cls_replay_exact_all4977")
        or dense_summary.get("center_cls_max_absolute_difference") != 0.0
        or dense_summary.get("classifier_fits") != 0
        or dense_summary.get("backbone_updates") != 0
        or dense_summary.get("execution_lock_sha256")
        != dense_receipts["execution_lock.json"]["sha256"]
    ):
        raise RuntimeError("Dense DINO cache completion contract is invalid")
    source_summary_path = source_data / "summary.json"
    source_summary = read_json(source_summary_path)
    source_memory = source_data / "memory_data.npz"
    if source_summary.get("output_sha256", {}).get("memory_data.npz") != file_sha256(
        source_memory
    ) or source_summary.get("output_sha256", {}).get("source_masks.npz") != file_sha256(
        SOURCE_MASKS
    ):
        raise RuntimeError("Source-swap video input changed")
    result_summary = read_json(SOURCE_RESULT_SUMMARY)
    source_execution_sha = file_sha256(SOURCE_EXECUTION_LOCK)
    if (
        not result_summary.get("complete")
        or result_summary.get("rows") != protocol["primary_rows"]
        or result_summary.get("fresh_memory_fits") != 75
        or result_summary.get("fresh_base_estimator_fits") != 3016
        or result_summary.get("source_swap_memory_execution_sha256") != source_execution_sha
        or result_summary.get("artifacts_sha256", {}).get("oof_probabilities.npz")
        != file_sha256(SOURCE_RESULT_OOF)
    ):
        raise RuntimeError("Completed source-swap reference contract is invalid")
    source_audit = read_json(SOURCE_FINAL_AUDIT)
    if (
        source_audit.get("status") != "INDEPENDENT_SAVED_SOURCE_MATRIX_AUDIT_PASSED"
        or source_audit.get("rows") != protocol["primary_rows"]
        or source_audit.get("verified_memory_receipts") != 75
        or not source_audit.get("new_source_seed_mean_bit_exact")
    ):
        raise RuntimeError("Source-swap reference lacks its independent audit")
    for path, expected in source_audit.get("input_sha256", {}).items():
        current = Path(path)
        if not current.is_file() or file_sha256(current) != expected:
            raise RuntimeError(f"Source-swap audited ancestry changed: {current}")
    replay_receipt = read_json(OLD_SOURCE_REPLAY_RECEIPT)
    if (
        replay_receipt.get("status") != "OLD_SOURCE_P6_AND_ALL75_M4_CHECKPOINTS_BIT_EXACT"
        or replay_receipt.get("rows") != protocol["primary_rows"]
        or replay_receipt.get("verified_m4_fits") != 75
        or replay_receipt.get("maximum_absolute_difference") != 0.0
        or replay_receipt.get("output_sha256") != file_sha256(OLD_SOURCE_REPLAY)
    ):
        raise RuntimeError("Old-source replay reference is invalid")
    legacy_receipt = read_json(LEGACY_STRATA_RECEIPT)
    if (
        legacy_receipt.get("status") != "LEGACY_REQUESTED_WINDOW_COMPARISON_STRATA_REPRODUCED"
        or legacy_receipt.get("rows") != protocol["primary_rows"]
        or legacy_receipt.get("artifact_sha256") != file_sha256(LEGACY_STRATA)
    ):
        raise RuntimeError("Legacy comparison strata are invalid")
    ids = np.load(dense_receipts["sample_ids.npy"]["path"], allow_pickle=False)
    completed = np.load(dense_receipts["completed.npy"]["path"], allow_pickle=False)
    local_valid = np.load(dense_receipts["local_valid.npy"]["path"], allow_pickle=False)
    dense_frames = np.load(dense_receipts["source_frames.npy"]["path"], allow_pickle=False)
    dense_fallback = np.load(
        dense_receipts["historical_short_fallback.npy"]["path"], allow_pickle=False
    )
    with np.load(source_memory, allow_pickle=False) as saved:
        population = {
            key: saved[key].copy()
            for key in (
                "sample_ids",
                "labels",
                "scenarios",
                "folds",
                "recordings",
                "tracks",
                "frames",
                "long_valid",
                "node_support_complete",
                "node_support_boundary",
                "node_interval_complete",
                "node_interval_boundary",
                "node_purity_target",
                "node_purity_valid",
                "quality",
            )
        }
        video = saved["features"][:, :1536].copy()
    if (
        len(ids) != protocol["primary_rows"]
        or not np.array_equal(ids, population["sample_ids"])
        or completed.dtype != np.bool_
        or not completed.all()
        or local_valid.shape != (len(ids),)
        or local_valid.dtype != np.bool_
        or not local_valid.all()
        or not np.array_equal(dense_frames, population["frames"])
        or not np.array_equal(dense_fallback, ~population["long_valid"])
        or video.shape != (len(ids), 1536)
        or video.dtype != np.float32
        or not np.isfinite(video).all()
        or not np.array_equal(np.bincount(population["labels"], minlength=3), [734, 2118, 2125])
        or not np.array_equal(np.unique(population["folds"]), protocol["splitting"]["outer_folds"])
        or any(
            len(np.unique(population["folds"][population["scenarios"] == scenario])) != 1
            for scenario in np.unique(population["scenarios"])
        )
        or protocol["statistics"]["scenario_swaps"] != 2 ** len(np.unique(population["scenarios"]))
    ):
        raise RuntimeError("Dense and source-swap cohorts are not exactly aligned and complete")
    patches = np.load(dense_receipts["patch_tokens.npy"]["path"], mmap_mode="r")
    center = np.load(dense_receipts["center_cls.npy"]["path"], mmap_mode="r")
    if patches.shape != (len(ids), 729, 768) or patches.dtype != np.float16:
        raise RuntimeError("Dense patch cache shape/dtype differs from protocol")
    if center.shape != (len(ids), 768) or center.dtype != np.float16:
        raise RuntimeError("Center CLS cache shape/dtype differs from protocol")
    for start in range(0, len(ids), 64):
        rows = local_valid[start : start + 64]
        if (
            not np.isfinite(patches[start : start + 64][rows]).all()
            or not np.isfinite(center[start : start + 64][rows]).all()
        ):
            raise RuntimeError("Available dense center evidence is nonfinite")
    identity_keys = ("sample_ids", "labels", "folds", "scenarios")
    with np.load(SOURCE_RESULT_OOF, allow_pickle=False) as saved:
        if any(not np.array_equal(saved[key], population[key]) for key in identity_keys):
            raise RuntimeError("Source result references do not match the SEAR population")
        references = {
            key: saved[key].copy()
            for key in ("old_source_p6", "old_source_m4", "new_source_p6", "new_source_m4")
        }
    with np.load(OLD_SOURCE_REPLAY, allow_pickle=False) as saved:
        if any(not np.array_equal(saved[key], population[key]) for key in identity_keys):
            raise RuntimeError("Old-source replay does not match the SEAR population")
        original_shared = (
            saved["old_source_components"].argmax(2) != population["labels"][:, None]
        ).all(1)
        if not np.array_equal(
            saved["old_source_p6"], references["old_source_p6"]
        ) or not np.array_equal(saved["old_source_m4"], references["old_source_m4"]):
            raise RuntimeError("Old-source result references do not replay exactly")
    with np.load(SOURCE_MASKS, allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], ids):
            raise RuntimeError("Source masks do not match the SEAR population")
        source_masks = {key: saved[key].copy() for key in saved.files if key != "sample_ids"}
    if (
        not np.array_equal(source_masks["historical_short_fallback"], ~population["long_valid"])
        or not np.array_equal(
            source_masks["changed_input"],
            population["long_valid"] & ~source_masks["native_source_fallback"],
        )
        or int(
            (
                source_masks["historical_short_fallback"] & source_masks["native_source_fallback"]
            ).sum()
        )
        != 27
    ):
        raise RuntimeError("Source fallback and changed-input masks are inconsistent")
    with np.load(LEGACY_STRATA, allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], ids):
            raise RuntimeError("Legacy strata do not match the SEAR population")
        legacy = {key: saved[key].copy() for key in saved.files if key != "sample_ids"}
    with np.load(EVIDENCE_OOF, allow_pickle=False) as saved:
        if any(not np.array_equal(saved[key], population[key]) for key in identity_keys):
            raise RuntimeError("Historical memory results do not match the SEAR population")
        persistent = np.ones(len(ids), dtype=bool)
        for arm in (
            "temporal_conv",
            "query_attention",
            "survival_memory",
            "corroborated_memory",
        ):
            persistent &= saved[arm].argmax(1) != population["labels"]
    sampled_pure = population["node_purity_valid"] & np.isclose(
        population["node_purity_target"], 1.0
    )
    sampled_mixed = population["node_purity_valid"] & ~sampled_pure
    pure_persistent = persistent & original_shared & sampled_pure
    height = population["quality"][:, 0].astype(np.float64) * 720
    medium_height = (height > 32 + 1e-4) & (height <= 64 + 1e-4)
    expected_counts = {
        "original_shared": (original_shared, 508),
        "historical_short_fallback": (source_masks["historical_short_fallback"], 467),
        "native_source_fallback": (source_masks["native_source_fallback"], 29),
        "changed_input": (source_masks["changed_input"], 4508),
        "legacy_boundary": (legacy["complete_target_boundary"], 671),
        "legacy_stable": (legacy["complete_stable_target_window"], 3730),
        "legacy_unknown": (legacy["unknown_target"], 576),
        "sampled_pure": (sampled_pure, 4184),
        "sampled_mixed": (sampled_mixed, 682),
        "sampled_unknown": (~population["node_purity_valid"], 111),
        "dense_interval_stable": (
            population["node_interval_complete"] & ~population["node_interval_boundary"],
            4184,
        ),
        "dense_interval_boundary": (
            population["node_interval_complete"] & population["node_interval_boundary"],
            681,
        ),
        "dense_interval_unknown": (~population["node_interval_complete"], 112),
        "pure_persistent": (pure_persistent, 266),
        "medium_pure_persistent": (pure_persistent & medium_height, 164),
    }
    if any(int(mask.sum()) != expected for mask, expected in expected_counts.values()):
        raise RuntimeError("Frozen SEAR diagnostic stratum counts changed")
    data_path = output / "population.npz"
    population_arrays = {**population, "video": video, "local_valid": local_valid}
    immutable_npz(data_path, population_arrays)
    evaluation_path = output / "evaluation_inputs.npz"
    evaluation = {
        "sample_ids": ids,
        **references,
        "original_shared_p6_failures": original_shared,
        **source_masks,
        **legacy,
        "sampled_node_pure": sampled_pure,
        "sampled_node_mixed": sampled_mixed,
        "sampled_node_unknown": ~population["node_purity_valid"],
        "dense_interval_stable": population["node_interval_complete"]
        & ~population["node_interval_boundary"],
        "dense_interval_boundary": population["node_interval_complete"]
        & population["node_interval_boundary"],
        "dense_interval_unknown": ~population["node_interval_complete"],
        "pure_persistent_error": pure_persistent,
        "medium_pure_persistent_error": pure_persistent & medium_height,
    }
    immutable_npz(evaluation_path, evaluation)
    provenance_paths = (
        *CODE_PATHS,
        PROTOCOL,
        dense_summary_path,
        source_summary_path,
        source_memory,
        SOURCE_RESULT_SUMMARY,
        SOURCE_RESULT_OOF,
        SOURCE_FINAL_AUDIT,
        SOURCE_EXECUTION_LOCK,
        OLD_SOURCE_REPLAY,
        OLD_SOURCE_REPLAY_RECEIPT,
        SOURCE_MASKS,
        LEGACY_STRATA,
        LEGACY_STRATA_RECEIPT,
        EVIDENCE_OOF,
        EVIDENCE_SUMMARY,
        output / "resource_pilot.json",
        data_path,
        evaluation_path,
    )
    source_files = {str(path.resolve()): artifact(path) for path in provenance_paths}
    lock = {
        "status": "SEAR_SIX_ARM_450_FIT_LOCKED_BEFORE_CLASSIFIER_FITTING",
        "protocol": protocol,
        "parameter_counts": counts,
        "dense_artifacts": dense_receipts,
        "source_artifacts": source_files,
        "sample_ids_sha256": canonical_hash(ids.tolist()),
        "labels_sha256": canonical_hash(population["labels"].tolist()),
        "folds_sha256": canonical_hash(population["folds"].tolist()),
        "scenarios_sha256": canonical_hash(population["scenarios"].tolist()),
        "runtime": resource["runtime"],
        "device": resource["device"],
        "gpu": resource["gpu"],
        "diagnostic_stratum_counts": {
            name: int(mask.sum()) for name, (mask, _expected) in expected_counts.items()
        },
        "classifier_fits_completed_at_lock": 0,
        "protected_rows_read": 0,
    }
    immutable_json(output / "execution_lock.json", lock)
    return lock


def load_locked(output: Path):
    lock = read_json(output / "execution_lock.json")
    if lock["status"] != "SEAR_SIX_ARM_450_FIT_LOCKED_BEFORE_CLASSIFIER_FITTING":
        raise RuntimeError("SEAR execution lock is invalid")
    for entry in [*lock["dense_artifacts"].values(), *lock["source_artifacts"].values()]:
        checked(entry)
    current_code = code_sha256()
    if lock.get("runtime") != configure_runtime() or current_code != {
        path: entry["sha256"]
        for path, entry in lock["source_artifacts"].items()
        if path in current_code
    }:
        raise RuntimeError("SEAR runtime or source code changed after execution lock")
    with np.load(output / "population.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    if (
        canonical_hash(data["sample_ids"].tolist()) != lock["sample_ids_sha256"]
        or canonical_hash(data["labels"].tolist()) != lock["labels_sha256"]
        or canonical_hash(data["folds"].tolist()) != lock["folds_sha256"]
        or canonical_hash(data["scenarios"].tolist()) != lock["scenarios_sha256"]
    ):
        raise RuntimeError("SEAR locked population identity changed")
    patches = np.load(lock["dense_artifacts"]["patch_tokens.npy"]["path"], mmap_mode="r")
    center = np.load(lock["dense_artifacts"]["center_cls.npy"]["path"], mmap_mode="r")
    return lock, patches, center, data


def batch_inputs(patches, center, data, rows, device):
    return (
        torch.from_numpy(np.asarray(patches[rows])).to(device),
        torch.from_numpy(np.asarray(center[rows])).to(device),
        torch.from_numpy(data["video"][rows]).to(device),
        torch.from_numpy(data["local_valid"][rows]).to(device),
    )


def build_fit_request(
    data,
    arm,
    train,
    held,
    learning_rate,
    weight_decay,
    seed,
    device,
    execution_lock_sha256,
    *,
    outer_fold,
    inner_fold=None,
    epochs=None,
):
    """Bind training labels always and selection labels only for inner fits."""
    return {
        "execution_lock_sha256": execution_lock_sha256,
        "arm": arm,
        "outer_fold": int(outer_fold),
        "inner_fold": None if inner_fold is None else int(inner_fold),
        "stage": "inner" if epochs is None else "outer_refit",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "epochs": epochs,
        "train_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "selection_labels_sha256": canonical_hash(data["labels"][held].tolist())
        if epochs is None
        else None,
        "device": device,
    }


def _diagnostic_coordinates(protocol, count, device, dtype):
    coordinates = torch.stack(
        torch.meshgrid(
            (torch.arange(int(np.sqrt(count)), device=device, dtype=dtype) + 0.5)
            * (2 / int(np.sqrt(count)))
            - 1,
            (torch.arange(int(np.sqrt(count)), device=device, dtype=dtype) + 0.5)
            * (2 / int(np.sqrt(count)))
            - 1,
            indexing="ij",
        ),
        dim=-1,
    )[..., [1, 0]].reshape(count, 2)
    permutation = np.random.default_rng(
        protocol["fixed_diagnostics"]["coordinate_shuffle_seed"]
    ).permutation(count)
    return coordinates[torch.as_tensor(permutation, device=device)]


def _outer_border_matchability(protocol, local_valid, count, dtype):
    side = int(np.sqrt(count))
    if side * side != count:
        raise RuntimeError("Fixed SEAR image-context mask requires a square patch grid")
    margin = protocol["fixed_diagnostics"]["outer_border_mask_patches"]
    if margin < 1 or 2 * margin >= side:
        raise RuntimeError("Fixed SEAR outer-border diagnostic is invalid")
    mask = torch.zeros((side, side), dtype=dtype, device=local_valid.device)
    mask[margin:-margin, margin:-margin] = 1
    return local_valid[:, None].to(dtype) * mask.flatten()[None]


def _softmax_numpy(logits):
    shifted = logits - logits.max(1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(1, keepdims=True)


@torch.no_grad()
def predict(model, arm, patches, center, data, rows, protocol, device, *, condition="original"):
    model.eval()
    if condition not in ("original", "coordinate_shuffle", "outer_border_mask"):
        raise ValueError("Unknown fixed SEAR inference diagnostic")
    collected = {"probabilities": []}
    if condition == "original":
        collected.update({"local_logits": [], "video_logits": []})
        if arm in TEMPLATE_ARMS:
            collected.update({"slot_masses": [], "slot_positions": [], "slot_descriptors": []})
    batch = protocol["training"]["batch_size"]
    for start in range(0, len(rows), batch):
        selected = rows[start : start + batch]
        values = batch_inputs(patches, center, data, selected, device)
        kwargs = {}
        if condition == "coordinate_shuffle":
            kwargs["coordinates"] = _diagnostic_coordinates(
                protocol, values[0].shape[1], values[0].device, torch.float32
            )
        elif condition == "outer_border_mask":
            kwargs["patch_matchability"] = _outer_border_matchability(
                protocol, values[3], values[0].shape[1], torch.float32
            )
        result = forward_arm(model, arm, *values, **kwargs)
        for name in collected:
            collected[name].append(result[name].detach().cpu().numpy())
    return {name: np.concatenate(values) for name, values in collected.items()}


def load_completed_fit(receipt_path, request, held_ids, expected_parameters):
    receipt = read_json(receipt_path)
    directory = receipt_path.parent
    if (
        receipt.get("status") != "SEAR_FIT_COMPLETE"
        or receipt.get("request") != request
        or directory.name != canonical_hash(request)
        or receipt.get("parameters") != expected_parameters
        or set(receipt.get("artifacts", {}))
        != {"request.json", "checkpoint.pt", "predictions.npz", "progress.pt"}
    ):
        raise RuntimeError(f"Retained SEAR fit receipt changed: {receipt_path}")
    for entry in receipt["artifacts"].values():
        checked(entry)
    expected_files = {*receipt["artifacts"], "receipt.json"}
    if {path.name for path in directory.iterdir()} != expected_files:
        raise RuntimeError("Retained SEAR fit directory inventory changed")
    if read_json(directory / "request.json") != request:
        raise RuntimeError("Retained SEAR fit request artifact changed")
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("request") != request or checkpoint.get("epoch") != receipt["selected_epoch"]:
        raise RuntimeError("Retained SEAR checkpoint ancestry changed")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], held_ids):
            raise RuntimeError("Retained SEAR fit prediction identity changed")
        predicted = {key: saved[key].copy() for key in saved.files if key != "sample_ids"}
    expected = {"probabilities", "local_logits", "video_logits"}
    if request["arm"] in TEMPLATE_ARMS:
        expected.update({"slot_masses", "slot_positions", "slot_descriptors"})
    if request["stage"] == "outer_refit":
        expected.update(
            {
                "coordinate_shuffle_probabilities",
                "outer_border_mask_probabilities",
                "video_removed_probabilities",
            }
        )
    if set(predicted) != expected:
        raise RuntimeError("Retained SEAR fit prediction schema changed")
    for name, values in predicted.items():
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise RuntimeError(f"Retained SEAR fit contains invalid {name}")
    for name in ("local_logits", "video_logits"):
        if predicted[name].shape != (len(held_ids), 3):
            raise RuntimeError("Retained SEAR fit logit shape changed")
    for name in (key for key in predicted if key.endswith("probabilities")):
        values = predicted[name]
        if (
            values.shape != (len(held_ids), 3)
            or (values < 0).any()
            or not np.allclose(values.sum(1), 1, atol=1e-6, rtol=0)
        ):
            raise RuntimeError("Retained SEAR fit probability shape/simplex changed")
    if request["arm"] in TEMPLATE_ARMS and (
        predicted["slot_masses"].shape != (len(held_ids), 6)
        or predicted["slot_positions"].shape != (len(held_ids), 6, 2)
        or predicted["slot_descriptors"].shape != (len(held_ids), 6, 32)
    ):
        raise RuntimeError("Retained SEAR slot-evidence shape changed")
    return predicted, receipt


def fit_model(
    output,
    lock,
    patches,
    center,
    data,
    arm,
    train,
    held,
    learning_rate,
    weight_decay,
    seed,
    device,
    *,
    outer_fold,
    inner_fold=None,
    epochs=None,
):
    protocol = lock["protocol"]
    request = build_fit_request(
        data,
        arm,
        train,
        held,
        learning_rate,
        weight_decay,
        seed,
        device,
        file_sha256(output / "execution_lock.json"),
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        epochs=epochs,
    )
    directory = output / "fits" / canonical_hash(request)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        return load_completed_fit(
            receipt_path, request, data["sample_ids"][held], lock["parameter_counts"][arm]
        )
    allowed_partial = {
        "request.json",
        "progress.pt",
        "progress.tmp",
        "checkpoint.pt",
        "predictions.npz",
    }
    if directory.exists() and any(path.name not in allowed_partial for path in directory.iterdir()):
        raise RuntimeError(f"Unrecognized partial SEAR fit retained without overwrite: {directory}")
    if set(data["scenarios"][train]) & set(data["scenarios"][held]):
        raise RuntimeError("SEAR fit and held scenarios overlap")
    directory.mkdir(parents=True, exist_ok=True)
    immutable_json(directory / "request.json", request)
    seed_training(seed)
    model = make_model(protocol, arm, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    counts = np.bincount(data["labels"][train], minlength=3)
    if np.any(counts == 0):
        raise RuntimeError("SEAR training partition lacks a class")
    class_weights = torch.tensor(len(train) / (3 * counts), dtype=torch.float32, device=device)
    maximum = epochs or protocol["training"]["max_epochs"]
    generator = np.random.default_rng(seed)
    best_key = None
    best_epoch = 0
    best_state = None
    history = []
    first_epoch = 1
    elapsed_before = 0.0
    progress_path = directory / "progress.pt"
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("request") != request or progress.get("maximum_epochs") != maximum:
            raise RuntimeError("Retained SEAR progress request changed")
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        history = progress["history"]
        best_key = progress["best_key"]
        best_epoch = progress["best_epoch"]
        best_state = progress["best_state"]
        generator.bit_generator.state = progress["numpy_generator_state"]
        torch.set_rng_state(progress["torch_rng_state"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(progress["cuda_rng_state"])
        first_epoch = progress["next_epoch"]
        elapsed_before = float(progress.get("elapsed_seconds_total", 0.0))
        if first_epoch != len(history) + 1 or not 1 <= first_epoch <= maximum + 1:
            raise RuntimeError("Retained SEAR epoch progress is inconsistent")
        if (
            epochs is None
            and history
            and history[-1]["epoch"] - best_epoch >= protocol["training"]["early_stopping_patience"]
        ):
            first_epoch = maximum + 1
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(first_epoch, maximum + 1):
        model.train()
        order = generator.permutation(train)
        totals = {"loss": 0.0, "classification": 0.0, "rows": 0}
        for begin in range(0, len(order), protocol["training"]["batch_size"]):
            selected = order[begin : begin + protocol["training"]["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            values = forward_arm(model, arm, *batch_inputs(patches, center, data, selected, device))
            labels = torch.from_numpy(data["labels"][selected]).to(device)
            loss, pieces = training_loss(values, labels, class_weights, arm, protocol)
            if not torch.isfinite(loss):
                raise RuntimeError("SEAR training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                protocol["training"]["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            totals["loss"] += float(loss.detach()) * len(selected)
            totals["classification"] += float(pieces["classification"].detach()) * len(selected)
            totals["rows"] += len(selected)
        row = {
            "epoch": epoch,
            "training_loss": totals["loss"] / totals["rows"],
            "classification_loss": totals["classification"] / totals["rows"],
        }
        if epochs is None:
            values = predict(model, arm, patches, center, data, held, protocol, device)
            measured = probability_metrics(data["labels"][held], values["probabilities"])
            row["validation_metrics"] = measured
            key = (-measured["macro_f1"], measured["nll"], epoch)
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
        history.append(row)
        progress = {
            "request": request,
            "maximum_epochs": maximum,
            "next_epoch": epoch + 1,
            "model_state": {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_key": best_key,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "numpy_generator_state": generator.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            "elapsed_seconds_total": elapsed_before + time.perf_counter() - started,
        }
        temporary = directory / "progress.tmp"
        torch.save(progress, temporary)
        os.replace(temporary, progress_path)
        event(
            "sear_epoch",
            arm=arm,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            epoch=epoch,
            maximum=maximum,
            training_loss=row["training_loss"],
            elapsed_seconds=time.perf_counter() - started,
        )
        if epochs is None and epoch - best_epoch >= protocol["training"]["early_stopping_patience"]:
            break
    if epochs is not None:
        best_epoch = maximum
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
    if best_state is None:
        raise RuntimeError("SEAR fit produced no selected state")
    model.load_state_dict(best_state)
    predicted = predict(model, arm, patches, center, data, held, protocol, device)
    if epochs is not None:
        for condition in ("coordinate_shuffle", "outer_border_mask"):
            diagnostic = predict(
                model, arm, patches, center, data, held, protocol, device, condition=condition
            )
            predicted[condition + "_probabilities"] = diagnostic["probabilities"]
        predicted["video_removed_probabilities"] = _softmax_numpy(predicted["local_logits"])
    validate_or_write_checkpoint(
        directory / "checkpoint.pt",
        {"state_dict": best_state, "request": request, "epoch": best_epoch},
    )
    immutable_npz(
        directory / "predictions.npz",
        {"sample_ids": data["sample_ids"][held], **predicted},
    )
    receipt = {
        "status": "SEAR_FIT_COMPLETE",
        "request": request,
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "train_rows": len(train),
        "held_rows": len(held),
        "parameters": model.trainable_parameters,
        "seconds": elapsed_before + time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else 0,
        "artifacts": {
            name: artifact(directory / name)
            for name in ("request.json", "checkpoint.pt", "predictions.npz", "progress.pt")
        },
    }
    if epochs is None:
        receipt["held_metrics"] = probability_metrics(
            data["labels"][held], predicted["probabilities"]
        )
    else:
        receipt["outer_held_metrics_embargoed"] = True
    immutable_json(receipt_path, receipt)
    event(
        "sear_fit_complete",
        arm=arm,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        selected_epoch=best_epoch,
        seconds=receipt["seconds"],
    )
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return predicted, receipt


def inner_splits(data, outer_train, protocol):
    splitter = StratifiedGroupKFold(
        n_splits=protocol["splitting"]["inner_folds"],
        shuffle=True,
        random_state=protocol["splitting"]["split_seed"],
    )
    return [
        (outer_train[fit], outer_train[held])
        for fit, held in splitter.split(
            np.zeros(len(outer_train)),
            data["labels"][outer_train],
            data["scenarios"][outer_train],
        )
    ]


def run_workload(output, lock, patches, center, data, arm, fold, device):
    protocol = lock["protocol"]
    directory = output / "workloads" / arm / f"fold-{fold}"
    receipt_path = directory / "receipt.json"
    allowed_partial = {"predictions.npz"}
    if directory.exists() and any(path.name not in allowed_partial for path in directory.iterdir()):
        if not receipt_path.exists():
            raise RuntimeError(f"Partial SEAR workload retained without overwrite: {directory}")
    outer_train = np.flatnonzero(data["folds"] != fold)
    outer_held = np.flatnonzero(data["folds"] == fold)
    request = {
        "execution_lock_sha256": file_sha256(output / "execution_lock.json"),
        "arm": arm,
        "fold": int(fold),
    }
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["request"] != request:
            raise RuntimeError("SEAR workload request changed")
        checked(receipt["artifact"])
        if len(receipt["fit_receipts"]) != 15:
            raise RuntimeError("SEAR workload fit inventory changed")
        for entry in receipt["fit_receipts"]:
            checked(entry)
        return receipt
    splits = inner_splits(data, outer_train, protocol)
    candidates = []
    fit_receipts = []
    for learning_rate in protocol["training"]["learning_rates"]:
        for weight_decay in protocol["training"]["weight_decays"]:
            oof = np.full((len(data["labels"]), 3), np.nan)
            selected_epochs = []
            for inner, (train, held) in enumerate(splits):
                predicted, receipt = fit_model(
                    output,
                    lock,
                    patches,
                    center,
                    data,
                    arm,
                    train,
                    held,
                    learning_rate,
                    weight_decay,
                    protocol["training"]["inner_seed"],
                    device,
                    outer_fold=fold,
                    inner_fold=inner,
                )
                oof[held] = predicted["probabilities"]
                selected_epochs.append(receipt["selected_epoch"])
                fit_receipts.append(
                    artifact(output / "fits" / canonical_hash(receipt["request"]) / "receipt.json")
                )
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "inner_metrics": probability_metrics(
                        data["labels"][outer_train], oof[outer_train]
                    ),
                    "inner_best_epochs": selected_epochs,
                }
            )
    selected = min(
        candidates,
        key=lambda item: (
            -item["inner_metrics"]["macro_f1"],
            item["inner_metrics"]["nll"],
            item["learning_rate"],
            item["weight_decay"],
        ),
    )
    epochs = max(1, int(np.rint(np.median(selected["inner_best_epochs"]))))
    seed_outputs = []
    for seed in protocol["training"]["outer_seeds"]:
        predicted, receipt = fit_model(
            output,
            lock,
            patches,
            center,
            data,
            arm,
            outer_train,
            outer_held,
            selected["learning_rate"],
            selected["weight_decay"],
            seed,
            device,
            outer_fold=fold,
            epochs=epochs,
        )
        seed_outputs.append(predicted)
        fit_receipts.append(
            artifact(output / "fits" / canonical_hash(receipt["request"]) / "receipt.json")
        )
    arrays = {
        "sample_ids": data["sample_ids"][outer_held],
        "seed_probabilities": np.stack([value["probabilities"] for value in seed_outputs]),
        "seed_local_logits": np.stack([value["local_logits"] for value in seed_outputs]),
        "seed_video_logits": np.stack([value["video_logits"] for value in seed_outputs]),
        "seed_coordinate_shuffle_probabilities": np.stack(
            [value["coordinate_shuffle_probabilities"] for value in seed_outputs]
        ),
        "seed_outer_border_mask_probabilities": np.stack(
            [value["outer_border_mask_probabilities"] for value in seed_outputs]
        ),
        "seed_video_removed_probabilities": np.stack(
            [value["video_removed_probabilities"] for value in seed_outputs]
        ),
    }
    if arm in TEMPLATE_ARMS:
        for name in ("slot_masses", "slot_positions", "slot_descriptors"):
            arrays["seed_" + name] = np.stack([value[name] for value in seed_outputs])
    if len(fit_receipts) != 15:
        raise RuntimeError("SEAR workload did not execute exactly fifteen fits")
    directory.mkdir(parents=True, exist_ok=True)
    immutable_npz(directory / "predictions.npz", arrays)
    receipt = {
        "status": "SEAR_WORKLOAD_COMPLETE",
        "request": request,
        "fit_device": device,
        "selected": selected,
        "outer_refit_epochs": epochs,
        "candidates": candidates,
        "unique_fits": len(fit_receipts),
        "fit_receipts": fit_receipts,
        "artifact": artifact(directory / "predictions.npz"),
    }
    immutable_json(receipt_path, receipt)
    event("sear_workload_complete", arm=arm, fold=fold, epochs=epochs)
    return receipt


def verify_workload(output, lock, data, arm, fold):
    """Reconstruct one workload exclusively from its immutable fit evidence."""
    protocol = lock["protocol"]
    lock_sha = file_sha256(output / "execution_lock.json")
    directory = output / "workloads" / arm / f"fold-{fold}"
    receipt = read_json(directory / "receipt.json")
    if {path.name for path in directory.iterdir()} != {"receipt.json", "predictions.npz"}:
        raise RuntimeError("SEAR workload directory inventory changed")
    expected_workload_request = {
        "execution_lock_sha256": lock_sha,
        "arm": arm,
        "fold": int(fold),
    }
    if (
        receipt.get("status") != "SEAR_WORKLOAD_COMPLETE"
        or receipt.get("request") != expected_workload_request
        or receipt.get("unique_fits") != 15
        or len(receipt.get("fit_receipts", ())) != 15
    ):
        raise RuntimeError("SEAR workload receipt ancestry is incomplete")
    outer_train = np.flatnonzero(data["folds"] != fold)
    outer_held = np.flatnonzero(data["folds"] == fold)
    splits = inner_splits(data, outer_train, protocol)
    candidates = []
    expected_receipts = []
    for learning_rate in protocol["training"]["learning_rates"]:
        for weight_decay in protocol["training"]["weight_decays"]:
            oof = np.full((len(data["labels"]), 3), np.nan)
            selected_epochs = []
            for inner, (train, held) in enumerate(splits):
                request = build_fit_request(
                    data,
                    arm,
                    train,
                    held,
                    learning_rate,
                    weight_decay,
                    protocol["training"]["inner_seed"],
                    receipt["fit_device"],
                    lock_sha,
                    outer_fold=fold,
                    inner_fold=inner,
                )
                receipt_path = output / "fits" / canonical_hash(request) / "receipt.json"
                predicted, fit_receipt = load_completed_fit(
                    receipt_path,
                    request,
                    data["sample_ids"][held],
                    lock["parameter_counts"][arm],
                )
                measured = probability_metrics(data["labels"][held], predicted["probabilities"])
                if fit_receipt.get("held_metrics") != measured:
                    raise RuntimeError("SEAR inner held metrics do not replay")
                oof[held] = predicted["probabilities"]
                selected_epochs.append(fit_receipt["selected_epoch"])
                expected_receipts.append(artifact(receipt_path))
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "inner_metrics": probability_metrics(
                        data["labels"][outer_train], oof[outer_train]
                    ),
                    "inner_best_epochs": selected_epochs,
                }
            )
    selected = min(
        candidates,
        key=lambda item: (
            -item["inner_metrics"]["macro_f1"],
            item["inner_metrics"]["nll"],
            item["learning_rate"],
            item["weight_decay"],
        ),
    )
    epochs = max(1, int(np.rint(np.median(selected["inner_best_epochs"]))))
    seed_outputs = []
    for seed in protocol["training"]["outer_seeds"]:
        request = build_fit_request(
            data,
            arm,
            outer_train,
            outer_held,
            selected["learning_rate"],
            selected["weight_decay"],
            seed,
            receipt["fit_device"],
            lock_sha,
            outer_fold=fold,
            epochs=epochs,
        )
        receipt_path = output / "fits" / canonical_hash(request) / "receipt.json"
        predicted, fit_receipt = load_completed_fit(
            receipt_path,
            request,
            data["sample_ids"][outer_held],
            lock["parameter_counts"][arm],
        )
        if (
            fit_receipt.get("selected_epoch") != epochs
            or fit_receipt.get("outer_held_metrics_embargoed") is not True
            or "held_metrics" in fit_receipt
        ):
            raise RuntimeError("SEAR outer-refit epoch or metric embargo changed")
        seed_outputs.append(predicted)
        expected_receipts.append(artifact(receipt_path))
    if (
        receipt.get("candidates") != candidates
        or receipt.get("selected") != selected
        or receipt.get("outer_refit_epochs") != epochs
        or receipt.get("fit_receipts") != expected_receipts
    ):
        raise RuntimeError("SEAR workload selection does not reconstruct")
    expected_arrays = {
        "sample_ids": data["sample_ids"][outer_held],
        "seed_probabilities": np.stack([value["probabilities"] for value in seed_outputs]),
        "seed_local_logits": np.stack([value["local_logits"] for value in seed_outputs]),
        "seed_video_logits": np.stack([value["video_logits"] for value in seed_outputs]),
        "seed_coordinate_shuffle_probabilities": np.stack(
            [value["coordinate_shuffle_probabilities"] for value in seed_outputs]
        ),
        "seed_outer_border_mask_probabilities": np.stack(
            [value["outer_border_mask_probabilities"] for value in seed_outputs]
        ),
        "seed_video_removed_probabilities": np.stack(
            [value["video_removed_probabilities"] for value in seed_outputs]
        ),
    }
    if arm in TEMPLATE_ARMS:
        for name in ("slot_masses", "slot_positions", "slot_descriptors"):
            expected_arrays["seed_" + name] = np.stack([value[name] for value in seed_outputs])
    prediction_path = checked(receipt["artifact"])
    with np.load(prediction_path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected_arrays) or any(
            not arrays_equal(saved[name], values) for name, values in expected_arrays.items()
        ):
            raise RuntimeError("SEAR workload arrays do not match originating outer fits")
    return expected_arrays, {Path(entry["path"]) for entry in expected_receipts}


def summarize(output, lock, data):
    protocol = lock["protocol"]
    seed_count = len(protocol["training"]["outer_seeds"])
    rows = len(data["labels"])
    seeds = {arm: np.full((seed_count, rows, 3), np.nan) for arm in ARMS}
    diagnostic_seeds = {
        condition: {arm: np.full((seed_count, rows, 3), np.nan) for arm in ARMS}
        for condition in ("coordinate_shuffle", "outer_border_mask", "video_removed")
    }
    slot_values = {
        arm: {
            "masses": np.full((seed_count, rows, 6), np.nan, dtype=np.float32),
            "positions": np.full((seed_count, rows, 6, 2), np.nan, dtype=np.float32),
            "descriptors": np.full((seed_count, rows, 6, 32), np.nan, dtype=np.float32),
        }
        for arm in TEMPLATE_ARMS
    }
    expected_fit_receipts = set()
    workload_receipts = {}
    for arm in ARMS:
        for fold in protocol["splitting"]["outer_folds"]:
            arrays, fit_receipts = verify_workload(output, lock, data, arm, fold)
            if expected_fit_receipts & fit_receipts:
                raise RuntimeError("A SEAR fit was reused across declared workloads")
            expected_fit_receipts |= fit_receipts
            held = np.flatnonzero(data["folds"] == fold)
            seeds[arm][:, held] = arrays["seed_probabilities"]
            for condition in diagnostic_seeds:
                diagnostic_seeds[condition][arm][:, held] = arrays[
                    "seed_" + condition + "_probabilities"
                ]
            if arm in TEMPLATE_ARMS:
                for name in slot_values[arm]:
                    slot_values[arm][name][:, held] = arrays["seed_slot_" + name]
            receipt_path = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            workload_receipts[str(receipt_path.resolve())] = file_sha256(receipt_path)
    fit_count = len(expected_fit_receipts)
    actual_fit_receipts = {path.resolve() for path in (output / "fits").glob("*/receipt.json")}
    actual_fit_directories = {
        path.resolve() for path in (output / "fits").iterdir() if path.is_dir()
    }
    actual_workloads = {
        path.resolve() for path in (output / "workloads").glob("*/fold-*/receipt.json")
    }
    actual_workload_directories = {
        path.resolve() for path in (output / "workloads").glob("*/fold-*") if path.is_dir()
    }
    expected_workloads = {Path(path).resolve() for path in workload_receipts}
    if (
        fit_count != protocol["training"]["maximum_classifier_fits"]
        or actual_fit_receipts != {path.resolve() for path in expected_fit_receipts}
        or actual_fit_directories != {path.resolve().parent for path in expected_fit_receipts}
        or actual_workloads != expected_workloads
        or actual_workload_directories != {path.parent for path in expected_workloads}
        or any(not np.isfinite(values).all() for values in seeds.values())
        or any(
            not np.isfinite(values).all()
            for condition in diagnostic_seeds.values()
            for values in condition.values()
        )
        or any(
            not np.isfinite(values).all()
            for arm_values in slot_values.values()
            for values in arm_values.values()
        )
    ):
        raise RuntimeError("SEAR exact 450-fit matrix is incomplete")
    probabilities = {arm: values.mean(0) for arm, values in seeds.items()}
    diagnostics = {
        condition: {arm: values.mean(0) for arm, values in arm_values.items()}
        for condition, arm_values in diagnostic_seeds.items()
    }
    labels = data["labels"]
    results = {arm: probability_metrics(labels, values) for arm, values in probabilities.items()}
    comparisons = {}
    pvalues = {}
    for candidate, reference in protocol["statistics"]["directional_contrasts"]:
        name = candidate + "_vs_" + reference
        value = paired_statistics(
            labels,
            probabilities[candidate],
            probabilities[reference],
            data["scenarios"],
            bootstrap_resamples=protocol["statistics"]["bootstrap_resamples"],
            bootstrap_seed=protocol["statistics"]["seed"],
        )
        comparisons[name] = value
        pvalues[name] = value["one_sided_exact_swap_pvalue"]
    adjusted = holm_adjust(pvalues)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_pvalue"] = value
    scenario_metrics = {
        arm: {
            str(group): probability_metrics(
                labels[data["scenarios"] == group], values[data["scenarios"] == group]
            )
            for group in np.unique(data["scenarios"])
        }
        for arm, values in probabilities.items()
    }
    evaluation_path = output / "evaluation_inputs.npz"
    with np.load(evaluation_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], data["sample_ids"]):
            raise RuntimeError("SEAR evaluation input identity changed")
        evaluation = {name: saved[name].copy() for name in saved.files if name != "sample_ids"}
    reference_names = ("old_source_p6", "old_source_m4", "new_source_p6", "new_source_m4")
    references = {name: evaluation[name] for name in reference_names}
    source_reference_results = {
        name: probability_metrics(labels, values) for name, values in references.items()
    }
    strata = {
        "all": np.ones(rows, dtype=bool),
        "historical_short_fallback": evaluation["historical_short_fallback"],
        "native_source_fallback": evaluation["native_source_fallback"],
        "changed_source_input": evaluation["changed_input"],
        "unchanged_source_input": ~evaluation["changed_input"],
        "legacy_boundary": evaluation["complete_target_boundary"],
        "legacy_stable": evaluation["complete_stable_target_window"],
        "legacy_unknown": evaluation["unknown_target"],
        "sampled_node_pure": evaluation["sampled_node_pure"],
        "sampled_node_mixed": evaluation["sampled_node_mixed"],
        "sampled_node_unknown": evaluation["sampled_node_unknown"],
        "dense_interval_stable": evaluation["dense_interval_stable"],
        "dense_interval_boundary": evaluation["dense_interval_boundary"],
        "dense_interval_unknown": evaluation["dense_interval_unknown"],
        "pure_persistent_error": evaluation["pure_persistent_error"],
        "medium_pure_persistent_error": evaluation["medium_pure_persistent_error"],
    }
    height = data["quality"][:, 0].astype(np.float64) * 720
    strata.update(
        {
            "native_height_le32": height <= 32 + 1e-4,
            "native_height_32to64": (height > 32 + 1e-4) & (height <= 64 + 1e-4),
            "native_height_gt64": height > 64 + 1e-4,
        }
    )
    for scenario in np.unique(data["scenarios"]):
        strata["scenario_" + str(scenario)] = data["scenarios"] == scenario
    for label, name in enumerate(protocol["class_order"]):
        strata["class_" + name] = labels == label
    reference_diagnostics = {
        arm: reference_strata_summary(
            labels,
            values,
            references,
            strata,
            evaluation["original_shared_p6_failures"],
            candidate_name=arm,
        )
        for arm, values in probabilities.items()
    }
    fixed_inference_diagnostics = {
        arm: {
            condition: {
                "metrics": probability_metrics(labels, diagnostics[condition][arm]),
                "vs_unperturbed": reference_strata_summary(
                    labels,
                    diagnostics[condition][arm],
                    {"unperturbed": probabilities[arm]},
                    {"all": strata["all"]},
                    evaluation["original_shared_p6_failures"],
                    candidate_name=arm + "_" + condition,
                )["references"]["unperturbed"]["strata"]["all"],
            }
            for condition in diagnostics
        }
        for arm in ARMS
    }
    slot_reports = {
        arm: slot_diagnostics(
            values["masses"],
            seed_positions=values["positions"],
            seed_descriptors=values["descriptors"],
            local_valid=data["local_valid"],
            seed_ids=protocol["training"]["outer_seeds"],
            active_share_threshold=protocol["fixed_diagnostics"]["slot_active_share_threshold"],
            descriptor_cosine_threshold=protocol["fixed_diagnostics"][
                "slot_descriptor_cosine_threshold"
            ],
            position_distance_threshold=protocol["fixed_diagnostics"][
                "slot_position_distance_threshold"
            ],
            position_space="normalized DINO patch-center coordinates [-1,1]",
        )
        for arm, values in slot_values.items()
    }
    primary = protocol["primary_arm"]
    primary_reference = reference_diagnostics[primary]["references"]["new_source_m4"]["strata"]
    scenario_accuracy_deltas = {
        str(scenario): scenario_metrics[primary][str(scenario)]["accuracy"]
        - probability_metrics(
            labels[data["scenarios"] == scenario],
            references["new_source_m4"][data["scenarios"] == scenario],
        )["accuracy"]
        for scenario in np.unique(data["scenarios"])
    }
    decision_gates = {
        "macro_f1_at_least_0_84": results[primary]["macro_f1"] >= 0.84,
        "macro_f1_at_least_0_85": results[primary]["macro_f1"] >= 0.85,
        "beats_a1_dense_cnn_by_at_least_0_5_points": results[primary]["macro_f1"]
        - results["a1_dense_cnn"]["macro_f1"]
        >= 0.005,
        "beats_a2_dense_transformer_by_at_least_0_5_points": results[primary]["macro_f1"]
        - results["a2_dense_transformer"]["macro_f1"]
        >= 0.005,
        "no_nll_regression_vs_best_ordinary_dense": results[primary]["nll"]
        <= min(results["a1_dense_cnn"]["nll"], results["a2_dense_transformer"]["nll"]),
        "no_brier_regression_vs_best_ordinary_dense": results[primary]["brier"]
        <= min(results["a1_dense_cnn"]["brier"], results["a2_dense_transformer"]["brier"]),
        "positive_net_corrections_vs_new_source_m4": primary_reference["all"]["net_corrections"]
        > 0,
        "scenario_accuracy_improves_at_least_7_of_11": sum(
            value > 0 for value in scenario_accuracy_deltas.values()
        )
        >= 7,
        "no_scenario_accuracy_loss_over_2_points": min(scenario_accuracy_deltas.values()) >= -0.02,
        "legacy_boundary_harms_fewer_than_historical_73_vs_old_source_p6": reference_diagnostics[
            primary
        ]["references"]["old_source_p6"]["strata"]["legacy_boundary"]["harms"]
        < 73,
        "repairs_at_least_one_pure_persistent_error": primary_reference["pure_persistent_error"][
            "original_shared_repairs"
        ]
        > 0,
    }
    result_dir = output / "results/v0001"
    result_dir.mkdir(parents=True, exist_ok=True)
    allowed_result_files = {"oof_probabilities.npz", "slot_evidence.npz", "summary.json"}
    if any(path.name not in allowed_result_files for path in result_dir.iterdir()):
        raise RuntimeError("SEAR result directory contains an undeclared artifact")
    oof_arrays = {
        "sample_ids": data["sample_ids"],
        "labels": labels,
        "scenarios": data["scenarios"],
        "folds": data["folds"],
        **references,
        **probabilities,
        **{arm + "_seeds": values for arm, values in seeds.items()},
        **{
            arm + "_" + condition: diagnostics[condition][arm]
            for condition in diagnostics
            for arm in ARMS
        },
    }
    immutable_npz(result_dir / "oof_probabilities.npz", oof_arrays)
    slot_arrays = {
        "sample_ids": data["sample_ids"],
        "outer_seeds": np.asarray(protocol["training"]["outer_seeds"], dtype=np.int64),
        **{
            arm + "_seed_slot_" + name: values
            for arm, arm_values in slot_values.items()
            for name, values in arm_values.items()
        },
    }
    immutable_npz(result_dir / "slot_evidence.npz", slot_arrays)
    summary = {
        "status": "SEAR_SIX_ARM_450_FIT_COMPLETE",
        "scope": protocol["scope"],
        "rows": rows,
        "classifier_fits": fit_count,
        "execution_lock_sha256": file_sha256(output / "execution_lock.json"),
        "workload_receipt_sha256": workload_receipts,
        "results": results,
        "source_reference_results": source_reference_results,
        "seed_metrics": {
            arm: [probability_metrics(labels, values) for values in arm_seeds]
            for arm, arm_seeds in seeds.items()
        },
        "scenario_metrics": scenario_metrics,
        "scenario_accuracy_delta_vs_new_source_m4": scenario_accuracy_deltas,
        "comparisons": comparisons,
        "reference_diagnostics": reference_diagnostics,
        "fixed_inference_diagnostics": fixed_inference_diagnostics,
        "slot_diagnostics": slot_reports,
        "decision_gates": decision_gates,
        "primary_arm": primary,
        "primary_macro_f1": results[primary]["macro_f1"],
        "no_outer_selected_fusion": True,
        "protected_rows_read": 0,
        "outer_held_metrics_released_only_after_complete_matrix": True,
        "fixed_diagnostic_policy": protocol["fixed_diagnostics"],
        "artifacts": {
            name: artifact(result_dir / name)
            for name in ("oof_probabilities.npz", "slot_evidence.npz")
        },
    }
    immutable_json(result_dir / "summary.json", summary)
    if {path.name for path in result_dir.iterdir()} != allowed_result_files:
        raise RuntimeError("SEAR result artifact inventory is incomplete")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("resource", "prepare", "train", "summarize"), required=True
    )
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--dense-dir", type=Path, default=DENSE)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(PROTOCOL)
    validate_protocol(protocol)
    torch.set_num_threads(2)
    if args.stage == "resource":
        result = resource_pilot(protocol, output, args.device)
        event("sear_resource_complete", passed=result["passed"], results=result["results"])
        return
    if args.stage == "prepare":
        lock = prepare(output, args.dense_dir.resolve(), args.source_data.resolve())
        event("sear_prepare_complete", parameters=lock["parameter_counts"])
        return
    lock, patches, center, data = load_locked(output)
    if args.stage == "train":
        if (
            args.device != lock["device"]
            or not args.device.startswith("cuda")
            or not torch.cuda.is_available()
            or torch.cuda.get_device_name(torch.device(args.device)) != lock["gpu"]
        ):
            raise RuntimeError(
                "SEAR training must use the CUDA device that passed the resource gate"
            )
        arms = (args.arm,) if args.arm else ARMS
        folds = (
            (args.fold,) if args.fold is not None else lock["protocol"]["splitting"]["outer_folds"]
        )
        for fold in folds:
            for arm in arms:
                run_workload(output, lock, patches, center, data, arm, fold, args.device)
        if args.arm is None and args.fold is None:
            summary = summarize(output, lock, data)
            event("sear_matrix_complete", primary_macro_f1=summary["primary_macro_f1"])
        return
    summary = summarize(output, lock, data)
    event("sear_summary_complete", primary_macro_f1=summary["primary_macro_f1"])


if __name__ == "__main__":
    main()
