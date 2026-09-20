"""Run the preregistered label-blind center-completion reconstruction screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.center_completion_screen import (  # noqa: E402
    ARMS,
    FEATURE_DIM,
    MAX_UPDATES,
    SCREEN_SEED,
    EvidenceCompletionModel,
    PackedScreenData,
    build_gate_features_21,
    fit_outer_fold,
    matched_parameter_counts,
    outer_scenario_splits,
    pack_dense_arm,
    relative_cosine_error_reduction,
    repeated_masked_center_control,
    spatial_reassignment_control,
    time_reversal_control,
    wrong_track_control,
)
from hac.center_evidence_completion import (  # noqa: E402
    AnchorMatches,
    bilinear_transport,
    donor_confidence_logits,
    fit_visible_anchor_transport,
    transport_consensus,
)
from hac.okutama_native_video import EXPECTED_SCENARIOS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_center_evidence_completion_protocol.json"
PLAN = ROOT / ".runs/research_20260913/center_evidence_completion_v1/plan_v8.json"
CACHE = ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1"
CACHE_AUDIT = (
    ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1_audit/audit_receipt.json"
)
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/center_evidence_completion_screen_v1"
DEFAULT_TRANSPORT_CACHE = (
    ROOT / ".runs/research_20260913/center_evidence_completion_transport_v1"
)
DEFAULT_TRANSPORT_AUDIT = (
    ROOT
    / ".runs/research_20260913/center_evidence_completion_transport_v1_audit/audit_receipt.json"
)

EXPECTED = {
    "protocol": "f2e95656cfd179e72522d02103d2542e96b4124ccb76c5f4c0c1706783505244",
    "plan": "b7e148c8efe1e72fc0000ffd0713aecf16a3a4e21387d9f520a9dac6e5e6b63c",
    "cache_summary": "5eec07a433c189de949abeff70e1987fb8fbfb873cb4c0f308921b5833d4f201",
    "cache_request": "09fa581f4dcadaa27c9b41f647912da22566f69d25a4d796b0ccb1d3f740c894",
    "cache_audit": "269eccee054f3313ed254445e5fb5a9cf71a581bf14dc1b4f1cd9306c3e6b1e2",
    "screen_core": "582a3f81b56dae749fb98691dc07eea65f0e409bd25f95158ce9afd870a37db7",
    "transport_core": "3e284fefbc1150744212427a86cb9308e0ba137f6ac2cf3b7b706de87994256b",
}
EXPECTED_TRANSPORT = {
    "summary": "1e2d3f1af9a096b80283a8d30cd00d06199d6e439064a62639eee3f0d6f7e401",
    "request": "99b420b47f09fd94d0aa619783375a2d07b63586d4c75dbe3f9d5c10b81ee3ed",
    "receipts": "2eaa63e8a7f649e7b5d02b6d1bbb959df3b18da5272d689425e5d65d6e072c3e",
    "materializer": "c56972e3384912aec82162b104e0c2363e557f681e0939d89b53fdd39ffabe2e",
    "screen_core": "0353fcc082a90faa12902a2c96d1dc5d9a8972a8076860888b96fc4c6a749b45",
}
ARRAY_NAMES = (
    "teacher_tokens.npy",
    "masked_tokens.npy",
    "neighbor_tokens.npy",
    "wrong_neighbor_tokens.npy",
    "target_masks.npy",
    "visible_masks.npy",
    "neighbor_valid.npy",
    "wrong_neighbor_valid.npy",
    "wrong_slot_available.npy",
)
CONTROL_NAMES = (
    "repeated_masked_center",
    "wrong_track",
    "spatial_reassignment",
    "time_reversal",
)
TRANSPORT_ARRAY_NAMES = (
    "true_transport_tokens.npy",
    "true_transport_available.npy",
    "true_gate_features.npy",
    "wrong_transport_tokens.npy",
    "wrong_transport_available.npy",
    "wrong_gate_features.npy",
    "repeated_transport_tokens.npy",
    "repeated_transport_available.npy",
    "repeated_gate_features.npy",
)


@dataclass(frozen=True)
class DenseTransport:
    tokens: np.ndarray
    available: np.ndarray
    gate_features: np.ndarray
    diagnostics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transport-cache", type=Path, default=DEFAULT_TRANSPORT_CACHE)
    parser.add_argument("--transport-audit", type=Path, default=DEFAULT_TRANSPORT_AUDIT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--smoke-updates",
        type=int,
        help="NON-SCIENTIFIC startup check; requires a nondefault output directory",
    )
    args = parser.parse_args()
    if args.smoke_updates is not None:
        if not 1 <= args.smoke_updates < MAX_UPDATES:
            parser.error("--smoke-updates must be in 1..399")
        if args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
            parser.error("a smoke run requires an explicit nondefault --output-dir")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_cache_locks() -> dict[str, Any]:
    paths = {
        "protocol": PROTOCOL,
        "plan": PLAN,
        "cache_summary": CACHE / "summary.json",
        "cache_request": CACHE / "request.json",
        "cache_audit": CACHE_AUDIT,
        "screen_core": ROOT / "src/hac/center_completion_screen.py",
        "transport_core": ROOT / "src/hac/center_evidence_completion.py",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED:
        changed = {name: value for name, value in observed.items() if value != EXPECTED[name]}
        raise RuntimeError(f"prospective screen lock changed: {changed}")
    protocol = read_json(PROTOCOL)
    plan = read_json(PLAN)
    summary = read_json(CACHE / "summary.json")
    request = read_json(CACHE / "request.json")
    audit = read_json(CACHE_AUDIT)
    if (
        protocol.get("schema_version") != 8
        or protocol.get("status")
        != "LABEL_BLIND_128_CENTER_EXTRACTION_AUTHORIZED_NO_TASK_FIT"
        or plan.get("protocol_sha256") != EXPECTED["protocol"]
        or summary.get("request_sha256") != EXPECTED["cache_request"]
        or summary.get("status") != "CENTER_EVIDENCE_COMPLETION_FULL_CACHE_COMPLETE"
        or summary.get("scientific_gate_evaluated") is not False
        or summary.get("task_training_authorized") is not False
        or audit.get("status") != "INDEPENDENT_FULL_CACHE_REPLAY_AUDIT_PASS"
        or audit.get("array_comparison", {}).get("files_byte_identical") != "9/9"
        or not audit.get("locks", {}).get("all_direct_hashes_match_both_requests")
        or not audit.get("locks", {}).get("all_model_snapshot_file_hashes_and_sizes_match")
    ):
        raise RuntimeError("cache authorization or independent replay receipt is not valid")
    forbidden_counts = summary.get("zero_access_counters", {})
    if any(forbidden_counts.get(key) != 0 for key in forbidden_counts):
        raise RuntimeError("cache extraction crossed its declared zero-access boundary")
    artifacts = summary.get("artifacts", {})
    if set(artifacts) != set(ARRAY_NAMES):
        raise RuntimeError("cache artifact inventory changed")
    checked_arrays = {}
    for name in ARRAY_NAMES:
        path = CACHE / name
        receipt = artifacts[name]
        if path.stat().st_size != receipt["size_bytes"] or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"cache array changed: {name}")
        audited = audit["arrays"][name]
        if audited["sha256"] != receipt["sha256"]:
            raise RuntimeError(f"cache/audit array receipt mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != receipt["shape"] or str(array.dtype) != receipt["dtype"]:
            raise RuntimeError(f"cache array schema changed: {name}")
        checked_arrays[name] = receipt
    view_receipts = read_json(CACHE / "view_receipts.json")["rows"]
    sample_ids = request.get("sample_ids", [])
    if (
        len(sample_ids) != 128
        or len(set(sample_ids)) != 128
        or [row["sample_id"] for row in view_receipts] != sample_ids
        or [row["center_index"] for row in view_receipts] != list(range(128))
    ):
        raise RuntimeError("cache identity/order receipt changed")
    scenarios = [row["scenario"] for row in view_receipts]
    outer_scenario_splits(scenarios)
    dependency = read_json(CACHE / "dependency_audit.json")["rows"]
    if len(dependency) != 256 or not all(
        row["altered_hidden_pixels"] > 0
        and row["masked_raw_bit_exact"]
        and row["letterbox_bit_exact"]
        and row["normalized_bit_exact"]
        for row in dependency
    ):
        raise RuntimeError("hidden-target dependency audit is not bit-exact")
    deterministic = audit.get("deterministic_json_artifacts", {})
    for name in ("dependency_audit.json", "member_receipts.json", "view_receipts.json"):
        if sha256_file(CACHE / name) != deterministic.get(name):
            raise RuntimeError(f"audited cache receipt changed: {name}")
    return {
        "status": "SCREEN_INPUT_LOCKS_PASS",
        "hashes": observed,
        "arrays": checked_arrays,
        "sample_ids": sample_ids,
        "scenarios": scenarios,
        "dependency_rows_bit_exact": len(dependency),
        "independent_cache_replay": True,
        "zero_access_counters": forbidden_counts,
        "request_locks": {
            key: request[key]
            for key in (
                "protocol_sha256",
                "plan_sha256",
                "wrong_track_map_sha256",
                "pilot_selection_sha256",
                "frame_manifest_sha256",
                "image_allowlist_sha256",
                "materialization_lock_sha256",
            )
        },
    }


def verify_transport_locks(transport_cache: Path, transport_audit: Path) -> dict[str, Any]:
    """Require the independently replayed, label-blind transport cache."""

    summary_path = transport_cache / "summary.json"
    request_path = transport_cache / "request.json"
    receipts_path = transport_cache / "transport_receipts.json"
    direct_hashes = {
        "summary": sha256_file(summary_path),
        "request": sha256_file(request_path),
        "receipts": sha256_file(receipts_path),
    }
    if direct_hashes != {key: EXPECTED_TRANSPORT[key] for key in direct_hashes}:
        raise RuntimeError("prospective transport cache lock changed")
    summary = read_json(summary_path)
    request = read_json(request_path)
    audit = read_json(transport_audit)
    if (
        summary.get("status") != "CENTER_EVIDENCE_COMPLETION_TRANSPORT_CACHE_COMPLETE"
        or summary.get("scientific_gate_evaluated") is not False
        or summary.get("task_training_authorized") is not False
        or summary.get("request_sha256") != sha256_file(request_path)
        or request.get("protocol_sha256") != EXPECTED["protocol"]
        or request.get("plan_sha256") != EXPECTED["plan"]
        or request.get("cache_summary_sha256") != EXPECTED["cache_summary"]
        or request.get("cache_audit_sha256") != EXPECTED["cache_audit"]
        or request.get("conditions")
        != ["true_track", "wrong_track", "repeated_masked_center"]
        or request.get("sample_ids") != read_json(CACHE / "request.json").get("sample_ids")
        or summary.get("centers") != 128
        or summary.get("mask_rows") != 256
        or audit.get("status")
        not in {
            "INDEPENDENT_TRANSPORT_CACHE_REPLAY_AUDIT_PASS",
            "INDEPENDENT_TRANSPORT_REPLAY_AUDIT_PASS",
        }
    ):
        raise RuntimeError("transport cache authorization or replay audit is invalid")
    if any(value != 0 for value in summary.get("zero_access_counters", {}).values()):
        raise RuntimeError("transport cache crossed its zero-access boundary")
    if request.get("source_hashes", {}).get(
        "src/hac/center_completion_screen.py"
    ) != EXPECTED_TRANSPORT[
        "screen_core"
    ] or request.get("source_hashes", {}).get(
        "src/hac/center_evidence_completion.py"
    ) != EXPECTED["transport_core"]:
        raise RuntimeError("transport cache was built with different mechanism code")
    if request.get("source_hashes", {}).get(
        "experiments/materialize_okutama_center_completion_transport.py"
    ) != EXPECTED_TRANSPORT["materializer"]:
        raise RuntimeError("transport cache materializer changed")
    artifacts = summary.get("artifacts", {})
    if set(artifacts) != set(TRANSPORT_ARRAY_NAMES):
        raise RuntimeError("transport cache artifact inventory changed")
    audit_arrays = audit.get("arrays", {})
    checked = {}
    for name in TRANSPORT_ARRAY_NAMES:
        path = transport_cache / name
        receipt = artifacts[name]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        observed_hash = sha256_file(path)
        if (
            observed_hash != receipt["sha256"]
            or path.stat().st_size != receipt["size_bytes"]
            or list(array.shape) != receipt["shape"]
            or str(array.dtype) != receipt["dtype"]
        ):
            raise RuntimeError(f"transport artifact changed: {name}")
        audited = audit_arrays.get(name, {})
        if audited.get("sha256") != observed_hash:
            raise RuntimeError(f"transport replay receipt mismatch: {name}")
        checked[name] = receipt
    if audit.get("array_comparison", {}).get("files_byte_identical") != "9/9":
        raise RuntimeError("transport replay was not byte-identical for all nine arrays")
    return {
        "status": "TRANSPORT_INPUT_LOCKS_PASS",
        "cache_path": str(transport_cache.resolve()),
        "audit_path": str(transport_audit.resolve()),
        "summary_sha256": direct_hashes["summary"],
        "request_sha256": direct_hashes["request"],
        "transport_receipts_sha256": direct_hashes["receipts"],
        "audit_sha256": sha256_file(transport_audit),
        "artifacts": checked,
        "availability": summary["availability"],
        "valid_affine_fits": summary["valid_affine_fits"],
        "independent_replay": True,
    }


def _local_statistics(matches: AnchorMatches, target_mask: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_xy = np.argwhere(target_mask)[:, ::-1].astype(np.float64)
    cosine = np.full(len(target_xy), np.nan, dtype=np.float32)
    distance = np.full(len(target_xy), np.inf, dtype=np.float32)
    for index in np.flatnonzero(observed):
        distances = np.linalg.norm(matches.center_xy - target_xy[index], axis=1)
        nearest = np.argsort(distances, kind="stable")[:4]
        cosine[index] = np.float32(np.median(matches.cosine[nearest]))
        distance[index] = np.float32(distances[nearest[0]])
    return cosine, distance


def build_partial_transport(
    masked_tokens: np.ndarray,
    donor_tokens: np.ndarray,
    donor_valid: np.ndarray,
    target_masks: np.ndarray,
    visible_masks: np.ndarray,
) -> DenseTransport:
    """Build fixed visible-anchor transport and exact 21-feature gate inputs."""

    centers = masked_tokens.shape[0]
    expected = (centers, 2, 27, 27)
    if (
        masked_tokens.shape != (*expected, FEATURE_DIM)
        or donor_tokens.shape != (centers, 4, 27, 27, FEATURE_DIM)
        or donor_valid.shape != (centers, 4, 27, 27)
        or target_masks.shape != expected
        or visible_masks.shape != expected
        or donor_valid.dtype != np.bool_
        or target_masks.dtype != np.bool_
        or visible_masks.dtype != np.bool_
    ):
        raise ValueError("dense cache differs from the locked transport geometry")
    result = np.zeros((*expected, FEATURE_DIM), dtype=np.float16)
    available = np.zeros(expected, dtype=bool)
    gate = np.zeros((*expected, 21), dtype=np.float32)
    affine_valid = 0
    affine_total = centers * 2 * 4
    observed_targets = 0
    target_total = int(target_masks.sum())
    for center in range(centers):
        for mask_id in range(2):
            target = np.asarray(target_masks[center, mask_id])
            center_tokens = np.asarray(masked_tokens[center, mask_id], dtype=np.float32)
            target_yx = np.argwhere(target)
            count = len(target_yx)
            donor_values = np.zeros((4, count, FEATURE_DIM), dtype=np.float32)
            donor_observed = np.zeros((4, count), dtype=bool)
            donor_logits = np.full((4, count), -np.inf, dtype=np.float64)
            residuals = np.full((count, 4), np.inf, dtype=np.float32)
            local_cosine = np.full((count, 4), np.nan, dtype=np.float32)
            anchor_distance = np.full((count, 4), np.inf, dtype=np.float32)
            for donor in range(4):
                transform, matches = fit_visible_anchor_transport(
                    center_tokens,
                    np.asarray(donor_tokens[center, donor], dtype=np.float32),
                    target,
                    center_valid=np.asarray(visible_masks[center, mask_id]),
                    neighbor_valid=np.asarray(donor_valid[center, donor]),
                )
                affine_valid += int(transform.valid)
                values, observed = bilinear_transport(
                    np.asarray(donor_tokens[center, donor], dtype=np.float32),
                    target,
                    transform,
                    neighbor_valid=np.asarray(donor_valid[center, donor]),
                )
                logits = donor_confidence_logits(matches, transform, target, observed)
                cosine, distance = _local_statistics(matches, target, observed)
                donor_values[donor] = values
                donor_observed[donor] = observed
                donor_logits[donor] = logits
                residuals[observed, donor] = np.float32(transform.weighted_residual)
                local_cosine[observed, donor] = cosine[observed]
                anchor_distance[observed, donor] = distance[observed]
            masked_target = center_tokens[target]
            consensus = transport_consensus(
                donor_values, donor_observed, donor_logits, masked_target
            )
            gate_rows = build_gate_features_21(
                torch.from_numpy(donor_values.transpose(1, 0, 2)),
                torch.from_numpy(donor_observed.T),
                torch.from_numpy(residuals),
                torch.from_numpy(local_cosine),
                torch.from_numpy(anchor_distance),
            ).numpy()
            y, x = target_yx[:, 0], target_yx[:, 1]
            result[center, mask_id, y, x] = consensus.features.astype(np.float16)
            available[center, mask_id, y, x] = consensus.available
            gate[center, mask_id, y, x] = gate_rows
            observed_targets += int(consensus.available.sum())
    return DenseTransport(
        result,
        available,
        gate,
        {
            "affine_valid": affine_valid,
            "affine_total": affine_total,
            "available_target_tokens": observed_targets,
            "target_tokens": target_total,
            "coverage": observed_targets / target_total,
        },
    )


def repeated_center_transport(
    masked_tokens: np.ndarray,
    target_masks: np.ndarray,
    visible_masks: np.ndarray,
) -> DenseTransport:
    """Run the identical transport path with four masked-center donors."""

    centers = masked_tokens.shape[0]
    outputs = np.zeros((centers, 2, 27, 27, FEATURE_DIM), dtype=np.float16)
    available = np.zeros((centers, 2, 27, 27), dtype=bool)
    gates = np.zeros((centers, 2, 27, 27, 21), dtype=np.float32)
    affine_valid = 0
    observed_targets = 0
    target_total = int(target_masks.sum())
    for center in range(centers):
        for mask_id in range(2):
            target = np.asarray(target_masks[center, mask_id])
            # The repeated donor is the complete masked-center token grid.  Its
            # hidden locations contain mask evidence, not teacher evidence, and
            # therefore remain valid donor samples for this negative control.
            valid = np.ones((27, 27), dtype=bool)
            tokens = np.asarray(masked_tokens[center, mask_id], dtype=np.float32)
            donor_bank = np.repeat(tokens[None], 4, axis=0)
            donor_valid = np.repeat(valid[None], 4, axis=0)
            built = build_partial_transport(
                masked_tokens[center : center + 1, mask_id : mask_id + 1].repeat(2, axis=1),
                donor_bank[None],
                donor_valid[None],
                target_masks[center : center + 1, mask_id : mask_id + 1].repeat(2, axis=1),
                visible_masks[center : center + 1, mask_id : mask_id + 1].repeat(2, axis=1),
            )
            # The helper requires two mask slots; both repeated slots are identical.
            y, x = np.argwhere(target)[:, 0], np.argwhere(target)[:, 1]
            outputs[center, mask_id, y, x] = built.tokens[0, 0, y, x]
            available[center, mask_id, y, x] = built.available[0, 0, y, x]
            gates[center, mask_id, y, x] = built.gate_features[0, 0, y, x]
            affine_valid += built.diagnostics["affine_valid"] // 2
            observed_targets += int(built.available[0, 0].sum())
    return DenseTransport(
        outputs,
        available,
        gates,
        {
            "affine_valid": affine_valid,
            "affine_total": centers * 2 * 4,
            "available_target_tokens": observed_targets,
            "target_tokens": target_total,
            "coverage": observed_targets / target_total,
        },
    )


def load_cache() -> dict[str, np.ndarray]:
    return {
        name.removesuffix(".npy"): np.load(CACHE / name, mmap_mode="r", allow_pickle=False)
        for name in ARRAY_NAMES
    }


def load_transport_cache(transport_cache: Path) -> tuple[DenseTransport, DenseTransport, DenseTransport]:
    arrays = {
        name.removesuffix(".npy"): np.load(
            transport_cache / name, mmap_mode="r", allow_pickle=False
        )
        for name in TRANSPORT_ARRAY_NAMES
    }

    def view(prefix: str) -> DenseTransport:
        available = arrays[f"{prefix}_transport_available"]
        targets = int(np.load(CACHE / "target_masks.npy", mmap_mode="r", allow_pickle=False).sum())
        observed = int(available.sum())
        return DenseTransport(
            arrays[f"{prefix}_transport_tokens"],
            available,
            arrays[f"{prefix}_gate_features"],
            {
                "available_target_tokens": observed,
                "target_tokens": targets,
                "coverage": observed / targets,
                "source": "independently_replayed_transport_cache",
            },
        )

    return view("true"), view("wrong"), view("repeated")


def pack_all(
    cache: dict[str, np.ndarray],
    true: DenseTransport,
    wrong: DenseTransport,
    repeated: DenseTransport,
) -> dict[str, PackedScreenData]:
    common = {
        "masked_tokens": cache["masked_tokens"],
        "teacher_tokens": cache["teacher_tokens"],
        "neighbor_tokens": cache["neighbor_tokens"],
        "neighbor_valid": cache["neighbor_valid"],
        "target_masks": cache["target_masks"],
        "visible_masks": cache["visible_masks"],
    }
    packed = {
        arm: pack_dense_arm(
            arm=arm,
            **common,
            **(
                {
                    "transport_tokens": true.tokens,
                    "transport_available": true.available,
                    "gate_features": true.gate_features,
                }
                if arm == "P3_partial_transport"
                else {}
            ),
        )
        for arm in ARMS
    }
    packed["wrong_track"] = pack_dense_arm(
        arm="P3_partial_transport",
        masked_tokens=cache["masked_tokens"],
        teacher_tokens=cache["teacher_tokens"],
        neighbor_tokens=cache["wrong_neighbor_tokens"],
        neighbor_valid=cache["wrong_neighbor_valid"],
        target_masks=cache["target_masks"],
        visible_masks=cache["visible_masks"],
        transport_tokens=wrong.tokens,
        transport_available=wrong.available,
        gate_features=wrong.gate_features,
    )
    packed["repeated_masked_center"] = pack_dense_arm(
        arm="P3_partial_transport",
        **common,
        transport_tokens=repeated.tokens,
        transport_available=repeated.available,
        gate_features=repeated.gate_features,
    )
    return packed


@torch.no_grad()
def evaluate_detail(model: torch.nn.Module, data: PackedScreenData, device: str) -> dict[str, np.ndarray]:
    batch = data.to(device)
    model.to(device).eval()
    output = model(batch)
    teacher = batch.teacher_tokens.detach().float()
    cosine = 1.0 - F.cosine_similarity(output.prediction, teacher, dim=-1, eps=1e-12)
    huber = F.smooth_l1_loss(
        F.layer_norm(output.prediction, (FEATURE_DIM,)),
        F.layer_norm(teacher, (FEATURE_DIM,)),
        reduction="none",
        beta=1.0,
    ).mean(-1)
    valid = batch.target_valid
    count = valid.sum(1)
    center_mean = lambda values: (values * valid).sum(1) / count  # noqa: E731
    return {
        "cosine": center_mean(cosine).cpu().numpy(),
        "huber": center_mean(huber).cpu().numpy(),
        "coverage": center_mean(output.evidence_available.float()).cpu().numpy(),
        "gate": center_mean(output.gate).cpu().numpy(),
        "target_count": count.cpu().numpy(),
    }


def checkpoint(path: Path, fit: Any, arm: str, fold: int) -> None:
    torch.save(
        {
            "arm": arm,
            "held_fold": fold,
            "seed": fit.seed,
            "updates": fit.updates,
            "train_rows": fit.train_rows,
            "held_rows": fit.held_rows,
            "training_history": fit.training_history,
            "state_dict": fit.model.state_dict(),
        },
        path,
    )


def _empty_metrics() -> dict[str, np.ndarray]:
    return {
        key: np.full(128, np.nan, dtype=np.float64)
        for key in ("cosine", "huber", "coverage", "gate", "target_count")
    }


def assign_rows(destination: dict[str, np.ndarray], rows: np.ndarray, values: dict[str, np.ndarray]) -> None:
    for key in destination:
        destination[key][rows] = values[key]


def metric_view(values: dict[str, np.ndarray]) -> dict[str, Any]:
    weights = values["target_count"]
    return {
        "mean_target_token_cosine_error": float(np.mean(values["cosine"])),
        "layernorm_huber_error": float(np.mean(values["huber"])),
        "mean_center_coverage": float(np.mean(values["coverage"])),
        "mean_intervention": float(np.mean(values["gate"])),
        "target_weighted_coverage": float(np.average(values["coverage"], weights=weights)),
        "target_weighted_intervention": float(np.average(values["gate"], weights=weights)),
        "per_center_gate_percentiles": {
            str(percentile): float(np.percentile(values["gate"], percentile))
            for percentile in (0, 10, 25, 50, 75, 90, 100)
        },
    }


def scenario_view(values: dict[str, np.ndarray], scenarios: list[str]) -> dict[str, Any]:
    identities = np.asarray(scenarios)
    return {
        scenario: {
            "centers": int((identities == scenario).sum()),
            "mean_target_token_cosine_error": float(np.mean(values["cosine"][identities == scenario])),
            "layernorm_huber_error": float(np.mean(values["huber"][identities == scenario])),
            "mean_center_coverage": float(np.mean(values["coverage"][identities == scenario])),
            "mean_intervention": float(np.mean(values["gate"][identities == scenario])),
        }
        for scenario in sorted(EXPECTED_SCENARIOS)
    }


def gain_fraction(
    base: np.ndarray,
    nominal: np.ndarray,
    control: np.ndarray,
    rows: np.ndarray | None = None,
) -> float | None:
    if rows is None:
        rows = np.arange(len(base))
    denominator = float(np.mean(base[rows]) - np.mean(nominal[rows]))
    if denominator <= 0:
        return None
    return float((np.mean(base[rows]) - np.mean(control[rows])) / denominator)


def evaluate_gates(
    metrics: dict[str, dict[str, np.ndarray]],
    controls: dict[str, dict[str, np.ndarray]],
    scenarios: list[str],
    wrong_slot_available: np.ndarray,
    locks: dict[str, Any],
) -> dict[str, Any]:
    p0, p1, p2, p3 = (metrics[arm] for arm in ARMS)
    scenario_array = np.asarray(scenarios)
    p3_available = p3["coverage"] > 0
    wrong_rows = np.flatnonzero(np.asarray(wrong_slot_available).any(axis=1))
    reductions_p1 = []
    better_p2 = []
    for scenario in sorted(EXPECTED_SCENARIOS):
        rows = np.flatnonzero(scenario_array == scenario)
        reductions_p1.append(
            (np.mean(p1["cosine"][rows]) - np.mean(p3["cosine"][rows]))
            / np.mean(p1["cosine"][rows])
        )
        better_p2.append(np.mean(p3["cosine"][rows]) < np.mean(p2["cosine"][rows]))
    repeated_fraction = gain_fraction(p0["cosine"], p3["cosine"], controls["repeated_masked_center"]["cosine"])
    wrong_fraction = gain_fraction(
        p0["cosine"], p3["cosine"], controls["wrong_track"]["cosine"], wrong_rows
    )
    nominal_gain = float(np.mean(p0["cosine"]) - np.mean(p3["cosine"]))
    spatial_removed: float | None = (
        float(np.mean(controls["spatial_reassignment"]["cosine"]) - np.mean(p3["cosine"]))
        / nominal_gain
        if nominal_gain > 0
        else None
    )
    checks = {
        "exact_source_and_hash_replay": bool(locks["independent_cache_replay"]),
        "usable_centers_min_103": int(p3_available.sum()) >= 103,
        "usable_center_each_scenario_min_8": min(
            int(p3_available[scenario_array == scenario].sum()) for scenario in EXPECTED_SCENARIOS
        ) >= 8,
        "P3_vs_P0_relative_reduction_min_0.05": relative_cosine_error_reduction(metric_view(p0), metric_view(p3)) >= 0.05,
        "P3_vs_P1_relative_reduction_min_0.10": relative_cosine_error_reduction(metric_view(p1), metric_view(p3)) >= 0.10,
        "P3_vs_P1_scenarios_min_8": int(np.sum(np.asarray(reductions_p1) >= 0.10)) >= 8,
        "P3_vs_P2_relative_reduction_min_0.05": relative_cosine_error_reduction(metric_view(p2), metric_view(p3)) >= 0.05,
        "P3_better_than_P2_scenarios_min_7": int(np.sum(better_p2)) >= 7,
        "repeated_gain_fraction_lt_0.5": repeated_fraction is not None
        and repeated_fraction < 0.5,
        "wrong_track_gain_fraction_lt_0.5": wrong_fraction is not None
        and wrong_fraction < 0.5,
        "spatial_reassignment_removes_gain_fraction_min_0.5": spatial_removed is not None
        and spatial_removed >= 0.5,
        "target_dependency_audit_bit_exact": locks["dependency_rows_bit_exact"] == 256,
    }
    return {
        "all_pass": all(checks.values()),
        "checks": checks,
        "measurements": {
            "usable_centers": int(p3_available.sum()),
            "minimum_usable_centers_per_scenario": min(
                int(p3_available[scenario_array == scenario].sum()) for scenario in EXPECTED_SCENARIOS
            ),
            "P3_relative_reduction_vs_P0": relative_cosine_error_reduction(metric_view(p0), metric_view(p3)),
            "P3_relative_reduction_vs_P1": relative_cosine_error_reduction(metric_view(p1), metric_view(p3)),
            "P3_relative_reduction_vs_P2": relative_cosine_error_reduction(metric_view(p2), metric_view(p3)),
            "scenarios_P3_reduction_vs_P1_at_least_0.10": int(np.sum(np.asarray(reductions_p1) >= 0.10)),
            "scenarios_P3_strictly_better_than_P2": int(np.sum(better_p2)),
            "repeated_masked_center_gain_fraction": repeated_fraction,
            "wrong_track_gain_fraction_paired_124_centers": wrong_fraction,
            "wrong_track_paired_centers": int(len(wrong_rows)),
            "spatial_reassignment_removed_gain_fraction": spatial_removed,
        },
    }


def save_per_center_csv(path: Path, sample_ids: list[str], scenarios: list[str], folds: np.ndarray, metrics: dict[str, dict[str, np.ndarray]], controls: dict[str, dict[str, np.ndarray]]) -> None:
    fields = ["center_index", "sample_id", "scenario", "held_fold"]
    for name in (*ARMS, *CONTROL_NAMES):
        fields.extend(f"{name}_{metric}" for metric in ("cosine", "huber", "coverage", "gate", "target_count"))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (sample_id, scenario) in enumerate(zip(sample_ids, scenarios, strict=True)):
            row: dict[str, Any] = {"center_index": index, "sample_id": sample_id, "scenario": scenario, "held_fold": int(folds[index])}
            for name, values in {**metrics, **controls}.items():
                for metric, array in values.items():
                    row[f"{name}_{metric}"] = float(array[index])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SCREEN_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SCREEN_SEED)
    started = time.perf_counter()
    locks = verify_cache_locks()
    transport_locks = verify_transport_locks(args.transport_cache, args.transport_audit)
    print(
        json.dumps(
            {
                "cache_status": locks["status"],
                "transport_status": transport_locks["status"],
                "centers": len(locks["sample_ids"]),
            },
            sort_keys=True,
        )
    )
    if args.validate_only:
        return
    scientific = args.smoke_updates is None
    updates = MAX_UPDATES if scientific else args.smoke_updates
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "checkpoints").mkdir()
    request = {
        "status": "CENTER_COMPLETION_SCREEN_REQUEST_LOCKED",
        "scientific": scientific,
        "non_scientific_smoke": not scientific,
        "updates_per_fit": updates,
        "exact_updates_required_for_gate": MAX_UPDATES,
        "fits": 20,
        "batch_size": 64,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "seed": SCREEN_SEED,
        "device": args.device,
        "input_locks": locks,
        "transport_locks": transport_locks,
        "source_hashes": {
            "experiments/screen_okutama_center_evidence_completion.py": sha256_file(Path(__file__)),
            "src/hac/center_completion_screen.py": EXPECTED["screen_core"],
            "src/hac/center_evidence_completion.py": EXPECTED["transport_core"],
        },
        "zero_access_contract": {
            "task_labels": 0,
            "arftr_probability_arrays": 0,
            "pose_outputs": 0,
            "support_categories": 0,
            "router_fits": 0,
        },
    }
    write_json(args.output_dir / "request.json", request)
    cache = load_cache()
    true, wrong, repeated = load_transport_cache(args.transport_cache)
    transport_seconds = 0.0
    packed = pack_all(cache, true, wrong, repeated)
    scenarios = locks["scenarios"]
    sample_ids = locks["sample_ids"]
    splits = outer_scenario_splits(scenarios)
    metrics = {arm: _empty_metrics() for arm in ARMS}
    controls = {name: _empty_metrics() for name in CONTROL_NAMES}
    histories: dict[str, Any] = {}
    fits_started = time.perf_counter()
    for fold, (_, held) in enumerate(splits):
        base_fit = fit_outer_fold(
            packed["P0_center_only"], scenarios, fold,
            arm="P0_center_only", updates=updates, batch_size=64,
            learning_rate=3e-4, weight_decay=0.01, seed=SCREEN_SEED, device=args.device,
        )
        checkpoint(args.output_dir / "checkpoints" / f"fold{fold}_P0_center_only.pt", base_fit, "P0_center_only", fold)
        histories[f"fold{fold}_P0_center_only"] = list(base_fit.training_history)
        assign_rows(metrics["P0_center_only"], held, evaluate_detail(base_fit.model, packed["P0_center_only"].select(held), args.device))
        for arm in ARMS[1:]:
            fit = fit_outer_fold(
                packed[arm], scenarios, fold, arm=arm, fitted_p0=base_fit.model,
                updates=updates, batch_size=64, learning_rate=3e-4,
                weight_decay=0.01, seed=SCREEN_SEED, device=args.device,
            )
            checkpoint(args.output_dir / "checkpoints" / f"fold{fold}_{arm}.pt", fit, arm, fold)
            histories[f"fold{fold}_{arm}"] = list(fit.training_history)
            held_data = packed[arm].select(held)
            assign_rows(metrics[arm], held, evaluate_detail(fit.model, held_data, args.device))
            if arm == "P3_partial_transport":
                if not isinstance(fit.model, EvidenceCompletionModel):
                    raise RuntimeError("P3 fit returned the wrong model type")
                repeated_view = repeated_masked_center_control(
                    held_data,
                    repeated_gate_features=packed["repeated_masked_center"].select(held).gate_features,
                    repeated_available=packed["repeated_masked_center"].select(held).evidence_available,
                )
                views = {
                    "repeated_masked_center": repeated_view,
                    "wrong_track": wrong_track_control(held_data, packed["wrong_track"].select(held)),
                    "spatial_reassignment": spatial_reassignment_control(held_data, [sample_ids[index] for index in held]),
                    "time_reversal": time_reversal_control(held_data),
                }
                for name, view in views.items():
                    assign_rows(controls[name], held, evaluate_detail(fit.model, view, args.device))
        print(json.dumps({"held_fold_complete": fold, "updates_per_fit": updates}, sort_keys=True), flush=True)
    fit_seconds = time.perf_counter() - fits_started
    if any(not np.isfinite(array).all() for group in (*metrics.values(), *controls.values()) for array in group.values()):
        raise RuntimeError("OOF population is incomplete or nonfinite")
    fold_ids = np.empty(128, dtype=np.int64)
    for fold, (_, held) in enumerate(splits):
        fold_ids[held] = fold
    save_per_center_csv(args.output_dir / "per_center_oof.csv", sample_ids, scenarios, fold_ids, metrics, controls)
    np.savez_compressed(
        args.output_dir / "oof_metrics.npz",
        sample_ids=np.asarray(sample_ids), scenarios=np.asarray(scenarios), held_fold=fold_ids,
        **{f"{name}__{metric}": values for name, group in {**metrics, **controls}.items() for metric, values in group.items()},
    )
    metric_summary = {name: metric_view(values) for name, values in {**metrics, **controls}.items()}
    scenario_summary = {name: scenario_view(values, scenarios) for name, values in {**metrics, **controls}.items()}
    gates = evaluate_gates(metrics, controls, scenarios, cache["wrong_slot_available"], locks) if scientific else {"all_pass": None, "status": "NOT_EVALUATED_NON_SCIENTIFIC_SMOKE"}
    summary = {
        "status": "CENTER_COMPLETION_RECONSTRUCTION_SCREEN_COMPLETE" if scientific else "CENTER_COMPLETION_NON_SCIENTIFIC_SMOKE_COMPLETE",
        "scientific": scientific,
        "label_blind_result_only": True,
        "task_training_authorized": False,
        "next_authorization_requires_new_versioned_lock": bool(scientific and gates["all_pass"]),
        "retained_arftr_changed": False,
        "centers": 128,
        "scenarios": 11,
        "fits": 20,
        "updates_per_fit": updates,
        "total_optimizer_updates": 20 * updates,
        "parameter_counts": matched_parameter_counts(),
        "transport_diagnostics": {"true_track": true.diagnostics, "wrong_track": wrong.diagnostics, "repeated_masked_center": repeated.diagnostics},
        "metrics": metric_summary,
        "by_scenario": scenario_summary,
        "gates": gates,
        "runtime_seconds": {"transport": transport_seconds, "fits_and_evaluation": fit_seconds, "total": time.perf_counter() - started},
        "request_sha256": sha256_file(args.output_dir / "request.json"),
        "zero_access_counters": request["zero_access_contract"],
    }
    write_json(args.output_dir / "training_history.json", histories)
    write_json(args.output_dir / "summary.json", summary)
    snapshot = args.output_dir / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), ROOT / "src/hac/center_completion_screen.py", ROOT / "src/hac/center_evidence_completion.py", PROTOCOL, PLAN):
        shutil.copy2(source, snapshot / source.name)
    print(json.dumps({"status": summary["status"], "all_gates_pass": gates["all_pass"], "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
