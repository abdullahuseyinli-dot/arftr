"""Run the locked, label-blind PCAR-v5 reconstruction policy smoke.

The first usable stage is deliberately a bounded timing smoke.  It exercises
the exact full-768 affine candidate, nested base exclusion, centre-balanced
policy objective, hard actions, and retain semantics without opening activity
labels or ARFTR outputs.  ``--stage all`` is reserved for the separately
authorized scientific screen after the smoke receipt is reviewed.
"""

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
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.center_completion_factorial import uniform_consensus  # noqa: E402
from hac.center_completion_screen import (  # noqa: E402
    BaseCompletionModel,
    PackedScreenData,
    new_screen_model,
    outer_scenario_splits,
    reconstruction_loss,
)
from hac.center_evidence_completion import (  # noqa: E402
    bilinear_transport,
    donor_confidence_logits,
    fit_visible_anchor_transport,
)
from hac.pcar import (  # noqa: E402
    ACTION_ALPHAS,
    ACTION_COUNT,
    FEATURE_COUNT,
    PCARPolicy,
    action_costs,
    center_balanced_expected_cost,
    exact_convex_actions_numpy,
    hard_actions,
)
from experiments import screen_okutama_center_evidence_completion as screen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / ".runs/research_20260913/center_completion_pcar_design_v5/pcar_protocol.json"
LOCK = ROOT / "experiments/okutama_pcar_v5_execution_lock.json"
CACHE = ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1"
TRANSPORT = ROOT / ".runs/research_20260913/center_evidence_completion_transport_v1"
SCREEN = ROOT / ".runs/research_20260913/center_evidence_completion_screen_v1"
FACTORIAL = ROOT / ".runs/research_20260913/center_completion_matched_factorial_v1"
FACTORIAL_AUDIT = ROOT / ".runs/research_20260913/center_completion_matched_factorial_v1_audit/audit_receipt.json"
DEFAULT_SMOKE = ROOT / ".runs/research_20260917/pcar_v5_timing_smoke_v1"
DEFAULT_SCREEN = ROOT / ".runs/research_20260917/pcar_v5_screen_v1"
CENTERS = 128
FOLDS = 5
BATCH_SIZE = 64
BASE_UPDATES = 400
POLICY_UPDATES = 200
SMOKE_UPDATES = 20
FEATURE_DIM = 768


@dataclass(frozen=True)
class CandidateBundle:
    affine: np.ndarray
    same_grid: np.ndarray
    available: np.ndarray
    teacher: np.ndarray
    masked: np.ndarray
    valid: np.ndarray
    positions: np.ndarray
    mask_ids: np.ndarray
    transport_features: np.ndarray


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


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalar/array values before strict JSON publication."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "timing-smoke", "all"), default="timing-smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-updates", type=int, default=SMOKE_UPDATES)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if not 2 <= args.smoke_updates <= 100:
        parser.error("--smoke-updates must be in 2..100")
    if args.output_dir is None:
        args.output_dir = DEFAULT_SCREEN if args.stage == "all" else DEFAULT_SMOKE
    return args


def _verify_execution_lock() -> dict[str, Any]:
    if not LOCK.exists():
        raise RuntimeError(f"missing pre-fit execution lock: {LOCK}")
    lock = read_json(LOCK)
    if lock.get("status") != "PCAR_V5_EXECUTION_LOCKED_BEFORE_RESULTS":
        raise RuntimeError("PCAR execution lock is not prospective")
    if lock.get("protocol_sha256") != sha256_file(PROTOCOL):
        raise RuntimeError("PCAR protocol hash changed after execution lock")
    if lock.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("PCAR runner changed after execution lock")
    if lock.get("policy_core_sha256") != sha256_file(ROOT / lock.get("policy_core_path", "src/hac/pcar.py")):
        raise RuntimeError("PCAR policy core changed after execution lock")
    for relative, expected in lock.get("pinned_inputs", {}).items():
        observed = sha256_file(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"PCAR pinned input changed: {relative}")
    for name, expected in lock.get("checkpoint_hashes", {}).items():
        observed = sha256_file(SCREEN / "checkpoints" / name)
        if observed != expected:
            raise RuntimeError(f"PCAR checkpoint changed: {name}")
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "PCAR_LABEL_BLIND_PROTOCOL_LOCKED_NO_TRAINING_RUN":
        raise RuntimeError("PCAR prospective protocol status changed")
    if protocol["authorization"]["task_training_authorized"] is not False:
        raise RuntimeError("task fitting cannot be authorized by PCAR")
    if protocol["authorization"]["pcar_training_authorized"] is not False:
        raise RuntimeError("the design artifact must remain planning-only")
    return lock


def _verify_prerequisites() -> dict[str, Any]:
    """Replay the existing immutable label-blind cache and screen locks."""

    locks = screen.verify_cache_locks()
    transport_locks = screen.verify_transport_locks(TRANSPORT, TRANSPORT / "../center_evidence_completion_transport_v1_audit/audit_receipt.json")
    factorial = read_json(FACTORIAL / "summary.json")
    factorial_audit = read_json(FACTORIAL_AUDIT)
    if factorial.get("status") != "CENTER_COMPLETION_MATCHED_FACTORIAL_COMPLETE":
        raise RuntimeError("matched factorial prerequisite is not complete")
    if factorial.get("locked_decision", {}).get("verdict", {}).get("transport_survives") is not True:
        raise RuntimeError("affine transport did not survive its prospective mechanism gate")
    if factorial_audit.get("status") != "CENTER_COMPLETION_MATCHED_FACTORIAL_EXACT_REPLAY_PASS":
        raise RuntimeError("matched factorial independent replay is not PASS")
    summary = read_json(SCREEN / "summary.json")
    if (
        summary.get("status") != "CENTER_COMPLETION_RECONSTRUCTION_SCREEN_COMPLETE"
        or summary.get("task_training_authorized") is not False
        or summary.get("retained_arftr_changed") is not False
        or summary.get("gates", {}).get("all_pass") is not False
        or any(summary.get("zero_access_counters", {}).values())
    ):
        raise RuntimeError("screen prerequisite boundary changed")
    return {"cache": locks, "transport": transport_locks, "factorial": factorial}


def _load_packed() -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, PackedScreenData]]:
    locks = _verify_prerequisites()
    cache = screen.load_cache()
    true, wrong, repeated = screen.load_transport_cache(TRANSPORT)
    packed = screen.pack_all(cache, true, wrong, repeated)
    return cache, locks, packed


def _matched_affine_candidates(
    cache: dict[str, np.ndarray], packed: dict[str, PackedScreenData]
) -> CandidateBundle:
    """Recompute the exact matched-support A/C vectors used by the factorial."""

    targets = np.asarray(cache["target_masks"])
    visible = np.asarray(cache["visible_masks"])
    neighbors = np.asarray(cache["neighbor_tokens"])
    neighbor_valid = np.asarray(cache["neighbor_valid"])
    p2_data = packed["P2_same_grid_pool"]
    valid = p2_data.target_valid.cpu().numpy().astype(bool)
    masked = p2_data.masked_tokens.cpu().numpy().astype(np.float32)
    teacher = p2_data.teacher_tokens.cpu().numpy().astype(np.float32)
    positions = p2_data.positions_yx.cpu().numpy().astype(np.int64)
    mask_ids = p2_data.mask_ids.cpu().numpy().astype(np.int64)
    transport_features = packed["P3_partial_transport"].gate_features.cpu().numpy().astype(np.float32)
    affine = np.zeros_like(masked, dtype=np.float32)
    same_grid = np.zeros_like(masked, dtype=np.float32)
    available = np.zeros(valid.shape, dtype=bool)
    for center in range(CENTERS):
        cursor = 0
        for mask_id in range(2):
            target = targets[center, mask_id].astype(bool)
            target_yx = np.argwhere(target)
            count = len(target_yx)
            donor_affine = np.zeros((4, count, FEATURE_DIM), dtype=np.float32)
            donor_same = np.zeros_like(donor_affine)
            observed = np.zeros((4, count), dtype=bool)
            logits = np.full((4, count), -np.inf, dtype=np.float64)
            center_tokens = masked[center, cursor : cursor + count]
            # The packed target row order is mask 0 followed by mask 1; the
            # dense masked cache is indexed by the same center/mask grid.
            center_tokens = np.asarray(cache["masked_tokens"][center, mask_id], dtype=np.float32)
            center_valid = visible[center, mask_id].astype(bool)
            for donor in range(4):
                donor_tokens = np.asarray(neighbors[center, donor], dtype=np.float32)
                donor_ok = neighbor_valid[center, donor].astype(bool)
                transform, matches = fit_visible_anchor_transport(
                    center_tokens,
                    donor_tokens,
                    target,
                    center_valid=center_valid,
                    neighbor_valid=donor_ok,
                )
                transported, affine_observed = bilinear_transport(
                    donor_tokens, target, transform, neighbor_valid=donor_ok
                )
                y, x = target_yx[:, 0], target_yx[:, 1]
                donor_affine[donor] = transported
                donor_same[donor] = donor_tokens[y, x]
                observed[donor] = affine_observed
                same_valid = donor_ok[y, x]
                observed[donor] &= same_valid
                logits[donor] = donor_confidence_logits(
                    matches, transform, target, affine_observed
                )
            result_c = uniform_consensus(donor_affine, observed, logits)
            result_a = uniform_consensus(donor_same, observed, logits)
            rows = slice(cursor, cursor + count)
            affine[center, rows] = result_c.features
            same_grid[center, rows] = result_a.features
            available[center, rows] = result_c.available
            cursor += count
        if cursor != int(valid[center].sum()):
            raise RuntimeError(f"candidate packing diverged at center {center}")
    if not np.isfinite(affine).all() or not np.isfinite(same_grid).all():
        raise RuntimeError("matched affine candidate contains nonfinite values")
    return CandidateBundle(
        affine=affine,
        same_grid=same_grid,
        available=available,
        teacher=teacher,
        masked=masked,
        valid=valid,
        positions=positions,
        mask_ids=mask_ids,
        transport_features=transport_features,
    )


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    denominator = np.maximum(left_norm * right_norm, 1e-12)
    return 1.0 - np.clip(np.sum(left * right, axis=-1) / denominator, -1.0, 1.0)


def _normalized_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.linalg.norm(left, axis=-1) + np.linalg.norm(right, axis=-1), 1e-12)
    return np.linalg.norm(left - right, axis=-1) / denominator


def _policy_features(
    p0: np.ndarray, p2: np.ndarray, bundle: CandidateBundle
) -> np.ndarray:
    if p0.shape != p2.shape or p0.shape != bundle.teacher.shape:
        raise ValueError("base predictions and candidate bundle are not aligned")
    features = np.zeros((*bundle.valid.shape, FEATURE_COUNT), dtype=np.float32)
    features[..., :21] = bundle.transport_features
    features[..., 21] = _cosine_distance(p2, p0)
    features[..., 22] = _normalized_l2(p2, p0)
    features[..., 23] = _cosine_distance(bundle.affine, p2)
    features[..., 24] = _normalized_l2(bundle.affine, p2)
    features[..., 25] = _cosine_distance(bundle.affine, bundle.same_grid)
    features[..., 26] = _normalized_l2(bundle.affine, bundle.same_grid)
    features[..., 27] = _cosine_distance(p2, bundle.masked)
    features[..., 28] = _cosine_distance(bundle.affine, bundle.masked)
    features[..., 29] = bundle.mask_ids.astype(np.float32)
    features[..., 30] = bundle.positions[..., 0].astype(np.float32) / 26.0
    features[..., 31] = bundle.positions[..., 1].astype(np.float32) / 26.0
    features[~bundle.valid] = 0.0
    if not np.isfinite(features).all():
        raise RuntimeError("PCAR policy features became nonfinite")
    return features


def _policy_costs(p2: np.ndarray, bundle: CandidateBundle) -> np.ndarray:
    actions = exact_convex_actions_numpy(p2, bundle.affine, available=bundle.available)
    costs = np.zeros((*bundle.valid.shape, ACTION_COUNT), dtype=np.float32)
    # The NumPy path is kept in float64 for the metric then written back to
    # float32 for the policy objective, exactly as the protocol specifies raw
    # full-dimensional cosine costs rather than a binary win target.
    p2_t = torch.from_numpy(actions)
    teacher_t = torch.from_numpy(bundle.teacher)
    with torch.inference_mode():
        costs[:] = action_costs(p2_t, teacher_t).numpy().astype(np.float32)
    costs[~bundle.valid] = 0.0
    if not np.isfinite(costs).all():
        raise RuntimeError("PCAR action costs became nonfinite")
    return costs


def _fit_subset(
    data: PackedScreenData,
    rows: np.ndarray,
    *,
    arm: str,
    base: BaseCompletionModel | None,
    updates: int,
    device: str,
) -> tuple[nn.Module, list[float]]:
    """Fit a locked screen model on exactly the supplied scenario groups."""

    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("subset fit requires nonempty one-dimensional rows")
    model = new_screen_model(arm, base=base, seed=42, device=device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-4,
        weight_decay=0.01,
    )
    generator = np.random.default_rng(42)
    order = np.empty(0, dtype=np.int64)
    cursor = 0
    history: list[float] = []
    for _ in range(updates):
        if cursor >= len(order):
            order = generator.permutation(len(rows))
            cursor = 0
        local = order[cursor : cursor + BATCH_SIZE]
        cursor += len(local)
        batch = data.select(rows[local]).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = reconstruction_loss(output, batch.teacher_tokens, batch.target_valid)
        if not torch.isfinite(loss.objective):
            raise RuntimeError("nested base loss became nonfinite")
        loss.objective.backward()
        optimizer.step()
        history.append(float(loss.objective.detach().cpu()))
    model.eval()
    return model, history


def _predict_dense(model: nn.Module, data: PackedScreenData, rows: np.ndarray, device: str) -> np.ndarray:
    selected = data.select(rows)
    with torch.inference_mode():
        output = model(selected.to(device)).prediction.detach().cpu().numpy().astype(np.float32)
    return output


def _fit_policy(
    features: np.ndarray,
    costs: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
    *,
    updates: int,
    device: str,
) -> tuple[PCARPolicy, list[float]]:
    if features.shape[-1] != FEATURE_COUNT or costs.shape[-1] != ACTION_COUNT:
        raise ValueError("policy tensors differ from locked 32/5 geometry")
    if rows.ndim != 1 or len(rows) == 0 or not valid[rows].any(axis=1).all():
        raise ValueError("policy fit requires nonempty valid centers")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = PCARPolicy().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    feature_tensor = torch.from_numpy(features)
    cost_tensor = torch.from_numpy(costs)
    valid_tensor = torch.from_numpy(valid)
    generator = np.random.default_rng(42)
    order = np.empty(0, dtype=np.int64)
    cursor = 0
    history: list[float] = []
    for _ in range(updates):
        if cursor >= len(order):
            order = generator.permutation(len(rows))
            cursor = 0
        local = order[cursor : cursor + BATCH_SIZE]
        cursor += len(local)
        batch_rows = rows[local]
        batch_features = feature_tensor[batch_rows].to(device=device, dtype=torch.float32)
        batch_costs = cost_tensor[batch_rows].to(device=device, dtype=torch.float32)
        batch_valid = valid_tensor[batch_rows].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = center_balanced_expected_cost(logits, batch_costs, batch_valid)
        if not torch.isfinite(loss):
            raise RuntimeError("PCAR policy loss became nonfinite")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    return model, history


def _nested_pair_predictions(
    packed: dict[str, PackedScreenData],
    fold_ids: np.ndarray,
    *,
    updates: int,
    device: str,
) -> tuple[dict[tuple[int, int], dict[int, tuple[np.ndarray, np.ndarray]]], list[dict[str, Any]]]:
    """Fit each unordered excluded-fold pair once and predict both members.

    The PCAR protocol permits the checkpoint for ``{O,J}`` to serve both
    orientations.  Reusing it is important: fitting the 20 directed pairs
    would double the declared optimizer budget and would no longer be the
    locked 9,000-update experiment.
    """

    result: dict[tuple[int, int], dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    receipts: list[dict[str, Any]] = []
    for left in range(FOLDS):
        for right in range(left + 1, FOLDS):
            train_rows = np.flatnonzero((fold_ids != left) & (fold_ids != right))
            p0, p0_history = _fit_subset(
                packed["P0_center_only"], train_rows, arm="P0_center_only", base=None,
                updates=updates, device=device
            )
            p2, p2_history = _fit_subset(
                packed["P2_same_grid_pool"], train_rows, arm="P2_same_grid_pool", base=p0,
                updates=updates, device=device
            )
            excluded_predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for excluded in (left, right):
                rows = np.flatnonzero(fold_ids == excluded)
                excluded_predictions[excluded] = (
                    _predict_dense(p0, packed["P0_center_only"], rows, device),
                    _predict_dense(p2, packed["P2_same_grid_pool"], rows, device),
                )
            result[(left, right)] = excluded_predictions
            receipts.append({
                "excluded_pair": [left, right],
                "fit_rows": int(len(train_rows)),
                "predicted_rows": {str(key): int(len(np.flatnonzero(fold_ids == key))) for key in (left, right)},
                "p0_updates": len(p0_history),
                "p2_updates": len(p2_history),
                "outer_held_teacher_read_during_base_fit": 0,
            })
            del p0, p2
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    if len(result) != 10:
        raise RuntimeError("nested PCAR pair cache does not contain exactly ten unordered pairs")
    return result, receipts


def _bundle_with_affine(bundle: CandidateBundle, affine: np.ndarray) -> CandidateBundle:
    if affine.shape != bundle.affine.shape:
        raise ValueError("replacement candidate does not share bundle geometry")
    return CandidateBundle(
        affine=affine,
        same_grid=bundle.same_grid,
        available=bundle.available,
        teacher=bundle.teacher,
        masked=bundle.masked,
        valid=bundle.valid,
        positions=bundle.positions,
        mask_ids=bundle.mask_ids,
        transport_features=bundle.transport_features,
    )


def _bundle_control(bundle: CandidateBundle, control: PackedScreenData) -> CandidateBundle:
    if control.gate_features is None:
        raise ValueError("PCAR control requires frozen transport features")
    evidence = control.evidence_tokens.cpu().numpy().astype(np.float32)
    available = control.evidence_available.cpu().numpy().astype(bool)
    gate = control.gate_features.cpu().numpy().astype(np.float32)
    if evidence.shape != bundle.affine.shape or available.shape != bundle.available.shape:
        raise ValueError("control evidence geometry differs from nominal bundle")
    return CandidateBundle(
        affine=evidence,
        same_grid=bundle.same_grid,
        available=available,
        teacher=bundle.teacher,
        masked=bundle.masked,
        valid=bundle.valid,
        positions=bundle.positions,
        mask_ids=bundle.mask_ids,
        transport_features=gate,
    )


def _evaluate_policy(
    policy: PCARPolicy,
    p0: np.ndarray,
    p2: np.ndarray,
    bundle: CandidateBundle,
    *,
    device: str,
) -> dict[str, Any]:
    features = _policy_features(p0, p2, bundle)
    costs = _policy_costs(p2, bundle)
    with torch.inference_mode():
        logits = policy(torch.from_numpy(features).to(device=device, dtype=torch.float32)).cpu()
        actions = hard_actions(
            logits,
            torch.from_numpy(bundle.valid),
            available=torch.from_numpy(bundle.available),
        ).numpy()
    selected = np.take_along_axis(costs, actions[..., None], axis=-1)[..., 0]
    valid = bundle.valid
    denominator = valid.sum(axis=1)
    per_center = (selected * valid).sum(axis=1) / denominator
    p2_per_center = (costs[..., 0] * valid).sum(axis=1) / denominator
    candidate_per_center = (costs[..., -1] * valid).sum(axis=1) / denominator
    return {
        "features": features,
        "costs": costs,
        "actions": actions,
        "selected": selected,
        "per_center": per_center.astype(np.float64),
        "p2_per_center": p2_per_center.astype(np.float64),
        "candidate_per_center": candidate_per_center.astype(np.float64),
        "valid": valid,
    }


def _load_screen_models(fold: int, device: str) -> tuple[nn.Module, nn.Module]:
    return (
        _load_screen_checkpoint("P0_center_only", fold, device),
        _load_screen_checkpoint("P2_same_grid_pool", fold, device),
    )


def _paired_bootstrap_lower(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20260917,
) -> tuple[float, float, float]:
    if reference.shape != candidate.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError("bootstrap inputs must be aligned centre vectors")
    rng = np.random.default_rng(seed)
    differences = reference - candidate
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        values[start : start + count] = differences[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _save_scientific_csv(
    path: Path,
    sample_ids: list[str],
    scenarios: list[str],
    fold_ids: np.ndarray,
    nominal: dict[str, np.ndarray],
    controls: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = ["center_index", "sample_id", "scenario", "held_fold"]
    names = ["P2", "C_affine", "PCAR"] + list(controls)
    for name in names:
        fields.extend([f"{name}_center_cosine", f"{name}_actions_alpha0", f"{name}_actions_alpha025", f"{name}_actions_alpha050", f"{name}_actions_alpha075", f"{name}_actions_alpha1"])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(sample_ids)):
            row: dict[str, Any] = {
                "center_index": index,
                "sample_id": sample_ids[index],
                "scenario": scenarios[index],
                "held_fold": int(fold_ids[index]),
            }
            for name in names:
                values = nominal if name in {"P2", "C_affine", "PCAR"} else controls[name]
                row[f"{name}_center_cosine"] = float(values["per_center"][index] if name == "PCAR" else values["p2_per_center"][index] if name == "P2" else values["candidate_per_center"][index])
                actions = values.get("actions")
                if actions is None:
                    fractions = np.zeros(5, dtype=np.float64)
                else:
                    fractions = np.eye(5, dtype=np.float64)[actions[index][values["valid"][index]]].mean(axis=0)
                for action_index, suffix in enumerate(("alpha0", "alpha025", "alpha050", "alpha075", "alpha1")):
                    row[f"{name}_actions_{suffix}"] = float(fractions[action_index])
            writer.writerow(row)


def _scientific_screen(output: Path, *, device: str) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    _verify_execution_lock()
    cache, locks, packed = _load_packed()
    bundle = _matched_affine_candidates(cache, packed)
    alignment = _validate_candidate_alignment(bundle, locks)
    split_pairs = outer_scenario_splits(locks["cache"]["scenarios"])
    fold_ids = np.empty(CENTERS, dtype=np.int64)
    for fold, (_, held) in enumerate(split_pairs):
        fold_ids[held] = fold
    pair_predictions, pair_receipts = _nested_pair_predictions(
        packed, fold_ids, updates=BASE_UPDATES, device=device
    )
    pcar_per_center = np.full(CENTERS, np.nan, dtype=np.float64)
    p2_per_center = np.full(CENTERS, np.nan, dtype=np.float64)
    c_per_center = np.full(CENTERS, np.nan, dtype=np.float64)
    action_matrix = np.zeros((CENTERS, bundle.valid.shape[1]), dtype=np.int64)
    control_results: dict[str, dict[str, np.ndarray]] = {}
    capacity_per_center = np.full(CENTERS, np.nan, dtype=np.float64)
    fold_stats: dict[str, Any] = {}
    policy_receipts: list[dict[str, Any]] = []
    sample_ids = locks["cache"]["sample_ids"]
    scenarios = locks["cache"]["scenarios"]
    spatial_control = screen.spatial_reassignment_control(packed["P3_partial_transport"], sample_ids)
    full_controls = {
        "repeated_masked_center": packed["repeated_masked_center"],
        "wrong_track": packed["wrong_track"],
        "spatial_reassignment": spatial_control,
    }
    for outer_fold in range(FOLDS):
        meta_features = np.zeros((*bundle.valid.shape, FEATURE_COUNT), dtype=np.float32)
        meta_costs = np.zeros((*bundle.valid.shape, ACTION_COUNT), dtype=np.float32)
        meta_valid = np.zeros(bundle.valid.shape, dtype=bool)
        meta_cap_features = np.zeros_like(meta_features)
        meta_cap_costs = np.zeros_like(meta_costs)
        for inner_fold in range(FOLDS):
            if inner_fold == outer_fold:
                continue
            pair = tuple(sorted((outer_fold, inner_fold)))
            try:
                p0_dense, p2_dense = pair_predictions[pair][inner_fold]
            except KeyError as exc:
                raise RuntimeError(f"missing nested prediction for pair {pair}/{inner_fold}") from exc
            rows = np.flatnonzero(fold_ids == inner_fold)
            sliced = _slice_bundle(bundle, rows)
            features = _policy_features(p0_dense, p2_dense, sliced)
            costs = _policy_costs(p2_dense, sliced)
            capacity_bundle = _bundle_with_affine(sliced, sliced.same_grid)
            capacity_features = _policy_features(p0_dense, p2_dense, capacity_bundle)
            capacity_costs = _policy_costs(p2_dense, capacity_bundle)
            meta_features[rows] = features
            meta_costs[rows] = costs
            meta_valid[rows] = sliced.valid
            meta_cap_features[rows] = capacity_features
            meta_cap_costs[rows] = capacity_costs
        if meta_valid[fold_ids == outer_fold].any():
            raise RuntimeError("outer-held centers entered PCAR policy meta fit")
        meta_rows = np.flatnonzero(meta_valid.any(axis=1))
        policy, policy_history = _fit_policy(
            meta_features, meta_costs, meta_valid, meta_rows,
            updates=POLICY_UPDATES, device=device
        )
        capacity_policy, capacity_history = _fit_policy(
            meta_cap_features, meta_cap_costs, meta_valid, meta_rows,
            updates=POLICY_UPDATES, device=device
        )
        held = np.flatnonzero(fold_ids == outer_fold)
        p0_outer, p2_outer = _load_screen_models(outer_fold, device)
        p0_dense = _predict_dense(p0_outer, packed["P0_center_only"], held, device)
        p2_dense = _predict_dense(p2_outer, packed["P2_same_grid_pool"], held, device)
        held_bundle = _slice_bundle(bundle, held)
        nominal = _evaluate_policy(policy, p0_dense, p2_dense, held_bundle, device=device)
        capacity = _evaluate_policy(capacity_policy, p0_dense, p2_dense, _bundle_with_affine(held_bundle, held_bundle.same_grid), device=device)
        pcar_per_center[held] = nominal["per_center"]
        p2_per_center[held] = nominal["p2_per_center"]
        c_per_center[held] = nominal["candidate_per_center"]
        action_matrix[held] = nominal["actions"]
        capacity_per_center[held] = capacity["per_center"]
        valid_tokens = nominal["valid"]
        selected = nominal["selected"]
        c_cost = nominal["costs"][..., -1]
        rescue = int(((selected < c_cost) & valid_tokens).sum())
        harm = int(((selected > c_cost) & valid_tokens).sum())
        fold_stats[str(outer_fold)] = {
            "held_centers": int(len(held)),
            "valid_tokens": int(valid_tokens.sum()),
            "P2_mean_center_error": float(nominal["p2_per_center"].mean()),
            "C_affine_mean_center_error": float(nominal["candidate_per_center"].mean()),
            "PCAR_mean_center_error": float(nominal["per_center"].mean()),
            "capacity_same_grid_policy_mean_center_error": float(capacity["per_center"].mean()),
            "rescues_vs_C": rescue,
            "harms_vs_C": harm,
            "net_corrections_vs_C": rescue - harm,
            "action_fractions": {
                str(alpha): float(np.mean(nominal["actions"][valid_tokens] == index))
                for index, alpha in enumerate(ACTION_ALPHAS)
            },
            "policy_updates": len(policy_history),
            "capacity_policy_updates": len(capacity_history),
            "outer_labels_read_during_fit": 0,
        }
        policy_receipts.append({
            "outer_fold": outer_fold,
            "meta_centers": int(len(meta_rows)),
            "outer_held_teacher_read_during_policy_fit": 0,
            "policy_history_final": float(policy_history[-1]),
            "capacity_policy_history_final": float(capacity_history[-1]),
        })
        for name, control_data in full_controls.items():
            control_bundle = _bundle_control(bundle, control_data)
            control_eval = _evaluate_policy(policy, p0_dense, p2_dense, _slice_bundle(control_bundle, held), device=device)
            if name not in control_results:
                control_results[name] = {
                    "per_center": np.full(CENTERS, np.nan, dtype=np.float64),
                    "p2_per_center": np.full(CENTERS, np.nan, dtype=np.float64),
                    "candidate_per_center": np.full(CENTERS, np.nan, dtype=np.float64),
                    "actions": np.zeros((CENTERS, bundle.valid.shape[1]), dtype=np.int64),
                    "valid": np.zeros(bundle.valid.shape, dtype=bool),
                }
            control_results[name]["per_center"][held] = control_eval["per_center"]
            control_results[name]["p2_per_center"][held] = control_eval["p2_per_center"]
            control_results[name]["candidate_per_center"][held] = control_eval["candidate_per_center"]
            control_results[name]["actions"][held] = control_eval["actions"]
            control_results[name]["valid"][held] = control_eval["valid"]
        torch.save(
            {"state_dict": policy.state_dict(), "outer_fold": outer_fold, "seed": 42, "updates": POLICY_UPDATES},
            output / f"policy_fold{outer_fold}.pt",
        )
        del policy, capacity_policy, p0_outer, p2_outer, p0_dense, p2_dense
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not np.isfinite(pcar_per_center).all() or not np.isfinite(c_per_center).all():
        raise RuntimeError("PCAR outer population is incomplete")
    global_p2 = float(p2_per_center.mean())
    global_c = float(c_per_center.mean())
    global_pcar = float(pcar_per_center.mean())
    bootstrap_mean, bootstrap_lower, bootstrap_upper = _paired_bootstrap_lower(c_per_center, pcar_per_center)
    token_rescues = int(sum(fold_stats[str(f)]["rescues_vs_C"] for f in range(FOLDS)))
    token_harms = int(sum(fold_stats[str(f)]["harms_vs_C"] for f in range(FOLDS)))
    net_by_fold = [int(fold_stats[str(f)]["net_corrections_vs_C"]) for f in range(FOLDS)]
    scenario_array = np.asarray(scenarios)
    better_scenarios = int(sum(
        float(pcar_per_center[scenario_array == scenario].mean()) < float(p2_per_center[scenario_array == scenario].mean())
        for scenario in sorted(set(scenarios))
    ))
    nominal_gain = global_p2 - global_pcar
    controls_summary: dict[str, Any] = {}
    for name, values in control_results.items():
        control_mean = float(values["per_center"].mean())
        control_gain = global_p2 - control_mean
        controls_summary[name] = {
            "mean_center_error": control_mean,
            "gain_fraction_vs_nominal": float(control_gain / nominal_gain) if nominal_gain > 0 else None,
            "available_centers": int(values["valid"].any(axis=1).sum()),
        }
    spatial_removed = (
        (float(control_results["spatial_reassignment"]["per_center"].mean()) - global_pcar) / nominal_gain
        if nominal_gain > 0 else None
    )
    gates = {
        "alignment_replay": True,
        "primary_vs_P2_relative_reduction_min_0.05": (global_p2 - global_pcar) / global_p2 >= 0.05,
        "primary_scenarios_strictly_better_min_10": better_scenarios >= 10,
        "incremental_vs_C_relative_reduction_min_0.02": (global_c - global_pcar) / global_c >= 0.02,
        "incremental_mean_error_ceiling": global_pcar <= 0.20877999674473652,
        "positive_net_each_outer_fold": all(value > 0 for value in net_by_fold),
        "paired_bootstrap_lower_positive": bootstrap_lower > 0,
        "rescue_harm_ratio_min_3": token_harms == 0 or token_rescues / token_harms >= 3.0,
        "same_grid_capacity_control_measured": np.isfinite(capacity_per_center).all(),
        "repeated_control_measured": np.isfinite(control_results["repeated_masked_center"]["per_center"]).all(),
        "wrong_track_control_measured": np.isfinite(control_results["wrong_track"]["per_center"]).all(),
        "spatial_control_removed_fraction_min_0.5": spatial_removed is not None and spatial_removed >= 0.5,
        "zero_outer_label_access": True,
    }
    result = {
        "status": "PCAR_V5_SCIENTIFIC_SCREEN_COMPLETE",
        "scientific": True,
        "task_training_authorized": False,
        "retained_arftr_changed": False,
        "alignment": alignment,
        "metrics": {
            "P2_same_grid_mean_center_error": global_p2,
            "C_affine_uniform_mean_center_error": global_c,
            "PCAR_mean_center_error": global_pcar,
            "PCAR_relative_reduction_vs_P2": (global_p2 - global_pcar) / global_p2,
            "PCAR_relative_reduction_vs_C": (global_c - global_pcar) / global_c,
            "better_scenarios_vs_P2": better_scenarios,
            "rescues_vs_C": token_rescues,
            "harms_vs_C": token_harms,
            "net_corrections_by_outer_fold": net_by_fold,
            "paired_bootstrap_delta_C_minus_PCAR": {
                "mean": bootstrap_mean,
                "lower_95": bootstrap_lower,
                "upper_95": bootstrap_upper,
                "replicates": 10_000,
            },
        },
        "controls": controls_summary,
        "spatial_removed_fraction": spatial_removed,
        "gates": {**gates, "all_promotion_gates": all(gates.values())},
        "folds": fold_stats,
        "pair_receipts": pair_receipts,
        "policy_receipts": policy_receipts,
        "zero_access_counters": {
            "task_labels": 0,
            "ARFTR_probability_arrays": 0,
            "annotation_derived_support_categories": 0,
            "pose_outputs": 0,
            "clip_index": 0,
            "held_outer_teacher_during_fit": 0,
        },
        "runtime_seconds": {"total": time.perf_counter() - started},
    }
    request = {
        "status": "PCAR_V5_SCIENTIFIC_SCREEN_REQUEST_LOCKED",
        "stage": "all",
        "device": device,
        "execution_lock_sha256": sha256_file(LOCK),
        "protocol_sha256": sha256_file(PROTOCOL),
        "outer_held_teacher_access_during_fit": 0,
        "zero_access_contract": result["zero_access_counters"],
    }
    write_json_exclusive(output / "request.json", request)
    write_json_exclusive(output / "summary.json", result)
    np.savez_compressed(
        output / "per_center.npz",
        sample_ids=np.asarray(sample_ids),
        scenarios=np.asarray(scenarios),
        held_fold=fold_ids,
        p2=p2_per_center,
        c_affine=c_per_center,
        pcar=pcar_per_center,
        pcar_actions=action_matrix,
        capacity=capacity_per_center,
    )
    _save_scientific_csv(output / "per_center.csv", sample_ids, scenarios, fold_ids, {"per_center": pcar_per_center, "p2_per_center": p2_per_center, "candidate_per_center": c_per_center, "actions": action_matrix, "valid": bundle.valid}, control_results)
    (output / "source_snapshot").mkdir()
    for source in (Path(__file__), ROOT / "src/hac/pcar.py", ROOT / "src/hac/center_completion_screen.py", ROOT / "src/hac/center_evidence_completion.py", PROTOCOL, LOCK):
        shutil.copy2(source, output / "source_snapshot" / source.name)
    return result


def _load_screen_checkpoint(arm: str, fold: int, device: str) -> nn.Module:
    path = SCREEN / "checkpoints" / f"fold{fold}_{arm}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != arm or checkpoint.get("held_fold") != fold or checkpoint.get("updates") != 400:
        raise RuntimeError(f"screen checkpoint contract changed: {path.name}")
    if arm == "P0_center_only":
        model = new_screen_model("P0_center_only", device=device)
    else:
        dummy = new_screen_model("P0_center_only", device="cpu")
        model = new_screen_model(arm, base=dummy, device=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _validate_candidate_alignment(bundle: CandidateBundle, locks: dict[str, Any]) -> dict[str, float]:
    factorial = locks["factorial"]
    teacher = bundle.teacher[bundle.valid]
    affine = bundle.affine[bundle.valid]
    same = bundle.same_grid[bundle.valid]
    c_error = float(_cosine_distance(affine, teacher).mean())
    a_error = float(_cosine_distance(same, teacher).mean())
    expected_c = float(factorial["mean_equal_center_cosine_error"]["C"])
    expected_a = float(factorial["mean_equal_center_cosine_error"]["A"])
    # The vector replay is token-weighted; compare against the saved tokenwise
    # means as a broad alignment check and keep the centre means for the report.
    if abs(c_error - float(factorial["mean_token_weighted_cosine_error"]["C"])) > 2e-5:
        raise RuntimeError(f"affine candidate replay mismatch: {c_error} vs saved token mean")
    if abs(a_error - float(factorial["mean_token_weighted_cosine_error"]["A"])) > 2e-5:
        raise RuntimeError(f"same-grid candidate replay mismatch: {a_error} vs saved token mean")
    return {
        "token_weighted_affine_error": c_error,
        "token_weighted_same_grid_error": a_error,
        "saved_equal_center_affine_error": expected_c,
        "saved_equal_center_same_grid_error": expected_a,
    }


def _timing_smoke(
    output: Path,
    *,
    device: str,
    smoke_updates: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    _verify_execution_lock()
    cache, locks, packed = _load_packed()
    bundle = _matched_affine_candidates(cache, packed)
    alignment = _validate_candidate_alignment(bundle, locks)
    # Use the authoritative split helper rather than parsing scenario names.
    split_pairs = outer_scenario_splits(locks["cache"]["scenarios"])
    fold_ids = np.empty(CENTERS, dtype=np.int64)
    for fold, (_, held) in enumerate(split_pairs):
        fold_ids[held] = fold
    meta_features = np.zeros((*bundle.valid.shape, FEATURE_COUNT), dtype=np.float32)
    meta_costs = np.zeros((*bundle.valid.shape, ACTION_COUNT), dtype=np.float32)
    meta_valid = np.zeros(bundle.valid.shape, dtype=bool)
    pair_receipts: list[dict[str, Any]] = []
    # One complete outer-training rotation exercises all four inner groups,
    # while remaining far below the full scientific budget.
    outer_fold = 0
    for inner_fold in range(1, FOLDS):
        train_rows = np.flatnonzero((fold_ids != outer_fold) & (fold_ids != inner_fold))
        held_rows = np.flatnonzero(fold_ids == inner_fold)
        p0, p0_history = _fit_subset(
            packed["P0_center_only"], train_rows, arm="P0_center_only", base=None,
            updates=smoke_updates, device=device
        )
        p2, p2_history = _fit_subset(
            packed["P2_same_grid_pool"], train_rows, arm="P2_same_grid_pool", base=p0,
            updates=smoke_updates, device=device
        )
        p0_dense = _predict_dense(p0, packed["P0_center_only"], held_rows, device)
        p2_dense = _predict_dense(p2, packed["P2_same_grid_pool"], held_rows, device)
        # Copy only the held inner group into the full meta population.  Outer
        # fold 0 is absent by construction and is never read for policy fit.
        features = _policy_features(p0_dense, p2_dense, _slice_bundle(bundle, held_rows))
        costs = _policy_costs(p2_dense, _slice_bundle(bundle, held_rows))
        meta_features[held_rows] = features
        meta_costs[held_rows] = costs
        meta_valid[held_rows] = bundle.valid[held_rows]
        pair_receipts.append({
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "fit_rows": int(len(train_rows)),
            "meta_rows": int(len(held_rows)),
            "p0_updates": len(p0_history),
            "p2_updates": len(p2_history),
            "outer_rows_read_for_fit": 0,
            "inner_held_teacher_used_for_costs": True,
        })
        del p0, p2, p0_dense, p2_dense
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    meta_rows = np.flatnonzero(meta_valid.any(axis=1))
    policy, policy_history = _fit_policy(
        meta_features, meta_costs, meta_valid, meta_rows,
        updates=smoke_updates, device=device
    )
    outer_rows = np.flatnonzero(fold_ids == outer_fold)
    p0_outer = _load_screen_checkpoint("P0_center_only", outer_fold, device)
    p2_outer = _load_screen_checkpoint("P2_same_grid_pool", outer_fold, device)
    p0_outer_dense = _predict_dense(p0_outer, packed["P0_center_only"], outer_rows, device)
    p2_outer_dense = _predict_dense(p2_outer, packed["P2_same_grid_pool"], outer_rows, device)
    outer_bundle = _slice_bundle(bundle, outer_rows)
    outer_features = _policy_features(p0_outer_dense, p2_outer_dense, outer_bundle)
    outer_costs = _policy_costs(p2_outer_dense, outer_bundle)
    with torch.inference_mode():
        logits = policy(torch.from_numpy(outer_features).to(device=device, dtype=torch.float32)).cpu()
        actions = hard_actions(
            logits,
            torch.from_numpy(outer_bundle.valid),
            available=torch.from_numpy(outer_bundle.available),
        ).numpy()
    selected_cost = np.take_along_axis(outer_costs, actions[..., None], axis=-1)[..., 0]
    valid_outer = outer_bundle.valid
    selected_mean = float(selected_cost[valid_outer].mean())
    direct_c_mean = float(outer_costs[..., -1][valid_outer].mean())
    p2_mean = float(outer_costs[..., 0][valid_outer].mean())
    # Exercise the explicit retain path on a synthetic all-tie policy output.
    tie_actions = hard_actions(
        torch.zeros_like(logits), torch.from_numpy(outer_bundle.valid),
        available=torch.from_numpy(outer_bundle.available),
    ).numpy()
    if np.any(tie_actions[valid_outer] != 0):
        raise RuntimeError("PCAR tie action did not retain P2")
    elapsed = time.perf_counter() - started
    # The smoke uses nine fits at ``smoke_updates`` (eight nested bases plus
    # one policy).  The locked scientific screen uses 20 base fits at 400
    # updates and five policy fits at 200 updates: 9,000 optimizer updates in
    # total.  Scale by update budget, not merely by fit count.
    smoke_optimizer_updates = 9 * smoke_updates
    scientific_optimizer_updates = 20 * BASE_UPDATES + 5 * POLICY_UPDATES
    projected = elapsed * (scientific_optimizer_updates / smoke_optimizer_updates) / 60.0
    result = {
        "status": "PCAR_V5_TIMING_SMOKE_PASS" if projected <= 20.0 else "PCAR_V5_TIMING_SMOKE_FAIL_RUNTIME",
        "scientific": False,
        "task_training_authorized": False,
        "retained_arftr_changed": False,
        "outer_labels_read_during_fit": 0,
        "outer_fold_used_for_smoke": outer_fold,
        "pair_receipts": pair_receipts,
        "policy_updates": len(policy_history),
        "alignment": alignment,
        "smoke_metrics": {
            "outer_p2_token_error": p2_mean,
            "outer_direct_affine_token_error": direct_c_mean,
            "outer_pcar_selected_token_error": selected_mean,
            "outer_valid_tokens": int(valid_outer.sum()),
            "outer_action_fractions": {
                str(alpha): float(np.mean(actions[valid_outer] == index))
                for index, alpha in enumerate(ACTION_ALPHAS)
            },
        },
        "runtime_seconds": elapsed,
        "projected_full_screen_minutes": projected,
        "projection_basis": (
            f"{scientific_optimizer_updates} scientific optimizer updates / "
            f"{smoke_optimizer_updates} smoke updates; conservative linear estimate"
        ),
        "zero_access_counters": {
            "task_labels": 0,
            "ARFTR_probability_arrays": 0,
            "annotation_derived_support_categories": 0,
            "pose_outputs": 0,
            "clip_index": 0,
            "held_outer_teacher_during_fit": 0,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    request = {
        "status": "PCAR_V5_TIMING_SMOKE_REQUEST_LOCKED",
        "stage": "timing-smoke",
        "smoke_updates": smoke_updates,
        "device": device,
        "execution_lock_sha256": sha256_file(LOCK),
        "protocol_sha256": sha256_file(PROTOCOL),
        "zero_access_contract": result["zero_access_counters"],
    }
    write_json_exclusive(output / "request.json", request)
    write_json_exclusive(output / "summary.json", result)
    (output / "source_snapshot").mkdir()
    for source in (Path(__file__), ROOT / "src/hac/pcar.py", ROOT / "src/hac/center_completion_screen.py", ROOT / "src/hac/center_evidence_completion.py", PROTOCOL, LOCK):
        shutil.copy2(source, output / "source_snapshot" / source.name)
    return result


def _slice_bundle(bundle: CandidateBundle, rows: np.ndarray) -> CandidateBundle:
    return CandidateBundle(
        affine=bundle.affine[rows],
        same_grid=bundle.same_grid[rows],
        available=bundle.available[rows],
        teacher=bundle.teacher[rows],
        masked=bundle.masked[rows],
        valid=bundle.valid[rows],
        positions=bundle.positions[rows],
        mask_ids=bundle.mask_ids[rows],
        transport_features=bundle.transport_features[rows],
    )


def main() -> None:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.stage == "validate":
        _verify_execution_lock()
        _verify_prerequisites()
        print(json.dumps({"status": "PCAR_V5_PREFLIGHT_PASS", "device": args.device}, sort_keys=True))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output_dir}")
    if args.stage == "all":
        result = _scientific_screen(args.output_dir, device=args.device)
        print(json.dumps({"status": result["status"], "output": str(args.output_dir), "all_promotion_gates": result["gates"]["all_promotion_gates"]}, sort_keys=True))
        return
    result = _timing_smoke(args.output_dir, device=args.device, smoke_updates=args.smoke_updates)
    print(json.dumps({"status": result["status"], "output": str(args.output_dir), "projected_full_screen_minutes": result["projected_full_screen_minutes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
