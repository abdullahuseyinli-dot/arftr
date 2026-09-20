"""Label-blind feasibility smoke for Persistent Token Gauge (PTG).

This script only reads the verified dense-token caches and the fixed 128-center
witness selection.  It never opens labels, ARFTR predictions, support
annotations, or human-review fields.  The output is a measurement receipt, not
a task-classification result and cannot authorize a model fit by itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runs/research_20260917/ptg_preflight_v1"
DENSE = ROOT / ".runs/research_20260908/dense_tokens_full"
PAIRED = ROOT / ".runs/research_20260908/native4k_paired_full/features"
SELECTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def _grid() -> np.ndarray:
    y, x = np.meshgrid(np.arange(12, dtype=np.float32), np.arange(12, dtype=np.float32), indexing="ij")
    return np.stack((x.reshape(-1), y.reshape(-1)), axis=1)


def _match(source: np.ndarray, target: np.ndarray, coords: np.ndarray, *, threshold: float = 0.35, radius: float = 1.5):
    s = _normalize(source.astype(np.float32))
    t = _normalize(target.astype(np.float32))
    cosine = s @ t.T
    spatial = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1) / 121.0
    distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    cost = 1.0 - cosine + 0.02 * spatial
    # PTG is a local transport hypothesis: a slot may move to its immediate
    # grid neighborhood, but cannot jump across the crop as in PCAR's global
    # pair-local reassignment.
    cost[distance > radius] = 1e3
    rows, cols = linear_sum_assignment(cost)
    accepted = cosine[rows, cols] >= threshold
    rows, cols = rows[accepted], cols[accepted]
    if not len(rows):
        return rows, cols, cosine, np.empty(0, np.float32)
    return rows, cols, cosine, cosine[rows, cols].astype(np.float32)


def _one_clip(tokens: np.ndarray) -> dict[str, float]:
    # The verified cache has eight monotonically sampled frames; index four is
    # the center.  Pairwise assignments are composed in the PTG implementation;
    # this smoke measures the stricter adjacent identity/cycle witness first.
    coords = _grid()
    center = 4
    survival = []
    cycle_errors = []
    forward_errors = []
    perm_penalties = []
    nontrivial = []
    for left in range(tokens.shape[0] - 1):
        right = left + 1
        a, b, cosine, accepted_cos = _match(tokens[left], tokens[right], coords)
        if len(a):
            survival.append(len(a) / 144.0)
            forward_errors.append(float(np.mean(1.0 - accepted_cos)))
            nontrivial.append(float(np.mean(np.linalg.norm(tokens[left, a] - tokens[right, b], axis=1) > 0.05)))
            # Reverse matching tests whether a physical slot can close locally.
            rb, ra, rev_cos, _ = _match(tokens[right], tokens[left], coords)
            reverse = {int(r): int(c) for r, c in zip(rb, ra)}
            closed = []
            for r, c in zip(a, b):
                # Reverse rows index the target slot; compare its returned
                # source column to the original source row.
                if int(c) in reverse:
                    closed.append(float(np.linalg.norm(coords[reverse[int(c)]] - coords[r])))
            if closed:
                cycle_errors.extend(closed)
            # Identity permutation: shuffle the target token identities while
            # preserving its spatial grid and measure the same assignment cost.
            perm = np.roll(tokens[right], 17, axis=0)
            # Preserve the learned assignment and permute only token identity;
            # re-solving would erase the very identity test being measured.
            perm_cosine = _normalize(tokens[left].astype(np.float32)) @ _normalize(perm.astype(np.float32)).T
            if len(accepted_cos):
                base = max(float(np.mean(1.0 - accepted_cos)), 1e-6)
                perm_error = float(np.mean(1.0 - perm_cosine[a, b]))
                perm_penalties.append(100.0 * (perm_error - base) / base)
    if not survival:
        return {"survival": 0.0, "cycle_median": float("inf"), "cycle_p95": float("inf"), "forward_error": float("inf"), "identity_penalty": -100.0, "nontrivial": 0.0}
    return {
        "survival": float(np.mean(survival)),
        "cycle_median": float(np.median(cycle_errors)) if cycle_errors else float("inf"),
        "cycle_p95": float(np.quantile(cycle_errors, 0.95)) if cycle_errors else float("inf"),
        "forward_error": float(np.mean(forward_errors)),
        "identity_penalty": float(np.median(perm_penalties)) if perm_penalties else -100.0,
        "nontrivial": float(np.mean(nontrivial)) if nontrivial else 0.0,
    }


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if len(selection) < 128:
        raise RuntimeError("The fixed witness must contain 128 centers")
    ids = [row["sample_id"] for row in selection[:16]]
    sample_ids = np.load(DENSE / "sample_ids.npy", allow_pickle=False)
    lookup = {str(value): i for i, value in enumerate(sample_ids.tolist())}
    if any(value not in lookup for value in ids):
        raise RuntimeError("Witness sample IDs are not present in the dense cache")
    indices = [lookup[value] for value in ids]
    tokens = np.load(DENSE / "tokens_grid12.npy", mmap_mode="r", allow_pickle=False)
    coarse = np.load(DENSE / "tokens_grid3.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != (4977, 8, 144, 768) or coarse.shape != (4977, 8, 9, 768):
        raise RuntimeError("Dense cache shape contract changed")
    measurements = [_one_clip(np.asarray(tokens[i], dtype=np.float32)) for i in indices]
    values = {key: np.asarray([row[key] for row in measurements], dtype=np.float64) for key in measurements[0]}
    # Source-paired statistics are label-blind.  Compare a compact motion norm
    # statistic, not task labels or ARFTR outputs; a missing paired cache is a
    # hard preflight failure rather than a silent substitution.
    paired_paths = {
        "exact_native_downsample720": PAIRED / "exact4k_downsample720_grid12.npy",
        "native4k": PAIRED / "native4k_grid12.npy",
    }
    source_stats = {}
    base_stat = float(np.mean(np.abs(np.asarray(tokens[indices], dtype=np.float32)[:, 1:] - np.asarray(tokens[indices], dtype=np.float32)[:, :-1])))
    for name, path in paired_paths.items():
        if not path.is_file():
            raise RuntimeError(f"Paired source cache missing: {path}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.shape != tokens.shape:
            raise RuntimeError(f"Paired source shape changed: {path}")
        stat = float(np.mean(np.abs(np.asarray(arr[indices], dtype=np.float32)[:, 1:] - np.asarray(arr[indices], dtype=np.float32)[:, :-1])))
        source_stats[name] = {"motion_abs_mean": stat, "relative_to_supplied": abs(stat - base_stat) / max(base_stat, 1e-8)}
    source_consistency = max(item["relative_to_supplied"] for item in source_stats.values()) <= 0.20
    gates = {
        "survival_smoke": bool(np.min(values["survival"]) >= 0.65 and np.median(values["survival"]) >= 0.80),
        "cycle_smoke": bool(np.median(values["cycle_median"]) <= 1.5 and np.quantile(values["cycle_p95"], 0.95) <= 4.0),
        "identity_smoke": bool(np.median(values["identity_penalty"]) >= 25.0),
        "nontrivial_smoke": bool(np.mean(values["nontrivial"]) >= 0.50),
        "source_consistency_smoke": bool(source_consistency),
    }
    receipt = {
        "status": "PTG_PHASE_A_LABEL_BLIND_SMOKE_COMPLETE",
        "run": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "selection": {"source": str(SELECTION.relative_to(ROOT)).replace("\\", "/"), "rows": 16, "first_16_only": True, "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "inputs": {
            "tokens_grid12": {"path": str((DENSE / "tokens_grid12.npy").relative_to(ROOT)).replace("\\", "/"), "shape": list(tokens.shape), "sha256": sha256(DENSE / "tokens_grid12.npy")},
            "tokens_grid3": {"path": str((DENSE / "tokens_grid3.npy").relative_to(ROOT)).replace("\\", "/"), "shape": list(coarse.shape), "sha256": sha256(DENSE / "tokens_grid3.npy")},
            "paired": {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)} for name, path in paired_paths.items()},
        },
        "labels_read": 0,
        "arftr_outputs_read": 0,
        "support_annotations_read": 0,
        "human_review_fields_read": 0,
        "classifier_fits": 0,
        "measurements": measurements,
        "aggregate": {key: {"median": float(np.median(value)), "min": float(np.min(value)), "max": float(np.max(value))} for key, value in values.items()},
        "source_stats": {"supplied720": {"motion_abs_mean": base_stat}, **source_stats},
        "gates": gates,
        "all_smoke_gates_pass": bool(all(gates.values())),
        "task_fit_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "preflight.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "all_smoke_gates_pass": receipt["all_smoke_gates_pass"], "gates": gates, "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
