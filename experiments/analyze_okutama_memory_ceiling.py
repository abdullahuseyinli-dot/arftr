"""Label-oracle ceilings for a fixed +/-2s probability-mixture hypothesis class.

No fitted model or inference rule is produced. This diagnostic deliberately uses
the true center label and, for the restricted ceiling, true interval boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from hac.actor_memory_base import file_sha256, probability_metrics


def convex_margin(probabilities: np.ndarray, target: int):
    """Maximum margin of the true class over both alternatives in the convex hull."""
    alternatives = [index for index in range(3) if index != target]
    count = len(probabilities)
    inequalities = np.column_stack(
        (-(probabilities[:, target, None] - probabilities[:, alternatives]).T, np.ones(2))
    )
    result = linprog(
        np.r_[np.zeros(count), -1.0],
        A_ub=inequalities,
        b_ub=np.zeros(2),
        A_eq=np.r_[np.ones(count), 0.0][None],
        b_eq=np.ones(1),
        bounds=[(0, 1)] * count + [(-1, 1)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Convex diagnostic solver failed: {result.message}")
    return float(result.x[-1]), result.x[:-1] @ probabilities


def continuous_slots(data, row):
    observed = np.flatnonzero(data["valid"][row])
    center = int(np.flatnonzero(observed == 2)[0])
    allowed = np.zeros(5, bool)
    allowed[2] = True
    for position, slot in enumerate(observed):
        left, right = sorted((position, center))
        edges = observed[left + 1 : right + 1]
        if (
            data["boundary_valid"][row, edges].all()
            and not data["boundary_targets"][row, edges].any()
        ):
            allowed[slot] = True
    return allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(".runs/research_20260908/evidence_memory"))
    args = parser.parse_args()
    with np.load(args.run / "data/memory_data.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(args.run / "base_replay.npz", allow_pickle=False) as saved:
        p, components = saved["probabilities"], saved["components"]
        if not np.array_equal(saved["sample_ids"], data["sample_ids"]):
            raise RuntimeError("Diagnostic row identity mismatch")
    y = data["labels"]
    wrong = p.argmax(1) != y
    shared = (components.argmax(2) != y[:, None]).all(1)
    records = []
    oracle_probabilities = {name: p.copy() for name in ("unrestricted", "continuous_known_state")}
    repaired = {name: np.zeros(len(y), bool) for name in oracle_probabilities}
    for row in np.flatnonzero(wrong):
        result = {
            "sample_id": str(data["sample_ids"][row]),
            "label": int(y[row]),
            "shared": bool(shared[row]),
        }
        for name in oracle_probabilities:
            valid = data["valid"][row] if name == "unrestricted" else continuous_slots(data, row)
            values = p[data["neighbor_indices"][row, valid]]
            margin, oracle = convex_margin(values, int(y[row]))
            can_repair = margin > 1e-10
            repaired[name][row] = can_repair
            if can_repair:
                oracle_probabilities[name][row] = oracle
            result[name] = {
                "margin": margin,
                "can_repair": can_repair,
                "any_single_correct": bool((values.argmax(1) == y[row]).any()),
            }
        records.append(result)
    receipt = {
        "diagnostic_only_not_achieved_accuracy": True,
        "warning": "Uses true labels to choose mixtures and true interval boundaries for restricted availability. Not deployable, not a trained result, not external confirmation.",
        "radius_seconds": 2,
        "baseline_errors": int(wrong.sum()),
        "shared_errors": int(shared.sum()),
        "inputs": {
            name: file_sha256(args.run / name)
            for name in ("data/memory_data.npz", "base_replay.npz")
        },
        "ceilings": {
            name: {
                "repairable_errors": int(mask.sum()),
                "repairable_shared_errors": int((mask & shared).sum()),
                "oracle_metrics": probability_metrics(y, oracle_probabilities[name]),
                "extra_over_any_single_correct": sum(
                    row[name]["can_repair"] and not row[name]["any_single_correct"]
                    for row in records
                ),
            }
            for name, mask in repaired.items()
        },
        "error_rows": records,
    }
    directory = args.run / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "convex_memory_ceiling.json"
    if path.exists():
        raise RuntimeError("Do not overwrite an existing diagnostic")
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in receipt.items() if key != "error_rows"}, indent=2)
    )


if __name__ == "__main__":
    main()
