"""Replay bounded-correction checkpoints and quantify CPU/CUDA drift."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import audit_okutama_bounded_factor_correction as audit_v1
import numpy as np

DRIFT: dict[str, dict[str, float | int]] = {}


def numerically_same(left: Any, right: Any, message: str) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
            right_array.dtype, np.floating
        ):
            difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
            record = DRIFT.setdefault(message, {"comparisons": 0, "maximum_absolute": 0.0})
            record["comparisons"] = int(record["comparisons"]) + 1
            record["maximum_absolute"] = max(
                float(record["maximum_absolute"]), float(difference.max(initial=0.0))
            )
            if message == "Checkpoint probability replay changed":
                changed = int(np.sum(left_array.argmax(axis=1) != right_array.argmax(axis=1)))
                record["argmax_changes"] = int(record.get("argmax_changes", 0)) + changed
                audit_v1.require(changed == 0, f"{message}: class decisions changed")
                tolerance = 1e-5
            else:
                tolerance = 1e-4
            audit_v1.require(
                np.allclose(left_array, right_array, rtol=1e-4, atol=tolerance, equal_nan=False),
                f"{message} (rtol=1e-4, atol={tolerance})",
            )
        else:
            audit_v1.require(np.array_equal(left_array, right_array), message)
    else:
        audit_v1.require(left == right, message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=audit_v1.trial.DEFAULT_RUN)
    parser.add_argument(
        "--output",
        type=Path,
        default=audit_v1.trial.ROOT
        / ".runs/research_20260912/bounded_factor_correction_v1_audit_v3",
    )
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    run.relative_to(audit_v1.trial.ROOT.resolve())
    output.relative_to(audit_v1.trial.ROOT.resolve())
    audit_v1.same = numerically_same
    audit_v1.audit(run, output)
    audit_v1.immutable_json(
        output / "numerical_replay.json",
        {
            "status": "BOUNDED_FACTOR_CORRECTION_NUMERICAL_REPLAY_PASS",
            "cpu_cuda_comparison": DRIFT,
            "floating_rtol": 1e-4,
            "latent_absolute_tolerance": 1e-4,
            "probability_absolute_tolerance": 1e-5,
            "required_probability_argmax_changes": 0,
        },
    )


if __name__ == "__main__":
    main()
