"""Replay the bounded-correction result with cross-device numerical checks.

The frozen v1 auditor required bit-identical CPU and CUDA forward passes.  This
wrapper preserves every identity/hash check while allowing only floating-point
arrays to differ within an explicit tolerance.  Downstream metrics and
statistics are still recomputed from the immutable published probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import audit_okutama_bounded_factor_correction as audit_v1
import numpy as np


def numerically_same(left: Any, right: Any, message: str) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
            right_array.dtype, np.floating
        ):
            audit_v1.require(
                np.allclose(left_array, right_array, rtol=1e-5, atol=1e-6, equal_nan=False),
                f"{message} (rtol=1e-5, atol=1e-6)",
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
        / ".runs/research_20260912/bounded_factor_correction_v1_audit_v2",
    )
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    run.relative_to(audit_v1.trial.ROOT.resolve())
    output.relative_to(audit_v1.trial.ROOT.resolve())
    audit_v1.same = numerically_same
    audit_v1.audit(run, output)


if __name__ == "__main__":
    main()
