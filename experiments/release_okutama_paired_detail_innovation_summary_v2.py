"""Release the PDI summary without changing any execution-locked producer code.

The original summary calculation completed but its JSON writer rejected a
NumPy boolean in the promotion gates and left ``summary.json`` truncated.
This release adapter recursively converts NumPy scalars to Python scalars and
redirects only that output to the versioned ``summary_v2.json`` artifact.  The
partial original is deliberately preserved as failure evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import run_okutama_paired_detail_innovation as runner
from hac.actor_memory_base import file_sha256


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(runner.ROOT.resolve())

    partial = run / "summary.json"
    audit = run / "independent_audit.json"
    if not partial.is_file():
        raise RuntimeError("expected preserved partial summary.json evidence")
    if not audit.is_file():
        raise RuntimeError("independent audit is missing")

    original_write_json = runner.write_json
    original_json_dumps = runner.json.dumps

    def versioned_write_json(path: Path, value: dict[str, Any]) -> None:
        target = path.with_name("summary_v2.json") if path.name == "summary.json" else path
        original_write_json(target, _native(value))

    runner.write_json = versioned_write_json
    runner.json.dumps = lambda value, *args, **kwargs: original_json_dumps(
        _native(value), *args, **kwargs
    )
    try:
        summary = runner.summarize(run)
    finally:
        runner.write_json = original_write_json
        runner.json.dumps = original_json_dumps

    summary_path = run / "summary_v2.json"
    receipt = {
        "status": "PDI_SUMMARY_V2_RELEASE_COMPLETE",
        "reason": "serialization-only repair; locked calculation and producer code unchanged",
        "partial_summary_preserved": {
            "path": str(partial.relative_to(runner.ROOT)).replace("\\", "/"),
            "bytes": partial.stat().st_size,
            "sha256": file_sha256(partial),
        },
        "independent_audit_sha256": file_sha256(audit),
        "locked_runner_sha256": file_sha256(Path(runner.__file__)),
        "summary_v2_sha256": file_sha256(summary_path),
        "result": summary["status"],
        "promoted": bool(summary["promoted"]),
    }
    original_write_json(run / "summary_v2_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
