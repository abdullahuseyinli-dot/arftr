"""Run the remaining fixed source/posture folds, replay, then summarize."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260913/source_posture_fixed_primary_v1"
RUNNER = ROOT / "experiments/run_okutama_source_posture.py"
AUDITOR = ROOT / "experiments/audit_okutama_source_posture.py"


def invoke(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print(json.dumps({"event": "source_posture_queue_command", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def fold_done(fold: int) -> bool:
    return (RUN / f"fold-{fold}/receipt.json").is_file()


def execute(*, dry_run: bool) -> dict:
    pending = [fold for fold in range(5) if not fold_done(fold)]
    result = {
        "status": "SOURCE_POSTURE_QUEUE_DRY_RUN" if dry_run else "SOURCE_POSTURE_QUEUE_START",
        "folds_complete_before_start": 5 - len(pending),
        "pending_folds": pending,
        "fail_fast": True,
        "metrics_embargoed_until_independent_audit": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    if dry_run:
        return result
    for fold in pending:
        print(json.dumps({"event": "source_posture_fold_start", "fold": fold}), flush=True)
        invoke([str(RUNNER), "--stage", "fit-fold", "--fold", str(fold)])
        if not fold_done(fold):
            raise RuntimeError(f"Fold receipt did not persist: {fold}")
    invoke([str(AUDITOR)])
    invoke([str(RUNNER), "--stage", "summarize"])
    final = {
        "status": "SOURCE_POSTURE_QUEUE_COMPLETE",
        "folds_complete": sum(fold_done(fold) for fold in range(5)),
        "independent_audit": str(RUN / "independent_audit.json"),
        "summary": str(RUN / "results/v0001/summary.json"),
    }
    print(json.dumps(final, indent=2), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    execute(dry_run=not args.execute)


if __name__ == "__main__":
    main()
