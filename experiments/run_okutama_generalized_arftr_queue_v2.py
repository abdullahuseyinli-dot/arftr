"""Fit and independently audit the remaining corrected F(S) populations.

The queue is single-GPU, smallest-population-first, resumable, and fail-fast.
It never launches a later population unless the preceding population's
independent replay passes.  When all 14 new populations pass, it emits the
aggregate independent audit required by the downstream posture task.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v2"
RUNNER = ROOT / "experiments/run_okutama_generalized_arftr_ancestors_v2.py"
AUDITOR = ROOT / "experiments/audit_okutama_generalized_arftr_ancestors_v2.py"
POPULATION_AUDIT_STATUS = "GENERALIZED_F_POPULATION_V2_INDEPENDENT_AUDIT_PASS"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def populations() -> list[tuple[str, int]]:
    lock = read_json(RUN / "execution_lock.json")
    result = [
        (population_id, specification["rows"])
        for population_id, specification in lock["F_populations"].items()
        if not specification["canonical_reuse"]
    ]
    if len(result) != 14:
        raise RuntimeError("Corrected generalized population inventory changed")
    return sorted(result, key=lambda item: (item[1], item[0]))


def audit_passed(population_id: str) -> bool:
    path = RUN / "populations" / population_id / "independent_audit.json"
    return path.is_file() and read_json(path).get("status") == POPULATION_AUDIT_STATUS


def invoke(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print(json.dumps({"event": "queue_command", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def execute(*, dry_run: bool) -> dict:
    inventory = populations()
    pending = [item for item in inventory if not audit_passed(item[0])]
    initial = {
        "status": "GENERALIZED_ARFTR_V2_QUEUE_DRY_RUN" if dry_run else "GENERALIZED_ARFTR_V2_QUEUE_START",
        "populations_total": len(inventory),
        "audited_before_start": len(inventory) - len(pending),
        "pending": [
            {"population_id": population_id, "training_rows": rows}
            for population_id, rows in pending
        ],
        "ordering": "training_rows_ascending_then_population_id",
        "fail_fast": True,
    }
    print(json.dumps(initial, indent=2), flush=True)
    if dry_run:
        return initial
    for population_id, rows in pending:
        print(
            json.dumps(
                {
                    "event": "population_start",
                    "population_id": population_id,
                    "training_rows": rows,
                }
            ),
            flush=True,
        )
        invoke(
            [
                str(RUNNER),
                "--stage",
                "fit-one",
                "--population-id",
                population_id,
            ]
        )
        invoke([str(AUDITOR), "--population-id", population_id])
        if not audit_passed(population_id):
            raise RuntimeError(f"Population audit did not persist PASS: {population_id}")
        print(
            json.dumps(
                {"event": "population_fit_and_audit_pass", "population_id": population_id}
            ),
            flush=True,
        )
    invoke([str(AUDITOR), "--all"])
    result = {
        "status": "GENERALIZED_ARFTR_V2_QUEUE_COMPLETE",
        "populations_total": len(inventory),
        "populations_independently_audited": sum(
            audit_passed(population_id) for population_id, _rows in inventory
        ),
        "aggregate_audit": str(RUN / "independent_audit.json"),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    execute(dry_run=not args.execute)


if __name__ == "__main__":
    main()
