"""Complete the locked PDI experiment: full audit, then metric release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import audit_okutama_paired_detail_innovation_v2 as auditor
from experiments import run_okutama_paired_detail_innovation as runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(runner.ROOT.resolve())
    audit = auditor.audit(run, args.device)
    if audit["status"] != "PDI_INDEPENDENT_FULL_PRODUCER_AND_VERIFIER_REPLAY_PASS":
        raise RuntimeError("PDI audit did not pass; metrics remain embargoed")
    summary = runner.summarize(run)
    print(
        json.dumps(
            {
                "status": "PDI_ALL_REMAINING_PHASES_COMPLETE",
                "audit": audit["status"],
                "result": summary["status"],
                "promoted": summary["promoted"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
