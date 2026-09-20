"""Reject new Ruff findings without rewriting hash-locked historical recipes."""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/quality/ruff_baseline.json"


def diagnostics(python: str) -> list[dict]:
    completed = subprocess.run(
        [python, "-m", "ruff", "check", ".", "--output-format", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr or completed.stdout)
    items = json.loads(completed.stdout)
    return [
        {
            "path": Path(i["filename"]).relative_to(ROOT).as_posix(),
            "code": i["code"],
            "line": i["location"]["row"],
            "column": i["location"]["column"],
            "message": i["message"],
        }
        for i in items
    ]


def signature(item: dict) -> str:
    return json.dumps(item, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Interpreter with Ruff installed")
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help="One-time audited migration only; refuses replacement",
    )
    args = parser.parse_args()
    version = subprocess.check_output([args.python, "-m", "ruff", "--version"], text=True).strip()
    if args.record_baseline:
        current = diagnostics(args.python)
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with BASELINE.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "ruff_version": version,
                    "scope": "Pre-existing research-source style diagnostics recorded during portfolio cleanup; no newly introduced findings allowed",
                    "diagnostics": current,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        print(json.dumps({"baseline_recorded": len(current), "ruff_version": version}))
        return
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if version != baseline["ruff_version"]:
        raise RuntimeError(
            "Ruff version differs from baseline; install the project's pinned dev extra"
        )
    current = diagnostics(args.python)
    allowed = collections.Counter(map(signature, baseline["diagnostics"]))
    seen = collections.Counter(map(signature, current))
    new = seen - allowed
    if new:
        for item, count in new.items():
            print(f"NEW ({count}): {item}")
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "PASS_NO_NEW_FINDINGS",
                "baseline_findings": sum(allowed.values()),
                "remaining_legacy_findings": sum(seen.values()),
                "resolved_findings": sum((allowed - seen).values()),
            }
        )
    )


if __name__ == "__main__":
    main()
