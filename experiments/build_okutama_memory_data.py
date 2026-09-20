"""Build locked actor memory features, graph and dense annotation supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.actor_memory_data import materialize_memory_data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=root / ".runs/research_20260908/evidence_memory/data"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(r"C:\Users\DELL\hac_external_data\OkutamaAction\TrainSetFrames.zip"),
    )
    args = parser.parse_args()
    result = materialize_memory_data(root, args.output_dir, args.archive)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "rows",
                    "graph",
                    "supervision",
                    "diagnostics",
                    "support",
                    "artifacts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
