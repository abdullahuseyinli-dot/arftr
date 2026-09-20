"""Materialize labels and physical identities for the frozen DINO frame caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.cached_frame_supervision import materialize_cached_frame_supervision


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / ".runs/research_20260912/cached_frame_supervision_v2/data",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(r"C:\Users\DELL\hac_external_data\OkutamaAction\TrainSetFrames.zip"),
    )
    args = parser.parse_args()
    receipt = materialize_cached_frame_supervision(root, args.output_dir, args.archive)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "rows",
                    "center_index",
                    "target_counts",
                    "weight_audit",
                    "artifact",
                    "access_accounting",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
