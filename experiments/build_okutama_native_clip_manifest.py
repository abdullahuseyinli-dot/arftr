"""Materialize only the 4,977 permitted Okutama OOF native clip identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hac.okutama_native_video import build_native_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--eligible-index",
        type=Path,
        default=Path(".runs/research_20260907/cptr_replay_r0/eligible_index.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    # Lock validator is maintained independently of the manifest implementation.
    from tools.lock_okutama_video_protocol import validate_materialization_lock

    lock = validate_materialization_lock(root, args.source_lock.resolve())
    summary = build_native_manifest(
        eligible_index=args.eligible_index.resolve(),
        archive_path=args.archive.resolve(),
        output_dir=args.output_dir.resolve(),
        source_lock=lock,
        source_lock_path=args.source_lock.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
