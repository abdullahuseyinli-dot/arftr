"""Declare a P3 source lock or build its annotation-only two-second clip manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.okutama_long_video import build_long_manifest, create_materialization_lock


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-source-lock", type=Path)
    action.add_argument("--source-lock", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.write_source_lock is not None:
        if args.archive is None or args.output_dir is not None:
            parser.error("Lock declaration requires --archive and forbids --output-dir")
        result = create_materialization_lock(
            root, args.archive.resolve(), args.write_source_lock.resolve()
        )
    else:
        if args.output_dir is None or args.archive is not None:
            parser.error(
                "Manifest construction requires --output-dir; archive identity comes only from its lock"
            )
        result = build_long_manifest(
            source_lock_path=args.source_lock.resolve(),
            output_dir=args.output_dir.resolve(),
            root=root,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
