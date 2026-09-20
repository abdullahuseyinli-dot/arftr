"""Copy public working-tree files for validation without local runs or datasets.

This is a local export, not a release archive or an independent backup. Untracked
source files are included because a Git archive of HEAD would omit recent research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {".git", ".runs", "__pycache__", ".pytest_cache", ".ruff_cache"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".mp4", ".avi", ".zip", ".whl"}


def export(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise ValueError("Validation copy already exists; never overwrite or delete it")
    files = sorted(
        set(
            subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
            )
            .decode("utf-8")
            .split("\0")
        )
        - {""}
    )
    records = []
    for name in files:
        path = Path(name)
        if any(part in FORBIDDEN_PARTS or part.startswith(".venv") for part in path.parts):
            raise ValueError(f"Private state is exposed to Git: {name}")
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            or path.name.startswith(".env")
            and name != ".env.example"
        ):
            raise ValueError(f"Forbidden public artifact: {name}")
        if path.parts[0] == "data" and name not in {"data/README.md", "data/manifest.csv"}:
            raise ValueError(f"Unexpected dataset artifact: {name}")
        source = (ROOT / path).resolve()
        if not source.is_relative_to(ROOT) or source.is_symlink() or not source.is_file():
            raise ValueError(f"Unsupported source entry: {name}")
        if source.stat().st_size > 50 * 1024 * 1024:
            raise ValueError(f"Large public artifact requires explicit review: {name}")
        records.append(
            {
                "path": name,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    output.mkdir(parents=True)
    for item in records:
        destination = output / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / item["path"], destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Export copy is not byte-identical")
    result = {
        "status": "PUBLIC_WORKING_TREE_COPY_CREATED",
        "files": len(records),
        "bytes": sum(r["bytes"] for r in records),
        "records": records,
        "scope": "No Git history, private runs, environments, datasets or checkpoint weights",
        "publication_performed": False,
    }
    manifest_path = output.parent / (output.name + "_inventory.json")
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return {k: v for k, v in result.items() if k != "records"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(export(parser.parse_args().output), indent=2))
