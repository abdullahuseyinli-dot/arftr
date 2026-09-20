"""Materialize the no-fit fixed-recipe ARFTR ancestry manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.nested_arftr_plan import fixed_recipe_plan

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_source_posture_protocol.json"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/source_posture_preflight_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    authority = ROOT / protocol["design_authority"]["path"]
    metadata = ROOT / protocol["population"]["metadata_path"]
    anchor = ROOT / protocol["anchor"]["path"]
    for path, expected in (
        (authority, protocol["design_authority"]["sha256"]),
        (metadata, protocol["population"]["metadata_sha256"]),
        (anchor, protocol["anchor"]["sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"Pinned source-posture input changed: {path}")
    with np.load(metadata, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in ("labels", "scenarios", "folds", "sample_ids")}
    result = fixed_recipe_plan(**arrays, enforce_canonical_counts=True)
    result.update(
        protocol_sha256=sha256(PROTOCOL),
        design_authority_sha256=sha256(authority),
        metadata_sha256=sha256(metadata),
        retained_anchor_sha256=sha256(anchor),
        task_fits=0,
        model_probabilities_read=0,
    )
    write_new(args.output_dir.resolve() / "ancestry_manifest.json", result)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
