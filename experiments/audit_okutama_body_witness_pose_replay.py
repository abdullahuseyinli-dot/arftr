"""Independent label-free replay audit of the frozen body-witness pose cache."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.extract_okutama_body_witness_pose import infer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
POSE_RUN = ROOT / ".runs/research_20260913/body_witness_pose_pilot_v1"
OUTPUT = (
    ROOT
    / ".runs/research_20260913/body_witness_review_geometry_v1/pose_replay_audit_v1.json"
)
SCHEMA4 = (
    ROOT
    / ".runs/research_20260913/body_witness_review_geometry_v1/"
    "okutama_body_witness_protocol_schema4.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((PILOT / "crop_manifest.json").read_text(encoding="utf-8"))
    replay, details = infer(manifest, batch_size=16)
    cache = POSE_RUN / "pose_measurements.npz"
    comparisons: dict[str, Any] = {}
    with np.load(cache, allow_pickle=False) as saved:
        for name, value in replay.items():
            reference = saved[name]
            exact = np.array_equal(value, reference)
            comparisons[name] = {
                "shape": list(value.shape),
                "byte_exact": exact,
                "max_abs": (
                    0.0
                    if exact or value.dtype == np.bool_
                    else float(np.max(np.abs(value.astype(np.float64) - reference)))
                ),
            }
    receipt = {
        "status": (
            "INDEPENDENT_POSE_REPLAY_BIT_EXACT"
            if all(value["byte_exact"] for value in comparisons.values())
            else "FAIL"
        ),
        "details": details,
        "arrays": comparisons,
        "pose_cache_sha256": sha256_file(cache),
        "schema4_protocol_sha256": sha256_file(SCHEMA4),
        "current_protocol_sha256": sha256_file(
            ROOT / "experiments/okutama_body_witness_protocol.json"
        ),
        "source_hashes": {
            path: sha256_file(ROOT / path)
            for path in (
                "experiments/extract_okutama_body_witness_pose.py",
                "experiments/pilot_okutama_body_witness.py",
                "src/hac/vitpose_measurement.py",
                "src/hac/vitpose_parity.py",
            )
        },
        "task_labels_read": 0,
        "arftr_outputs_read": 0,
        "task_optimizer_updates": 0,
    }
    if receipt["status"] != "INDEPENDENT_POSE_REPLAY_BIT_EXACT":
        raise RuntimeError("Pose replay differs from the frozen cache")
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
