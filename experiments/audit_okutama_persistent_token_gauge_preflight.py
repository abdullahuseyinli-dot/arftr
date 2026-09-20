"""Independent receipt audit for the PTG label-blind smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260917/ptg_preflight_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    receipt = json.loads((RUN / "preflight.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for key in ("labels_read", "arftr_outputs_read", "support_annotations_read", "human_review_fields_read", "classifier_fits"):
        if receipt.get(key) != 0:
            failures.append(f"{key}={receipt.get(key)}")
    if receipt.get("status") != "PTG_PHASE_A_LABEL_BLIND_SMOKE_COMPLETE":
        failures.append("status")
    measurements = receipt.get("measurements", [])
    if len(measurements) != 16:
        failures.append("selection_count")
    for item in measurements:
        if not all(np.isfinite(float(item[k])) for k in ("survival", "cycle_median", "cycle_p95", "forward_error", "identity_penalty", "nontrivial")):
            failures.append("nonfinite_measurement")
    for name, record in receipt["inputs"]["paired"].items():
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            failures.append(f"hash:{name}")
    token_record = receipt["inputs"]["tokens_grid12"]
    token_path = ROOT / token_record["path"]
    if not token_path.is_file() or sha256(token_path) != token_record["sha256"]:
        failures.append("hash:tokens_grid12")
    # Recompute the gate booleans from the saved measurements, independently of
    # the producer's gate dictionary.
    values = {k: np.asarray([float(row[k]) for row in measurements]) for k in ("survival", "cycle_median", "cycle_p95", "identity_penalty", "nontrivial")}
    recomputed = {
        "survival_smoke": bool(values["survival"].min() >= 0.65 and np.median(values["survival"]) >= 0.80),
        "cycle_smoke": bool(np.median(values["cycle_median"]) <= 1.5 and np.quantile(values["cycle_p95"], 0.95) <= 4.0),
        "identity_smoke": bool(np.median(values["identity_penalty"]) >= 25.0),
        "nontrivial_smoke": bool(values["nontrivial"].mean() >= 0.50),
        "source_consistency_smoke": bool(max(float(v["relative_to_supplied"]) for v in receipt["source_stats"].values() if "relative_to_supplied" in v) <= 0.20),
    }
    if recomputed != receipt.get("gates"):
        failures.append("gate_recompute")
    audit = {
        "status": "PTG_PHASE_A_INDEPENDENT_REPLAY_AUDIT_PASS" if not failures else "PTG_PHASE_A_INDEPENDENT_REPLAY_AUDIT_FAIL",
        "run": str(RUN.relative_to(ROOT)).replace("\\", "/"),
        "receipt_sha256": sha256(RUN / "preflight.json"),
        "checks": {"hashes": not any(item.startswith("hash:") for item in failures), "zero_forbidden_reads": not any("=" in item for item in failures), "gate_recompute": "gate_recompute" not in failures, "measurement_rows": len(measurements)},
        "failures": failures,
        "labels_read": 0,
        "arftr_outputs_read": 0,
        "support_annotations_read": 0,
        "human_review_fields_read": 0,
        "classifier_fits": 0,
    }
    (RUN / "independent_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
