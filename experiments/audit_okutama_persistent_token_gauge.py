"""Independent exact replay audit for the PTG discovery matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.persistent_token_gauge import PTGDiscoveryHead


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260917/ptg_task_probe_v1"
ARMS = ("G0_arftr_context", "G1_coarse_temporal_control", "G2_raw_persistent_paths", "G3_gauge_parity")
KEYS = {ARMS[0]: "g0", ARMS[1]: "g1", ARMS[2]: "g2", ARMS[3]: "g3"}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(RUN / "feature_cache.npz", allow_pickle=False) as z:
        features = {key: z[key] for key in z.files}
    checks = []
    failures = []
    for fold in range(5):
        for arm in ARMS:
            directory = RUN / f"fold-{fold}" / arm
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                held = saved["held_rows"]
                expected = saved["probabilities"]
            checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
            model = PTGDiscoveryHead(features[KEYS[arm]].shape[1]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            values = (features[KEYS[arm]][held] - checkpoint["mean"]) / checkpoint["scale"]
            with torch.no_grad():
                replay = torch.softmax(model(torch.from_numpy(values.astype(np.float32)).to(device)), dim=1).cpu().numpy()
            delta = float(np.max(np.abs(replay - expected)))
            receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
            checks.append({"fold": fold, "arm": arm, "rows": int(len(held)), "max_probability_delta": delta, "outer_held_labels_read": receipt.get("outer_held_labels_read"), "optimizer_updates": 0})
            if delta != 0.0 or receipt.get("outer_held_labels_read") != 0:
                failures.append({"fold": fold, "arm": arm, "delta": delta, "outer_labels": receipt.get("outer_held_labels_read")})
    audit = {
        "status": "PTG_DISCOVERY_INDEPENDENT_REPLAY_AUDIT_PASS" if not failures else "PTG_DISCOVERY_INDEPENDENT_REPLAY_AUDIT_FAIL",
        "run": str(RUN.relative_to(ROOT)).replace("\\", "/"),
        "device": device,
        "checks": checks,
        "max_probability_delta": max((item["max_probability_delta"] for item in checks), default=0.0),
        "fits": 0,
        "optimizer_updates": 0,
        "labels_read": 0,
        "outer_held_labels_read": 0,
        "human_review_fields_read": 0,
        "annotation_support_read": 0,
        "failures": failures,
    }
    (RUN / "independent_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
