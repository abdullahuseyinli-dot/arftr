"""Independent no-fit replay audit for the fine-density v2 probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import run_okutama_fine_detail_probe as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_fine_detail_probe_v2_protocol.json"
LOCK = ROOT / "experiments/okutama_fine_detail_execution_v2_lock.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run = args.run.resolve()
    base.PROTOCOL_PATH = PROTOCOL
    base.LOCK_PATH = LOCK
    protocol = read_json(PROTOCOL)
    data = base.validate_inputs(protocol)
    if not (run / "summary.json").is_file():
        raise RuntimeError("Probe summary is missing")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    max_delta = {"coarse_control": 0.0, "fine_both": 0.0}
    checks = []
    for fold in range(5):
        for arm_index, arm in enumerate(("coarse_control", "fine_both")):
            folder = run / f"fold-{fold}" / arm
            saved = np.load(folder / "predictions.npz", allow_pickle=False)
            rows = saved["held_rows"].astype(np.int64)
            expected_ids = data["sample_ids"][rows]
            if not np.array_equal(saved["sample_ids"].astype(str), expected_ids):
                raise RuntimeError(f"Saved identity mismatch: {arm}/fold-{fold}")
            normalization = np.load(folder / "normalization.npz", allow_pickle=False)
            checkpoint = torch.load(folder / "checkpoint.pt", map_location=args.device, weights_only=False)
            model = base.make_model(arm, args.device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            replay = []
            with torch.no_grad():
                for start in range(0, len(rows), int(protocol["training"]["batch_size"])):
                    subset = rows[start : start + int(protocol["training"]["batch_size"])]
                    fine, times, quality, coarse = base.batch_inputs(
                        data,
                        subset,
                        normalization["mean"],
                        normalization["std"],
                        normalization["qmean"],
                        normalization["qstd"],
                        arm,
                        args.device,
                    )
                    replay.append(model(fine, times, quality=quality, coarse_reference=coarse)["probabilities"].cpu().numpy())
            actual = np.concatenate(replay, axis=0)
            delta = float(np.max(np.abs(actual.astype(np.float64) - saved["probabilities"].astype(np.float64))))
            max_delta[arm] = max(max_delta[arm], delta)
            checks.append({"fold": fold, "arm": arm, "rows": int(len(rows)), "max_probability_delta": delta})
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    result = {
        "status": "FINE_DETAIL_V2_INDEPENDENT_REPLAY_AUDIT_PASS" if max(max_delta.values()) == 0.0 else "FINE_DETAIL_V2_INDEPENDENT_REPLAY_AUDIT_FAIL",
        "run": str(run.relative_to(ROOT)).replace("\\", "/"),
        "fits": 0,
        "optimizer_updates": 0,
        "labels_read": 0,
        "human_review_fields_read": 0,
        "max_probability_delta": max_delta,
        "checks": checks,
    }
    output = run / "independent_audit.json"
    if output.exists():
        if read_json(output) != result:
            raise RuntimeError("Existing independent audit differs")
    else:
        write_json(output, result)
    if result["status"].endswith("FAIL"):
        raise RuntimeError(json.dumps(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
