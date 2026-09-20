"""Corrected PDI replay audit with explicit dtype-boundary semantics.

V1 correctly replayed neural outputs, but incorrectly required the raw float32
model action to be byte-identical to the source float64 anchor.  The deployed
retain action lives in the fixed verifier and copies the float64 anchor.  This
audit preserves both checks: raw action 0 must equal the documented float32
round trip, while every final retain output must equal the source anchor bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import audit_okutama_paired_detail_innovation as v1
from experiments import run_okutama_paired_detail_innovation as runner
from hac.actor_memory_base import file_sha256
from hac.paired_detail_innovation import PairedDetailInnovation


def replay_producer(data: dict, directory: Path, device: str) -> dict:
    receipt = runner.read_json(directory / "receipt.json")
    saved = runner._load_predictions(directory / "predictions.npz")
    if receipt["predictions_sha256"] != file_sha256(directory / "predictions.npz"):
        raise RuntimeError("PDI prediction hash changed")
    with np.load(directory / "normalization.npz", allow_pickle=False) as norm:
        stats = tuple(norm[name] for name in ("mean", "std", "qmean", "qstd", "class_weights"))
    checkpoint = torch.load(directory / "checkpoint.pt", map_location=device, weights_only=True)
    model = PairedDetailInnovation().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rows = saved["held_rows"].astype(np.int64)
    replay = {name: [] for name in ("probabilities", "actions", "correction", "scaffold_norms")}
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            subset = rows[start : start + 128]
            anchor = saved["anchor_probabilities"][start : start + len(subset)]
            batch = runner._batch(data, subset, stats, receipt["arm"], anchor, device)
            output = model(*batch, arm=receipt["arm"])
            replay["probabilities"].append(output["probabilities"].cpu().numpy())
            replay["actions"].append(output["actions"].cpu().numpy())
            replay["correction"].append(output["correction"].cpu().numpy())
            replay["scaffold_norms"].append(output["coarse_scaffold_norms"].cpu().numpy())
    maxima = {}
    for name in replay:
        value = np.concatenate(replay[name]).astype(np.float64)
        maxima[name] = float(np.max(np.abs(value - saved[name])))
        if maxima[name] > 1e-7:
            raise RuntimeError(f"PDI producer replay mismatch {directory}/{name}: {maxima[name]}")
    expected_raw_retain = saved["anchor_probabilities"].astype(np.float32).astype(np.float64)
    if not np.array_equal(saved["actions"][:, 0], expected_raw_retain):
        raise RuntimeError("raw retain action differs from the locked float32 model boundary")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"directory": str(directory), "rows": len(rows), "maximum_deltas": maxima,
            "raw_retain_semantics": "exact_float64_to_float32_to_float64_roundtrip"}


def audit(run: Path, device: str) -> dict:
    runner.validate_lock(run)
    data = runner.data()
    producer_results, verifier_results, stopped = [], [], []
    final_retain_rows = 0
    for arm in runner.ARMS:
        for outer in range(5):
            fold_dir = run / arm / f"fold-{outer}"
            receipt = runner.read_json(fold_dir / "receipt.json")
            for inner in range(5):
                producer_results.append(replay_producer(data, fold_dir / f"inner-{inner}", device))
            if receipt["status"] == "PDI_ARM_FOLD_STOPPED_INNER_CAPACITY_NO_GO":
                stopped.append({"arm": arm, "outer": outer})
                continue
            if receipt["status"] != "PDI_ARM_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED":
                raise RuntimeError("unexpected PDI arm/fold state")
            producer_results.append(replay_producer(data, fold_dir / "final", device))
            replay = v1.replay_verifier(fold_dir)
            final = runner._load_predictions(fold_dir / "predictions.npz")
            retain = final["choices"] == 0
            if not np.array_equal(final["routed_probabilities"][retain], final["anchor_probabilities"][retain]):
                raise RuntimeError("final verifier retention is not float64 bit-exact")
            final_retain_rows += int(retain.sum())
            verifier_results.append({"arm": arm, "outer": outer, **replay})
    failed_log = run / "completion.stderr.log"
    result = {
        "status": "PDI_INDEPENDENT_FULL_PRODUCER_AND_VERIFIER_REPLAY_PASS",
        "audit_version": 2,
        "v1_failure_preserved": str(failed_log.relative_to(runner.ROOT)).replace("\\", "/"),
        "v1_failure_sha256": file_sha256(failed_log),
        "v1_correction": "raw float32 action checked at its dtype boundary; deployed float64 retain checked bit-exact",
        "producer_replays": len(producer_results), "verifier_replays": len(verifier_results),
        "final_bit_exact_retain_rows": final_retain_rows,
        "stopped_arm_folds": stopped, "outer_labels_read_during_replay_fit": 0,
        "maximum_probability_delta": max((item["maximum_deltas"]["probabilities"] for item in producer_results), default=0.0),
        "details": producer_results, "verifiers": verifier_results,
    }
    runner.write_json(run / "independent_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("details", "verifiers")}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    audit(args.run.resolve(), args.device)


if __name__ == "__main__":
    main()
