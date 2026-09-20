"""Label-free check of reference tensor layout in the separate replay auditor."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import audit_okutama_motion_null_contrast as audit
from experiments import run_okutama_motion_null_contrast as trial
from hac.motion_null_contrast import MotionNullContrast


def diagnose():
    run = trial.RUN
    cohort = trial.arrays(run / "inputs/cohort.npz")
    cache = trial.load_native_motion_cache(trial.legacy.CACHE, cohort["sample_ids"])
    geometry = trial.motion_geometry(cache.geometry, cache.camera)
    inputs = trial.arrays(run / "inputs/fold-0.npz")
    rows = inputs["held_rows"]
    transform = trial.arrays(run / "transforms/outer-0_final.npz")
    saved = trial.arrays(run / "paired_null/fold-0/predictions.npz")
    model = MotionNullContrast().cuda().eval()
    model.load_state_dict(torch.load(run / "paired_null/fold-0/checkpoint.pt", weights_only=True)["state_dict"])
    actual, original_replay, layout_replay = [], [], []
    layouts = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            x = torch.from_numpy(audit.independent_inputs(cache.crops[batch])).cuda()
            g = torch.from_numpy(((geometry[batch] - transform["mean"]) / transform["scale"]).astype(np.float32)).cuda()
            old_reference = torch.cat([x[:, :, :3], torch.zeros_like(x[:, :, 3:])], dim=2)
            same_layout_reference = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
            same_layout_reference.zero_()
            same_layout_reference[:, :, :3].copy_(x[:, :, :3])
            if not torch.equal(old_reference, same_layout_reference):
                raise RuntimeError("Reference values differ; layout is not isolated")
            actual.append(model(x, g).cpu().numpy())
            original_replay.append((0.5 * torch.tanh(audit.raw(model, x, g) - audit.raw(model, old_reference, g))).cpu().numpy())
            layout_replay.append((0.5 * torch.tanh(audit.raw(model, x, g) - audit.raw(model, same_layout_reference, g))).cpu().numpy())
            layouts.append({"input_stride": list(x.stride()), "old_reference_stride": list(old_reference.stride()),
                            "corrected_reference_stride": list(same_layout_reference.stride()), "reference_values_identical": True})
    results = {}
    for name, values in (("actual_trained_forward", actual), ("original_auditor", original_replay), ("layout_preserving_auditor", layout_replay)):
        delta = np.concatenate(values).astype(np.float64)
        results[name] = {"matches_saved_bit_exactly": bool(np.array_equal(delta, saved["delta"])),
                         "max_abs_delta_difference": float(np.max(np.abs(delta - saved["delta"])))}
    result = {"status": "LABEL_FREE_REPLAY_LAYOUT_DIAGNOSIS", "arm": "paired_null", "fold": 0,
              "new_fits": 0, "new_held_scores_computed": False, "checks": results, "layouts": layouts,
              "scope": "Only reference memory layout varied; same checkpoint, values, forward formula and batches",
              "original_auditor_sha256": trial.file_sha256(Path(audit.__file__)),
              "diagnostic_source_sha256": trial.file_sha256(Path(__file__))}
    trial.write(run / "replay_layout_diagnosis.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2))
