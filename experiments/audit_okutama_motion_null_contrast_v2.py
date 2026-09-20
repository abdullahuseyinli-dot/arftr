"""Separate-formula replay v2: preserve reference tensor strides, without changing any checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for search in (ROOT, ROOT / "src", ROOT / "experiments"):
    sys.path.insert(0, str(search))

from experiments import run_okutama_motion_null_contrast as trial
from hac.native_motion_data import load_native_motion_cache
from hac.native_motion_innovation import NativeMotionInnovation


def independent_inputs(crops):
    rgb = np.asarray(crops, dtype=np.float32) / 255
    means = (rgb[:, :-1] + rgb[:, 1:]) / 2
    differences = rgb[:, 1:] - rgb[:, :-1]
    return np.concatenate([means, differences], axis=-1).transpose(0, 1, 4, 2, 3).astype(np.float32)


def raw(model, x, geometry):
    encoded = model.encoder(x.reshape(-1, 6, 64, 64)).reshape(len(x), -1)
    return model.head(torch.cat([encoded, geometry], dim=1)).reshape(-1)


def replay(run, device):
    trial.validate(run)
    if trial.read(run / "queue_receipt.json")["head_fits"] != 10:
        raise RuntimeError("Exactly ten completed fits required")
    cohort = trial.arrays(run / "inputs" / "cohort.npz")
    cache = load_native_motion_cache(trial.legacy.CACHE, cohort["sample_ids"])
    geometry = np.concatenate([cache.geometry.astype(np.float32), cache.camera.astype(np.float32).reshape(len(cohort["sample_ids"]), -1)], axis=1)
    reports = []
    for fold in range(5):
        inputs = trial.arrays(run / "inputs" / f"fold-{fold}.npz")
        transform = trial.arrays(run / "transforms" / f"outer-{fold}_final.npz")
        schedules = []
        for arm in trial.ARMS:
            folder = run / arm / f"fold-{fold}"
            receipt = trial.read(folder / "receipt.json")
            for name in ("checkpoint", "predictions"):
                suffix = ".pt" if name == "checkpoint" else ".npz"
                if trial.file_sha256(folder / (name + suffix)) != receipt[name + "_sha256"]:
                    raise RuntimeError("Fit artifact hash changed")
            schedules.append(receipt["training_schedule_sha256"])
            saved = trial.arrays(folder / "predictions.npz")
            rows = inputs["held_rows"]
            if not np.array_equal(rows, saved["held_rows"]) or not np.array_equal(saved["sample_ids"], cohort["sample_ids"][rows]):
                raise RuntimeError("Prediction identity mismatch")
            checkpoint = torch.load(folder / "checkpoint.pt", map_location=device, weights_only=True)
            model = NativeMotionInnovation().to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            values = []
            null_values = []
            with torch.no_grad():
                for start in range(0, len(rows), 256):
                    batch = rows[start:start + 256]
                    x = torch.from_numpy(independent_inputs(cache.crops[batch])).to(device)
                    g = torch.from_numpy(((geometry[batch] - transform["mean"]) / transform["scale"]).astype(np.float32)).to(device)
                    reference = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
                    reference.zero_()
                    reference[:, :, :3].copy_(x[:, :, :3])
                    if arm == "paired_null":
                        delta = 0.5 * torch.tanh(raw(model, x, g) - raw(model, reference, g))
                        null_delta = 0.5 * torch.tanh(raw(model, reference, g) - raw(model, reference.clone(), g))
                        null_values.append(null_delta.cpu().numpy())
                    else:
                        delta = 0.5 * torch.tanh(raw(model, x, g))
                    values.append(delta.cpu().numpy())
            delta = np.concatenate(values).astype(np.float64)
            if not np.array_equal(delta, saved["delta"]):
                raise RuntimeError(f"Forward replay not bit-exact: {arm} fold {fold}")
            if arm == "paired_null" and (np.count_nonzero(np.concatenate(null_values)) or torch.count_nonzero(model.head[-1].bias).item()):
                raise RuntimeError("Trained paired nullness/bias invariant failed")
            p = cohort["anchor"][rows]
            eligible = cache.phase_available[rows].all(axis=1) & (p[:, 0] < np.minimum(p[:, 1], p[:, 2]))
            q = p[:, 1] + p[:, 2]
            log_odds = np.log(np.clip(p[:, 1], 1e-12, None) / np.clip(p[:, 2], 1e-12, None))
            ratio = 1 / (1 + np.exp(-(log_odds + delta)))
            candidate = np.stack([p[:, 0], q * ratio, q * (1 - ratio)], axis=1)
            retain = (~eligible) | (delta == 0)
            candidate[retain] = p[retain]
            if not np.array_equal(candidate, saved["candidate_probabilities"]) or not np.array_equal(eligible, saved["eligible"]):
                raise RuntimeError("Independent probability mapping disagrees")
            if not np.array_equal(candidate[retain], p[retain]) or not np.array_equal(candidate[:, 0], p[:, 0]):
                raise RuntimeError("Exact-retain or sitting-mass invariant failed")
            if not np.allclose(candidate.sum(1), p.sum(1), atol=1e-15, rtol=0) or np.max(np.abs(delta)) > 0.5:
                raise RuntimeError("Pair mass or bound failed")
            if arm == "paired_null":
                # Exercise the actual trained forward implementation as well.
                from hac.motion_null_contrast import MotionNullContrast
                actual = MotionNullContrast().to(device)
                actual.load_state_dict(checkpoint["state_dict"])
                actual.eval()
                with torch.no_grad():
                    if torch.count_nonzero(actual(reference, g)).item():
                        raise RuntimeError("Actual trained forward does not satisfy null invariant")
                del actual
            reports.append({"arm": arm, "fold": fold, "held_rows": len(rows), "delta_bit_exact": True,
                            "probabilities_bit_exact": True, "null_invariant": arm == "paired_null",
                            "retain_rows": int(retain.sum()), "active_parameters": receipt["active_parameters"]})
            del model
        if len(set(schedules)) != 1:
            raise RuntimeError("Paired arms used different minibatches")
    result = {"status": "MOTION_NULL_SEPARATE_FORMULA_REPLAY_PASS", "head_replays": len(reports),
              "trained_null_heads_verified": 5, "details": reports, "matched_schedules": True,
              "new_held_scores_computed": False, "independence_scope": "separate formula implementation; same author, cache and network modules",
              "preserved_files_verified": trial.check_preservation()}
    result.update({"auditor_revision": "v2_reference_memory_layout_only",
                   "auditor_source_sha256": trial.file_sha256(Path(__file__)),
                   "original_auditor_sha256": trial.file_sha256(ROOT / "experiments/audit_okutama_motion_null_contrast.py"),
                   "layout_diagnosis_sha256": trial.file_sha256(run / "replay_layout_diagnosis.json"),
                   "replay_tolerance_changed": False, "training_or_saved_predictions_changed": False})
    trial.write(run / "replay_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=trial.RUN)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(replay(args.run.resolve(), args.device), indent=2))
