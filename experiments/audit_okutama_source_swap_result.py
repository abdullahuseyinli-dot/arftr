"""Independent saved-output identity/metric audit; zero training or new fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from hac.actor_memory_base import file_sha256
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260908/source_swap_v1"


def independent_metrics(labels, probabilities):
    prediction = probabilities.argmax(1)
    return {
        "rows": len(labels),
        "macro_f1": float(f1_score(labels, prediction, labels=[0, 1, 2], average="macro")),
        "accuracy": float(accuracy_score(labels, prediction)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "brier": float(np.square(probabilities - np.eye(3)[labels]).sum(1).mean()),
        "confusion": confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist(),
    }


def transitions(labels, old, new, mask):
    old_correct, new_correct = old.argmax(1) == labels, new.argmax(1) == labels
    return {
        "rows": int(mask.sum()),
        "rescues": int((mask & ~old_correct & new_correct).sum()),
        "harms": int((mask & old_correct & ~new_correct).sum()),
        "persistent_errors": int((mask & ~old_correct & ~new_correct).sum()),
        "class_prediction_changes": int((mask & (old.argmax(1) != new.argmax(1))).sum()),
        "probability_vector_changes": int((mask & np.any(old != new, axis=1)).sum()),
    }


def audit(run, result_dir):
    provenance = {}

    def checked(path, expected=None):
        actual = file_sha256(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Bound bytes changed: {path}")
        provenance[str(path)] = actual
        return path

    checked(Path(__file__))
    summary = read_json(checked(result_dir / "summary.json"))
    if not summary["complete"] or summary["rows"] != 4977 or summary["fresh_memory_fits"] != 75:
        raise RuntimeError("Audit requires the complete predeclared source matrix")
    for name, expected in summary["artifacts_sha256"].items():
        checked(result_dir / name, expected)
    lock_path = run / "memory_execution_lock.json"
    lock = read_json(checked(lock_path, summary["source_swap_memory_execution_sha256"]))
    for path, expected in lock["source_sha256"].items():
        checked(Path(path), expected)
    protocol = lock["unchanged_learning_protocol"]
    original_run = ROOT / ".runs/research_20260908/evidence_memory"
    with np.load(checked(run / "data/memory_data.npz"), allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files if key != "features"}
    with np.load(checked(original_run / "data/memory_data.npz"), allow_pickle=False) as saved:
        if any(not np.array_equal(saved[key], value) for key, value in data.items()):
            raise RuntimeError("Source intervention altered original nonfeature metadata")
    with np.load(result_dir / "oof_probabilities.npz", allow_pickle=False) as saved:
        final = {key: saved[key] for key in saved.files}
    with np.load(checked(run / "old_source_replay.npz"), allow_pickle=False) as saved:
        old = {key: saved[key] for key in saved.files}
    with np.load(checked(run / "new_source_p6_oof.npz"), allow_pickle=False) as saved:
        new = {key: saved[key] for key in saved.files}
    for archive in (old, new, final):
        for key in ("sample_ids", "labels", "scenarios", "folds"):
            if not np.array_equal(archive[key], data[key]):
                raise RuntimeError("Independent source-output identity mismatch")
    labels = data["labels"]
    seed_probabilities = np.full((3, 4977, 3), np.nan)
    fit_receipts = set()
    for path, expected in summary["completion_manifest_sha256"].items():
        manifest = read_json(checked(Path(path), expected))
        if manifest["fit_receipts"] != 15 or len(manifest["files_sha256"]) != 61:
            raise RuntimeError("Fold does not bind exactly15fits plus its selection")
        for name, expected_hash in manifest["files_sha256"].items():
            checked(Path(name), expected_hash)
            if Path(name).name == "receipt.json":
                fit_receipts.add(Path(name))
        fold = manifest["fold"]
        rows = np.flatnonzero(data["folds"] == fold)
        directory = Path(path).parent
        selection = read_json(directory / "selection.json")
        if selection["outer_held_used_for_selection"] or len(selection["candidates"]) != 4:
            raise RuntimeError("Source M4 selection contract changed")
        for seed_index, seed in enumerate(protocol["outer_seeds"]):
            fit_dir = directory / f"refit-seed-{seed}"
            receipt = read_json(fit_dir / "receipt.json")
            if (
                receipt["request"]["selection_labels_sha256"] is not None
                or receipt["selected_epoch"] != selection["refit_epochs"]
            ):
                raise RuntimeError("Outer refit accessed labels for checkpoint selection")
            with np.load(fit_dir / "predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["held_rows"], rows) or not np.array_equal(
                    saved["sample_ids"], data["sample_ids"][rows]
                ):
                    raise RuntimeError("Outer source fit identity changed")
                seed_probabilities[seed_index, rows] = saved["probabilities"]
    if len(fit_receipts) != 75 or fit_receipts != set((run / "models").rglob("receipt.json")):
        raise RuntimeError("Independent fit receipt census is not exactly75")
    if not np.array_equal(seed_probabilities, final["new_source_m4_seeds"]) or not np.array_equal(
        seed_probabilities.mean(0), final["new_source_m4"]
    ):
        raise RuntimeError("Final M4 output is not the exact retained3seed mean")
    for key in ("old_source_p6", "old_source_m4"):
        if not np.array_equal(old[key], final[key]):
            raise RuntimeError("Final source comparison changed an original reference")
    if not np.array_equal(new["probabilities"], final["new_source_p6"]):
        raise RuntimeError("Final source P6 differs from its fixed outer checkpoint")
    metrics = {name: independent_metrics(labels, final[name]) for name in summary["results"]}
    for name, values in metrics.items():
        for key, value in values.items():
            if key in ("rows", "confusion"):
                if summary["results"][name][key] != value:
                    raise RuntimeError("Independent metric/count check failed")
            elif abs(summary["results"][name][key] - value) > 1e-12:
                raise RuntimeError("Independent sklearn metric check failed")
    components = {}
    all_rows = np.ones(4977, bool)
    for index, name in enumerate(
        ("long_vjepa_mean", "dual_scale_vjepa_dino", "orthogonal_moments_factorized")
    ):
        before, after = old["old_source_components"][:, index], new["components"][:, index]
        components[name] = {
            "old": independent_metrics(labels, before),
            "new": independent_metrics(labels, after),
            **transitions(labels, before, after, all_rows),
        }
    with np.load(checked(run / "data/source_masks.npz"), allow_pickle=False) as saved:
        masks = {key: saved[key] for key in saved.files if key != "sample_ids"}
    masks["unchanged_input"] = ~masks["changed_input"]
    receipt = {
        "status": "INDEPENDENT_SAVED_SOURCE_MATRIX_AUDIT_PASSED",
        "model_fits": 0,
        "new_checkpoint_inference": 0,
        "new_fusion_candidates": 0,
        "rows": 4977,
        "verified_memory_receipts": 75,
        "new_source_seed_mean_bit_exact": True,
        "original_nonfeature_metadata_bit_exact": True,
        "independent_metrics": metrics,
        "P6_component_intervention": components,
        "fallback_and_changed_input_predictions": {
            candidate: {
                key: transitions(labels, final[reference], final[candidate], mask)
                for key, mask in masks.items()
            }
            for candidate, reference in (
                ("new_source_p6", "old_source_p6"),
                ("new_source_m4", "old_source_m4"),
            )
        },
        "interpretation": "Preserved fallback FEATURES do not imply preserved predictions after globally fitted weights change. Long-only source intervention retains467 old short-fallback features and is not identical to earlier probes that re-encoded them. Component scores are existing predictors, not new fusion candidates.",
        "input_sha256": provenance,
    }
    index = 1
    while (run / "diagnostics" / f"final_audit_v{index:04d}").exists():
        index += 1
    output = run / "diagnostics" / f"final_audit_v{index:04d}" / "summary.json"
    immutable_json(output, receipt)
    print(
        json.dumps(
            {"output": str(output), "sha256": file_sha256(output), "status": receipt["status"]},
            indent=2,
        )
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir or args.run / "results/v0001"
    audit(args.run, result_dir)


if __name__ == "__main__":
    main()
