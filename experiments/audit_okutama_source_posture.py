"""Replay every fixed source/posture outer prediction without held labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_okutama_source_posture as runner

from hac.actor_memory_base import file_sha256
from hac.nested_arftr_plan import fixed_recipe_plan
from hac.posture_witness import PostureHead, Projection, apply_posture_policy, apply_projection


def _projection(saved: dict[str, np.ndarray]) -> Projection | None:
    if saved["projection_components"].size == 0:
        return None
    return Projection(
        saved["projection_feature_mean"],
        saved["projection_components"],
        saved["projection_projected_mean"],
        saved["projection_projected_scale"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(runner.ROOT.resolve())
    lock = runner.validate_lock(run)
    data = runner._data()
    cache = runner._feature_cache(data)
    plan = fixed_recipe_plan(
        data["labels"],
        data["scenarios"],
        data["folds"],
        data["sample_ids"],
        enforce_canonical_counts=True,
    )
    fold_results = []
    for fold in range(5):
        train, held, _training_prior, _ancestry = runner._fold_priors(data, plan, fold)
        del train
        with np.load(runner.ARFTR / f"fold-{fold}/predictions.npz", allow_pickle=False) as saved:
            outer_prior = saved["seed_probabilities"][5].mean(0)
        reproduced_probabilities = []
        reproduced_interventions = []
        reproduced_deltas = []
        reproduced_available = []
        arm_results = []
        for arm, keys in runner.ARM_KEYS.items():
            directory = run / f"fold-{fold}" / arm
            receipt = runner._read_json(directory / "receipt.json")
            prediction_path = directory / "predictions.npz"
            checkpoint_path = directory / "checkpoint.npz"
            if (
                receipt.get("status")
                != "SOURCE_POSTURE_ARM_COMPLETE_OUTER_METRICS_EMBARGOED"
                or receipt.get("outer_held_labels_read") != 0
                or receipt.get("predictions_sha256") != file_sha256(prediction_path)
                or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
            ):
                raise RuntimeError("Source/posture arm receipt changed")
            if keys is None:
                probabilities = outer_prior.copy()
                intervention = np.zeros(len(held), dtype=bool)
                delta = np.zeros(len(held), dtype=np.float64)
                available = np.ones(len(held), dtype=bool)
            else:
                with np.load(checkpoint_path, allow_pickle=False) as saved:
                    checkpoint = {name: saved[name] for name in saved.files}
                if keys:
                    descriptor, all_available = cache.descriptor(keys)
                    held_features = descriptor[held]
                    available = all_available[held]
                    projection = _projection(checkpoint)
                    if projection is None:
                        raise RuntimeError("A pixel arm lacks its fitted projection")
                    held_features = apply_projection(held_features, projection)
                else:
                    held_features = np.empty((len(held), 0), dtype=np.float64)
                    available = np.ones(len(held), dtype=bool)
                head = PostureHead(
                    checkpoint["coefficients"],
                    float(checkpoint["intercept"][0]),
                    float(checkpoint["cap"][0]),
                    float(checkpoint["dead_zone"][0]),
                    float(checkpoint["l2_weight"][0]),
                    receipt["optimizer"],
                )
                probabilities, intervention, delta = apply_posture_policy(
                    outer_prior,
                    held_features,
                    head,
                    available=available,
                )
            with np.load(prediction_path, allow_pickle=False) as saved:
                exact = {
                    "sample_ids": bool(
                        np.array_equal(saved["sample_ids"], data["sample_ids"][held])
                    ),
                    "probabilities": bool(np.array_equal(saved["probabilities"], probabilities)),
                    "intervention": bool(np.array_equal(saved["intervention"], intervention)),
                    "continuous_delta": bool(
                        np.array_equal(saved["continuous_delta"], delta)
                    ),
                    "available": bool(np.array_equal(saved["available"], available)),
                }
                maximum = float(
                    np.max(np.abs(saved["probabilities"] - probabilities))
                )
            if not all(exact.values()):
                raise RuntimeError(f"Independent source/posture replay failed: fold{fold} {arm}")
            reproduced_probabilities.append(probabilities)
            reproduced_interventions.append(intervention)
            reproduced_deltas.append(delta)
            reproduced_available.append(available)
            arm_results.append(
                {"arm": arm, "exact": exact, "maximum_probability_difference": maximum}
            )
        fold_path = run / f"fold-{fold}/predictions.npz"
        fold_receipt = runner._read_json(run / f"fold-{fold}/receipt.json")
        with np.load(fold_path, allow_pickle=False) as saved:
            aggregate_exact = all(
                (
                    np.array_equal(saved["sample_ids"], data["sample_ids"][held]),
                    np.array_equal(saved["held_rows"], held),
                    np.array_equal(saved["arms"], np.asarray(tuple(runner.ARM_KEYS))),
                    np.array_equal(saved["probabilities"], np.stack(reproduced_probabilities)),
                    np.array_equal(saved["interventions"], np.stack(reproduced_interventions)),
                    np.array_equal(saved["continuous_deltas"], np.stack(reproduced_deltas)),
                    np.array_equal(saved["available"], np.stack(reproduced_available)),
                )
            )
        if (
            not aggregate_exact
            or fold_receipt.get("predictions_sha256") != file_sha256(fold_path)
            or fold_receipt.get("outer_held_labels_read") != 0
        ):
            raise RuntimeError("Independent fold aggregate replay failed")
        fold_results.append(
            {"fold": fold, "aggregate_exact": aggregate_exact, "arms": arm_results}
        )
    result = {
        "status": "SOURCE_POSTURE_INDEPENDENT_REPLAY_PASS",
        "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
        "folds": fold_results,
        "all_probabilities_exact": True,
        "maximum_probability_difference": 0.0,
        "outer_held_labels_read": 0,
        "auditor_sha256": file_sha256(Path(__file__)),
        "locked_auditor_sha256": next(
            record["sha256"]
            for record in lock["inputs"]
            if record["path"] == runner._relative(Path(__file__))
        ),
    }
    if result["auditor_sha256"] != result["locked_auditor_sha256"]:
        raise RuntimeError("Independent auditor changed after task-fit lock")
    path = run / "independent_audit.json"
    if path.exists():
        raise RuntimeError("Independent audit output already exists")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "folds": len(fold_results)}, indent=2))


if __name__ == "__main__":
    main()
