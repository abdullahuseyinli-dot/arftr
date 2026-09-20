"""Evaluate one pre-hashed, post-matrix M2/M4 equal-weight consensus; never fit.

This follow-up was motivated by already observed complementary development
errors. Its uncertainty estimates do not provide independent confirmation.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import analyze_okutama_memory_results as audit
import numpy as np
import run_okutama_video_probe as statistics

from hac.actor_memory_base import file_sha256, probability_metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/okutama_fixed_memory_consensus_protocol.json"
PROTOCOL_SHA256 = "fcb643cbea1827b3529d5e339ccf6d6949b6e735bb9f4476267920eaef93960a"
RUN = ROOT / ".runs/research_20260908/evidence_memory"


def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def fixed_consensus(m2, m4):
    """No labels, fitted parameters, clipping, or renormalization enter this rule."""
    m2, m4 = np.asarray(m2, dtype=np.float64), np.asarray(m4, dtype=np.float64)
    if m2.ndim != 2 or m2.shape[1] != 3 or m2.shape != m4.shape:
        raise ValueError("Expected aligned three-class probability arrays")
    for value in (m2, m4):
        if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Invalid source probability")
        if not np.allclose(value.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError("Source probabilities do not sum to one")
    return np.float64(0.5) * m2 + np.float64(0.5) * m4


def masks_for(data, legacy):
    labels = data["labels"]
    height = data["quality"][:, 0].astype(float) * 720
    masks = {
        "all": np.ones(len(labels), bool),
        "long_feature_fallback": ~data["long_valid"],
        "long_feature_valid": data["long_valid"],
        "native_height_le32": height <= 32 + 1e-4,
        "native_height_32to64": (height > 32 + 1e-4) & (height <= 64 + 1e-4),
        "native_height_gt64": height > 64 + 1e-4,
    }
    masks.update({"legacy_" + name: mask for name, mask in legacy.items()})
    for index, name in enumerate(("sitting", "standing", "walking_running")):
        masks["true_class_" + name] = labels == index
    for prefix in ("node_support", "memory_support", "node_interval", "memory_interval"):
        complete, boundary = data[prefix + "_complete"], data[prefix + "_boundary"]
        masks[prefix + "_complete_boundary"] = complete & boundary
        masks[prefix + "_complete_stable"] = complete & ~boundary
        masks[prefix + "_unknown"] = ~complete
    return masks


def comparison(labels, candidate, reference, shared, mask):
    if not mask.any():
        return {"rows": 0}
    correct, base_correct = candidate.argmax(1) == labels, reference.argmax(1) == labels
    return {
        "rows": int(mask.sum()),
        "candidate_metrics": probability_metrics(labels[mask], candidate[mask]),
        "reference_metrics": probability_metrics(labels[mask], reference[mask]),
        "rescues": int((correct & ~base_correct & mask).sum()),
        "harms": int((~correct & base_correct & mask).sum()),
        "net_corrections": int((correct & mask).sum() - (base_correct & mask).sum()),
        "shared_p6_component_failures": int((shared & mask).sum()),
        "shared_failure_repairs": int((correct & shared & mask).sum()),
    }


def accuracy_bootstrap(labels, candidate, reference, scenarios, config):
    """Use the same paired scenario draws as the existing F1/loss bootstrap."""
    groups = np.unique(scenarios)
    difference = (candidate.argmax(1) == labels).astype(float) - (reference.argmax(1) == labels)
    counts = np.asarray([(scenarios == group).sum() for group in groups])
    sums = np.asarray([difference[scenarios == group].sum() for group in groups])
    draws = np.random.default_rng(config["seed"]).integers(
        0, len(groups), size=(config["bootstrap_resamples"], len(groups))
    )
    sampled = sums[draws].sum(1) / counts[draws].sum(1)
    return {
        "accuracy_delta": float(difference.mean()),
        "accuracy_delta_two_sided_95pct": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "accuracy_delta_one_sided_95pct_lower": float(np.quantile(sampled, 0.05)),
    }


def main():
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("The pre-score fixed-consensus protocol changed")
    protocol = audit.read_json(PROTOCOL_PATH)
    output = ROOT / protocol["output_directory"]
    if output.exists():
        raise FileExistsError(
            "Retain this one follow-up result; never overwrite or select another weight"
        )
    if file_sha256(RUN / "execution_lock.json") != protocol["source_lock_sha256"]:
        raise RuntimeError("Original memory execution lock changed")
    if protocol["sources"] != ["temporal_conv", "survival_memory"] or protocol["weights"] != [
        0.5,
        0.5,
    ]:
        raise RuntimeError("Only the declared exact equal-weight consensus is implemented")
    provenance = {
        str(PROTOCOL_PATH): PROTOCOL_SHA256,
        str(Path(__file__)): file_sha256(Path(__file__)),
        str(Path(audit.__file__)): file_sha256(Path(audit.__file__)),
        str(Path(statistics.__file__)): file_sha256(Path(statistics.__file__)),
    }
    for name in ("v0005/summary.json", "v0005/rows.csv", "access_accounting_clarification_v1.json"):
        path = RUN / "diagnostics" / name
        provenance[str(path)] = file_sha256(path)
    if (
        provenance[str(RUN / "diagnostics/v0005/summary.json")]
        != protocol["motivation_observed_before_protocol"]["summary_sha256"]
    ):
        raise RuntimeError("The diagnostic motivating the follow-up changed")
    data, baseline, legacy, original_lock = audit.load_inputs(RUN, provenance)
    originals, coverage = {}, {}
    for arm in protocol["sources"]:
        values, status = audit.load_arm(RUN, data, original_lock, arm, provenance)
        if values is None or not status["complete_primary_population"]:
            raise RuntimeError("Cannot evaluate a partial consensus")
        if not np.array_equal(values["rows"], np.arange(protocol["primary_rows"])):
            raise RuntimeError("Consensus source row ordering mismatch")
        originals[arm] = values["probabilities"].mean(0)
        coverage[arm] = status
    if (
        len(data["labels"]) != protocol["primary_rows"]
        or len(np.unique(data["scenarios"])) != protocol["scenarios"]
    ):
        raise RuntimeError("Consensus primary population changed")

    # This receipt is written and hashed BEFORE constructing or scoring the
    # consensus. Loading verified sources above performs no new model fit.
    output.mkdir(parents=True, exist_ok=False)
    pre_evaluation = {
        "status": "LOCKED BEFORE FIRST CONSENSUS COMPUTATION",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": provenance,
        "source_coverage": coverage,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn")
            },
        },
    }
    lock_path = output / "pre_evaluation_lock.json"
    write_json_new(lock_path, pre_evaluation)
    lock_hash = file_sha256(lock_path)
    print(
        json.dumps({"event": "fixed_consensus_locked_before_score", "sha256": lock_hash}),
        flush=True,
    )

    candidate = fixed_consensus(originals["temporal_conv"], originals["survival_memory"])
    labels, scenarios = data["labels"], data["scenarios"]
    reference = baseline["probabilities"]
    shared = (baseline["components"].argmax(2) != labels[:, None]).all(1)
    strata = masks_for(data, legacy)
    references = {"survival_memory": originals["survival_memory"], "p6": reference}
    comparisons, groups = {}, {}
    for name, probabilities in references.items():
        comparisons[name] = {
            **comparison(labels, candidate, probabilities, shared, strata["all"]),
            "strata": {
                key: comparison(labels, candidate, probabilities, shared, mask)
                for key, mask in strata.items()
            },
            "paired_statistics": {
                **statistics.paired_statistics(
                    labels,
                    candidate,
                    probabilities,
                    scenarios,
                    bootstrap_resamples=protocol["statistics"]["bootstrap_resamples"],
                    bootstrap_seed=protocol["statistics"]["seed"],
                ),
                **accuracy_bootstrap(
                    labels, candidate, probabilities, scenarios, protocol["statistics"]
                ),
                "interpretation": "Post-matrix exploratory comparison; intervals and p-values do not adjust for all preceding research decisions",
            },
        }
    adjusted = statistics.holm_adjust(
        {
            name: value["paired_statistics"]["one_sided_exact_swap_pvalue"]
            for name, value in comparisons.items()
        }
    )
    for name, pvalue in adjusted.items():
        comparisons[name]["paired_statistics"]["holm_adjusted_pvalue_two_followup_comparisons"] = (
            pvalue
        )
    for group_name, identities in {
        "scenario": scenarios,
        "recording": data["recordings"],
        "track": np.char.add(np.char.add(data["recordings"], "::"), data["tracks"]),
    }.items():
        groups[group_name] = []
        for value in np.unique(identities):
            mask = identities == value
            row = {group_name: str(value), "rows": int(mask.sum())}
            for name, probabilities in references.items():
                result = comparison(labels, candidate, probabilities, shared, mask)
                row.update(
                    {
                        f"vs_{name}_{key}": result[key]
                        for key in ("rescues", "harms", "net_corrections", "shared_failure_repairs")
                    }
                )
                row[f"vs_{name}_accuracy_delta"] = (
                    result["candidate_metrics"]["accuracy"]
                    - result["reference_metrics"]["accuracy"]
                )
            row["macro_f1"] = probability_metrics(labels[mask], candidate[mask])["macro_f1"]
            groups[group_name].append(row)
    p6_comparison = comparisons["p6"]
    limits = protocol["engineering_checks"]
    deltas = [row["vs_p6_accuracy_delta"] for row in groups["scenario"]]
    metrics = probability_metrics(labels, candidate)
    checks = {
        "target_84_macro_f1": metrics["macro_f1"] >= limits["target_macro_f1"],
        "stretch_85_macro_f1": metrics["macro_f1"] >= limits["stretch_macro_f1"],
        "positive_net_corrections_vs_p6": p6_comparison["net_corrections"] > 0,
        "shared_repairs": p6_comparison["shared_failure_repairs"]
        >= limits["shared_failure_repairs_minimum"],
        "scenario_breadth": sum(delta > 0 for delta in deltas)
        >= limits["improved_scenarios_minimum"],
        "worst_scenario": min(deltas) >= limits["worst_scenario_accuracy_delta_minimum"],
        "boundary_harm_reduced": p6_comparison["strata"]["legacy_complete_target_boundary"]["harms"]
        < limits["legacy_boundary_harms_less_than"],
        "stable_net_positive": p6_comparison["strata"]["legacy_complete_stable_target_window"][
            "net_corrections"
        ]
        > 0,
        "nll_no_worse": metrics["nll"] <= p6_comparison["reference_metrics"]["nll"],
        "brier_no_worse": metrics["brier"] <= p6_comparison["reference_metrics"]["brier"],
    }
    row_table = []
    for index in range(len(labels)):
        row_table.append(
            {
                "sample_id": str(data["sample_ids"][index]),
                "scenario": str(scenarios[index]),
                "fold": int(data["folds"][index]),
                "recording": str(data["recordings"][index]),
                "track": str(data["tracks"][index]),
                "center_frame": int(data["frames"][index]),
                "label": int(labels[index]),
                "consensus_prediction": int(candidate[index].argmax()),
                "m2_prediction": int(originals["temporal_conv"][index].argmax()),
                "m4_prediction": int(originals["survival_memory"][index].argmax()),
                "p6_prediction": int(reference[index].argmax()),
                "shared_failure": bool(shared[index]),
                "legacy_boundary": bool(legacy["complete_target_boundary"][index]),
                "legacy_stable": bool(legacy["complete_stable_target_window"][index]),
            }
        )
    audit.write_csv(output / "rows.csv", row_table)
    for name, rows in groups.items():
        audit.write_csv(output / (name + ".csv"), rows)
    np.savez_compressed(
        output / "probabilities.npz",
        sample_ids=data["sample_ids"],
        labels=labels,
        scenarios=scenarios,
        folds=data["folds"],
        consensus_fixed_half=candidate,
        m2=originals["temporal_conv"],
        m4=originals["survival_memory"],
        p6=reference,
    )
    summary = {
        "status": "ONE FIXED POST-MATRIX CONSENSUS EVALUATED AND RETAINED",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": protocol["scope"],
        "pre_evaluation_lock_sha256": lock_hash,
        "protocol_sha256": PROTOCOL_SHA256,
        "new_training_fits": 0,
        "new_model_inferences": 0,
        "weights_evaluated": [[0.5, 0.5]],
        "retained_even_if_checks_fail": True,
        "metrics": metrics,
        "comparisons": comparisons,
        "engineering_checks": checks,
        "all_nonstretch_engineering_checks_passed": all(
            value for name, value in checks.items() if name != "stretch_85_macro_f1"
        ),
        "improved_scenarios_vs_p6": sum(delta > 0 for delta in deltas),
        "worst_scenario_accuracy_delta_vs_p6": min(deltas),
        "group_tables": {name: name + ".csv" for name in groups},
        "source_sha256": provenance,
        "artifacts_sha256": {
            path.name: file_sha256(path) for path in sorted(output.iterdir()) if path.is_file()
        },
        "interpretation": "A post-matrix equal-weight combination chosen after observing complementary development errors, not a new trained model, new architecture, independently confirmed result, or authorized weight search",
    }
    write_json_new(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "summary_sha256": file_sha256(output / "summary.json"),
                "metrics": metrics,
                "engineering_checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
