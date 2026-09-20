"""Build the post-fit evidence-utility forensic package and graph update."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE_GRAPH = ROOT / ".runs/research_20260913/invention_council_20260913_135728/knowledge_graph.json"
BASE_LEDGER = (
    ROOT
    / ".runs/research_20260913/invention_council_20260913_135728/lineage/normalized_experiment_ledger.json"
)
AUDIT = ROOT / ".runs/research_20260913/evidence_utility_screen_v1_audit"
CACHE = ROOT / ".runs/research_20260913/evidence_utility_cache_v2"
RUN = ROOT / ".runs/research_20260913/evidence_utility_screen_v1"
ARMS = (
    "B0_center_repeat",
    "B1_unaligned",
    "B2_phase_control",
    "B3_aligned_primary",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.bincount(3 * labels + predictions, minlength=9).reshape(3, 3)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(
        2 * confusion.diagonal(),
        denominator,
        out=np.zeros(3, dtype=float),
        where=denominator > 0,
    )
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(np.mean(labels == predictions)),
        "errors": int(np.sum(labels != predictions)),
        "per_class_f1": f1.tolist(),
        "confusion": confusion.tolist(),
    }


def pairwise_overlap(errors: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    result = []
    names = list(errors)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            intersection = int(np.sum(errors[left] & errors[right]))
            union = int(np.sum(errors[left] | errors[right]))
            result.append(
                {
                    "left": left,
                    "right": right,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union,
                    "left_conditional_on_right": intersection / int(errors[right].sum()),
                    "right_conditional_on_left": intersection / int(errors[left].sum()),
                }
            )
    return result


def load_quality() -> tuple[dict[str, np.ndarray], np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "coverage",
            "transform_valid",
            "match_count",
            "weighted_residual",
            "contrasts",
        )
    }
    ids = []
    for directory in sorted(CACHE.glob("shard-*")):
        with np.load(directory / "features.npz", allow_pickle=False) as saved:
            ids.append(saved["sample_ids"].astype(str))
            for name in parts:
                parts[name].append(saved[name])
    values = {name: np.concatenate(value) for name, value in parts.items()}
    residual = values["weighted_residual"]
    residual_finite = np.isfinite(residual)
    residual_safe = np.where(residual_finite, residual, np.nan)
    with np.errstate(invalid="ignore"):
        residual_mean = np.nanmean(residual_safe, axis=1)
    residual_mean = np.nan_to_num(residual_mean, nan=10.0, posinf=10.0, neginf=10.0)
    contrasts = values["contrasts"].astype(np.float64)
    result = {
        "coverage_mean": values["coverage"].mean(1),
        "coverage_min": values["coverage"].min(1),
        "valid_transform_count": values["transform_valid"].sum(1).astype(float),
        "match_count_mean": values["match_count"].mean(1),
        "residual_mean_finite": residual_mean,
        "B1_contrast_rms": np.sqrt(np.mean(np.square(contrasts[:, 1]), axis=(1, 2))),
        "B2_contrast_rms": np.sqrt(np.mean(np.square(contrasts[:, 2]), axis=(1, 2))),
        "B3_contrast_rms": np.sqrt(np.mean(np.square(contrasts[:, 3]), axis=(1, 2))),
    }
    return result, np.concatenate(ids)


def associations(
    features: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    result = []
    for target_name, target in targets.items():
        for feature_name, feature in features.items():
            selected = np.isfinite(feature)
            y, x = target[selected].astype(int), feature[selected].astype(float)
            if len(np.unique(y)) != 2 or np.ptp(x) == 0:
                continue
            correlation, p_value = pointbiserialr(y, x)
            result.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "rows": int(len(y)),
                    "positives": int(y.sum()),
                    "roc_auc_raw_direction": float(roc_auc_score(y, x)),
                    "point_biserial_r": float(correlation),
                    "point_biserial_p_unadjusted": float(p_value),
                    "status": "post_hoc_association_not_causal_or_router_performance",
                    "feature_available_at_inference": True,
                }
            )
    return result


def graph_update(base: dict[str, Any], forensic: dict[str, Any]) -> dict[str, Any]:
    graph = json.loads(json.dumps(base))
    graph["graph"]["title"] = "HAC council graph after aligned-evidence action falsification"
    graph["graph"]["ancestor"] = str(BASE_GRAPH.relative_to(ROOT)).replace("\\", "/")
    artifact = str((AUDIT / "summary.json").relative_to(ROOT)).replace("\\", "/")
    new_nodes = [
        {
            "id": "experiment:evidence_utility_v1",
            "kind": "experiment",
            "label": "Natural-center evidence utility screen; 20 locked fits; independent replay",
            "artifact": artifact,
            "outcome": "failed preregistered discovery and opportunity gates",
        },
        {
            "id": "finding:alignment_action_disconnect",
            "kind": "failure_mechanism",
            "label": "Affine alignment improves masked-token reconstruction but not pooled action reading",
            "artifact": artifact,
            "confidence": "controlled task evidence on adaptive internal folds",
        },
        {
            "id": "finding:contrast_nuisance",
            "kind": "failure_mechanism",
            "label": "Every temporal contrast arm worsens NLL/calibration versus center/context B0",
            "artifact": artifact,
            "confidence": "observed; causal source among nuisance/reader/regularization unresolved",
        },
        {
            "id": "finding:no_unique_alignment_residual",
            "kind": "failure_mechanism",
            "label": "B3 has only 6 unique ARFTR rescues; 183 rescues versus 474 harms",
            "artifact": artifact,
            "confidence": "exact post-fit transition accounting",
        },
        {
            "id": "next:persistent_articulation_gauge",
            "kind": "proposed_invention",
            "label": "Gauge-invariant persistent articulation: background camera, actor root, local deformation",
            "artifact": "PROTOCOL.md",
            "status": "primary measurement-first proposal; no task fit authorized",
        },
        {
            "id": "fallback:source_fidelity_factor_swap",
            "kind": "proposed_invention",
            "label": "Regenerated-source factor replacement inside matched ARFTR ancestors",
            "artifact": "docs/HAC_SOURCE_VIEW_AUDIT_20260908.md",
            "status": "fallback; source effect exists but retained-system contribution unisolated",
        },
        {
            "id": "fallback:ambiguity_adjudication",
            "kind": "proposed_experiment",
            "label": "Track-context ambiguity adjudication on a representative label-blind sample",
            "artifact": "PROTOCOL.md",
            "status": "fallback diagnostic; never an inference routing feature",
        },
    ]
    for index, arm in enumerate(ARMS):
        arm_metric = forensic["metrics"][arm]
        new_nodes.append(
            {
                "id": f"experiment:evidence_utility_v1/{arm}",
                "kind": "experiment_arm",
                "label": f"{arm}: {100 * arm_metric['macro_f1']:.4f}% macro-F1",
                "artifact": artifact,
                "macro_f1": arm_metric["macro_f1"],
                "accuracy": arm_metric["accuracy"],
                "errors": int(
                    arm_metric["rows"]
                    - np.trace(np.asarray(arm_metric["confusion"], dtype=np.int64))
                ),
                "arm_index": index,
            }
        )
    existing = {node["id"] for node in graph["nodes"]}
    if any(node["id"] in existing for node in new_nodes):
        raise RuntimeError("Postmortem graph node already exists")
    graph["nodes"].extend(new_nodes)

    def edge(source: str, target: str, relation: str, **attributes: Any) -> dict[str, Any]:
        return {
            "source": source,
            "target": target,
            "key": 0,
            "relation": relation,
            "artifact": artifact,
            "confidence": "controlled_internal_or_explicit_hypothesis",
            "evidence_strength": "artifact_or_calculation",
            **attributes,
        }

    graph["edges"].extend(
        [
            edge("council:primary", "experiment:evidence_utility_v1", "suggests-next-test"),
            *[
                edge(
                    "experiment:evidence_utility_v1",
                    f"experiment:evidence_utility_v1/{arm}",
                    "has-arm",
                )
                for arm in ARMS
            ],
            edge(
                "experiment:evidence_utility_v1/B3_aligned_primary",
                "experiment:evidence_utility_v1/B2_phase_control",
                "regresses",
                effect_size_macro_f1_points=-0.23747401608636975,
            ),
            edge(
                "experiment:evidence_utility_v1/B3_aligned_primary",
                "experiment:evidence_utility_v1/B0_center_repeat",
                "regresses",
                effect_size_macro_f1_points=-0.4741927447938998,
            ),
            edge(
                "experiment:evidence_utility_v1/B3_aligned_primary",
                "council:unchanged_anchor",
                "rescues",
                effect_size_rows=183,
            ),
            edge(
                "experiment:evidence_utility_v1/B3_aligned_primary",
                "council:unchanged_anchor",
                "harms",
                effect_size_rows=474,
            ),
            edge(
                "experiment:evidence_utility_v1",
                "finding:alignment_action_disconnect",
                "supports-hypothesis",
            ),
            edge(
                "experiment:evidence_utility_v1",
                "finding:contrast_nuisance",
                "supports-hypothesis",
            ),
            edge(
                "experiment:evidence_utility_v1",
                "finding:no_unique_alignment_residual",
                "supports-hypothesis",
            ),
            edge(
                "finding:alignment_action_disconnect",
                "root_evidence_motion_reader",
                "falsifies-hypothesis",
                limitation="falsifies this source/decomposition/readout, not every registration method",
            ),
            edge(
                "finding:no_unique_alignment_residual",
                "next:persistent_articulation_gauge",
                "suggests-next-test",
            ),
            edge(
                "errors:motion",
                "next:persistent_articulation_gauge",
                "suggests-next-test",
                effect_size_error_rows=457,
            ),
            edge(
                "finding:alignment_action_disconnect",
                "fallback:source_fidelity_factor_swap",
                "suggests-next-test",
            ),
            edge(
                "labels:support",
                "fallback:ambiguity_adjudication",
                "requires-unavailable-signal",
                limitation="diagnostic only; prohibited at inference",
            ),
        ]
    )
    return graph


def write_graphml(graph: dict[str, Any], path: Path) -> None:
    converted = nx.MultiDiGraph()
    converted.graph.update(
        {
            key: value
            for key, value in graph["graph"].items()
            if isinstance(value, (str, int, float, bool))
        }
    )
    for record in graph["nodes"]:
        node = record["id"]
        converted.add_node(
            node,
            **{
                key: value
                for key, value in record.items()
                if key != "id" and isinstance(value, (str, int, float, bool))
            },
        )
    for record in graph["edges"]:
        converted.add_edge(
            record["source"],
            record["target"],
            **{
                key: value
                for key, value in record.items()
                if key not in {"source", "target", "key"}
                and isinstance(value, (str, int, float, bool))
            },
        )
    nx.write_graphml(converted, path)


def append_ledger(forensic: dict[str, Any]) -> list[dict[str, Any]]:
    ledger = json.loads(BASE_LEDGER.read_text(encoding="utf-8"))
    transitions = forensic["ARFTR_transitions"]
    for arm in ARMS:
        arm_metrics = forensic["metrics"][arm]
        ledger.append(
            {
                "experiment_id": f"evidence_utility_v1::{arm}",
                "family": "evidence_utility_v1",
                "arm": arm,
                "parent_baseline": "ARFTR retained; B0/B1/B2 matched task controls",
                "data": "Okutama 4977 centers; all rows; supplied boxes/tracks; no support filtering",
                "rows": 4977,
                "split": "five fixed scenario-grouped outer folds; 11 adaptive development scenarios",
                "features": "frozen 2304 context + six center DINO bins + arm contrast + common coverage",
                "architecture": "Linear(11526,64)-GELU-Linear(64,2) factor reader",
                "loss": "outer-train class-weighted stable factor NLL",
                "training_settings": "AdamW lr3e-4 wd1e-4 batch64 400 updates seed42; identical schedules",
                "seed_count": 1,
                "metric": "pooled OOF fixed-three-class macro-F1",
                "macro_f1": arm_metrics["macro_f1"],
                "accuracy": arm_metrics["accuracy"],
                "nll": arm_metrics["nll"],
                "brier": arm_metrics["brier"],
                "errors": int(
                    arm_metrics["rows"]
                    - np.trace(np.asarray(arm_metrics["confusion"], dtype=np.int64))
                ),
                "uncertainty": (
                    forensic["primary_B3_minus_B2"]
                    if arm == "B3_aligned_primary"
                    else "primary paired uncertainty applies only to preregistered B3-B2 contrast"
                ),
                "runtime_seconds": 59.38147600000957,
                "audit_status": "independent inventory/scaler/weight/checkpoint replay; exact decisions",
                "outcome": (
                    "primary failed discovery and opportunity gates; no confirmation"
                    if arm == "B3_aligned_primary"
                    else "matched diagnostic control; not retained"
                ),
                "result_class": "failed_preregistered_task_probe",
                "artifact": ".runs/research_20260913/evidence_utility_screen_v1_audit/summary.json",
                "protocol_artifact": "experiments/okutama_evidence_utility_protocol.json",
                "audit_artifact": ".runs/research_20260913/evidence_utility_screen_v1_audit/summary.json",
                "complete": True,
                "prediction_artifact": ".runs/research_20260913/evidence_utility_screen_v1_audit/oof_probabilities.npz",
                "rescue_vs_arftr": transitions[arm]["rescues"],
                "harm_vs_arftr": transitions[arm]["harms"],
                "net_corrections_vs_arftr": transitions[arm]["net"],
                "benchmark_comparability": "same 4977 labels/folds as ARFTR; separate single-seed fixed reader",
                "deployment_status": "inference-available inputs but inferior probe; never integrated",
                "checkpoint_verification_this_review": "20/20 CPU replays; maximum drift 8.34465e-7; decisions exact",
                "runtime_scope": "all 20 task fits",
                "council_verification": "prospective gates frozen before results; no ARFTR access before fit inventory",
                "metric_domain": "activity_classification",
            }
        )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT.resolve())
    output.mkdir(parents=True, exist_ok=False)

    audit_summary = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    with np.load(AUDIT / "oof_probabilities.npz", allow_pickle=False) as saved:
        sample_ids = saved["sample_ids"].astype(str)
        labels = saved["labels"].astype(np.int64)
        scenarios = saved["scenarios"].astype(str)
        folds = saved["folds"].astype(np.int64)
        probabilities = saved["probabilities"].astype(np.float64)
        arftr_probabilities = saved["arftr_probabilities"].astype(np.float64)
    predictions = {arm: probabilities[index].argmax(1) for index, arm in enumerate(ARMS)}
    predictions["ARFTR"] = arftr_probabilities.argmax(1)
    error_masks = {name: value != labels for name, value in predictions.items()}
    quality, quality_ids = load_quality()
    if not np.array_equal(quality_ids, sample_ids):
        raise RuntimeError("Cache and task prediction identities differ")
    for name, values in (
        ("ARFTR", arftr_probabilities),
        ("B3", probabilities[3]),
    ):
        ordered = np.sort(values, axis=1)
        quality[f"{name}_confidence"] = values.max(1)
        quality[f"{name}_margin"] = ordered[:, -1] - ordered[:, -2]
        quality[f"{name}_entropy"] = -np.sum(values * np.log(np.maximum(values, 1e-12)), axis=1)
    quality["B3_ARFTR_probability_L1"] = np.abs(probabilities[3] - arftr_probabilities).sum(1)
    quality["B3_ARFTR_decision_disagreement"] = (
        probabilities[3].argmax(1) != arftr_probabilities.argmax(1)
    ).astype(float)

    b3_correct = ~error_masks["B3_aligned_primary"]
    b2_correct = ~error_masks["B2_phase_control"]
    arftr_error = error_masks["ARFTR"]
    b3_rescue = arftr_error & b3_correct
    b3_harm = ~arftr_error & ~b3_correct
    targets = {
        "ARFTR_error": arftr_error,
        "B3_only_correct_vs_B2": b3_correct & ~b2_correct,
        "B2_only_correct_vs_B3": b2_correct & ~b3_correct,
        "B3_rescues_ARFTR_all_rows": b3_rescue,
        "B3_harms_ARFTR_all_rows": b3_harm,
    }
    feature_associations = associations(quality, targets)
    restricted_associations = []
    for name, mask, target in (
        ("B3_rescue_within_ARFTR_errors", arftr_error, b3_rescue),
        ("B3_harm_within_ARFTR_correct", ~arftr_error, b3_harm),
    ):
        rows = associations(
            {key: value[mask] for key, value in quality.items()}, {name: target[mask]}
        )
        restricted_associations.extend(rows)

    arftr_confidence = arftr_probabilities.max(1)
    sorted_confidence = np.argsort(-arftr_confidence)
    selective_risk = []
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = int(np.floor(len(labels) * coverage))
        kept = sorted_confidence[:count]
        selective_risk.append(
            {
                "coverage": coverage,
                "rows": count,
                "risk": float(error_masks["ARFTR"][kept].mean()),
                "minimum_retained_confidence": float(arftr_confidence[kept].min()),
            }
        )

    oracle_prediction = predictions["ARFTR"].copy()
    oracle_prediction[b3_rescue] = labels[b3_rescue]
    b3_b2_counts = {
        "both_correct": int(np.sum(b3_correct & b2_correct)),
        "B3_only_correct": int(np.sum(b3_correct & ~b2_correct)),
        "B2_only_correct": int(np.sum(~b3_correct & b2_correct)),
        "both_wrong": int(np.sum(~b3_correct & ~b2_correct)),
        "different_decisions": int(
            np.sum(predictions["B3_aligned_primary"] != predictions["B2_phase_control"])
        ),
    }
    scenario_delta = {}
    fold_delta = {}
    for _group_name, group_values, destination in (
        ("scenario", scenarios, scenario_delta),
        ("fold", folds, fold_delta),
    ):
        for group in np.unique(group_values):
            chosen = group_values == group
            left = metrics(labels[chosen], predictions["B3_aligned_primary"][chosen])
            right = metrics(labels[chosen], predictions["B2_phase_control"][chosen])
            destination[str(group)] = {
                "rows": int(chosen.sum()),
                "B3_macro_f1": left["macro_f1"],
                "B2_macro_f1": right["macro_f1"],
                "B3_minus_B2": left["macro_f1"] - right["macro_f1"],
            }

    forensic = {
        "status": "ALIGNED_EVIDENCE_UTILITY_POSTMORTEM_COMPLETE",
        "facts": {
            "retained_ARFTR_unchanged": True,
            "confirmation_authorized": False,
            "router_or_integration_authorized": False,
            "diagnostic_support_used_at_inference": False,
            "task_fits": 20,
            "fit_seconds": 59.38147600000957,
        },
        "metrics": audit_summary["metrics"],
        "ARFTR_metrics": audit_summary["ARFTR_metrics"],
        "primary_B3_minus_B2": audit_summary["primary_B3_minus_B2"],
        "exact_scenario_signs": audit_summary["exact_scenario_signs_descriptive"],
        "fold_B3_minus_B2": fold_delta,
        "scenario_B3_minus_B2": scenario_delta,
        "B3_vs_B2_correctness_partition": b3_b2_counts,
        "ARFTR_transitions": audit_summary["ARFTR_transitions"],
        "B3_unique_ARFTR_rescues_beyond_controls": audit_summary[
            "B3_unique_ARFTR_error_rescues_beyond_all_controls"
        ],
        "pairwise_error_overlap": pairwise_overlap(error_masks),
        "inference_quality_feature_associations": feature_associations + restricted_associations,
        "ARFTR_selective_risk": selective_risk,
        "diagnostic_oracles": {
            "ARFTR_retain_unless_B3_is_known_correct": {
                **metrics(labels, oracle_prediction),
                "rescues": int(b3_rescue.sum()),
                "harms": 0,
                "deployable": False,
                "reason": "uses held correctness to select interventions",
            }
        },
        "interpretation": {
            "fact": "B3 failed both prospective gates and is worse than every matched control.",
            "supported_inference": "The tested local aligned-DINO contrast is not unique action evidence and should not proceed to a correction router.",
            "rival_explanations_not_separated": [
                "aligned difference is nuisance for activity",
                "six-bin averaging removes the useful articulation structure",
                "single-frame DINO patches are semantically invariant to the needed motion",
                "the small reader cannot regularize the added 4608-dimensional contrast",
            ],
        },
        "inputs_sha256": {
            "audit_summary": sha256(AUDIT / "summary.json"),
            "audit_oof": sha256(AUDIT / "oof_probabilities.npz"),
            "fit_inventory": sha256(RUN / "fit_inventory.json"),
            "cache_audit": sha256(CACHE / "cache_audit_receipt.json"),
            "base_graph": sha256(BASE_GRAPH),
            "base_ledger": sha256(BASE_LEDGER),
        },
    }
    write_json(output / "forensics.json", forensic)

    with (output / "feature_associations.csv").open("x", encoding="utf-8", newline="") as stream:
        rows = forensic["inference_quality_feature_associations"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ledger = append_ledger(forensic)
    write_json(output / "normalized_experiment_ledger.json", ledger)
    fields = []
    for row in ledger:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output / "normalized_experiment_ledger.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ledger:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )

    graph = graph_update(json.loads(BASE_GRAPH.read_text(encoding="utf-8")), forensic)
    write_json(output / "knowledge_graph.json", graph)
    write_graphml(graph, output / "knowledge_graph.graphml")
    receipt = {
        "status": "POSTMORTEM_GRAPH_AND_LEDGER_COMPLETE",
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "ledger_rows": len(ledger),
        "graphml_nodes": nx.read_graphml(output / "knowledge_graph.graphml").number_of_nodes(),
        "graphml_edges": nx.read_graphml(output / "knowledge_graph.graphml").number_of_edges(),
        "outputs_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()},
    }
    write_json(output / "artifact_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
