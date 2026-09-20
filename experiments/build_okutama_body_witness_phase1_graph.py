"""Extend the verified HAC graph with prospective Body Witness phase-1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260912/invention_synthesis_20260912_234907/knowledge_graph.graphml"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/body_witness_phase1_20260913"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
PROTOCOL = ROOT / "experiments/okutama_body_witness_protocol.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_node(graph: nx.MultiDiGraph, node: str, kind: str, label: str, artifact: str) -> None:
    graph.add_node(node, kind=kind, label=label, artifact=artifact)


def add_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    relation: str,
    artifact: str,
    *,
    effect_size: str = "not_applicable",
    confidence: str = "prospective_or_structural",
    evidence_strength: str = "artifact_or_calculation",
) -> None:
    graph.add_edge(
        source,
        target,
        relation=relation,
        artifact=artifact,
        effect_size=effect_size,
        confidence=confidence,
        evidence_strength=evidence_strength,
    )


def build(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    graph = nx.read_graphml(SOURCE, force_multigraph=True)
    before = (graph.number_of_nodes(), graph.number_of_edges())
    nodes = [
        ("experiment:body_witness_native_pilot", "experiment", "128-center exact-native body observation pilot", ".runs/research_20260913/body_witness_pilot_v1/extraction_receipt.json"),
        ("data:body_witness_pilot_128", "dataset", "Deterministic label-blind 128-center pilot", ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"),
        ("features:six_slot_body_bank", "feature", "Two unmasked plus four masked 768D body descriptors", "experiments/okutama_body_witness_protocol.json"),
        ("features:body_quality_36", "feature", "Shared crop geometry, transforms and validity", "src/hac/body_witness_data.py"),
        ("measurement:pose_hidden_witness", "feature", "Eight-landmark hidden-region categorical pose witness", "experiments/okutama_body_witness_protocol.json"),
        ("mechanism:conditional_body_energy", "component", "Competing sitting/standing hypotheses predict excluded body evidence", "src/hac/body_witness.py"),
        ("architecture:body_witness_V0", "architecture", "V0 context-only control", "experiments/run_okutama_body_witness.py"),
        ("architecture:body_witness_V1", "architecture", "V1 equal RGB/geometry control; 790,153 active params", "experiments/run_okutama_body_witness.py"),
        ("architecture:body_witness_V2", "architecture", "V2 equal-input direct pose reader; 796,361 active params", "experiments/run_okutama_body_witness.py"),
        ("architecture:body_witness_V3", "architecture", "V3 conditional verifier; 821,287 active params", "experiments/run_okutama_body_witness.py"),
        ("experiment:body_witness_screen_v1", "experiment", "Superseded synthetic preflight with invalid controls", ".runs/research_20260913/body_witness_screen_v1/synthetic_preflight_receipt.json"),
        ("experiment:body_witness_screen_v4", "experiment", "Corrected 80-update synthetic preflight; no task data", ".runs/research_20260913/body_witness_screen_v4/synthetic_preflight_receipt.json"),
        ("failure:body_witness_capacity_v1", "failure_mechanism", "V3 full active capacity exceeded V1 by 36.11%", ".runs/research_20260913/body_witness_screen_v1/screen_execution_plan.json"),
        ("failure:body_witness_equal_input_v1", "failure_mechanism", "V3 alone indirectly accessed four masked descriptors", "experiments/okutama_body_witness_protocol.json"),
        ("gate:body_human_review", "gate", "Two independent blinded landmark reviews", ".runs/research_20260913/body_witness_pilot_v1/blinded_review/reviewer_a.csv"),
        ("gate:pose_exact_parity", "gate", "Official-byte identity plus full-heatmap parity", ".runs/research_20260913/body_witness_pilot_v1/review/POSE_MODEL_LOCK_CANDIDATE.json"),
        ("failure:hf_vitpose_epsilon", "failure_mechanism", "HF converted default epsilon 1e-12 differs from official 1e-6", ".runs/research_20260913/body_witness_pilot_v1/review/POSE_MODEL_PROVENANCE_DECISION.md"),
        ("gate:body_availability", "gate", "At least 103/128 usable; every scenario represented", "experiments/okutama_body_witness_protocol.json"),
        ("gate:body_target_sensitivity", "gate", "TV >0.05 for at least half of eligible witnesses", "experiments/audit_okutama_body_witness.py"),
        ("failure:body_nested_integration", "failure_mechanism", "40/59 required F populations structurally infeasible", ".runs/research_20260913/body_witness_nested_plan_v3/summary.json"),
        ("action:retain_arftr", "decision", "Copy retained ARFTR probability bytes exactly", "src/hac/body_witness.py"),
        ("decision:body_witness_next", "decision", "Complete blinded review then exact pose parity; no task fit yet", ".runs/research_20260913/body_witness_phase1_20260913/REPORT.md"),
    ]
    for node in nodes:
        add_node(graph, *node)

    edge_rows = [
        ("experiment:okutama_temporal_controls/t2_fixed_distinct_probabilities", "experiment:arftr_v1/r5_arftr_full", "precedes-improvement", ".runs/research_20260912/invention_synthesis_20260912_234907/REPORT.md", "+13.46_macro_f1_points", "verified_internal_comparison"),
        ("experiment:arftr_v1/r5_arftr_full", "experiment:body_witness_native_pilot", "suggests-next-test", ".runs/research_20260912/invention_synthesis_20260912_234907/portfolio.json", "targets_224_posture_swaps", "hypothesis"),
        ("experiment:body_witness_native_pilot", "data:body_witness_pilot_128", "evaluated-on", ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json", "128_centers", "replayed"),
        ("experiment:body_witness_native_pilot", "gate:body_human_review", "suggests-next-test", ".runs/research_20260913/body_witness_pilot_v1/blinded_review/index_v2.html", "two_reviewers_pending", "observed_blocker"),
        ("gate:pose_exact_parity", "measurement:pose_hidden_witness", "enables", ".runs/research_20260913/body_witness_pilot_v1/review/POSE_MODEL_LOCK_CANDIDATE.json", "not_yet", "blocked"),
        ("failure:hf_vitpose_epsilon", "gate:pose_exact_parity", "falsifies-hypothesis", ".runs/research_20260913/body_witness_pilot_v1/review/POSE_MODEL_PROVENANCE_DECISION.md", "stock_HF_not_exact", "primary_source_code"),
        ("gate:body_human_review", "gate:body_availability", "enables", "experiments/okutama_body_witness_protocol.json", "not_yet", "prospective"),
        ("gate:pose_exact_parity", "gate:body_availability", "enables", "experiments/okutama_body_witness_protocol.json", "not_yet", "prospective"),
        ("gate:pose_exact_parity", "gate:body_target_sensitivity", "enables", "experiments/okutama_body_witness_protocol.json", "not_yet", "prospective"),
        ("features:six_slot_body_bank", "architecture:body_witness_V1", "input-to", "experiments/run_okutama_body_witness.py", "4608_dimensions", "implemented"),
        ("features:six_slot_body_bank", "architecture:body_witness_V2", "input-to", "experiments/run_okutama_body_witness.py", "4608_dimensions", "implemented"),
        ("features:six_slot_body_bank", "architecture:body_witness_V3", "input-to", "experiments/run_okutama_body_witness.py", "4608_dimensions", "implemented"),
        ("features:body_quality_36", "architecture:body_witness_V1", "input-to", "experiments/run_okutama_body_witness.py", "36_dimensions", "implemented"),
        ("features:body_quality_36", "architecture:body_witness_V2", "input-to", "experiments/run_okutama_body_witness.py", "36_dimensions", "implemented"),
        ("features:body_quality_36", "architecture:body_witness_V3", "input-to", "experiments/run_okutama_body_witness.py", "36_dimensions", "implemented"),
        ("measurement:pose_hidden_witness", "architecture:body_witness_V2", "input-to", "experiments/okutama_body_witness_protocol.json", "direct_measurement_control", "prospective"),
        ("measurement:pose_hidden_witness", "mechanism:conditional_body_energy", "evaluated-by", "src/hac/body_witness.py", "normalized_CE_over_193_bins", "implemented"),
        ("mechanism:conditional_body_energy", "architecture:body_witness_V3", "adds-component", "src/hac/body_witness.py", "V3_minus_V2_primary_contrast", "prospective"),
        ("architecture:body_witness_V2", "architecture:body_witness_V3", "compared-to", "experiments/okutama_body_witness_protocol.json", "required_plus_0.5_BA_point", "prospective_gate"),
        ("failure:body_witness_capacity_v1", "experiment:body_witness_screen_v1", "falsifies-hypothesis", ".runs/research_20260913/body_witness_screen_v1/screen_execution_plan.json", "36.11_percent_range", "independent_audit"),
        ("failure:body_witness_equal_input_v1", "experiment:body_witness_screen_v1", "falsifies-hypothesis", "experiments/okutama_body_witness_protocol.json", "four_unmatched_masked_descriptors", "independent_audit"),
        ("experiment:body_witness_screen_v1", "experiment:body_witness_screen_v4", "superseded-by", ".runs/research_20260913/body_witness_screen_v4/screen_execution_plan.json", "controls_corrected_before_task_fit", "verified"),
        ("architecture:body_witness_V0", "experiment:body_witness_screen_v4", "evaluated-by", ".runs/research_20260913/body_witness_screen_v4/synthetic_preflight_receipt.json", "20_synthetic_updates", "preflight_only"),
        ("architecture:body_witness_V1", "experiment:body_witness_screen_v4", "evaluated-by", ".runs/research_20260913/body_witness_screen_v4/synthetic_preflight_receipt.json", "20_synthetic_updates", "preflight_only"),
        ("architecture:body_witness_V2", "experiment:body_witness_screen_v4", "evaluated-by", ".runs/research_20260913/body_witness_screen_v4/synthetic_preflight_receipt.json", "20_synthetic_updates", "preflight_only"),
        ("architecture:body_witness_V3", "experiment:body_witness_screen_v4", "evaluated-by", ".runs/research_20260913/body_witness_screen_v4/synthetic_preflight_receipt.json", "20_synthetic_updates", "preflight_only"),
        ("gate:body_availability", "experiment:body_witness_screen_v4", "required-before", "experiments/okutama_body_witness_protocol.json", "not_yet", "blocked"),
        ("gate:body_target_sensitivity", "architecture:body_witness_V3", "required-before", "experiments/okutama_body_witness_protocol.json", "not_yet", "blocked"),
        ("architecture:body_witness_V3", "action:retain_arftr", "preserves", "src/hac/body_witness.py", "exact_on_nonintervention", "tested"),
        ("failure:body_nested_integration", "architecture:body_witness_V3", "blocks-integration-of", ".runs/research_20260913/body_witness_nested_plan_v3/summary.json", "40_of_59_F_populations", "structural_proof"),
        ("gate:body_human_review", "decision:body_witness_next", "suggests-next-test", ".runs/research_20260913/body_witness_phase1_20260913/REPORT.md", "complete_first", "direct_recommendation"),
        ("gate:pose_exact_parity", "decision:body_witness_next", "suggests-next-test", ".runs/research_20260913/body_witness_phase1_20260913/REPORT.md", "resolve_before_pose", "direct_recommendation"),
        ("decision:body_witness_next", "action:retain_arftr", "preserves", ".runs/research_20260913/body_witness_phase1_20260913/REPORT.md", "until_all_gates_pass", "decision"),
    ]
    for source, target, relation, artifact, effect, confidence in edge_rows:
        add_edge(graph, source, target, relation, artifact, effect_size=effect, confidence=confidence)

    graphml = output / "knowledge_graph.graphml"
    graph_json = output / "knowledge_graph.json"
    nx.write_graphml(graph, graphml)
    serial = {
        "directed": True,
        "multigraph": True,
        "nodes": [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)],
        "edges": [
            {"source": source, "target": target, "key": key, **attrs}
            for source, target, key, attrs in graph.edges(keys=True, data=True)
        ],
    }
    graph_json.write_text(json.dumps(serial, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = {
        "status": "PASS",
        "source_graph": {"nodes": before[0], "edges": before[1], "sha256": sha256_file(SOURCE)},
        "updated_graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        "required_nodes_present": all(node[0] in graph for node in nodes),
        "arftr_sha256": sha256_file(ARFTR),
        "arftr_matches_retained_pin": sha256_file(ARFTR) == "ec9957a9393e6f1803d274d281549c20290ed59343b308c5be3446be37cec720",
        "protocol_sha256": sha256_file(PROTOCOL),
        "graphml_sha256": sha256_file(graphml),
        "graph_json_sha256": sha256_file(graph_json),
        "report_exists": (output / "REPORT.md").is_file(),
        "mind_map_exists": (output / "MIND_MAP.md").is_file(),
        "ledger_rows": len(json.loads((output / "phase1_ledger.json").read_text(encoding="utf-8"))),
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        },
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
