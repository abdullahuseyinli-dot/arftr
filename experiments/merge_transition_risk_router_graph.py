"""Create an additive final knowledge graph for the 2026-09-16 router cycle.

The historical graph is immutable.  This script copies it into the dated run,
adds the transition-risk-router evidence and final measured result, and writes
JSON/GraphML plus a receipt with source hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260916/source_posture_failure_router_v1"
PARENT = ROOT / ".runs/research_20260913/continuation_after_review_v5"
OUT_JSON = RUN / "knowledge_graph_final.json"
OUT_GRAPHML = RUN / "knowledge_graph_final.graphml"
OUT_RECEIPT = RUN / "knowledge_graph_final_receipt.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (OUT_JSON, OUT_GRAPHML, OUT_RECEIPT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    graph = nx.read_graphml(PARENT / "knowledge_graph.graphml", force_multigraph=True)
    delta = json.loads((RUN / "knowledge_graph_delta.json").read_text(encoding="utf-8"))
    additions = list(delta["nodes"])
    additions.extend(
        [
            {
                "id": "result:transition_risk_router_v1",
                "kind": "metric",
                "label": "Nested router 0.854632593 macro-F1; +0.0796 pp vs ARFTR; net +2",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/summary.json",
                "effect_size": 0.0007961120509410099,
                "effect_units": "macro_f1_delta_vs_ARFTR",
                "evidence_strength": "independent_replay_and_saved_predictions",
            },
            {
                "id": "audit:transition_risk_router_independent_replay",
                "kind": "audit",
                "label": "Independent replay pass; five folds; exact router/action replay",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/independent_audit.json",
                "evidence_strength": "independent_exact_replay",
            },
            {
                "id": "failure:transition_risk_router_gate",
                "kind": "failed_gate",
                "label": "Router not promoted: negative folds and bootstrap interval crosses zero",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/summary.json",
                "evidence_strength": "predeclared_gate_evaluation",
            },
        ]
    )
    edges = list(delta["edges"])
    edges.extend(
        [
            {
                "source": "architecture:arftr_transition_risk_router_v1",
                "target": "result:transition_risk_router_v1",
                "relation": "measured-as",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/summary.json",
                "effect_size": 0.0007961120509410099,
                "effect_units": "macro_f1_delta_vs_ARFTR",
                "evidence_strength": "independent_replay_and_saved_predictions",
            },
            {
                "source": "result:transition_risk_router_v1",
                "target": "audit:transition_risk_router_independent_replay",
                "relation": "audited-by",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/independent_audit.json",
                "evidence_strength": "independent_exact_replay",
            },
            {
                "source": "result:transition_risk_router_v1",
                "target": "failure:transition_risk_router_gate",
                "relation": "fails",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/summary.json",
                "effect_size": 2,
                "effect_units": "net_corrections; per_fold [-1,-2,0,5,0]",
                "evidence_strength": "predeclared_gate_evaluation",
            },
            {
                "source": "failure:transition_risk_router_gate",
                "target": "action:retain_arftr",
                "relation": "preserves",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/summary.json",
                "evidence_strength": "decision",
            },
            {
                "source": "audit:transition_risk_router_independent_replay",
                "target": "protocol:router_nested_crossfit",
                "relation": "supports",
                "artifact": ".runs/research_20260916/source_posture_failure_router_v1/router_results/independent_audit.json",
                "evidence_strength": "zero_outer_label_reads",
            },
        ]
    )

    for node in additions:
        node_id = str(node["id"])
        attrs = {k: str(v) for k, v in node.items() if k != "id"}
        if node_id in graph:
            # Delta files may restate an existing historical anchor/action.
            # Preserve the parent attributes and add only genuinely new fields.
            for key, value in attrs.items():
                graph.nodes[node_id].setdefault(key, value)
        else:
            graph.add_node(node_id, **attrs)
    for index, edge in enumerate(edges):
        source, target = edge["source"], edge["target"]
        if source not in graph or target not in graph:
            raise ValueError(f"Graph edge endpoint missing: {source} -> {target}")
        attrs = {k: str(v) for k, v in edge.items() if k not in {"source", "target"}}
        attrs.setdefault("key", str(index))
        graph.add_edge(source, target, **attrs)

    nodes = [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)]
    graph_edges = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        graph_edges.append({"source": source, "target": target, "key": key, **attrs})
    OUT_JSON.write_text(
        json.dumps(
            {
                "graph_scope": "historical_parent_plus_transition_risk_router_final",
                "parent_graph": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
                "nodes": nodes,
                "edges": graph_edges,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    nx.write_graphml(graph, OUT_GRAPHML)
    receipt = {
        "status": "KNOWLEDGE_GRAPH_FINAL_MERGE_COMPLETE",
        "parent_graph": str((PARENT / "knowledge_graph.graphml").relative_to(ROOT)).replace("\\", "/"),
        "parent_sha256": sha256_file(PARENT / "knowledge_graph.graphml"),
        "delta_sha256": sha256_file(RUN / "knowledge_graph_delta.json"),
        "final_json_sha256": sha256_file(OUT_JSON),
        "final_graphml_sha256": sha256_file(OUT_GRAPHML),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "router_summary": "router_results/summary.json",
    }
    OUT_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
