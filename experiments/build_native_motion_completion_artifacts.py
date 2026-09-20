"""Extend the council graph and write the fixed-trial decision record."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import networkx as nx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.actor_memory_base import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260920/native_motion_innovation_v1"
BASE_GRAPH = ROOT / ".runs/research_20260920/aerial_council_v1/knowledge_graph.graphml"
RGB = ROOT / ".runs/research_20260920/rgb_witness_v1/summary_v2.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    motion = read(RUN / "summary.json")
    postmortem = read(RUN / "postmortem.json")
    rgb = read(RGB)
    graph = nx.read_graphml(BASE_GRAPH)
    artifact = ".runs/research_20260920/native_motion_innovation_v1/summary.json"
    nodes = {
        "experiment:rgb_witness_v1_executed": {
            "kind": "experiment", "label": "Pose-free conditional RGB witness (executed, closed)",
            "artifact": ".runs/research_20260920/rgb_witness_v1/summary_v2.json",
            "macro_f1": rgb["scores"]["conditional"]["raw"]["macro_f1"],
            "promoted": str(rgb["promoted"]).lower(),
        },
        "mechanism:fixed_rgb_center_witness": {
            "kind": "mechanism", "label": "Disjoint upper/lower frozen-RGB center witness",
            "artifact": "src/hac/rgb_witness.py",
        },
        "experiment:native_motion_innovation_v1": {
            "kind": "experiment", "label": "Camera-compensated exact-native motion innovation",
            "artifact": artifact, "promoted": str(motion["promoted"]).lower(),
        },
        "mechanism:signed_physical_time_motion": {
            "kind": "mechanism", "label": "Ordered signed local RGB change at +/-0.13347s",
            "artifact": "src/hac/native_motion_innovation.py",
        },
        "control:motion_appearance": {
            "kind": "control", "label": "Matched pair-mean appearance control", "artifact": artifact,
            "macro_f1": motion["scores"]["appearance"]["raw"]["macro_f1"],
        },
        "control:motion_phase_destroyed": {
            "kind": "control", "label": "Matched absolute-difference phase-destroyed control", "artifact": artifact,
            "macro_f1": motion["scores"]["phase_destroyed"]["raw"]["macro_f1"],
        },
        "result:native_motion_primary": {
            "kind": "result", "label": "Signed native-motion primary result", "artifact": artifact,
            "macro_f1": motion["scores"]["signed_motion"]["routed"]["macro_f1"],
            "errors": motion["scores"]["signed_motion"]["routed"]["errors"],
            "net": motion["details"]["signed_motion"]["net"],
        },
        "capacity:native_motion_fixed_action": {
            "kind": "diagnostic_bound", "label": "153 reachable ARFTR pair errors under fixed cap (oracle only)",
            "artifact": ".runs/research_20260920/native_motion_innovation_v1/capacity_before_results.json",
        },
        "decision:retain_arftr_after_motion": {
            "kind": "decision", "label": "Retain ARFTR unless every prospective gate passes", "artifact": artifact,
        },
    }
    for node, attrs in nodes.items():
        graph.add_node(node, **attrs)

    def edge(source: str, target: str, relation: str, evidence: str, effect: float | None = None) -> None:
        attrs = {"relation": relation, "artifact": artifact, "evidence_strength": evidence,
                 "confidence": "prospective_nested_outer_evaluation"}
        if effect is not None:
            attrs["effect_size"] = float(effect)
            attrs["effect_metric"] = "macro_f1_difference"
        graph.add_edge(source, target, **attrs)

    edge("mechanism:fixed_rgb_center_witness", "experiment:rgb_witness_v1_executed", "evaluated-by", "120 producers + independent replay")
    edge("experiment:rgb_witness_v1_executed", "experiment:native_motion_innovation_v1", "suggests-next-test", "RGB objective failed; larger pair-error capacity")
    edge("capacity:native_motion_fixed_action", "experiment:native_motion_innovation_v1", "supports-test", "calculated before motion results")
    edge("mechanism:signed_physical_time_motion", "experiment:native_motion_innovation_v1", "adds-component", "fixed protocol")
    edge("control:motion_appearance", "experiment:native_motion_innovation_v1", "matched-control", "same architecture and parameter count")
    edge("control:motion_phase_destroyed", "experiment:native_motion_innovation_v1", "matched-control", "same architecture and parameter count")
    signed_raw = motion["scores"]["signed_motion"]["raw"]["macro_f1"]
    edge("mechanism:signed_physical_time_motion", "control:motion_appearance", "compared-with", "outer OOF raw predictions",
         signed_raw - motion["scores"]["appearance"]["raw"]["macro_f1"])
    edge("mechanism:signed_physical_time_motion", "control:motion_phase_destroyed", "compared-with", "outer OOF raw predictions",
         signed_raw - motion["scores"]["phase_destroyed"]["raw"]["macro_f1"])
    edge("experiment:native_motion_innovation_v1", "result:native_motion_primary", "produces", "independently replayed")
    if motion["promoted"]:
        edge("result:native_motion_primary", "arch_arftr_factor_anchor", "improves", "all preregistered gates passed",
             motion["scores"]["signed_motion"]["routed"]["macro_f1"] - motion["scores"]["arftr"]["macro_f1"])
    else:
        edge("result:native_motion_primary", "decision:retain_arftr_after_motion", "fails-promotion", "one or more preregistered gates failed",
             motion["scores"]["signed_motion"]["routed"]["macro_f1"] - motion["scores"]["arftr"]["macro_f1"])
    graph_path = RUN / "knowledge_graph_extended.graphml"
    nx.write_graphml(graph, graph_path)
    graph_json = {
        "schema_version": "hac_knowledge_extension_v1", "date": "2026-09-20",
        "base_graph": str(BASE_GRAPH.relative_to(ROOT)).replace("\\", "/"),
        "node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges(),
        "nodes": [{"id": node, **dict(attrs)} for node, attrs in graph.nodes(data=True)],
        "edges": [{"source": source, "target": target, **dict(attrs)} for source, target, attrs in graph.edges(data=True)],
    }
    (RUN / "knowledge_graph_extended.json").write_text(json.dumps(graph_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [
        ["Retained ARFTR", "deployable", motion["scores"]["arftr"]["macro_f1"], motion["scores"]["arftr"]["accuracy"], motion["scores"]["arftr"]["errors"], "retained"],
        ["RGB direct raw", "screen", rgb["scores"]["direct"]["raw"]["macro_f1"], rgb["scores"]["direct"]["raw"]["accuracy"], rgb["scores"]["direct"]["raw"]["errors"], "closed"],
        ["RGB conditional raw", "screen", rgb["scores"]["conditional"]["raw"]["macro_f1"], rgb["scores"]["conditional"]["raw"]["accuracy"], rgb["scores"]["conditional"]["raw"]["errors"], "closed"],
    ]
    for arm in ARMS_ORDER:
        score = motion["scores"][arm]["raw"]
        rows.append([f"Native motion {arm} raw", "screen", score["macro_f1"], score["accuracy"], score["errors"], "promote" if arm == "signed_motion" and motion["promoted"] else "control/closed"])
    routed = motion["scores"]["signed_motion"]["routed"]
    rows.append(["Native motion signed routed", "deployable candidate", routed["macro_f1"], routed["accuracy"], routed["errors"], "promote" if motion["promoted"] else "retain ARFTR"])
    with (RUN / "summary_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["result", "kind", "macro_f1", "accuracy", "errors", "decision"]); writer.writerows(rows)

    primary = motion["scores"]["signed_motion"]
    recommendation = ("Run the two fixed confirmation seeds; do not alter the mechanism." if motion["promoted"]
                      else "Close this fixed motion design and move to the preregistered center-identity reacquisition fallback; retain ARFTR.")
    mind = f"""# HAC decision mind map — 2026-09-20

```text
T2 (~71.92 macro-F1)
└── multi-source evidence + grouped OOF factorization
    └── ARFTR 85.383648% / 702 errors  [RETAINED]
        ├── bounded generic corrections: 82.08–85.10%  [harm > rescue]
        ├── sequence family: 64.16–71.78%  [correlated temporal noise / weak supervision]
        ├── pose-free RGB witness
        │   ├── direct raw {rgb['scores']['direct']['raw']['macro_f1']:.6f}
        │   └── conditional raw {rgb['scores']['conditional']['raw']['macro_f1']:.6f}  [mechanism gate failed]
        └── exact-native motion innovation
            ├── appearance raw {motion['scores']['appearance']['raw']['macro_f1']:.6f}
            ├── signed raw {primary['raw']['macro_f1']:.6f}  [primary]
            ├── phase-destroyed raw {motion['scores']['phase_destroyed']['raw']['macro_f1']:.6f}
            └── signed routed {primary['routed']['macro_f1']:.6f} / {primary['routed']['errors']} errors
                └── decision: {'PROMOTE TO CONFIRMATION' if motion['promoted'] else 'NO-GO; EXACT ARFTR RETAIN'}
```

Direct recommendation: {recommendation}
"""
    (RUN / "MIND_MAP.md").write_text(mind, encoding="utf-8")
    report = f"""# Native-motion innovation decision record

The trial is complete and independently replayed. The retained ARFTR result remains {motion['scores']['arftr']['macro_f1']:.9f} macro-F1 with {motion['scores']['arftr']['errors']} errors.

The signed arm produced raw macro-F1 {primary['raw']['macro_f1']:.9f}; its matched appearance and phase-destroyed controls produced {motion['scores']['appearance']['raw']['macro_f1']:.9f} and {motion['scores']['phase_destroyed']['raw']['macro_f1']:.9f}. The leakage-safe routed output produced {primary['routed']['macro_f1']:.9f}, {primary['routed']['errors']} errors, and net {motion['details']['signed_motion']['net']} corrections with fold nets {motion['details']['signed_motion']['per_fold_net']}.

Mechanism gates: `{json.dumps(motion['mechanism_gates'], sort_keys=True)}`. Promotion gates: `{json.dumps(motion['promotion_gates'], sort_keys=True)}`. Promoted: **{motion['promoted']}**.

The earlier conditional RGB witness is closed: conditional raw macro-F1 was {rgb['scores']['conditional']['raw']['macro_f1']:.9f}, below ARFTR, and all routed arms retained exactly because the training-only crossing support gate did not pass. This is evidence against adding another broad center-RGB correction head, not evidence that RGB contains no useful signal.

The action-capacity diagnostic counted 436 ARFTR errors in eligible rows: 409 target-pair errors, 27 sitting errors invariantly unreachable, and 153 target-pair errors reachable under the fixed ±0.5 logit cap. These are oracle bounds only, never achieved performance.

Recommendation: **{recommendation}**
"""
    (RUN / "RESULT_REPORT.md").write_text(report, encoding="utf-8")
    outputs = [graph_path, RUN / "knowledge_graph_extended.json", RUN / "summary_table.csv", RUN / "MIND_MAP.md", RUN / "RESULT_REPORT.md"]
    receipt = {"status": "NATIVE_MOTION_COMPLETION_ARTIFACTS_WRITTEN", "outputs": [
        {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in outputs
    ], "postmortem_status": postmortem["status"]}
    (RUN / "completion_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


ARMS_ORDER = ("appearance", "signed_motion", "phase_destroyed")


if __name__ == "__main__":
    main()
