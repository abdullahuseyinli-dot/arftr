"""Append the completed-review findings and next branch to the HAC graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / ".runs/research_20260913/body_witness_phase1_20260913/knowledge_graph.graphml"
)
OUTPUT = ROOT / ".runs/research_20260913/continuation_after_review_v4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    graph = nx.read_graphml(BASE, force_multigraph=True)
    nodes = {
        "review:body_reviewer_a": {
            "kind": "human_reference",
            "label": "Reviewer A: 128 visibility decisions, v5 coordinates corrupted",
            "artifact": ".runs/research_20260913/body_witness_pilot_v1/reviewer_a_provisional_receipt_v1.json",
        },
        "failure:review_v5_horizontal_compression": {
            "kind": "failure_mechanism",
            "label": "70vh object-fit content narrower than full-width canvas compressed exported x",
            "artifact": ".runs/research_20260913/body_witness_review_geometry_v1/css_inverse_receipt_v1.json",
        },
        "diagnostic:pose_raw_7_of_128": {
            "kind": "diagnostic_result",
            "label": "7/128 over corrupted v5 reference; not a pose-capability result",
            "artifact": ".runs/research_20260913/body_witness_pose_pilot_v1/provisional_summary.json",
        },
        "diagnostic:pose_css_inverse_65_of_128": {
            "kind": "diagnostic_result",
            "label": "CSS inverse: 65/128, 662/822 joints, median error 0.04257H; not a gate",
            "artifact": ".runs/research_20260913/body_witness_review_geometry_v1/css_inverse_pose_diagnostic_v1.json",
        },
        "audit:pose_cache_bit_exact_replay": {
            "kind": "audit",
            "label": "All 768 pose inputs replayed bit-exact; heatmap stream unchanged",
            "artifact": ".runs/research_20260913/body_witness_review_geometry_v1/pose_replay_audit_v1.json",
        },
        "gate:reviewer_a_anatomical_ceiling": {
            "kind": "gate",
            "label": "Reviewer A anatomical ceiling 92/128 versus required 103",
            "artifact": ".runs/research_20260913/body_witness_pose_pilot_v1/provisional_availability_receipt.json",
        },
        "interface:reviewer_v6": {
            "kind": "component",
            "label": "Geometry-corrected blinded reviewer v6 with isolated browser state",
            "artifact": ".runs/research_20260913/body_witness_pilot_v1/blinded_review/index_v6_interactive.html",
        },
        "decision:body_witness_pending": {
            "kind": "decision",
            "label": "Keep pose cache; no Body Witness task fit; second review deferred and outside the current critical path",
            "artifact": "experiments/okutama_body_witness_protocol.json",
        },
        "decision:reviewer_b_deferred": {
            "kind": "decision",
            "label": "Reviewer B may close the pose branch later but is not required for the non-pose branch",
            "artifact": ".runs/research_20260913/body_witness_pilot_v1/reviewer_a_provisional_receipt_v1.json",
        },
        "architecture:center_evidence_completion": {
            "kind": "architecture",
            "label": "Visible-Anchor Partial Transport Completion with dustbin/no-transport",
            "artifact": "experiments/okutama_center_evidence_completion_protocol.json",
        },
        "protocol:center_completion_schema7_executed": {
            "kind": "protocol",
            "label": "Immutable schema-7 execution lock; SHA 930fc6c9...",
            "artifact": ".runs/research_20260913/center_evidence_completion_preflight_v4/source_snapshot/okutama_center_evidence_completion_protocol.json",
        },
        "plan:center_evidence_completion_v7": {
            "kind": "experiment",
            "label": "Schema 7 / plan v7: 128 centers, 1392 frozen-DINO forwards, fixed wrong-track control",
            "artifact": ".runs/research_20260913/center_evidence_completion_v1/plan_v7.json",
        },
        "protocol:center_completion_schema8_authorized": {
            "kind": "protocol",
            "label": "Schema 8 authorizes one immutable label-blind 128-center cache; task fits remain forbidden",
            "artifact": "experiments/okutama_center_evidence_completion_protocol.json",
        },
        "plan:center_evidence_completion_v8": {
            "kind": "experiment",
            "label": "Plan v8: full extraction authorized, 1392 frozen-DINO forwards, task fits zero",
            "artifact": ".runs/research_20260913/center_evidence_completion_v1/plan_v8.json",
        },
        "control:wrong_track_map_v4": {
            "kind": "control",
            "label": "Wrong-track map v4: 512 rows; donor rows identical to v3; schema-8 ancestry updated",
            "artifact": ".runs/research_20260913/center_evidence_completion_v1/wrong_track_donor_map_v4.json",
        },
        "data:provider_720_jpeg_source": {
            "kind": "dataset",
            "label": "Pinned provider-supplied 1280x720 JPEG source; archive SHA c021ce8a...",
            "artifact": ".runs/research_20260907/okutama_native_video_p0_r1/materialization_lock.json",
        },
        "gate:center_completion_strong_controls": {
            "kind": "gate",
            "label": "P3 must beat P0, P1, P2 and repeated-center; identity and spatial controls must fail",
            "artifact": "experiments/okutama_center_evidence_completion_protocol.json",
        },
        "failure:hard_donor_entropy_abstention": {
            "kind": "failure_mechanism",
            "label": "Hard donor-identity entropy confused redundant agreement with uncertainty; 0/32 rows transported",
            "artifact": ".runs/research_20260913/center_evidence_completion_preflight_v1/summary.json",
        },
        "component:learned_label_blind_transport_gate": {
            "kind": "component",
            "label": "Explicit retain/transport gate trained only by masked reconstruction and donor disagreement",
            "artifact": "experiments/okutama_center_evidence_completion_protocol.json",
        },
        "audit:center_completion_preflight_v2": {
            "kind": "audit",
            "label": "Schema-6 preflight v2: corrected consensus, 112 inputs, 126/128 affine fits, 32/32 rows exercised",
            "artifact": ".runs/research_20260913/center_evidence_completion_preflight_v2/summary.json",
        },
        "audit:center_completion_preflight_v2_replay": {
            "kind": "audit",
            "label": "Independent v2 replay: six arrays and four substantive receipts byte-identical",
            "artifact": ".runs/research_20260913/center_evidence_completion_preflight_v2_audit/audit_receipt.json",
        },
        "audit:center_completion_preflight_v4": {
            "kind": "audit",
            "label": "Control-inclusive schema-7 preflight: 176 inputs; true 126/128 fits; wrong-track 93/128",
            "artifact": ".runs/research_20260913/center_evidence_completion_preflight_v4/summary.json",
        },
        "audit:center_completion_preflight_v4_replay": {
            "kind": "audit",
            "label": "Independent v4 replay: all eight arrays and four substantive receipts byte-identical",
            "artifact": ".runs/research_20260913/center_evidence_completion_preflight_v4_audit/audit_receipt.json",
        },
        "boundary:center_completion_operational_only": {
            "kind": "evidence_boundary",
            "label": "Operational readiness only: no reconstruction score, task fit, router fit, or macro-F1 result",
            "artifact": ".runs/research_20260913/center_evidence_completion_preflight_v4_audit/audit_receipt.json",
        },
        "action:center_completion_full_cache": {
            "kind": "next_action",
            "label": "Next: full 128-center label-blind cache and reconstruction screen; task training remains forbidden",
            "artifact": "experiments/okutama_center_evidence_completion_protocol.json",
        },
    }
    graph.add_nodes_from(nodes.items())

    def edge(source: str, target: str, relation: str, artifact: str, effect: str) -> None:
        graph.add_edge(
            source,
            target,
            relation=relation,
            artifact=artifact,
            effect_size=effect,
            confidence="high" if "diagnostic" not in source else "provisional",
            evidence_strength="artifact_or_calculation",
        )

    edge(
        "review:body_reviewer_a",
        "diagnostic:pose_raw_7_of_128",
        "evaluated-on",
        nodes["diagnostic:pose_raw_7_of_128"]["artifact"],
        "7_of_128",
    )
    edge(
        "failure:review_v5_horizontal_compression",
        "review:body_reviewer_a",
        "corrupts-coordinate-field-of",
        nodes["failure:review_v5_horizontal_compression"]["artifact"],
        "113_height_capped_rows",
    )
    edge(
        "failure:review_v5_horizontal_compression",
        "diagnostic:pose_raw_7_of_128",
        "invalidates-capability-inference-from",
        nodes["failure:review_v5_horizontal_compression"]["artifact"],
        "median_x_residual_0.13021H",
    )
    edge(
        "diagnostic:pose_css_inverse_65_of_128",
        "diagnostic:pose_raw_7_of_128",
        "contradicts-pose-failure-explanation",
        nodes["diagnostic:pose_css_inverse_65_of_128"]["artifact"],
        "7_to_65;225_to_662_joints",
    )
    edge(
        "audit:pose_cache_bit_exact_replay",
        "diagnostic:pose_css_inverse_65_of_128",
        "supports-measurement-integrity-of",
        nodes["audit:pose_cache_bit_exact_replay"]["artifact"],
        "768_of_768_inputs_bit_exact",
    )
    edge(
        "gate:reviewer_a_anatomical_ceiling",
        "decision:body_witness_pending",
        "blocks-current-single-review-promotion",
        nodes["gate:reviewer_a_anatomical_ceiling"]["artifact"],
        "92_vs_103",
    )
    edge(
        "interface:reviewer_v6",
        "failure:review_v5_horizontal_compression",
        "corrects",
        nodes["interface:reviewer_v6"]["artifact"],
        "image_canvas_aspect_locked",
    )
    edge(
        "decision:body_witness_pending",
        "action:retain_arftr",
        "preserves",
        nodes["decision:body_witness_pending"]["artifact"],
        "85.383648_macro_f1",
    )
    edge(
        "decision:body_witness_pending",
        "decision:reviewer_b_deferred",
        "removes-from-current-critical-path",
        nodes["decision:reviewer_b_deferred"]["artifact"],
        "non_pose_branch_proceeds_now",
    )
    edge(
        "decision:body_witness_pending",
        "architecture:center_evidence_completion",
        "suggests-next-test",
        nodes["architecture:center_evidence_completion"]["artifact"],
        "fallback_planning_only",
    )
    edge(
        "architecture:center_evidence_completion",
        "plan:center_evidence_completion_v7",
        "evaluated-by-planned",
        nodes["plan:center_evidence_completion_v7"]["artifact"],
        "128_centers;1392_forwards",
    )
    edge(
        "protocol:center_completion_schema7_executed",
        "audit:center_completion_preflight_v4",
        "governs-executed",
        nodes["protocol:center_completion_schema7_executed"]["artifact"],
        "schema_7_sha_930fc6c9",
    )
    edge(
        "plan:center_evidence_completion_v7",
        "audit:center_completion_preflight_v4",
        "specifies-executed",
        nodes["plan:center_evidence_completion_v7"]["artifact"],
        "plan_v7_sha_3a218aed",
    )
    edge(
        "data:provider_720_jpeg_source",
        "plan:center_evidence_completion_v7",
        "evaluated-on",
        nodes["plan:center_evidence_completion_v7"]["artifact"],
        "128_center_and_512_neighbor_slots",
    )
    edge(
        "gate:center_completion_strong_controls",
        "architecture:center_evidence_completion",
        "falsifies-unless",
        nodes["gate:center_completion_strong_controls"]["artifact"],
        "5pct_vs_P0_and_P2;10pct_vs_P1",
    )
    edge(
        "failure:hard_donor_entropy_abstention",
        "component:learned_label_blind_transport_gate",
        "suggests-next-test",
        nodes["failure:hard_donor_entropy_abstention"]["artifact"],
        "0_of_32_to_32_of_32_consensus_rows",
    )
    edge(
        "component:learned_label_blind_transport_gate",
        "architecture:center_evidence_completion",
        "adds-component",
        nodes["component:learned_label_blind_transport_gate"]["artifact"],
        "explicit_retain_per_target",
    )
    edge(
        "audit:center_completion_preflight_v2",
        "architecture:center_evidence_completion",
        "supports-operational-feasibility-of",
        nodes["audit:center_completion_preflight_v2"]["artifact"],
        "126_of_128_fits;32_of_32_consensus",
    )
    edge(
        "failure:hard_donor_entropy_abstention",
        "audit:center_completion_preflight_v2",
        "corrected-by",
        nodes["audit:center_completion_preflight_v2"]["artifact"],
        "0_of_32_to_32_of_32_rows",
    )
    edge(
        "audit:center_completion_preflight_v2_replay",
        "audit:center_completion_preflight_v2",
        "independently-reproduces",
        nodes["audit:center_completion_preflight_v2_replay"]["artifact"],
        "6_arrays_and_4_receipts_byte_exact",
    )
    edge(
        "audit:center_completion_preflight_v4",
        "architecture:center_evidence_completion",
        "supports-operational-feasibility-of",
        nodes["audit:center_completion_preflight_v4"]["artifact"],
        "true_3413_of_3413;wrong_3107_of_3413_targets",
    )
    edge(
        "audit:center_completion_preflight_v4_replay",
        "audit:center_completion_preflight_v4",
        "independently-reproduces",
        nodes["audit:center_completion_preflight_v4_replay"]["artifact"],
        "8_arrays_and_4_receipts_byte_exact",
    )
    edge(
        "audit:center_completion_preflight_v4_replay",
        "boundary:center_completion_operational_only",
        "limited-by",
        nodes["boundary:center_completion_operational_only"]["artifact"],
        "scientific_gate_not_evaluated",
    )
    edge(
        "boundary:center_completion_operational_only",
        "action:retain_arftr",
        "requires-preservation-of",
        nodes["boundary:center_completion_operational_only"]["artifact"],
        "85.383648_macro_f1_unchanged",
    )
    edge(
        "audit:center_completion_preflight_v4_replay",
        "action:center_completion_full_cache",
        "suggests-next-test",
        nodes["action:center_completion_full_cache"]["artifact"],
        "128_centers;projected_402.36_seconds",
    )
    edge(
        "audit:center_completion_preflight_v4_replay",
        "protocol:center_completion_schema8_authorized",
        "authorizes-successor",
        nodes["protocol:center_completion_schema8_authorized"]["artifact"],
        "operational_pass_to_label_blind_cache_only",
    )
    edge(
        "protocol:center_completion_schema8_authorized",
        "plan:center_evidence_completion_v8",
        "governs",
        nodes["plan:center_evidence_completion_v8"]["artifact"],
        "schema_8_sha_f2e95656;plan_v8_sha_b7e148c8",
    )
    edge(
        "plan:center_evidence_completion_v8",
        "control:wrong_track_map_v4",
        "uses-control",
        nodes["control:wrong_track_map_v4"]["artifact"],
        "496_of_512_available;rows_identical_to_v3",
    )
    edge(
        "plan:center_evidence_completion_v8",
        "action:center_completion_full_cache",
        "specifies",
        nodes["plan:center_evidence_completion_v8"]["artifact"],
        "1392_frozen_DINO_forwards;task_fits_0",
    )
    edge(
        "failure:body_nested_integration",
        "architecture:center_evidence_completion",
        "blocks-integration-until-resolved",
        ".runs/research_20260913/body_witness_nested_plan_v3/summary.json",
        "representation_screen_only_until_feasible",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    graphml = OUTPUT / "knowledge_graph.graphml"
    graph_json = OUTPUT / "knowledge_graph.json"
    receipt_path = OUTPUT / "graph_receipt.json"
    existing = [path for path in (graphml, graph_json, receipt_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite graph artifacts: {existing}")
    nx.write_graphml(graph, graphml)
    with graph_json.open("x", encoding="utf-8") as stream:
        json.dump(nx.node_link_data(graph, edges="edges"), stream, indent=2, sort_keys=True)
        stream.write("\n")
    receipt = {
        "status": "POST_REVIEW_KNOWLEDGE_GRAPH_COMPLETE",
        "base_graph_sha256": sha256_file(BASE),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "graphml_sha256": sha256_file(graphml),
        "json_sha256": sha256_file(graph_json),
        "source_sha256": sha256_file(Path(__file__)),
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
