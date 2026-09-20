"""Extend the HAC evidence graph through the completed screen and causal audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".runs/research_20260913/continuation_after_review_v4/knowledge_graph.graphml"
OUTPUT = ROOT / ".runs/research_20260913/continuation_after_review_v5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    graph = nx.read_graphml(BASE, force_multigraph=True)
    nodes = {
        "cache:center_completion_full_v1": {
            "kind": "feature_cache",
            "label": "Full label-blind cache: 128 centers, 1392 frozen-DINO inputs, nine arrays",
            "artifact": ".runs/research_20260913/center_evidence_completion_cache_v1/summary.json",
            "sha256": "5eec07a433c189de949abeff70e1987fb8fbfb873cb4c0f308921b5833d4f201",
        },
        "audit:center_completion_full_cache_replay": {
            "kind": "audit",
            "label": "Independent full-cache replay: 9/9 arrays byte- and element-exact",
            "artifact": ".runs/research_20260913/center_evidence_completion_cache_v1_audit/audit_receipt.json",
            "sha256": "269eccee054f3313ed254445e5fb5a9cf71a581bf14dc1b4f1cd9306c3e6b1e2",
        },
        "correction:cache_view_access_accounting": {
            "kind": "accounting_correction",
            "label": "1136 is view accesses; actual JPEG decodes 640; scientific arrays unchanged",
            "artifact": ".runs/research_20260913/center_evidence_completion_cache_v1/accounting_correction_v1.json",
            "sha256": "95ad996e7636b4185bf158137d4b70ca6b800e27c606d647c0d79fb931cf48b0",
        },
        "cache:center_completion_transport_v1": {
            "kind": "feature_cache",
            "label": "Transport cache: 27407 targets; true/repeated full; wrong-track 25218",
            "artifact": ".runs/research_20260913/center_evidence_completion_transport_v1/summary.json",
            "sha256": "1e2d3f1af9a096b80283a8d30cd00d06199d6e439064a62639eee3f0d6f7e401",
        },
        "audit:center_completion_transport_replay": {
            "kind": "audit",
            "label": "Independent immutable-source transport replay: 9/9 arrays byte-exact",
            "artifact": ".runs/research_20260913/center_evidence_completion_transport_v1_audit/audit_receipt.json",
            "sha256": "2d3719fc2e8473da5a64a56d40af352b5ef6d324f501761c17c66adfbaa94859",
        },
        "failure:transport_live_source_changed": {
            "kind": "failed_run",
            "label": "First replay interrupted after live screen source changed; excluded from audit",
            "artifact": ".runs/research_20260913/center_evidence_completion_transport_v1_replay1_live_source_changed_failed/request.json",
            "sha256": "99b420b47f09fd94d0aa619783375a2d07b63586d4c75dbe3f9d5c10b81ee3ed",
        },
        "lock:center_completion_screen_v1": {
            "kind": "execution_lock",
            "label": "Prospective scientific screen lock: four arms, 20 fits, all gates required",
            "artifact": "experiments/okutama_center_completion_screen_execution_lock.json",
            "sha256": "bcdd3c8a1f9d600e8e32d00a288f57131d81058e475bfda40738d62a01d9acdd",
        },
        "experiment:center_completion_screen_v1": {
            "kind": "experiment",
            "label": "Label-blind reconstruction screen: 128 centers, 5 folds, seed 42, 8000 updates",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1/summary.json",
            "sha256": "002ad504f36de7e59ac5d345abacbaff548eca9fac718e1661dca95dad0306cd",
        },
        "result:completion_P0": {
            "kind": "metric",
            "label": "P0 center-only mean center cosine error 0.464298",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1/summary.json",
            "value": "0.46429786109365523",
        },
        "result:completion_P1": {
            "kind": "metric",
            "label": "P1 unordered-pool mean center cosine error 0.451158",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1/summary.json",
            "value": "0.4511581806000322",
        },
        "result:completion_P2": {
            "kind": "metric",
            "label": "P2 same-grid mean center cosine error 0.383857",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1/summary.json",
            "value": "0.3838571538217366",
        },
        "result:completion_P3": {
            "kind": "metric",
            "label": "P3 partial-transport mean center cosine error 0.369195",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1/summary.json",
            "value": "0.3691946086473763",
        },
        "failure:completion_P3_vs_P2_gate": {
            "kind": "failed_gate",
            "label": "Only locked gate failure: P3 reduced P2 error 3.819792%, below required 5%",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1_audit/audit_receipt.json",
            "effect_size": "0.038197920836899574_vs_0.05",
        },
        "audit:center_completion_screen_replay": {
            "kind": "audit",
            "label": "No-refit screen audit: 40/40 OOF arrays exact; maximum difference zero",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1_audit/audit_receipt.json",
            "sha256": "af61861442ddeeb8019e863752912b0ebc4e89898c7c12f3b73ed9151ac2c62d",
        },
        "forensics:center_completion_screen_v1": {
            "kind": "statistical_finding",
            "label": "Paired forensics: P3 better on 111/128 centers; adaptive bootstrap CI 3.04%-4.66%",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_forensics_v1/forensics.json",
            "sha256": "77850fc70a4b32d5c173cc2ea7edda409501f03e097649de5cc39222eb139c4b",
        },
        "diagnostic:center_selector_oracle": {
            "kind": "diagnostic_oracle",
            "label": "Per-center min(P2,P3) oracle reaches only 4.032611% vs P2; nondeployable",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_forensics_v1/forensics.json",
            "effect_size": "0.04032611157067323",
        },
        "diagnostic:token_selector_oracle": {
            "kind": "diagnostic_oracle",
            "label": "Old P2/P3 per-token oracle error 0.360598, 6.0594% below P2; obsolete and nondeployable",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v5/tokenwise_forensics.json",
            "sha256": "b103c2ab594e005f9dc2e7affacda377e61fcfb9915919c5d5721da10c365029",
            "effect_size": "0.060594231671688606",
        },
        "finding:P3_token_gate_weak": {
            "kind": "statistical_finding",
            "label": "P3 wins 64.78% of tokens, but existing token gate correctness AUC is only 0.5169",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v5/tokenwise_forensics.json",
            "sha256": "b103c2ab594e005f9dc2e7affacda377e61fcfb9915919c5d5721da10c365029",
            "effect_size": "AUC_0.516927",
        },
        "diagnostic:affine_uniform_convex_oracle": {
            "kind": "diagnostic_oracle",
            "label": "P2-to-C convex token oracle error 0.205218; 3.6722% below direct C; nondeployable",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v5/tokenwise_forensics.json",
            "sha256": "b103c2ab594e005f9dc2e7affacda377e61fcfb9915919c5d5721da10c365029",
        },
        "hypothesis:factorial_architecture_confound": {
            "kind": "rival_hypothesis",
            "label": "P2-P3 comparison changes transport and correction/gating together",
            "artifact": ".runs/research_20260913/continuation_after_review_v5/NEXT_ACTIONS.json",
        },
        "hypothesis:global_affine_too_rigid": {
            "kind": "rival_hypothesis",
            "label": "Global affine alignment cannot express local articulated deformation",
            "artifact": ".runs/research_20260913/continuation_after_review_v5/NEXT_ACTIONS.json",
        },
        "hypothesis:short_offsets_near_duplicate": {
            "kind": "rival_hypothesis",
            "label": "Short offsets provide state-preserving but temporally near-duplicate evidence",
            "artifact": ".runs/research_20260913/continuation_after_review_v5/NEXT_ACTIONS.json",
        },
        "hypothesis:reconstruction_not_task_relevance": {
            "kind": "rival_hypothesis",
            "label": "DINO reconstruction gain may encode appearance rather than action-discriminative evidence",
            "artifact": ".runs/research_20260913/continuation_after_review_v5/NEXT_ACTIONS.json",
        },
        "plan:matched_2x2_causal_factorial": {
            "kind": "experiment",
            "label": "Completed matched 2x2 sampling x aggregation factorial, label-blind and decoder-free",
            "artifact": ".runs/research_20260913/center_completion_matched_factorial_v1/summary.json",
            "rank": "1",
        },
        "lock:matched_2x2_causal_factorial": {
            "kind": "execution_lock",
            "label": "Factorial cells, metrics, and decision rules locked before results",
            "artifact": "experiments/okutama_center_completion_matched_factorial_execution_lock.json",
            "sha256": "355140b1dd169b4057ede15463573f26c6261c1671eb0f8105782d389a2aa439",
        },
        "result:factorial_affine_uniform": {
            "kind": "scientific_result",
            "label": "Affine-uniform C beats same-grid-uniform A by 13.3159%; positive 5/5 folds",
            "artifact": ".runs/research_20260913/center_completion_matched_factorial_v1/summary.json",
            "sha256": "70b8fa965f24058c8617c651f8132d89d416cba1495dd675633792e41c6bb919",
        },
        "failure:factorial_robust_aggregation": {
            "kind": "falsified_component",
            "label": "Confidence-weighted componentwise median hurts: B/A -7.9537%, D/C -6.5831%",
            "artifact": ".runs/research_20260913/center_completion_matched_factorial_v1/summary.json",
        },
        "audit:matched_factorial_replay": {
            "kind": "audit",
            "label": "Matched factorial exact replay: every output artifact hash exact; zero forbidden access",
            "artifact": ".runs/research_20260913/center_completion_matched_factorial_v1_audit/audit_receipt.json",
            "sha256": "55718cc04f902ebec6c8904dce29763ddf45e95fa55f299cc217fcc8e850fba7",
        },
        "audit:factorial_pcar_skeptical_v1": {
            "kind": "skeptical_audit",
            "label": "Factorial passes as adaptive reconstruction evidence; PCAR v5 passes design review only",
            "artifact": ".runs/research_20260913/center_completion_factorial_pcar_skeptical_audit_v1/audit.json",
            "sha256": "165b476258a31c0b85594cc00d834b497fe92c5da4ba71705186f1d7edcbbbde",
        },
        "finding:factorial_robustness": {
            "kind": "statistical_finding",
            "label": "C beats A for 115/128 centers, 5/5 folds and 11/11 scenarios; clustered intervals exclude zero",
            "artifact": ".runs/research_20260913/center_completion_factorial_pcar_skeptical_audit_v1/audit.json",
            "sha256": "165b476258a31c0b85594cc00d834b497fe92c5da4ba71705186f1d7edcbbbde",
        },
        "lineage:PCAR_v1_to_v5": {
            "kind": "protocol_lineage",
            "label": "Immutable PCAR lineage: v1-v4 rejected; leakage-safe planning-only v5 is canonical",
            "artifact": ".runs/research_20260913/center_completion_pcar_lineage_v1/protocol_lineage.json",
            "sha256": "243ab55aceb542a68d8d6ac884770dab432fd281fbede49f5b997a091a79ab8a",
        },
        "failure:PCAR_v1_obsolete_basis": {
            "kind": "rejected_design",
            "label": "PCAR v1 rejected: obsolete P3 median basis, random bottleneck and learned residual direction",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v1/pcar_protocol.json",
            "sha256": "cfdd375778a2222d53a2ccfb2403e3c09c10c518e5417e59f225eb43a2b0afc3",
        },
        "failure:PCAR_v2_incomplete_controls": {
            "kind": "rejected_design",
            "label": "PCAR v2 rejected: endpoints, fixed costs, controls and inner base OOF were incomplete",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v2/pcar_protocol.json",
            "sha256": "14b099977e0afa8ef470e0c486e1918710ed34269a6d97b4b818c77bfd191c43",
        },
        "failure:PCAR_v3_base_stacking_leakage": {
            "kind": "rejected_design",
            "label": "PCAR v3 rejected: policy meta rows used in-sample saved outer-base predictions",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v3/pcar_protocol.json",
            "sha256": "d59da0390fc6430f9d3467a9f64f916f68779e7775b7377ef8f2caaa09439752",
        },
        "failure:PCAR_v4_inner_selection_contamination": {
            "kind": "rejected_design",
            "label": "PCAR v4 rejected: learned-versus-fixed inner selection retained meta-dependence",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v4/pcar_protocol.json",
            "sha256": "d5552d75159cf0d073694ba019884c9fb0828ee635ed23cb6ec2c9eafd8e446d",
        },
        "architecture:PCAR": {
            "kind": "prospective_architecture",
            "label": "PRIMARY PROSPECTIVE: v5 pair-cross-fitted bounded P2-to-affine-uniform fusion; tie retains",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v5/pcar_protocol.json",
            "sha256": "a6df37fe788ee260d4e5f001c9f0730ddb7ee9499cca92bc9a9bf79170034b55",
            "rank": "2",
        },
        "plan:PCAR_v5_runner_and_smoke": {
            "kind": "prospective_experiment",
            "label": "NEXT AUTHORIZED WORK: implement v5 runner and bounded timing smoke; do not fit policy yet",
            "artifact": ".runs/research_20260913/center_completion_pcar_design_v5/pcar_protocol.json",
            "sha256": "a6df37fe788ee260d4e5f001c9f0730ddb7ee9499cca92bc9a9bf79170034b55",
            "rank": "2",
        },
        "plan:local_deformation_pre_gate": {
            "kind": "fallback_experiment",
            "label": "FALLBACK: leave-one-anchor-out local-deformation geometry pre-gate",
            "artifact": ".runs/research_20260913/continuation_after_review_v5/NEXT_ACTIONS.json",
            "rank": "3",
        },
        "decision:no_completion_task_fit": {
            "kind": "decision",
            "label": "Failed reconstruction gate forbids task-label and router fitting",
            "artifact": ".runs/research_20260913/center_evidence_completion_screen_v1_audit/audit_receipt.json",
        },
        "decision:no_PCAR_fit_authorized": {
            "kind": "decision",
            "label": "PCAR v5 is planning-only; runner, tests, execution lock and timing smoke required before fit",
            "artifact": ".runs/research_20260913/center_completion_pcar_lineage_v1/protocol_lineage.json",
            "sha256": "243ab55aceb542a68d8d6ac884770dab432fd281fbede49f5b997a091a79ab8a",
        },
    }
    graph.add_nodes_from(nodes.items())

    def edge(
        source: str,
        target: str,
        relation: str,
        artifact: str,
        effect: str,
        confidence: str = "high",
    ) -> None:
        graph.add_edge(
            source,
            target,
            relation=relation,
            artifact=artifact,
            effect_size=effect,
            confidence=confidence,
            evidence_strength="artifact_or_calculation",
        )

    edge(
        "protocol:center_completion_schema8_authorized",
        "cache:center_completion_full_v1",
        "authorizes",
        nodes["cache:center_completion_full_v1"]["artifact"],
        "128_centers;1392_inputs",
    )
    edge(
        "cache:center_completion_full_v1",
        "audit:center_completion_full_cache_replay",
        "independently-reproduced-by",
        nodes["audit:center_completion_full_cache_replay"]["artifact"],
        "9_of_9_arrays_byte_exact",
    )
    edge(
        "correction:cache_view_access_accounting",
        "cache:center_completion_full_v1",
        "corrects-label-only",
        nodes["correction:cache_view_access_accounting"]["artifact"],
        "1136_views;640_decodes;600_unique",
    )
    edge(
        "audit:center_completion_full_cache_replay",
        "cache:center_completion_transport_v1",
        "authorizes-label-blind-materialization",
        nodes["cache:center_completion_transport_v1"]["artifact"],
        "27407_targets",
    )
    edge(
        "cache:center_completion_transport_v1",
        "audit:center_completion_transport_replay",
        "independently-reproduced-by",
        nodes["audit:center_completion_transport_replay"]["artifact"],
        "9_of_9_arrays_byte_exact",
    )
    edge(
        "failure:transport_live_source_changed",
        "audit:center_completion_transport_replay",
        "excluded-and-corrected-by-immutable-source",
        nodes["audit:center_completion_transport_replay"]["artifact"],
        "failed_attempt_not_evidence",
    )
    edge(
        "audit:center_completion_transport_replay",
        "lock:center_completion_screen_v1",
        "supports-execution-of",
        nodes["lock:center_completion_screen_v1"]["artifact"],
        "all_input_pins_match",
    )
    edge(
        "lock:center_completion_screen_v1",
        "experiment:center_completion_screen_v1",
        "governs",
        nodes["lock:center_completion_screen_v1"]["artifact"],
        "5_folds;20_fits;seed42",
    )
    for result in ("result:completion_P0", "result:completion_P1", "result:completion_P2", "result:completion_P3"):
        edge(
            "experiment:center_completion_screen_v1",
            result,
            "produces",
            nodes[result]["artifact"],
            nodes[result]["value"],
        )
    edge(
        "result:completion_P3",
        "result:completion_P2",
        "improves",
        nodes["forensics:center_completion_screen_v1"]["artifact"],
        "3.819792_percent_relative;111_of_128_centers",
    )
    edge(
        "failure:completion_P3_vs_P2_gate",
        "experiment:center_completion_screen_v1",
        "fails-preregistered-gate-of",
        nodes["failure:completion_P3_vs_P2_gate"]["artifact"],
        "3.819792_percent_below_5_percent",
    )
    edge(
        "audit:center_completion_screen_replay",
        "failure:completion_P3_vs_P2_gate",
        "independently-confirms",
        nodes["audit:center_completion_screen_replay"]["artifact"],
        "only_gate_failed;40_of_40_arrays_exact",
    )
    edge(
        "forensics:center_completion_screen_v1",
        "failure:completion_P3_vs_P2_gate",
        "quantifies",
        nodes["forensics:center_completion_screen_v1"]["artifact"],
        "adaptive_bootstrap_CI_3.035_to_4.661_percent",
    )
    edge(
        "diagnostic:center_selector_oracle",
        "failure:completion_P3_vs_P2_gate",
        "cannot-rescue",
        nodes["diagnostic:center_selector_oracle"]["artifact"],
        "4.032611_percent_below_5_percent",
    )
    edge(
        "diagnostic:token_selector_oracle",
        "architecture:PCAR",
        "obsolete-secondary-motivation-for",
        nodes["diagnostic:token_selector_oracle"]["artifact"],
        "6.0594_percent_oracle_vs_required_5_percent",
        "diagnostic_only",
    )
    edge(
        "finding:P3_token_gate_weak",
        "architecture:PCAR",
        "motivates",
        nodes["finding:P3_token_gate_weak"]["artifact"],
        "legacy_AUC_0.516927;legacy_oracle_secondary",
    )
    edge(
        "hypothesis:factorial_architecture_confound",
        "lock:matched_2x2_causal_factorial",
        "suggests-next-test",
        nodes["lock:matched_2x2_causal_factorial"]["artifact"],
        "separate_evidence_from_correction",
        "locked_pre_result",
    )
    edge(
        "lock:matched_2x2_causal_factorial",
        "plan:matched_2x2_causal_factorial",
        "governs",
        nodes["lock:matched_2x2_causal_factorial"]["artifact"],
        "four_decoder_free_cells;zero_fits",
    )
    edge(
        "plan:matched_2x2_causal_factorial",
        "result:factorial_affine_uniform",
        "produces",
        nodes["result:factorial_affine_uniform"]["artifact"],
        "C_vs_A_13.3159_percent;5_of_5_folds",
    )
    edge(
        "plan:matched_2x2_causal_factorial",
        "failure:factorial_robust_aggregation",
        "falsifies-component",
        nodes["failure:factorial_robust_aggregation"]["artifact"],
        "B_vs_A_-7.9537;D_vs_C_-6.5831_percent",
    )
    edge(
        "audit:matched_factorial_replay",
        "result:factorial_affine_uniform",
        "independently-confirms",
        nodes["audit:matched_factorial_replay"]["artifact"],
        "all_artifacts_exact",
    )
    edge(
        "audit:factorial_pcar_skeptical_v1",
        "result:factorial_affine_uniform",
        "independently-stress-tests",
        nodes["audit:factorial_pcar_skeptical_v1"]["artifact"],
        "PASS_adaptive_reconstruction_only",
    )
    edge(
        "finding:factorial_robustness",
        "result:factorial_affine_uniform",
        "supports",
        nodes["finding:factorial_robustness"]["artifact"],
        "115_of_128_centers;5_of_5_folds;11_of_11_scenarios",
    )
    edge(
        "result:factorial_affine_uniform",
        "diagnostic:affine_uniform_convex_oracle",
        "suggests-next-test",
        nodes["result:factorial_affine_uniform"]["artifact"],
        "recompute_headroom_for_surviving_C",
    )
    edge(
        "diagnostic:affine_uniform_convex_oracle",
        "architecture:PCAR",
        "supports-precondition-for",
        nodes["diagnostic:affine_uniform_convex_oracle"]["artifact"],
        "3.6722_percent_vs_C;positive_5_of_5;required_2_percent",
        "diagnostic_only",
    )
    edge(
        "failure:factorial_robust_aggregation",
        "architecture:PCAR",
        "constrains-design-of",
        nodes["failure:factorial_robust_aggregation"]["artifact"],
        "uniform_full768_only;no_weighted_median",
    )
    for rejected in (
        "failure:PCAR_v1_obsolete_basis",
        "failure:PCAR_v2_incomplete_controls",
        "failure:PCAR_v3_base_stacking_leakage",
        "failure:PCAR_v4_inner_selection_contamination",
    ):
        edge(
            rejected,
            "lineage:PCAR_v1_to_v5",
            "recorded-as-superseded-by",
            nodes["lineage:PCAR_v1_to_v5"]["artifact"],
            "do_not_implement",
            "skeptically_confirmed",
        )
    edge(
        "lineage:PCAR_v1_to_v5",
        "architecture:PCAR",
        "selects-canonical-design",
        nodes["lineage:PCAR_v1_to_v5"]["artifact"],
        "v5_only;pair_crossfit;no_inner_selection",
        "design_only",
    )
    edge(
        "audit:factorial_pcar_skeptical_v1",
        "architecture:PCAR",
        "passes-design-review-for",
        nodes["audit:factorial_pcar_skeptical_v1"]["artifact"],
        "implementation_and_timing_smoke_only",
        "planning_only",
    )
    edge(
        "architecture:PCAR",
        "plan:PCAR_v5_runner_and_smoke",
        "suggests-next-test",
        nodes["plan:PCAR_v5_runner_and_smoke"]["artifact"],
        "30_fits_projected_10_to_13_minutes;not_authorized",
        "prospective",
    )
    edge(
        "decision:no_PCAR_fit_authorized",
        "plan:PCAR_v5_runner_and_smoke",
        "limits-to",
        nodes["decision:no_PCAR_fit_authorized"]["artifact"],
        "runner_tests_lock_smoke_only",
        "planning_only",
    )
    edge(
        "hypothesis:global_affine_too_rigid",
        "plan:local_deformation_pre_gate",
        "suggests-next-test",
        nodes["plan:local_deformation_pre_gate"]["artifact"],
        "LOAO_geometry_before_reconstruction",
        "prospective",
    )
    edge(
        "architecture:PCAR",
        "plan:local_deformation_pre_gate",
        "falls-back-to-on-failure",
        nodes["plan:local_deformation_pre_gate"]["artifact"],
        "LOAO_geometry_gate_before_reconstruction",
        "prospective",
    )
    edge(
        "hypothesis:short_offsets_near_duplicate",
        "failure:completion_P3_vs_P2_gate",
        "rival-explanation-for",
        nodes["hypothesis:short_offsets_near_duplicate"]["artifact"],
        "time_reversal_delta_0.000228",
        "hypothesis",
    )
    edge(
        "hypothesis:reconstruction_not_task_relevance",
        "decision:no_completion_task_fit",
        "reinforces",
        nodes["hypothesis:reconstruction_not_task_relevance"]["artifact"],
        "reconstruction_gate_failed;task_relevance_unknown",
        "hypothesis",
    )
    edge(
        "failure:completion_P3_vs_P2_gate",
        "decision:no_completion_task_fit",
        "enforces",
        nodes["decision:no_completion_task_fit"]["artifact"],
        "task_training_unauthorized",
    )
    edge(
        "decision:no_completion_task_fit",
        "action:retain_arftr",
        "preserves",
        nodes["decision:no_completion_task_fit"]["artifact"],
        "85.383648_macro_f1;702_errors",
    )
    edge(
        "decision:reviewer_b_deferred",
        "plan:matched_2x2_causal_factorial",
        "not-required-by",
        nodes["plan:matched_2x2_causal_factorial"]["artifact"],
        "reviewer_B_off_critical_path",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    graphml = OUTPUT / "knowledge_graph.graphml"
    graph_json = OUTPUT / "knowledge_graph.json"
    receipt_path = OUTPUT / "graph_receipt.json"
    existing = [path for path in (graphml, graph_json, receipt_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite graph artifacts: {existing}")
    graph_artifacts = {
        node["artifact"]
        for node in nodes.values()
        if not node["artifact"].startswith(".runs/research_20260913/continuation_after_review_v5/")
    }
    missing_artifacts = sorted(
        artifact for artifact in graph_artifacts if not (ROOT / artifact).is_file()
    )
    if missing_artifacts:
        raise FileNotFoundError(f"Missing graph evidence artifacts: {missing_artifacts}")
    bad_hashes = []
    for node in nodes.values():
        artifact = node["artifact"]
        expected = node.get("sha256")
        if expected and not artifact.startswith(
            ".runs/research_20260913/continuation_after_review_v5/"
        ):
            actual = sha256_file(ROOT / artifact)
            if actual != expected:
                bad_hashes.append((artifact, expected, actual))
    if bad_hashes:
        raise ValueError(f"Graph evidence hash mismatches: {bad_hashes}")
    nx.write_graphml(graph, graphml)
    with graph_json.open("x", encoding="utf-8") as stream:
        json.dump(nx.node_link_data(graph, edges="edges"), stream, indent=2, sort_keys=True)
        stream.write("\n")
    receipt = {
        "status": "POST_COMPLETION_FACTORIAL_KNOWLEDGE_GRAPH_COMPLETE",
        "base_graph_sha256": sha256_file(BASE),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "graphml_sha256": sha256_file(graphml),
        "json_sha256": sha256_file(graph_json),
        "source_sha256": sha256_file(Path(__file__)),
        "referenced_external_artifacts_checked": len(graph_artifacts),
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
