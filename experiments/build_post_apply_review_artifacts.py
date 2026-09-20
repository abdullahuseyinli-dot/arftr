"""Build the dated post-apply ledger and knowledge graph.

This is an evidence packaging script.  It never fits a model and never edits
any prior run.  The parent graph/ledger are copied logically (the parent graph
is loaded and extended; the full ledger is referenced by hash) so that the
new report makes the two 2026-09-17 probes visible without rewriting history.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runs/research_20260917/post_apply_review_v1"
GRAPH_DIR = OUT / "graph"
PARENT_GRAPH = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/knowledge_graph_final.json"
FULL_LEDGER = ROOT / ".runs/research_20260912/invention_audit_20260912_220926/lineage/normalized_experiment_ledger.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def node(node_id: str, kind: str, label: str, artifact: str, **extra):
    value = {"id": node_id, "kind": kind, "label": label, "artifact": artifact}
    value.update(extra)
    return value


def edge(source: str, target: str, relation: str, artifact: str, **extra):
    value = {
        "source": source,
        "target": target,
        "relation": relation,
        "artifact": artifact,
        "evidence_strength": extra.pop("evidence_strength", "artifact_or_calculation"),
        "confidence": extra.pop("confidence", "conditional_internal_evidence"),
    }
    value.update(extra)
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    parent = json.loads(PARENT_GRAPH.read_text(encoding="utf-8"))
    graph = {
        "graph_scope": "parent_521_node_graph_plus_20260917_fdcrg_and_ptg_review",
        "parent_graph": str(PARENT_GRAPH.relative_to(ROOT)).replace("\\", "/"),
        "nodes": list(parent["nodes"]),
        "edges": list(parent["edges"]),
    }
    node_ids = {n["id"] for n in graph["nodes"]}
    for n in [
        node("archive:continuation_20260906", "archive", "Original continuation archive (manifest inspected; 48 zip entries)", "C:/Users/DELL/Downloads/HAC_Research_Continuation_20260906.zip", evidence="manifest and guide entries inspected; payload not rewritten"),
        node("cache:dense_tokens_grid12", "feature_cache", "Verified DINO dense grid12 tokens (4977 x 8 x 144 x 768, float16)", ".runs/research_20260908/dense_tokens_full/verification.json", shape="4977,8,144,768", sha256="f97d582ddfe0fcc42a7476aa1af60a6317d427bccf3ffcefb27dbb19e9e85f97"),
        node("architecture:fine_local_motion", "architecture", "FineLocalMotion pre-pooling spatial/temporal moment reader", "src/hac/fine_local_motion.py", status="implemented_and_unit_tested"),
        node("experiment:fine_detail_probe_v2/coarse_control", "experiment", "Fine-detail v2 repeated-coarse control", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", macro_f1=0.7872250675086286, accuracy=0.7984729756881656, nll=0.5142965370152435, brier=0.09777447936429583, errors=1003, seeds=5, folds=5, result_class="observed_internal_probe"),
        node("experiment:fine_detail_probe_v2/fine_both", "experiment", "Fine-detail v2 genuine 12x12 spatial+motion reader", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", macro_f1=0.7879714088513957, accuracy=0.7996785211975085, nll=0.49609758627211353, brier=0.09695063232624615, errors=997, seeds=5, folds=5, result_class="observed_internal_probe"),
        node("metric:fine_detail_probe_v2/standalone_delta", "metric", "Fine minus repeated-coarse: +0.0746 macro-F1 points", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", effect_size=0.0007463413427670767, effect_units="macro_f1_delta", positive_outer_folds=3, outer_folds=5, positive_scenarios=4, scenarios=11),
        node("metric:fine_detail_probe_v2/proper_loss_delta", "metric", "Fine improves NLL and Brier versus control", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", nll_delta=-0.01819895074312946, brier_delta=-0.00082384603804968),
        node("result:fine_detail_probe_v2/arftr_blend", "result", "Fixed 50/50 ARFTR + fine blend", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", macro_f1=0.8337587278334784, accuracy=0.8414707655213984, nll=0.39852540804757647, brier=0.07682521525096093, errors=789, rescues=199, harms=494, net=-295, result_class="diagnostic_fixed_blend_not_deployable"),
        node("audit:fine_detail_probe_v2/replay", "audit", "Fine-detail v2 independent replay audit PASS", ".runs/research_20260917/fine_detail_probe_v2_retry/independent_audit.json", max_probability_delta=0.0, fits=0, labels_read=0, optimizer_updates=0),
        node("failure:fine_detail_probe_v2/nonpromotion", "failure_mechanism", "Fine detail is complementary in proper loss but unsafe as global intervention", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", reason="199 rescues versus 494 harms against ARFTR; fixed blend regresses"),
        node("experiment:pcar_v5_screen_v2", "experiment", "PCAR v5 persistent correspondence screen", ".runs/research_20260917/pcar_v5_screen_v2/summary.json", relative_reduction_vs_C=0.0001160361960563488, rescues=61, harms=80, net=-19, outer_net=[-5,-6,29,-6,-31], result_class="scientific_screen_not_promoted"),
        node("failure:pcar_v5/nonpromotion", "failure_mechanism", "PCAR reconstruction gain did not translate to stable classification utility", ".runs/research_20260917/pcar_v5_screen_v2/summary.json"),
        node("failure:sequence_family/closed", "failure_mechanism", "Standalone timestamp/actor-relative sequence family closed", "docs/HAC_ACTOR_RELATIVE_SEQUENCE_RESULTS_20260912.md", result="68.48/71.78/69.30/64.16/64.22 macro-F1"),
        node("failure:router_family/closed", "failure_mechanism", "Source/posture transition-risk router not robust", ".runs/research_20260916/source_posture_failure_router_v1/REPORT.md", result="+0.0796 pp; fold nets [-1,-2,0,5,0]"),
        node("experiment:gipa_persistent_articulation_v1_dense_knownbg", "experiment", "GIPA Phase-A dense persistent-articulation smoke", ".runs/research_20260916/gipa_persistent_articulation_v1_dense_knownbg/FAILURE_ANALYSIS.json", result_class="measurement_only_failed_gates", labels_read=0, task_fit=False),
        node("audit:gipa_persistent_articulation/replay", "audit", "GIPA replay audit PASS", ".runs/research_20260916/gipa_persistent_articulation_v1_dense_knownbg/independent_audit.json", labels_read=0),
        node("failure:gipa_persistent_articulation/smoke", "failure_mechanism", "Persistent LK tracking failed survival, actor-residual retention, held reduction, and wrong-track specificity", ".runs/research_20260916/gipa_persistent_articulation_v1_dense_knownbg/FAILURE_ANALYSIS.json", downsample_survival=0.75, native_survival=0.5625, downsample_actor_retention=0.3449772188815837, native_actor_retention=0.31911094600210055),
        node("experiment:source_causal_body_parts_v1/phase0", "experiment", "Source-Causal Body Parts parity preflight", ".runs/research_20260916/source_causal_body_parts_v1/phase0_preflight/preflight.json", result_class="label_blind_preflight_pass", rows=4977, classifier_fits=0),
        node("failure:source_causal_body_parts_v1/duplicate", "failure_mechanism", "Planned body-parts Phase 1 was a duplicate of evaluated source/posture evidence", ".runs/research_20260916/source_causal_body_parts_v1/DUPLICATE_CLOSURE.md", result="phase1 not launched"),
        node("error:arftr_residual_702", "error_group", "ARFTR residual: 702 errors; 457 standing/locomotion, 224 sitting/standing, 21 sitting/locomotion", ".runs/research_20260912/arftr_v1/results/v0001/summary.json", errors=702, dominant_pairs={"standing_locomotion":457,"sitting_standing":224,"sitting_locomotion":21}),
        node("hypothesis:fine_complementary_signal", "hypothesis", "Fine density contains weak complementary/calibration signal but not a safe global replacement", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", status="supported_for_loss_only; unproven_for_selective_utility"),
        node("invention:persistent_token_gauge", "invention_root", "Persistent Token Gauge (PTG): high-resolution token identity paths plus gauge-invariant residuals", "experiments/okutama_persistent_token_gauge_protocol.json", status="tested_closed_after_discovery_failure"),
        node("protocol:persistent_token_gauge_v1", "protocol", "Label-blind PTG preflight then one-seed four-arm probe", "experiments/okutama_persistent_token_gauge_protocol.json", outer_folds=5, max_task_fits=20, inference_inputs="dense tokens, timestamps, ARFTR context; no annotation support"),
        node("experiment:ptg_preflight_v1", "experiment", "PTG 16-center label-blind local-token smoke", ".runs/research_20260917/ptg_preflight_v1/preflight.json", result_class="measurement_only_pass", all_smoke_gates_pass=True, labels_read=0, classifier_fits=0),
        node("audit:ptg_preflight_v1/replay", "audit", "PTG preflight independent replay PASS", ".runs/research_20260917/ptg_preflight_v1/independent_audit.json", labels_read=0, classifier_fits=0),
        node("experiment:ptg_task_probe_v1", "experiment", "PTG one-seed four-arm task discovery", ".runs/research_20260917/ptg_task_probe_v1/summary.json", result_class="scientific_probe_failed_discovery_gate", macro_f1_g0=0.7507371448514489, macro_f1_g2=0.7301487200103115, macro_f1_g3=0.7287411124583864, fits=20),
        node("audit:ptg_task_probe_v1/replay", "audit", "PTG task probe independent replay PASS", ".runs/research_20260917/ptg_task_probe_v1/independent_audit.json", max_probability_delta=0.0, outer_labels_read=0),
        node("failure:ptg_task_probe_v1/discovery", "failure_mechanism", "PTG token paths have proper-loss signal but no stable macro-F1 advantage", ".runs/research_20260917/ptg_task_probe_v1/summary.json", g3_vs_g0=-0.02199603239306258, g3_vs_g2_outer=[0.0012952094945034576,-0.0008545494411447141,0.002212904211082223,-0.007663013427717802,-0.003274882213356145]),
        node("invention:fine_density_residual_gate", "invention_root", "Fine-Density Counterfactual Residual Gate (FDCRG)", "experiments/okutama_fdcrg_protocol.json", status="executed_failed_promotion_gates"),
        node("architecture:fine_density_residual_gate", "architecture", "Bounded fine-minus-coarse factor residual with utility gate", "src/hac/fine_density_residual_gate.py", actions=["retain_ARFTR","posture","motion","both"], residual_cap=0.25),
        node("experiment:fdcrg_v1_full", "experiment", "FDCRG one-shot nested residual probe", ".runs/research_20260917/fdcrg_v1_full/summary.json", macro_f1=0.854276299199945, accuracy=0.8593530239099859, nll=0.3774763184527237, brier=0.07118450612414413, errors=700, rescues=2, harms=0, net=2, interventions=135, runtime_seconds=1299.8386746000033, result_class="scientific_probe_failed_promotion"),
        node("audit:fdcrg_v1/replay", "audit", "FDCRG independent replay audit PASS", ".runs/research_20260917/fdcrg_v1_full/independent_audit.json", max_probability_delta=0.0, fits=0, optimizer_updates=0, outer_labels_read=0),
        node("failure:fdcrg_v1/nonpromotion", "failure_mechanism", "FDCRG produced only +0.04398 macro-F1 points and worsened NLL/Brier", ".runs/research_20260917/fdcrg_v1_full/summary.json", net_corrections=2, positive_net_outer_folds=1, scenario_bootstrap_lower=0.0),
        node("invention:question_directed_observation", "invention_root", "Question-Directed Observation (QDO), deferred fallback", "docs/HAC_NEXT_ARCHITECTURE_MAP_20260912.md", status="deferred_until_new_evidence_bank"),
        node("action:retain_arftr", "action", "Explicit retain_ARFTR default action", "docs/HAC_NEXT_ARCHITECTURE_MAP_20260912.md", invariant=True),
    ]:
        if n["id"] not in node_ids:
            graph["nodes"].append(n)
            node_ids.add(n["id"])

    edge_specs = [
        edge("archive:continuation_20260906", "data:okutama", "documents", "C:/Users/DELL/Downloads/HAC_Research_Continuation_20260906.zip", evidence_strength="manifest_inspection"),
        edge("data:okutama", "cache:dense_tokens_grid12", "derived-from", ".runs/research_20260908/dense_tokens_full/verification.json", effect_size=4977, effect_units="aligned_rows", evidence_strength="independent_replay"),
        edge("cache:dense_tokens_grid12", "architecture:fine_local_motion", "input-to", "src/hac/fine_local_motion.py", evidence_strength="implementation_and_hash"),
        edge("architecture:fine_local_motion", "experiment:fine_detail_probe_v2/coarse_control", "evaluated-as", "experiments/okutama_fine_detail_probe_v2_protocol.json", evidence_strength="locked_protocol"),
        edge("architecture:fine_local_motion", "experiment:fine_detail_probe_v2/fine_both", "evaluated-as", "experiments/okutama_fine_detail_probe_v2_protocol.json", evidence_strength="locked_protocol"),
        edge("experiment:fine_detail_probe_v2/fine_both", "metric:fine_detail_probe_v2/standalone_delta", "improves", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", effect_size=0.0007463413427670767, effect_units="macro_f1_delta", evidence_strength="recalculated_from_saved_predictions"),
        edge("experiment:fine_detail_probe_v2/fine_both", "metric:fine_detail_probe_v2/proper_loss_delta", "improves", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", effect_size={"nll":-0.01819895074312946,"brier":-0.00082384603804968}, evidence_strength="recalculated_from_saved_predictions"),
        edge("experiment:fine_detail_probe_v2/fine_both", "result:fine_detail_probe_v2/arftr_blend", "evaluated-with", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", evidence_strength="fixed_post_fit_diagnostic"),
        edge("result:fine_detail_probe_v2/arftr_blend", "failure:fine_detail_probe_v2/nonpromotion", "harms", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", effect_size=-295, effect_units="net_corrections", evidence_strength="saved_OOF_transition_count"),
        edge("audit:fine_detail_probe_v2/replay", "experiment:fine_detail_probe_v2/fine_both", "audits", ".runs/research_20260917/fine_detail_probe_v2_retry/independent_audit.json", evidence_strength="independent_exact_replay"),
        edge("failure:fine_detail_probe_v2/nonpromotion", "hypothesis:fine_complementary_signal", "supports-hypothesis", ".runs/research_20260917/fine_detail_probe_v2_retry/summary.json", confidence="moderate"),
        edge("failure:fine_detail_probe_v2/nonpromotion", "invention:fine_density_residual_gate", "suggests-next-test", "experiments/okutama_fdcrg_protocol.json", evidence_strength="mechanism_targeted"),
        edge("error:arftr_residual_702", "invention:persistent_token_gauge", "suggests-next-test", "experiments/okutama_persistent_token_gauge_protocol.json", effect_size=457, effect_units="dominant_swap_count", evidence_strength="error_forensics"),
        edge("failure:sequence_family/closed", "invention:persistent_token_gauge", "contradicts-broad-replacement", "docs/HAC_ACTOR_RELATIVE_SEQUENCE_RESULTS_20260912.md", evidence_strength="repeated_outer_screen"),
        edge("failure:pcar_v5/nonpromotion", "invention:persistent_token_gauge", "suggests-different-representation", ".runs/research_20260917/pcar_v5_screen_v2/summary.json", evidence_strength="failure_analysis"),
        edge("failure:router_family/closed", "invention:fine_density_residual_gate", "constrains", ".runs/research_20260916/source_posture_failure_router_v1/REPORT.md", relation_detail="gate must not repeat broad score routing", evidence_strength="negative_evidence"),
        edge("experiment:gipa_persistent_articulation_v1_dense_knownbg", "failure:gipa_persistent_articulation/smoke", "fails", ".runs/research_20260916/gipa_persistent_articulation_v1_dense_knownbg/FAILURE_ANALYSIS.json", effect_size="survival/retention/held-reduction/wrong-track gates", evidence_strength="direct_measurement"),
        edge("audit:gipa_persistent_articulation/replay", "experiment:gipa_persistent_articulation_v1_dense_knownbg", "audits", ".runs/research_20260916/gipa_persistent_articulation_v1_dense_knownbg/independent_audit.json", evidence_strength="independent_exact_replay"),
        edge("failure:gipa_persistent_articulation/smoke", "invention:persistent_token_gauge", "suggests-different-identity-carrier", "experiments/okutama_persistent_token_gauge_protocol.json", relation_detail="avoid raw LK tracker and test ViT-token persistence with its own gates", evidence_strength="failure_directed_invention"),
        edge("experiment:source_causal_body_parts_v1/phase0", "failure:source_causal_body_parts_v1/duplicate", "precedes-closure", ".runs/research_20260916/source_causal_body_parts_v1/DUPLICATE_CLOSURE.md", evidence_strength="parity_pass_then_duplicate_audit"),
        edge("failure:source_causal_body_parts_v1/duplicate", "invention:persistent_token_gauge", "requires-new-signal", ".runs/research_20260916/source_causal_body_parts_v1/DUPLICATE_CLOSURE.md", evidence_strength="negative_evidence"),
        edge("invention:persistent_token_gauge", "protocol:persistent_token_gauge_v1", "realized-by", "experiments/okutama_persistent_token_gauge_protocol.json", evidence_strength="prospective_falsifiable_design"),
        edge("protocol:persistent_token_gauge_v1", "experiment:ptg_preflight_v1", "evaluated-by", ".runs/research_20260917/ptg_preflight_v1/preflight.json", evidence_strength="label_blind_measurement"),
        edge("experiment:ptg_preflight_v1", "audit:ptg_preflight_v1/replay", "audited-by", ".runs/research_20260917/ptg_preflight_v1/independent_audit.json", evidence_strength="independent_exact_replay"),
        edge("experiment:ptg_preflight_v1", "protocol:persistent_token_gauge_v1", "passes-phase-a", ".runs/research_20260917/ptg_preflight_v1/preflight.json", effect_size="all five smoke gates", evidence_strength="direct_measurement"),
        edge("experiment:ptg_preflight_v1", "protocol:persistent_token_gauge_v1", "suggests-next-test", "experiments/okutama_persistent_token_gauge_protocol.json", relation_detail="authorize one-seed four-arm task discovery only", evidence_strength="prospective_gate"),
        edge("experiment:ptg_task_probe_v1", "failure:ptg_task_probe_v1/discovery", "fails", ".runs/research_20260917/ptg_task_probe_v1/summary.json", effect_size=-0.02199603239306258, effect_units="G3-G0 macro_f1_delta", evidence_strength="saved_OOF_metrics"),
        edge("audit:ptg_task_probe_v1/replay", "experiment:ptg_task_probe_v1", "audits", ".runs/research_20260917/ptg_task_probe_v1/independent_audit.json", evidence_strength="independent_exact_replay"),
        edge("failure:ptg_task_probe_v1/discovery", "invention:fine_density_residual_gate", "suggests-next-test", "experiments/okutama_fdcrg_protocol.json", relation_detail="use fine signal only as bounded nested residual; no PTG replacement", evidence_strength="failure_directed_invention"),
        edge("invention:fine_density_residual_gate", "architecture:fine_density_residual_gate", "realized-by", "src/hac/fine_density_residual_gate.py", evidence_strength="implemented_protocol"),
        edge("architecture:fine_density_residual_gate", "experiment:fdcrg_v1_full", "evaluated-as", "experiments/okutama_fdcrg_protocol.json", evidence_strength="locked_nested_protocol"),
        edge("experiment:fdcrg_v1_full", "audit:fdcrg_v1/replay", "audited-by", ".runs/research_20260917/fdcrg_v1_full/independent_audit.json", evidence_strength="independent_exact_replay"),
        edge("experiment:fdcrg_v1_full", "failure:fdcrg_v1/nonpromotion", "fails", ".runs/research_20260917/fdcrg_v1_full/summary.json", effect_size=0.000439818462582688, effect_units="macro_f1_delta", evidence_strength="saved_outer_metrics"),
        edge("failure:fdcrg_v1/nonpromotion", "invention:question_directed_observation", "suggests-next-test", "docs/HAC_NEXT_ARCHITECTURE_MAP_20260912.md", relation_detail="cached density correction branch closed; require a new observation bank", evidence_strength="promotion_gate_failure"),
        edge("invention:fine_density_residual_gate", "action:retain_arftr", "includes-action", "experiments/okutama_fdcrg_protocol.json", evidence_strength="locked_safety_boundary"),
        edge("invention:persistent_token_gauge", "action:retain_arftr", "preserves", "experiments/okutama_persistent_token_gauge_protocol.json", evidence_strength="locked_safety_boundary"),
        edge("invention:question_directed_observation", "action:retain_arftr", "preserves", "docs/HAC_NEXT_ARCHITECTURE_MAP_20260912.md", evidence_strength="prospective_requirement"),
    ]
    existing = {(e.get("source"), e.get("target"), e.get("relation")) for e in graph["edges"]}
    next_key = max((int(e.get("key", -1)) for e in graph["edges"] if str(e.get("key", "-1")).lstrip("-").isdigit()), default=-1) + 1
    for e in edge_specs:
        signature = (e["source"], e["target"], e["relation"])
        if signature not in existing:
            e["key"] = next_key
            next_key += 1
            graph["edges"].append(e)
            existing.add(signature)

    graph["node_count"] = len(graph["nodes"])
    graph["edge_count"] = len(graph["edges"])
    graph["generated_from"] = {
        "parent_graph_sha256": sha256(PARENT_GRAPH),
        "full_ledger_sha256": sha256(FULL_LEDGER),
        "generator": "experiments/build_post_apply_review_artifacts.py",
    }
    (OUT / "knowledge_graph.json").write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")

    # Preserve a compact, auditable ledger while pointing to the complete 168-row ledger.
    full = json.loads(FULL_LEDGER.read_text(encoding="utf-8"))
    keep_terms = ("t2_fixed", "vjepa21_real_clip__linear", "video_dino_mean", "dual_scale_factorized", "spatial_contrast", "ocvc_uniform_diverse", "temporal_mean", "temporal_conv", "survival_memory", "new_source_m4", "arftr_v1::r5", "bounded_factor_correction_v1::b", "actor_relative_sequence_v2::s", "matr_cached_gate_v3::r", "sear_matrix_v1::a", "native4k_source_effect")
    selected = [r for r in full if any(term in r.get("experiment_id", "") for term in keep_terms)]
    selected.extend([
        {"experiment_id":"fine_detail_probe_v2::coarse_control", "parent_baseline":"fine_detail_probe_v2::fine_both", "data":"Okutama 4,977 centers; 11 scenarios; 5 grouped outer folds", "features":"repeated 3x3 tokens expanded to 12x12; shared coarse reference", "architecture":"FineLocalMotion rank32 width128 2-layer temporal reader", "loss":"class-weighted cross entropy", "training_settings":"2 epochs; batch128; AdamW 3e-4; seed per fold", "seed_count":5, "metric":"pooled OOF macro-F1", "macro_f1":0.7872250675086286, "accuracy":0.7984729756881656, "nll":0.5142965370152435, "brier":0.09777447936429583, "errors":1003, "uncertainty":"scenario bootstrap delta recorded in post-apply report", "runtime_seconds":59.0, "audit_status":"independent replay PASS; zero labels during audit", "outcome":"control; not promoted", "result_class":"observed_internal_probe", "artifact":".runs/research_20260917/fine_detail_probe_v2_retry/summary.json"},
        {"experiment_id":"fine_detail_probe_v2::fine_both", "parent_baseline":"fine_detail_probe_v2::coarse_control", "data":"Okutama 4,977 centers; 11 scenarios; 5 grouped outer folds", "features":"verified 12x12 dense tokens; pre-pooling spatial and temporal residual moments", "architecture":"FineLocalMotion rank32 width128 2-layer temporal reader", "loss":"class-weighted cross entropy", "training_settings":"2 epochs; batch128; AdamW 3e-4; seed per fold", "seed_count":5, "metric":"pooled OOF macro-F1", "macro_f1":0.7879714088513957, "accuracy":0.7996785211975085, "nll":0.49609758627211353, "brier":0.09695063232624615, "errors":997, "uncertainty":"fine-minus-coarse scenario bootstrap 95% [-0.02892,+0.03947] F1 points (unweighted scenario delta)", "runtime_seconds":198.9, "audit_status":"independent replay PASS; zero labels/human fields during fit", "outcome":"not promoted; informative complementary loss signal", "result_class":"observed_internal_probe", "artifact":".runs/research_20260917/fine_detail_probe_v2_retry/summary.json"},
        {"experiment_id":"fine_detail_probe_v2::fixed_half_arftr_blend", "parent_baseline":"arftr_v1::r5_arftr_full", "data":"same 4,977-center OOF rows; post-fit diagnostic only", "features":"ARFTR probabilities + fine_both probabilities", "architecture":"fixed 50/50 probability blend", "loss":"none", "training_settings":"no fit; coefficient fixed before post-fit scoring", "seed_count":0, "metric":"pooled OOF macro-F1", "macro_f1":0.8337587278334784, "accuracy":0.8414707655213984, "nll":0.39852540804757647, "brier":0.07682521525096093, "errors":789, "uncertainty":"199 rescues; 494 harms; net -295", "runtime_seconds":0.0, "audit_status":"descriptive only; not a deployable router", "outcome":"failed safety/promotion gate", "result_class":"fixed_posthoc_diagnostic", "artifact":".runs/research_20260917/fine_detail_probe_v2_retry/summary.json"},
        {"experiment_id":"pcar_v5_screen_v2::capacity_policy", "parent_baseline":"affine center reconstruction", "data":"128 deterministic centers; 5 scenario folds", "features":"persistent correspondence proxy and capacity policy", "architecture":"PCAR v5", "loss":"unsupervised reconstruction proxy", "training_settings":"200 policy updates per outer fold", "seed_count":1, "metric":"mean center reconstruction error", "macro_f1":None, "accuracy":None, "nll":None, "brier":None, "errors":None, "uncertainty":"relative reduction vs C 0.0116%; net corrections [-5,-6,+29,-6,-31]", "runtime_seconds":478.64, "audit_status":"alignment replay PASS; no task promotion", "outcome":"closed; reconstruction is not task utility", "result_class":"diagnostic_non_task", "artifact":".runs/research_20260917/pcar_v5_screen_v2/summary.json"},
        {"experiment_id":"ptg_task_probe_v1::gauge_parity", "parent_baseline":"ptg_task_probe_v1::g0_context_control", "data":"Okutama 4,977 centers; 11 scenarios; 5 grouped outer folds", "features":"persistent token paths, gauge/parity/cycle channels", "architecture":"Persistent Token Gauge four-arm discovery", "loss":"class-weighted cross entropy", "training_settings":"one seed; 2 epochs; 20 fits", "seed_count":1, "metric":"pooled OOF macro-F1", "macro_f1":0.7287411124583864, "accuracy":None, "nll":0.60135634, "brier":0.11584073, "errors":None, "uncertainty":"G3-G0 -2.1996 points; G3-G2 not positive every fold", "runtime_seconds":None, "audit_status":"independent replay PASS; zero outer-label reads during fit", "outcome":"closed after discovery gate", "result_class":"scientific_probe_failed_discovery", "artifact":".runs/research_20260917/ptg_task_probe_v1/summary.json"},
        {"experiment_id":"fdcrg_v1::nested_residual_gate", "parent_baseline":"arftr_v1::r5_arftr_full", "data":"Okutama 4,977 centers; 11 scenarios; 5 grouped outer folds", "features":"anchor/fine/coarse probabilities, factor contrast, entropy/margin/JS, quality and clock", "architecture":"Fine-Density Counterfactual Residual Gate; four actions with retain_ARFTR", "loss":"inner cross-fitted action utility from NLL", "training_settings":"5x5 inner gate; 50 fresh reader fits; residual cap 0.25", "seed_count":1, "metric":"pooled OOF macro-F1", "macro_f1":0.854276299199945, "accuracy":0.8593530239099859, "nll":0.3774763184527237, "brier":0.07118450612414413, "errors":700, "uncertainty":"2 rescues; 0 harms; net +2; outer positive net only fold 3; scenario bootstrap lower 0", "runtime_seconds":1299.8386746000033, "audit_status":"independent replay PASS; zero outer-label reads during fit", "outcome":"not promoted; all promotion gates failed except class-F1 and intervention caps", "result_class":"scientific_probe_failed_promotion", "artifact":".runs/research_20260917/fdcrg_v1_full/summary.json"},
    ])
    ledger = {
        "schema_version": "post_apply_ledger_v1",
        "generated": "2026-09-17",
        "full_ledger": str(FULL_LEDGER.relative_to(ROOT)).replace("\\", "/"),
        "full_ledger_sha256": sha256(FULL_LEDGER),
        "full_record_count": len(full),
        "compact_record_count": len(selected),
        "records": selected,
        "deployment_rule": "Only ARFTR r5 is retained as default; oracle/annotation-only/fixed post-hoc diagnostics are not deployable.",
    }
    (OUT / "experiment_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")

    # GraphML is deliberately generated from the JSON so both machine formats agree.
    g = nx.MultiDiGraph()
    for n in graph["nodes"]:
        attrs = {k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)) for k, v in n.items() if k != "id"}
        g.add_node(n["id"], **attrs)
    for i, e in enumerate(graph["edges"]):
        attrs = {k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)) for k, v in e.items() if k not in {"source", "target", "key"}}
        g.add_edge(e["source"], e["target"], key=str(e.get("key", i)), **attrs)
    nx.write_graphml(g, GRAPH_DIR / "knowledge_graph.graphml")
    (GRAPH_DIR / "README.txt").write_text("knowledge_graph.json is the authoritative JSON; knowledge_graph.graphml is generated from the same nodes/edges.\n", encoding="utf-8")

    print(json.dumps({"out": str(OUT), "nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "ledger_records": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
