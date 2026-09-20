"""Release the completed optional study and append its result to the evidence graph."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_okutama_motion_null_contrast as trial


def write_text(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def report():
    run = trial.RUN
    trial.validate(run)
    result = trial.read(run / "summary.json")
    queue = trial.read(run / "queue_receipt.json")
    audit = trial.read(run / "replay_audit.json")
    amendment = trial.read(run / "numerical_parity_exception_v2.json")
    if result["status"] != "MOTION_NULL_EXPERIMENT_COMPLETE" or audit["head_replays"] != 10 or queue["head_fits"] != 10:
        raise RuntimeError("All fits, replay and locked evaluation must complete first")
    if trial.file_sha256(ROOT / "experiments/complete_okutama_motion_null_contrast.py") != amendment["completion_source_sha256"]:
        raise RuntimeError("Completion adapter changed")
    if trial.file_sha256(run / "numerical_parity_exception_v2.json") != queue["numerical_exception_sha256"]:
        raise RuntimeError("Amendment binding changed")
    for item in amendment["preexisting_artifacts"]:
        trial.verify_record(item)
    scores = result["scores"]
    base, plain, paired = (scores[k] for k in ("ARFTR", "plain", "paired_null"))
    passed = result["raw_continuation_gate_pass"]
    decision = "Consider a separately authorized nested-policy phase; do not promote." if passed else "Close this optional branch for this search cycle; retain ARFTR and return to repository release-readiness work."
    cohort = trial.arrays(run / "inputs/cohort.npz")
    y = trial.arrays(trial.legacy.ARFTR)["labels"].astype(int)
    predictions = {}
    for arm in trial.ARMS:
        p = np.full_like(cohort["anchor"], np.nan)
        for fold in range(5):
            saved = trial.arrays(run / arm / f"fold-{fold}/predictions.npz")
            p[saved["held_rows"]] = saved["candidate_probabilities"]
        predictions[arm] = p
    comparison = trial.transitions(y, predictions["plain"], predictions["paired_null"], cohort["folds"])
    trial.write(run / "paired_vs_plain_transitions.json", comparison)
    rows = []
    for name in ("ARFTR", "plain", "paired_null"):
        s = scores[name]
        t = result["transitions"].get(name, {"rescues": 0, "harms": 0, "net": 0, "per_fold_net": [0] * 5})
        rows.append({"model": name, "macro_f1_percent": 100 * s["macro_f1"], "accuracy_percent": 100 * s["accuracy"],
                     "errors": s["errors"], "rescues_vs_arftr": t["rescues"], "harms_vs_arftr": t["harms"],
                     "net_vs_arftr": t["net"], "macro_f1_gain_pp": 100 * (s["macro_f1"] - base["macro_f1"]),
                     "nll": s["nll"], "brier_sum": s["brier_sum"], "fold_nets": json.dumps(t["per_fold_net"])})
    with (run / "summary_table.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "| Model | Macro-F1 | Accuracy | Errors | Rescues / harms vs ARFTR | Net |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    table += "".join(f"| {r['model']} | {r['macro_f1_percent']:.6f}% | {r['accuracy_percent']:.6f}% | {r['errors']} | {r['rescues_vs_arftr']} / {r['harms_vs_arftr']} | {r['net_vs_arftr']:+d} |\n" for r in rows)
    intervals = "\n".join(f"- {name}: [{100*v['interval'][0]:+.6f}, {100*v['interval'][1]:+.6f}] macro-F1 percentage points." for name, v in result["bootstrap"].items())
    gates = "\n".join(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in result["gates"].items())
    parity = "\n".join(f"| {v['fold']} | {v['max_abs_delta']:.12g} | {v['max_abs_probability']:.12g} | {v['class_prediction_disagreements']} | {'exception' if v['accepted_by_user_exception'] else 'original tolerance passed'} |" for v in queue["historical_control_checks"])
    detail = "\n".join(f"| {name} | {scores[name]['nll']:.9f} | {scores[name]['brier_sum']:.9f} | " + " / ".join(f"{100*f:.6f}%" for f in scores[name]["per_class_f1"]) + " |" for name in ("ARFTR", "plain", "paired_null"))
    text = f"""# Optional shared motion-null experiment — completed

## Decision

{decision} No candidate was promoted and no router was fitted.

All ten first-attempt head fits completed: five plain controls and five shared
paired-null heads. Three preexisting fits were preserved; seven were completed
under the second user-approved numerical-only amendment. No retries, selected
checkpoints, seed sweeps, threshold searches, or changed performance gates.

## Results on all 4,977 outer-held rows

{table}
Paired-null minus matched plain: {100*(paired['macro_f1']-plain['macro_f1']):+.6f} pp.
Paired-null minus ARFTR: {100*(paired['macro_f1']-base['macro_f1']):+.6f} pp.
Paired-null fold nets versus ARFTR: {result['transitions']['paired_null']['per_fold_net']}.
Plain fold nets versus ARFTR: {result['transitions']['plain']['per_fold_net']}.
Paired-null versus plain: {comparison['rescues']} rescues, {comparison['harms']} harms,
{comparison['net']:+d} net; fold nets {comparison['per_fold_net']}.

Scenario-bootstrap 95% intervals: 100,000 resamples of 11 scenarios, fixed seed
20260920; matched scenario resampling, not independent-row resampling.

{intervals}

| Model | NLL | Brier (sum over classes) | Class F1: sitting / standing / walking |
| --- | ---: | ---: | --- |
{detail}

All precision, recall, confusion matrices, supports, and per-fold metrics are in
[summary.json](summary.json). No seed-variance estimate is possible from seed 42 alone.

## Unchanged locked gates

| Gate | Result |
| --- | --- |
{gates}

The required ARFTR gain was at least +0.50 macro-F1 pp and +25 net corrections,
strictly positive net in every outer fold, positive scenario-interval lower
bounds against ARFTR and matched plain, NLL/Brier deterioration no greater than
1e-6, and class-F1 loss no greater than 0.005. The matched plain point estimate
also had to be beaten. Gates were fixed before new performance scores were opened.

## What was tested, and what can be concluded

The architectural intervention was training from initialization with
`delta = 0.5*tanh(h(a,b,g)-h(a,0,g))`, sharing weights across both calls.
Inputs, ancestry-safe training priors, outer folds, seed 42, minibatches, 256
updates, optimizer and loss matched the plain arm. The algebraically cancelled
scalar bias was frozen: 120,128 active parameters versus 120,129 for plain.
Ordered pair means and geometry still contain temporal information: this is an
explicit-difference-carrier null invariant, not proof of a motion-free reference.

This is now a faithful training test, unlike the earlier fixed-weight post-hoc
subtraction. The matched comparison estimates this whole recipe change on this
development cohort; it does not isolate every gradient/optimization effect or
establish a general ceiling on motion recognition. Exact nullness and replay are
mechanism checks, not performance evidence. A miss rules out proceeding under
these prespecified gates in this cycle, not all possible future architectures.

## Numerical exceptions and integrity

| Plain fold | Max residual difference vs historical | Max probability difference | Class disagreements | Numerical status |
| --- | ---: | ---: | ---: | --- |
{parity}

The original 1e-6 residual-parity rule remains visible in the unchanged protocol.
Its numerical exceptions were expressly user-approved before releasing outcomes;
they are not reported as passing that original rule. Identical class predictions,
population/ancestry checks, normalized inputs and exact anchor references remained
mandatory. The underlying source of the small training differences is not proven.
Actual NLL/Brier are evaluated on the saved probabilities, not copied from history.

Separate-formula replay reconstructed all ten checkpoints' saved residuals and
probabilities bit-for-bit. All five trained paired heads satisfy the zero-carrier
null invariant; exact retain, bounds, class-pair mass and matched row schedules
passed. This is separately written replay by the same author, not a second-person
audit or an independent implementation of the feature cache/network modules.

The original auditor initially failed paired-null fold 0 because constructing the
reference with channel concatenation changed its memory strides. A label-free,
same-checkpoint experiment held reference values fixed and varied only layout:
the original auditor differed by 0.000563522335 in residuals, while the actual
trained forward and layout-preserving separate formula both matched saved outputs
exactly. Auditor v2 explicitly preserves reference strides. The original auditor
remains unchanged and hash-bound; the bit-exact replay requirement was not relaxed.
This explains the replay discrepancy, not the separate historical-training drift.
See [layout diagnosis](replay_layout_diagnosis.json) and the source hashes in the
replay receipt. No held-out performance scores were opened until v2 replay passed.

The first two plain controls lost their in-memory loss traces and elapsed times
when the original runner raised after saving checkpoint/predictions. Those fields
remain null. Their minibatch schedule hashes are reconstructed from the locked
recipe; later parity exceptions capture actual in-memory telemetry without reruns.
All 218 protected historical files and the original execution-lock dependencies
remain unchanged. All seven new fits ran on CUDA. No training is left running.

## Scope and next action

These are raw producer diagnostics on an adaptively reused internal development
cohort, not untouched confirmation or deployable router scores. No annotation-only
signal was added to inference and no held labels trained an intervention policy.
ARFTR remains retained. {decision}

Evidence: [locked summary](summary.json), [replay audit](replay_audit.json),
[queue receipt](queue_receipt.json), [original exception](numerical_parity_exception.json),
[extended exception](numerical_parity_exception_v2.json),
[protocol](../../../experiments/okutama_motion_null_contrast_protocol.json).
Earlier stop notes are immutable historical snapshots, superseded by this completed report.
"""
    write_text(run / "RESULT_REPORT.md", text)
    base_graph_path = ROOT / ".runs/research_20260920/viability_reassessment_v1/knowledge_graph.json"
    graph = trial.read(base_graph_path)
    old_node = "reassessment:20260920:contrast_training_unknown"
    for node in graph["nodes"]:
        if node["id"] == old_node:
            node["pre_completion_snapshot"] = dict(node)
            node.update(kind="tested_hypothesis", status="TESTED_FIXED_RECIPE_CONTINUATION_PASS" if passed else "TESTED_FIXED_RECIPE_CONTINUATION_FAIL",
                        label="Faithful paired-null training completed; see locked matched comparison",
                        artifact=str((run / "RESULT_REPORT.md").relative_to(ROOT)).replace("\\", "/"))
    prefix = "motion_null_contrast:20260920:"
    summary_path = str((run / "summary.json").relative_to(ROOT)).replace("\\", "/")
    new_nodes = [{"id": prefix + name, "kind": "experiment", "label": name,
                  "macro_f1": scores[name]["macro_f1"], "errors": scores[name]["errors"],
                  "artifact": summary_path, "scope": result["scope"], "promoted": False}
                 for name in trial.ARMS]
    new_nodes += [{"id": prefix + "decision", "kind": "decision", "label": decision, "artifact": summary_path},
                  {"id": prefix + "null_invariant", "kind": "verified_mechanism", "label": "Five trained heads: exact zero-carrier nullness; ten checkpoint replays pass", "artifact": str((run / "replay_audit.json").relative_to(ROOT)).replace("\\", "/")},
                  {"id": prefix + "numerical_exception", "kind": "protocol_amendment", "label": "User-approved numeric-only exceptions; class/provenance/performance gates unchanged", "artifact": str((run / "numerical_parity_exception_v2.json").relative_to(ROOT)).replace("\\", "/")}]
    old_ids = {n["id"] for n in graph["nodes"]}
    if any(n["id"] in old_ids for n in new_nodes):
        raise RuntimeError("Graph result already exists")
    graph["nodes"].extend(new_nodes)
    def edge(source, target, relation, **attrs):
        graph["edges"].append({"source": source, "target": target, "relation": relation,
                              "artifact": summary_path, "evidence_strength": "matched fixed-recipe 5-fold single-seed adaptive-development study", **attrs})
    edge(prefix + "paired_null", prefix + "plain", "improves" if paired["macro_f1"] > plain["macro_f1"] else "regresses",
         effect_size_pp=100*(paired["macro_f1"]-plain["macro_f1"]), confidence_interval95=result["bootstrap"]["paired_minus_plain"]["interval"], confidence_interval_units="macro_f1_fraction")
    edge(prefix + "paired_null", "experiment:arftr_v1/r5_arftr_full", "improves" if paired["macro_f1"] > base["macro_f1"] else "regresses",
         effect_size_pp=100*(paired["macro_f1"]-base["macro_f1"]), confidence_interval95=result["bootstrap"]["paired_minus_ARFTR"]["interval"], confidence_interval_units="macro_f1_fraction")
    edge(prefix + "paired_null", old_node, "tests-hypothesis", continuation_gate_pass=passed)
    edge(prefix + "null_invariant", prefix + "paired_null", "verifies-mechanism")
    edge(prefix + "numerical_exception", prefix + "plain", "qualifies-comparability")
    edge(prefix + "paired_null", prefix + "decision", "supports-decision")
    graph.update(base_graph=str(base_graph_path.relative_to(ROOT)).replace("\\", "/"), base_graph_sha256=trial.file_sha256(base_graph_path),
                 schema_version="motion_null_completion_v1", current_decision=prefix + "decision", current_primary_experiment=None,
                 decision_scope="completed_optional_experiment_no_automatic_publication_or_promotion",
                 node_count=len(graph["nodes"]), edge_count=len(graph["edges"]))
    ids = {n["id"] for n in graph["nodes"]}
    if any(e["source"] not in ids or e["target"] not in ids for e in graph["edges"]):
        raise RuntimeError("Graph has dangling edges")
    trial.write(run / "knowledge_graph.json", graph)
    nx_graph = nx.MultiDiGraph()
    def scalar(value):
        return value if isinstance(value, (str, int, float, bool)) else json.dumps(value, sort_keys=True)
    for node in graph["nodes"]:
        nx_graph.add_node(node["id"], **{k: scalar(v) for k, v in node.items() if k != "id"})
    for e in graph["edges"]:
        nx_graph.add_edge(e["source"], e["target"], **{k: scalar(v) for k, v in e.items() if k not in ("source", "target")})
    with (run / "knowledge_graph.graphml").open("xb") as stream:
        nx.write_graphml(nx_graph, stream)
    mind = f"""# Completed optional branch in the research tree

```text
Retained ARFTR: {100*base['macro_f1']:.6f}% macro-F1, {base['errors']} errors [UNCHANGED]
  |
  +-- Earlier signed-motion head: omitted shared null subtraction
  |     Post-hoc fixed-weight subtraction did not test training the intended model
  |
  +-- Authorized matched training: same cache / priors / folds / seed / schedule / loss
        |
        +-- Plain:       {100*plain['macro_f1']:.6f}%, {plain['errors']} errors
        +-- Paired-null: {100*paired['macro_f1']:.6f}%, {paired['errors']} errors
              |
              +-- Exact trained nullness and all ten checkpoint replays: PASS
              +-- Net vs ARFTR: {result['transitions']['paired_null']['net']:+d}
              +-- Fold nets: {result['transitions']['paired_null']['per_fold_net']}
              +-- Locked continuation gate: {'PASS' if passed else 'FAIL'}
              +-- Decision: {decision}
```

Full historical nodes/edges are retained in [knowledge_graph.json](knowledge_graph.json)
and [GraphML](knowledge_graph.graphml); only the formerly open contrast-training
question receives a current result, with its prior state preserved.
See [RESULT_REPORT.md](RESULT_REPORT.md) for intervals, gates, and limitations.
"""
    write_text(run / "MIND_MAP.md", mind)
    artifacts = [p for p in sorted(run.rglob("*")) if p.is_file()]
    receipt = {"status": "OPTIONAL_MOTION_NULL_EXPERIMENT_FINISHED", "decision": decision,
               "head_fits": 10, "new_fits_this_authorization": 7, "retrained_fits": 0,
               "raw_continuation_gate_pass": passed, "ARFTR_retained": True, "promoted": False,
               "publication_performed": False, "new_router_fits": 0, "running_training_jobs": 0,
               "graph_nodes": len(graph["nodes"]), "graph_edges": len(graph["edges"]),
               "preserved_files_verified": trial.check_preservation(),
               "report_source": trial.record(Path(__file__)), "artifacts": [trial.record(p) for p in artifacts]}
    trial.write(run / "completion_receipt.json", receipt)
    return {k: v for k, v in receipt.items() if k not in ("artifacts", "report_source")}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
