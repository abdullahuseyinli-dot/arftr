"""Preserve pre-cleanup documents and export aggregate-only research evidence.

This local maintenance command never trains, deletes, moves, or publishes anything.
It requires private run artifacts. Public validation does not require this exporter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / ".runs/research_20260920/repo_polish_20260920_v1"
PUBLIC = ROOT / "results/arftr_development"
PRIVATE = ROOT / ".runs/research_20260920"
DOCUMENTS = (
    "README.md",
    ".gitignore",
    ".gitattributes",
    ".github/workflows/ci.yml",
    "pyproject.toml",
    "CITATION.cff",
    ".zenodo.json",
    "CHANGELOG.md",
    "experiments/README.md",
    "docs/RESULT_LINEAGE.md",
    "docs/PORTFOLIO_ARTICLE.md",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare() -> None:
    if (SNAPSHOT / "before_cleanup.json").exists():
        raise RuntimeError("Snapshot exists; do not overwrite pre-cleanup evidence")
    records = []
    for relative in DOCUMENTS:
        source = ROOT / relative
        target = SNAPSHOT / "original_documents" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"Snapshot target already exists: {relative}")
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError("Snapshot copy changed bytes")
        records.append(
            {
                "path": relative,
                "sha256": digest(source),
                "snapshot": target.relative_to(ROOT).as_posix(),
            }
        )
    write_json(
        SNAPSHOT / "before_cleanup.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "authorization": "Clean up the repo, documents and relevant parts for research/portfolio use",
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "documents": records,
            "data_or_runs_deleted": False,
            "scope": "In-place presentation cleanup; historical numerical evidence and locked research code stay unchanged",
        },
    )
    summary_path = PRIVATE / "motion_null_contrast_v1/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit_path = PRIVATE / "motion_null_contrast_v1/replay_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    queue_path = PRIVATE / "motion_null_contrast_v1/queue_receipt.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    export = {
        "schema_version": 1,
        "study": "ARFTR development and matched motion-null closeout",
        "evaluation_scope": "adaptive_internal_development_not_independent_confirmation",
        "classes": ["sitting", "standing", "walking_running"],
        "rows": 4977,
        "outer_folds": 5,
        "scenarios": 11,
        "retained_model": "ARFTR",
        "scores": summary["scores"],
        "transitions": summary["transitions"],
        "bootstrap": summary["bootstrap"],
        "gates": summary["gates"],
        "continuation_pass": summary["raw_continuation_gate_pass"],
        "head_fits": 10,
        "router_fits": 0,
        "promoted": False,
        "replay": {
            k: audit[k]
            for k in (
                "status",
                "head_replays",
                "trained_null_heads_verified",
                "matched_schedules",
                "independence_scope",
                "replay_tolerance_changed",
            )
        },
        "numerical_exceptions": queue["historical_control_checks"],
        "sources": [
            {"local_artifact": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
            for p in (summary_path, audit_path, queue_path)
        ],
        "limitations": [
            "Aggregate metric replay is not checkpoint reproduction",
            "No row labels, sample identities, pixels, feature caches or checkpoints included",
            "Scenario confidence intervals are exported, not recomputable from these aggregate matrices",
            "One seed for the final matched comparison; no seed-variance estimate",
        ],
    }
    write_json(PUBLIC / "metrics.json", export)
    ledger_path = PRIVATE / "viability_reassessment_v1/fresh_metric_ledger.csv"
    with ledger_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (PRIVATE / "motion_null_contrast_v1/summary_table.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        for item in csv.DictReader(stream):
            if item["model"] == "ARFTR":
                continue
            rows.append(
                {
                    "result": "matched_" + item["model"],
                    **{
                        k: item[k]
                        for k in (
                            "macro_f1_percent",
                            "accuracy_percent",
                            "errors",
                            "nll",
                            "brier_sum",
                        )
                    },
                    "rescues_vs_ARFTR": item["rescues_vs_arftr"],
                    "harms_vs_ARFTR": item["harms_vs_arftr"],
                    "net_vs_ARFTR": item["net_vs_arftr"],
                    "fold_nets": item["fold_nets"],
                    "scope": "matched_optional_experiment_not_promoted",
                }
            )
    with (PUBLIC / "experiment_ledger.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    graph = json.loads(
        (PRIVATE / "motion_null_contrast_v1/knowledge_graph.json").read_text(encoding="utf-8")
    )
    # Keep the full graph locally. Export an intentionally small, auditable result map.
    nodes = [
        {
            "id": r["result"],
            "kind": "experiment",
            "macro_f1_percent": float(r["macro_f1_percent"]),
            "errors": int(r["errors"]),
            "scope": r["scope"],
            "artifact": "experiment_ledger.csv",
            "retained": r["result"] == "ARFTR",
        }
        for r in rows
    ]
    edges = [
        {
            "source": "T2",
            "target": "ARFTR",
            "relation": "historical_bundle_gain",
            "effect_pp": summary["scores"]["ARFTR"]["macro_f1"] * 100
            - float(next(r for r in rows if r["result"] == "T2")["macro_f1_percent"]),
            "causal_scope": "Multiple changes; not a single-component causal attribution",
        },
        {
            "source": "matched_plain",
            "target": "matched_paired_null",
            "relation": "matched_architectural_intervention",
            "effect_pp": 100
            * (
                summary["scores"]["paired_null"]["macro_f1"]
                - summary["scores"]["plain"]["macro_f1"]
            ),
            "continuation_pass": False,
        },
    ]
    for r in rows:
        if r["result"] not in ("ARFTR", "T2"):
            edges.append(
                {
                    "source": r["result"],
                    "target": "ARFTR",
                    "relation": "compared_with_not_promoted",
                    "net_corrections": int(r["net_vs_ARFTR"]),
                    "artifact": "experiment_ledger.csv",
                }
            )
    write_json(
        PUBLIC / "knowledge_graph.json",
        {
            "schema_version": 1,
            "scope": "curated_public_result_map_not_full_experimental_history",
            "full_local_graph": {
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "sha256": digest(PRIVATE / "motion_null_contrast_v1/knowledge_graph.json"),
            },
            "nodes": nodes,
            "edges": edges,
        },
    )
    files = [
        PUBLIC / name for name in ("metrics.json", "experiment_ledger.csv", "knowledge_graph.json")
    ]
    write_json(
        PUBLIC / "evidence_manifest.json",
        {
            "schema_version": 1,
            "hash_semantics": "UTF-8 text with CRLF normalized to LF for checkout portability",
            "artifacts": {
                p.name: {
                    "sha256": hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                }
                for p in files
            },
            "ledger_source_sha256": digest(ledger_path),
        },
    )
    print(
        json.dumps(
            {
                "snapshot_documents": len(records),
                "public_evidence_files": len(files) + 1,
                "experiment_rows": len(rows),
                "training_started": False,
            }
        )
    )


if __name__ == "__main__":
    prepare()
