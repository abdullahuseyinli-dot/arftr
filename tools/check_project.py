"""Validate the current public checkout using only Python's standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
# Immutable release inventories, not hashes regenerated from the current checkout.
# Pinning their bytes prevents omitted entries from silently weakening validation.
LOCKED_INVENTORIES = {
    "results/arftr_development/evidence_manifest.json": "c65ba1d633a0479e4719afabb2f81b6a7cefbb8622731d449494cf179815a6d3",
    "results/arftr_development/architecture_manifest.json": "ecb61efa4f5c0ed4a906f02be531f79075a7966c9708e3243ccfd248fb69c004",
    "results/human_activity_study_v3.0.0_manifest.json": "9be8160b6a56fa1eb944956a13edb3bbe42877579110812631b86597f04025fd",
    "experiments/okutama_motion_null_contrast_protocol.json": "6f98699dea12f7067ef1672958e41abeaa2f7b85fbb6f0550bf261532daf936a",
    ".runs/research_20260920/repo_polish_20260920_v1/before_cleanup.json": "c0f8ba2869a9f3f7eb2c8514f61d45980a35c6b7c1e7d9e2181f53330fe2a1eb",
    ".runs/research_20260920/research_closeout_v1/preservation_manifest.json": "2ef2eaa0523f4527ecf38c178a85076e2cb217fb64636ad99dab7bb7c61fdbd3",
    ".runs/research_20260920/motion_null_contrast_v1/completion_receipt.json": "d8054421cff18a96c8e501f58d325b7dc931a16df2977cd86ed76a28c391dbba",
    ".runs/research_20260920/motion_null_contrast_v1/execution_lock.json": "bd93d032b51af6224961af541081ae46d57faf21b65db86d17a4c23413990d15",
}
CLASSES = ["sitting", "standing", "walking_running"]
SCORE_FIELDS = {
    "accuracy",
    "brier_sum",
    "confusion",
    "errors",
    "macro_f1",
    "nll",
    "per_class_f1",
    "precision",
    "recall",
    "support",
}
CURRENT_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/README.md",
    "docs/RESEARCH_OVERVIEW.md",
    "docs/ARCHITECTURE.md",
    "docs/ARFTR_REPORT.md",
    "docs/MODEL_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/REPOSITORY_MAINTENANCE.md",
    "docs/VALIDATION.md",
    "docs/RESULTS.md",
    "docs/PROJECT_HISTORY.md",
    "assets/README.md",
    "data/README.md",
    "experiments/README.md",
    "output/pdf/README.md",
    "results/README.md",
    "results/arftr_development/README.md",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the repository: {relative}")
    return path


def sha256(path: Path, *, normalized: bool = False) -> str:
    data = path.read_bytes()
    if normalized:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def locked_json(root: Path, relative: str) -> dict:
    path = within(root, relative)
    if sha256(path, normalized=True) != LOCKED_INVENTORIES[relative]:
        raise ValueError(f"Locked evidence inventory or protocol changed: {relative}")
    return read_json(path)


def confusion_metrics(matrix: list[list[int]]) -> dict:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected a three-class confusion matrix")
    if any(type(n) is not int or n < 0 for row in matrix for n in row):
        raise ValueError("Confusion counts must be nonnegative integers")
    total = sum(map(sum, matrix))
    if total <= 0:
        raise ValueError("Empty evaluation population")
    correct = sum(matrix[i][i] for i in range(3))
    f1 = []
    for i in range(3):
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    return {
        "rows": total,
        "errors": total - correct,
        "accuracy": correct / total,
        "macro_f1": sum(f1) / 3,
        "per_class_f1": f1,
        "precision": [
            matrix[i][i] / sum(row[i] for row in matrix) if sum(row[i] for row in matrix) else 0.0
            for i in range(3)
        ],
        "recall": [matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else 0.0 for i in range(3)],
    }


def check_scores(score: dict, rows: int) -> None:
    calculated = confusion_metrics(score["confusion"])
    for key in ("errors", "accuracy", "macro_f1"):
        if not math.isclose(calculated[key], score[key], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Confusion matrix disagrees with {key}")
    if (
        calculated["rows"] != rows
        or score["support"] != list(map(sum, score["confusion"]))
        or score.get("rows", rows) != rows
    ):
        raise ValueError("Evaluation population changed")
    for key in ("per_class_f1", "precision", "recall"):
        if key not in score and key != "per_class_f1":
            continue  # The older architecture export has no precision/recall columns.
        if len(score[key]) != 3 or any(
            not math.isclose(a, b, abs_tol=1e-12, rel_tol=0)
            for a, b in zip(calculated[key], score[key], strict=True)
        ):
            raise ValueError(f"Per-class {key} differs")


def evidence(root: Path) -> dict:
    folder = root / "results/arftr_development"
    manifest = locked_json(root, "results/arftr_development/evidence_manifest.json")
    if manifest["schema_version"] != 1 or set(manifest["artifacts"]) != {
        "experiment_ledger.csv",
        "knowledge_graph.json",
        "metrics.json",
    }:
        raise ValueError("Public evidence inventory differs")
    for name, item in manifest["artifacts"].items():
        if sha256(within(folder, name), normalized=True) != item["sha256"]:
            raise ValueError(f"Public evidence hash changed: {name}")
    result = read_json(folder / "metrics.json")
    if result["classes"] != CLASSES:
        raise ValueError("Class mapping differs")
    if (
        result["schema_version"] != 1
        or result["rows"] != 4977
        or result["outer_folds"] != 5
        or result["scenarios"] != 11
        or set(result["scores"]) != {"ARFTR", "plain", "paired_null"}
        or set(result["transitions"]) != {"plain", "paired_null"}
    ):
        raise ValueError("Development population or model inventory differs")
    if result["evaluation_scope"] != "adaptive_internal_development_not_independent_confirmation":
        raise ValueError("Development scope must remain explicit")
    for score in result["scores"].values():
        if set(score) != SCORE_FIELDS:
            raise ValueError("Development score schema differs")
        check_scores(score, result["rows"])
    base, plain, paired = (result["scores"][k] for k in ("ARFTR", "plain", "paired_null"))
    for arm, transition in result["transitions"].items():
        if len(transition["per_fold_net"]) != 5 or len(transition["per_fold_metrics"]) != 5:
            raise ValueError("Outer-fold inventory differs")
        if transition["rescues"] - transition["harms"] != transition["net"]:
            raise ValueError("Rescue/harm arithmetic differs")
        if sum(transition["per_fold_net"]) != transition["net"]:
            raise ValueError("Fold net arithmetic differs")
        if base["errors"] - result["scores"][arm]["errors"] != transition["net"]:
            raise ValueError("Error-count transition differs")
        for score in transition["per_fold_metrics"]:
            if set(score) != SCORE_FIELDS:
                raise ValueError("Fold score schema differs")
            check_scores(score, sum(score["support"]))
        summed = [
            [sum(s["confusion"][i][j] for s in transition["per_fold_metrics"]) for j in range(3)]
            for i in range(3)
        ]
        if summed != result["scores"][arm]["confusion"]:
            raise ValueError("Fold confusion matrices do not sum to aggregate")
    protocol = locked_json(root, "experiments/okutama_motion_null_contrast_protocol.json")
    criteria = protocol["continuation_gates"]
    intervals = result["bootstrap"]
    gates = {
        "paired_above_plain": paired["macro_f1"] > plain["macro_f1"],
        "paired_minus_plain_interval": intervals["paired_minus_plain"]["interval"][0] > 0,
        "macro_f1_gain": paired["macro_f1"] - base["macro_f1"]
        >= criteria["macro_f1_gain_vs_arftr_min"],
        "net": result["transitions"]["paired_null"]["net"] >= criteria["net_corrections_min"],
        "positive_every_fold": all(
            n > 0 for n in result["transitions"]["paired_null"]["per_fold_net"]
        ),
        "paired_minus_ARFTR_interval": intervals["paired_minus_ARFTR"]["interval"][0] > 0,
        "nll": paired["nll"] <= base["nll"] + criteria["proper_score_tolerance"],
        "brier": paired["brier_sum"] <= base["brier_sum"] + criteria["proper_score_tolerance"],
        "class_f1": min(
            a - b for a, b in zip(paired["per_class_f1"], base["per_class_f1"], strict=True)
        )
        >= -criteria["maximum_class_f1_drop"],
    }
    if gates != result["gates"] or all(gates.values()) != result["continuation_pass"]:
        raise ValueError("Locked gate decision differs")
    if result["promoted"] or result["retained_model"] != "ARFTR" or result["head_fits"] != 10:
        raise ValueError("Retained-system or fit-count claim differs")
    graph = read_json(folder / "knowledge_graph.json")
    ids = [n["id"] for n in graph["nodes"]]
    if len(ids) != len(set(ids)) or any(
        e["source"] not in ids or e["target"] not in ids for e in graph["edges"]
    ):
        raise ValueError("Public graph has duplicate nodes or dangling edges")
    return {
        "metric_rows": result["rows"],
        "models_recomputed": len(result["scores"]),
        "graph_nodes": len(ids),
        "continuation_pass": result["continuation_pass"],
    }


def navigation(root: Path) -> int:
    checked = 0
    for name in CURRENT_DOCS:
        path = root / name
        text = path.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:[\\/]Users[\\/]|/home/[^/]+/", text):
            raise ValueError(f"Machine-local user path in current documentation: {name}")
        # Deliberately check current navigation, not archival .runs provenance links.
        for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or target.startswith("#") or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                raise ValueError(f"Broken current documentation link: {name} -> {target}")
            checked += 1
    return checked


def metadata(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r'^version:\s*[\'"]?([^\'"\s]+)', cff, re.MULTILINE)
    if (
        not citation_version
        or citation_version.group(1) != version
        or read_json(root / ".zenodo.json")["version"] != version
    ):
        raise ValueError("Package/citation/deposit versions disagree")
    if (
        "dev" in version
        and "unreleased" not in (root / "README.md").read_text(encoding="utf-8").lower()
    ):
        raise ValueError("Development version must be identified as unreleased")
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    if any(token not in ignore.splitlines() for token in (".venv-*/", ".runs/", "*.pt", ".env")):
        raise ValueError("Missing local-artifact exclusions")
    return version


def local_preservation(root: Path) -> dict:
    original = locked_json(
        root, ".runs/research_20260920/repo_polish_20260920_v1/before_cleanup.json"
    )
    documents = {r["path"]: r for r in original["documents"]}
    for item in original["documents"]:
        if sha256(within(root, item["snapshot"])) != item["sha256"]:
            raise ValueError("Pre-cleanup document snapshot changed")
    manifest = locked_json(
        root, ".runs/research_20260920/research_closeout_v1/preservation_manifest.json"
    )
    substitutions = []
    for record in manifest["files"]:
        if sha256(within(root, record["path"])) == record["sha256"]:
            continue
        saved = documents.get(record["path"])
        if not saved or not record["path"].endswith(".md") or saved["sha256"] != record["sha256"]:
            raise ValueError(f"Protected non-document evidence changed: {record['path']}")
        substitutions.append(
            {
                "original_path": record["path"],
                "snapshot": saved["snapshot"],
                "sha256": saved["sha256"],
            }
        )
    completion = locked_json(
        root, ".runs/research_20260920/motion_null_contrast_v1/completion_receipt.json"
    )
    for item in completion["artifacts"]:
        if sha256(within(root, item["path"])) != item["sha256"]:
            raise ValueError(f"Completed-experiment artifact changed: {item['path']}")
    lock = locked_json(root, ".runs/research_20260920/motion_null_contrast_v1/execution_lock.json")
    for item in lock["dependencies"]:
        if sha256(within(root, item["path"])) != item["sha256"]:
            raise ValueError(f"Locked experiment dependency changed: {item['path']}")
    return {
        "protected_files": len(manifest["files"]),
        "document_snapshot_substitutions": substitutions,
        "completed_experiment_files": len(completion["artifacts"]),
        "locked_experiment_dependencies": len(lock["dependencies"]),
    }


def historical_evidence(root: Path) -> int:
    """Verify frozen numerical exports and reports, not mutable project metadata."""
    manifest = locked_json(root, "results/human_activity_study_v3.0.0_manifest.json")
    checked = 0
    for name, item in manifest["artifacts"].items():
        if not name.startswith(("results/", "output/pdf/", "assets/")) or name.endswith(
            "README.md"
        ):
            continue
        path = within(root, name)
        normalized = path.suffix.lower() in {".csv", ".json", ".svg", ".txt", ".md"}
        if sha256(path, normalized=normalized) != item["sha256"]:
            raise ValueError(f"Historical numerical evidence/report changed: {name}")
        checked += 1
    return checked


def architecture_evidence(root: Path) -> int:
    """Check the original ARFTR component export without opening private artifacts."""
    folder = root / "results/arftr_development"
    manifest = locked_json(root, "results/arftr_development/architecture_manifest.json")
    if manifest["schema_version"] != 1 or set(manifest["artifacts"]) != {"architecture_study.json"}:
        raise ValueError("Architecture evidence inventory differs")
    for name, item in manifest["artifacts"].items():
        if sha256(within(folder, name), normalized=True) != item["sha256"]:
            raise ValueError(f"Architecture evidence hash changed: {name}")
    recipe = manifest["protocol"]
    protocol_path = within(root, recipe["path"])
    if sha256(protocol_path, normalized=True) != recipe["sha256"]:
        raise ValueError("Architecture protocol changed")
    protocol = read_json(protocol_path)
    study = read_json(folder / "architecture_study.json")
    if (
        set(study["scores"]) != set(protocol["arms"])
        or len(study["scores"]) != 7
        or study["schema_version"] != 1
        or study["primary_arm"] != protocol["primary_arm"]
        or study["evaluation_scope"] != protocol["status"]
        or study["outer_folds"] != len(protocol["population"]["outer_folds"])
        or study["scenarios"] != 11
        or study["classes"] != protocol["population"]["classes"]
        or study["rows"] != protocol["population"]["rows"]
        or study["prediction_seeds"] != protocol["population"]["prediction_seeds"]
        or study["ablation_contract"] != protocol["control_contract"]
    ):
        raise ValueError("Architecture comparison contract differs")
    for score in study["scores"].values():
        if set(score) != (SCORE_FIELDS - {"precision", "recall"}) | {"rows"}:
            raise ValueError("Architecture score schema differs")
        check_scores(score, study["rows"])
    primary = study["scores"][study["primary_arm"]]
    retained = read_json(folder / "metrics.json")["scores"]["ARFTR"]
    for key in ("macro_f1", "accuracy", "errors", "confusion", "nll", "brier_sum"):
        if primary[key] != retained[key]:
            raise ValueError(f"Architecture primary differs from retained ARFTR: {key}")
    anchor = study["scores"]["r0_exact_m4"]
    transitions = study["transitions_vs_exact_m4"]
    if (
        transitions["rescues"] - transitions["harms"] != transitions["net_corrections"]
        or anchor["errors"] - primary["errors"] != transitions["net_corrections"]
        or not math.isclose(
            100 * (primary["macro_f1"] - anchor["macro_f1"]),
            study["effects"]["macro_f1_gain_points"],
            abs_tol=1e-12,
            rel_tol=0,
        )
    ):
        raise ValueError("Architecture transition/effect arithmetic differs")
    return len(study["scores"])


def figures(root: Path) -> int:
    """Bind the current development figures to their public sources and renderer."""
    folder = root / "assets"
    manifest = read_json(folder / "arftr_figure_manifest.json")
    expected = {
        f"arftr_{name}.{suffix}"
        for name in (
            "development_summary",
            "confusion_matrix",
            "architecture_results",
            "report_components",
            "system_overview",
        )
        for suffix in ("png", "svg")
    }
    if set(manifest["artifacts"]) != expected or set(manifest["sources"]) != {
        "results/arftr_development/metrics.json",
        "results/arftr_development/experiment_ledger.csv",
        "results/arftr_development/architecture_study.json",
        "tools/render_arftr_figures.py",
    }:
        raise ValueError("Current figure inventory differs")
    for name, item in manifest["sources"].items():
        if sha256(within(root, name), normalized=True) != item["sha256"]:
            raise ValueError(f"Figure source changed; regenerate current figures: {name}")
    for name, item in manifest["artifacts"].items():
        if sha256(within(folder, name), normalized=Path(name).suffix == ".svg") != item["sha256"]:
            raise ValueError(f"Figure artifact changed: {name}")
    return len(expected)


def report(root: Path) -> int:
    """Validate the current report separately from immutable historical reports."""
    manifest = read_json(root / "output/pdf/arftr_report_v1.0.0.manifest.json")
    expected = {
        "docs/ARFTR_REPORT.md",
        "tools/build_study_papers.py",
        "assets/arftr_figure_manifest.json",
        "output/pdf/arftr_report_v1.0.0.pdf",
    }
    if (
        manifest["schema_version"] != 1
        or manifest["project_version"] != metadata(root)
        or set(manifest["artifacts"]) != expected
    ):
        raise ValueError("Current report inventory or version differs")
    for name, item in manifest["artifacts"].items():
        normalized = not name.endswith(".pdf")
        if item["normalized_lf"] is not normalized or sha256(
            within(root, name), normalized=normalized
        ) != item["sha256"]:
            raise ValueError(f"Current report source/artifact differs: {name}")
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--local-preservation", action="store_true")
    args = parser.parse_args()
    root = args.repository.resolve()
    result = {
        "status": "PASS",
        "version": metadata(root),
        "current_links_checked": navigation(root),
        "evidence": evidence(root),
        "architecture_arms_recomputed": architecture_evidence(root),
        "historical_evidence_files": historical_evidence(root),
        "current_figure_files": figures(root),
        "current_report_bindings": report(root),
        "scope": "portable aggregate validation; not model execution",
    }
    if args.local_preservation:
        result["local_preservation"] = local_preservation(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
