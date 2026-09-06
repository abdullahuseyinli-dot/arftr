"""Fail fast on portability, notebook, evidence, and repository-size regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pandas as pd
from pypdf import PdfReader

TEXT_SUFFIXES = {".csv", ".ipynb", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
LEGACY_FINGERPRINT_TEXT_SUFFIXES = {".csv", ".json", ".py", ".svg"}
EXCLUDED_PARTS = {
    ".git",
    ".runs",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
REQUIRED = {
    ".github/workflows/ci.yml",
    ".gitignore",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "data/manifest.csv",
    "docs/EXPERIMENT_PROTOCOL.md",
    "experiments/evaluate_faithfulness.py",
    "human_activity_classification.ipynb",
    "pyproject.toml",
    "src/hac/explainability.py",
    "assets/faithfulness_method_selection.png",
    "assets/faithfulness_perturbation_curves.png",
    "results/locked_test_metrics.csv",
    "results/faithfulness_selection_lock.json",
    "results/faithfulness_test_summary.csv",
    "results/faithfulness_test_per_image.csv",
    "results/faithfulness_replay_validation.csv",
    "results/faithfulness_sanity_summary.csv",
    "results/faithfulness_stability_summary.csv",
    "results/faithfulness_checkpoint_manifest.csv",
    "results/faithfulness_oof_selection_cohort.csv",
    "results/faithfulness_provenance.json",
    "results/oof_replay_validation.csv",
    "results/run_provenance.json",
}
EXCLUDED_THIRD_PARTY_MEDIA = frozenset(
    {
        "assets/champion_error_gallery.png",
        "assets/convnext_small_faithfulness_gallery.jpg",
        "assets/dinov2_small_faithfulness_gallery.jpg",
        "assets/probability_blend_faithfulness_gallery.jpg",
    }
)
EXCLUDED_FAITHFULNESS_MEDIA_SHA256 = {
    "convnext_small_faithfulness_gallery.jpg": (
        "35865cc879b212a4f4690467c9100321b3d11ea16794280c31a47462e562ce32"
    ),
    "dinov2_small_faithfulness_gallery.jpg": (
        "ebd41b2e289377ae470dc72fbe0fddc82d72d44109c631b06cdfa66e2ffacc91"
    ),
    "probability_blend_faithfulness_gallery.jpg": (
        "cf787683493e90817ec1ac86ff44ec4d752480e0d2f8a03356f0fa9141175849"
    ),
}
POLAR_REQUIRED = {
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "docs/LEGACY_COCO_STUDY.md",
    "docs/POLAR_SCALE_STUDY_PROTOCOL.md",
    "docs/POLAR_PUBLIC_REPORT.md",
    "docs/PORTFOLIO_ARTICLE.md",
    "docs/RESULT_LINEAGE.md",
    "docs/releases/POLAR_STUDY_V1.0.0.md",
    "experiments/analyze_polar_exploratory.py",
    "experiments/evaluate_polar_faithfulness.py",
    "experiments/evaluate_polar_fault_robustness.py",
    "experiments/evaluate_polar_final.py",
    "experiments/polar_study_protocol.json",
    "output/pdf/README.md",
    "output/pdf/polar_public_report_v1.0.0.pdf",
    "release/POLAR_STUDY_V1.0.0_SHA256SUMS.txt",
    "results/polar_data_audit.json",
    "results/polar_final_evidence_manifest.json",
    "results/polar_final_fit_manifest.json",
    "results/polar_final_selection_lock.json",
    "results/polar_exploratory_summary.json",
    "results/polar_study_v1.0.0_manifest.json",
    "results/polar_test_access_gate.json",
    "results/polar_test_metrics.csv",
    "results/polar_test_summary.json",
    "results/polar_test_uncertainty.json",
    "results/polar_external_overlap_audit.json",
    "results/polar_external_summary.json",
    "results/polar_faithfulness_summary.json",
    "results/polar_fault_summary.json",
    "assets/polar_attribution_sanity.png",
    "assets/polar_confusion_matrix.png",
    "assets/polar_external_validation.png",
    "assets/polar_faithfulness.png",
    "assets/polar_fault_robustness.png",
    "assets/polar_scale_curve.png",
    "assets/polar_test_comparison.png",
    "tools/build_study_papers.py",
    "tools/build_study_release_manifest.py",
}
VCOCO_V2_REQUIRED = {
    "docs/VCOCO_V2_EXTERNAL_TRANSFER.md",
    "docs/releases/POLAR_STUDY_V2.0.0.md",
    "output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf",
    "release/POLAR_STUDY_V2.0.0_SHA256SUMS.txt",
    "results/polar_study_v2.0.0_manifest.json",
    "tools/export_vcoco_v2_results.py",
    "tools/render_vcoco_v2_figures.py",
    "assets/vcoco_v2_development_comparison.png",
    "assets/vcoco_v2_development_comparison.svg",
    "assets/vcoco_v2_fewshot_curve.png",
    "assets/vcoco_v2_fewshot_curve.svg",
    "assets/vcoco_v2_official_test_comparison.png",
    "assets/vcoco_v2_official_test_comparison.svg",
    "assets/vcoco_v2_scale_gain.png",
    "assets/vcoco_v2_scale_gain.svg",
    "assets/vcoco_v2_selective_prediction.png",
    "assets/vcoco_v2_selective_prediction.svg",
    "results/vcoco_v2/README.md",
    "results/vcoco_v2/development_candidates.csv",
    "results/vcoco_v2/evidence_manifest.json",
    "results/vcoco_v2/factorized_fusion.csv",
    "results/vcoco_v2/factorized_fusion_per_class.csv",
    "results/vcoco_v2/factorized_fusion_uncertainty.json",
    "results/vcoco_v2/fewshot_curve.csv",
    "results/vcoco_v2/final_selection_lock.json",
    "results/vcoco_v2/mechanism_correlations.csv",
    "results/vcoco_v2/mechanism_error_transitions.csv",
    "results/vcoco_v2/mechanism_strata.csv",
    "results/vcoco_v2/official_test_confusions.json",
    "results/vcoco_v2/official_test_metrics.csv",
    "results/vcoco_v2/official_test_per_class.csv",
    "results/vcoco_v2/official_test_selective_metrics.json",
    "results/vcoco_v2/official_test_strata.csv",
    "results/vcoco_v2/official_test_summary.json",
    "results/vcoco_v2/official_test_uncertainty.json",
    "results/vcoco_v2/protocol_lock.json",
    "results/vcoco_v2/test_access_gate.json",
}
VCOCO_V3_REQUIRED = {
    "docs/DINOV3_ACCESS.md",
    "docs/POLIMI_ITW_S_ACCESS_AND_STORAGE.md",
    "docs/VCOCO_V3_EXECUTION_RUNBOOK.md",
    "docs/VCOCO_V3_EXTERNAL_CUDA_AMENDMENT.md",
    "docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md",
    "docs/VCOCO_V3_RESEARCH_PROTOCOL.md",
    "experiments/okutama_action_protocol.json",
    "experiments/okutama_temporal_grid.json",
    "experiments/vcoco_v3_protocol.json",
    "assets/vcoco_v3_confirmation_comparison.png",
    "assets/vcoco_v3_confirmation_comparison.svg",
    "assets/vcoco_v3_routing_curve.png",
    "assets/vcoco_v3_routing_curve.svg",
    "results/vcoco_v3/README.md",
    "results/vcoco_v3/annotation_summary.json",
    "results/vcoco_v3/confirmation_metrics.csv",
    "results/vcoco_v3/confirmation_per_class.csv",
    "results/vcoco_v3/confirmation_routing_curve.csv",
    "results/vcoco_v3/confirmation_subgroup_deltas.csv",
    "results/vcoco_v3/confirmation_subgroups.csv",
    "results/vcoco_v3/confirmation_summary.json",
    "results/vcoco_v3/confirmation_uncertainty.json",
    "results/vcoco_v3/evidence_manifest.json",
    "results/vcoco_v3/okutama_dataset_summary.json",
    "results/vcoco_v3/protocol_lineage.json",
    "results/vcoco_v3/source_tag_development_metrics.csv",
    "results/vcoco_v3/temporal_crossfit_summary.json",
    "results/vcoco_v3/temporal_development_metrics.csv",
    "tools/export_vcoco_v3_results.py",
    "tools/render_vcoco_v3_figures.py",
}
V3_RELEASE_REQUIRED = {
    "docs/SCIENTIFIC_VALIDATION_PLAN.md",
    "docs/releases/HUMAN_ACTIVITY_STUDY_V3.0.0.md",
    "output/pdf/okutama_cptr_development_v3.0.0.pdf",
    "output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf",
    "release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt",
    "requirements-v3-lock.txt",
    "results/human_activity_study_v3.0.0_manifest.json",
    "tools/build_v3_release_manifest.py",
    "tools/verify_v3_release_archive.py",
}
CPTR_REQUIRED = {
    "docs/OKUTAMA_CPTR_DEVELOPMENT.md",
    "experiments/audit_okutama_cptr_baseline.py",
    "experiments/cache_okutama_cptr_motion.py",
    "experiments/cache_okutama_cptr_parts.py",
    "experiments/cache_okutama_cptr_siglip.py",
    "experiments/crossfit_okutama_cptr.py",
    "experiments/evaluate_okutama_cptr_faithfulness.py",
    "experiments/fit_okutama_cptr_router.py",
    "experiments/okutama_cptr_adaptive_grid.json",
    "experiments/okutama_cptr_crossfit_plan.json",
    "experiments/okutama_cptr_grid.json",
    "experiments/okutama_cptr_protocol.json",
    "experiments/okutama_cptr_stage2_grid.json",
    "experiments/okutama_cptr_stage3_grid.json",
    "experiments/okutama_cptr_stage4_grid.json",
    "experiments/pretrain_okutama_cptr_masked.py",
    "experiments/train_okutama_cptr_candidate.py",
    "experiments/train_okutama_cptr_lora_specialist.py",
    "results/okutama_cptr/README.md",
    "results/okutama_cptr/component_ablation.csv",
    "results/okutama_cptr/development_decision.json",
    "results/okutama_cptr/evidence_manifest.json",
    "results/okutama_cptr/faithfulness_metrics.csv",
    "results/okutama_cptr/faithfulness_summary.json",
    "results/okutama_cptr/fold_seed_metrics.csv",
    "results/okutama_cptr/headline_metrics.csv",
    "results/okutama_cptr/provenance.json",
    "results/okutama_cptr/recording_metrics.csv",
    "results/okutama_cptr/subgroup_metrics.csv",
    "results/okutama_cptr/uncertainty.json",
    "src/hac/cptr.py",
    "src/hac/cptr_features.py",
    "src/hac/cptr_training.py",
    "tests/test_cptr.py",
    "tests/test_okutama_cptr_portable_evidence.py",
    "tools/export_okutama_cptr_results.py",
    "tools/finalize_okutama_cptr_development.py",
    "tools/lock_okutama_cptr_adaptive_grid.py",
    "tools/lock_okutama_cptr_crossfit_plan.py",
    "tools/lock_okutama_cptr_development.py",
    "tools/lock_okutama_cptr_grid.py",
    "tools/lock_okutama_cptr_protocol.py",
    "tools/lock_okutama_cptr_stage2_grid.py",
    "tools/lock_okutama_cptr_stage3_grid.py",
    "tools/lock_okutama_cptr_stage4_grid.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    return parser.parse_args()


def included_files(repository: Path):
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    relative_paths = sorted(
        Path(value.decode("utf-8")) for value in completed.stdout.split(b"\0") if value
    )
    for relative in relative_paths:
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts[:2] == ("data", "images"):
            continue
        path = repository / relative
        if path.is_file():
            yield path, relative


def validate_third_party_media_boundary(repository: Path) -> None:
    for relative in sorted(EXCLUDED_THIRD_PARTY_MEDIA):
        path = repository / PurePosixPath(relative)
        if path.exists():
            raise RuntimeError(f"Third-party qualitative media is publicly present: {relative}")

        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if tracked.returncode == 0:
            raise RuntimeError(f"Third-party qualitative media remains tracked: {relative}")

        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", relative],
            cwd=repository,
            check=False,
        )
        if ignored.returncode != 0:
            raise RuntimeError(f"Third-party qualitative media is not ignored: {relative}")

        attribute = subprocess.run(
            ["git", "check-attr", "export-ignore", "--", relative],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not attribute.endswith(": export-ignore: set"):
            raise RuntimeError(f"Third-party media lacks export-ignore defense: {relative}")


def validate_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if int(notebook.get("nbformat", 0)) != 4:
        raise RuntimeError(f"Unsupported notebook version: {path}")
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    if kernelspec.get("name") != "python3":
        raise RuntimeError(f"Non-portable notebook kernel: {path}")
    notebook_text = path.read_text(encoding="utf-8")
    if path.name == "human_activity_classification.ipynb":
        if "scenario-cluster interval" not in notebook_text:
            raise RuntimeError("Notebook is missing the scenario-level inference qualification")
        if "recording-cluster interval" in notebook_text:
            raise RuntimeError("Notebook contains obsolete recording-cluster terminology")
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        identity = f"{index}\0{cell.get('cell_type')}\0{source}".encode()
        expected_id = hashlib.sha256(identity).hexdigest()[:8]
        if path.name == "human_activity_classification.ipynb" and (
            cell.get("id") != expected_id or "execution" in cell.get("metadata", {})
        ):
            raise RuntimeError("Notebook cell identity or execution metadata is not deterministic")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(f"Notebook contains an error output: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_release_hashes(path: Path) -> set[str]:
    """Return byte-equivalent hashes across historical Git line-ending checkouts."""

    encoded = path.read_bytes()
    candidates = {hashlib.sha256(encoded).hexdigest()}
    if path.suffix.lower() in LEGACY_FINGERPRINT_TEXT_SUFFIXES:
        linefeed = encoded.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        candidates.add(hashlib.sha256(linefeed).hexdigest())
        candidates.add(hashlib.sha256(linefeed.replace(b"\n", b"\r\n")).hexdigest())
    return candidates


def validate_faithfulness(repository: Path, expected_test_ids: set[str]) -> None:
    results = repository / "results"
    with (results / "faithfulness_selection_lock.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "LOCKED_FROM_OOF_BEFORE_FAITHFULNESS_TEST_EVALUATION":
        raise RuntimeError("Invalid attribution-selection lock status")
    if selection.get("test_used_for_selection") is not False:
        raise RuntimeError("Attribution selection is not independent of the test split")
    selected = selection.get("selected", {})
    if set(selected) != {"convnext_small", "dinov2_small"}:
        raise RuntimeError("Attribution selection does not cover both model families")

    cohort = pd.read_csv(results / "faithfulness_oof_selection_cohort.csv", dtype={"image_id": str})
    if (
        len(cohort) != 36
        or cohort["image_id"].duplicated().any()
        or not cohort.groupby("label").size().eq(12).all()
    ):
        raise RuntimeError("Invalid OOF attribution-selection cohort")

    per_image = pd.read_csv(results / "faithfulness_test_per_image.csv", dtype={"image_id": str})
    expected_models = {"convnext_small", "dinov2_small", "probability_blend"}
    if set(per_image["model"]) != expected_models:
        raise RuntimeError("Unexpected model set in faithfulness test evidence")
    for model, rows in per_image.groupby("model"):
        if len(rows) != 43 or rows["image_id"].duplicated().any():
            raise RuntimeError(f"Invalid faithfulness row count for {model}")
        if set(rows["image_id"]) != expected_test_ids:
            raise RuntimeError(f"Faithfulness IDs do not match the test split for {model}")

    summary = pd.read_csv(results / "faithfulness_test_summary.csv")
    if set(summary["model"]) != expected_models or not summary["test_rows"].eq(43).all():
        raise RuntimeError("Faithfulness summary does not cover the fixed test split")
    for name in ("oof_replay_validation.csv", "faithfulness_replay_validation.csv"):
        replay = pd.read_csv(results / name)
        if replay.empty or not replay["passed"].astype(str).str.casefold().eq("true").all():
            raise RuntimeError(f"Probability replay validation failed in {name}")

    checkpoints = pd.read_csv(results / "faithfulness_checkpoint_manifest.csv")
    if len(checkpoints) != 36:
        raise RuntimeError("Faithfulness checkpoint manifest must contain 36 checkpoints")
    for value in checkpoints["artifact_relative_path"].astype(str):
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Invalid relative checkpoint reference: {value}")

    provenance_path = results / "faithfulness_provenance.json"
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("status") != "LOCKED_TEST_EVALUATED_AFTER_OOF_ATTRIBUTION_SELECTION":
        raise RuntimeError("Faithfulness provenance is incomplete")
    release = provenance.get("release_export", {})
    if release.get("status") != "VALIDATED_PATH_SANITIZED_EVIDENCE_PROMOTED":
        raise RuntimeError("Faithfulness release evidence was not validated before export")
    tracked_evidence = release.get("tracked_evidence", {})
    if not isinstance(tracked_evidence, dict) or not tracked_evidence:
        raise RuntimeError("Faithfulness release tracked-evidence inventory is missing")
    forbidden_evidence = {
        f"assets/{name}" for name in EXCLUDED_FAITHFULNESS_MEDIA_SHA256
    }.intersection(tracked_evidence)
    if forbidden_evidence:
        raise RuntimeError(
            f"Third-party galleries were promoted as release evidence: {sorted(forbidden_evidence)}"
        )
    for relative, expected_hash in tracked_evidence.items():
        path = repository / PurePosixPath(relative)
        if not path.is_file() or expected_hash not in legacy_release_hashes(path):
            raise RuntimeError(f"Faithfulness release fingerprint mismatch: {relative}")

    excluded = release.get("excluded_third_party_media", {})
    reason = excluded.get("reason", "")
    if (
        excluded.get("artifacts") != EXCLUDED_FAITHFULNESS_MEDIA_SHA256
        or "COCO/Flickr" not in reason
        or "not redistributed" not in reason
    ):
        raise RuntimeError("Faithfulness third-party media exclusion is incomplete")
    for relative, expected_hash in release.get("implementation_fingerprints", {}).items():
        path = repository / PurePosixPath(relative)
        if not path.is_file() or expected_hash not in legacy_release_hashes(path):
            raise RuntimeError(f"Faithfulness implementation fingerprint mismatch: {relative}")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_polar_release(repository: Path) -> None:
    results = repository / "results"
    pdf_path = repository / "output" / "pdf" / "polar_public_report_v1.0.0.pdf"
    with pdf_path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise RuntimeError("POLAR Study Report is not a PDF")
        handle.seek(max(0, pdf_path.stat().st_size - 1024))
        if b"%%EOF" not in handle.read():
            raise RuntimeError("POLAR Study Report PDF is incomplete")
    evidence = read_json(results / "polar_final_evidence_manifest.json")
    if (
        evidence.get("status") != "LOCKED_POLAR_PORTFOLIO_EVIDENCE"
        or evidence.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR portable-evidence manifest")
    lock_hash = evidence.get("selection_lock_sha256")
    selection_path = results / "polar_final_selection_lock.json"
    if lock_hash not in legacy_release_hashes(selection_path):
        raise RuntimeError("POLAR selection-lock fingerprint mismatch")

    exported = evidence.get("exported_files", {})
    if len(exported) < 20:
        raise RuntimeError("POLAR portable-evidence export is incomplete")
    for name, expected_hash in exported.items():
        if Path(name).name != name:
            raise RuntimeError(f"Invalid POLAR evidence filename: {name}")
        path = results / name
        if not path.is_file() or expected_hash not in legacy_release_hashes(path):
            raise RuntimeError(f"POLAR evidence fingerprint mismatch: {name}")

    summaries = {
        "polar_test_summary.json": "LOCKED_FINAL_TEST_COMPLETE",
        "polar_external_summary.json": "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
        "polar_faithfulness_summary.json": "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
        "polar_fault_summary.json": "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
    }
    loaded = {}
    for name, status in summaries.items():
        payload = read_json(results / name)
        loaded[name] = payload
        if payload.get("status") != status:
            raise RuntimeError(f"Incomplete POLAR result: {name}")
        if payload.get("selection_lock_sha256") != lock_hash:
            raise RuntimeError(f"POLAR result has a different selection lock: {name}")
        if payload.get("test_used_for_selection") not in {None, False}:
            raise RuntimeError(f"POLAR test-selected result cannot be released: {name}")

    audit = read_json(results / "polar_data_audit.json")
    if (
        audit.get("status") != "PRE_SUPERVISED_FIT_DATA_LOCK"
        or audit.get("clean_rows") != 16614
        or audit.get("quarantine_images") != 125
        or audit.get("test_used_for_model_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR pre-fit data audit")

    fits = read_json(results / "polar_final_fit_manifest.json")
    if (
        fits.get("status") != "LOCKED_POLAR_FINAL_FITS_VERIFIED"
        or fits.get("development_rows") != 13285
        or fits.get("test_rows_read") != 0
        or fits.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR final-fit gate")
    expected_neural = {"convnext_small_full", "dinov2_base_top4", "dinov2_small_moderate"}
    expected_probes = {
        "dinov2_base_multilayer_logistic",
        "dinov2_base_multilayer_logistic_3class",
        "dinov2_base_multilayer_rbf",
    }
    if set(fits.get("neural", {})) != expected_neural or any(
        set(item.get("seeds", {})) != {"42", "52", "62"} for item in fits.get("neural", {}).values()
    ):
        raise RuntimeError("POLAR final neural fits are incomplete")
    if set(fits.get("probes", {})) != expected_probes:
        raise RuntimeError("POLAR final probes are incomplete")

    gate = read_json(results / "polar_test_access_gate.json")
    if (
        gate.get("status") != "POLAR_TEST_GATE_OPEN"
        or gate.get("official_test_manifest_open_count") != 1
        or gate.get("test_rows_read") != 3329
        or gate.get("selection_lock_sha256") != lock_hash
    ):
        raise RuntimeError("Invalid POLAR one-time test-access gate")

    test = loaded["polar_test_summary.json"]
    if (
        test.get("primary_candidate") != "locked_ensemble"
        or test.get("primary_candidate_locked_pre_test") is not True
        or test.get("test_rows_read") != 3329
        or test.get("official_test_manifest_open_count") != 1
        or test.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid locked POLAR test summary")
    metrics = pd.read_csv(results / "polar_test_metrics.csv")
    expected_candidates = {
        "locked_ensemble",
        "convnext_small_full",
        "dinov2_small_moderate",
        "dinov2_base_top4",
        "dinov2_base_multilayer_logistic",
        "dinov2_base_multilayer_rbf",
    }
    if (
        len(metrics) != len(expected_candidates)
        or set(metrics["candidate"]) != expected_candidates
        or metrics["candidate"].duplicated().any()
    ):
        raise RuntimeError("Invalid POLAR held-out candidate table")
    primary_row = metrics.loc[metrics["candidate"].eq("locked_ensemble")].iloc[0]
    if abs(float(primary_row["macro_f1"]) - float(test["primary_metrics"]["macro_f1"])) > 1e-12:
        raise RuntimeError("POLAR primary metric disagrees with its summary")
    uncertainty = read_json(results / "polar_test_uncertainty.json")
    if uncertainty.get("locked_ensemble", {}).get("resamples") != 10000:
        raise RuntimeError("POLAR uncertainty does not use the locked resample count")
    paired = uncertainty.get("locked_ensemble_paired_deltas", {})
    if set(paired) != expected_candidates - {"locked_ensemble"} or any(
        float(item["ci_95_low"]) <= 0.0 for item in paired.values()
    ):
        raise RuntimeError("POLAR paired component comparisons are incomplete")

    overlap = read_json(results / "polar_external_overlap_audit.json")
    if (
        overlap.get("status") != "POLAR_VCOCO_CROSS_DATASET_OVERLAP_AUDITED"
        or overlap.get("exact_overlap_pairs") != 0
        or overlap.get("perceptual_candidates") != 0
        or overlap.get("confirmed_source_related_pairs") != 0
        or overlap.get("model_predictions_read") != 0
        or overlap.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR/V-COCO overlap audit")
    external = loaded["polar_external_summary.json"]
    if (
        external.get("image_level_rows") != 3761
        or external.get("person_rows") != 6640
        or external.get("primary_candidate") != "locked_ensemble_collapsed"
        or external.get("primary_candidate_locked_pre_test") is not True
        or external.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid locked V-COCO external evaluation")

    faithfulness = loaded["polar_faithfulness_summary.json"]
    if (
        faithfulness.get("cohort_rows") != 256
        or faithfulness.get("selection_role") != "none"
        or faithfulness.get("parameter_randomization_rows_per_family") != 16
        or faithfulness.get("test_used_for_attribution_selection") is not False
        or faithfulness.get("test_used_for_model_selection") is not False
        or float(faithfulness.get("max_probability_parity_absolute_error", 1.0)) > 0.002
    ):
        raise RuntimeError("Invalid locked POLAR faithfulness summary")
    faith_rows = pd.read_csv(results / "polar_faithfulness_per_image.csv", dtype={"image_id": str})
    faith_families = {"convnext_small_full", "dinov2_base_top4"}
    if set(faith_rows["family"]) != faith_families:
        raise RuntimeError("Unexpected POLAR faithfulness model family")
    cohort_ids = None
    for family, rows in faith_rows.groupby("family"):
        ids = set(rows["image_id"])
        if len(rows) != 256 or len(ids) != 256:
            raise RuntimeError(f"Invalid POLAR faithfulness rows for {family}")
        cohort_ids = ids if cohort_ids is None else cohort_ids
        if ids != cohort_ids:
            raise RuntimeError("POLAR faithfulness families use different cohorts")

    fault = loaded["polar_fault_summary.json"]
    if (
        fault.get("cohort_rows") != 256
        or fault.get("cohort_sha256") != faithfulness.get("cohort_sha256")
        or fault.get("reported_separately_from_faithfulness") is not True
        or fault.get("selection_role") != "none"
        or fault.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid locked POLAR fault audit")
    fault_metrics = pd.read_csv(results / "polar_fault_robustness_metrics.csv")
    aggregate_faults = fault_metrics[fault_metrics["fault_seed"].astype(str).eq("aggregate")]
    if set(aggregate_faults["family"]) != faith_families or set(aggregate_faults["condition"]) != {
        "uint8_input_bit_flip_rate",
        "symmetric_int8_head_weight_bit_flips",
    }:
        raise RuntimeError("POLAR aggregate fault evidence is incomplete")


def validate_pdf(path: Path, label: str) -> None:
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise RuntimeError(f"{label} is not a PDF")
        handle.seek(max(0, path.stat().st_size - 1024))
        if b"%%EOF" not in handle.read():
            raise RuntimeError(f"{label} PDF is incomplete")


def validate_current_pdf(path: Path, *, title: str, text_marker: str) -> None:
    validate_pdf(path, title)
    reader = PdfReader(path)
    if len(reader.pages) < 4:
        raise RuntimeError(f"{title} has too few pages")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    if text_marker not in extracted:
        raise RuntimeError(f"{title} is missing expected report text")
    metadata_title = str((reader.metadata or {}).get("/Title", ""))
    if metadata_title != title:
        raise RuntimeError(f"{title} PDF metadata is stale: {metadata_title!r}")


def validate_study_report_release(repository: Path) -> None:
    v1_report = repository / "docs/POLAR_PUBLIC_REPORT.md"
    v1_text = v1_report.read_text(encoding="utf-8")
    if "version: 1.0.0" not in v1_text or "status: Independent technical report" not in v1_text:
        raise RuntimeError("POLAR Study Report metadata is not final v1.0.0")

    v2_report = repository / "docs/VCOCO_V2_EXTERNAL_TRANSFER.md"
    v2_text = v2_report.read_text(encoding="utf-8")
    for marker in (
        "date: 24 August 2026",
        "version: 2.0.0",
        "status: Independent technical report",
        "0.8663 official-test macro-F1",
        "+0.1592 macro-F1",
    ):
        if marker not in v2_text:
            raise RuntimeError(f"V-COCO v2 report is missing: {marker}")

    release_narratives = (
        "README.md",
        "CHANGELOG.md",
        "docs/POLAR_PUBLIC_REPORT.md",
        "docs/OKUTAMA_CPTR_DEVELOPMENT.md",
        "docs/PORTFOLIO_ARTICLE.md",
        "docs/RESULT_LINEAGE.md",
        "docs/SCIENTIFIC_VALIDATION_PLAN.md",
        "docs/VCOCO_V2_EXTERNAL_TRANSFER.md",
        "docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md",
        "docs/releases/POLAR_STUDY_V1.0.0.md",
        "docs/releases/POLAR_STUDY_V2.0.0.md",
        "docs/releases/HUMAN_ACTIVITY_STUDY_V3.0.0.md",
        "experiments/README.md",
        "output/pdf/README.md",
        "results/README.md",
        "results/vcoco_v2/README.md",
        "results/vcoco_v3/README.md",
        "results/okutama_cptr/README.md",
    )
    stale_markers = (
        "0.9.0-review",
        "pre-release review copy",
        "review status:",
    )
    for relative in release_narratives:
        text = (repository / relative).read_text(encoding="utf-8")
        lowered = text.lower()
        for marker in stale_markers:
            if marker in lowered:
                raise RuntimeError(f"Release narrative contains '{marker}': {relative}")
        if any(ord(character) >= 128 for character in text):
            raise RuntimeError(f"Release narrative must remain ASCII-safe: {relative}")

    validate_pdf(
        repository / "output/pdf/polar_public_report_v1.0.0.pdf",
        "POLAR Study Report v1.0.0",
    )
    validate_pdf(
        repository / "output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf",
        "V-COCO Study Report v2.0.0",
    )
    validate_current_pdf(
        repository / "output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf",
        title="When a Still Image Is Not Enough",
        text_marker="0.7854 macro-F1",
    )
    validate_current_pdf(
        repository / "output/pdf/okutama_cptr_development_v3.0.0.pdf",
        title="Camera-Compensated Part-Trajectory Residual Development",
        text_marker="0.7144 for the new component",
    )

    zenodo = read_json(repository / ".zenodo.json")
    zenodo_keywords = set(zenodo.get("keywords", []))
    if (
        zenodo.get("title") != "Human Activity Classification Under Domain and Temporal Shift"
        or zenodo.get("version") != "3.0.0"
        or "publication_date" in zenodo
        or zenodo.get("access_right") != "open"
        or zenodo.get("license") != "mit"
        or zenodo.get("upload_type") != "software"
        or zenodo.get("creators") != [{"name": "Huseyinli, Abdulla"}]
        or not {"POLAR", "V-COCO", "Okutama-Action", "DINOv3"}.issubset(zenodo_keywords)
    ):
        raise RuntimeError("Zenodo metadata does not match the v3.0.0 release candidate")

    citation = (repository / "CITATION.cff").read_text(encoding="utf-8")
    for marker in (
        'title: "Human Activity Classification Under Domain and Temporal Shift"',
        'version: "3.0.0"',
        'message: "If you use this software or its evidence, cite this versioned record."',
    ):
        if marker not in citation:
            raise RuntimeError(f"Citation metadata is missing: {marker}")
    if "preferred-citation:" in citation or "date-released:" in citation:
        raise RuntimeError("Candidate citation metadata contains premature release fields")

    validation_plan = (repository / "docs/SCIENTIFIC_VALIDATION_PLAN.md").read_text(
        encoding="utf-8"
    )
    for marker in (
        "## Current claim boundary",
        "## Limitation and evidence-gate matrix",
        "## 1. Annotation reliability",
        "## 3. Independent replication on POLIMI-ITW-S",
        "## 4. Grouped inference and prospective precision",
        "## 5. Matched model comparison",
        "## 6. CPTR occlusion study",
        "## 7. Runtime, energy, and memory",
        "## 8. Reproducibility check",
        "## 9. Multiplicity and analysis discipline",
        "## Publication decision rule",
        "The sealed temporal result has five confirmation scenarios",
        "These are evidence-completeness gates, not success gates.",
    ):
        if marker not in validation_plan:
            raise RuntimeError(f"Scientific validation plan is missing: {marker}")

    v3_notes = (repository / "docs/releases/HUMAN_ACTIVITY_STUDY_V3.0.0.md").read_text(
        encoding="utf-8"
    )
    if "Release tag:" in v3_notes:
        raise RuntimeError("Release candidate notes claim a tag that does not exist")

    exploratory = read_json(repository / "results/polar_exploratory_summary.json")
    if (
        exploratory.get("analysis_seed") != 20260823
        or exploratory.get("bootstrap_resamples") != 10000
        or not str(exploratory.get("analysis_role", "")).startswith("Hypothesis-generating only")
    ):
        raise RuntimeError("Exploratory evidence role or deterministic settings changed")

    v1_manifest_path = repository / "results/polar_study_v1.0.0_manifest.json"
    v1_manifest = read_json(v1_manifest_path)
    if (
        v1_manifest.get("release_id") != "polar-study-v1.0.0"
        or v1_manifest.get("report_version") != "1.0.0"
        or not isinstance(v1_manifest.get("artifacts"), dict)
    ):
        raise RuntimeError("Historical v1.0.0 release manifest is invalid")
    v1_pdf_path = repository / "output/pdf/polar_public_report_v1.0.0.pdf"
    expected_v1_checksums = (
        f"{sha256_file(v1_pdf_path)}  output/pdf/polar_public_report_v1.0.0.pdf\n"
        f"{sha256_file(v1_manifest_path)}  results/polar_study_v1.0.0_manifest.json\n"
    )
    v1_checksums_path = repository / "release/POLAR_STUDY_V1.0.0_SHA256SUMS.txt"
    if v1_checksums_path.read_text(encoding="utf-8") != expected_v1_checksums:
        raise RuntimeError("Historical v1.0.0 release checksums changed")

    sys.path.insert(0, str(repository / "tools"))
    from build_readme import build_readme
    from build_study_release_manifest import checksum_text as historical_checksum_text
    from build_v3_release_manifest import verify_frozen_release as verify_v3_frozen_release

    if (repository / "README.md").read_text(encoding="utf-8") != build_readme(repository):
        raise RuntimeError("README is stale relative to locked evidence")

    manifest_path = repository / "results/polar_study_v2.0.0_manifest.json"
    release_manifest = read_json(manifest_path)
    if (
        release_manifest.get("release_id") != "polar-study-v2.0.0"
        or release_manifest.get("report_version") != "2.0.0"
    ):
        raise RuntimeError("Study v2.0.0 release manifest identity changed")
    for relative, evidence in release_manifest.get("artifacts", {}).items():
        blob = subprocess.run(
            ["git", "show", f"polar-study-v2.0.0:{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != evidence.get("sha256"):
            raise RuntimeError(f"Tagged v2.0.0 artifact hash differs: {relative}")
        if len(blob) != int(evidence.get("size_bytes", -1)):
            raise RuntimeError(f"Tagged v2.0.0 artifact size differs: {relative}")
    checksums_path = repository / "release/POLAR_STUDY_V2.0.0_SHA256SUMS.txt"
    if checksums_path.read_text(encoding="utf-8") != historical_checksum_text(
        repository, manifest_path
    ):
        raise RuntimeError("Study v2.0.0 release checksums are stale")

    v3_manifest_path = repository / "results/human_activity_study_v3.0.0_manifest.json"
    v3_checksums_path = repository / "release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt"
    verify_v3_frozen_release(repository, v3_manifest_path, v3_checksums_path)

    notebook_text = (repository / "human_activity_classification.ipynb").read_text(encoding="utf-8")
    for marker in (
        "0.8663 official-test macro-F1",
        "## 5. Person-level V-COCO study",
        "## 6. Motion identifiability on Okutama-Action",
        "## 7. CPTR architecture development",
        "0.7854",
        "0.7144",
    ):
        if marker not in notebook_text:
            raise RuntimeError(f"Executed notebook is missing: {marker}")


def validate_vcoco_v2_release(repository: Path) -> None:
    results = repository / "results" / "vcoco_v2"
    manifest = read_json(results / "evidence_manifest.json")
    protocol = read_json(results / "protocol_lock.json")
    selection = read_json(results / "final_selection_lock.json")
    summary = read_json(results / "official_test_summary.json")
    gate = read_json(results / "test_access_gate.json")
    protocol_hash = protocol.get("source_lock_sha256")
    if manifest.get("status") != "VCOCO_V2_PORTABLE_EVIDENCE_COMPLETE":
        raise RuntimeError("V-COCO v2 evidence export is incomplete")
    if manifest.get("exporter_sha256") != sha256_file(
        repository / "tools" / "export_vcoco_v2_results.py"
    ):
        raise RuntimeError("V-COCO v2 exporter changed after evidence generation")
    if protocol.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("V-COCO v2 protocol lock is invalid")
    if selection.get("status") != "VCOCO_V2_FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("V-COCO v2 selection lock is invalid")
    if summary.get("status") != "VCOCO_V2_OFFICIAL_TEST_EVALUATION_COMPLETE":
        raise RuntimeError("V-COCO v2 official test result is incomplete")
    if gate.get("status") != "VCOCO_V2_OFFICIAL_TEST_GATE_OPEN":
        raise RuntimeError("V-COCO v2 test gate is invalid")
    if (
        not isinstance(protocol_hash, str)
        or manifest.get("protocol_lock_sha256") != protocol_hash
        or selection.get("protocol_lock_sha256") != protocol_hash
        or summary.get("protocol_lock_sha256") != protocol_hash
    ):
        raise RuntimeError("V-COCO v2 protocol lineage does not align")
    selection_hash = manifest.get("selection_lock_sha256")
    if (
        selection.get("source_lock_sha256") != selection_hash
        or summary.get("selection_lock_sha256") != selection_hash
        or gate.get("selection_lock_sha256") != selection_hash
    ):
        raise RuntimeError("V-COCO v2 selection lineage does not align")
    if (
        manifest.get("official_test_label_open_count") != 1
        or summary.get("official_test_label_open_count") != 1
        or gate.get("official_test_label_open_count") != 1
        or manifest.get("test_used_for_selection") is not False
        or summary.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("V-COCO v2 official test access contract changed")

    artifact_names = {
        path.name
        for path in results.iterdir()
        if path.is_file() and path.name != "evidence_manifest.json"
    }
    if set(manifest.get("artifacts", {})) != artifact_names:
        raise RuntimeError("V-COCO v2 evidence inventory is incomplete")
    for name, record in manifest["artifacts"].items():
        path = results / name
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get(
            "size_bytes"
        ):
            raise RuntimeError(f"V-COCO v2 evidence artifact drift: {name}")

    metrics = pd.read_csv(results / "official_test_metrics.csv").set_index("method")
    if set(metrics.index) != {"scale_conditioned_stacking", "historical_v1_dino"}:
        raise RuntimeError("Unexpected V-COCO v2 official test candidates")
    champion = float(metrics.loc["scale_conditioned_stacking", "macro_f1"])
    baseline = float(metrics.loc["historical_v1_dino", "macro_f1"])
    uncertainty = read_json(results / "official_test_uncertainty.json")
    if (
        abs(champion - float(summary["primary_metrics"]["macro_f1"])) > 1e-12
        or abs(baseline - float(summary["baseline_metrics"]["macro_f1"])) > 1e-12
        or abs((champion - baseline) - float(uncertainty["point_estimate"])) > 1e-12
        or float(uncertainty["point_estimate"]) < 0.01
        or float(uncertainty["ci_95_low"]) <= 0.0
        or uncertainty.get("resamples") != 10000
        or uncertainty.get("clusters") != 3708
        or summary.get("confirmatory_success") is not True
    ):
        raise RuntimeError("V-COCO v2 confirmatory result changed")
    per_class = pd.read_csv(results / "official_test_per_class.csv")
    if set(per_class["method"]) != set(metrics.index) or set(per_class["class"]) != {
        "sitting",
        "standing",
        "walking_running",
    }:
        raise RuntimeError("V-COCO v2 per-class evidence is incomplete")

    report = (repository / "docs" / "VCOCO_V2_EXTERNAL_TRANSFER.md").read_text(encoding="utf-8")
    for marker in (
        "version: 2.0.0",
        "status: Independent technical report",
        "0.8663 official-test macro-F1",
        "+0.1592 macro-F1",
    ):
        if marker not in report:
            raise RuntimeError(f"V-COCO v2 report is missing: {marker}")


def validate_vcoco_v3_evidence(repository: Path) -> None:
    results = repository / "results" / "vcoco_v3"
    manifest = read_json(results / "evidence_manifest.json")
    summary = read_json(results / "confirmation_summary.json")
    lineage = read_json(results / "protocol_lineage.json")
    annotation = read_json(results / "annotation_summary.json")
    dataset = read_json(results / "okutama_dataset_summary.json")

    if manifest.get("status") != "VCOCO_V3_PORTABLE_EVIDENCE_COMPLETE":
        raise RuntimeError("V-COCO v3 evidence export is incomplete")
    if manifest.get("exporter_sha256") != sha256_file(
        repository / "tools" / "export_vcoco_v3_results.py"
    ):
        raise RuntimeError("V-COCO v3 exporter changed after evidence generation")
    if summary.get("status") != "VCOCO_V3_TEMPORAL_CONFIRMATION_COMPLETE":
        raise RuntimeError("V-COCO v3 confirmation is incomplete")
    if lineage.get("status") != "VCOCO_V3_PORTABLE_PROTOCOL_LINEAGE_COMPLETE":
        raise RuntimeError("V-COCO v3 protocol lineage is incomplete")
    if (
        manifest.get("confirmation_open_number") != 1
        or summary.get("confirmation_open_number") != 1
        or lineage.get("confirmation_open_number") != 1
        or manifest.get("confirmation_used_for_selection") is not False
        or lineage.get("confirmation_used_for_selection") is not False
    ):
        raise RuntimeError("V-COCO v3 confirmation access contract changed")
    if {
        manifest.get("pipeline_lock_sha256"),
        summary.get("pipeline_lock_sha256"),
        lineage.get("pipeline_lock_sha256"),
    } != {"9c2ff4715b87b9ee8854caa31f4f15ebf77306201911d839533aa3aab3be4068"}:
        raise RuntimeError("V-COCO v3 pipeline lineage does not align")

    artifact_names = {
        path.name
        for path in results.iterdir()
        if path.is_file() and path.name != "evidence_manifest.json"
    }
    if set(manifest.get("artifacts", {})) != artifact_names:
        raise RuntimeError("V-COCO v3 evidence inventory is incomplete")
    for name, record in manifest["artifacts"].items():
        path = results / name
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get(
            "size_bytes"
        ):
            raise RuntimeError(f"V-COCO v3 evidence artifact drift: {name}")

    cuda = lineage.get("cuda", {})
    if (
        cuda.get("available") is not True
        or cuda.get("cpu_fallback_permitted") is not False
        or cuda.get("device") != "NVIDIA GeForce RTX 4060 Laptop GPU"
        or cuda.get("torch_version") != "2.11.0+cu128"
        or lineage.get("verified_cuda_temporal_runs") != 100
        or lineage.get("spatial_cuda_logistic_fits") != 25380
        or lineage.get("representation_cuda_logistic_fits") != 6750
    ):
        raise RuntimeError("V-COCO v3 CUDA provenance changed")

    expected_families = {
        "static",
        "teacher",
        "classification_student",
        "routing_student",
        "source_only_static",
        "hybrid_budget_0",
        "hybrid_budget_0.1",
        "hybrid_budget_0.25",
        "hybrid_budget_0.5",
        "hybrid_budget_1",
    }
    metrics = pd.read_csv(results / "confirmation_metrics.csv").set_index("family")
    if set(metrics.index) != expected_families or metrics.index.duplicated().any():
        raise RuntimeError("Unexpected V-COCO v3 confirmation families")
    expected_macro = {
        "source_only_static": 0.5734824448414355,
        "static": 0.7457893380602498,
        "classification_student": 0.7456372774713865,
        "hybrid_budget_0.5": 0.7817431178785431,
        "teacher": 0.7854392126392127,
    }
    for family, expected in expected_macro.items():
        if abs(float(metrics.loc[family, "macro_f1"]) - expected) > 1e-12:
            raise RuntimeError(f"V-COCO v3 macro-F1 changed for {family}")
    if (
        abs(float(metrics.loc["hybrid_budget_1", "macro_f1"]) - expected_macro["teacher"]) > 1e-12
        or abs(
            float(metrics.loc["hybrid_budget_0", "macro_f1"])
            - expected_macro["classification_student"]
        )
        > 1e-12
    ):
        raise RuntimeError("V-COCO v3 routing endpoints do not match their models")

    uncertainty = read_json(results / "confirmation_uncertainty.json")
    if set(uncertainty) != {
        "classification_student",
        "hybrid_budget_0.1",
        "hybrid_budget_0.25",
        "hybrid_budget_0.5",
        "source_only_static",
        "teacher",
    }:
        raise RuntimeError("V-COCO v3 paired uncertainty is incomplete")
    teacher = uncertainty["teacher"]
    half = uncertainty["hybrid_budget_0.5"]
    student = uncertainty["classification_student"]
    if (
        teacher.get("resamples") != 10000
        or teacher.get("clusters") != 5
        or float(teacher["macro_f1"]["ci_95_low"]) <= 0.0
        or float(half["macro_f1"]["ci_95_low"]) <= 0.0
        or float(teacher["holm_adjusted_macro_p"]) >= 0.01
        or float(half["holm_adjusted_macro_p"]) >= 0.01
        or not (
            float(student["macro_f1"]["ci_95_low"])
            <= 0.0
            <= float(student["macro_f1"]["ci_95_high"])
        )
    ):
        raise RuntimeError("V-COCO v3 confirmation uncertainty changed")

    routing = pd.read_csv(results / "confirmation_routing_curve.csv")
    if (
        list(routing["requested_clip_fraction"]) != [0.0, 0.1, 0.25, 0.5, 1.0]
        or not routing["macro_f1"].is_monotonic_increasing
    ):
        raise RuntimeError("V-COCO v3 fixed-budget routing curve changed")
    per_class = pd.read_csv(results / "confirmation_per_class.csv")
    if set(per_class["family"]) != expected_families or set(per_class["class"]) != {
        "sitting",
        "standing",
        "walking_running",
    }:
        raise RuntimeError("V-COCO v3 per-class evidence is incomplete")

    deltas = pd.read_csv(results / "confirmation_subgroup_deltas.csv")
    scenarios = deltas[deltas["family"].eq("teacher") & deltas["axis"].eq("scenario")]
    if (
        len(scenarios) != 5
        or int((scenarios["macro_f1_delta_vs_static"] > 0.0).sum()) != 4
        or float(
            scenarios.loc[
                scenarios["value"].astype(str).eq("1.9"), "macro_f1_delta_vs_static"
            ].iloc[0]
        )
        >= 0.0
    ):
        raise RuntimeError("V-COCO v3 scenario evidence changed")

    if (
        annotation.get("status") != "VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE"
        or annotation.get("primary_task_presentations") != 130
        or annotation.get("unique_content_rows") != 126
        or annotation.get("human_pilot_labels_used_for_candidate_selection") is not False
        or annotation.get("interrater_agreement_available") is not False
    ):
        raise RuntimeError("V-COCO v3 annotation audit changed")
    if (
        dataset.get("development", {}).get("samples") != 8339
        or dataset.get("confirmation", {}).get("samples") != 1771
        or dataset.get("confirmation", {}).get("scenarios") != 5
        or dataset.get("confirmation", {}).get("open_number") != 1
    ):
        raise RuntimeError("Okutama dataset evidence changed")

    source_metrics = pd.read_csv(results / "source_tag_development_metrics.csv").set_index(
        ["stage", "family"]
    )
    dino2 = float(source_metrics.loc[("representations", "dinov2_base"), "macro_f1"])
    dino3 = float(source_metrics.loc[("representations", "dinov3_base"), "macro_f1"])
    nested = float(
        source_metrics.loc[
            ("nested_stacks", "dino_siglip_factorized_reliability_stack"), "macro_f1"
        ]
    )
    if not (dino2 > dino3 and abs(nested - 0.8696815257533729) <= 1e-12):
        raise RuntimeError("V-COCO v3 source-tag development evidence changed")

    report = (repository / "docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md").read_text(encoding="utf-8")
    for marker in (
        "status: Independent technical report",
        "0.7854 macro-F1",
        "0.0396 [0.0202, 0.0568]",
        "All 100 temporal",
        "# 8. Limitations and next measurements",
    ):
        if marker not in report:
            raise RuntimeError(f"V-COCO v3 report is missing: {marker}")
    runbook = (repository / "docs/VCOCO_V3_EXECUTION_RUNBOOK.md").read_text(encoding="utf-8")
    if "features\\dinov3_base" in runbook or "--model-kind dinov3_base" in runbook:
        raise RuntimeError("V-COCO v3 runbook still replays the rejected backbone")


def validate_okutama_cptr_evidence(repository: Path) -> None:
    results = repository / "results" / "okutama_cptr"
    manifest = read_json(results / "evidence_manifest.json")
    decision = read_json(results / "development_decision.json")
    provenance = read_json(results / "provenance.json")
    faithfulness = read_json(results / "faithfulness_summary.json")
    uncertainty = read_json(results / "uncertainty.json")
    if (
        manifest.get("study") != "okutama_cptr_development"
        or decision.get("status") != "OKUTAMA_CPTR_DEVELOPMENT_LOCKED_NO_PROMOTION"
        or decision.get("promotion_passed") is not False
        or provenance.get("status") != "OKUTAMA_CPTR_PORTABLE_EVIDENCE_COMPLETE"
        or faithfulness.get("status") != "OKUTAMA_CPTR_FAITHFULNESS_COMPLETE"
    ):
        raise RuntimeError("CPTR portable evidence is incomplete")
    if provenance.get("source_sha256", {}).get("exporter") != sha256_file(
        repository / "tools" / "export_okutama_cptr_results.py"
    ):
        raise RuntimeError("CPTR exporter changed after evidence generation")
    if any(
        int(payload.get("calibration_samples_read", -1)) != 0
        or int(payload.get("confirmation_samples_read", -1)) != 0
        for payload in (decision, faithfulness)
    ):
        raise RuntimeError("Closed CPTR evaluation data entered development evidence")

    artifact_names = {
        path.name
        for path in results.iterdir()
        if path.is_file() and path.name != "evidence_manifest.json"
    }
    if (
        int(manifest.get("artifact_count", -1)) != len(artifact_names)
        or set(manifest.get("artifacts", {})) != artifact_names
    ):
        raise RuntimeError("CPTR portable evidence inventory is incomplete")
    for name, record in manifest["artifacts"].items():
        path = results / name
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get(
            "size_bytes"
        ):
            raise RuntimeError(f"CPTR portable artifact drift: {name}")

    metrics = pd.read_csv(results / "headline_metrics.csv").set_index(["scope", "model"])
    expected = {
        ("development_validation", "v3_temporal_baseline"): 0.7806159956588115,
        ("development_validation", "centre_short_parts"): 0.7886666661527836,
        ("grouped_crossfit_oof", "v3_temporal_baseline"): 0.7164832120716224,
        ("grouped_crossfit_oof", "centre_short_parts"): 0.7144451499106025,
    }
    if set(metrics.index) != set(expected):
        raise RuntimeError("Unexpected CPTR headline model scopes")
    for key, value in expected.items():
        if abs(float(metrics.loc[key, "macro_f1"]) - value) > 1e-12:
            raise RuntimeError(f"CPTR headline metric changed: {key}")

    oof_uncertainty = uncertainty.get("crossfit_oof_cluster_bootstrap", {})
    exact = uncertainty.get("crossfit_oof_exact_group_swap", {})
    if (
        oof_uncertainty.get("resamples") != 10000
        or oof_uncertainty.get("clusters") != 11
        or not (
            float(oof_uncertainty["macro_f1"]["ci_95_low"])
            < 0.0
            < float(oof_uncertainty["macro_f1"]["ci_95_high"])
        )
        or exact.get("permutations") != 2048
        or exact.get("recordings") != 11
    ):
        raise RuntimeError("CPTR grouped uncertainty changed")

    components = pd.read_csv(results / "component_ablation.csv")
    expected_components = {
        "raw_trajectory",
        "camera_compensated_trajectory",
        "centre_short",
        "dual_clock",
        "dual_clock_specialized",
        "centre_short_trajectory",
        "centre_short_parts",
        "integrated_cptr",
        "counterfactual_original",
        "counterfactual_refined",
        "masked_initialisation",
        "siglip_posture_specialist",
        "group_dro",
        "top_block_lora",
    }
    if set(components["component_id"]) != expected_components:
        raise RuntimeError("CPTR component evidence is incomplete")
    lineage_columns = {
        "run_status",
        "summary_sha256",
        "request_payload_sha256",
        "request_file_sha256",
        "model_module_sha256",
        "neural_module_sha256",
        "feature_module_sha256",
        "training_module_sha256",
        "runner_sha256",
        "grid_sha256",
        "grid_lock_sha256",
    }
    if not lineage_columns.issubset(components.columns):
        raise RuntimeError("CPTR component source lineage is incomplete")
    if (
        components[["summary_sha256", "request_payload_sha256", "request_file_sha256"]]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError("CPTR component request lineage contains missing hashes")
    if components["model_module_sha256"].dropna().nunique() < 2:
        raise RuntimeError("CPTR sequential source revisions are no longer visible")
    contract = provenance.get("crossfit_contract_audit", {})
    if (
        contract.get("status") != "VERIFIED_WITH_HISTORICAL_SCHEMA_LIMITATIONS"
        or contract.get("requests_verified") != 25
        or contract.get("candidate_grid_matches_adaptive_lock") is not True
        or contract.get("historical_request_schema_omissions") != ["src/hac/cptr_training.py"]
    ):
        raise RuntimeError("CPTR cross-fit contract audit is incomplete")
    fold_seed = pd.read_csv(results / "fold_seed_metrics.csv", dtype=str)
    individual = fold_seed[
        fold_seed["fold"].isin({"0", "1", "2", "3", "4"})
        & fold_seed["seed"].isin({"42", "43", "44", "45", "46"})
    ]
    if len(individual) != 25 or len(fold_seed[fold_seed["seed"].eq("ensemble")]) != 5:
        raise RuntimeError("CPTR fold/seed evidence is incomplete")

    subgroups = pd.read_csv(results / "subgroup_metrics.csv").set_index(["scope", "subgroup"])
    if (
        float(subgroups.loc[("crossfit_oof", "window_occluded"), "macro_f1_delta"]) >= 0.0
        or float(subgroups.loc[("crossfit_oof", "window_clear"), "macro_f1_delta"]) <= 0.0
    ):
        raise RuntimeError("CPTR visibility failure evidence changed")
    motion_null = faithfulness["diagnostics"]["motion_null"]
    if (
        float(motion_null["macro_f1_delta_real_minus_intervention"]) < 0.04
        or float(motion_null["mean_true_class_log_probability_gain_real_minus_intervention"]) < 0.10
    ):
        raise RuntimeError("CPTR temporal faithfulness evidence changed")

    report = (repository / "docs" / "OKUTAMA_CPTR_DEVELOPMENT.md").read_text(encoding="utf-8")
    for marker in (
        "0.7887 on the fixed validation split",
        "0.7144 for the new component and 0.7165",
        "2,048-permutation recording-swap test",
        "Calibration remains unopened",
        "## 7. Limitations and validation requirements",
        "SCIENTIFIC_VALIDATION_PLAN.md",
    ):
        if marker not in report:
            raise RuntimeError(f"CPTR development report is missing: {marker}")


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    missing = sorted(
        name
        for name in REQUIRED
        | POLAR_REQUIRED
        | VCOCO_V2_REQUIRED
        | VCOCO_V3_REQUIRED
        | V3_RELEASE_REQUIRED
        | CPTR_REQUIRED
        if not (repository / name).is_file()
    )
    if missing:
        raise RuntimeError(f"Missing release files: {missing}")

    license_text = (repository / "LICENSE").read_text(encoding="utf-8")
    for marker in (
        "MIT License",
        "Copyright (c) 2026 Abdulla Huseyinli",
        "Permission is hereby granted, free of charge",
        'THE SOFTWARE IS PROVIDED "AS IS"',
    ):
        if marker not in license_text:
            raise RuntimeError(f"LICENSE is missing: {marker}")

    notices = (repository / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for marker in (
        "cocodataset.org/#termsofuse",
        "facebookresearch/dinov2",
        "facebookresearch/dinov3",
        "Barekatain_Okutama-Action",
        "google/siglip-base-patch16-224",
        "google/siglip2-base-patch16-224",
        "pytorch/vision",
        "jacobgil/pytorch-grad-cam",
    ):
        if marker not in notices:
            raise RuntimeError(f"Third-party notice is missing: {marker}")

    validate_third_party_media_boundary(repository)

    windows_user_path = re.compile(r"[A-Za-z]:[\\/]+" + "Users" + r"[\\/]", flags=re.IGNORECASE)
    local_file_scheme = "file" + "://"
    oversized = []
    for path, relative in included_files(repository):
        if path.stat().st_size > 10 * 1024 * 1024:
            oversized.append((str(relative), path.stat().st_size))
        if path.name in {".gitignore", ".gitattributes"} or path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            invalid_control = next(
                (
                    character
                    for character in text
                    if ord(character) < 32 and character not in "\n\r\t"
                ),
                None,
            )
            if invalid_control is not None:
                raise RuntimeError(
                    f"Control character U+{ord(invalid_control):04X} leaked into {relative}"
                )
            if windows_user_path.search(text) or local_file_scheme in text.lower():
                raise RuntimeError(f"Local absolute path leaked into {relative}")
            if "\ufffd" in text:
                raise RuntimeError(f"Unicode replacement character leaked into {relative}")
        if path.suffix.lower() == ".ipynb":
            validate_notebook(path)
    if oversized:
        raise RuntimeError(f"Files exceed the 10 MiB portfolio limit: {oversized}")

    with (repository / "results" / "selection_lock.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "LOCKED_BEFORE_FINAL_TEST":
        raise RuntimeError("Invalid model-selection lock status")
    with (repository / "results" / "downstream_selection_lock.json").open(
        encoding="utf-8"
    ) as handle:
        downstream = json.load(handle)
    if downstream.get("status") != "LOCKED_FROM_OOF_BEFORE_DOWNSTREAM_TEST_EVALUATION":
        raise RuntimeError("Invalid downstream-selection lock status")

    sys.path.insert(0, str(repository / "src"))
    from hac.protocol import load_and_validate_manifest

    manifest, protocol = load_and_validate_manifest(
        repository / "data" / "manifest.csv", require_images=False
    )
    if protocol.development_rows != 242 or protocol.test_rows != 43:
        raise RuntimeError("Tracked manifest no longer matches the fixed protocol")
    expected_test_ids = set(manifest.loc[manifest["split"].eq("test"), "image_id"].astype(str))
    validate_faithfulness(repository, expected_test_ids)
    validate_polar_release(repository)
    validate_study_report_release(repository)
    validate_vcoco_v2_release(repository)
    validate_vcoco_v3_evidence(repository)
    validate_okutama_cptr_evidence(repository)
    print("Repository validation passed")


if __name__ == "__main__":
    main()
