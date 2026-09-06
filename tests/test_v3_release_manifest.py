import hashlib
import subprocess
from pathlib import Path

import pytest

from tools.build_v3_release_manifest import (
    EXCLUDED_THIRD_PARTY_MEDIA,
    PREPARED_DATE,
    RELEASE_ID,
    REPORT_VERSION,
    SOFTWARE_VERSION,
    artifact_bytes,
    checksum_text,
    encoded_manifest,
    verify_frozen_release,
)

FROZEN_PDFS = (
    "output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf",
    "output/pdf/okutama_cptr_development_v3.0.0.pdf",
)
MANIFEST_RELATIVE = "results/human_activity_study_v3.0.0_manifest.json"
CHECKSUMS_RELATIVE = "release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt"


def run_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def frozen_release_fixture(
    tmp_path: Path,
    *,
    identity_overrides: dict[str, object] | None = None,
    artifact_count_override: int | None = None,
    omit_artifact: str | None = None,
    evidence_overrides: dict[str, dict[str, object]] | None = None,
    extra_artifacts: dict[str, dict[str, object]] | None = None,
    checksum_override: str | None = None,
) -> tuple[Path, Path, Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    payloads = {
        "artifact.bin": b"release evidence\x00",
        FROZEN_PDFS[0]: b"temporal-pdf",
        FROZEN_PDFS[1]: b"cptr-pdf",
    }
    for relative, payload in payloads.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    artifacts = {
        relative: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for relative, payload in payloads.items()
        if relative != omit_artifact
    }
    for relative, override in (evidence_overrides or {}).items():
        artifacts[relative].update(override)
    artifacts.update(extra_artifacts or {})
    manifest: dict[str, object] = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "report_version": REPORT_VERSION,
        "software_version": SOFTWARE_VERSION,
        "prepared_date": PREPARED_DATE,
        "artifact_count": (
            len(artifacts) if artifact_count_override is None else artifact_count_override
        ),
        "artifacts": artifacts,
    }
    manifest.update(identity_overrides or {})
    manifest_path = repository / MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(encoded_manifest(manifest))
    checksums_path = repository / CHECKSUMS_RELATIVE
    checksums_path.parent.mkdir(parents=True)
    checksums_path.write_text(
        checksum_text(repository, manifest_path)
        if checksum_override is None
        else checksum_override,
        encoding="utf-8",
        newline="\n",
    )

    run_git(repository, "init", "--quiet")
    run_git(repository, "config", "user.name", "Release Test")
    run_git(repository, "config", "user.email", "release-test@example.invalid")
    run_git(repository, "config", "core.autocrlf", "false")
    run_git(repository, "config", "commit.gpgsign", "false")
    run_git(repository, "config", "core.hooksPath", ".git/disabled-hooks")
    run_git(repository, "add", ".")
    run_git(repository, "commit", "--quiet", "-m", "frozen release")
    revision = run_git(repository, "rev-parse", "HEAD")
    return repository, manifest_path, checksums_path, revision


@pytest.mark.parametrize(
    "filename",
    [
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "LICENSE",
        "module.py",
    ],
)
def test_text_artifacts_use_git_normalized_line_endings(tmp_path: Path, filename: str) -> None:
    artifact = tmp_path / filename
    artifact.write_bytes(b"first\r\nsecond\r\n")

    assert artifact_bytes(artifact) == b"first\nsecond\n"


def test_binary_artifacts_retain_original_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.png"
    payload = b"binary\r\npayload\x00"
    artifact.write_bytes(payload)

    assert artifact_bytes(artifact) == payload


def test_non_distributed_media_contract_matches_the_release_tree() -> None:
    repository = Path(__file__).resolve().parents[1]
    expected = {
        "assets/champion_error_gallery.png",
        "assets/convnext_small_faithfulness_gallery.jpg",
        "assets/dinov2_small_faithfulness_gallery.jpg",
        "assets/probability_blend_faithfulness_gallery.jpg",
    }

    assert {path.as_posix() for path in EXCLUDED_THIRD_PARTY_MEDIA} == expected
    assert all(len(digest) == 64 for digest in EXCLUDED_THIRD_PARTY_MEDIA.values())
    assert all(not (repository / path).exists() for path in EXCLUDED_THIRD_PARTY_MEDIA)


def test_release_checksums_use_downloadable_asset_names(tmp_path: Path) -> None:
    (tmp_path / "output" / "pdf").mkdir(parents=True)
    (tmp_path / "output" / "pdf" / "vcoco_v3_motion_identifiability_v3.0.0.pdf").write_bytes(
        b"temporal"
    )
    (tmp_path / "output" / "pdf" / "okutama_cptr_development_v3.0.0.pdf").write_bytes(b"cptr")
    manifest = tmp_path / "human_activity_study_v3.0.0_manifest.json"
    manifest.write_bytes(b"manifest")

    lines = checksum_text(tmp_path, manifest).splitlines()

    assert [line.split("  ", 1)[1] for line in lines] == [
        "vcoco_v3_motion_identifiability_v3.0.0.pdf",
        "okutama_cptr_development_v3.0.0.pdf",
        "human_activity_study_v3.0.0_manifest.json",
    ]


def test_frozen_check_uses_release_commit_and_allows_continuation_files(tmp_path: Path) -> None:
    repository, manifest, checksums, revision = frozen_release_fixture(tmp_path)
    (repository / "artifact.bin").write_bytes(b"continued evidence")
    continuation = repository / "docs" / "continuation.md"
    continuation.parent.mkdir()
    continuation.write_text("post-release work\n", encoding="utf-8")
    run_git(repository, "add", "artifact.bin", continuation.relative_to(repository).as_posix())
    run_git(repository, "commit", "--quiet", "-m", "continuation")
    continuation_revision = run_git(repository, "rev-parse", "HEAD")
    assert continuation_revision != revision

    verify_frozen_release(
        repository,
        manifest,
        checksums,
        release_revision=revision,
    )
    with pytest.raises(RuntimeError, match="does not cover the immutable release tree"):
        verify_frozen_release(
            repository,
            manifest,
            checksums,
            release_revision=continuation_revision,
        )


@pytest.mark.parametrize(
    "fixture_options, message",
    [
        ({"identity_overrides": {"report_version": "3.0.1"}}, "identity changed"),
        ({"artifact_count_override": 99}, "artifact inventory is invalid"),
        ({"omit_artifact": "artifact.bin"}, "does not cover the immutable release tree"),
        (
            {
                "extra_artifacts": {
                    "phantom.bin": {
                        "sha256": hashlib.sha256(b"").hexdigest(),
                        "size_bytes": 0,
                    }
                }
            },
            "does not cover the immutable release tree",
        ),
        (
            {"evidence_overrides": {"artifact.bin": {"sha256": "0" * 64}}},
            "artifact hash differs",
        ),
        (
            {"evidence_overrides": {"artifact.bin": {"size_bytes": 999}}},
            "artifact size differs",
        ),
    ],
)
def test_frozen_check_rejects_manifest_identity_inventory_and_blob_drift(
    tmp_path: Path,
    fixture_options: dict[str, object],
    message: str,
) -> None:
    repository, manifest, checksums, revision = frozen_release_fixture(tmp_path, **fixture_options)

    with pytest.raises(RuntimeError, match=message):
        verify_frozen_release(
            repository,
            manifest,
            checksums,
            release_revision=revision,
        )


def test_frozen_check_rejects_modified_manifest_and_checksums(tmp_path: Path) -> None:
    repository, manifest, checksums, revision = frozen_release_fixture(tmp_path)
    original_manifest = manifest.read_bytes()
    original_checksums = checksums.read_bytes()
    pdf_path = repository / FROZEN_PDFS[0]
    original_pdf = pdf_path.read_bytes()
    manifest.write_bytes(original_manifest + b" ")
    with pytest.raises(RuntimeError, match="manifest differs"):
        verify_frozen_release(
            repository,
            manifest,
            checksums,
            release_revision=revision,
        )

    manifest.write_bytes(original_manifest)
    checksums.write_bytes(original_checksums)
    pdf_path.write_bytes(b"modified-pdf")
    with pytest.raises(RuntimeError, match="checksum targets are stale"):
        verify_frozen_release(
            repository,
            manifest,
            checksums,
            release_revision=revision,
        )

    pdf_path.write_bytes(original_pdf)
    checksums.write_text("0" * 64 + "  invalid\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksums differ"):
        verify_frozen_release(
            repository,
            manifest,
            checksums,
            release_revision=revision,
        )


def test_frozen_check_rejects_internally_invalid_committed_checksums(tmp_path: Path) -> None:
    repository, manifest, checksums, revision = frozen_release_fixture(
        tmp_path,
        checksum_override="0" * 64 + "  invalid\n",
    )

    with pytest.raises(RuntimeError, match="internally inconsistent"):
        verify_frozen_release(
            repository,
            manifest,
            checksums,
            release_revision=revision,
        )


def test_scientific_validation_plan_binds_the_remaining_evidence() -> None:
    repository = Path(__file__).resolve().parents[1]
    plan = (repository / "docs" / "SCIENTIFIC_VALIDATION_PLAN.md").read_text(encoding="utf-8")

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
        assert marker in plan
