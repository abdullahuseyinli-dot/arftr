"""Build or verify the Human Activity Classification Study v3.0.0 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "human-activity-study-v3.0.0"
REPORT_VERSION = "3.0.0"
SOFTWARE_VERSION = "3.0.0"
PREPARED_DATE = "2026-08-24"
RELEASE_BASE_COMMIT = "2697126e3887f99d0b815ada68eb3c4a3c861822"
DEFAULT_OUTPUT = PurePosixPath("results/human_activity_study_v3.0.0_manifest.json")
DEFAULT_CHECKSUMS = PurePosixPath("release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt")

REQUIRED_ARTIFACTS = frozenset(
    {
        PurePosixPath(".zenodo.json"),
        PurePosixPath("CHANGELOG.md"),
        PurePosixPath("CITATION.cff"),
        PurePosixPath("README.md"),
        PurePosixPath("docs/OKUTAMA_CPTR_DEVELOPMENT.md"),
        PurePosixPath("docs/SCIENTIFIC_VALIDATION_PLAN.md"),
        PurePosixPath("docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md"),
        PurePosixPath("docs/releases/HUMAN_ACTIVITY_STUDY_V3.0.0.md"),
        PurePosixPath("human_activity_classification.ipynb"),
        PurePosixPath("output/pdf/okutama_cptr_development_v3.0.0.pdf"),
        PurePosixPath("output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf"),
        PurePosixPath("pyproject.toml"),
        PurePosixPath("requirements-v3-lock.txt"),
        PurePosixPath("results/okutama_cptr/evidence_manifest.json"),
        PurePosixPath("results/vcoco_v3/evidence_manifest.json"),
        PurePosixPath("tools/build_v3_release_manifest.py"),
        PurePosixPath("tools/verify_v3_release_archive.py"),
    }
)

EXCLUDED_THIRD_PARTY_MEDIA = {
    PurePosixPath("assets/champion_error_gallery.png"): (
        "b0ab0ccf114b50cf69a2e7fdcc99153b4fbfedf10ecfc79c5c75e1f344358865"
    ),
    PurePosixPath("assets/convnext_small_faithfulness_gallery.jpg"): (
        "35865cc879b212a4f4690467c9100321b3d11ea16794280c31a47462e562ce32"
    ),
    PurePosixPath("assets/dinov2_small_faithfulness_gallery.jpg"): (
        "ebd41b2e289377ae470dc72fbe0fddc82d72d44109c631b06cdfa66e2ffacc91"
    ),
    PurePosixPath("assets/probability_blend_faithfulness_gallery.jpg"): (
        "cf787683493e90817ec1ac86ff44ec4d752480e0d2f8a03356f0fa9141175849"
    ),
}

TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".cff",
        ".css",
        ".csv",
        ".html",
        ".ipynb",
        ".js",
        ".json",
        ".md",
        ".py",
        ".svg",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
TEXT_ARTIFACT_FILENAMES = frozenset(
    {
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "LICENSE",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_bytes(path: Path) -> bytes:
    """Return bytes as Git will store them under the repository attributes."""

    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES or path.name in TEXT_ARTIFACT_FILENAMES:
        payload = payload.replace(b"\r\n", b"\n")
        if b"\r" in payload:
            raise RuntimeError(f"Text artifact contains a lone carriage return: {path}")
    return payload


def artifact_paths(repository: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    relative_paths = {
        PurePosixPath(value.decode("utf-8")) for value in completed.stdout.split(b"\0") if value
    }
    relative_paths.difference_update({DEFAULT_OUTPUT, DEFAULT_CHECKSUMS})
    missing_required = sorted(REQUIRED_ARTIFACTS.difference(relative_paths))
    if missing_required:
        formatted = ", ".join(path.as_posix() for path in missing_required)
        raise RuntimeError(f"Required release artifacts are missing or ignored: {formatted}")
    paths = {repository / relative for relative in relative_paths}
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        formatted = ", ".join(path.relative_to(repository).as_posix() for path in missing)
        raise RuntimeError(f"Release candidate files are missing: {formatted}")
    if not paths:
        raise RuntimeError("The release candidate inventory is empty")
    return sorted(paths, key=lambda path: path.relative_to(repository).as_posix())


def validate_versions(repository: Path) -> None:
    with (repository / "pyproject.toml").open("rb") as handle:
        package_version = tomllib.load(handle)["project"]["version"]
    if package_version != SOFTWARE_VERSION:
        raise RuntimeError(f"Expected software version {SOFTWARE_VERSION}, found {package_version}")

    for relative in (
        "docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md",
        "docs/OKUTAMA_CPTR_DEVELOPMENT.md",
    ):
        report = (repository / relative).read_text(encoding="utf-8")
        if f"version: {REPORT_VERSION}" not in report:
            raise RuntimeError(f"Report version is missing or stale: {relative}")

    citation = (repository / "CITATION.cff").read_text(encoding="utf-8")
    if f'version: "{SOFTWARE_VERSION}"' not in citation:
        raise RuntimeError("Citation metadata does not match the v3 release")

    zenodo = json.loads((repository / ".zenodo.json").read_text(encoding="utf-8"))
    if zenodo.get("version") != REPORT_VERSION or "publication_date" in zenodo:
        raise RuntimeError("Zenodo metadata does not match the v3 release")

    present_exclusions = [
        path for path in EXCLUDED_THIRD_PARTY_MEDIA if (repository / path).exists()
    ]
    if present_exclusions:
        formatted = ", ".join(path.as_posix() for path in sorted(present_exclusions))
        raise RuntimeError(f"Non-distributable third-party media are present: {formatted}")


def build_manifest(repository: Path) -> dict:
    repository = repository.resolve()
    validate_versions(repository)
    artifacts = {}
    for path in artifact_paths(repository):
        relative = path.relative_to(repository).as_posix()
        payload = artifact_bytes(path)
        artifacts[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "report_version": REPORT_VERSION,
        "software_version": SOFTWARE_VERSION,
        "prepared_date": PREPARED_DATE,
        "artifact_scope": (
            "Complete nonignored repository release candidate, including the v3 "
            "motion-identifiability and CPTR reports, portable evidence, figures, "
            "protocols, implementation, tests, and reproducibility entry points"
        ),
        "text_digest_policy": (
            "CRLF is canonicalized to LF for source files and repository metadata "
            "declared as text by .gitattributes"
        ),
        "distribution_exclusions": {
            path.as_posix(): {
                "sha256": digest,
                "reason": (
                    "Qualitative composite containing third-party COCO/Flickr imagery; "
                    "retained only in ignored local evidence and historical Git history"
                ),
            }
            for path, digest in sorted(
                EXCLUDED_THIRD_PARTY_MEDIA.items(), key=lambda item: item[0].as_posix()
            )
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def encoded_manifest(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def checksum_text(repository: Path, manifest_path: Path) -> str:
    targets = (
        repository / "output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf",
        repository / "output/pdf/okutama_cptr_development_v3.0.0.pdf",
        manifest_path,
    )
    return "".join(f"{sha256_file(path)}  {path.name}\n" for path in targets)


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Git command failed ({' '.join(arguments)}){suffix}") from error


def _verify_commit(repository: Path, revision: str) -> str:
    resolved = (
        _git_bytes(repository, "rev-parse", "--verify", f"{revision}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    if resolved.lower() != revision.lower():
        raise RuntimeError(
            f"Release base revision resolved to {resolved}, expected exact commit {revision}"
        )
    return resolved


def _repository_relative(repository: Path, path: Path) -> PurePosixPath:
    try:
        return PurePosixPath(path.resolve().relative_to(repository.resolve()).as_posix())
    except ValueError as error:
        raise RuntimeError(f"Release control file is outside the repository: {path}") from error


def _git_blob(repository: Path, revision: str, relative: PurePosixPath) -> bytes:
    return _git_bytes(repository, "cat-file", "blob", f"{revision}:{relative.as_posix()}")


def _git_tree_paths(repository: Path, revision: str) -> set[PurePosixPath]:
    payload = _git_bytes(repository, "ls-tree", "-r", "--name-only", "-z", revision)
    return {PurePosixPath(value.decode("utf-8")) for value in payload.split(b"\0") if value}


def _validate_frozen_manifest_identity(manifest: object) -> dict[str, dict[str, object]]:
    if not isinstance(manifest, dict):
        raise RuntimeError("Release manifest root must be an object")
    expected_identity = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "report_version": REPORT_VERSION,
        "software_version": SOFTWARE_VERSION,
        "prepared_date": PREPARED_DATE,
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("Release manifest identity changed")
    artifacts = manifest.get("artifacts")
    artifact_count = manifest.get("artifact_count")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(artifact_count, int)
        or isinstance(artifact_count, bool)
        or artifact_count < 1
        or artifact_count != len(artifacts)
    ):
        raise RuntimeError("Release manifest artifact inventory is invalid")
    for relative, evidence in artifacts.items():
        if (
            not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or not isinstance(evidence, dict)
        ):
            raise RuntimeError(f"Release manifest artifact entry is invalid: {relative!r}")
        digest = evidence.get("sha256")
        size = evidence.get("size_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise RuntimeError(f"Release manifest evidence is invalid: {relative}")
    return artifacts


def _frozen_checksum_text(
    repository: Path,
    revision: str,
    manifest_relative: PurePosixPath,
) -> str:
    targets = (
        PurePosixPath("output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf"),
        PurePosixPath("output/pdf/okutama_cptr_development_v3.0.0.pdf"),
        manifest_relative,
    )
    return "".join(
        f"{hashlib.sha256(_git_blob(repository, revision, path)).hexdigest()}  {path.name}\n"
        for path in targets
    )


def verify_frozen_release(
    repository: Path,
    manifest_path: Path,
    checksums_path: Path,
    *,
    release_revision: str = RELEASE_BASE_COMMIT,
) -> None:
    """Verify the historical release against its immutable Git commit, not the worktree."""

    repository = repository.resolve()
    revision = _verify_commit(repository, release_revision)
    manifest_relative = _repository_relative(repository, manifest_path)
    checksums_relative = _repository_relative(repository, checksums_path)
    if not manifest_path.is_file():
        raise RuntimeError(f"Release manifest is missing: {manifest_path}")
    if not checksums_path.is_file():
        raise RuntimeError(f"Release checksums are missing: {checksums_path}")

    manifest_bytes = manifest_path.read_bytes()
    frozen_manifest_bytes = _git_blob(repository, revision, manifest_relative)
    if manifest_bytes != frozen_manifest_bytes:
        raise RuntimeError(f"Release manifest differs from {revision}: {manifest_path}")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Release manifest is not valid UTF-8 JSON") from error
    artifacts = _validate_frozen_manifest_identity(manifest)

    tree_inventory = _git_tree_paths(repository, revision)
    expected_inventory = tree_inventory.difference({manifest_relative, checksums_relative})
    recorded_inventory = {PurePosixPath(relative) for relative in artifacts}
    if recorded_inventory != expected_inventory:
        missing = sorted(path.as_posix() for path in expected_inventory - recorded_inventory)
        unexpected = sorted(path.as_posix() for path in recorded_inventory - expected_inventory)
        raise RuntimeError(
            "Release manifest does not cover the immutable release tree; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for relative, evidence in artifacts.items():
        blob = _git_blob(repository, revision, PurePosixPath(relative))
        if hashlib.sha256(blob).hexdigest() != evidence["sha256"]:
            raise RuntimeError(f"Release-base artifact hash differs: {relative}")
        if len(blob) != evidence["size_bytes"]:
            raise RuntimeError(f"Release-base artifact size differs: {relative}")

    expected_checksums = _frozen_checksum_text(repository, revision, manifest_relative)
    frozen_checksums = _git_blob(repository, revision, checksums_relative)
    if frozen_checksums != expected_checksums.encode("utf-8"):
        raise RuntimeError(f"Release-base checksums are internally inconsistent at {revision}")
    current_checksums = checksums_path.read_bytes()
    if current_checksums != frozen_checksums:
        raise RuntimeError(f"Release checksums differ from {revision}: {checksums_path}")
    if current_checksums.decode("utf-8") != checksum_text(repository, manifest_path):
        raise RuntimeError(f"Release checksum targets are stale: {checksums_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=PurePosixPath, default=DEFAULT_OUTPUT)
    parser.add_argument("--checksums", type=PurePosixPath, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    output = repository / args.output
    checksums = repository / args.checksums
    if args.check:
        verify_frozen_release(repository, output, checksums)
        print(f"Release manifest verified: {output}")
        return

    expected = encoded_manifest(build_manifest(repository))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    checksums.write_text(checksum_text(repository, output), encoding="utf-8", newline="\n")
    print(f"Release manifest written: {output}")
    print(f"Release checksums written: {checksums}")


if __name__ == "__main__":
    main()
