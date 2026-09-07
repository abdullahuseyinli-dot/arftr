"""Bind the committed P6 protocol and named P3/P5 OOF evidence before replay."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hac.video_consensus import ARM_COMPONENTS, ARMS  # noqa: E402
from tools import lock_hac_continuation_protocols as common  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_p6_protocol.json"
PROTOCOL_STATUS = "DECLARED_AFTER_P5_AND_RETROSPECTIVE_FUSION_SCREEN_BEFORE_P6_REPLAY"
STATUS = "OKUTAMA_VIDEO_P6_LOCKED_BEFORE_DETERMINISTIC_REPLAY"
AUTHORIZATION = {
    "read_named_p3_p5_oof_artifacts": True,
    "deterministic_probability_fusion": True,
    "model_fitting": False,
    "feature_extraction": False,
    "raw_image_access": False,
    "protected_data_access": False,
}
SOURCES = {
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_video_p6.py",
    "runner": "experiments/run_okutama_video_p6.py",
    "consensus_module": "src/hac/video_consensus.py",
    "p1_statistics_runner": "experiments/run_okutama_video_probe.py",
    "p2_statistics_runner": "experiments/run_okutama_video_p2a.py",
    "common_locker": "tools/lock_hac_continuation_protocols.py",
    "requirements": "requirements-video-lock.txt",
}


def expected_comparisons() -> list[dict[str, str]]:
    return [
        {"candidate": ARMS[0], "reference": "p3_best"},
        {"candidate": ARMS[0], "reference": "p5_best"},
        {"candidate": ARMS[1], "reference": "p3_best"},
        {"candidate": ARMS[1], "reference": "p5_best"},
        {"candidate": ARMS[0], "reference": ARMS[1]},
    ]


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    architecture = spec.get("architecture", {})
    expected_arms = {
        arm: [f"{phase}.{name}" for phase, name in ARM_COMPONENTS[arm]] for arm in ARMS
    }
    if (spec.get("protocol_version"), spec.get("status")) != (
        "1.0.0",
        PROTOCOL_STATUS,
    ):
        raise RuntimeError("Unexpected P6 protocol/version")
    if (
        spec.get("primary_rows") != 4977
        or architecture.get("arms") != expected_arms
        or architecture.get("primary_arm") != ARMS[0]
        or architecture.get("learned_parameters") != 0
        or architecture.get("fitted_weights") is not False
        or architecture.get("label_dependent_routing") is not False
        or spec.get("statistics", {}).get("comparisons") != expected_comparisons()
        or spec.get("statistics", {}).get("bootstrap_resamples") != 10000
        or spec.get("execution_budget", {}).get("model_fits") != 0
        or spec.get("authorization", {}).get("protected_data_access") is not False
        or "already_observed_primary_macro_f1" not in spec.get("adaptation_disclosure", {})
    ):
        raise RuntimeError("P6 adaptive replay contract changed")
    return spec


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before P6")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def _path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    common.assert_role_safe_input_path(root, path)
    return path


def _artifact_receipt(root: Path, declared: dict[str, Any]) -> dict[str, Any]:
    path = _path(root, declared["path"])
    digest, size = common._sha256_file(path)
    if digest != declared["sha256"] or size != int(declared["size_bytes"]):
        raise RuntimeError(f"P6 source artifact receipt changed: {declared['path']}")
    return {
        "path": common._relative_path(root, path),
        "sha256": digest,
        "size_bytes": size,
    }


def _evidence_receipts(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for phase, expected_status in (
        ("p3", "OKUTAMA_VIDEO_P3_EXPLORATORY_CROSSFIT_COMPLETE"),
        ("p5", "OKUTAMA_VIDEO_P5_ADAPTIVE_CROSSFIT_COMPLETE"),
    ):
        declared = spec["sources"][phase]
        summary_receipt = _artifact_receipt(root, declared["summary"])
        oof_receipt = _artifact_receipt(root, declared["oof"])
        summary = json.loads(
            _path(root, declared["summary"]["path"]).read_text(encoding="utf-8")
        )
        if summary.get("status") != expected_status:
            raise RuntimeError(f"Unexpected {phase.upper()} evidence status")
        if summary.get("artifacts", {}).get("oof_probabilities.npz") != oof_receipt["sha256"]:
            raise RuntimeError(f"{phase.upper()} summary does not bind its OOF artifact")
        receipts[phase] = {
            "summary": summary_receipt,
            "oof": oof_receipt,
            "required_arrays": declared["required_arrays"],
            "status": expected_status,
        }
    return receipts


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, path, commit) for name, path in SOURCES.items()
    }
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "evidence": _evidence_receipts(root, spec),
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "probability_rows_decoded": 0,
            "probability_fusions": 0,
            "model_fits": 0,
            "raw_images_read": 0,
            "protected_rows_read": 0,
        },
    }


def compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained P6 lock changed: " + ", ".join(changed))


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing P6 lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the P6 pre-replay execution lock")
    compare(retained, build_payload(root))
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.mode == "check":
        payload = validate_lock(root, output)
    else:
        payload = build_payload(root)
        if args.mode == "lock":
            write_lock(root, output, payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload["status"],
                "rows": 4977,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
