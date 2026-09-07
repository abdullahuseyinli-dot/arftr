"""Bind the P3 token caches and P5 moment hypotheses before fitting."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT, _ROOT / "experiments"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import run_okutama_video_probe as p1  # noqa: E402

from hac.video_token_moments import ARMS  # noqa: E402
from tools import lock_hac_continuation_protocols as common  # noqa: E402
from tools import lock_okutama_video_p4_probe as p4_lineage  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_p5_protocol.json"
PROTOCOL_STATUS = "DECLARED_AFTER_P4_BEFORE_P5_MOMENT_DERIVATION_OR_FITTING"
STATUS = "OKUTAMA_VIDEO_P5_LOCKED_BEFORE_MOMENT_DERIVATION_OR_FITTING"
AUTHORIZATION = {
    "frozen_feature_read": True,
    "fixed_moment_derivation": True,
    "probe_fitting": True,
    "raw_image_access": False,
    "feature_extraction": False,
    "backbone_fitting": False,
    "protected_data_access": False,
}
SOURCES = {
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_video_p5.py",
    "runner": "experiments/run_okutama_video_p5.py",
    "moment_module": "src/hac/video_token_moments.py",
    "moment_input_type": "src/hac/video_kinematic.py",
    "multiscale_module": "src/hac/video_multiscale.py",
    "p4_lineage_helper": "tools/lock_okutama_video_p4_probe.py",
    "p3_runner": "experiments/run_okutama_video_p3.py",
    "p2_runner": "experiments/run_okutama_video_p2a.py",
    "p1_runner": "experiments/run_okutama_video_probe.py",
    "common_locker": "tools/lock_hac_continuation_protocols.py",
    "requirements": "requirements-video-lock.txt",
}


def expected_comparisons() -> list[dict[str, str]]:
    return [
        *({"candidate": arm, "reference": "p3_best"} for arm in ARMS),
        *(
            {"candidate": arm, "reference": "mean_factorized_refit"}
            for arm in ARMS[1:]
        ),
        {
            "candidate": "orthogonal_moments_factorized",
            "reference": "spatial_contrast_factorized",
        },
        {
            "candidate": "orthogonal_moments_factorized",
            "reference": "temporal_spectrum_factorized",
        },
        {
            "candidate": "orthogonal_moments_multinomial",
            "reference": "orthogonal_moments_factorized",
        },
    ]


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if (spec.get("protocol_version"), spec.get("status")) != (
        "1.0.0",
        PROTOCOL_STATUS,
    ):
        raise RuntimeError("Unexpected P5 protocol/version")
    probe = spec.get("probe", {})
    if (
        tuple(spec.get("arms", ())) != ARMS
        or spec.get("primary_rows") != 4977
        or probe.get("C_values") != [1e-5, 1e-4, 1e-3, 1e-2]
        or probe.get("maximum_unique_estimator_fits") != 585
        or probe.get("inner_folds") != 3
        or probe.get("inner_seed") != 42
        or spec.get("statistics", {}).get("comparisons") != expected_comparisons()
        or spec.get("authorization", {}).get("protected_data_access") is not False
    ):
        raise RuntimeError("P5 hypothesis, budget, or comparison contract changed")
    recipe = spec.get("feature_recipe", {})
    if (
        recipe.get("dct_definition")
        != "sqrt(2/T)*cos(pi*(t+0.5)*k/T), k=1..4, t=0..T-1"
        or recipe.get("dc_coefficient_excluded") is not True
        or recipe.get("learned_feature_operations") is not False
    ):
        raise RuntimeError("P5 fixed moment recipe changed")
    return spec


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before P5")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def _path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    common.assert_role_safe_input_path(root, path)
    return path


def receipt(root: Path, value: str | Path) -> dict[str, Any]:
    path = _path(root, value)
    digest, size = common._sha256_file(path)
    return {
        "path": common._relative_path(root, path),
        "sha256": digest,
        "size_bytes": size,
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, path, commit)
        for name, path in SOURCES.items()
    }
    p3_lock, data, p3_reference = p4_lineage._validate_p3(root, spec, commit)
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "p3_reference": p3_reference,
        "p3_long_caches": p3_lock["long_caches"],
        "p3_fold_map_rows": p3_lock["fold_map_rows"],
        "p3_fold_map_sha256": p3_lock["fold_map_sha256"],
        "p3_randomization": p3_lock["randomization"],
        "sample_ids_sha256": p1.canonical_digest(data.sample_ids.tolist()),
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "fixed_moment_derivations": 0,
            "model_fits": 0,
            "raw_images_read": 0,
            "protected_rows_read": 0,
            "p3_reference_probabilities_decoded": 0,
        },
    }


def compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained P5 lock changed: " + ", ".join(changed))


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing P5 lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the P5 pre-fit execution lock")
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
