"""Audit source/posture cache shards without reading task labels or ARFTR."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.posture_witness import PROJECTION_DIM, SUPERVISED_COEFFICIENTS

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    lock = json.loads((cache / "execution_lock.json").read_text(encoding="utf-8"))
    if lock.get("status") != "SOURCE_POSTURE_LABEL_BLIND_EXTRACTION_LOCK":
        raise RuntimeError("Cache extraction lock missing")
    shards = []
    all_ids, coverage = [], []
    with np.load(METADATA, allow_pickle=False) as saved:
        canonical_ids = saved["sample_ids"]
    if lock.get("rows") != len(canonical_ids):
        raise RuntimeError("Cache lock and canonical cohort row counts differ")
    reference_by_id: dict[str, np.ndarray] = {}
    reference_keys: list[str] | None = None
    if args.reference_preflight is not None:
        with np.load(
            args.reference_preflight.resolve() / "features.npz", allow_pickle=False
        ) as reference:
            reference_keys = reference["crop_keys"].tolist()
            reference_by_id = {
                str(sample_id): feature.copy()
                for sample_id, feature in zip(
                    reference["sample_ids"], reference["features"], strict=True
                )
            }
    replay_matches = []
    for receipt_path in sorted(cache.glob("shard-*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        npz_path = receipt_path.with_suffix(".npz")
        if (
            receipt.get("status") != "SOURCE_POSTURE_SHARD_COMPLETE"
            or not isinstance(receipt.get("start"), int)
            or not isinstance(receipt.get("stop"), int)
            or not 0 <= receipt["start"] < receipt["stop"] <= len(canonical_ids)
            or receipt.get("rows") != receipt["stop"] - receipt["start"]
            or receipt["npz_sha256"] != sha256(npz_path)
        ):
            raise RuntimeError(f"Shard hash changed: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as saved:
            ids, keys, features, available = (saved[name] for name in ("sample_ids", "crop_keys", "features", "available"))
        if (
            features.shape != (len(ids), len(keys), 768)
            or available.shape != features.shape[:2]
            or features.dtype != np.float16
            or available.dtype != np.bool_
            or keys.tolist() != lock["crop_keys"]
            or not np.isfinite(features).all()
            or len(np.unique(ids)) != len(ids)
            or receipt["rows"] != len(ids)
        ):
            raise RuntimeError("Malformed source-posture shard")
        norms = np.linalg.norm(features.astype(np.float32), axis=-1)
        if np.any(available & ((norms < 0.999) | (norms > 1.001))) or np.any(~available & (features != 0).any(2)):
            raise RuntimeError("Feature normalization or missingness contract failed")
        all_ids.extend(ids.tolist())
        shard_coverage = list(range(receipt["start"], receipt["stop"]))
        if not np.array_equal(ids, canonical_ids[shard_coverage]):
            raise RuntimeError("Shard identities do not occupy their canonical row range")
        for local, sample_id in enumerate(ids):
            expected = reference_by_id.get(str(sample_id))
            if expected is not None:
                difference = np.abs(
                    features[local].astype(np.float32) - expected.astype(np.float32)
                )
                replay_matches.append(
                    {
                        "sample_id": str(sample_id),
                        "feature_exact": bool(np.array_equal(features[local], expected)),
                        "max_abs_difference": float(difference.max()),
                    }
                )
        coverage.extend(shard_coverage)
        shards.append({"receipt": receipt_path.name, "npz": npz_path.name, "start": receipt["start"], "stop": receipt["stop"], "rows": len(ids), "available_rows": int(available.all(1).sum()), "norm_min": float(norms[available].min()) if available.any() else None, "norm_max": float(norms[available].max()) if available.any() else None})
    if (
        len(set(all_ids)) != len(all_ids)
        or len(set(coverage)) != len(coverage)
        or sorted(coverage) != list(range(len(canonical_ids)))
        or not np.array_equal(np.asarray(all_ids), canonical_ids)
    ):
        raise RuntimeError("Cache shards do not cover the canonical cohort exactly once")
    parity = None
    if args.reference_preflight is not None:
        if reference_keys != lock["crop_keys"]:
            raise RuntimeError("Reference and cache crop inventories differ")
        if len(replay_matches) != len(reference_by_id):
            raise RuntimeError("Cache does not cover every canonical preflight identity")
        parity = {
            "rows": len(replay_matches),
            "sample_ids": [row["sample_id"] for row in replay_matches],
            "identity_exact": True,
            "feature_exact": all(row["feature_exact"] for row in replay_matches),
            "max_abs_difference": max(row["max_abs_difference"] for row in replay_matches),
        }
        if not parity["feature_exact"]:
            raise RuntimeError("Cache startup does not replay canonical preflight")
    result = {
        "status": "SOURCE_POSTURE_CACHE_AUDIT_PASS",
        "cache": str(cache), "lock_sha256": sha256(cache / "execution_lock.json"),
        "shards": shards, "rows": len(all_ids), "coverage_indices": sorted(coverage),
        "reference_preflight_parity": parity,
        "posture_head_contract": {"projection_dim": PROJECTION_DIM, "supervised_coefficients": SUPERVISED_COEFFICIENTS},
        "labels_read": 0, "ARFTR_outputs_read": 0, "model_fits": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "rows", "reference_preflight_parity")}, indent=2))


if __name__ == "__main__":
    main()
