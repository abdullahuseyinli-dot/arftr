from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import lock_okutama_video_p3_extraction as lock

ROOT = Path(__file__).resolve().parents[1]


def test_p3_extraction_protocol_freezes_scope_and_fit_free_authority() -> None:
    spec = lock.load_protocol(ROOT)
    assert spec["authorization"] == lock.AUTHORIZATION
    assert spec["execution_sources"] == lock.EXPECTED_SOURCES
    assert spec["cohort"]["rows"] == 4977
    assert spec["cohort"]["all_long_frames_valid_rows"] == 4510
    assert spec["cohort"]["incomplete_long_rows"] == 467
    assert spec["sampling"]["source_offsets"] == list(range(-32, 29, 4))
    assert not spec["authorization"]["model_fitting"]
    assert not spec["authorization"]["annotation_payload_access"]


def test_lock_write_never_overwrites(tmp_path: Path) -> None:
    root = tmp_path
    output = root / "lock.json"
    payload = {"status": lock.STATUS}
    lock.write_lock(root, output, payload)
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        lock.write_lock(root, output, {"status": "changed"})
    assert output.read_bytes() == before


def test_historical_materialization_hash_checked_before_json_decode(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "materialization.json"
    path.write_bytes(b"not-json")
    spec = {"materialization_lock": {"path": "materialization.json", "size_bytes": 8, "sha256": "0" * 64}}
    monkeypatch.setattr(lock, "_repo_path", lambda *_: path)
    with pytest.raises(RuntimeError, match="changed before decoding"):
        lock._validate_historical_materialization(tmp_path, spec, "a" * 40)


def test_retained_lock_detects_top_level_change() -> None:
    retained = {"status": lock.STATUS, "locked_at_utc": "old", "manifest": {"x": 1}}
    current = {"status": lock.STATUS, "locked_at_utc": "new", "manifest": {"x": 2}}
    with pytest.raises(RuntimeError, match="manifest"):
        lock._compare(retained, current)
    current["manifest"] = {"x": 1}
    lock._compare(retained, current)


def test_probe_protocol_declared_before_extraction_and_has_fixed_budget() -> None:
    spec = json.loads(
        (ROOT / "experiments/okutama_video_p3_probe_protocol.json").read_text(encoding="utf-8")
    )
    assert spec["status"] == "DECLARED_BEFORE_OKUTAMA_P3_LONG_FEATURE_EXTRACTION_OR_PROBE_FITTING"
    assert len(spec["arms"]) == 6
    assert len(spec["statistics"]["comparisons"]) == 16
    assert spec["probe"]["C_values"] == [1e-5, 1e-4, 1e-3, 1e-2]
    assert spec["probe"]["maximum_unique_estimator_fits"] == 455
    assert not spec["feature_recipe"]["missing_indicator_as_feature"]
