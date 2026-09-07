from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import lock_okutama_video_p3_probe as lock

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_freezes_six_arms_budget_and_exact_comparison_family() -> None:
    spec = lock.load_protocol(ROOT)
    assert tuple(spec["arms"]) == lock.ARMS
    assert spec["statistics"]["comparisons"] == lock.expected_comparisons()
    assert len(spec["statistics"]["comparisons"]) == 16
    assert spec["probe"]["maximum_unique_estimator_fits"] == 455
    assert spec["references"]["p2a_best"]["role"].startswith("comparison only after")


def test_comparison_family_drift_fails_closed(tmp_path: Path) -> None:
    spec = json.loads((ROOT / lock.PROTOCOL_PATH).read_text(encoding="utf-8"))
    spec["statistics"]["comparisons"][0]["candidate"] = "long_dino_mean"
    protocol = tmp_path / lock.PROTOCOL_PATH
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="comparison family"):
        lock.load_protocol(tmp_path)


def test_p2_reference_is_hashed_but_not_decoded_before_fitting(monkeypatch) -> None:
    reference = {
        "execution_lock": {"path": "execution.json"},
        "summary": {"path": "summary.json"},
        "oof": {"path": "oof.npz"},
        "arm": "video_dino_mean_multinomial",
        "macro_f1": 0.75,
    }
    spec = {"references": {"p2a_best": reference}}
    p2_lock = {"authorization": {"probe_fitting": True}}
    data = lock.p1.PrimaryData(
        np.asarray(["a"]),
        np.asarray([0]),
        np.asarray(["scene"]),
        np.asarray([0]),
        {},
        {"short": np.ones(1, dtype=bool)},
        np.full((1, 3), 1 / 3),
        np.zeros(1, dtype=bool),
        np.zeros(1, dtype=bool),
    )
    summary = {
        "status": "OKUTAMA_VIDEO_P2A_EXPLORATORY_CROSSFIT_COMPLETE",
        "rows": 4977,
        "reference_predictions_used_for_fitting": False,
        "models": {reference["arm"]: {"metrics": {"macro_f1": 0.75}}},
    }
    accessed = []

    def checked(_root, item):
        accessed.append(item["path"])
        raw = json.dumps(summary).encode() if item["path"] == "summary.json" else b"opaque"
        return Path(item["path"]), raw

    monkeypatch.setattr(lock, "_historical_lock", lambda *_: (p2_lock, b"lock"))
    monkeypatch.setattr(lock.p1, "load_primary_data", lambda *_: data)
    monkeypatch.setattr(lock, "checked", checked)
    monkeypatch.setattr(lock.np, "load", lambda *_args, **_kwargs: pytest.fail("OOF decoded"))
    _, _, receipt = lock._validate_p2(ROOT, spec, "a" * 40)
    assert accessed == ["summary.json", "oof.npz"]
    assert receipt["prefit_validation"] == "sha256_and_size_only_no_probability_decode"


def test_lock_write_never_overwrites_and_compare_ignores_only_time(tmp_path: Path) -> None:
    output = tmp_path / "lock.json"
    payload = {"status": lock.STATUS, "locked_at_utc": "old", "x": 1}
    lock.write_lock(tmp_path, output, payload)
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        lock.write_lock(tmp_path, output, payload)
    assert output.read_bytes() == before
    lock.compare(payload, {**payload, "locked_at_utc": "new"})
    with pytest.raises(RuntimeError, match="x"):
        lock.compare(payload, {**payload, "x": 2})
