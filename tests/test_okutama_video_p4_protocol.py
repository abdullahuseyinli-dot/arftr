from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import lock_okutama_video_p4_materialization as materialization
from tools import lock_okutama_video_p4_probe as probe

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_freezes_seven_arms_eighteen_contrasts_and_910_fits() -> None:
    spec = materialization.load_protocol(ROOT)
    assert tuple(spec["arms"]) == materialization.ARMS
    assert spec["statistics"]["comparisons"] == materialization.expected_comparisons()
    assert len(spec["statistics"]["comparisons"]) == 18
    assert spec["probe"]["maximum_unique_estimator_fits"] == 910
    assert spec["statistics"]["primary_candidate"] == "camera_reference_marginalized"
    assert spec["adaptation_disclosure"]["observed_before_declaration"]
    assert not spec["authorization"]["protected_data_access"]


def test_protocol_comparison_or_budget_drift_fails_closed(tmp_path: Path) -> None:
    spec = json.loads(
        (ROOT / materialization.PROTOCOL_PATH).read_text(encoding="utf-8")
    )
    protocol_path = tmp_path / materialization.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    spec["statistics"]["comparisons"].reverse()
    protocol_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="comparison family"):
        materialization.load_protocol(tmp_path)
    spec["statistics"]["comparisons"].reverse()
    spec["probe"]["maximum_unique_estimator_fits"] = 911
    protocol_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="budget"):
        materialization.load_protocol(tmp_path)


@pytest.mark.parametrize("module,status", [(materialization, materialization.STATUS), (probe, probe.STATUS)])
def test_lock_outputs_are_nonoverwriting(tmp_path: Path, module, status: str) -> None:
    path = tmp_path / "lock.json"
    module.write_lock(tmp_path, path, {"status": status})
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        module.write_lock(tmp_path, path, {"status": "changed"})
    assert path.read_bytes() == before


def test_materialization_lock_requires_clean_tree_before_protocol_access(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(materialization.common, "_git", lambda *_: "?? dirty.py")
    monkeypatch.setattr(
        materialization,
        "load_protocol",
        lambda *_: pytest.fail("Protocol/input access preceded clean-tree check"),
    )
    with pytest.raises(RuntimeError, match="clean committed"):
        materialization.build_payload(tmp_path)


def test_checked_bytes_rejected_before_json_decode(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b"not-json")
    item = {"path": "bad.json", "sha256": "0" * 64, "size_bytes": 8}
    with pytest.raises(RuntimeError, match="changed before decoding"):
        probe.checked(tmp_path, item)
