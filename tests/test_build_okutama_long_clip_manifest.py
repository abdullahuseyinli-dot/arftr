from __future__ import annotations

import json

import build_okutama_long_clip_manifest as cli
import pytest


def test_cli_declares_lock_without_materializing(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(
        cli, "create_materialization_lock", lambda *args: seen.append(args) or {"status": "locked"}
    )
    monkeypatch.setattr(
        cli,
        "build_long_manifest",
        lambda **args: pytest.fail("Declaration attempted annotation access"),
    )
    cli.main(
        [
            "--write-source-lock",
            str(tmp_path / "lock.json"),
            "--archive",
            str(tmp_path / "TrainSetFrames.zip"),
        ]
    )
    assert len(seen) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "locked"}


def test_cli_materializes_only_from_existing_lock(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(
        cli, "build_long_manifest", lambda **args: seen.append(args) or {"status": "complete"}
    )
    monkeypatch.setattr(
        cli,
        "create_materialization_lock",
        lambda *args: pytest.fail("Manifest silently created a lock"),
    )
    cli.main(
        ["--source-lock", str(tmp_path / "lock.json"), "--output-dir", str(tmp_path / "manifest")]
    )
    assert seen[0]["source_lock_path"] == tmp_path / "lock.json"
    assert seen[0]["output_dir"] == tmp_path / "manifest"
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--write-source-lock", "lock.json"],
        [
            "--write-source-lock",
            "lock.json",
            "--archive",
            "TrainSetFrames.zip",
            "--output-dir",
            "manifest",
        ],
        ["--source-lock", "lock.json"],
        [
            "--source-lock",
            "lock.json",
            "--output-dir",
            "manifest",
            "--archive",
            "TrainSetFrames.zip",
        ],
        ["--source-lock", "lock.json", "--write-source-lock", "other.json"],
    ],
)
def test_cli_rejects_mixed_declaration_and_access_modes(args, monkeypatch):
    monkeypatch.setattr(
        cli, "build_long_manifest", lambda **kw: pytest.fail("Invalid CLI materialized")
    )
    monkeypatch.setattr(
        cli, "create_materialization_lock", lambda *a: pytest.fail("Invalid CLI declared")
    )
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
