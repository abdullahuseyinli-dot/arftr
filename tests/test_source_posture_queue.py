from __future__ import annotations

import pytest
import run_okutama_source_posture_queue as queue


@pytest.mark.parametrize("completed", [[], [0], [0, 1, 2, 3, 4]])
def test_source_posture_queue_uses_only_its_run_directory(tmp_path, monkeypatch, completed):
    monkeypatch.setattr(queue, "RUN", tmp_path)
    for fold in completed:
        folder = tmp_path / f"fold-{fold}"
        folder.mkdir()
        (folder / "receipt.json").write_text("{}", encoding="utf-8")
    result = queue.execute(dry_run=True)
    assert result["folds_complete_before_start"] == len(completed)
    assert result["pending_folds"] == [fold for fold in range(5) if fold not in completed]
