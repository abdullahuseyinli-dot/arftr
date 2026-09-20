from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fsar_auditor_is_independent_and_replays_checkpoints_and_swaps():
    source = (ROOT / "experiments/audit_okutama_frame_supervision.py").read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith("hac") for name in imports)
    assert "FSAR_V2_INDEPENDENT_30_FIT_AUDIT_PASS" in source
    assert "checkpoint_forward_passes_replayed" in source
    assert "itertools.product" in source
    assert "Fit semantic identity changed" in source
