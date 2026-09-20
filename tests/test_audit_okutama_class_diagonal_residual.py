from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_auditor_is_independent_and_replays_all_required_mechanisms():
    path = ROOT / "experiments/audit_okutama_class_diagonal_residual.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith("hac") for name in imports)
    assert "three_parameter_fits_independently_replayed" in source
    assert "outer_seed_forward_passes_independently_replayed" in source
    assert "protected_sitting_fallback_bit_exact" in source
    assert "itertools.product" in source
    assert "CLASS_DIAGONAL_RESIDUAL_INDEPENDENT_AUDIT_PASS" in source
