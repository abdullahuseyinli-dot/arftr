"""Label-blind schema preflight for the ARFTR transition-risk router.

This command reads only identifiers, folds, arm names, probabilities, and
intervention masks from the completed source/posture OOF artifact. It never
loads labels and performs no fit or threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.actor_memory_base import file_sha256
from hac.transition_risk_router import router_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / ".runs/research_20260913/source_posture_fixed_primary_v1/results/v0001/oof_probabilities.npz"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/preflight.json"


def run(source: Path, output: Path) -> dict[str, object]:
    with np.load(source, allow_pickle=False) as saved:
        required = {"sample_ids", "folds", "arms", "probabilities", "interventions"}
        missing = sorted(required - set(saved.files))
        if missing:
            raise RuntimeError(f"OOF artifact is missing label-blind fields: {missing}")
        sample_ids = saved["sample_ids"]
        folds = saved["folds"]
        arms = saved["arms"]
        probabilities = saved["probabilities"]
        interventions = saved["interventions"]
        if probabilities.ndim != 3 or probabilities.shape[2] != 3:
            raise RuntimeError("Probability array must have shape [arms, rows, 3]")
        names = [str(value) for value in arms.tolist()]
        if "A0_retain" not in names or "R_parts" not in names:
            raise RuntimeError("Required anchor/candidate arms are absent")
        anchor_index = names.index("A0_retain")
        candidate_index = names.index("R_parts")
        anchor = probabilities[anchor_index]
        candidate = probabilities[candidate_index]
        if len(sample_ids) != len(folds) or len(sample_ids) != len(anchor):
            raise RuntimeError("Identity, fold, and probability rows are not aligned")
        if len(np.unique(sample_ids)) != len(sample_ids) or not np.array_equal(np.unique(folds), np.arange(5)):
            raise RuntimeError("Locked identity/fold contract changed")
        features, feature_names = router_features(anchor, candidate)
        if interventions.shape != (len(names), len(sample_ids)):
            raise RuntimeError("Intervention mask shape changed")
    receipt: dict[str, object] = {
        "status": "TRANSITION_RISK_ROUTER_LABEL_BLIND_PREFLIGHT_PASS",
        "source": str(source.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "source_sha256": file_sha256(source),
        "rows": int(len(sample_ids)),
        "outer_folds": sorted({int(value) for value in folds.tolist()}),
        "anchor": "A0_retain",
        "candidate": "R_parts",
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names,
        "max_abs_feature": float(np.max(np.abs(features))),
        "outer_label_reads": 0,
        "fit_performed": False,
        "threshold_selected": False,
        "annotation_support_used": False,
        "intervention_masks_loaded": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
