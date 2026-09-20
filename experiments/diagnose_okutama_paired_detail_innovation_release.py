"""Post-release diagnostics for the completed PDI experiment.

This analysis deliberately reads outer labels only after the prospective result
has been released.  It separates producer capacity from verifier behavior and
writes a clearly marked diagnostic artifact; it cannot affect promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import run_okutama_paired_detail_innovation as runner
from hac.actor_memory_base import file_sha256


def _transition(labels: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    anchor_correct = anchor.argmax(1) == labels
    candidate_correct = candidate.argmax(1) == labels
    rescues = int((candidate_correct & ~anchor_correct).sum())
    harms = int((~candidate_correct & anchor_correct).sum())
    return {"rescues": rescues, "harms": harms, "net": rescues - harms}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"immutable diagnostic differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(runner.ROOT.resolve())
    summary = run / "summary_v2.json"
    if not summary.is_file():
        raise RuntimeError("versioned result must be released before post-hoc diagnosis")

    data = runner.data()
    labels = data["labels"]
    anchor = data["anchor"]
    anchor_class = anchor.argmax(1)
    anchor_wrong = anchor_class != labels
    result: dict[str, Any] = {
        "status": "PDI_POSTHOC_OUTER_LABEL_DIAGNOSTIC_COMPLETE",
        "diagnostic_only": True,
        "outer_labels_read_after_result_release": True,
        "promotion_eligible": False,
        "summary_v2_sha256": file_sha256(summary),
        "anchor_errors": int(anchor_wrong.sum()),
        "arms": {},
    }
    for arm in runner.ARMS:
        arrays: dict[str, np.ndarray] = {}
        for name in ("raw_probabilities", "actions", "routed_probabilities", "choices"):
            shape = (len(labels), 4, 3) if name == "actions" else (
                (len(labels), 3) if name.endswith("probabilities") else (len(labels),)
            )
            arrays[name] = np.empty(shape, dtype=np.float64 if name != "choices" else np.int64)
        for outer in range(5):
            saved = runner._load_predictions(run / arm / f"fold-{outer}" / "predictions.npz")
            rows = saved["held_rows"]
            for name in arrays:
                arrays[name][rows] = saved[name]

        actions = arrays["actions"]
        choices = arrays["choices"].astype(np.int64)
        routed = arrays["routed_probabilities"]
        action_class = actions.argmax(2)
        selected = anchor.copy()
        active_rows = np.flatnonzero(choices != 0)
        selected[active_rows] = actions[active_rows, choices[active_rows]]
        if not np.allclose(selected, routed, atol=0, rtol=0):
            raise RuntimeError(f"stored route does not match choices for {arm}")

        reachable = anchor_wrong & (action_class[:, 1:] == labels[:, None]).any(1)
        any_class_flip = (action_class[:, 1:] != anchor_class[:, None]).any(1)
        selected_class_flip = routed.argmax(1) != anchor_class
        chosen_nonretain = choices != 0
        chosen_nll_delta = np.log(np.clip(routed[np.arange(len(labels)), labels], 1e-12, 1.0)) - np.log(
            np.clip(anchor[np.arange(len(labels)), labels], 1e-12, 1.0)
        )

        oracle = anchor.copy()
        for row in np.flatnonzero(reachable):
            choice = np.flatnonzero(action_class[row, 1:] == labels[row])[0] + 1
            oracle[row] = actions[row, choice]

        fixed_actions = {}
        for action in range(1, 4):
            fixed_actions[str(action)] = {
                "metrics": runner.metrics(labels, actions[:, action]),
                "transitions": _transition(labels, anchor, actions[:, action]),
                "class_flips": int((action_class[:, action] != anchor_class).sum()),
            }
        result["arms"][arm] = {
            "choice_counts": {str(action): int((choices == action).sum()) for action in range(4)},
            "nonretain_choices": int(chosen_nonretain.sum()),
            "rows_with_any_proposed_class_flip": int(any_class_flip.sum()),
            "selected_class_flips": int(selected_class_flip.sum()),
            "reachable_anchor_errors": int(reachable.sum()),
            "reachable_errors_selected_as_class_flip": int((reachable & selected_class_flip).sum()),
            "reachable_errors_retained_or_same_class": int((reachable & ~selected_class_flip).sum()),
            "selected_nll_improvements": int((chosen_nll_delta > 0).sum()),
            "selected_nll_regressions": int((chosen_nll_delta < 0).sum()),
            "mean_selected_nll_utility": float(chosen_nll_delta.mean()),
            "routed_metrics": runner.metrics(labels, routed),
            "routed_transitions": _transition(labels, anchor, routed),
            "raw_metrics": runner.metrics(labels, arrays["raw_probabilities"]),
            "raw_transitions": _transition(labels, anchor, arrays["raw_probabilities"]),
            "fixed_actions": fixed_actions,
            "oracle_reachable_metrics": runner.metrics(labels, oracle),
        }

    path = run / "posthoc_release_diagnostics_v1.json"
    _write_new(path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
