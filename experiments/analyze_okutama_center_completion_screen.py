"""Recalculate paired forensics for the label-blind completion screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260913/center_evidence_completion_screen_v1"
OUTPUT = (
    ROOT / ".runs/research_20260913/center_evidence_completion_screen_forensics_v1"
)
EXPECTED = {
    "summary.json": "002ad504f36de7e59ac5d345abacbaff548eca9fac718e1661dca95dad0306cd",
    "oof_metrics.npz": "6e6bf70e31d011ce69a5e359ee9beb572e758a41e3c5190b22ef9498cdb54b40",
    "per_center_oof.csv": "dce86e36d3186df3a415e124f8bbc76e728b17d8a17a41b1b3762a933069c355",
    "request.json": "7212655af4d7c0d0ad0fd9f73c6fce96dffb61e9708f177adc2c73672a9459ad",
}
BOOTSTRAPS = 100_000
PERMUTATIONS = 200_000
SEED = 20_260_913


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Forensics output already exists")
    observed = {name: sha256_file(RUN / name) for name in EXPECTED}
    if observed != EXPECTED:
        raise RuntimeError("Scientific screen artifacts changed before forensics")
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    if (
        summary["status"] != "CENTER_COMPLETION_RECONSTRUCTION_SCREEN_COMPLETE"
        or summary["scientific"] is not True
        or summary["gates"]["all_pass"] is not False
        or summary["task_training_authorized"] is not False
        or summary["retained_arftr_changed"] is not False
    ):
        raise RuntimeError("Scientific result boundary changed")
    saved = np.load(RUN / "oof_metrics.npz", allow_pickle=False)
    sample_ids = saved["sample_ids"].astype(str)
    scenarios = saved["scenarios"].astype(str)
    folds = saved["held_fold"].astype(np.int64)

    def metric(name: str, kind: str = "cosine") -> np.ndarray:
        values = saved[f"{name}__{kind}"].astype(np.float64)
        if values.shape != (128,) or not np.isfinite(values).all():
            raise RuntimeError(f"OOF metric is incomplete: {name}/{kind}")
        return values

    p0 = metric("P0_center_only")
    p1 = metric("P1_unordered_pool")
    p2 = metric("P2_same_grid_pool")
    p3 = metric("P3_partial_transport")
    gate = metric("P3_partial_transport", "gate")
    wrong = metric("wrong_track")
    repeated = metric("repeated_masked_center")
    spatial = metric("spatial_reassignment")
    reversed_time = metric("time_reversal")
    improvement = p2 - p3
    rng = np.random.default_rng(SEED)
    bootstrap_rows = rng.integers(0, len(p2), size=(BOOTSTRAPS, len(p2)))
    bootstrap_relative = (
        p2[bootstrap_rows].mean(axis=1) - p3[bootstrap_rows].mean(axis=1)
    ) / p2[bootstrap_rows].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(PERMUTATIONS, len(p2)))
    permutation_means = (signs * improvement).mean(axis=1)
    permutation_p = (np.sum(permutation_means >= improvement.mean()) + 1) / (
        PERMUTATIONS + 1
    )
    oracle = np.minimum(p2, p3)
    spearman = stats.spearmanr(gate, improvement)
    results = {
        "status": "CENTER_COMPLETION_SCREEN_PAIRED_FORENSICS_COMPLETE",
        "scope": "post-hoc label-blind reconstruction diagnostics; not a task or deployable result",
        "input_sha256": observed,
        "population": {
            "centers": len(sample_ids),
            "unique_sample_ids": len(set(sample_ids)),
            "scenarios": len(set(scenarios)),
            "folds": len(set(folds.tolist())),
        },
        "mean_center_cosine_error": {
            "P0_center_only": float(p0.mean()),
            "P1_unordered_pool": float(p1.mean()),
            "P2_same_grid_pool": float(p2.mean()),
            "P3_partial_transport": float(p3.mean()),
            "wrong_track": float(wrong.mean()),
            "repeated_masked_center": float(repeated.mean()),
            "spatial_reassignment": float(spatial.mean()),
            "time_reversal": float(reversed_time.mean()),
        },
        "P3_vs_P2": {
            "absolute_error_reduction": float(improvement.mean()),
            "relative_error_reduction": float(improvement.mean() / p2.mean()),
            "bootstrap_seed": SEED,
            "bootstrap_replicates": BOOTSTRAPS,
            "bootstrap_95_percent_relative_reduction": [
                float(np.quantile(bootstrap_relative, 0.025)),
                float(np.quantile(bootstrap_relative, 0.975)),
            ],
            "paired_sign_permutation_replicates": PERMUTATIONS,
            "paired_sign_permutation_p_one_sided": float(permutation_p),
            "wilcoxon_p_one_sided": float(
                stats.wilcoxon(improvement, alternative="greater").pvalue
            ),
            "centers_better": int((improvement > 0).sum()),
            "centers_equal": int((improvement == 0).sum()),
            "centers_worse": int((improvement < 0).sum()),
            "sign_test_p_one_sided": float(
                stats.binomtest(
                    int((improvement > 0).sum()),
                    int((improvement != 0).sum()),
                    0.5,
                    alternative="greater",
                ).pvalue
            ),
            "per_center_oracle_relative_reduction_vs_P2": float(
                (p2.mean() - oracle.mean()) / p2.mean()
            ),
            "per_center_router_can_reach_five_percent_gate": bool(
                (p2.mean() - oracle.mean()) / p2.mean() >= 0.05
            ),
            "spearman_mean_gate_vs_P3_improvement": {
                "rho": _finite(spearman.statistic),
                "p_value": _finite(spearman.pvalue),
            },
        },
        "relative_reduction_by_fold": {
            str(fold): float(
                (p2[folds == fold].mean() - p3[folds == fold].mean())
                / p2[folds == fold].mean()
            )
            for fold in sorted(set(folds.tolist()))
        },
        "relative_reduction_by_scenario": {
            scenario: float(
                (p2[scenarios == scenario].mean() - p3[scenarios == scenario].mean())
                / p2[scenarios == scenario].mean()
            )
            for scenario in sorted(set(scenarios))
        },
        "control_absolute_degradation_vs_P3": {
            "wrong_track": float(wrong.mean() - p3.mean()),
            "repeated_masked_center": float(repeated.mean() - p3.mean()),
            "spatial_reassignment": float(spatial.mean() - p3.mean()),
            "time_reversal": float(reversed_time.mean() - p3.mean()),
        },
        "locked_gate_verdict": summary["gates"],
        "interpretive_limits": [
            "The confidence interval quantifies paired center variability in this adaptive 128-center cohort; it is not a new untouched-dataset interval.",
            "The per-center oracle is diagnostic and not deployable.",
            "The failed five-percent P3-versus-P2 gate may not be relaxed post hoc.",
            "No action labels, ARFTR probabilities, pose outputs, or support categories were opened.",
        ],
    }
    OUTPUT.mkdir(parents=True)
    write_json_exclusive(OUTPUT / "forensics.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
