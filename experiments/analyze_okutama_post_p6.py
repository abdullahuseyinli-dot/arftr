"""Read-only analysis of retained P1-P6 evidence; write a fresh review directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/research_20260907"
PHASE_HASHES = {
    "p1": "6f69baac232a76df89096e29f1f697cc33e34f048c30fd34cbb28dff41ec6096",
    "p2a": "bf7d8311887dfc460b25fd00e3baa97911f07ef8b1f95ebd436072350602f63a",
    "p3": "262418ab23e1c39494fa4e88d70f517f36be2b5f05a23cd7f122317aaa2f8f89",
    "p4": "38b8a442e7f52151b3ee097d9426b18bc365383b1c8c6fe0206884b2c51bc33a",
    "p5": "8981f14c9a68640a7819e5ecfe8d9a3221dc3f629afa1ca81de0172f3e9cd7ab",
    "p6": "61fc852d1da29670032e8e01356d7231bc7801437e9478a80ca47484eaa47256",
}
INDEX_HASH = "5bf6d0cc11a18d3e986b714f8a3de71cadd9fef3bd233eb9b5f212f701aadbbb"
SELECTED = {
    "p1": "vjepa21_real_clip__linear",
    "p2a": "video_dino_mean_multinomial",
    "p3": "dual_scale_factorized",
    "p4": "visual_comp_track_factorized",
    "p5": "spatial_contrast_factorized",
    "p6": "ocvc_uniform_diverse_triad",
}


def checked_bytes(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError(f"Retained evidence changed: {path}")
    return content


def macro_f1(cm: np.ndarray) -> float:
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.divide(2 * cm.diagonal(), denominator, out=np.zeros(3), where=denominator > 0).mean())


def confusion(labels: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.bincount(3 * labels + prediction, minlength=9).reshape(3, 3)


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output: Path) -> dict:
    output = output.resolve()
    if not output.is_relative_to(RUN) or output == RUN or output.exists():
        raise RuntimeError("Use a new dedicated directory below .runs/research_20260907")
    summaries, arrays, receipts = {}, {}, {}
    for phase, digest in PHASE_HASHES.items():
        folder = RUN / f"okutama_native_video_{phase}/results"
        summaries[phase] = json.loads(checked_bytes(folder / "summary.json", digest))
        receipts[f"{phase}/summary.json"] = digest
        if phase in {"p3", "p5", "p6"}:
            npz_hash = summaries[phase]["artifacts"]["oof_probabilities.npz"]
            checked_bytes(folder / "oof_probabilities.npz", npz_hash)
            with np.load(folder / "oof_probabilities.npz", allow_pickle=False) as data:
                arrays[phase] = {key: data[key].copy() for key in data.files}
            receipts[f"{phase}/oof_probabilities.npz"] = npz_hash
    p3, p5, p6 = (arrays[phase] for phase in ("p3", "p5", "p6"))
    for other in (p3, p5):
        for key in ("sample_ids", "labels", "recording_ids", "folds", "long_valid"):
            if not np.array_equal(p6[key], other[key]):
                raise RuntimeError(f"OOF alignment mismatch: {key}")
    labels, groups = p6["labels"], p6["recording_ids"]
    if len(labels) != 4977 or len(np.unique(groups)) != 11:
        raise RuntimeError("Review population changed")
    index = {}
    content = checked_bytes(RUN / "cptr_replay_r0/eligible_index.csv", INDEX_HASH)
    for row in csv.DictReader(content.decode("utf-8").splitlines()):
        if row["scope"] == "grouped_crossfit_oof":
            if row["sample_id"] in index:
                raise RuntimeError("Duplicate eligible identity")
            index[row["sample_id"]] = row
    ordered = [index[str(sample)] for sample in p6["sample_ids"]]
    if any(int(row["label_index"]) != int(label) for row, label in zip(ordered, labels, strict=True)):
        raise RuntimeError("Eligible label mismatch")
    for row, group in zip(ordered, groups, strict=True):
        if row["recording_id"] != str(group):
            raise RuntimeError("Eligible scenario mismatch")
    occluded = np.asarray([row["window_any_occluded"].lower() == "true" for row in ordered])
    transition = np.asarray([row["transition_window"].lower() == "true" for row in ordered])
    probabilities = p6[SELECTED["p6"]]
    prediction = probabilities.argmax(axis=1)
    cm = confusion(labels, prediction)
    np.testing.assert_array_equal(cm, summaries["p6"]["models"][SELECTED["p6"]]["metrics"]["confusion"])
    errors = prediction != labels
    component_predictions = np.stack([
        p3["long_vjepa_mean"].argmax(axis=1),
        p3["dual_scale_vjepa_dino"].argmax(axis=1),
        p5["orthogonal_moments_factorized"].argmax(axis=1),
    ])
    all_wrong = np.all(component_predictions != labels, axis=0)
    unanimous = np.all(component_predictions == component_predictions[0], axis=0)
    strata = {}
    for name, mask in {
        "occluded": occluded, "clear": ~occluded,
        "transition": transition, "stable": ~transition,
        "clear_and_stable": ~occluded & ~transition,
        "long_valid": p6["long_valid"], "short_fallback": ~p6["long_valid"],
        "unanimous_components": unanimous, "disagreeing_components": ~unanimous,
    }.items():
        strata[name] = {"rows": int(mask.sum()), "errors": int((mask & errors).sum()),
                        "macro_f1": macro_f1(confusion(labels[mask], prediction[mask]))}
    phase_rows = [{"phase": "Original", "arm": "distinct_frame_teacher", **{
        key: summaries["p6"]["baseline"][key] for key in ("macro_f1", "accuracy", "nll", "brier")}}]
    for phase, arm in SELECTED.items():
        phase_rows.append({"phase": phase.upper(), "arm": arm, **{
            key: summaries[phase]["models"][arm]["metrics"][key]
            for key in ("macro_f1", "accuracy", "nll", "brier")}})
    scenario_rows = []
    for group in np.unique(groups):
        mask = groups == group
        scenario_rows.append({"scenario": str(group), "rows": int(mask.sum()),
            "errors": int((mask & errors).sum()),
            "sitting_support": int((mask & (labels == 0)).sum()),
            "macro_f1": macro_f1(confusion(labels[mask], prediction[mask])),
            "p3_macro_f1": macro_f1(confusion(labels[mask], p3["dual_scale_factorized"][mask].argmax(axis=1))),
            "p5_macro_f1": macro_f1(confusion(labels[mask], p5["spatial_contrast_factorized"][mask].argmax(axis=1)))})
    scenario_rows.sort(key=lambda row: row["errors"], reverse=True)
    repairs = {}
    for target in (0.84, 0.85):
        lo, hi = 0.0, 1.0
        off_diagonal = cm - np.diag(cm.diagonal())
        for _ in range(60):
            fraction = (lo + hi) / 2
            repaired = cm.astype(float) - fraction * off_diagonal
            repaired += np.diag(fraction * off_diagonal.sum(axis=1))
            if macro_f1(repaired) < target:
                lo = fraction
            else:
                hi = fraction
        repairs[str(target)] = {"fraction": hi, "fractional_errors_repaired": hi * int(errors.sum())}
    result = {
        "status": "POST_P6_DESCRIPTIVE_REVIEW_COMPLETE", "rows": len(labels), "scenarios": 11,
        "phase_metrics": phase_rows, "confusion": cm.tolist(), "errors": int(errors.sum()),
        "standing_locomotion_errors": int(cm[1, 2] + cm[2, 1]), "strata": strata,
        "component_error_partition": {"all_three_wrong": int(all_wrong.sum()),
            "p6_errors_all_three_wrong": int((all_wrong & errors).sum()),
            "p6_errors_any_component_correct": int((~all_wrong & errors).sum()),
            "p6_correct_all_three_wrong": int((all_wrong & ~errors).sum())},
        "scenarios_by_error_count": scenario_rows,
        "proportional_no_harm_repair_thought_experiment": repairs,
        "repair_interpretation": "Fractional no-harm confusion-matrix thought experiment, not a forecast or minimum-error bound",
        "selection_caveat": "P6 was selected after a retrospective ensemble screen; this analysis adds no independent confirmation",
        "inputs": {**receipts, "eligible_index.csv": INDEX_HASH},
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_fits": 0, "raw_images_read": 0, "protected_rows_read": 0,
    }
    output.mkdir(parents=True)
    csv_write(output / "phase_metrics.csv", phase_rows)
    csv_write(output / "scenario_metrics.csv", scenario_rows)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    scores = [100 * row["macro_f1"] for row in phase_rows]
    colors = ["#9aa5b1"] + ["#336e99"] * 5 + ["#16836c"]
    axes[0].bar([row["phase"] for row in phase_rows], scores, color=colors)
    axes[0].set_ylim(65, 88)
    axes[0].axhline(82, color="#876026", linestyle="--", linewidth=1)
    axes[0].axhline(85, color="#876026", linestyle=":", linewidth=1)
    for i, score in enumerate(scores):
        axes[0].text(i, score + .4, f"{score:.2f}", ha="center", fontsize=10)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[0].set_title("Retained development results\nP4 shows its best added-trajectory arm")
    axes[1].imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        axes[1].text(j, i, str(value), ha="center", va="center", color="white" if value > 900 else "#1a2734", fontsize=14)
    names = ["Sitting", "Standing", "Walk / run"]
    axes[1].set_xticks(range(3), names)
    axes[1].set_yticks(range(3), names)
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    axes[1].set_title(f"P6 remaining errors: {errors.sum()}\nStanding / locomotion boundary: {cm[1, 2] + cm[2, 1]}")
    fig.suptitle("Okutama: 4,977 centers, 11 reused development scenarios", fontsize=15)
    fig.supxlabel("P6 was selected after inspecting these OOF results. These are adaptive development estimates.", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"evidence_overview.{extension}", dpi=180)
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = analyze(parser.parse_args().output_dir)
    print(json.dumps({key: result[key] for key in ("status", "errors", "strata", "component_error_partition", "proportional_no_harm_repair_thought_experiment")}, indent=2))
