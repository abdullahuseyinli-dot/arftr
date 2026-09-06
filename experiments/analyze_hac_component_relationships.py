"""Exploratory component and error relationships from four immutable development exports.

No fitting, routing, checkpoint access, or new evaluation split is involved. Intervals
condition on historical candidate selection and fitted folds; they are not confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("sitting", "standing", "walking_running")
METHODS = {
    "flat": "dino_flat_probability_stack",
    "factorized": "dino_factorized_probability_stack",
    "linear_mixed": "dino_siglip_linear_svm_control",
    "full_mixed": "dino_siglip_factorized_reliability_stack",
}
SOURCES = {
    "vcoco_rows": (
        ".runs/research_20260906/vcoco_development_evidence/development_joined.csv",
        "ca32a6cd9f60524d9cbb122dce969ebede167cd6900d5772c757f1131a94ec9b",
    ),
    "vcoco_summary": (
        ".runs/research_20260906/vcoco_development_evidence/summary.json",
        "1a5e7c4d307e6fa872a8fc25a352502bda93b35583cfb24c10c702c062dbf7e3",
    ),
    "cptr_rows": (
        ".runs/cptr/baseline_preservation_diagnostic/paired_rows.csv",
        "4be1537f19fba6f4524618485da65044d1645228ae56fbbc1a11e4a64fbfdc46",
    ),
    "cptr_summary": (
        ".runs/cptr/baseline_preservation_diagnostic/summary.json",
        "32dcf384865983651c317fe961cfe6187013b13f6910df83330f804f6ee85e73",
    ),
}
CONTRASTS = (
    ("factorized", "flat"),
    ("linear_mixed", "flat"),
    ("full_mixed", "flat"),
    ("full_mixed", "factorized"),
    ("full_mixed", "linear_mixed"),
)


def locked_bytes(path: Path, digest: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Development export hash mismatch: {path.name}")
    return content


def probabilities(frame: pd.DataFrame, prefix: str, separator: str) -> np.ndarray:
    values = frame[[f"{prefix}{separator}p_{label}" for label in CLASSES]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values < -1e-6).any()
        or (values > 1 + 1e-6).any()
        or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("Invalid retained probability matrix")
    return values


def macro_f1(confusion: np.ndarray) -> np.ndarray:
    """Fixed three-class F1; missing denominators contribute zero."""
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-2) + confusion.sum(axis=-1)
    scores = np.divide(
        2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0
    )
    return scores.mean(axis=-1)


def metrics(labels: np.ndarray, values: np.ndarray) -> dict:
    pred = values.argmax(axis=1)
    confusion = np.bincount(3 * labels + pred, minlength=9).reshape(3, 3)
    nll = -np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1))
    return {
        "macro_f1": float(macro_f1(confusion)),
        "accuracy": float(np.mean(pred == labels)),
        "nll": float(nll.mean()),
        "confusion": confusion.tolist(),
    }


def interval(point: float, draws: np.ndarray) -> dict:
    valid = draws[np.isfinite(draws)]
    bounds = np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else [None, None]
    return {
        "point": float(point),
        "ci95": bounds,
        "valid_draws": len(valid),
        "invalid_draws": int(len(draws) - len(valid)),
    }


def clustered_model_draws(labels, models, group_index, indices):
    """Bootstrap sufficient statistics, preserving all rows in each sampled group."""
    groups = int(group_index.max()) + 1
    counts = np.bincount(group_index, minlength=groups)
    sufficient = []
    for values in models.values():
        confusions = np.bincount(
            group_index * 9 + 3 * labels + values.argmax(axis=1), minlength=groups * 9
        ).reshape(groups, 9)
        losses = np.bincount(
            group_index,
            weights=-np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1)),
            minlength=groups,
        )
        sufficient.append(np.column_stack((confusions, losses)))
    statistics = np.column_stack((counts, *sufficient)).astype(float)
    output = np.empty((len(indices), len(models), 2))
    for start in range(0, len(indices), 64):
        batch = indices[start : start + 64]
        weights = np.zeros((len(batch), groups))
        for j, row in enumerate(batch):
            weights[j] = np.bincount(row, minlength=groups)
        totals = weights @ statistics
        for model in range(len(models)):
            offset = 1 + 10 * model
            output[start : start + len(batch), model, 0] = macro_f1(
                totals[:, offset : offset + 9].reshape(-1, 3, 3)
            )
            output[start : start + len(batch), model, 1] = totals[:, offset + 9] / totals[:, 0]
    return output


def error_overlap(labels, baseline, candidate) -> dict:
    b = baseline.argmax(axis=1) == labels
    c = candidate.argmax(axis=1) == labels
    rescue, harm = int((~b & c).sum()), int((b & ~c).sum())
    shared = int((~b & ~c).sum())
    return {
        "rows": len(labels),
        "both_correct": int((b & c).sum()),
        "rescued": rescue,
        "harmed": harm,
        "both_wrong": shared,
        "rescue_fraction_of_baseline_errors": rescue / int((~b).sum()) if (~b).any() else None,
        "harm_fraction_of_baseline_correct": harm / int(b.sum()) if b.any() else None,
        "shared_fraction_of_baseline_errors": shared / int((~b).sum()) if (~b).any() else None,
        "prediction_disagreement_fraction": float(
            np.mean(baseline.argmax(1) != candidate.argmax(1))
        ),
        "either_correct_accuracy_oracle": float(np.mean(b | c)),
        "oracle_note": "Label-informed accuracy ceiling of selecting these two argmax outputs; not a macro-F1 bound, not deployable, and not evidence of a learnable router.",
    }


def subgroup_contrast(labels, baseline, candidate, groups, indices, indicator):
    """Difference in paired mean effects: subgroup minus complement, same cluster draws."""
    count_groups = int(groups.max()) + 1
    delta_error = (candidate.argmax(1) != labels).astype(float) - (baseline.argmax(1) != labels)
    delta_nll = -np.log(np.clip(candidate[np.arange(len(labels)), labels], 1e-12, 1))
    delta_nll += np.log(np.clip(baseline[np.arange(len(labels)), labels], 1e-12, 1))
    stats = []
    for mask in (indicator, ~indicator):
        stats.extend(
            [
                np.bincount(groups[mask], minlength=count_groups),
                *[
                    np.bincount(groups[mask], weights=d[mask], minlength=count_groups)
                    for d in (delta_error, delta_nll)
                ],
            ]
        )
    stats = np.column_stack(stats)
    totals = stats[indices].sum(axis=1)
    valid = (totals[:, 0] > 0) & (totals[:, 3] > 0)
    result = {}
    for col, (name, delta) in enumerate((("error_rate", delta_error), ("nll", delta_nll)), 1):
        draws = np.full(len(indices), np.nan)
        draws[valid] = (
            totals[valid, col] / totals[valid, 0] - totals[valid, col + 3] / totals[valid, 3]
        )
        result[name] = interval(delta[indicator].mean() - delta[~indicator].mean(), draws)
    result["orientation"] = (
        "(candidate-minus-baseline in subgroup) minus (candidate-minus-baseline in complement); positive means relatively more harm in subgroup"
    )
    return result


def descriptive_slices(dataset, frame, labels, baseline, candidate, group_column, masks):
    rows = []
    for name, mask in masks.items():
        selected = labels[mask]
        if not len(selected):
            continue
        base, cand = metrics(selected, baseline[mask]), metrics(selected, candidate[mask])
        overlap = error_overlap(selected, baseline[mask], candidate[mask])
        single_true_class = len(np.unique(selected)) == 1
        rows.append(
            {
                "dataset": dataset,
                "slice": name,
                "rows": len(selected),
                "groups": int(frame.loc[mask, group_column].nunique()),
                "sitting": int((selected == 0).sum()),
                "standing": int((selected == 1).sum()),
                "walking_running": int((selected == 2).sum()),
                "macro_f1_delta": None
                if single_true_class
                else cand["macro_f1"] - base["macro_f1"],
                "single_class_recall_delta": cand["accuracy"] - base["accuracy"]
                if single_true_class
                else None,
                "metric_scope": "true_class_recall_only"
                if single_true_class
                else "fixed_three_class_macro_f1",
                "accuracy_delta": cand["accuracy"] - base["accuracy"],
                "nll_delta": cand["nll"] - base["nll"],
                "rescued": overlap["rescued"],
                "harmed": overlap["harmed"],
            }
        )
    return rows


def class_standardized_contrast(labels, baseline, candidate, groups, indices, indicator):
    """Fix class weights to cohort prevalence; remaining confounding is not removed."""
    group_count = indices.shape[1]
    weights = np.bincount(labels, minlength=3) / len(labels)
    error = (candidate.argmax(1) != labels).astype(float) - (baseline.argmax(1) != labels)
    nll = -np.log(np.clip(candidate[np.arange(len(labels)), labels], 1e-12, 1))
    nll += np.log(np.clip(baseline[np.arange(len(labels)), labels], 1e-12, 1))
    stats = np.zeros((group_count, 2, 3, 3))
    for arm, mask in enumerate((indicator, ~indicator)):
        np.add.at(stats[:, arm, :, 0], (groups[mask], labels[mask]), 1)
        np.add.at(stats[:, arm, :, 1], (groups[mask], labels[mask]), error[mask])
        np.add.at(stats[:, arm, :, 2], (groups[mask], labels[mask]), nll[mask])
    totals = stats[indices].sum(axis=1)
    observed = stats.sum(axis=0)
    if (observed[:, :, 0] == 0).any():
        raise ValueError("Class standardization requires each true class in both observed arms")
    valid = (totals[:, :, :, 0] > 0).all(axis=(1, 2))
    result = {}
    for col, name in ((1, "error_rate"), (2, "nll")):
        draws = np.full(len(indices), np.nan)
        per_class = totals[valid, :, :, col] / totals[valid, :, :, 0]
        draws[valid] = (per_class[:, 0] - per_class[:, 1]) @ weights
        observed_means = observed[:, :, col] / observed[:, :, 0]
        result[name] = interval((observed_means[0] - observed_means[1]) @ weights, draws)
    result["class_weights"] = weights.tolist()
    result["interpretation"] = (
        "Occluded-minus-clear paired effect standardized to fixed full-cohort true-class prevalence; observational, not adjusted for scenario, confidence, or other confounders. Require all three classes in both arms for a valid draw."
    )
    return result


def confidence_masks(baseline):
    """Fixed confidence intervals, no outcome-dependent threshold selection."""
    confidence = baseline.max(axis=1)
    return {
        f"base_confidence_{lo}_{hi}": (confidence >= lo) & (confidence < hi)
        for lo, hi in ((0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.000001))
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def plot_results(summary, slices, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    comparisons = summary["vcoco"]["contrasts"]
    labels = [
        "DINO factorization − flat",
        "Mixed linear − flat",
        "Full mixed − flat",
        "Full mixed − DINO factorized",
        "Full mixed − mixed linear",
    ]
    for y, ((_, effect), _label) in enumerate(zip(comparisons.items(), labels, strict=True)):
        value = effect["macro_f1"]
        low, high = value["ci95"]
        axes[0].errorbar(
            value["point"] * 100,
            y,
            xerr=[[100 * (value["point"] - low)], [100 * (high - value["point"])]],
            fmt="o",
            color="#2166ac",
            capsize=3,
        )
    axes[0].set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Macro-F1 change (percentage points)",
        title="VCOCO: the relevant comparisons",
    )
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="grey", linewidth=1)
    table = pd.DataFrame(slices).query("dataset == 'cptr'").set_index("slice")
    for i, name in enumerate(("clear", "occluded")):
        row = table.loc[name]
        axes[1].bar(
            i - 0.18,
            1000 * row.rescued / row.rows,
            width=0.35,
            color="#2166ac",
            label="Rescued" if i == 0 else None,
        )
        axes[1].bar(
            i + 0.18,
            1000 * row.harmed / row.rows,
            width=0.35,
            color="#b35806",
            label="Harmed" if i == 0 else None,
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Clear", "Occluded"],
        ylabel="Windows per 1,000 in subgroup",
        title="CPTR: changes concentrate under occlusion",
    )
    axes[1].legend(frameon=False)
    for i, name in enumerate(("vcoco", "cptr")):
        overlap = summary[name]["error_overlap"]
        total = overlap["rescued"] + overlap["both_wrong"]
        correction = 100 * overlap["rescued"] / total
        axes[2].barh(i, correction, color="#2166ac", label="Candidate corrects" if i == 0 else None)
        axes[2].barh(
            i,
            100 - correction,
            left=correction,
            color="#ddd",
            label="Both still wrong" if i == 0 else None,
        )
        axes[2].text(
            correction + (100 - correction) / 2,
            i,
            f"{100 - correction:.1f}% shared",
            ha="center",
            va="center",
        )
    axes[2].set(
        yticks=[0, 1],
        yticklabels=["VCOCO", "CPTR"],
        xlim=(0, 100),
        xlabel="Percentage of baseline errors",
        title="How much new information is present?",
    )
    axes[2].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    fig.suptitle(
        "HAC development review · exploratory, conditional on retained models", fontsize=14
    )
    fig.savefig(output / "component_review.png", dpi=180)
    fig.savefig(output / "component_review.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def run(repository: Path, output: Path, resamples=10_000, seed=20260906):
    repository, output = repository.resolve(), output.resolve()
    if resamples < 1:
        raise ValueError("Positive resample count required")
    # A fresh leaf directory prevents source clobbering and accidental reuse of old output.
    if output.exists():
        raise ValueError("Output must be a new directory")
    for relative, _ in SOURCES.values():
        source = (repository / relative).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output overlaps a locked input")
    captured = {
        name: locked_bytes(repository / path, digest) for name, (path, digest) in SOURCES.items()
    }
    vcoco = pd.read_csv(BytesIO(captured["vcoco_rows"]), dtype={"image_id": str})
    cptr = pd.read_csv(BytesIO(captured["cptr_rows"]), dtype={"recording_id": str})
    vs = json.loads(captured["vcoco_summary"])
    cs = json.loads(captured["cptr_summary"])
    if len(vcoco) != 6640 or vs["people"] != 6640 or cs["model_fits"] != 0:
        raise ValueError("Unexpected locked scope")
    cptr = cptr.loc[cptr.scope.eq("grouped_crossfit_oof")].reset_index(drop=True)
    if len(cptr) != 4977 or set(cptr.development_role) != {"train"}:
        raise ValueError("Unexpected CPTR scope")
    output.mkdir(parents=True)
    summary = {
        "status": "HAC_EXPLORATORY_COMPONENT_RELATIONSHIPS_COMPLETE",
        "interpretation": "Post hoc conditional development analysis; no selection-adjusted intervals, no confirmatory tests, no causal attribution from correlations.",
        "inputs": {
            key: {"path": path, "sha256": digest} for key, (path, digest) in SOURCES.items()
        },
        "model_fits": 0,
        "checkpoint_feature_image_reads": 0,
        "protected_data_reads": 0,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resampling": {
            "draws": resamples,
            "seed": seed,
            "numpy": np.__version__,
            "bit_generator": "PCG64",
            "method": "paired cluster percentile bootstrap",
            "row_weighting": "sample groups uniformly, retain all constituent rows",
            "macro_f1_labels": list(CLASSES),
            "zero_division": 0,
            "streams": "Separate PCG64 instances initialized with the same seed; shared draws within each dataset; no cross-dataset joint inference.",
        },
    }
    all_slices, all_groups = [], []
    for dataset, frame, group_column in (
        ("vcoco", vcoco, "image_id"),
        ("cptr", cptr, "recording_id"),
    ):
        labels = frame.label_index.to_numpy(int)
        if dataset == "vcoco":
            models = {key: probabilities(frame, prefix, "__") for key, prefix in METHODS.items()}
            baseline, candidate = models["flat"], models["full_mixed"]
        else:
            models = {key: probabilities(frame, key, "_") for key in ("baseline", "candidate")}
            baseline, candidate = models.values()
        order, group_index = np.unique(frame[group_column].astype(str), return_inverse=True)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(order), size=(resamples, len(order)), dtype=np.int32)
        np.save(output / f"{dataset}_resample_indices.npy", indices, allow_pickle=False)
        write_json(output / f"{dataset}_group_order.json", order.tolist())
        draws = clustered_model_draws(labels, models, group_index, indices)
        point = {key: metrics(labels, values) for key, values in models.items()}
        contrasts = CONTRASTS if dataset == "vcoco" else (("candidate", "baseline"),)
        effects = {}
        names = list(models)
        for cand, base in contrasts:
            effects[f"{cand}_minus_{base}"] = {
                metric: interval(
                    point[cand][metric] - point[base][metric],
                    draws[:, names.index(cand), j] - draws[:, names.index(base), j],
                )
                for j, metric in enumerate(("macro_f1", "nll"))
            }
        summary[dataset] = {
            "rows": len(labels),
            "groups": len(order),
            "metrics": point,
            "contrasts": effects,
            "error_overlap": error_overlap(labels, baseline, candidate),
        }
        masks = {"all": np.ones(len(frame), dtype=bool), **confidence_masks(baseline)}
        if dataset == "cptr":
            occluded = frame.window_any_occluded.to_numpy(bool)
            transition = frame.transition_window.to_numpy(bool)
            masks.update(
                {
                    "occluded": occluded,
                    "clear": ~occluded,
                    "transition": transition,
                    "stable": ~transition,
                    "occluded_transition": occluded & transition,
                    "occluded_stable": occluded & ~transition,
                }
            )
            summary[dataset]["subgroup_effect_contrasts"] = {
                key: subgroup_contrast(labels, baseline, candidate, group_index, indices, mask)
                for key, mask in (
                    ("occluded_minus_clear", occluded),
                    ("transition_minus_stable", transition),
                )
            }
            summary[dataset]["class_standardized_occluded_minus_clear"] = (
                class_standardized_contrast(
                    labels, baseline, candidate, group_index, indices, occluded
                )
            )
            summary[dataset]["inference_limit"] = (
                "11 scenario IDs stored as recording_id according to hash-bound generation-code lineage; original video mapping not re-enumerated. Intervals assume scenario independence and condition on retained fitted folds. See docs/HAC_SCENARIO_LINEAGE_REVIEW_20260906.md."
            )
            for label, name in enumerate(CLASSES):
                masks[f"occluded_{name}"] = occluded & (labels == label)
                masks[f"clear_{name}"] = ~occluded & (labels == label)
        else:
            masks.update(
                {
                    f"source_{value}": frame.v2_split.eq(value).to_numpy()
                    for value in ("train", "val")
                }
            )
            summary[dataset]["inference_limit"] = (
                "Image grouping verified in this export; historical outer training/evaluation fold assignments only attested, not reconstructed."
            )
        all_slices.extend(
            descriptive_slices(dataset, frame, labels, baseline, candidate, group_column, masks)
        )
        # Group-level rank associations avoid pretending individual video frames are independent.
        delta_nll = -np.log(np.clip(candidate[np.arange(len(labels)), labels], 1e-12, 1))
        delta_nll += np.log(np.clip(baseline[np.arange(len(labels)), labels], 1e-12, 1))
        group_values = pd.DataFrame(
            {
                "group": frame[group_column],
                "delta_nll": delta_nll,
                "baseline_confidence": baseline.max(1),
                "baseline_error": baseline.argmax(1) != labels,
            }
        )
        if dataset == "cptr":
            group_values["occlusion_fraction"] = occluded
            group_values["transition_fraction"] = transition
        else:
            group_values["mean_log_area_fraction"] = np.log(
                frame.bbox_area_fraction.to_numpy(float)
            )
        aggregate = group_values.groupby("group", sort=True).mean()
        aggregate["rows"] = group_values.groupby("group", sort=True).size()
        correlations = {}
        for column in aggregate.columns:
            if column in {"delta_nll", "rows"}:
                continue
            value = aggregate[column].corr(aggregate.delta_nll, method="spearman")
            correlations[column] = float(value) if np.isfinite(value) else None
        summary[dataset]["group_level_spearman_with_delta_nll"] = correlations
        summary[dataset]["correlation_limit"] = (
            "Equal-weight group means; descriptive ecological rank correlations, no p-values, no causal or routing claim. Baseline difficulty is mathematically coupled to a baseline-referenced outcome."
        )
        aggregate.insert(0, "dataset", dataset)
        all_groups.extend(aggregate.reset_index().to_dict(orient="records"))
    pd.DataFrame(all_slices).to_csv(
        output / "descriptive_slices.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(all_groups).to_csv(
        output / "group_relationships.csv", index=False, lineterminator="\n"
    )
    plot_results(summary, all_slices, output)
    summary["artifacts"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    result = run(args.repository, args.output_dir, args.resamples)
    print(json.dumps({key: result[key]["contrasts"] for key in ("vcoco", "cptr")}, indent=2))


if __name__ == "__main__":
    main()
