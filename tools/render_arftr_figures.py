"""Render current development figures from public aggregates; no training or private data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "results/arftr_development/metrics.json",
    "results/arftr_development/experiment_ledger.csv",
    "results/arftr_development/architecture_study.json",
)
GENERATOR = "tools/render_arftr_figures.py"
FIGURES = (
    "arftr_development_summary",
    "arftr_confusion_matrix",
    "arftr_architecture_results",
    "arftr_report_components",
    "arftr_system_overview",
)


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix != ".png":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_data(root: Path) -> dict:
    """Read hash-verified, aggregate-only evidence used by both plots."""
    folder = root / "results/arftr_development"
    manifest = json.loads((folder / "evidence_manifest.json").read_text(encoding="utf-8"))
    supplement = json.loads((folder / "architecture_manifest.json").read_text(encoding="utf-8"))
    for relative in SOURCES:
        path = root / relative
        source_manifest = supplement if path.name == "architecture_study.json" else manifest
        if digest(path) != source_manifest["artifacts"][path.name]["sha256"]:
            raise ValueError(f"Figure source differs from the evidence manifest: {relative}")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    with (folder / "experiment_ledger.csv").open(encoding="utf-8", newline="") as stream:
        ledger = {row["result"]: row for row in csv.DictReader(stream)}
    confusion = metrics["scores"]["ARFTR"]["confusion"]
    return {
        "rows": metrics["rows"],
        "scenarios": metrics["scenarios"],
        "history": [
            {
                "name": name,
                "macro_f1_percent": float(ledger[name]["macro_f1_percent"]),
                "errors": int(ledger[name]["errors"]),
            }
            for name in ("T2", "ARFTR")
        ],
        "fold_nets": {
            arm: metrics["transitions"][arm]["per_fold_net"] for arm in ("plain", "paired_null")
        },
        "confusion": confusion,
        "errors": metrics["scores"]["ARFTR"]["errors"],
        "upright_confusions": confusion[1][2] + confusion[2][1],
        "architecture": json.loads(
            (folder / "architecture_study.json").read_text(encoding="utf-8")
        ),
    }


def development_figure(data: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (history, folds) = plt.subplots(1, 2, figsize=(11.2, 4.9), width_ratios=(1, 1.25))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.73, wspace=0.38)
    fig.suptitle("ARFTR development: gains and the final correction test", y=0.97, weight="bold")
    fig.text(
        0.5,
        0.88,
        f"{data['rows']:,} centers / {data['scenarios']} scenarios / adaptive internal development",
        ha="center",
        color="#475569",
    )
    bars = history.barh(
        [1, 0],
        [r["macro_f1_percent"] for r in data["history"]],
        height=0.48,
        color=["#94a3b8", "#00796b"],
        zorder=3,
    )
    for bar, row in zip(bars, data["history"], strict=True):
        history.text(
            bar.get_width() - 2,
            bar.get_y() + bar.get_height() / 2,
            f"{row['macro_f1_percent']:.2f}%",
            ha="right",
            va="center",
            color="white",
            weight="bold",
        )
    history.set_yticks([1, 0], [f"{r['name']}\n{r['errors']:,} errors" for r in data["history"]])
    history.set_xlim(0, 100)
    history.set_ylim(-0.65, 1.8)
    history.set_xticks([0, 25, 50, 75, 100])
    history.set_xlabel("Macro-F1 (%)")
    gain = data["history"][1]["macro_f1_percent"] - data["history"][0]["macro_f1_percent"]
    history.set_title(f"Historical gain: +{gain:.2f} pp", loc="left", pad=14)
    history.grid(axis="x", alpha=0.2, zorder=0)

    indices = np.arange(len(data["fold_nets"]["plain"]))
    for offset, arm, label, color, hatch in (
        (-0.19, "plain", "Plain control", "#0072b2", None),
        (0.19, "paired_null", "Paired-null", "#c66b00", "//"),
    ):
        values = data["fold_nets"][arm]
        bars = folds.bar(
            indices + offset,
            values,
            width=0.35,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.4,
            label=f"{label} (net {sum(values):+d})",
            zorder=3,
        )
        folds.bar_label(bars, labels=[f"{value:+d}" for value in values], padding=3, fontsize=10)
    folds.axhline(0, color="#334155", linewidth=1)
    folds.set_xticks(indices, [f"Fold {i}" for i in indices])
    folds.set_ylim(-10, 11)
    folds.set_yticks([-8, -4, 0, 4, 8])
    folds.set_ylabel("Net corrections versus ARFTR")
    folds.set_title("Neither correction wins every fold", loc="left", pad=14)
    folds.grid(axis="y", alpha=0.2, zorder=0)
    folds.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=10
    )
    for axis in (history, folds):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=0, pad=7)
    fig.text(
        0.105, 0.045, "Multiple changes; not one isolated effect.", fontsize=10, color="#475569"
    )
    fig.text(
        0.56,
        0.015,
        "Net = rescues minus harms; counts, not F1 points.",
        fontsize=10,
        color="#475569",
    )
    return fig


def confusion_figure(data: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    counts = np.asarray(data["confusion"])
    fractions = counts / counts.sum(axis=1, keepdims=True)
    labels = ["Sitting", "Standing", "Walking /\nrunning"]
    fig, axis = plt.subplots(figsize=(7.2, 6.1))
    fig.subplots_adjust(left=0.20, bottom=0.26, right=0.90, top=0.80)
    fig.suptitle(f"Retained ARFTR: {data['errors']:,} remaining errors", y=0.97, weight="bold")
    fig.text(
        0.5,
        0.90,
        "Cells show counts and percentages within each true class",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    heatmap = axis.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(
                column,
                row,
                f"{counts[row, column]:,}\n{100 * fractions[row, column]:.1f}%",
                ha="center",
                va="center",
                color="white" if fractions[row, column] > 0.5 else "#172554",
                fontsize=12,
            )
    axis.set_xticks(range(3), labels)
    axis.set_yticks(range(3), labels)
    axis.set_xlabel("Predicted class", labelpad=10)
    axis.set_ylabel("True class", labelpad=10)
    axis.tick_params(length=0, pad=8)
    colorbar = fig.colorbar(heatmap, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    share = 100 * data["upright_confusions"] / data["errors"]
    fig.text(
        0.5,
        0.06,
        f"Standing vs walking/running: {data['upright_confusions']} of {data['errors']} errors ({share:.1f}%).",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.015,
        "Adaptive development; error concentration is not evidence of a cause.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    return fig


def architecture_figure(data: dict):
    import matplotlib.pyplot as plt

    study = data["architecture"]
    arms = (
        ("r0_exact_m4", "M4 anchor"),
        ("r1_p6_restoration_only", "+ P6 restoration only"),
        ("r2_a3_motion_only", "+ A3 template motion only"),
        ("r3_temporal_only", "+ Temporal update only"),
        ("r4_residual_no_temporal", "+ Factor residuals, no temporal update"),
        ("r5_arftr_full", "ARFTR: full architecture"),
        ("r6_shuffled_neighbor_control", "Shuffled-neighbor control"),
    )
    fig, axis = plt.subplots(figsize=(11.2, 5.5))
    fig.subplots_adjust(left=0.34, right=0.96, top=0.78, bottom=0.21)
    fig.suptitle("ARFTR: original component study", y=0.97, weight="bold", fontsize=17)
    fig.text(
        0.5,
        0.88,
        f"{study['rows']:,} centers / {study['outer_folds']} scenario-grouped folds / "
        f"{len(study['prediction_seeds'])}-seed predictions",
        ha="center",
        color="#475569",
    )
    anchor = 100 * study["scores"]["r0_exact_m4"]["macro_f1"]
    axis.axvline(anchor, linestyle="--", linewidth=1, color="#94a3b8", zorder=1)
    axis.axhspan(4.55, 5.45, color="#e4f3ef", zorder=0)
    for row, (arm, _) in enumerate(arms):
        score = 100 * study["scores"][arm]["macro_f1"]
        full = arm == "r5_arftr_full"
        control = arm == "r6_shuffled_neighbor_control"
        color = "#00796b" if full else "#9b5900" if control else "#0072b2"
        axis.scatter(
            score, row, color=color, s=95 if full else 60, marker="D" if full else "o", zorder=3
        )
        axis.annotate(
            f"{score:.2f}%",
            (score, row),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            color=color,
            weight="bold" if full else "normal",
        )
    axis.set_yticks(range(len(arms)), [label for _, label in arms])
    axis.invert_yaxis()
    axis.set_xlim(80, 86.35)
    axis.set_xticks(range(80, 87))
    axis.set_xlabel("Macro-F1 (%) — point estimates; zoomed scale", labelpad=10)
    axis.grid(axis="x", alpha=0.18)
    axis.tick_params(length=0, pad=10)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.5,
        0.07,
        "Dashed line: M4 anchor. Component ablations reuse the full model's selected coefficients.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    quantiles = study["paired_scenario_bootstrap"]["delta_quantiles_2_5_50_97_5"]
    gain = study["effects"]["macro_f1_gain_points"]
    fig.text(
        0.5,
        0.025,
        f"Adaptive development; full vs anchor: {gain:+.2f} pp "
        f"(95% interval [{100 * quantiles[0]:+.2f}, {100 * quantiles[2]:+.2f}] pp).",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    return fig


def report_components_figure(data: dict):
    """Print-sized companion to the original, unchanged web component chart."""
    import matplotlib.pyplot as plt

    study = data["architecture"]
    arms = (
        ("r0_exact_m4", "M4 anchor"),
        ("r1_p6_restoration_only", "+ P6 restoration only"),
        ("r2_a3_motion_only", "+ A3 motion only"),
        ("r3_temporal_only", "+ Temporal update only"),
        ("r4_residual_no_temporal", "+ Factor residuals\n(no temporal update)"),
        ("r5_arftr_full", "Full ARFTR"),
        ("r6_shuffled_neighbor_control", "Shuffled-neighbor control"),
    )
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    fig.subplots_adjust(left=0.38, right=0.96, top=0.81, bottom=0.28)
    fig.suptitle("ARFTR: original component evidence", y=0.965, fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.885,
        f"{study['rows']:,} centers / {study['outer_folds']} grouped folds / "
        f"{len(study['prediction_seeds'])}-seed predictions",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    anchor = 100 * study["scores"]["r0_exact_m4"]["macro_f1"]
    axis.axvline(anchor, linestyle="--", linewidth=1, color="#94a3b8", zorder=1)
    axis.axhspan(4.58, 5.42, color="#e4f3ef", zorder=0)
    for row, (arm, _) in enumerate(arms):
        score = 100 * study["scores"][arm]["macro_f1"]
        full = arm == "r5_arftr_full"
        control = arm == "r6_shuffled_neighbor_control"
        color = "#00796b" if full else "#9b5900" if control else "#0072b2"
        axis.scatter(
            score, row, color=color, s=75 if full else 40, marker="D" if full else "o", zorder=3
        )
        axis.annotate(
            f"{score:.2f}%",
            (score, row),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=10.5,
            weight="bold" if full else "normal",
        )
    axis.set_yticks(range(len(arms)), [label for _, label in arms], fontsize=10.5)
    axis.set_ylim(6.5, -0.5)
    axis.set_xlim(80, 86.9)
    axis.set_xticks([80, 82, 84, 86])
    axis.set_xlabel("Macro-F1 (%) · zoomed scale", labelpad=9, fontsize=10.5)
    axis.grid(axis="x", alpha=0.18)
    axis.tick_params(length=0, pad=8, labelsize=10.5)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.5,
        0.105,
        "Points: observed scores. Dashed line: M4 anchor.\n"
        "Removal arms reuse the full model's selected coefficients.",
        ha="center",
        fontsize=10.5,
        color="#475569",
        linespacing=1.5,
    )
    quantiles = study["paired_scenario_bootstrap"]["delta_quantiles_2_5_50_97_5"]
    fig.text(
        0.5,
        0.035,
        f"Full − anchor: {study['effects']['macro_f1_gain_points']:+.2f} pp; "
        f"95% interval [{100 * quantiles[0]:+.2f}, {100 * quantiles[2]:+.2f}] pp.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    return fig


def system_overview_figure(data: dict):
    """Show probability-level inference and its selection/context boundaries."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    fig.suptitle("ARFTR: anchored factor correction", y=0.965, fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.895,
        "Probability fusion above the upstream visual and actor-memory models",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )

    def box(x, y, width, height, title, details, *, core=False):
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.009,rounding_size=0.012",
                linewidth=1.1,
                facecolor="#e4f3ef" if core else "#f1f5f9",
                edgecolor="#00796b" if core else "#94a3b8",
            )
        )
        axis.text(
            x + width / 2,
            y + height * 0.73,
            title,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
        )
        axis.text(
            x + width / 2,
            y + height * 0.33,
            details,
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.25,
        )

    def arrow(start, end):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color="#475569",
                shrinkA=4,
                shrinkB=5,
            )
        )

    for x, title, details in (
        (0.035, "M4 anchor", "Actor-memory\nprobabilities"),
        (0.355, "P6 restoration", "Posture and motion\nprobabilities"),
        (0.675, "A3 template expert", "Upright-motion\nresidual only"),
    ):
        box(x, 0.705, 0.29, 0.125, title, details)
    box(
        0.15,
        0.49,
        0.70,
        0.145,
        "Factor correction in log-odds",
        "Posture: sitting versus upright\nMotion: standing versus walking/running",
        core=True,
    )
    for start, end in (
        ((0.18, 0.705), (0.28, 0.635)),
        ((0.50, 0.705), (0.50, 0.635)),
        ((0.82, 0.705), (0.72, 0.635)),
        ((0.50, 0.49), (0.50, 0.425)),
        ((0.50, 0.28), (0.50, 0.21)),
    ):
        arrow(start, end)
    box(
        0.15,
        0.28,
        0.70,
        0.145,
        "One same-track temporal update",
        "Exact −1 / +1 second neighbors\nSame recording, scenario, fold and track",
        core=True,
    )
    axis.text(
        0.5,
        0.18,
        "Decode: sitting · standing · walking/running",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#f1f5f9", "edgecolor": "#94a3b8"},
    )
    fig.text(
        0.5,
        0.09,
        "Four coefficients selected from inner out-of-fold predictions only.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.5,
        0.045,
        "All-zero correction retains M4 exactly. Offline / look-ahead inference.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    return fig


def render(root: Path, output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_data(root)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.labelcolor": "#1e293b",
            "text.color": "#1e293b",
            "svg.hashsalt": "hac-arftr-20260920",
            "savefig.dpi": 180,
        }
    ):
        for name, builder in zip(
            FIGURES,
            (
                development_figure,
                confusion_figure,
                architecture_figure,
                report_components_figure,
                system_overview_figure,
            ),
            strict=True,
        ):
            figure = builder(data)
            try:
                for suffix in ("png", "svg"):
                    path = output / f"{name}.{suffix}"
                    options = {"metadata": {"Date": None}} if suffix == "svg" else {}
                    figure.savefig(path, **options)
                    if suffix == "svg":
                        # Matplotlib path output contains trailing spaces; keep generated diffs clean.
                        svg = path.read_text(encoding="utf-8")
                        path.write_text(
                            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                            encoding="utf-8",
                            newline="\n",
                        )
                    artifacts[path.name] = {"sha256": digest(path)}
            finally:
                plt.close(figure)
    manifest = {
        "schema_version": 1,
        "scope": "Figures of existing aggregate development evidence; no new model evaluation",
        "hash_convention": "SHA-256; CRLF normalized to LF except for PNG bytes",
        "sources": {name: {"sha256": digest(root / name)} for name in (*SOURCES, GENERATOR)},
        "matplotlib_version": matplotlib.__version__,
        "artifacts": artifacts,
    }
    (output / "arftr_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {"figures": len(FIGURES), "artifacts": len(artifacts), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    print(json.dumps(render(ROOT, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
