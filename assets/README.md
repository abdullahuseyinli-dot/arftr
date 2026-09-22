# Figures

## Current ARFTR development figures

| Figure | What it shows | Public input |
| --- | --- | --- |
| [ARFTR component study](arftr_architecture_results.png) ([SVG](arftr_architecture_results.svg)) | The retained architecture, its original component ablations and shuffled-neighbor control | [Original component export](../results/arftr_development/architecture_study.json) |
| [Historical gain and fold stability](arftr_development_summary.png) ([SVG](arftr_development_summary.svg)) | T2 to ARFTR, then the inconsistent final matched corrections | [Ledger](../results/arftr_development/experiment_ledger.csv) and [fold results](../results/arftr_development/metrics.json) |
| [Retained confusion matrix](arftr_confusion_matrix.png) ([SVG](arftr_confusion_matrix.svg)) | Counts and true-class-normalized error concentration | [Aggregate confusion matrix](../results/arftr_development/metrics.json) |
| [Report component study](arftr_report_components.png) ([SVG](arftr_report_components.svg)) | The same seven-arm evidence in a print-readable layout | [Original component export](../results/arftr_development/architecture_study.json) |
| [System overview](arftr_system_overview.png) ([SVG](arftr_system_overview.svg)) | Anchor, factor residuals, temporal boundaries and output decoding | [Architecture guide](../docs/ARCHITECTURE.md) and [locked protocol](../experiments/okutama_arftr_protocol.json) |

Regenerate only these new figures with:

```bash
python tools/render_arftr_figures.py
python tools/check_project.py
```

The renderer needs Matplotlib and NumPy, but no GPU, dataset, checkpoint or local run
directory. It reads hash-verified public evidence, does not run a model, and does not
change historical figures. The [figure manifest](arftr_figure_manifest.json) binds
inputs, renderer, output hashes and Matplotlib version. Rendering bytes can differ
across library versions; the statistical content is taken directly from the inputs.

These are adaptively reused **development** results, not confirmation. The historical
gain combines multiple changes. Fold nets are counts, not macro-F1 differences or
confidence intervals. The confusion heatmap normalizes by true class (rows).

## Historical study figures

Existing v1/v2/v3 figures remain unchanged and are bound by historical manifests.
The commands below document their original builders; render to a separate directory
when exploring a rebuild rather than replacing the archived assets.

POLAR charts are generated from tracked, locked evidence with:

```bash
python tools/render_polar_final_figures.py --results-dir results --output-dir assets
```

Each chart is stored as PNG for GitHub/notebook rendering and SVG for vector-quality
reuse. The final set covers held-out model comparison, confusion, data scale, external
transfer, bbox-aware faithfulness, attribution sanity, and bit-flip robustness.

The motion-identifiability confirmation comparison and fixed-budget routing curve are
rendered from the portable Okutama tables with:

```bash
python tools/render_vcoco_v3_figures.py --results results/vcoco_v3 --output-dir assets
```

The current distributable assets contain aggregate charts and architecture diagrams;
they do not contain dataset photographs or video frames. The v1 and v2 tags included
four qualitative COCO composites. Those files remain identifiable in the historical
record and in retained local run evidence, but are excluded from the v3 distributable
tree because the source photographs retain their individual Flickr terms. See
`THIRD_PARTY_NOTICES.md` for the release boundary. Raw POLAR, V-COCO, and
Okutama-Action media are not copied into the repository.
