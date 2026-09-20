# ARFTR development evidence

This aggregate-only package documents ARFTR's original component study and its
subsequent development history. Retained ARFTR: **85.383648% macro-F1**,
**702 / 4,977 errors**. These are
adaptively reused development results, not an independent confirmation set.

## Original ARFTR architecture

Start with the [architecture guide](../../docs/ARCHITECTURE.md), which explains the
factor residual, exact anchor control and same-track temporal pass.

| File | Contents |
| --- | --- |
| [architecture_study.json](architecture_study.json) | All seven original arms, confusion matrices, matched M4 comparison, intervals and significance screen |
| [architecture_manifest.json](architecture_manifest.json) | Supplement checksum, original local summary hash and public protocol binding |

This supplement exports already completed results; it is not a new experiment and
does not alter the earlier evidence package. Its full ARFTR outputs agree exactly
with the retained score and confusion matrix in `metrics.json`.

## Retained scores and follow-up history

| File | Contents |
| --- | --- |
| [metrics.json](metrics.json) | Confusion matrices, proper scores, fold results, paired intervals, gates and numerical exceptions |
| [experiment_ledger.csv](experiment_ledger.csv) | Sixteen key saved-prediction results; not an exhaustive fit inventory |
| [knowledge_graph.json](knowledge_graph.json) | Curated public result relationships and the digest of the full local graph |
| [evidence_manifest.json](evidence_manifest.json) | Original package hashes and lineage; the architecture supplement has its own manifest |

Check with `python tools/check_project.py`. Macro-F1, accuracy and error counts can
be recomputed from aggregate confusion matrices. NLL/Brier and scenario bootstrap
intervals are reported from original saved predictions; this export does not include
enough row-level data to independently recompute them.

The T2 comparison represents a historical bundle of changes, not one isolated
architectural effect. Later raw scores and diagnostic post-hoc scores are not
promoted results. The final matched paired-null model reached 85.311396% macro-F1,
rescued 40 errors, harmed 45 correct predictions and failed continuation.

Source references beginning `.runs/` are hash-bound local provenance pointers, not
downloadable public files. No media, per-example annotations, feature tensors or
model weights are included. See the [research overview](../../docs/RESEARCH_OVERVIEW.md)
and [reproduction guide](../../docs/REPRODUCIBILITY.md) for scope and limitations.
