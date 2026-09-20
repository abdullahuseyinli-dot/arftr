# Human Activity Classification Under Domain and Temporal Shift

[![Quality gates](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml?query=branch%3Amain)
[![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.12-3776AB.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0F766E.svg)](LICENSE)

A research and engineering portfolio investigating what improves human activity
classification—and when extra visual or temporal evidence makes it worse. The work
spans source-overlap auditing, person-centric representations, factorized classifiers,
temporal inference, actor memory and bounded residual correction across POLAR,
V-COCO and Okutama-Action.

**Latest development result:** Anchor-Restored Factorized Temporal Residual (ARFTR)
retains
**85.383648% macro-F1, 85.895118% accuracy and 702 errors on 4,977 examples**.
This is adaptively reused internal development, not untouched confirmation. The
search cycle is closed; the final optional motion-null experiment did not improve it.

## Explore the project

- [Research overview](docs/RESEARCH_OVERVIEW.md): findings, negative results and claim boundaries.
- [Architecture and knowledge map](docs/ARCHITECTURE.md): components and experimental relationships.
- [Model card](docs/MODEL_CARD.md): intended use, evaluation and limitations.
- [Reproduce and verify](docs/REPRODUCIBILITY.md): public checks versus full local replay.
- [Documentation index](docs/README.md): current guides and historical reports.

## Results and their evaluation boundaries

| Study | Macro-F1 | Population and interpretation |
| --- | ---: | --- |
| [POLAR v1](docs/POLAR_PUBLIC_REPORT.md) | 94.0% | Four-class held-out source test; 3,329 images |
| [V-COCO v2](docs/VCOCO_V2_EXTERNAL_TRANSFER.md) | 86.63% | Person-level official test; 6,077 people |
| [Temporal v3](docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md) | 78.54% | Separate Okutama confirmation; 1,771 examples |
| [ARFTR development](results/arftr_development/README.md) | **85.383648%** | Five-fold adaptive development; 4,977 centers / 11 scenarios |

These are separate studies, not a single leaderboard. Different populations, label
spaces, modalities and selection histories make direct score comparisons invalid.

![Earlier sealed Okutama confirmation: target static 74.58%, temporal 78.54%, and 50% clip routing 78.17% macro-F1.](assets/vcoco_v3_confirmation_comparison.png)

In the earlier locked temporal study, clips improved over the matched static model
by **3.96 percentage points** (95% scenario-cluster interval: +2.02 to +5.68 pp).
A fixed 50% clip policy retained 90.7% of that gain. This is evidence for temporal
inference in that study, **not external validation of ARFTR**.
[Report and uncertainty](docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md).

Within the ARFTR development line, saved predictions show a historical improvement
from **71.923768% to 85.383648%** (+13.459880 percentage points). This combines
several changes; it is not the causal gain of one isolated component.

## The final experiment—and why it was rejected

A matched ten-fit study tested a shared motion-null residual trained from
initialization, holding data, priors, folds, seed, minibatches and optimization fixed.

| System | Macro-F1 | Errors | Rescues / harms versus ARFTR |
| --- | ---: | ---: | ---: |
| Retained ARFTR | 85.383648% | 702 | — |
| Matched plain control | 85.467181% | 698 | 40 / 36 |
| Shared motion-null residual | 85.311396% | 707 | 40 / 45 |

The new mechanism passed exact checkpoint replay and its architectural null
invariant, but failed the fixed performance gates. Neither correction justified
replacing ARFTR. The [portable evidence](results/arftr_development/README.md) retains
the gate decision, fold results, uncertainty and approved numerical exceptions.

![ARFTR's historical gain from 71.92% to 85.38% macro-F1, followed by inconsistent fold corrections: plain control +7, +2, -2, +4, -7; paired-null +1, +1, -1, -2, -4.](assets/arftr_development_summary.png)

Both final corrections harm more predictions than they rescue in at least two
outer folds. A higher aggregate point estimate alone is not enough to replace the
retained model. [Figure sources and regeneration](assets/README.md).

## Engineering work demonstrated

- **Data integrity:** source-overlap audits, grouped splits, label-access controls,
  ancestry-safe cross-fitting and hash-bound execution records.
- **Modeling:** frozen visual encoders, person-centric multiview features,
  factorized posture/locomotion targets, temporal readers and actor memory.
- **Evaluation:** paired grouped uncertainty, calibration, rescue-versus-harm analysis,
  exact-retain controls and explicit rejection of fragile improvements.
- **Reproducibility:** synthetic tests, aggregate evidence, checkpoint replay,
  original experiment recipes and a machine-readable result map.

Oracle bounds and annotation-derived diagnostics are not deployable results.
No state-of-the-art claim or production-readiness claim is made.

## Reports and walkthrough

| Read or run | Scope |
| --- | --- |
| [Executed notebook](human_activity_classification.ipynb) | Historical POLAR, V-COCO and temporal studies; reads tracked evidence, no training |
| [Temporal study PDF](output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf) | Locked confirmation, distillation and budgeted inference |
| [V-COCO study PDF](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf) | Person-centric transfer and controlled representation comparisons |
| [POLAR study PDF](output/pdf/polar_public_report_v1.0.0.pdf) | Source-audited benchmark, calibration and attribution |
| [ARFTR overview and map](docs/RESEARCH_OVERVIEW.md) | Latest development outcome, failed corrections and links to code/evidence |

The notebook and versioned PDFs predate the September ARFTR continuation; they
remain unchanged historical artifacts. [All reports, including CPTR](output/pdf/README.md).

## Quick verification

No data download, GPU, model weights or third-party Python packages are needed:

```bash
python tools/check_project.py
```

For code-level tests, use Python 3.11 or 3.12 in a virtual environment:

```bash
python -m pip install -e ".[dev,notebook,report,research]"
python -m pytest
python tools/check_style.py
```

The [reproduction guide](docs/REPRODUCIBILITY.md) explains installation, legacy
style diagnostics and assets needed for historical replay. Aggregate metric checks
are not a substitute for replaying checkpoints or retraining models.

## Repository structure

```text
src/hac/       Reusable components and research implementations
experiments/   Versioned protocols, historical runners and audits
tests/         Unit, synthetic-data and evidence-contract tests
tools/         Data utilities, exporters and current checkout checks
results/       Compact public evidence; no model weights or dataset media
docs/          Current guides, dated research records and release history
assets/        Research figures
.runs/         Local-only evidence and caches (not distributed)
```

Historical reports and locked research source remain in place to preserve their
references. Dated plans are archival, not instructions to launch more training.

## Citation, license and maintenance

Author: **Abdulla Huseyinli**. See [CITATION.cff](CITATION.cff).
Working version **3.1.0.dev0** is unreleased; no new DOI or publication is claimed.
Earlier reports remain available through the documentation index.

Original code and documentation use the [MIT License](LICENSE). Dataset media and
pretrained checkpoints are not redistributed and retain their upstream terms;
see [third-party notices](THIRD_PARTY_NOTICES.md).

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) ·
[Repository and release policy](docs/REPOSITORY_MAINTENANCE.md)
