# ARFTR documentation and research archive

## ARFTR: start here

The current project centerpiece is **Anchor-Restored Factorized Temporal Residual**.
Begin with its design and original component study; earlier studies and later
follow-up experiments are supporting research records.

| Document | What it answers |
| --- | --- |
| [ARFTR architecture and component evidence](ARCHITECTURE.md) | The mechanism, equations, original controls and result interpretation |
| [Portable ARFTR evidence](../results/arftr_development/README.md) | Original component study, retained metrics and machine-readable evidence |
| [Model card](MODEL_CARD.md) | Intended use, evaluation, inputs, limits, and misuse risks |
| [Reproducibility](REPRODUCIBILITY.md) | What a public checkout can verify and what requires local research assets |
| [Repository maintenance](REPOSITORY_MAINTENANCE.md) | Public/local separation, evidence preservation, and release preparation |
| [Validation record](VALIDATION.md) | Exact checks performed, skip reasons, environment and remaining limits |
| [Research history and follow-up experiments](RESEARCH_OVERVIEW.md) | Broader lineage, subsequent corrections and their outcomes |

## Historical studies

These documents are retained as dated research records, not current instructions.
In particular, an old proposed experiment or “next phase” is not an active task.

| Study | Read online | Versioned PDF | Portable evidence |
| --- | --- | --- | --- |
| POLAR v1 | [Report](POLAR_PUBLIC_REPORT.md) | [PDF](../output/pdf/polar_public_report_v1.0.0.pdf) | [Evidence guide](../results/README.md) |
| V-COCO v2 | [Report](VCOCO_V2_EXTERNAL_TRANSFER.md) | [PDF](../output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf) | [Tables and locks](../results/vcoco_v2/README.md) |
| Temporal v3 | [Report](VCOCO_V3_MOTION_IDENTIFIABILITY.md) | [PDF](../output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf) | [Confirmation evidence](../results/vcoco_v3/README.md) |
| CPTR development | [Report](OKUTAMA_CPTR_DEVELOPMENT.md) | [PDF](../output/pdf/okutama_cptr_development_v3.0.0.pdf) | [Positive and negative results](../results/okutama_cptr/README.md) |

The [executed notebook](../human_activity_classification.ipynb) is a code-backed
walkthrough of these earlier studies, not the September ARFTR continuation.
[Figure guide](../assets/README.md) · [Dataset setup](../data/README.md) ·
[Historical experiment runners](../experiments/README.md).

- [POLAR report](POLAR_PUBLIC_REPORT.md) and [portfolio case study](PORTFOLIO_ARTICLE.md).
- [V-COCO v2](VCOCO_V2_EXTERNAL_TRANSFER.md): person-level official-test study.
- [Temporal v3](VCOCO_V3_MOTION_IDENTIFIABILITY.md): separate Okutama confirmation.
- [CPTR development](OKUTAMA_CPTR_DEVELOPMENT.md): a validation gain that did not
  survive grouped out-of-fold evaluation.
- [September architecture map](HAC_NEXT_ARCHITECTURE_MAP_20260912.md),
  [sequence results](HAC_ACTOR_RELATIVE_SEQUENCE_RESULTS_20260912.md), and
  [bounded correction results](HAC_BOUNDED_FACTOR_CORRECTION_RESULTS_20260912.md).
- [Initial September closeout](HAC_RESEARCH_CLOSEOUT_20260920.md): predates the
  subsequently authorized optional motion-null experiment, now completed.
- [Historical result lineage](RESULT_LINEAGE.md) and [release notes](releases/HUMAN_ACTIVITY_STUDY_V3.0.0.md).

Historical documents can reference local `.runs/` artifacts or the original
workstation. Those links are provenance pointers, not files promised in a public
checkout. Current documentation and portable evidence have their own checks.
