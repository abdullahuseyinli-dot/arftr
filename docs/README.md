# Documentation guide

## Start here

| Document | What it answers |
| --- | --- |
| [Research overview](RESEARCH_OVERVIEW.md) | What was learned, which results are comparable, and why the search closed |
| [Architecture and knowledge map](ARCHITECTURE.md) | What the retained system does and where later mechanisms failed |
| [Model card](MODEL_CARD.md) | Intended use, evaluation, inputs, limits, and misuse risks |
| [Reproducibility](REPRODUCIBILITY.md) | What a public checkout can verify and what requires local research assets |
| [Repository maintenance](REPOSITORY_MAINTENANCE.md) | Public/local separation, evidence preservation, and release preparation |
| [Validation record](VALIDATION.md) | Exact checks performed, skip reasons, environment and remaining limits |
| [Portable development evidence](../results/arftr_development/README.md) | Numbers, hashes, and a machine-readable result graph |

## Historical studies

These documents are retained as dated research records, not current instructions.
In particular, an old proposed experiment or “next phase” is not an active task.

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
