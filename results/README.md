# Tracked evidence

## Current development closeout

The [ARFTR development package](arftr_development/README.md) adds a versioned,
aggregate-only result ledger, confusion matrices, matched motion-null outcomes,
and a compact knowledge graph. These are adaptive development results, separate
from the earlier confirmation studies below. Validate the new package without
datasets or model dependencies using `python tools/check_project.py`.

This directory contains compact, path-sanitized evidence for the repository's
versioned studies. Dataset images, checkpoints, embeddings, fitted estimators, dense
probabilities, and full-resolution attribution arrays remain outside Git.

## Okutama CPTR architecture study

Begin with [`okutama_cptr/README.md`](okutama_cptr/README.md) and its
[`evidence_manifest.json`](okutama_cptr/evidence_manifest.json). The five-seed
center-conditioned short-window and body-region ensemble reaches 0.7887 macro-F1 on
the fixed validation split, compared with 0.7806 for the frozen temporal baseline.
The full 11-recording grouped OOF result is 0.7144 versus 0.7165, so the baseline
remains the default model. The package contains every component screen, all 25
fold/seed results, paired uncertainty, visibility and transition subgroups,
faithfulness interventions, CUDA latency, and the locked development decision.

## Motion-identifiability and temporal extension

Begin with [`vcoco_v3/README.md`](vcoco_v3/README.md) and its
[`evidence_manifest.json`](vcoco_v3/evidence_manifest.json). The sealed Okutama
confirmation contains 1,771 tracked people from five scenarios. The eight-frame
temporal model reaches 0.7854 macro-F1, compared with 0.7458 for the matched static
model. A fixed 50% clip-routing policy reaches 0.7817.

The package includes the human pilot aggregates, matched frozen-representation and
spatial screens, source-only and few-shot transfer, recording-grouped temporal
development, pre-confirmation calibration, paired scenario-cluster uncertainty, and
confirmation subgroup results.

## V-COCO v2 person-level study

Begin with [`vcoco_v2/README.md`](vcoco_v2/README.md) and the accompanying
[`evidence_manifest.json`](vcoco_v2/evidence_manifest.json). The official test package
records 6,077 people from 3,708 images. The selected scale-conditioned DINO stack
reached 0.8663 macro-F1 and 0.8795 accuracy; the historical source-only DINO baseline
reached 0.7071 macro-F1 and 0.7010 accuracy on the same rows.

The main files are:

- `vcoco_v2/protocol_lock.json`: endpoint, split, overlap, and test-access policy fixed
  before target-domain model fitting;
- `vcoco_v2/final_selection_lock.json`: selected stack, development comparisons,
  implementation hashes, and final-fit artifact binding;
- `vcoco_v2/test_access_gate.json`: persistent record of the single official test-label
  open;
- `vcoco_v2/official_test_summary.json`: headline metrics, paired uncertainty, and
  evidence lineage;
- `vcoco_v2/official_test_metrics.csv`, `official_test_per_class.csv`, and
  `official_test_confusions.json`: aggregate, class-level, and confusion evidence;
- `vcoco_v2/official_test_selective_metrics.json` and `official_test_strata.csv`:
  confidence-based coverage and subgroup performance;
- `vcoco_v2/development_candidates.csv`, `factorized_fusion.csv`, and
  `fewshot_curve.csv`: development-only model, target-structure, and data-scale studies;
- `vcoco_v2/mechanism_*.csv`: controlled person-scale, view, geometry, and error-shift
  analyses.

## POLAR v1 source benchmark

Files prefixed with `polar_` belong to the source-overlap-controlled four-class POLAR
study. Start with:

- `polar_final_selection_lock.json`: pre-test model, classifier, ensemble, and metric
  lock;
- `polar_final_fit_manifest.json`: checkpoint and probe hashes, configurations,
  runtimes, and artifact sizes;
- `polar_test_summary.json` and `polar_test_uncertainty.json`: locked primary metrics
  and 10,000-resample uncertainty;
- `polar_external_summary.json`: the original source-only V-COCO transfer audit;
- `polar_faithfulness_summary.json`: attribution localization, perturbation, and
  randomization evidence;
- `polar_fault_summary.json`: separately reported input and classifier-weight bit-flip
  experiments;
- `polar_final_evidence_manifest.json`: hash inventory for the primary and auxiliary
  evidence;
- `polar_study_v1.0.0_manifest.json`: frozen release inventory for the v1 report and
  analysis supplement.

Files prefixed with `polar_exploratory_` analyze the completed v1 prediction arrays.
They cover paired comparisons, blend sensitivity, error topology, person scale,
dataset composition, label semantics, mixed-person scenes, selective prediction,
faithfulness strata, and class-conditioned fault response. Their configurations and
source hashes are recorded in `polar_exploratory_summary.json`.

## Historical COCO benchmark

Files without the `polar_` or `vcoco_v2/` prefixes preserve the earlier 285-image COCO
benchmark. Its design and limitations are documented in
[`docs/LEGACY_COCO_STUDY.md`](../docs/LEGACY_COCO_STUDY.md).

## Local artifact boundary

Long-running stages write to ignored `.runs/` directories. Those files are retained
locally for replay and audit but are unsuitable for a portable repository because they
contain large binaries, rights-sensitive images, dense arrays, or machine-local paths.
Tracked summaries keep the metrics, exclusions, locks, hashes, and configurations
needed to audit the published results.
