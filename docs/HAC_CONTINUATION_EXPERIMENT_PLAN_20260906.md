# HAC continuation experiment plan - 2026-09-06

The later [team research review](HAC_ENGINEERING_RESEARCH_REVIEW_20260906.md) and
[next-run plan](HAC_NEXT_RUN_PLAN_20260906.md) refine this draft. They add the
zero-epoch attenuation finding, actual distinct-frame audit, and stronger matched
controls, and correct the scenario-group interpretation through a
[source-lineage receipt](HAC_SCENARIO_LINEAGE_REVIEW_20260906.md).

## Status

**DRAFT DESIGN ONLY.** This document is not a protocol, protocol amendment, lock,
selection record, or authorization to fit, calibrate, or evaluate a model. No
experiment proposed below has been run. Before any fitting, an approved continuation
protocol must be placed at a new path, committed from a clean tree, and covered by a
fresh lock as specified in Section 8.

This draft is subordinate to the repository's
[`SCIENTIFIC_VALIDATION_PLAN.md`](SCIENTIFIC_VALIDATION_PLAN.md). Execution requires
separate V-COCO and Okutama continuation protocols and locks, with dataset-specific
candidate families and multiplicity scopes. The F1/F2 coefficient diagnostics below
would deliberately precede the older fallback/reliability/occlusion repair sequence;
that ordering is a proposed amendment and must be approved explicitly rather than
treated as inherited authorization. F3 is an integrity control, while only a later
fitted repair can enter the general promotion gate.

The historical locks and consumed evaluation splits remain unchanged. In particular,
this draft does not repair or supersede the portability gap recorded in
[`HAC_CONTINUATION_PROVENANCE_20260906.md`](HAC_CONTINUATION_PROVENANCE_20260906.md).

## 1. Evidence that motivates, but does not authorize, the program

The following results are retained development evidence. They are associations from
grouped out-of-fold or fixed-development predictions, not causal component effects or
independent confirmation.

The intervals below are conditional on the retained fitted folds and on candidates
already selected in the historical development runs. The receiving analyses do not
repeat fitting or selection inside each resample, so the intervals are not
selection-adjusted.

| Evidence | Retained result | Bounded interpretation |
| --- | --- | --- |
| V-COCO joined development evidence | The factorized DINO+SigLIP reliability ensemble improves macro-F1 by `+0.015341` over the DINO flat probability stack; source-image cluster 95% CI `[0.008106, 0.022634]`. Mean NLL changes by `-0.027947`; CI `[-0.035230, -0.020680]`. Standing and locomotion F1 changes are positive. | The overall development association is strong enough to motivate structured static evidence. It is not a new component ablation or test result. |
| V-COCO scale slices | The candidate's Q4-large-minus-Q1-small recall contrasts have intervals crossing zero for every class. | There is no confirmed large-scale effect. Area quartile remains a descriptive subgroup, not a candidate-selection target. |
| Okutama CPTR grouped OOF | The frozen temporal baseline has macro-F1 `0.716483`; `centre_short_parts` has `0.714445` (`-0.002038`). Candidate NLL is `0.750344` versus `0.753891`, a small favorable proper-loss change. | The candidate is not promoted. The slight NLL improvement does not override the macro-F1 and worst-recording failures. |
| Okutama occlusion and fallback diagnostics | On OOF windows containing occlusion, candidate-minus-baseline macro-F1 is `-0.030816`, while NLL is only slightly lower (`0.945000` versus `0.945366`). A label-informed oracle occlusion fallback reaches OOF macro-F1 `0.717848`, only `+0.001365` over baseline, and its paired interval crosses zero. | Legacy-path attenuation is plausible, not established. The oracle uses outcome-adjacent metadata, is not deployable, and cannot justify a router. |

The V-COCO joined table is
`.runs/research_20260906/vcoco_development_evidence/development_joined.csv`
(SHA-256 `ca32a6cd9f60524d9cbb122dce969ebede167cd6900d5772c757f1131a94ec9b`).
Its summary has SHA-256
`1a5e7c4d307e6fa872a8fc25a352502bda93b35583cfb24c10c702c062dbf7e3`.
The analysis summary at
`.runs/research_20260906/vcoco_development_analysis/summary.json` has SHA-256
`b2e10cffcad0600a6601bf75f245e8011831cd4e62441ce6d2bf7083c40036ff`;
its 10,000 bootstrap resamples share source-image indices with SHA-256
`96898d21dbee0e6331f0f8a5a09930f538841a79678c238bf669e3398410021b`.
That receiving diagnostic records the hash but does not retain the index array, group
ordering, or NumPy bit-generator metadata, so the hash is an execution attestation
rather than independently replayable evidence. The fresh continuation lock must
archive the canonical indices and complete RNG metadata as required in Section 7.

The byte-locked V-COCO run reports five outer image-grouped folds, but its retained NPZ
does not contain row-level fold assignments or per-fold train/evaluation group hashes.
The receiving analysis therefore validates the reported provenance and exact
predictions; it cannot independently reconstruct fold disjointness. New continuation
runs must bind and test those assignments before fitting.

The portable CPTR anchors are
[`headline_metrics.csv`](../results/okutama_cptr/headline_metrics.csv),
[`subgroup_metrics.csv`](../results/okutama_cptr/subgroup_metrics.csv), and
[`uncertainty.json`](../results/okutama_cptr/uncertainty.json), whose hashes in
[`evidence_manifest.json`](../results/okutama_cptr/evidence_manifest.json) are,
respectively, `3dcfbc3db74d9904d4c125725b26d187cc72acd1feb199300af60cb706efdf72`,
`9bf09dcbfeb0c5cf689c354ab7fcb0191be1f769e83cbf332ddabc3c6ed26f08`,
and `aed88ab6f17ff5532ad6efc7765f1360631c749fe1c93ac23884b04d155877fd`.
The receiving-workspace oracle diagnostic summary is
`.runs/cptr/baseline_preservation_diagnostic/summary.json`, SHA-256
`32dcf384865983651c317fe961cfe6187013b13f6910df83330f804f6ee85e73`.
Its canonical recording-resample artifact has SHA-256
`5c1d48e4dede7351d00b1f3db5304cfbd18fca8b1a89cd7eda7a64ae3b6453fb`.
This is a retrospective comparison of retained prediction arrays. It does not execute
the proposed F1-F3 frozen replays or establish the model-level zero-gate identity
invariant in Section 3.

## 2. Data and grouping boundaries

These boundaries apply to every proposed diagnostic, representation, crop, and
architecture comparison.

| Study | Permitted development data | Independent unit and split rule | Forbidden inputs |
| --- | --- | --- | --- |
| V-COCO | Locked V-COCO v2 `train` plus `val` development rows only | `image_id` (the source-image identifier); all people from one image remain in one outer fold and one bootstrap cluster | Official v2 test, V-COCO temporal calibration or confirmation, historical test predictions, and any labels derived from them |
| Okutama | Provider-train archive's declared training rows (11 recordings) and fixed validation rows (3 recordings) only | `scenario_id` is the canonical split and inference unit; bind its recording/drone mapping, keep synchronized views together, and never split a track | The 1,979-row calibration role, provider test, prior confirmation, and every calibration/confirmation/test image, feature, probability, or checkpoint array |

The V-COCO `train` and `val` source tags are provenance strata inside a combined
development grouped-OOF analysis. Neither tag is an independent holdout. For Okutama,
five folds grouped by the bound canonical scenarios and seeds 42 through 46 are the
proposed matched design. Scenario-grouped OOF is primary; the three-recording
fixed-validation result and nested recording/track sensitivity are descriptive.

The combined V-COCO role is a proposed continuation reassignment, not authorization
inherited from the v2 lock. That lock assigns `train` to adaptation/inner CV and `val`
to external selection/calibration. The later published v3 protocol describes combined
nested development, and the retained run attests to that design, but its exact
lock-time protocol bytes have the portability gap recorded in the provenance note.
The fresh continuation protocol must therefore authorize the combined role explicitly;
the retrospective evidence above does not grant that permission for a new fit.

The retained CPTR diagnostic clusters OOF rows into 11 groups by the recorded
`recording_id`; its interval is conditional on those identifiers and does not establish
that paired drone views are independent. Before new Okutama fitting, bind the exact
`scenario_id` to `recording_id` to drone-view mapping and use the original protocol's
scenario as the canonical independent group, keeping synchronized Drone1/Drone2
recordings together. Tracks are nested descriptive sensitivity units and never replace
scenario inference. Fold and leave-one-fold-out summaries are likewise descriptive
influence checks, not a second population of independent clusters. The three fixed
validation recordings may be opened once only after all OOF decisions are frozen;
they cannot fit, select, or open a later stage.

The receiving-machine provenance record also discloses two process errors. An
over-broad stale-path search printed the first five calibration-role rows of the CPTR
development manifest. An interim diagnostic then raw-read all 1,979 calibration-row
strings, without CSV-parsing or retaining their protected fields, while routing on the
final `split` value; its focused integration test repeated that scan. No calibration
image, feature, probability, or checkpoint array and no confirmation input was opened.
The row contents must not guide this program, the calibration manifest must not be
described as pristine or unseen, and
[`analyze_okutama_cptr_baseline_preservation.py`](../experiments/analyze_okutama_cptr_baseline_preservation.py)
must retain no mixed-role manifest input. The calibration role remains forbidden as a
development input.

## 3. Priority A: isolate CPTR legacy attenuation

### Hypothesis and invariant

The current CPTR construction starts at static logits and multiplies the established
temporal residual by both a learned legacy gate and the observed-valid-frame fraction
`q_legacy`. This can attenuate the established model precisely when occlusion makes
`q_legacy < 1`.

With `r_h,legacy = z_h,temporal - z_h,static`, the frozen comparisons are defined
head by head as

- `z_h,original = z_h,static + g_h,legacy * q_h,legacy * r_h,legacy + sum_k(new_h,k)`;
- `z_h,F2 = z_h,static + r_h,legacy + sum_k(new_h,k)`; and
- `z_h,F3 = z_h,static + r_h,legacy`.

F1 changes only `q_h,legacy` to one in the original equation. Every `new_h,k` retains
its original mask, gate, quality multiplier, and residual for F1/F2.

For each factorized logit head `h` (posture and motion) and each fold-by-seed
checkpoint, the proposed architecture is

`z_h = z_h,established + sum_k(g_h,k * q_h,k * r_h,k)`,

where `k` ranges only over new residual experts. The whole coefficient on
`z_h,established` is fixed at `1`; `q_legacy` is fixed at `1` and is absent from the
established path. Reliability and missingness may suppress only a new residual.

Setting every new gate to exactly zero must return that checkpoint's established
temporal baseline pre-decoder logits and decoded probabilities exactly, with the same
preprocessing, class order, temperature, and factorized decoder. The unchanged
seed-probability ensemble is applied only after those per-checkpoint decodes; no unique
"ensemble logit" is assumed. Final ensemble probabilities must also match the retained
baseline exactly. These are hard implementation invariants, not approximate metric
checks. A direct baseline-return path and equality/hash tests over every eligible
development row are required. Any failure stops the study before fitting.

### Three frozen-checkpoint interventions

After a valid continuation lock exists, replay the retained CPTR checkpoints without
updating parameters, thresholds, temperatures, or normalization:

1. **F1 - reliability neutralization:** set only `q_legacy = 1`; retain the learned
   legacy gate and all new residual masks, gates, and values.
2. **F2 - complete anchor restoration:** fix the whole effective established-path
   coefficient `g_legacy * q_legacy = 1`; retain every new residual mask, gate, and
   value. Compare F2 with the original replay and F1 to distinguish gate attenuation
   from the validity multiplier.
3. **F3 - identity control:** use the F2 established path and set all new gates to
   exact zero. F3 must be exactly equal to the matched frozen baseline; it is an
   integrity control and is never eligible for a superiority claim.

Nothing in this draft authorizes those replays; the sequence describes work to bind in
the later protocol and lock.

The primary mechanistic contrast is F2 minus the original candidate on
canonical-scenario-grouped OOF occluded windows, with aggregate macro-F1 and NLL
reported beside it. F1 separates the `q_legacy` term. These are frozen-replay
diagnostics and remain exploratory/noncausal even if favorable; the oracle fallback
result does not count as support for F2.

F2 can open the later fitted-repair stage only if its occluded-window change versus the
original candidate is at least `+0.010` macro-F1, the one-sided 95% scenario-cluster
lower bound (the 5th percentile of the frozen resample distribution) is greater than
zero, the occluded NLL one-sided 95% upper bound (95th percentile) is at most zero, and
its aggregate change versus the established baseline has a one-sided lower bound above
`-0.005` macro-F1. Every NLL contrast is intervention minus comparator, so a negative
value favors the intervention. F1 and F2 are the two nontrivial frozen mechanistic
tests and enter one Okutama-only Holm family using one-sided exact scenario-vector
swaps for their directional stage-opening hypotheses. Failure of any condition records
attenuation as an insufficient explanation and stops the fitted CPTR repair in this
cycle.

If, and only if, the frozen replay supports continued work, the first fitted candidate
family under a later lock keeps the established coefficient fixed at one and permits
only new residual gates. New-stream **mask semantics** and new-stream **gate behavior**
must be evaluated as separate one-factor variants: a mask variant holds the gate
definition and inputs fixed, while a gate variant holds masks, missing-value handling,
and residual inputs fixed. Their combination is not tried in this cycle. Missing
modalities produce an explicit mask and a zero new gate; a numeric zero feature is not
allowed to stand ambiguously for missingness.

## 4. Priority B: anatomy and support evidence with frozen visual features

This stage asks whether spatial evidence adds information beyond the retained V-COCO
factorized ensemble. It does not treat a pose estimate as ground truth or a support
edge as physical-contact proof.

- Use a revision- and hash-pinned inference-time pose estimator to produce real
  predicted joints. Preserve a predeclared number of pose hypotheses per person,
  rather than silently selecting the label-favorable hypothesis. A top-one-pose view
  is a matched secondary ablation.
- Bind the estimator's training corpora, supervision, license, and preprocessing, and
  audit exact and perceptual target-image overlap before use. V-COCO is built on COCO
  imagery: an auxiliary trained or fine-tuned on overlapping COCO/V-COCO images or
  labels is target-exposed, must be reported separately, and cannot establish a clean
  transfer gain or promote the primary B3 path. If no provenance-clean pose estimator
  is available, pose-based results remain non-promotable mechanism diagnostics or
  must move to genuinely independent data.
- For every joint and hypothesis, retain location, confidence, visibility/missingness,
  spatial covariance or heatmap dispersion, hypothesis score, and hypothesis entropy.
  Aggregation must propagate these quantities and must fall back to zero residual when
  support is inadequate.
- Under a newly bound preprocessing run, extract spatial DINO patch tokens from the
  revision-pinned backbone at predicted joint locations and fixed local neighborhoods.
  The retained V-COCO caches contain pooled/CLS features, not a reusable spatial grid.
  Bind the encoder revision and weight hash, source-image content hashes, layer, crop
  transform, token-grid geometry, coordinate transform, interpolation, border
  handling, dtype, and ordered row identities. Do not substitute only a global CLS
  vector or separately encoded, higher-resolution crops in this stage.
- Form anatomical nodes for the major body chains and explicit support-context nodes
  at the feet, below-pelvis/bounding-box base, and seat/context regions. Relations are
  image-evidence features with confidence and missingness, not claims of measured
  contact, force, or ground geometry.

Before constructing B1-B3, either recover and hash the historical row-level fold
assignments and every fold-specific B0 checkpoint, or refit B0 on the newly locked
folds and label it as a new matched fit. A global OOF B0 score cannot be reused as a
residual-training input. For each outer fold, no score or feature used to train,
calibrate, or evaluate a residual may depend on a label from that outer-held fold.

The bounded matched set is: (B0) the exact recovered retained factorized ensemble or
its explicitly new matched refit, (B1) an equal-token/equal-parameter spatial DINO
control using fixed grid nodes rather than pose, (B2) predicted-joint tokens with
anatomy and support edges removed, and (B3) the full anatomy/support residual. B0-B3
share rows, grouped evaluation folds, labels, preprocessing, and evaluation code. The
fitted B1-B3 heads additionally share frozen encoders, feature width, regularization,
seeds, epoch/fit budget, and candidate-selection budget. Those training properties do
not apply retroactively if the recovered no-fit B0 is used; a newly refitted B0 must
match them wherever its model class permits.
Geometry-only, appearance-only, top-one-pose, and hypothesis-shuffle results are
diagnostic ablations, not additional paths for promotion. Person area quartile is
reported but cannot drive architecture choice because the retained scale contrasts are
unconfirmed.

Only a provenance-clean B3 is eligible to advance. It uses the same
baseline-preserving residual equation as Section 3: the retained baseline coefficient
is one and zero new gates return B0 exactly. A target-exposed pose auxiliary remains a
separately labeled diagnostic regardless of its development score.

For each outer-fold model, the new residual is added separately to the factorized
posture and motion logits before the unchanged decoder; decoded probabilities are then
combined by the fixed ensemble operator. Zero residual gates must reproduce each
fold-specific B0 decode and final ensemble probability exactly. No logit is invented
for an already-aggregated probability ensemble.

## 5. Conditional crop, ConvNeXt, and routing stages

Crop work begins with deterministic fixed views derived without labels: whole-person,
torso, foot/support, and fixed-context crops. They are compared with the no-new-crop
baseline under identical grouped folds and the same total decoded input-pixel budget
per person; projected feature width and trainable parameters are matched separately.
Separately encoded crops, crop scales, and fusion weights are fixed in the final
protocol; no outcome-guided crop adjustment is allowed.

Stop before a learned crop router unless at least one fixed-crop candidate passes the
full promotion gate in Section 7 and improves paired NLL. A label-informed best-crop
oracle may quantify an upper bound inside nested development folds, but it is never a
deployable result and cannot itself open the routing stage. If routing is later
authorized by a new amendment, router targets must be produced strictly out of fold,
all experts must be scored on the same people, and zero routing confidence must return
the established baseline.

Stop before integrating ConvNeXt into B3 unless the frozen-DINO anatomy/support stage
passes its gate. A separate, fresh-protocol-authorized, hash-locked complementarity
diagnostic then compares exactly one frozen ConvNeXt head with the full DINO model and
the parameter-matched spatial DINO control on identical OOF people and image folds. If
no such retained artifact exists, this diagnostic is a new fit that must be locked
before it runs; unmatched historical scores cannot substitute for it.

A unique correction is a row the full DINO model gets wrong and the comparator gets
right; a unique harm is a row DINO gets right and the comparator gets wrong. The
ConvNeXt unique-correction fraction uses all full-DINO errors as its denominator and
must be at least `0.10`; corrections divided by harms must be at least `1.25` (zero
harms yields infinity, but the correction-fraction gate still applies). Its correction
fraction must also exceed the spatial DINO control's by more than zero with a positive
paired image-cluster interval. The only fusion diagnostic is a fixed `0.5/0.5`
arithmetic probability average. It must improve macro-F1 by at least `0.005` with a
positive image-cluster interval, while its candidate-minus-DINO NLL interval has upper
bound at most zero. `argmax` ties use the fixed class order. These quantities form one
stop gate, not separate tuning targets. Every comparison uses the same crops, total
input pixels, folds, labels, tuning budget, and inference accounting. Failure records
a negative result; it does not trigger a larger backbone or broader router search.

## 6. Annotation agreement prerequisite

The required two-rater agreement cohort is incomplete. Human-harmonized labels,
support labels, or adjudicated anatomy labels therefore cannot be training targets,
selection endpoints, or confirmatory outcomes in this program.

Before any human-harmonized claim, two independent, prediction- and source-blind
raters must each complete the same 280 unique items plus 20 hidden repeats. Posture,
visible translation, gait, and visibility remain separate axes. Each axis requires
Krippendorff's alpha of at least `0.80`, and each rater requires at least `90%` exact
repeat agreement on every axis. A source- and prediction-blind pass adjudicates
disagreements without overwriting original ratings. If a threshold fails, revise the
guide and run a new pilot; do not relax the threshold after outcomes are seen.

Until this gate closes, provider/source labels remain unchanged and all anatomy,
support, scale, and visibility analyses are explicitly source-tag or machine-derived.

## 7. Analysis and decision rules to freeze in the protocol

The V-COCO primary endpoint is source-tag three-class macro-F1 over eligible people;
the Okutama primary endpoint is provider-label three-class macro-F1 over eligible track
windows. Inference uses `image_id` for V-COCO and the newly bound canonical scenario
group for Okutama. NLL is the primary proper loss; Brier score, accuracy, calibration
error, classwise F1, standing-to-locomotion and locomotion-to-standing errors,
rescue/harm counts, transition, occlusion, visibility, scale, and compute are
secondary. For Okutama, report every scenario and recording change, while treating
recordings as nested descriptive units. A single V-COCO image often lacks all three
classes, so report image-level paired loss/error contributions rather than calling a
one-image quantity macro-F1. A V-COCO provenance or area stratum enters a predeclared
worst-stratum gate only with at least 30 source images and 20 true rows from each class.

Primary uncertainty uses exactly 10,000 paired percentile bootstrap resamples with
seed `20260906`. Draw one master group-index array per dataset and reuse it for
macro-F1, per-class F1, NLL, Brier score, subgroup contrasts, and worst-group
summaries. Persist the arrays, canonical group ordering, label order, NumPy and
bit-generator versions, zero-division rule (`0`), valid-resample counts, and SHA-256
values.
Increasing the resample count after seeing an interval is forbidden. For losses, sum
the sampled group loss totals and divide by sampled row counts. A subgroup replicate
is valid only under its predeclared row and class-support rule; invalid draws remain
counted and visible rather than changing the estimand.

For a paired group-swap test, swap the two models' complete probability vectors for all
rows in each selected group and recompute global macro-F1. Do not sign-flip an
undefined per-image or per-scenario macro-F1 contribution. Use exact enumeration when
feasible. V-COCO uses 100,000 shared Monte Carlo image-group swaps with seed `20260906`,
an absolute two-sided statistic for general promotion, and the plus-one correction
`(extreme + 1) / (draws + 1)`. Frame- or person-row resampling cannot replace image or
scenario clustering. Okutama inference remains conditional on the fitted folds; fold
and leave-one-fold-out sensitivity is descriptive and not unconditional training-set
inference.

Each dataset protocol must enumerate its complete candidate family. Holm correction
at family-wise `alpha = 0.05` is applied separately within the V-COCO and Okutama
primary-test families; no alpha family crosses datasets. The F1/F2 stage-opening
family is defined separately in Section 3. Nominal and adjusted results are retained.
Unplanned slices and all mechanism interpretations are exploratory/noncausal.

A fitted Okutama repair or provenance-clean V-COCO B3 is promotable only when all of
the following predeclared gates pass; F1/F2 open a stage and F3 checks integrity but
none of them is itself a promoted model:

1. macro-F1 change is at least `+0.010`, its paired group-bootstrap 95% CI has lower
   bound greater than zero, and the Holm-adjusted paired test rejects at `0.05`;
2. paired mean NLL does not worsen: its 95% CI upper bound is at most zero;
3. no class F1 point estimate regresses by more than `0.010`; for Okutama, the
   candidate-minus-baseline occluded-window macro-F1 is the safety estimand and its
   one-sided 95% scenario-cluster lower bound must exceed the `-0.010` noninferiority
   margin (`H0: delta <= -0.010`). A bootstrap replicate is valid only when the
   resampled occluded rows contain support for all three true classes; fewer than 95%
   valid replicates makes the gate precision-limited and failed rather than silently
   changing the class convention;
4. worst-group macro-F1 change, every Okutama scenario and recording delta, V-COCO
   image-level loss/error contributions, and transition, visibility, area-quartile,
   rescue/harm, and directional-error tables are reported; no adequately supported
   predeclared V-COCO stratum or Okutama scenario may have a macro-F1 delta below
   `-0.010`;
5. all baseline-identity, provenance, no-forbidden-input, fold-disjointness, and
   matched-budget checks pass.

Before fitting, use only prior development evidence to simulate the attainable power
and interval width at the fixed group count, class balance, and observed between-group
variation. Target at least 80% power at family-wise `alpha = 0.05`; if the cohort cannot
support that target, freeze the attainable precision and report the result as
precision-limited. Do not change the independent unit or add rows after outcomes.

The study-specific protocol must also freeze an end-to-end resource harness beginning
with source-image/video decoding and including pose estimation, crops, backbone
encoding, residual heads, routing, and post-processing. On one fixed cohort and
hardware/software record, report cold start separately, fixed warm-up and CUDA
synchronization, median and 95th-percentile latency, throughput, host and GPU peak
memory, and energy with its sampling interval. Cached-feature timing is secondary.

If the available group count cannot support a useful interval, report the result as
precision-limited. A favorable subgroup cannot rescue a failed aggregate gate, and a
favorable development result cannot authorize opening a holdout.

## 8. Required protocols and fresh locks before any fitting

This draft may be revised through design review. Once approved, create separate
V-COCO and Okutama continuation protocols at new paths; neither protocol or lock can
authorize the other dataset. Complete all of the following before a model fit,
parameter-selection run, new feature extraction, or label-bearing operation:

1. enumerate data roles, protected paths, nonoverlapping output paths, group folds,
   seeds, checkpoints, candidate families, controls, budgets, endpoints, margins,
   multiplicity, stopping rules, and planned commands;
2. add tests for exact zero-gate baseline identity, group disjointness, label-blind
   feature construction, source hashes, and zero reads from forbidden roles;
3. commit each protocol and all bound source first, require a clean worktree, and bind
   the Git commit, Git blob IDs, byte-level SHA-256 values, environment, manifests,
   feature stores, code, and tests in a newly generated lock;
4. archive the exact bytes of every bound protocol/source document and validate them
   against the fresh lock on a clean checkout; retain a rights-safe receipt with the
   outer transfer, package-manifest, path-map, environment, dependency-lock, command,
   and derived-evidence hashes, while keeping private artifacts out of a public release;
5. rerun readiness successfully without weakening its integrity checks, while
   preserving the disclosed five-row calibration-manifest exposure and leaving the
   historical external-CUDA lock untouched; and
6. after execution but before any claim or release, replay the bound code and
   rights-safe evidence on a fresh clone/container and reproduce all declared tables,
   figures, locks, metrics, tests, and manifests within frozen tolerances.

Until those conditions hold, the only permitted work described here is design review,
safety/test implementation, and non-fitting verification or descriptive analysis of
already permitted development evidence. Calibration, confirmation, and test gates
remain closed. Clean committed protocols and fresh locks are necessary preconditions,
not administrative follow-ups to fitting.

## 9. Planned record and claim boundary

When execution is separately authorized, every candidate, control, stopped run,
failure, invariant result, group-level prediction, shared-resample index, metric table,
resource measurement, command, environment record, and artifact hash is retained.
Negative results do not expand the search budget.

This plan supports a bounded development question: whether preserving the entire
established temporal prediction and adding uncertainty-aware anatomy/support residuals
can improve development performance without subgroup or group-level harm. It does not
support claims of causal anatomy, measured physical support, independent replication,
calibration validity, test performance, or deployment benefit.
