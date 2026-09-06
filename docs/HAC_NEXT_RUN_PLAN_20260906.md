# HAC next-run plan — engineering decision, 2026-09-06

## Objective and status

Develop an activity model that adds useful evidence without suppressing an established
prediction when the added evidence is missing or ambiguous. The architecture hypothesis
is separate, uncertainty-aware posture and motion correction from spatial human/context
evidence. Success requires a measured advantage over simple matched controls and an
independent-domain result. Novelty and improvement remain hypotheses.

This is the concrete work sequence selected after
[the team review](HAC_ENGINEERING_RESEARCH_REVIEW_20260906.md). It refines the earlier
[continuation draft](HAC_CONTINUATION_EXPERIMENT_PLAN_20260906.md), retaining its data,
annotation, performance, and resource requirements except where explicitly corrected
below. It is not a completed execution lock: exact eligible feature indices, checkpoint
hashes, fold assignments, and auxiliary-model provenance still need binding. Those
are implementation tasks in R0, not permissions to reconstruct missing historical
documents or silently treat mixed-role files as development-only inputs.

## 1. Order of work and deliverables

| Stage | Concrete work | Size and control | Exit decision |
| --- | --- | --- | --- |
| R0 — inputs and replay contract | Create separate Okutama and V-COCO protocol documents; inventory eligible inputs, fold/seed checkpoints, environment, and exact sources. Build a development-only row-to-feature index without scanning the forbidden mixed-role manifest. | No fits. Close each required field; archive source bytes, then create new locks from a clean commit. | No dependent execution with an unresolved source or data role. |
| R1a — zero-epoch replay | Reproduce seed-43 original outputs and coefficients on all five historical folds; compare original, F1, F2, F3. | Five checkpoint sets; no updates, tuning, calibration or changed preprocessing. | Establish whether initialization attenuation reproduces in real eligible rows. |
| R1b — full coefficient intervention | Repeat original/F1/F2/F3 over all five folds × five seeds, reusing R1a results. Save head logits, gates, actual short-window valid fractions and decoded probabilities. | 25 checkpoint sets total, four fixed interventions; inference only. | F3 exact identity must pass. F2 must satisfy the mechanistic gate below before fitting a repair. |
| R2 — one fixed-anchor repair | Fit the same selected center/parts branch with the established coefficient fixed at one. Change no mask semantics, sampler, loss, or input source in this contrast. | 25 matched fold/seed fits; retain original predictions and teacher as controls. | Promote only if aggregate, occlusion, proper-loss and worst-group gates pass. Otherwise keep teacher. |
| T1 — is time order necessary? | Compare the matched teacher with a simple permutation-invariant feature-pooling head using the same cached frame sequence and the teacher's all-valid mask. | One new control family, five folds × five seeds. It can follow R0 independently of R2. | If pooling matches or exceeds teacher within the predeclared practical margin and is cheaper, prefer it; do not claim sequence reasoning. |
| T2 — distinct-frame sampling | Compare the declared repeated-index sequence with a fixed distinct-index sequence over the same temporal extent and center policy, refitting both matched heads. | Two locked sampler arms; five folds × five seeds each. Separate from R1/R2. | Require a benefit over its newly matched sampler control; never attach the gain to gate repair. |
| V0 — resolve static attribution | Fit mixed flat, mixed factorized without reliability, and mixed factorized with reliability on shared inner and outer folds. Retain a matched mixed linear SVM. | Four explicit head families; equal candidate-selection budget, fixed features. | Determine whether reliability/factorization justify their cost; no pose model yet. |
| V1 — add spatial evidence | Compare B0/B1/B2/B3 and uncertainty controls defined below, after binding spatial features and clean auxiliary provenance. | Frozen encoders; one common head configuration initially, five outer folds × seeds 42–46 per trained arm. | The full model must beat the ordinary spatial and joint-only controls to earn a structural claim. |
| V2 — mechanism and uncertainty checks | Test anatomy deletion, support deletion, top-one pose, equal averaging, and predicted uncertainty on fixed inputs and budgets. | Predeclared diagnostic comparisons, no promotion of the best ablation after seeing scores. | Remove ingredients whose benefit is unestablished. |
| X — independent replication and cost | Freeze one surviving system and run the separate independent-domain protocol plus end-to-end resource measurement. | One untouched external evaluation; same reference model and declared resource cohort. | Broader architecture/solution claim requires independent support. |

The plan is sequential by evidence, not a commitment to run every branch. R1a/R1b
are the first GPU work because they are inexpensive and distinguish a specific
mechanism. T1 tests whether the problem needs order-sensitive temporal complexity.
V0 avoids crediting a complex design for a simpler representation gain. T2 and V1
receive compute only after their own input contracts and resource estimates are complete.

Counts denote experimental units, not independent statistical samples. Where new
folds are needed, refit every comparator on those folds; an old unmatched score is
not a substitute. Any inner model selection requires common explicit inner folds and
the same predeclared number of evaluated settings. Measure one representative fit's
memory/time before scheduling a family and record the resulting compute budget;
do not invent unmeasured GPU-hour estimates or enlarge the grid after a failure.

## 2. R0: make execution possible without changing historical evidence

The [scenario receipt](HAC_SCENARIO_LINEAGE_REVIEW_20260906.md) establishes that the
11 OOF `recording_id` values denote scenarios in the historical code. Preserve these
groups and bind their eligible videos/tracks explicitly; do not group them again by
misreading the first component as a drone ID. Every synchronized view and track stays
inside its scenario's fold. Audit row identities and feature positions using permitted
role-specific exports or a provider-supplied development-only manifest. If those cannot
be constructed without a protected-role scan, record the missing artifact precisely
and continue independent synthetic/code work rather than silently broadening access.

Bind all 25 CPTR checkpoint sets and matching static/temporal anchors before loading
them. Filenames exist; their presence is not a verified weight hash or safe-loading
guarantee. Prefer tensor-only checkpoint loading and validate expected keys/shapes.
Record class order, temperatures, normalization, masks, feature dtype and source-index
mapping. The invariant is checkpoint-level logits and probabilities followed by the
unchanged probability ensemble; never invent a unique logit for an ensemble.

R1 must retain the exact short indices `[4, 6, 6, 8, 8, 10, 10, 12]`. Missingness,
sampling, gate reliability, and loss changes are separate factors. Freeze T2's distinct
indices and timestamp encoding before fitting; include any asymmetry imposed by eight
distinct samples and an explicitly retained center in the stated input contract.
T1 inherits the teacher's all-valid inference mask; adding provider-occlusion masking
would change another factor. R1/R2 retain historical provider-occlusion inputs for
matching and are metadata-assisted mechanism studies. A deployable image-derived
visibility arm needs its own locked comparison.
For T1, predeclare macro-F1 noninferiority margin 0.005: the one-sided 95% scenario
lower bound for pooling-minus-teacher must exceed −0.005, the NLL upper bound must
be at most zero, and measured end-to-end p95 latency must improve on the same cohort.
A nonsignificant superiority test does not establish equivalence or noninferiority.
Bind these criteria and the resource comparison procedure in R0 before fitting.

V-COCO R0 must explicitly authorize combined old `train`/`val` as development under a
new protocol and reconstruct shared image folds. Neither source tag is a fresh holdout.
If old fold-specific baseline models are unavailable, label B0 as a new matched refit.
No residual training feature or base score may use the outer-held fold's labels.

Do not reuse the incomplete historical external-CUDA document lock. Commit new
dataset-specific protocols and source first; lock commit, Git blobs, SHA-256 values,
environment and exact source snapshots. Preserve the prior access incident and the
remaining source-lineage gap in every new receipt. No new test/confirmation/calibration
data is part of this program's development inputs.

## 3. R1/R2: test the precise legacy-path mechanism

For each posture/motion head, let `r_legacy = z_teacher − z_static` and let `n` be
the sum of all original new-branch contributions, with masks and gates unchanged.

| Intervention | Head logit |
| --- | --- |
| Original | `z_static + g_legacy × q_legacy × r_legacy + n` |
| F1 | `z_static + g_legacy × r_legacy + n` |
| F2 | `z_teacher + n` |
| F3 | Direct return of the exact teacher output; all new contributions disabled |

Also audit the algebraic `z_static + r_legacy` path, but use the direct-return path
for bit-exact F3 identity, avoiding subtract/add rounding. Tests must cover each head,
decoder, temperature and ensemble, complete missingness, finite outputs, and identical
row ordering. Log the actual short-window `q`; full-window any-occlusion is a distinct
variable. Run numerical gradient checks before training the repair.

The engineering gate is exact F3 identity on every eligible row. The mechanistic
stage-opening gate retains the initial draft's fixed thresholds: F2 minus original
must improve occluded macro-F1 by at least 0.010, with one-sided 95% scenario-bootstrap
lower bound above zero; its occluded NLL upper bound must be at most zero; aggregate
F2-minus-teacher macro-F1 lower bound must exceed −0.005. Apply Holm to F1/F2's
directional whole-scenario swap tests within their declared diagnostic family. These
are tests of model interventions on retained development data, not causal claims
about physical occlusion or independent confirmation.

Failure of the performance gate stops fitted CPTR repair this cycle. It does not erase
the verified identity issue. Success only authorizes the R2 question; R2 still needs
its own matched development and later independent validation.

## 4. The proposed architecture, stated narrowly

Use an established factorized classifier as an anchor. For each person, construct
one frozen spatial token grid and exactly `K=4` label-blind pose hypotheses, with
explicit validity, geometry dispersion, and hypothesis weights. Four is a proposed
engineering budget to freeze in V1, not a number chosen from observed outcomes.
If the pinned auxiliary cannot supply defensible hypotheses, resolve that interface
before extraction; synthetic jitter is not automatically a pose posterior.

Each hypothesis samples the same count of spatial tokens at joint neighborhoods and
fixed support-context locations. A small common encoder produces separate two-logit
posture and motion residuals `r_h,k`. Anatomical/support relations are visual evidence
features, not measured contact, force, or proof of motion. Head-specific aggregation
may represent confident upright posture alongside uncertain locomotion:

`a_h = sum_k(w_h,k × r_h,k)`

`z_h = z_h,anchor + m_h × g_h × a_h`.

Here `m_h` is an explicit availability mask, and `g_h` depends on permitted evidence
quality, hypothesis disagreement and baseline diagnostics. The anchor coefficient is
always one. An inference path with unavailable evidence or deliberately disabled new
gates returns the exact anchor. Disagreement is only an uncertainty feature; call it
calibrated only after evaluating its relation to actual error on separate eligible
data. Consistent hypotheses can all be wrong.

Avoid a training dead branch: initialize residual output weights and biases to zero and the
available-evidence gate to a nonzero value. Initializing both multiplicative factors
to zero can remove useful gradients. The direct-return inference shortcut must not
detach all zero-initialized residuals during training. Add tests for nonzero residual
head gradients, missing-input behavior, and exact disabled-path predictions.

## 5. V1/V2 controls and attribution rules

| Arm | Input/structure | What it tests |
| --- | --- | --- |
| B0 | Matched V0 established classifier | Reference for aggregate benefit. |
| B1 | Fixed-grid tokens with equal trainable capacity, token count, input pixels and hypothesis slots | Extra spatial information/capacity without predicted anatomy. |
| B2 | Predicted-joint tokens with no anatomy or support relations | Pose-based sampling without claimed relational reasoning. |
| B3-average | Full anatomy/support residual, fixed averaging over the same four hypotheses | Structure with no learned uncertainty weighting. |
| B3-uncertainty | Full structure with separate posture/motion weighting | Single proposed full candidate. |
| Top-one control | One selected pose repeated into the same token budget and aggregation layout | Whether multiple plausible poses add information. |
| Anatomy deletion / support deletion | Remove only the named relation family; keep nodes, tokens and parameter allocation matched | Attribution to the claimed edge family. |

B3-uncertainty is the only architecture promotion candidate in this cycle. Primary
contrasts are against B0, B1, B2 and B3-average, entered into one predeclared V-COCO
family. Its point macro-F1 gain must be at least 0.010 over B0 with a positive paired
95% interval and Holm-adjusted whole-image swap test; NLL must not worsen. For claims
about structure or uncertainty, the corresponding control contrast must also have a
positive paired interval and pass the declared family correction. Do not attribute
an ingredient's value solely to the B0 contrast. Failure against B1/B2 retains a
simpler solution; failure against averaging removes learned uncertainty from the claim.

Keep five outer image folds and seeds 42–46, shared inner folds where used, one fixed
initial optimization configuration, frozen encoders, matched total decoded pixels,
tokens, head width/parameters, update count and selection budget. Do not mix new crops,
a larger backbone, extra part-state labels, and graph structure in the same comparison.

Auxiliary provenance is a hard input requirement. V-COCO uses COCO images, and HOT
explicitly contains V-COCO in its training data. Verify overlap and training corpora
before choosing pose/contact weights. Without a clean auxiliary, report a separate
target-exposed mechanism study or move to independent imagery; do not claim clean
transfer. The required DINO spatial grids are new features: old pooled/CLS caches do
not provide the proposed joint-local tokens.

The initial two-rater agreement prerequisite remains 280 unique items plus 20 hidden
repeats per rater, each axis alpha ≥0.80 and repeat agreement ≥90%. Add blinded
error-cause coding and correctly classified controls to a separately frozen design;
an enriched error audit cannot estimate population prevalence without its sampling
weights. Human annotation work requires actual independent raters. Until complete,
use provider/source labels and machine-derived evidence; no harmonized-label claim.

## 6. Statistical, resource and stop rules

Use exactly 10,000 paired group resamples with seed 20260906; save actual index arrays,
canonical group order, RNG versions, and hashes. Aggregate confusion counts before
macro-F1; aggregate loss sums divided by sampled row counts before NLL. Use fixed
three-label zero-division semantics. Single-true-class slices use recall and confusion.

Use whole-group probability-vector swaps for the null test, exact enumeration for
11 scenarios where feasible, and 100,000 fixed Monte Carlo image swaps with plus-one
correction for V-COCO. Historical uncentered bootstrap sign tails are not replacements
for null-based tests. Declare diagnostic and promoted families explicitly; no alpha
pooling across datasets. Repeated decisions on reused development evidence remain
development results even with correction. A fresh independent evaluation is needed
for a generalization claim.

Retain the earlier draft's final guardrails: no class point regression exceeding
0.010, occlusion macro-F1 one-sided lower bound above −0.010 for Okutama, at least 95%
valid subgroup resamples with all classes supported, and no adequately supported
predeclared scenario/stratum below −0.010. Report every scenario, rescue/harm counts,
transition/visibility slices, NLL and Brier alongside macro-F1. Better ECE alone cannot
justify a model with worse F1 or Brier.

Simulate attainable precision from prior development evidence before fitting. If
11 scenarios do not support the target precision, record the limit; more frames or
seeds cannot create independent scenarios. Thresholds are success criteria, not
predicted effect sizes or guarantees.

Measure decoding, pose extraction, spatial features, all heads and routing in the
resource harness. Record cold start, synchronized warm latency, p50/p95, throughput,
host/GPU memory and energy sampling. Cached-head timing is secondary. New complexity
must justify itself on an accuracy/proper-loss/resource frontier.

Do not expand dual-clock, trajectory integration, counterfactual losses, masked
pretraining, CPTR SigLIP, GroupDRO, or LoRA branches in this cycle. Fixed-crop routing
and ConvNeXt remain conditional later studies under their existing strict entry
gates. A negative result does not open a wider architecture search.

## 7. Work completed versus work next

Completed: independent implementation/statistics/prior-art review; exact development
component comparisons and conditional correlations; persisted shared resamples;
scenario-lineage correction; synthetic CPTR attenuation and duplicate-sampler tests;
the research report, figure and this sequence.

Implemented commands available now, with a fresh output path for each replay:

```powershell
.\.venv\Scripts\python.exe experiments/analyze_hac_component_relationships.py --output-dir .runs/research_20260906/component_review_replay --resamples 10000
.\.venv\Scripts\python.exe experiments/audit_cptr_initialization_contract.py --output .runs/research_20260906/initialization_replay/summary.json
.\.venv\Scripts\python.exe -m pytest tests/test_hac_component_relationships.py tests/test_cptr_initialization_contract.py
```

Next implementation is R0 followed by R1a: a role-safe input bundle, two new protocol
documents and source locks, then a frozen replay runner with explicit F1/F2/F3 modes.
These runner modes and new model fits have not been implemented or executed in this
review. The first real-data milestone is a faithful zero-epoch replay; the first
architecture milestone is a gain over the strongest matched simple control. Neither
milestone is represented as already achieved.
