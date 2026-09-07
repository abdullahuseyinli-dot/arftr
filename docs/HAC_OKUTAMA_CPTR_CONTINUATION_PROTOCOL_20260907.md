# Okutama CPTR continuation protocol — 2026-09-07

## Status and question

This protocol is declared before continuation replay or fitting. Its machine-readable
authority is
[`experiments/okutama_cptr_continuation_protocol.json`](../experiments/okutama_cptr_continuation_protocol.json).
It supersedes no historical lock and does not repair the missing historical amendment
documents.

The question is narrow: does preserving the established temporal path remove CPTR's
occlusion-sensitive regression while every other input, mask, decoder, and ensemble
operation remains fixed? R1 is a retained-development mechanism test. R2 becomes
eligible only if R1 passes its predeclared gate.

## Eligible data and protected boundary

| Scope | Role | Rows | Scenarios | Use |
| --- | --- | ---: | ---: | --- |
| Grouped cross-fit OOF | Primary reused development | 4,977 | 11 | R1 decisions and grouped inference |
| Fixed development validation | Descriptive reused development | 1,383 | 3 | Role-safe materialization/index consistency; not an R1 decision or output |
| Calibration, confirmation, test | Protected | 0 accessible | 0 accessible | No continuation input |

The 11 OOF groups are scenario identifiers stored under `recording_id`, as established
by the hash-matched historical source review. A scenario never crosses a fold. The
fixed folds remain:

| Fold | Held scenarios |
| --- | --- |
| 0 | `1.4`, `2.2`, `2.5` |
| 1 | `1.5`, `2.11` |
| 2 | `1.10`, `1.2` |
| 3 | `2.7`, `2.8` |
| 4 | `1.11`, `1.3` |

The mixed temporal manifest and the pre-split `development_metadata.csv` are forbidden
inputs. The eligible index is reconstructed from the 6,360 independently locked
sample IDs, the audited aggregate selected-centre counts, and the historical sort
order: scenario, source recording, provider track as a string, then center frame.
Each eligible source recording must contain exactly its audited number of rows. The
five checkpoint folds replay only the 4,977 grouped-crossfit OOF rows. The extra 1,383
rows complete the role-safe index and materialized bundle, but cannot enter an R1
decision or output. Neither scope is an independent estimate. The
builder persists source positions for eligible rows only to authorize the selective
copy, then writes those rows in contiguous eligible-only bundle order. It does not
persist complement identities or values.

The source archive contract is `TrainSetFrames.zip`, 5,770,432,522 bytes, with the
historically audited SHA-256
`c021ce8a12c84e083f359023ffd41c145561aaedb48b118e7c5416d5ddcecb73`.
The protocol locker verifies that prior receipt by hashing the supplied archive as an
opaque byte stream and checking its filename and exact byte count. It does not parse
any member. The bundle builder may open annotation members only for the 14 proven
eligible scenarios to construct the exact 17-frame occlusion masks.
After eligibility is proven, the source locker opaque-hashes each complete historical
array file and matches it to the store declaration without NumPy decoding, indexing,
or value interpretation. Feature materialization may then read only the proven
eligible positions. It may not decode, materialize, summarize, or persist a complement
identity or array value.

## Frozen replay inputs

The materialization lock binds all 75 development checkpoint sets: static, temporal
teacher, and CPTR for five folds and seeds 42–46. Each checkpoint hash is taken from
its role-specific run summary and verified against its bytes. Every summary must report
zero validation, calibration, and confirmation access. Checkpoints are loaded later
with `weights_only=True`, followed by exact state-key and shape validation.

The role-safe bundle binds the retained primary teacher ensemble explicitly: 4,977
Unicode sample IDs in exact primary bundle order, seeds `[42,43,44,45,46]` as int64,
and a finite normalized float32 probability tensor shaped `(4977,5,3)` in the declared
class order. The temporal-control fixed epochs are additionally tied to the role-safe
provenance file `.runs/vcoco_v3/temporal/development_final/summary.json`, SHA-256
`48301d3da04085f810d7fc7d146d4f61b770ff77b311c92cf13328083c726ece`.

The short stream remains `[4, 6, 6, 8, 8, 10, 10, 12]`. R1 preserves its historical
mask. The distinct T2 stream is `[4, 5, 6, 7, 8, 10, 11, 12]`; it retains center index 8
in slot four and intentionally omits offset `+1` while preserving both endpoints. Both
T2 arms retain the same sinusoidal slot-ordinal encoding for slots 0 through 7 and no
absolute/source-frame timestamp; only the selected source-frame indices change.

For each posture and motion head, define `r = z_teacher - z_static` and let `n` be all
new-branch contributions with their existing masks and gates:

| Mode | Output |
| --- | --- |
| Original | `z_static + g_legacy * q_legacy * r + n` |
| F1 | `z_static + g_legacy * r + n` |
| F2 | `z_teacher + n` |
| F3 | Return the exact retained teacher probabilities with every new contribution disabled |

R1a evaluates seed 43 over all five folds, with no updates. Discrete recovery of `q`
from retained outputs is permitted only after the zero-head and initialized-gate
invariants are verified. R1b evaluates all 25 fold/seed checkpoints and requires the
exact role-specific window-mask artifact. Both stages save head logits, gates, actual
short-window valid fractions, decoded probabilities, row order, and checkpoint-level
outputs before the unchanged probability ensemble.

## Decisions and later controls

F3's statistical probabilities and decoded outputs must match the retained teacher bit
for bit on every eligible row. A fresh direct-teacher forward pass on the current GPU is
only a reproduction check and may not replace that retained statistical input; its
maximum absolute probability drift must be at most `5e-4`. R2 opens only when F2 over
Original improves occluded macro-F1 by at least 0.010, its one-sided 95% scenario lower
bound is above zero, its occluded NLL upper bound is at most zero, and the aggregate
F2-minus-teacher macro-F1 lower bound exceeds −0.005. F1 and F2 directional scenario
swap tests form one Holm-corrected family.

R2 contains one fitted family: the same center/parts model with the established anchor
coefficient fixed at one. It uses folds 0–4, seeds 42–46, and historical fixed epochs
`{42: 2, 43: 0, 44: 1, 45: 9, 46: 1}`. No mask, sampler, loss, feature, or candidate
budget changes in this contrast.

T1 contains one permutation-invariant head. It takes the arithmetic mean over all eight
legacy slots, including their repeated weighting, uses the teacher's all-valid mask,
and receives no position or timestamp signal. Its macro-F1 noninferiority lower bound
must exceed −0.005, its NLL upper bound must be at most zero, and measured same-cohort
p95 latency must improve. Accuracy and proper loss compare against the exact retained
five-seed teacher probability ensemble. Latency compares against the newly matched T2
legacy-repeated fold-0/seed-42 head on the same cached-feature cohort. The authoritative
latency measurement loads both completed checkpoints with tensor-only loading and exact
request, artifact, key, shape, and dtype validation, then holds both models in one
process. It uses the same first 64 fold-0 rows and the identical all-valid legacy-short
batch for both arms, performs 20 shared warm-up pairs, and times 200 synchronized paired
rounds in an exactly balanced AB/BA order. The strict latency check requires both a lower
observed candidate p95 and a below-zero one-sided 95% upper bound for the paired-round
bootstrap p95 delta (10,000 resamples, seed 20260907). Raw paired timings, order strata,
p50/p95, deltas, and ratios are retained in `joint_latency_benchmark.json`; timings made
inside separate training workloads are diagnostics only. T1 must also pass every common
promotion guardrail. Because the role-safe input is a cached feature bundle, this
measurement is component-level evidence: a fully preferred T1 remains pending the
master plan's end-to-end decode, crop, feature, and head p95 measurement. Thus a cached
head pass is reported separately and never sets full preference eligibility by itself.
T2 refits two
matched arms, one for each declared index sequence, over the same five folds and seeds.
The distinct arm advances only if its aggregate macro-F1 two-sided 95% lower bound is
above zero, aggregate NLL one-sided 95% upper bound is at most zero, one-sided exact
whole-scenario swap p-value is at most 0.05, and every promotion guardrail passes. It
has no added point-size threshold, and any gain is attributed only to sampling.

The temporal-control implementation uses only the 4,977 primary cross-fit rows in the
historical scenario folds. T1 takes the raw arithmetic mean of all eight legacy slots,
then applies LayerNorm, a 256-wide linear layer, GELU, dropout 0.1, and the same
factorized classifier; it has no position, time, or attention path. Both T2 arms use a
`TemporalFactorizedTeacher` with model width 256, two layers, four heads, feed-forward
width 512, dropout 0.1, maximum length eight, and an all-valid mask.

All three temporal-control arms use AdamW at `2e-4`, weight decay 0.01, batch size 64,
10% linear warmup followed by cosine decay, gradient clipping at 1.0, label smoothing
0.02, fit-partition inverse-frequency weights, CUDA AMP, and no candidate selection.
Fixed epochs are `{42: 5, 43: 5, 44: 5, 45: 5, 46: 3}`. Before fitting, prepare mode
writes and receipts the exact 4,977-row fold map, 10,000-by-11 scenario bootstrap
indices from `default_rng(20260906)`, and all 2,048 exact 11-scenario swap assignments.
The separate `OKUTAMA_TEMPORAL_CONTROLS_LOCKED_BEFORE_FITTING` lock binds those inputs,
the role-safe bundle, committed runner/modules, repository tree, and environment.
Benchmark mode completes both fold-0/seed-42 T1 and T2-legacy workloads before the full
schedule, recording runtime and peak CUDA allocated/reserved memory for the simple and
expensive families, then runs the joint latency experiment.

All grouped bootstraps use exactly 10,000 whole-scenario draws with seed 20260906.
Confusion counts are aggregated before macro-F1, loss sums before division, and all
three classes use zero-division value zero. The primary 11-scenario swaps are enumerated
exactly where feasible. Per-class regression cannot exceed 0.010; the occlusion lower
bound cannot fall below −0.010; at least 95% of subgroup resamples must support all
classes; and no adequately supported declared scenario or stratum may fall below
−0.010.

## Lock sequence

The materialization and replay phases prevent a source/output cycle:

1. Commit the protocol, bundle builder, replay runner, model modules, and locker.
2. Build `eligible_index.csv`, `eligible_window_masks.npz`, its immutable
   `window_mask_summary.json`, and `input_inventory.json` without feature-array-value
   access.
3. Create `okutama_materialization_lock.json` with status
   `OKUTAMA_CPTR_REPLAY_MATERIALIZATION_LOCKED_BEFORE_FEATURE_ACCESS`; during this
   step, opaque-hash all five declared feature-array files and bind their observed
   paths, byte counts, and matching SHA-256 values without array decoding.
4. Materialize `eligible_feature_bundle.npz` from eligible positions only and write
   `bundle_summary.json` binding the phase-one lock.
5. Create the phase-two lock with status
   `OKUTAMA_CPTR_FROZEN_REPLAY_LOCKED_BEFORE_EXECUTION`, then run R1.
6. Independently prepare the primary-only temporal fold map and fixed scenario
   resampling from the locked bundle, then create
   `okutama_temporal_controls_execution_lock.json` with status
   `OKUTAMA_TEMPORAL_CONTROLS_LOCKED_BEFORE_FITTING` before any T1/T2 benchmark or fit.

The locker supports `--mode prepare`, `--mode lock`, and `--mode check`. A lock can be
created only from a clean worktree; every execution source must match its blob in the
bound commit. Both locks record the complete installed Python distribution set, Python
and platform versions, PyTorch/CUDA/device details, and the committed requirements
snapshot.

R1 writes immutable per-checkpoint `fold-{fold}/seed-{seed}/replay.npz` files beneath
`.runs/research_20260907/cptr_replay`, plus stage aggregate predictions, exact scenario
resample indices, and a stage summary. Per-checkpoint summaries have status
`OKUTAMA_CPTR_FROZEN_INTERVENTION_REPLAY_COMPLETE`; aggregate summaries have status
`OKUTAMA_CPTR_FROZEN_INTERVENTION_AGGREGATE_COMPLETE`. R1a and R1b use distinct
stage-prefixed aggregate files, while R1b deliberately reuses the five immutable
seed-43 checkpoint outputs. Therefore the stage-R1b execution lock is created up front
and authorizes both the R1a seed-43 smoke replay and its later all-seed completion; an
R1a-only lock must not be created and then replaced between those invocations.

## Recorded access history

The original receiving audit printed five calibration rows, and an interim router
raw-read 1,979 calibration strings solely to exclude them. A later implementation
review loaded the 8,339-row pre-split `development_metadata.csv` for schema/count
inspection and printed five rows believed to be training rows. During this protocol
audit, an aggregate receipt print exposed three complement scenario identifiers and
their per-video selected-centre counts. Those aggregate identifiers/counts are
authorized solely as structural evidence for cumulative packed-store offsets: they
therefore influence the numeric eligible source positions and the lock that hashes the
eligible index. No complement row identity, label, decoded array value, prediction, or
model outcome is read or persisted. Before the phase-one lock, five NumPy headers from
the mixed stores were opened with `mmap_mode='r'` to verify only public shapes and
dtypes; no array element was indexed or materialized (`feature_array_values_read=0`).
No protected image, feature-array value, probability, checkpoint, or confirmation/test
input was opened in these events. Current-run protected-value access counters must
remain zero and the forbidden paths are rejected before file access.
