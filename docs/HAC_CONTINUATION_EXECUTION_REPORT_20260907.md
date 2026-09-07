# HAC continuation execution report — 2026-09-07

## Executive decision

The locked continuation cycle is complete for every branch that its gates allowed to
run. It produced one strong architectural direction, but **not** a confirmatory
breakthrough claim:

- R0 successfully constructed a role-safe Okutama bundle and exact masks from 6,360
  authorized development rows. The primary temporal analysis used the 4,977 OOF rows
  from 11 scenarios; 1,383 fixed-validation rows from three scenarios remained
  descriptive-only.
- R1a failed its frozen numerical-reproduction gate before any intervention metric
  was emitted. The maximum Original probability drift was `0.0009456575`, above the
  locked `0.0005` limit. The error is consistent with Ada-to-Blackwell GPU/kernel
  portability drift, not bundle reconstruction. R1b and R2 therefore stayed closed.
- T1 completed and showed that arithmetic temporal pooling is a useful low-cost
  proper-loss and efficiency primitive, but an inadequate replacement decision head:
  it improved NLL and cached-head speed while failing macro-F1 noninferiority and
  robustness guardrails.
- T2 completed and showed a promising development signal for distinct-frame sampling
  (`+0.0034223` macro-F1 versus a newly matched repeated-frame fit), but the declared
  two-sided interval, scenario-swap test, occlusion bound, and scenario guardrail did
  not pass. It is a component lead, not a promoted system.
- V0 failed closed on its first bounded mixed-linear-SVM workload because the first
  grouped calibration fit reached the locked 2,000-iteration ceiling. No prediction,
  selection, checkpoint, or metric was written. The matched four-family V0 comparison
  is consequently unavailable; V1 remains blocked by its independently missing
  spatial and pose inputs.
- X, an untouched independent-domain replication with end-to-end resource
  measurement, remains unavailable. All conclusions below are reused-development
  evidence.

The best next architecture is a **baseline-anchored, visibility-aware dual-path
temporal residual**: retain the established transformer decision path; add true
relative-time, distinct-frame, visibility/repetition-aware residual evidence; and use
arithmetic pooling only as a probability-quality/confidence sidecar whose calibration
must be tested. Its residual must be bounded and initialized at zero so missing or
unreliable evidence returns the anchor exactly.

## Execution authority and immutable scope

All protocol-authorized fitting and primary inference was bound to clean commit
`458cd17717b4c77cfe830517f48d72b3bad26aea` (tree
`db8e39737fd5b5ccbbddc4b503dcca71d48cdec3`). The implementation commits are:

- `fd2c440` — implement the locked HAC continuation experiments.
- `458cd17` — correct retained-teacher sample-ID serialization before valid locking
  and execution.

The five retained lock hashes are:

| Lock | SHA-256 |
| --- | --- |
| Okutama materialization | `4c4f42756a7ca7f36a15c3fb20d086367624dcdfc91411ef5ace0811d12fddb6` |
| Okutama R1 execution | `58f20c8fd8bd8c8d7c83c25ab3a508fd665c9c04217dc46d3175d7266e023d1e` |
| Okutama T1/T2 execution | `63cf773ebe935f46767b990725cdf530e1f32ea17b3ab8dbffc40085f74495cb` |
| V-COCO source | `82affc275d0646b5e7f3d95d80653b14271bf2afc2a0bd05fc3ebb6c5e58b55b` |
| V-COCO V0 execution | `1dfeac2c3b895141a3e44fda272213e2a90aa1e1003f51ac7566f243c6d91d73` |

The official `TrainSetFrames.zip` was opaque-hashed at 5,770,432,522 bytes and
SHA-256 `c021ce8a12c84e083f359023ffd41c145561aaedb48b118e7c5416d5ddcecb73`.
Only the 54 explicitly allowed annotation members were used to make masks; no archive
image member or disallowed annotation member was opened.

Post-materialization R1/T1/T2 and V0 fitting runners report zero protected-role and
mixed-manifest access. R0 deliberately opaque-hashed five mixed feature stores and
selectively decoded eligible values; excluded/protected-value counters remained zero.
That does not erase the pre-lock incidents disclosed in
`HAC_CONTINUATION_PROVENANCE_20260906.md`: five historical calibration rows were
printed; 1,979 calibration strings were routed only to exclude them; the broad
8,339-row Okutama metadata table was loaded and its first five rows printed; aggregate
complement scenario names/counts became visible; and five mixed NumPy headers were
opened for shape/dtype only. No protected feature value, image, checkpoint,
prediction, confirmation input, or test input entered this cycle's fits or metrics.
The earlier public README exposure to historical aggregate V-COCO test metrics is also
retained as a protocol disclosure; those aggregates were excluded from every current
input, decision, and comparison.

## R0 — role-safe materialization

R0 passed. The retained bundle contains 6,360 rows from 14 eligible scenarios and has
SHA-256
`66388c64f987371d278d298199d13f02957f6dbf47f12feafa463b1da9504852`.
The bundle summary has SHA-256
`4edaefe263dfa3e0e82fbd965cf2af03520d70a7bfbf10eb2848c0bce4e5d6f5`.
It includes the exact retained teacher tensor `(4977, 5, 3)` in `float32`, the
historical repeated short indices `[4, 6, 6, 8, 8, 10, 10, 12]`, a separately locked
distinct sampler, exact masks, and opaque source-store hashes. The five mixed source
arrays were hashed before and after selective copying; only eligible values were
copied and the complement was never decoded.

An earlier generated bundle at commit `fd2c440` stored teacher sample IDs with NumPy
object dtype. The execution locker rejected it safely under `allow_pickle=False`
before any replay or fit. Those invalid, pre-fit artifacts were moved intact to
`.runs/research_20260907/quarantine_invalid_teacher_object_dtype_fd2c440`; the fix
uses a non-object Unicode dtype. The quarantine is recoverable and is not evidence.

## R1 — frozen CPTR replay

The executed command was:

```powershell
.\.venv\Scripts\python.exe experiments\replay_okutama_cptr_interventions.py `
  --protocol-lock .runs\research_20260907\protocol_locks\okutama_r1_execution_lock.json `
  --seeds 43
```

It stopped on fold 0 / seed 43 before writing replay output:

```text
Original replay exceeds the fixed reproduction tolerance: 0.00094565749168396
```

The locked maximum was `0.0005`. It was not changed after the result. Because R1a did
not reproduce, no F1/F2 intervention result was inspected or emitted, R1b was not run,
and the gated R2 repair was not fit.

### Post-failure exploratory numerical diagnosis

This no-write diagnosis was performed after the frozen gate failed. It used the same
clean commit but is exploratory evidence, not part of or a repair to the locked gate.
The prior result came from an RTX 4060 Ada GPU; this run used an RTX PRO 3000
Blackwell GPU with PyTorch `2.11.0+cu128`. Reinstating the historical batch size,
AMP, row order, normalization, seeding, cuDNN flags, and automatic SDPA left the same
failure:

| Diagnostic | Original | Direct teacher |
| --- | ---: | ---: |
| Maximum absolute probability drift | `0.0009456575` | `0.0009521246` |
| Mean absolute drift | `0.00006577` | `0.00006244` |
| p99 absolute drift | `0.0004661` | `0.0004630` |
| Rows over `0.0005` | 15 / 1,196 | 12 / 1,196 |
| Argmax changes | 0 | 1 |

Original and direct-teacher rowwise maximum errors correlate at about `r=0.856`, and
candidate error correlates with the expected `q × teacher-error` propagation at about
`r=0.839`. FP32, BF16, batch changes, and forced math SDPA did not satisfy the lock.
The retained F3 statistical teacher tensor remains bit-exact. This is strong evidence
for upstream GPU/kernel numerical drift, but it cannot convert the failed locked
replay into a passed result.

Decision: **R1 portability failure; CPTR mechanism unresolved on this hardware.** A
prospective portability study may predeclare max, p99, and argmax invariants or replay
on the original Ada GPU. It may not amend this result retrospectively.

## T1/T2 — completed temporal controls

The bounded resource run and the resumable full run were:

```powershell
.\.venv\Scripts\python.exe experiments\run_okutama_temporal_controls.py --mode benchmark
.\.venv\Scripts\python.exe experiments\run_okutama_temporal_controls.py --mode full
```

All 75 locked workloads completed across five folds, five seeds, and three arms. The
full invocation reused two benchmarked workloads and took 522.881 seconds. It covered
4,977 OOF rows and 11 scenarios. The summary SHA-256 is
`b8cf73e99f3c32486cfc21d4cb67fd8072b540d49806de80f1b0c2050b280fb9`;
the ensemble-prediction artifact SHA-256 is
`1dd9dbc85ca346c08ff0dc098c71bca7eb5ecc792d9c18e053990904a720a481`.

### Aggregate results

| Arm | Macro-F1 | Accuracy | NLL | Brier |
| --- | ---: | ---: | ---: | ---: |
| Exact retained teacher | `0.7164832` | `0.7130802` | `0.7538911` | `0.4090998` |
| T1 arithmetic mean | `0.7081586` | `0.7068515` | `0.7084791` | `0.4060002` |
| T2 newly matched repeated | `0.7158154` | `0.7126783` | `0.7552795` | `0.4094292` |
| T2 fixed distinct | `0.7192377` | `0.7162950` | `0.7488200` | `0.4050321` |

### T1 decision

Against the retained teacher, arithmetic pooling changed macro-F1 by `-0.0083246`.
Its two-sided 95% scenario-bootstrap interval was
`[-0.0181561, +0.0055641]`; the one-sided lower bound was `-0.0168146`, failing the
locked `>-0.005` noninferiority criterion. NLL improved by `-0.0454120`, with
one-sided 95% upper bound `-0.0129647`.

The cached-feature temporal head was materially faster in the same-process AB/BA
benchmark:

| Measure | Arithmetic mean | Repeated transformer | Result |
| --- | ---: | ---: | ---: |
| Batch p50 | `1.61475 ms` | `4.57325 ms` | `2.83×` p50 latency speedup |
| Batch p95 | `2.790495 ms` | `7.357235 ms` | `62.07%` lower |
| Median throughput | `39,634.6 rows/s` | `13,994.4 rows/s` | `2.83×` |

The paired p95 delta was `-4.56674 ms` with one-sided 95% upper
`-3.48340 ms`; the p95 ratio was `0.379286` with one-sided upper `0.446120`.
These are cached-head measurements only. Decode, crop, feature extraction, cold start,
host memory, energy, and end-to-end latency were not measured, so no full-pipeline
preference is authorized.

The representative fold-0/seed-42 bounded workloads measured 3.772485 seconds for T1
and 6.220458 seconds for the repeated T2 transformer. Peak CUDA allocated/reserved
memory was 81,169,408/92,274,688 bytes for T1 and
106,520,576/117,440,512 bytes for T2. The scope is model construction, fixed-epoch
fit, held inference, and diagnostic latency from cached features. Sequential initial
allocations differ, so these are representative peaks rather than full-pipeline cost.

T1 also failed promotion guardrails: sitting F1 changed `-0.015225`; occluded-window
F1 changed `-0.011729` with one-sided lower `-0.042155`; and seven supported scenarios
fell below the `-0.010` limit. The worst were scenario `2.2` (`-0.029123`), `1.10`
(`-0.027868`), and `2.5` (`-0.023976`).

Decision: **do not replace the teacher with arithmetic pooling.** Retain it only as a
probability-quality/confidence sidecar—after calibration is directly tested—or as a
compute-saving fallback whose effect is bounded by an anchored gate.

The newly matched repeated refit reproduced retained-teacher aggregate F1 closely:
delta `-0.0006678`, two-sided 95% interval
`[-0.0037231, +0.0023266]`. Its occlusion one-sided lower bound was nevertheless
`-0.0133283`, and scenario `2.11` changed `-0.0102746`; both failed guardrails. This
supports aggregate reproducibility while exposing scenario-level variance.

### T2 decision

Distinct sampling versus its newly matched repeated-sampling control changed
macro-F1 by `+0.0034223`, accuracy by `+0.0036166`, NLL by `-0.0064595`, and Brier by
`-0.0043972`. The NLL one-sided 95% upper bound was `-0.0005790`, so that declared
proper-loss check passed. All three class-F1 point estimates improved. However:

- The macro-F1 two-sided 95% interval was
  `[-0.0005517, +0.0077247]`, so its locked positive-lower-bound criterion failed.
- The exact whole-scenario swap p-value was `0.083984375`, above `0.05`.
- Eight of 11 scenarios improved, but scenario `1.11` regressed `-0.015378` and
  violated the scenario guardrail.
- Occluded-window F1 changed `-0.001785`, with one-sided lower `-0.016547`, violating
  the occlusion bound.
- Clear-window F1 changed `+0.003649`; the transition point estimate was encouraging
  at `+0.014891`, but its two-sided interval `[-0.008399, +0.037679]` was imprecise.

Decision: **do not promote the distinct sampler by itself.** It is a credible
coverage primitive for the next factorial experiment, especially with explicit time
offsets, but it did not solve robustness under occlusion.

Across the 11 scenarios, exploratory correlations between F1 gain and proper-loss
gain were approximately `r=+0.567` for T1 and `r=-0.246` for T2. These small-n
correlations are descriptive only. They show that T2's discrimination and
probability-quality changes do not consistently occur in the same scenarios.

## V0 — failed-closed static attribution

R0 created the shared, development-only V-COCO design for 6,640 people in 4,123 image
groups. No protected row was read. Its artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| Fold map | `80a8dffcc6a36971843d8a3321e8393a889e5a05a1915dadcc74147d3b648036` |
| Bootstrap group indices | `4539eb3323c7dd5cf9c23e399142e08b165370fe56723954d6e95f84624aaa3f` |
| Packed swap signs | `5bb1922682957ff4e1aa9f12d9148dc1fef02d19da2af443d20f72b47b4a7f83` |
| Randomization receipt | `7dc95cb4138e05c004c6d347540e4df0f7c71955e07f4a3e39fe1296bea7e6b0` |

The required bounded feasibility command was:

```powershell
.\.venv\Scripts\python.exe experiments\run_vcoco_continuation_v0.py `
  --mode run --max-new-family-folds 1 --benchmark-family mixed_linear_svm
```

It failed at mixed-linear-SVM outer fold 0, candidate index 0
`mixed_linear_svm__c-0p001__cw-none`, inner fold 0, calibration-stack fold 0,
seed `20270907`:

```text
RuntimeError: CUDA SVM reached its iteration limit; V0 cannot continue:
calibration-fold-0
```

The model reached the locked `n_iter=2000`. The check occurs before predictions; the
outer workload never returned, so `.runs/research_20260907/vcoco_v0` was not created
and no bounded receipt, family checkpoint, candidate selection, OOF probability,
metric, or optimization JSON exists. The `vcoco_v0_r0` fold/randomization artifacts
predate fitting and remain valid.

The protocol requires all four families, eight candidates per family, and convergence
before the ceiling for every calibration and final SVM fit. Therefore the primary V0
status is **FAILED_CLOSED**, and the full command was correctly not run. Running only
the three non-SVM families would be an authorized bounded diagnostic but could neither
complete nor rescue V0; doing so after this failure would add cost without answering
the declared matched question. Skipping the candidate, accepting its ceiling, changing
the tolerance/iteration budget, or dropping the SVM would be retrospective rule
changes and was not done.

## What improved, what did not, and why

### Useful signals

1. **Distinct temporal coverage:** consistent positive aggregate/class point estimates
   and gains in eight scenarios justify retaining unique-frame sampling as a factor.
2. **Arithmetic probability-quality evidence:** much better NLL and a 2.64× p95
   cached-head speed ratio justify testing order-invariant pooling as a confidence or
   abstention feature, or using it in a bounded cheap path. Calibration itself remains
   a hypothesis to validate.
3. **Anchor preservation:** the failed R1 replay reinforces the need to evaluate new
   residual paths against a retained, exact statistical anchor rather than regenerate
   a hardware-sensitive teacher whenever possible.
4. **Fail-closed infrastructure:** both the unsafe object serialization and
   non-converged SVM were stopped before they could become apparently valid evidence.

### Components that did not earn promotion

1. **Arithmetic pooling as the classifier:** too much aggregate, sitting, occlusion,
   and worst-scenario discrimination was lost.
2. **Distinct sampling alone:** the average gain was too small and unstable across
   scenarios, with no occlusion solution.
3. **Legacy CPTR coefficient surgery on this host:** its required numerical baseline
   was not reproduced; no mechanistic claim is available.
4. **The present CUDA primal SVM configuration:** it cannot satisfy its own convergence
   contract on the first nested calibration fit.
5. **Current V1 pose/spatial proposal:** it is not runnable from pooled features, a
   ground-truth pose oracle is ineligible, and clean `K=4` predicted poses, source
   images, pinned patch grids/weights, and a training-overlap audit are absent.

## Next architecture and breakthrough path

The evidence supports a narrow invention candidate rather than a larger undirected
model search: a **baseline-anchored visibility-aware dual-path temporal residual
(BAVTR)**.

For anchor logits `z0`, learned temporal residual `r`, visibility/repetition mask `m`,
hard availability `a`, and confidence feature `c` from the arithmetic path, to be
calibrated within training folds:

```text
z = z0 + a(m) * clip(g(c, m), 0, g_max) * r(x_unique, delta_t, m)
```

Required invariants:

- The anchor coefficient is exactly one.
- Residual output weights and biases initialize to zero.
- `a(m)` is exactly zero for missing, fully occluded, or wholly duplicated evidence,
  so those cases return `z0` exactly.
- Frames use real relative offsets `delta_t`, not ordinal slot IDs that conceal
  irregular gaps or duplicate indices.
- Visibility and repetition are explicit inputs, not inferred from feature magnitude.
- The arithmetic path supplies a confidence feature or bounded gate; any calibration
  claim requires an explicit within-fold calibration evaluation. It does not replace
  the discriminative path.

### Prospectively locked next experiments

1. **T3 representation factorial:** repeated versus distinct sampling crossed with
   ordinal versus true-relative-time encoding, on identical folds/seeds/budgets.
   This separates the sampling gain from correct temporal representation.
2. **T4 visibility residual:** add explicit visibility/repetition masks and compare
   anchor, unmasked residual, visibility-aware residual, and arithmetic-confidence
   gating. Require exact anchor return under missing evidence and nonzero residual
   gradients.
3. **T5 robustness:** retain aggregate noninferiority, class, occlusion, transition,
   and worst-scenario guardrails. Freeze the BAVTR configuration after this exploratory
   derivation, then require new independently held scenarios for any confirmatory
   architecture claim.
4. **V0.1 solver diagnostic:** before another matched static study, prospectively
   instrument convergence on authorized development folds and choose a deterministic,
   convergence-safe reference solver without using held-label performance. Bind the
   solver and settings under a new commit and execution lock. The failed V0 stays
   immutable. Then rerun all four matched families, rather than only favorable arms.
5. **V1 input acquisition:** acquire rights-safe source images, the pinned DINOv2
   weights, 16×16 patch grids, clean `K=4` predicted-pose hypotheses, and an auxiliary
   corpus overlap audit. Only then create a new V1 source lock.
6. **X replication and cost:** freeze exactly one survivor, evaluate once on a fresh
   untouched external domain, and measure decode-to-decision latency, throughput,
   peak host/device memory, and energy. This is the gate for a broader solution claim.

Promotion of BAVTR should require either a predeclared noninferiority lower bound above
its negative margin or a superiority lower bound above zero, no NLL worsening, intact
class/occlusion/worst-scenario guardrails, exact missing-evidence anchor identity, and
end-to-end—not cached-head only—resource evidence.

## Final gate ledger

| Stage | Outcome | Consequence |
| --- | --- | --- |
| R0 Okutama role-safe bundle | PASS | R1 and independent T1/T2 authorized |
| R1a numerical reproduction | FAIL | R1b and R2 closed; no CPTR mechanism result |
| T1 arithmetic mean | FAIL promotion | Keep only as probability-quality/efficiency sidecar |
| T2 distinct sampling | FAIL promotion | Retain as prospective factorial component |
| V0 bounded SVM feasibility | FAIL CLOSED | No four-family static attribution result |
| V1 spatial/pose inputs | BLOCKED | No V1 fitting authorized |
| X independent replication | BLOCKED | No breakthrough/generalization claim |

This cycle is fully executed under its stop rules. Its genuine contribution is not a
winning score; it is a sharply reduced design space and a falsifiable next architecture
that combines the two useful primitives while preserving the strongest baseline.
