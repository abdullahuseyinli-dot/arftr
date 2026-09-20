# Bounded Factor-Correction Trial — Final Result

Date: 2026-09-12

## Decision

Do not promote the bounded factor-correction candidate. The frozen ARFTR
reference remains the retained system at **85.3836% macro-F1**. The primary
candidate reached **84.9637%**, a change of **-0.4199 points**.

This was a complete 60-fit adaptive screen: four trained arms, five held-out
folds, and three seeds. All held-out predictions, metrics, resampling
statistics, and gate decisions passed an independent checkpoint replay audit.

## Arm comparison

| Arm | Macro-F1 | Delta vs ARFTR | Accuracy | NLL | Brier | Rescues | Harms | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact ARFTR | 85.3836% | — | 85.8951% | 0.37735 | 0.21350 | — | — | — |
| Bounded separate, center-only | 84.5988% | -0.7848 pt | 85.3325% | 0.37531 | 0.21370 | 71 | 99 | -28 |
| Bounded separate, all-frame (primary) | 84.9637% | -0.4199 pt | 85.6741% | 0.37337 | 0.21216 | 77 | 88 | -11 |
| Bounded shared, all-frame | 85.1024% | -0.2813 pt | 85.6138% | 0.37581 | 0.21297 | 78 | 92 | -14 |
| Unbounded separate, all-frame | 82.0767% | -3.3070 pt | 83.0621% | 0.45012 | 0.25097 | 158 | 299 | -141 |

## Mechanism conclusions

- All-frame supervision was useful: it added **+0.3649 points** over the
  center-only control, although the paired scenario test did not establish the
  gain (`p=0.0830`).
- The explicit bound was essential: bounded correction beat the matched
  unbounded arm by **+2.8871 points**. The unbounded arm reduced training loss
  aggressively while producing far more harms and much worse NLL/Brier.
- Separate presence/motion outputs did not help with the available static
  center feature: the shared-output control was **+0.1386 points** better than
  the separate-output primary.
- The primary improved NLL and Brier point estimates over ARFTR, but their
  scenario-resampled one-sided upper bounds crossed zero. The calibration gain
  therefore is not reliable evidence of improvement.
- The primary was stable across seeds (macro-F1 SD **0.3066 points**) but only
  improved 4 of 11 scenarios. Stability did not compensate for negative mean
  utility.

## Where the correction helped and harmed

| Support stratum | Rows | Primary delta vs ARFTR | Rescues | Harms | Net |
|---|---:|---:|---:|---:|---:|
| Known pure | 4,184 | -0.5025 pt | 46 | 59 | -13 |
| Known mixed | 682 | +0.4975 pt | 29 | 23 | +6 |
| Unknown | 111 | -3.4762 pt | 2 | 6 | -4 |

The useful signal is localized to ambiguous mixed-support examples. Applying
the residual globally damages the large pure-support population and is unsafe
on unknown support. The primary also lost **1.1471 class-F1 points** on class 0,
which dominates its aggregate deficit; class 1 changed by -0.1702 points and
class 2 by +0.0576 points.

## Frozen-gate outcome

The primary passed only the gain-over-center-control, mixed-support net,
seed-stability, and exact-ARFTR-replay checks. It failed the absolute score,
gain-over-ARFTR, net-correction, pure-support, paired-scenario, proper-loss,
worst-class, and scenario-coverage requirements. The overall promotion gate is
therefore false.

## Evidence-directed next trial

The next architecture should not be a larger global residual. It should be a
**selective, harm-aware temporal correction**:

1. Keep ARFTR exactly unchanged by default.
2. Add real short temporal inputs (ordered neighboring frame embeddings or
   motion tokens), because static center features did not support meaningful
   factor separation.
3. Learn correction and a separate apply/abstain utility gate using inner-OOF
   rescue-versus-harm targets only; outer labels remain embargoed.
4. Force the gate to zero on pure/high-confidence and unknown-support regions,
   and permit bounded correction primarily in mixed-support ambiguity.
5. Include oracle-coverage and selective-risk curves before training. If the
   mixed-support opportunity cannot support at least the frozen +0.45-point
   promotion margin, stop before a large run.
6. Run the same held-out folds, seeds, exact ARFTR reference, transition
   accounting, scenario resampling, and proper-loss gates.

This trial rules out unrestricted static factor correction as the breakthrough
path. It supports a narrower hypothesis: temporal evidence is useful only when
the system can identify where intervention has positive expected utility.

## Artifacts

- Protocol: `experiments/okutama_bounded_factor_correction_protocol.json`
- Runner: `experiments/run_okutama_bounded_factor_correction.py`
- Result: `.runs/research_20260912/bounded_factor_correction_v1/results/v0001/summary.json`
- OOF predictions: `.runs/research_20260912/bounded_factor_correction_v1/results/v0001/oof_probabilities.npz`
- Independent audit: `.runs/research_20260912/bounded_factor_correction_v1_audit_v3/summary.json`
- Numerical replay diagnostics: `.runs/research_20260912/bounded_factor_correction_v1_audit_v3/numerical_replay.json`

The audit observed maximum CPU/CUDA probability drift of
`2.891667242321816e-07` across 60 replays and zero argmax changes. The result is
adaptive repeated-development evidence and still requires external-dataset or
fresh held-out confirmation before any generalization claim.
