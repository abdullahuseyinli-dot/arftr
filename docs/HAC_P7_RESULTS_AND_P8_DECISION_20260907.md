# HAC P7 results and P8 decision

Date: 2026-09-07  
Development population: 4,977 Okutama rows, 11 scenarios, five fixed outer folds  
P7 status: **complete negative result; retain P6**

## Execution integrity

The Center-Aware Phase-Preserving Expert (CAPE) protocol was committed before
feature derivation or fitting. The execution used commit `03a410a`, pre-fit lock
SHA-256 `e88017a59d5d748d168ad72f2f84c617d3456fdae70957b9fcf47a417d64a89f`,
30 resumable outer-fold workloads, and exactly 715 audited estimator fits.

The A0 control reproduced the retained P5 spatial OOF probability array exactly.
The completed output was replay-validated without refitting. Its summary SHA-256 is
`16d6a106bfd19923f6eb5a9e1654fbc1e26c80903036f27611b665f1e7da2200`; its OOF
archive SHA-256 is
`0e5abcd02438ce62ddabcc47c329a1b42ad36b8881deaa905caaba94caba6ef7`.
P3/P5/P6 reference probabilities were decoded only after all 30 P7 workloads and
the 715-fit audit completed. The legacy baseline was loaded read-only with the
frozen feature population but was not accessed by fitting.

All repository tests and Ruff checks passed before locking. Recorded fitting time
across the 30 receipts was 974.63 seconds (16.24 minutes), excluding frozen-feature
loading and final statistical aggregation.

## Primary results

Percentages below are absolute percentages; deltas are percentage points.

| Output | Macro-F1 | Accuracy | NLL | Brier | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 spatial refit | 81.7665% | 82.6803% | 0.449524 | 0.257257 | reference expert |
| A1 center posture | 81.6869% | 82.6201% | 0.448488 | 0.257606 | -0.0796 vs A0 |
| A2 center + signed changes, primary expert | 81.4329% | 82.2986% | 0.452941 | 0.260155 | -0.3336 vs A0 |
| A3 off-center signed control | 81.6550% | 82.4593% | 0.461574 | 0.263521 | -0.1115 vs A0 |
| A4 center + unsigned control | 81.5815% | 82.4794% | 0.449390 | 0.257967 | -0.1850 vs A0 |
| A5 information-matched direct head | 81.0799% | 82.5598% | 0.463419 | 0.260923 | -0.6866 vs A0 |
| P6 retained OCVC | **82.5830%** | **83.5041%** | **0.416570** | **0.240902** | incumbent |
| Fixed P7 replacement triad | 82.5350% | 83.3836% | 0.418884 | 0.242334 | **-0.0480 vs P6** |

The primary system difference was -0.0480 points, with scenario-bootstrap 95%
interval [-0.3456, +0.3476], one-sided whole-scenario swap p=0.603516, and Holm
p=1.0. The replacement worsened NLL by 0.002314 and Brier by 0.001433.

## Six declared comparisons

| Candidate minus reference | F1 delta | 95% interval | Holm p |
| --- | ---: | ---: | ---: |
| Replacement triad minus P6 | -0.0480 | [-0.3456, +0.3476] | 1.0000 |
| A1 center posture minus A0 | -0.0796 | [-0.5594, +0.3016] | 1.0000 |
| A2 signed minus A1 posture | -0.2540 | [-0.6038, +0.0516] | 1.0000 |
| A2 center minus A3 off-center | -0.2221 | [-1.0779, +0.4508] | 1.0000 |
| A2 signed minus A4 unsigned | -0.1486 | [-0.4369, +0.1282] | 1.0000 |
| A2 factorized minus A5 direct | +0.3530 | [-0.2234, +1.1544] | 0.6973 |

Every interval includes zero. The result does not support the center, signed-phase,
or fixed-replacement hypotheses. A2's numerical advantage over A5 is compatible
with the earlier factorization evidence but is not statistically resolved and does
not compensate for A2's loss to A0.

## Guardrails and scenario behavior

Only two of seven engineering guardrails passed.

| Guardrail | Result | Evidence |
| --- | --- | --- |
| System macro-F1 at least 83.5% | Fail | 82.5350% |
| NLL strictly below P6 | Fail | 0.418884 versus 0.416570 |
| Brier strictly below P6 | Fail | 0.242334 versus 0.240902 |
| At least 7/11 scenarios improve | Fail | 5 improve, 5 worsen, 1 ties |
| No scenario decline over 2 points | Pass | worst decline was 0.5132 points |
| Positive overall net correction | Fail | 44 rescues, 50 harms, net -6 |
| Positive clear/stable net correction | Pass | 33 rescues, 32 harms, net +1 |

| Scenario | Replacement minus P6 F1 |
| --- | ---: |
| 1.10 | +0.0100 points |
| 1.11 | 0.0000 points |
| 1.2 | -0.5132 points |
| 1.3 | +0.2846 points |
| 1.4 | +0.0498 points |
| 1.5 | -0.4138 points |
| 2.11 | -0.0205 points |
| 2.2 | -0.2790 points |
| 2.5 | +1.5567 points |
| 2.7 | +0.5487 points |
| 2.8 | -0.3356 points |

The class-F1 changes were +0.1969 points for sitting, -0.2138 for standing, and
-0.1272 for walking/running. Thus the system did not solve the dominant
standing--locomotion boundary.

## What the error correlations say

The P7 triad rescued 44 P6 errors and introduced 50. Of the 508 rows where all
three P6 members were wrong, the replacement rescued only one. Its other 43
rescues came from the 313 P6 errors where an existing component was already correct.
Its error correlation with P6 was 0.93165.

The standalone A2 expert contained more diverse evidence: it rescued 139 P6 errors,
including 31 shared failures, but introduced 199 errors, for a net loss of 60.
Across all six P7 experts, a descriptive label oracle found a correct expert on only
66 of the 508 shared failures; 442 were wrong for every P7 expert. This oracle is
not an implementable score and must not be presented as a routing result.

A post-result diagnostic found that the fixed P6-confidence-below-0.5 slice contains
169 rows where A2 gives 39 rescues and 28 harms. The fold net changes are
`+7,+2,+5,-4,+1`. This is only a hypothesis for a future nested router. Selecting or
tuning a gate on these exposed OOF outcomes would be leakage; any gate would require
fresh inner-OOF component generation within every outer training population.

## What worked, what failed, and what to stop

What worked:

- The execution and lineage controls worked: exact A0 reproduction, fold isolation,
  deterministic single-threaded probes, immutable workload receipts, and immutable
  publication receipt all passed.
- P6 remains the incumbent at 82.5830% macro-F1 and had better NLL/Brier than the
  attempted replacements; no separate calibration assessment was performed.
- Factorized head allocation remained numerically better than the direct union in
  this comparison, though its interval includes no effect.
- A2's 31 shared-failure rescues show that its representation is not wholly
  redundant, but its precision is inadequate for use as an expert.

What failed:

- Preserving the frozen labeled-center token did not improve A0.
- Signed local DINO feature derivatives made the center-posture expert worse.
- The center arm did not beat either the off-center or unsigned matched control.
- The 16,128-dimensional direct head was slower and worst on macro-F1.
- Equal-weight replacement did not retain A2's rare shared-failure repairs.

One measured nuisance deserves a bounded control, not a rescue narrative. The mean
center-change L2 norm was 13,958.48 for short-fallback rows versus 1,082.80 for
long-valid rows, a 12.89-fold shift consistent with their different physical time
steps but not isolated from the accompanying input/cohort differences.
A2 lost 1.3128 points versus A1 on fallback rows, but also lost 0.1562 points on
long-valid rows. Therefore stride-aware normalization cannot explain away the
overall negative result.

Stop further C sweeps, center-index sweeps, derivative variants, ensemble-weight
screens, and threshold tuning on this exposed development OOF set. All 25
factorized workloads selected motion C=0.001; signed changes lowered selected inner
macro-F1 relative to A1 in every outer fold, and no convergence limit was approached.

## P8 decision: camera-compensated actor correspondence

The next high-value experiment must change the observation source. P4 measured box
trajectories; P7 differentiated frozen representation coordinates. Neither measured
motion of visible foreground points relative to the camera and to the actor's own
translation. The P8 candidate is therefore a **Camera-Compensated Actor
Correspondence (CCAC) expert**. This is a working name, not an originality claim.

### P8-0: input and independent-evaluation audit

1. Revalidate frame/video identities, annotation-center timing, crop-to-scene
   transforms, box coordinates, and scenario grouping.
2. Inventory external aerial/surveillance video separately as auxiliary training or
   untouched evaluation data. Verify licenses, duplicates, ontology, and actor scale
   before downloading or using labels.
3. Freeze a distinct evaluation population before using its outcomes. The 4,977-row
   Okutama population is now an adaptive development set.

### P8-1: 128-clip label-blind feasibility gate

1. Select about 128 eligible clips by fixed sample-ID hash, balanced by scenario,
   native actor-size bands, and declared visibility metadata. Do not use class labels,
   P6/P7 errors, confidence, or correctness in selection.
2. Track foreground points inside the actor region and background points outside a
   dilated actor region at native timestamps. Preserve the explicit crop-to-scene
   transform.
3. Measure forward/backward consistency, track survival, actor pixel height,
   background camera-model inliers and residuals, and compensated background
   residuals. Compare raw, ordinary single-camera compensation, and multiple-camera
   hypotheses on the identical sample.
4. Publish quality distributions before fitting a classifier. Set the full-extraction
   continuation threshold from label-blind measurement quality. Stop if native actor
   detail cannot support stable correspondences; upsampling is not evidence recovery.

### P8-2: fixed CCAC representation

Estimate camera motion robustly from background correspondences while excluding
actor boxes. Transform all tracks to scene coordinates, subtract camera-predicted
motion from foreground displacement, and decompose the residual into:

- coherent actor translation, normalized by elapsed time and actor height;
- within-actor residual motion after subtracting coherent translation;
- spatial distribution and temporal persistence of residual motion; and
- measurement reliability, retained as an explicit non-label feature.

These are foreground correspondence statistics, not claimed anatomical joints.
The existing visual posture head remains unchanged; CCAC features augment only the
conditional standing-versus-locomotion head.

### P8-3: bounded supervised trial

Lock five factorized arms before extraction/model outcomes: A0 spatial refit,
reliability-only control, raw foreground motion, camera-compensated translation,
and compensated translation plus within-actor residual motion as the primary arm.
Use the same four C values, five outer scenario folds, three inner grouped folds,
and at most 650 fits. Lock one equal-weight system that replaces P6's third member
with the primary CCAC expert.

Use no more than six declared directional comparisons: system versus P6; primary
versus A0; primary versus compensated translation; compensated translation versus
raw motion; primary versus reliability-only; and raw motion versus A0. Retain P7's
calibration, scenario, net-correction, and shared-failure guardrails. Do not proceed
to learned fusion unless the new expert first shows useful shared-failure repair.

Reaching 84% from P6 corresponds to about 66.5 proportional no-harm error repairs;
85% corresponds to about 113.5. These are confusion-matrix thought experiments, not
forecasts or minimum-error bounds. P7's entire six-expert oracle covered only 66
shared failures, which makes a genuinely new physical observation the preferred
next hypothesis; it does not prove that every other improvement route is exhausted.

## Evidence

- [P7 summary](../.runs/research_20260907/okutama_native_video_p7/results/summary.json)
- [P7 metrics](../.runs/research_20260907/okutama_native_video_p7/results/metrics.csv)
- [P7 paired statistics](../.runs/research_20260907/okutama_native_video_p7/results/paired_statistics.json)
- [P7 diagnostics](../.runs/research_20260907/okutama_native_video_p7/results/diagnostics.json)
- [P7 OOF probabilities](../.runs/research_20260907/okutama_native_video_p7/results/oof_probabilities.npz)
- [P7 protocol](../experiments/okutama_video_p7_protocol.json)
- [Pre-P7 research map](HAC_RESEARCH_MAP_AND_NEXT_PHASE_20260907.md)

All P7 performance evidence is adaptive development evidence. It does not establish
independent generalization or architectural originality.
