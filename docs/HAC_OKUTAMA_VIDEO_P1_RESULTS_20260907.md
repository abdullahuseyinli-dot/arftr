# Okutama native-video P1 result report

Date: 2026-09-07

## Decision

Advance the frozen real-clip V-JEPA representation with a strongly regularized
linear probe. It reaches 0.7715389093 pooled development OOF macro-F1, a gain of
5.2301 percentage points over the frozen historical reference. It does not reach
the 0.82 development milestone and it fails the proposed occlusion and per-scenario
promotion guardrails. The current attentive probe, repeated-center candidate, and
DINO-only candidate do not advance.

This is an exploratory result on reused development scenarios, not independent
confirmation. The V-JEPA and DINO candidates use one seed; the historical reference
is a five-seed ensemble.

## Locked cohort and provenance

- Rows: 4,977 original centers; no exclusions or fallback predictions.
- Scenario groups: 11, assigned to the unchanged five outer folds.
- Classes: sitting, standing, walking/running.
- Native input: 16 tracked actor crops spanning 0.5 seconds endpoint-to-endpoint.
- V-JEPA cache: two arrays of shape `4977 x 8 x 9 x 768`, FP16, all rows valid.
- DINOv2 cache: one array of shape `4977 x 16 x 1 x 768`, FP16, all rows valid.
- Feature extraction read no candidate labels and performed no fitting.
- Probe execution read no protected manifest or target image.
- P1 status: `OKUTAMA_VIDEO_P1_EXPLORATORY_CROSSFIT_COMPLETE`.

The full V-JEPA extraction took 914.3808 seconds at 5.4430 clips/s and peaked at
737,402,880 allocated CUDA bytes. DINOv2 took 642.7280 seconds at 7.7436 clips/s
and peaked at 489,344,000 allocated CUDA bytes.

## Primary results

| Representation | Probe | Macro-F1 | Change vs baseline | Accuracy | NLL | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V-JEPA real clip | Linear mean | **0.771539** | **+0.052301** | **0.783605** | **0.532933** | **0.305134** |
| V-JEPA real clip | Attentive | 0.761780 | +0.042543 | 0.770545 | 0.869403 | 0.378626 |
| Historical distinct-frame ensemble | Frozen reference | 0.719238 | -- | 0.716295 | 0.748820 | 0.405032 |
| DINOv2 native frames | Linear mean | 0.715964 | -0.003274 | 0.711674 | 0.681415 | 0.401366 |
| DINOv2 native frames | Attentive | 0.698276 | -0.020962 | 0.700020 | 1.720354 | 0.541715 |
| V-JEPA repeated center | Linear mean | 0.636522 | -0.082716 | 0.632309 | 0.869896 | 0.501461 |
| V-JEPA repeated center | Attentive | 0.615536 | -0.103702 | 0.613020 | 1.371938 | 0.623839 |

All 30 declared arm/probe/fold workloads completed. The winning linear probe averages
the frozen spatiotemporal tokens to 768 dimensions; it does not flatten the full grid.

## Paired evidence

| Directional comparison | Macro-F1 difference | Scenario-bootstrap 95% CI | Holm p |
| --- | ---: | ---: | ---: |
| Real V-JEPA linear minus baseline | +0.052301 | [+0.024645, +0.082729] | 0.0078125 |
| Real V-JEPA linear minus DINO linear | +0.055575 | [+0.019930, +0.086294] | 0.0375977 |
| Real V-JEPA linear minus repeated V-JEPA linear | +0.135017 | [+0.103262, +0.169095] | 0.0048828 |
| Real V-JEPA attentive minus repeated V-JEPA attentive | +0.146244 | [+0.105760, +0.198900] | 0.0048828 |

The winning candidate rescues 767 historical errors and introduces 432 new errors,
for 335 additional correct predictions. Its binary error indicator has correlation
0.3674 with the historical reference, compared with 0.6188 for DINO linear. This
supports a properly nested feature-fusion test but does not prove that a deployable
router can identify the complementary cases.

## Error mechanism

| Class | Baseline F1 | Real-video linear F1 | Difference | Rescued / harmed |
| --- | ---: | ---: | ---: | ---: |
| Sitting | 0.737726 | 0.728218 | -0.009508 | 57 / 68 |
| Standing | 0.678672 | 0.761053 | +0.082381 | 440 / 184 |
| Walking/running | 0.741314 | 0.825345 | +0.084030 | 270 / 180 |

Standing-to-locomotion and locomotion-to-standing errors fall from 1,006 to 659,
a 34.49% reduction. Sitting/standing errors rise, so the representation primarily
solves part of the motion boundary rather than improving every class uniformly.

| Diagnostic stratum | Rows | Baseline F1 | Real-video linear F1 | Difference |
| --- | ---: | ---: | ---: | ---: |
| Clear | 4,565 | 0.728266 | 0.791599 | +0.063333 |
| Occluded | 412 | 0.626653 | 0.586391 | **-0.040262** |
| Stable | 4,579 | 0.733609 | 0.786907 | +0.053298 |
| Transition | 398 | 0.525829 | 0.574688 | +0.048859 |

Nine of 11 scenarios improve. Scenarios 1.10 and 2.11 decline by 0.6532 and
1.4082 percentage points, respectively. DINO linear beats the video winner by 10.394 points in
scenario 2.8, which is direct evidence that appearance remains locally useful.

## What is retained and retired

Retain:

- Real native video and the frozen V-JEPA representation.
- Strong regularization and grouped nested selection.
- A simple mean-feature head as the matched reference.
- Appearance features as a possible posture/reliability expert.

Retire for the next screen:

- The current 15-epoch attentive head: it loses F1 for all three representations and
  is severely overconfident out of fold.
- Repeated-center V-JEPA as a production candidate; retain it only as a control and
  counterfactual feature source.
- DINO-only head tuning as the main direction.

Fourteen of 15 linear fits selected the smallest tested value, `C=0.01`. The next
locked screen therefore extends the grid downward instead of increasing capacity.

## Gate assessment and next work

| Gate | Result |
| --- | --- |
| At least +2 pp over a matched representation control | Pass |
| Development macro-F1 at least 0.82 | Fail; 4.8461 pp remain |
| No pooled NLL degradation | Pass |
| No class F1 decline worse than 0.010 | Pass narrowly; sitting -0.009508 |
| No scenario decline worse than 0.010 | Fail; scenario 2.11 |
| Occlusion lower-bound noninferiority | Fail |
| Transition lower-bound noninferiority | Fail |

The next sequence is:

1. A newly locked cached-feature screen of lower regularization, direct V-JEPA+DINO
   fusion, counterfactual motion-residual features, and a low-capacity factorized
   posture/motion specialist. Historical probabilities remain comparison-only.
2. A 2-second, 16-frame native-video arm with the same encoder token budget and a
   matched long-window DINO control.
3. Explicit foreground/background motion with compensated-flow, single-camera, and
   camera-hypothesis controls if the cached and duration screens leave complementary
   opportunity.
4. A matched five-seed run before any architecture-promotion claim.

Primary retained artifacts are under
`.runs/research_20260907/okutama_native_video_p1/results/`.
