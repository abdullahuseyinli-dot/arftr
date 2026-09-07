# HAC P8 CCAC full results and P9 decision

Date: 2026-09-07

Status: **P8 complete and replay-verified; retain P6**

## Executive decision

The full Camera-Compensated Actor Correspondence (CCAC) program was executed as
declared: 4,977 label-blind extraction workloads, a fixed five-arm supervised
ablation, 25 outer-fold workloads, and 470 estimator-fit invocations. All retained
artifacts replay and validate.

P8 did **not** produce the intended architectural breakthrough. Its fixed
replacement system reached 82.6269% macro-F1 versus P6's 82.5830%, a nominal gain
of 0.0439 percentage points, but the 95% scenario-bootstrap interval is
[-0.1695, +0.3133] points, every Holm-adjusted p-value is 1.0, accuracy is lower,
NLL and Brier are worse, and the system repairs none of P6's 508 shared failures.
Only one of seven engineering guardrails passes. P6 therefore remains the champion.

The useful finding is narrower: camera compensation improves the probability scores
of the raw-motion arm consistently, although it does not establish an argmax or
macro-F1 improvement. Keep the measurement machinery; retire the present six-number
within-actor residual summary and do not run another post-hoc CCAC weight sweep.

## Execution and integrity

| Audit | Result |
| --- | ---: |
| Full extraction clips | 4,977/4,977 |
| Requested / available adjacent pairs | 74,655 / 73,737 |
| New pixel workloads / exact pilot references | 4,849 / 128 |
| Extraction shards | 8/8 complete, zero stderr |
| Nonpilot exact replays | 2/2 exact for pair metrics, clip metrics, and correspondence bytes |
| Action labels / prior predictions / classifier fits during extraction | 0 / 0 / 0 |
| Supervised workloads | 25/25 |
| Estimator-fit invocations | 470/470 |
| Exact A0 reproduction of P5 spatial OOF | Pass |
| Exact reconstruction of P6 | Pass |
| Add-on posture checkpoints and held sitting probabilities equal A0 | 20/20 pass |
| Retained P8 rerun | Pass with zero refits |

Full extraction required about 45 minutes wall-clock across eight single-threaded
OpenCV shards. The 25 supervised workloads used 308.95 summed seconds; median
workload time was 11.87 seconds and the maximum was 18.94 seconds.

## Full-cohort measurement result

| Measurement | Full cohort | Pilot | Assessment |
| --- | ---: | ---: | --- |
| Camera-usable requested pairs | 67,945/74,655 = **91.012%** | 91.146% | Pilot generalized closely |
| Translation-usable requested pairs | 63,773/74,655 = **85.424%** | 86.354% | Strong pair coverage |
| Articulation-usable requested pairs | 61,664/74,655 = **82.599%** | 82.917% | Strong pair coverage |
| Translation-valid feature rows (at least 10 pairs) | 4,216/4,977 = **84.710%** | 87.500% clips | Adequate |
| Articulation-valid feature rows (at least 10 pairs) | 4,063/4,977 = **81.636%** | 83.594% clips | Adequate |
| Camera models H0 / H1 / H2 | 2,584 / 22,223 / 48,930 | 91 / 725 / 1,079 | Homography dominates |

The principal full-cohort failures are 5,792 camera-unusable pairs and 6,181 pairs
with insufficient actor correspondences; reasons overlap. Scenario-level
articulation-valid coverage ranges from 68.37% in 2.5 to 97.60% in 2.8. Coverage by
native actor-size band is 85.60% small, 80.06% medium, and 71.79% large. The pilot's
opposite-looking aggregate size ordering was therefore not population-representative;
size and scenario composition remain confounded.

Median retained actor support is 15 points. Median camera audit error is 0.286 native
pixels, median camera-motion reduction is 94.88%, median compensated translation is
0.183 actor-heights/second, and median within-actor residual is 0.0757
actor-heights/second. These establish measurable motion, not semantic articulation.

## Supervised results

| Model | Macro-F1 | Accuracy | NLL lower is better | Brier lower is better | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 spatial refit | 81.7665% | 82.6803% | 0.449524 | 0.257257 | Exact control |
| Q reliability | 81.7338% | 82.6401% | 0.450082 | 0.257700 | No benefit |
| R raw translation | 81.7039% | 82.6000% | 0.450175 | 0.257835 | No benefit |
| C camera-compensated translation | 81.7497% | 82.6602% | 0.448496 | 0.256794 | Small scoring signal |
| F compensated plus residual, primary | 81.7351% | 82.6401% | 0.448473 | 0.256787 | Does not beat A0 or C |
| P6 incumbent | **82.5830%** | **83.5041%** | **0.416570** | **0.240902** | Retain |
| P8 fixed replacement triad | 82.6269% | 83.4438% | 0.418920 | 0.242284 | Nominal F1 only; reject promotion |

All arms and folds selected motion C=0.001. Frozen A0 posture C values were
[0.01, 0.001, 0.01, 0.01, 0.01]. Motion fits converged in 47--58 iterations, far
below the 2,000-iteration ceiling. The negative result is not explained by a failed
optimizer or an accidentally disconnected feature branch.

## Six locked comparisons

Differences and intervals are macro-F1 percentage points.

| Locked contrast | Delta | 95% scenario-bootstrap interval | Directional exact p | Holm p | Rescues / harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| P8 system minus P6 | +0.0439 | [-0.1695, +0.3133] | 0.3555 | 1.000 | 29 / 32 |
| F minus A0 | -0.0314 | [-0.1280, +0.0689] | 0.7734 | 1.000 | 6 / 8 |
| F minus C | -0.0146 | [-0.0711, +0.0344] | 0.6250 | 1.000 | 2 / 3 |
| C minus R | +0.0458 | [-0.0290, +0.1251] | 0.2031 | 1.000 | 9 / 6 |
| F minus Q | +0.0014 | [-0.0822, +0.1097] | 0.4531 | 1.000 | 5 / 5 |
| R minus A0 | -0.0626 | [-0.1735, +0.0353] | 0.9063 | 1.000 | 3 / 7 |

The C-minus-R probability-score deltas are more consistent than its F1 delta:
NLL -0.001679 with a descriptive 95% interval [-0.002406, -0.000968], and Brier
-0.001041 with interval [-0.001510, -0.000468]. NLL improves in 10/11 scenarios and
5/5 folds. These are secondary, non-multiplicity-adjusted findings, so they support
retaining camera compensation as a component, not promoting the model.

## Correlation and mechanism map

```text
1280x720 scene pixels
        |
        +--> background LK --> robust camera model --> useful probability correction
        |                                             (C better scored than R)
        |
        +--> actor LK --> 15 pair rows --> temporal quantiles --> tiny added logits
                                                   |          |
                                                   |          +--> residual block: null
                                                   +--> only 15/4,977 A0 decisions change

P6 visual consensus --> 821 mistakes --> 508 all-components-wrong
                                           |
                                           +--> P8 system rescues 0
                                                (targeted bottleneck remains)
```

| Connected observation | Exact evidence | Interpretation |
| --- | --- | --- |
| F is almost the same classifier as A0 | Error correlation 0.99019; only 15 decisions change | CCAC is a small correction, not an independent expert |
| Residual statistics are redundant/weak | F-versus-C error correlation 0.99650; five decisions change | Six unordered magnitudes do not encode useful articulated structure |
| Camera compensation is doing real work | C beats R in NLL and Brier in every fold directionally | Preserve compensation, but not the current classifier claim |
| Reliability is not the answer | Q is below A0 on F1, accuracy, NLL, and Brier | Missingness/context is not a hidden shortcut that solves the task |
| Headline macro-F1 hides harm | P8 system has 29 rescues and 32 harms; errors rise 821 to 824 | Minority-class balance improves while total correctness falls |
| Shared coverage does not expand | A0 and F each rescue 15/508 P6 shared failures; 14 are identical, one gained, one lost | CCAC contributes zero net shared-error coverage |
| The system gain is marginal but not literally zero | Descriptive A0-triad to F-triad comparison changes three decisions, all rescues, +0.0589 points | Small ensemble perturbation; not a seventh confirmatory test |

The P8 system gains four correct sitting decisions and two standing decisions but
loses nine locomotion decisions relative to P6. Sitting F1 rises 0.3882 points;
standing and locomotion F1 fall 0.0977 and 0.1588 points. Six scenarios improve,
three worsen, and two tie. No P8 system rescue occurs in the 508 shared failures,
the 401 true-upright shared failures, or the 311 shared standing/locomotion-boundary
errors. All 29 rescues occur in the already-routable 313-error partition.

## Engineering gates

| Gate | Result |
| --- | --- |
| Macro-F1 at least 83.5% | Fail |
| NLL strictly below P6 | Fail |
| Brier strictly below P6 | Fail |
| At least 7/11 scenarios improve | Fail: 6 improve |
| No scenario decline beyond 2 points | **Pass** |
| Positive net corrections overall | Fail: -3 |
| Positive net corrections on clear/stable rows | Fail: -1 |

The 84% milestone and 85% stretch target were not reached. They remain targets, not
supported forecasts.

## What to retain and what to stop

Retain:

- the audited full-cohort correspondence artifact and deterministic extractor;
- disjoint camera fit/selection/audit correspondences;
- camera-compensated rather than raw actor translation;
- explicit measurement-validity fields and identity-safe joins; and
- the frozen-posture ablation/evaluation machinery.

Stop or deprioritize:

- the six unordered within-actor magnitude quantiles;
- another C, threshold, ensemble-weight, or learned global-OOF gate sweep;
- claims that sparse residual points are anatomical joints or gait trajectories;
- promotion of the 82.6269% triad; and
- interpreting adaptive same-scenario confidence intervals as independent evidence.

## P9: highest-value next phase

P8 shows that the remaining bottleneck is not merely camera motion. It is missing
actor-local visual detail and learned space-time structure. The next phase should
change those inputs rather than append more scalar statistics.

### P9-0: source-fidelity audit before training

The Okutama-Action paper reports 43 sequences at 30 FPS and 4K resolution, while the
current scene archive is 1280x720. Audit access to the genuine original 4K videos,
their license, recording identities, frame timing, and box scaling before any new
outcomes. The official dataset page historically lists separate 4K video and 720p
frame downloads, but current download availability must be verified rather than
assumed.

Lock a label-blind matched sample before reading new performance:

1. Match original 4K frames to the current recording/frame identities and publish
   checksums, timestamp offsets, and exact 3x box transforms.
2. For the same fixed centers, compare genuine 4K actor crops with crops from the
   exactly downsampled 720p control. Keep timestamps, boxes, encoder, and head fixed.
3. Measure actor pixels, blur, codec effects, token variation, and correspondence
   support. Continue only if the original source adds genuine observable detail.
4. Keep any external test recordings frozen and separate from auxiliary training.

Primary source: [Okutama-Action paper](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w34/html/Barekatain_Okutama-Action_An_Aerial_CVPR_2017_paper.html).

### P9-1: Actor-Local Space-Time Adapter (working name)

If source fidelity passes, retain per-frame actor-local spatial patch grids and learn
a small center-query temporal adapter over them. Do not first reduce them to temporal
means, absolute spectra, or six correspondence quantiles.

The first locked trial should contain:

- one frozen-encoder center-query space-time adapter with a strict parameter cap;
- a matched spatial-only adapter with the same trainable parameter count;
- a fixed temporal-order-shuffled control;
- a pooled-token MLP capacity control;
- five outer scenario folds, training-only inner selection, and multiple declared
  optimization seeds because the head is now stochastic; and
- shared-failure repair as a diagnostic, never as a sampling or fitting signal.

The adapter is not itself novel: parameter-efficient spatiotemporal adaptation is
established by [ST-Adapter](https://arxiv.org/abs/2206.13559) and
[AIM](https://arxiv.org/abs/2302.03024). A defensible contribution would require a
fixed actor-local/center-query design, matched controls, meaningful repair of the
shared standing/locomotion boundary, and independent transfer—not a new name.

Do not fit a new fusion model to the existing global OOF matrix. If learned fusion is
later justified, regenerate every component prediction inside each new outer-training
population.

## Evidence and immutable artifacts

- [Full extraction summary](../.runs/research_20260907/okutama_ccac_full_v2/results/summary.json)
- [Full CCAC features](../.runs/research_20260907/okutama_ccac_full_v2/results/ccac_features.npz)
- [Full feature manifest](../.runs/research_20260907/okutama_ccac_full_v2/results/feature_manifest.json)
- [P8 supervised summary](../.runs/research_20260907/okutama_native_video_p8/results/summary.json)
- [P8 metrics](../.runs/research_20260907/okutama_native_video_p8/results/metrics.csv)
- [P8 paired statistics](../.runs/research_20260907/okutama_native_video_p8/results/paired_statistics.json)
- [P8 diagnostics](../.runs/research_20260907/okutama_native_video_p8/results/diagnostics.json)
- [P8 OOF probabilities](../.runs/research_20260907/okutama_native_video_p8/results/oof_probabilities.npz)
- [P8 protocol](../experiments/okutama_video_p8_protocol.json)
- [P8 execution lock](../.runs/research_20260907/okutama_native_video_p8/execution_lock.json)

Critical SHA-256 receipts:

| Artifact | SHA-256 |
| --- | --- |
| Full extraction lock | `58649a03850af8cd86825ebf06b234b192fad257f447a2301b0013c4a4e0bbd9` |
| Full feature archive | `9c09d72a91027e1d72fcba923086803e34554783b4c79cba92792e4165ab9ecb` |
| Full extraction summary | `98c084c86d130b0cd99e660f85aa95cd578956d8a1d6e19efa16d0c6dbb84d60` |
| P8 supervised lock | `3f8c891aa49cd3812e25a9919c2467fc3e714b12cd7e22ba00128d980c38d628` |
| P8 OOF probabilities | `e649f35a22ccd1c7deb7c3bee62cc762a9beb22e3190672118d44e03a8dfdbe4` |
| P8 supervised summary | `05f1ee23a368fb2a7d9cb8a681355c9367de7bba3a075f624487386e86696b84` |

This remains adaptive development evidence on the same 11 Okutama scenarios. It is
not independent confirmation, a selection-adjusted claim, or proof of architectural
originality.
