# HAC P8 CCAC pilot results

Date: 2026-09-07  
Scope: label-blind adaptive-development measurement feasibility  
Status: **all declared gates passed; full CCAC extraction authorized**

## Result

The locked 128-clip Camera-Compensated Actor Correspondence (CCAC) pilot completed
all 128 immutable workloads and all 1,920 requested adjacent-frame rows. It decoded
only the 1,789 prelocked valid JPEG members, read no action labels or model
predictions, and fit no classifier. Two preselected clips reproduced their pair
metrics, clip metrics, correspondence NPZ bytes, and applicable overlay bytes
exactly.

| Locked gate | Observed | Threshold | Pass |
|---|---:|---:|:---:|
| Camera-usable requested pairs | 1,750/1,920 (91.15%) | at least 90% | yes |
| Articulation-usable clips | 107/128 (83.59%) | at least 70% | yes |
| Scenarios with at least half of clips usable | 11/11 | at least 8 | yes |
| Small native actors | 40/48 (83.33%) | at least 50% | yes |
| Medium native actors | 42/48 (87.50%) | at least 50% | yes |
| Large native actors | 25/32 (78.13%) | at least 50% | yes |
| Synthetic correctness and exact replay | all passed | all required | yes |

Translation was usable for 1,658/1,920 requested pairs and 112/128 clips. Within-
actor residual motion was usable for 1,592/1,920 pairs. Missing frames remained in
every denominator: 1,895 pairs had both frames and 25 did not.

The camera gate passed by only 22 pairs. That margin is sufficient under the frozen
rule but should not be described as overwhelming evidence.

## Scenario and size audit

| Scenario | Articulation-usable clips | Camera-usable pairs |
|---|---:|---:|
| 1.10 | 8/11 | 153/165 |
| 1.11 | 11/12 | 177/180 |
| 1.2 | 11/12 | 167/180 |
| 1.3 | 7/12 | 157/180 |
| 1.4 | 11/12 | 178/180 |
| 1.5 | 11/11 | 163/165 |
| 2.11 | 11/12 | 172/180 |
| 2.2 | 9/11 | 165/165 |
| 2.5 | 8/12 | 108/180 |
| 2.7 | 9/11 | 137/165 |
| 2.8 | 11/12 | 173/180 |

Scenario 2.5 is the principal camera-estimation weakness. Scenario 1.3 combines
camera failures with limited actor-region support. Six of the seven failed large-
actor clips occur in these two scenarios, while three scenarios contain no selected
large actors; the aggregate size ordering therefore does not establish a causal
actor-size effect.

The 12 deliberately retained incomplete clips achieved 144/180 camera-usable pairs
and 8/12 articulation-usable clips. Their conditional median residual was lower
than for complete clips, but their coverage was worse because unavailable pairs
correctly remained failures. This is a missingness/conditioning example, not a
quality advantage.

## Measurement behavior

| Quantity over available or usable pairs | Median | 10th–90th percentile |
|---|---:|---:|
| Background correspondences | 776 | 689–798 |
| Actor-region correspondences | 17 | 5.4–31 |
| Actor forward/backward error | 0.0497 px | 0.0128–0.1699 px |
| 7 px vs 15 px LK-window disagreement | 0.2503 px | 0.0774–0.5942 px |
| Disjoint camera-audit median error | 0.2752 px | 0.1440–0.4872 px |
| Camera-motion reduction | 95.08% | 74.37–98.01% |
| Compensated translation magnitude | 0.1836 heights/s | 0.0332–1.0221 |
| Within-actor residual magnitude | 0.0737 heights/s | 0.0215–0.2307 |

H0 identity, H1 similarity, and H2 homography were selected for 91, 725, and 1,079
available pairs respectively. H2 is selected under harder image motion: its lower
pass fraction is not a causal model comparison. Among 1,074 comparable H2 choices,
H2 improved the disjoint audit error over valid H1 each time.

The dominant failure counts were 145 camera-audit p90 failures, 154 insufficient
actor-region correspondences for the residual statistic, four insufficient vertical
regions, and 25 unavailable pairs. Failure categories overlap by design.

Long-horizon persistence is weaker than pairwise feasibility. The median clip kept
41.74% of center-seeded points for at least 12/15 links, and the median point survived
9.59 links. Therefore the next representation uses robust summaries of independently
reseeded adjacent pairs. It does not claim persistent anatomical tracks or gait
cycles.

## Fixed continuation

The full 4,977-row extraction keeps every pilot threshold and the disjoint
fit/model-selection/audit split unchanged. It preserves all 15 requested long-window
pairs, including failures, and reuses the 128 pilot measurements only after exact
source, frame, box, seed, code, and artifact validation.

Before any supervised access, it publishes four fixed float32 blocks:

- `Q[16]`: availability, camera/actor support, audit error, LK agreement, center
  survival, and native scale reliability;
- `R[9]`: 10th/50th/90th percentiles of raw x, y, and magnitude translation;
- `C[9]`: the same summaries after camera compensation, using exactly the same
  pair mask as `R`; and
- `A[6]`: 10th/50th/90th percentiles of median and p90 within-actor residual.

Unavailable motion blocks use a numeric zero sentinel only in the finite classifier
matrix and remain distinguishable through `Q` plus explicit validity arrays. The
source pair tables retain nulls, never fabricated observed zero motion.

This pilot authorizes measurement extraction only. It provides no action-recognition
score, no forecast of improvement over P6, and no independent confirmation.

## Evidence

- [Pilot protocol](../experiments/okutama_ccac_pilot_protocol.json)
- [Execution lock](../.runs/research_20260907/okutama_ccac_pilot/execution_lock.json)
- [Summary](../.runs/research_20260907/okutama_ccac_pilot/results/summary.json)
- [Pair metrics](../.runs/research_20260907/okutama_ccac_pilot/results/pair_metrics.csv)
- [Clip metrics](../.runs/research_20260907/okutama_ccac_pilot/results/clip_metrics.csv)
- [Quality overview](../.runs/research_20260907/okutama_ccac_pilot/results/quality_overview.png)
