# HAC experiment review and next decision

Date: 2026-09-12. This review inspects completed experiments and computes descriptive comparisons of fixed predictions. No training, model selection, parameter sweep, or alteration of an existing result was performed.

## Current improvement and comparison boundary

The best completed internal result is ARFTR: **85.383648% macro-F1 and 85.895118% accuracy** on all 4,977 original Okutama centers. The classes are sitting, standing, and walking/running.

The original retained temporal teacher scored **71.648321% macro-F1**. The remembered **71.923768%** result was the early T2 distinct-frame sampling control. Both saved prediction populations match the current sample IDs and labels exactly; historical scenario identities and the five-fold assignment also match.

| Comparator | Macro-F1 | Accuracy | Errors | Current F1 gain | Current error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original retained teacher | 71.6483% | 71.3080% | 1,428 | +13.7353 points | 50.84% |
| T2 distinct-frame control | 71.9238% | 71.6295% | 1,412 | +13.4599 points | 50.28% |
| Original P6 consensus | 82.5830% | 83.5041% | 821 | +2.8006 points | 14.49% |
| Current ARFTR | 85.3836% | 85.8951% | 702 | — | — |

Against the original teacher, ARFTR repairs 948 errors and introduces 222, for 726 additional correct predictions. Against T2, the corresponding counts are 940 repairs and 230 harms, for 710 additional correct predictions. These error reductions use classification errors, not one minus macro-F1.

The original teacher was already task-trained: frozen DINOv2 tight/context crops and box geometry fed a small temporal Transformer with factorized posture/motion heads. It used eight frames over 0.5 seconds and five seeds. Thus this is a comparison with the system before this continuation's additions, not an untouched pretrained model. Current downstream ensembles use three seeds. Inputs, temporal support, source processing, supervision, and model families changed across the progression.

The historical POLAR result around 93.99% belongs to a separate static-image task and evaluation. The old sealed Okutama temporal result around 78.54% belongs to a different confirmation population. Neither belongs on the current development leaderboard.

## What produced the gains

| Stage | Macro-F1 | Main change and interpretation |
| --- | ---: | --- |
| Retained original temporal teacher | 71.6483% | Starting system before continuation |
| T2 distinct-frame sampling | 71.9238% | Small sampling improvement |
| P1 real video / frozen V-JEPA | 77.1539% | Better video observations and representation; +5.23 points over T2 |
| P2 video plus DINO appearance | 78.3818% | Image and video evidence complement one another |
| P3 longer context and fusion | 81.4416% | Two-second context plus fusion; +3.06 points over P2 |
| P5 spatial contrast | 81.7665% | Modest benefit from spatial summaries |
| P6 diverse consensus | 82.5830% | Combination of complementary video, image, and spatial classifiers |
| M4, original sources | 83.8046% | Nearby observations of the same tracked actor; net +56 correct versus P6 |
| M4, regenerated sources | 84.8111% | Source processing change through the nested pipeline; net +30 correct |
| ARFTR | 85.3836% | Partial P6 restoration, restricted A3 motion correction, and temporal messages; net +33 correct |

This is a lineage of increasingly capable systems. Adjacent differences do not independently identify the effect of every changed component. P6 was chosen following an inspected ensemble screen; the isolated P3 factorization and P5 spatial contrasts had uncertainty intervals crossing zero.

The largest changes came from the information supplied to the model: real video, suitable pretrained video features, longer context, and improved source processing. P1's real-video control scored 77.15% versus 63.65% for repeated-center video. That comparison removes additional appearance observations as well as motion, so it does not isolate a pure motion effect.

Source regeneration improved matched P6 from 82.5830% to 83.7098%; its scenario interval was [+0.266, +2.427] points, with within-study Holm-adjusted p=.0293. M4 improved from 83.8046% to 84.8111%, but that smaller-group inference remained uncertain (interval [-0.099, +2.394], p=.0723). In isolated source probes, regenerated downsampled video scored 82.5642%, native 4K 82.1686%, and the supplied source 80.4254%. The supported intervention is the source-processing chain, not a claim that more pixels alone caused the gain.

The current architecture is a task-specific system built on frozen image/video encoders. P6 supplies complementary classifiers, M4 aggregates nearby actor evidence, and ARFTR adjusts posture and conditional motion scores. ARFTR is not a newly pretrained foundation backbone. Posture/motion factorization also existed in the original teacher; the recent change is how those coordinates constrain restoration and temporal correction.

## Additions that failed or only helped slightly

| Tested recipe | Macro-F1 | Assessment |
| --- | ---: | --- |
| CPTR parts model | 71.44% versus its 71.65% teacher | Validation gains did not survive grouped evaluation |
| P4 box/camera motion | 81.34% versus 81.44% control | No gain from this geometry recipe |
| P7 center/signed features | 82.5350% versus P6 82.5830% | No system gain |
| P8 compensated point summaries | 82.6269% | Essentially flat; no repair of original shared failures by the system |
| M5 extra corroboration | 82.9482% | Below M4 |
| Expected episode v2 | 83.4801% versus 83.6304% control | Better hazard/bound machinery did not improve classification |
| Ordered/correspondence blends | 82.4172% / 82.0771% | Below the simpler pooled blend at 83.2799% |
| SEAR shared geometry | 80.5837% | Below unrestricted templates, 81.3890%, and no-geometry control, 80.7902% |
| MATR boundary-risk routing | 84.4795% | More harms than repairs; below M4 84.8111% |
| MATR uncertainty/unrestricted routing | 84.4095% / 84.0039% | Gates failed to turn complementarity into a system improvement |
| Fixed 10% geometric A3 correction | 84.9495% | Small improvement; net +11 correct |
| Protected class-diagonal residual | 84.9768% | Small improvement; p=.2031, branch stop criterion triggered |
| FSAR center-only supervision | 82.5810% | Learned correction damaged its strong anchor |
| FSAR all-frame supervision | 84.2262% | Better than center-only, still below M4 and the 84.9495% starting anchor |

These results reject the tested implementations. They do not establish that all geometry, correspondence, adaptation, or routing methods are ineffective.

SEAR's coordinate shuffle changed 80.5837% to 80.4673%; removing video changed it to 61.0715%. This gives little support for the claimed geometry mechanism and strong evidence of reliance on video. Nevertheless, its unrestricted A3 control remains a useful source of motion evidence inside ARFTR. A failed full architecture can supply a useful component.

## What ARFTR's controls establish

| Fixed component removal | Macro-F1 |
| --- | ---: |
| Exact M4 | 84.8111% |
| P6 restoration only | 85.0357% |
| A3 motion only | 84.8565% |
| Temporal update only | 84.9945% |
| Residual without temporal update | 85.1024% |
| Full ARFTR | 85.3836% |
| Shuffled-neighbor control | 80.9212% |

These ablations reuse the full model's selected coefficients. They support complementary contributions, but are not separately optimized competing methods. The shuffle changes neighbor identity, including which actor is borrowed from, so it establishes sensitivity to valid temporal/actor relationships rather than identifying a single causal property of time order.

ARFTR improves four of five folds and nine of eleven scenarios. NLL improves .390042 to .377349; Brier improves .220730 to .213505. All three class F1 values improve, but sitting recall declines: correct sitting examples change 630 to 621, standing 1753 to 1778, and walking/running 1859 to 1876. Improved sitting precision compensates for that recall loss.

Its computational replay completed successfully. Its strict adaptive statistical screen did not pass: exact scenario-swap p=.0649414 and the scenario-bootstrap F1-difference interval is [-0.123, +1.350] points. The 85.38 result is an audited internal best on repeatedly inspected scenarios; independent generalization remains unmeasured.

## What the frame-supervision experiment actually teaches

Both learned arms completed all 30 combined fits and passed the independent checkpoint/output/statistics audit. They used the same frozen features and optimization budget. The all-frame arm used 114,951 unique supervised physical actor frames across 444 tracks, with zero frame/track crossings between outer folds. These are more supervised observations from the existing scenarios, not 114,951 independent scenes.

All-frame supervision improves the matched center-only arm by **1.6453 F1 points**, with scenario interval [+0.7165, +2.6711] and p=.0078125. Its standalone classifier also improves from 64.7548% to 67.5726%.

This supports the broader supervised-frame training recipe. Additional labels, crop/view diversity, track weighting, and reduced repeated-center overfitting are not fully separated. Both standalone heads remain weak. The earlier claim that integration alone explains the failure was too strong.

There are two concrete mechanisms worth testing:

1. **The residual was scaled but not bounded.** The code multiplies unbounded local logits by .25. Across the all-frame held predictions, .25 times the local logit range has median 2.134, 95th percentile 3.682, and maximum 5.622. The median largest pairwise odds change can therefore be about exp(2.134), or 8.45-fold. A .25 coefficient did not ensure a small correction.
2. **One output serves incompatible roles.** Auxiliary cross-entropy asks the local logits to predict the entire class; the center loss asks the same logits to correct an already strong prediction. This can double-count shared evidence or create competing gradients. It is a hypothesis to test, not a diagnosed cause. Inference sees only the center-frame DINO CLS vector: training on additional frames did not give this head temporal input for standing-versus-walking recognition.

Existing work shows that multiple objectives can produce detrimental gradient interference; that supports measuring this possibility here, without establishing that it occurred in our run. See [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html). Zero-initialized residual paths are also established practice, as illustrated by [ReZero](https://proceedings.mlr.press/v161/bachlechner21a.html); an initially exact fallback alone does not bound corrections after training.

## Remaining errors and correlations

The following fixed-model comparisons were recomputed after exact sample/label/fold alignment. No new combination or routing rule was fitted.

| Actual input-support stratum | Centers | M4 F1 | ARFTR without temporal step | Full ARFTR F1 | Full ARFTR errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Known pure sampled support | 4,184 | 88.2582% | 88.1527% | 88.5743% | 447 |
| Known mixed sampled support | 682 | 63.4673% | 66.0627% | 64.8958% | 227 |
| Unknown support | 111 | 72.3797% | 73.4784% | 75.3321% | 28 |

Pure/mixed refer to the exact frames supplying the node, not every frame in a continuous video interval. The ARFTR summary's broader non-boundary category includes unknown rows; this table separates them.

Temporal aggregation helps stable observations but still gives back some transition corrections. Relative to ARFTR without temporal aggregation, the full model removes 19 pure-support errors, adds six mixed-support errors, and removes two unknown-support errors. Most remaining errors are nevertheless on pure supports: 447/702. A boundary-only solution cannot resolve the entire problem.

ARFTR's confusion matrix contains 457 standing/locomotion swaps, 224 sitting/standing swaps, and 21 sitting/locomotion swaps. Motion ambiguity remains the largest error category. Scenarios 1.10 and 2.11 account for 217 errors on 1,070 rows; new data collection should span difficult appearance/view/transition conditions and entire unseen scenarios, not merely repeat already observed tracks.

The all-frame FSAR prediction repairs 126 ARFTR errors but would harm 177 ARFTR-correct rows if substituted. Their binary error correlation is .7568, and 576 errors are shared. Within pure support, FSAR has 79 unique repairs versus 122 harms; on mixed support, 44 versus 46. This explains why complementary predictions are interesting yet insufficient to justify direct combination. Those label-derived repair sets are diagnostics, not deployable routing features or attainable claimed gains.

## Next phase: one decisive controlled trial

Keep ARFTR as the frozen reference at 85.3836%. The next research question is whether frame supervision can train useful *corrections* without overwriting existing evidence.

Use zero-initialized correction outputs in posture and conditional-motion coordinates, separate from auxiliary frame-classification outputs. Enforce actual limits, for example delta_s = b_s * tanh(r_s) and delta_m = b_m * tanh(r_m). This bounds factor-score updates; it does not by itself prove calibration or correctness. Factorization alone is a reparameterization already present earlier in the project.

The minimum controlled matrix is:

| Arm | Purpose |
| --- | --- |
| Unchanged ARFTR | Exact fallback/reference |
| Bounded correction, separate auxiliary output, center-only supervision | Matched learning control |
| Identical bounded model, all-frame supervision | Primary test of the broader supervision recipe |
| Same all-frame model and bound, shared auxiliary/correction output | Test whether separating objectives helps |
| Same all-frame model and separate outputs, scaled but unbounded correction | Isolate the effect of enforcing the limit |

Match the feature input, model capacity as closely as possible, training steps, physical-frame weights, seeds, and scenario splits. Keep the first experiment's input fixed so constraint/objective effects remain interpretable. Include zero correction among the choices; choose any limits or stopping settings only within scenario-grouped inner folds.

Rebuild genuinely nested predictions for every added upstream stage. Existing global outer-fold FSAR or ARFTR predictions cannot simply become a meta-training cache for a new held fold: the models predicting other folds may have trained on the held fold. Every upstream fitting and coefficient-selection stage must exclude that held scenario population.

The primary endpoint is improvement over ARFTR, not merely over the weaker FSAR control. Predeclare positive net corrections, class precision/recall, pure/mixed-support harms, NLL/Brier safeguards, and scenario uncertainty. A practical next milestone is approximately 85.9% (about +0.5 points); that is a target, not a forecast. Retain the frozen anchor if the correction adds more harm than benefit.

If broader supervision helps the local head again but none of the controlled corrections improves the anchor, stop this center-CLS integration branch. The next information experiment should provide actual local temporal observations to the motion head, paired with center-frame posture, using the existing cached sequence before committing to a dense new extraction. Earlier signed-feature and coarse-correspondence failures mean this needs matched simple controls, not an assumption that more temporal features must help.

A separate untouched scenario set is needed to decide whether the latest gains generalize. More epochs on the same tracks will not answer that question. No new training is launched by this review; the user's rule remains to verify and hand off any future training expected to exceed 20 minutes.

## Evidence paths

- Earliest/T2 aligned probabilities: `.runs/research_20260907/okutama_temporal_controls/temporal_control_ensemble_predictions.npz`
- Original headline: `results/okutama_cptr/headline_metrics.csv`
- Original architecture: `docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md`
- P1: `docs/HAC_OKUTAMA_VIDEO_P1_RESULTS_20260907.md` and `.runs/research_20260907/okutama_native_video_p1/results/summary.json`
- P2/P3/P5/P6: `.runs/research_20260907/okutama_native_video_p2a/results/metrics.csv` and the corresponding `okutama_native_video_p{3,5,6}/results/metrics.csv`
- Historical mechanisms: `docs/HAC_RESEARCH_MAP_AND_NEXT_PHASE_20260907.md`, `docs/HAC_SHARED_EVIDENCE_RECONSTRUCTION_PLAN_20260908.md`
- Memory: `.runs/research_20260908/evidence_memory/results/summary.json`
- Source swap: `.runs/research_20260908/source_swap_v1/results/v0001/summary.json` and its `oof_probabilities.npz`
- SEAR: `.runs/research_20260908/sear_matrix_v1/results/v0001/summary.json`
- MATR: `.runs/research_20260912/matr_cached_gate_v3/results/v0001/summary.json`
- Class-diagonal residual: `.runs/research_20260912/class_diagonal_residual_v1/results/v0001/summary.json`
- ARFTR: `.runs/research_20260912/arftr_v1/results/v0001/summary.json` and `oof_probabilities.npz`; the separate `experiments/audit_okutama_arftr.py` program generates the final replayed summary rather than a separate audit-directory receipt.
- FSAR: `.runs/research_20260912/frame_supervised_anchor_residual_v2/results/v0001/summary.json` and `oof_predictions.npz`
- FSAR checkpoint audit: `.runs/research_20260912/frame_supervised_anchor_residual_v2_audit/summary.json`
- FSAR objective: `src/hac/frame_supervised_residual.py`
