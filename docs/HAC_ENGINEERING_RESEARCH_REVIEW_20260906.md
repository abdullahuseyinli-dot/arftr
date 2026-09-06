# HAC engineering and research review — 2026-09-06

Review started 2026-09-06 and finalized 2026-09-07. Filenames retain the continuation
date; the bounded literature review uses the stated September 6 cutoff.

## Decision

Continue with a narrow mechanism-led research program. The evidence justifies two
questions: can we preserve established temporal evidence when added inputs are
unreliable, and can uncertain spatial evidence supply corrections that simple visual
fusion cannot? It does not establish a new architecture win or a breakthrough.

Independent reviews covered implementation, statistical comparability, and primary
literature. A fourth analysis recomputed component contrasts and correlations from
immutable development exports. The executable next-run sequence is specified in
[the revised plan](HAC_NEXT_RUN_PLAN_20260906.md). This review supersedes the earlier
draft's prioritization and corrects its grouping uncertainty; historical protocols,
results, and locks retain their original meaning.

## 1. What actually improved on V-COCO

The new analysis uses the same 6,640 people from 4,123 source images. These are
retained development predictions; outer fold assignments are attested, not
independently reconstructed. Every comparison below uses the same 10,000 image
resamples. Intervals condition on the selected historical models and omit uncertainty
from repeating fitting and selection. These are exploratory family comparisons.

| Comparison | Macro-F1 difference | Conditional 95% interval | NLL difference | Decision implication |
| --- | ---: | --- | ---: | --- |
| DINO factorized − DINO flat | +0.001670 | [−0.002502, +0.005787] | −0.001975 | Factorization alone has unresolved incremental value. |
| DINO+SigLIP linear − DINO flat | +0.009710 | [+0.001534, +0.017902] | −0.012207 | A relatively simple mixed-representation control already captures substantial gain. |
| Full mixed reliability stack − DINO flat | +0.015341 | [+0.008106, +0.022634] | −0.027947 | Useful development association; not an isolated architectural effect. |
| Full mixed stack − DINO factorized | +0.013672 | [+0.006778, +0.020839] | −0.025972 | Adding the mixed family is promising, but multiple ingredients change together. |
| Full mixed stack − mixed linear | +0.005631 | [−0.001131, +0.012335] | −0.015740 | Extra macro-F1 beyond the simpler control remains uncertain. Its NLL interval [−0.022926, −0.008566] is favorable. |

The full model rescues 251 baseline errors and introduces 169 new errors. It corrects
30.17% of the baseline's errors, while 581 errors remain shared. Standing↔locomotion
confusions fall from 644 to 573: 71 of the 82 net errors removed. Sitting F1 changes
only +0.00290, with an interval crossing zero; standing and locomotion account for
the stronger gains. These are the error types future spatial evidence must address.

The headline gain cannot establish which of SigLIP, factorization, reliability
features, or the fitted combination is responsible. The historical evaluator shares
outer people but varies inner/stack split RNG by family and compares 12 stack
hyperparameter settings with eight SVM settings. An explicit mixed factorized model
without reliability features is missing. The new plan makes that attribution
control mandatory before claiming reliability as an invention.

Sources: [retained family metrics](../results/vcoco_v3/source_tag_development_metrics.csv),
[nested evaluator](../experiments/evaluate_vcoco_v3_nested_stacks.py),
[candidate grid](../experiments/vcoco_v3_candidate_grid.json), and the new numerical
receipt in Section 7.

## 2. CPTR's useful discovery is a failure mechanism

The established temporal baseline remains preferable: OOF macro-F1 0.716483 versus
0.714445 for CPTR. The candidate rescues 50 errors and introduces 54; 1,378 errors are
shared. Thus it corrects only 3.50% of baseline errors, leaving 96.50% shared. A router
choosing between these two argmax predictions cannot correct their shared mistakes.
Its label-informed accuracy ceiling is 0.723126 versus baseline 0.713080; this is
neither a macro-F1 bound nor evidence that such a router can be learned.

The strongest evidence is the agreement between three different observations:

1. **Implementation:** new residual heads initialize to zero, but the legacy temporal
   residual is multiplied by `sigmoid(5) × valid_fraction`. For each factorized head,
   initialization is `z_static + 0.99330717 × q × (z_temporal − z_static)`. It therefore
   does not implement an exact return to the established temporal model.
2. **Historical negative control:** all five seed-43 folds used zero fitting epochs,
   yet every fold regressed. Their pooled OOF macro-F1 difference is −0.00471334.
   Training cannot explain a change in those zero-epoch models, although checkpoint,
   preprocessing, and execution correspondence still require a real replay.
3. **Error localization:** occlusion affects 412/4,977 windows (8.28%) but accounts
   for 72/127 argmax changes (56.69%). The candidate moves much farther from the
   temporal baseline in these windows, predominantly toward the static prediction.

| OOF diagnostic | Clear | Occluded |
| --- | ---: | ---: |
| Rows | 4,565 | 412 |
| Macro-F1 difference | +0.001540 | −0.030816 |
| Rescued / harmed | 30 / 21 | 20 / 33 |
| Mean probability L1 movement | 0.02717 | 0.27073 |
| Fraction moving toward static | 67.40% | 94.17% |
| Median projection toward static | 0.02583 | approximately 1.000 |

The new synthetic harness executes the production CPTR forward implementation using
constant synthetic anchors and no data. At valid fractions 1, 0.5, and 0, the legacy
coefficients are 0.99330717, 0.49665359, and 0. All added residuals are exactly zero;
the half-valid case changes the synthetic predicted class, and the wholly masked
case returns static exactly. This proves initialization attenuation in code. It does
not measure a real-data repair gain or prove this explains the complete OOF loss.

The binary reporting subgroup is any occlusion over the full cached window; `q`
depends on the short-path mask. Do not infer an individual gate coefficient from
that binary subgroup or from probability-space projection.

Sources: [model](../src/hac/cptr.py),
[fold and seed table](../results/okutama_cptr/fold_seed_metrics.csv),
[epoch policy](../experiments/okutama_cptr_crossfit_plan.json),
[synthetic harness](../experiments/audit_cptr_initialization_contract.py), and
[existing diagnostic](../experiments/analyze_okutama_cptr_baseline_preservation.py).

## 3. Correlations: what survives conditioning, and what does not

The primary additional association is the difference in the candidate's error-rate
effect between occluded and clear windows. Positive means relatively more harm under
occlusion. Resampling whole declared scenarios gives:

| Exploratory effect contrast | Point | Conditional 95% interval |
| --- | ---: | --- |
| Occluded-minus-clear paired error-rate effect | +0.033525 | [+0.000499, +0.068233] |
| Same contrast standardized to the full cohort's true-class proportions | +0.040769 | [+0.004243, +0.077933] |
| Class-standardized NLL effect contrast | +0.007967 | [−0.041225, +0.066637] |
| Transition-minus-stable paired error-rate effect | −0.009066 | [−0.019871, +0.008551] |

Class mix alone therefore does not remove the observed occlusion error association.
Scenario, confidence, visibility, and other confounding remain; these are unadjusted,
post hoc intervals across several examined quantities, not a causal
difference-in-differences study or a confirmatory subgroup result. Standardization
fixes the class weights to the full OOF cohort, requires all three classes in both
arms per draw, and retains invalid-draw counts (zero invalid among these 10,000).
Occluded locomotion recall falls 0.06923 and standing recall falls 0.02564, while
sitting recall rises 0.01149. Class-conditioned slices report recall, not a misleading
single-class macro-F1.

Other correlations provide useful limits. Equal-weight image means give Spearman
rho +0.310 between baseline confidence and candidate-minus-baseline NLL: the candidate
tends to improve loss more on difficult images. Fixed descriptive bins show macro-F1
gains +0.10265 below 0.6 baseline confidence, but −0.00241 above 0.95. Bin class mixes
differ substantially; these numbers cannot select a router threshold. The correlation
also shares the baseline with its outcome definition and is mathematically coupled.

Image-level mean log person area has only rho +0.058 with NLL change; the earlier
class-specific scale contrasts all crossed zero. There is no strong evidence to
prioritize a size-conditioned architecture. At the 11-scenario level, occlusion
prevalence has rho −0.218 with NLL change, while the window-level error effect is
harmful. This illustrates why ecological correlations, different metrics, and
within-subgroup effects cannot be substituted for one another. No correlation
p-values or fitted prediction rules were produced.

## 4. Components that have not earned further complexity

These are stop decisions for this development cycle, not claims of universal
uselessness. Historical screens span code revisions and mostly one seed, so their
differences are engineering history rather than isolated causal effects.

| Branch | Recorded incremental validation macro-F1 | Current disposition |
| --- | ---: | --- |
| Camera compensation over raw trajectory | +0.00144 | Small, unconfirmed; retain only as a control if motion geometry becomes necessary. |
| Parts over short residual | +0.00312 | Weak single-seed signal; final combined OOF candidate failed. |
| Dual clocks over short residual | −0.00716 | Stop this cycle. |
| Integrated trajectory+parts over short+parts | −0.00973 | Stop this cycle. |
| Original / refined counterfactual objective | −0.02449 / −0.00810 | Stop this cycle. |
| Masked feature pretraining | −0.00624 | Stop this cycle. |
| SigLIP posture specialist | −0.00551 | Stop in CPTR; this does not negate V-COCO mixed-representation evidence. |
| GroupDRO | −0.00234 | Stop this cycle. |
| Top-block LoRA versus its legacy reference | −0.03018 | Stop this configuration; no larger adaptation search. |

The [faithfulness table](../results/okutama_cptr/faithfulness_metrics.csv) supports
multi-frame appearance more than precise order: repeating the short stream costs
0.04605 macro-F1, but shuffling improves 0.000454 and reversal costs only 0.002690.
Repeating part tokens costs 0.001206, removing them 0.007631, and zeroing geometry
0.000809. These are three-scenario validation interventions. Repeating short features
also alters the legacy teacher, and removing parts changes shared gate inputs, so
these are not isolated measurements of the new branch's contribution.

Two additional design facts matter:

- The declared eight-sample short-window sampler returns
  `[4, 6, 6, 8, 8, 10, 10, 12]`: five distinct cached frames. Both baseline and CPTR use
  this sequence, so it is a shared sampling limitation, not a proven mismatch.
- Current part confidence combines resolution with provider occlusion and assigns
  the same confidence to all seven regions. It is not predicted joint uncertainty.
  Provider occlusion also enters valid masks and gate inputs. A deployable study
  needs measured image-derived visibility or an explicit metadata-assisted claim.

Sources: [screen table](../results/okutama_cptr/component_ablation.csv),
[parts cache](../experiments/cache_okutama_cptr_parts.py),
[feature construction](../src/hac/cptr_features.py), and
[intervention code](../experiments/evaluate_okutama_cptr_faithfulness.py).

## 5. Two corrections to the earlier evidence interpretation

**Grouping:** the 11 values named `recording_id` are scenario identifiers in the
hash-matched historical generation code. Original drone-video identifiers are kept
separately. The [lineage receipt](HAC_SCENARIO_LINEAGE_REVIEW_20260906.md) closes the
column-semantics concern without opening the mixed manifest. It does not reconstruct
every original mapping or establish independence between distinct scenarios.

**Historical significance:** the V-COCO helper computes twice the smaller uncentered
bootstrap sign-tail proportion and calls it `two_sided_p`; the evaluator then applies
Holm. That is not the null-based paired permutation test specified for continuation.
Preserve those values as historical metadata, not independently validated permutation
significance. The new analysis reports conditional intervals and no p-values. Future
primary tests must implement whole-group probability-vector swaps. See
[the statistical helper](../src/hac/vcoco_v3_models.py) and
[the evaluator](../experiments/evaluate_vcoco_v3_nested_stacks.py).

## 6. What could constitute a new contribution

The proposed research question is: **Can uncertainty over plausible anatomy and
support evidence guide separate posture and motion corrections, adding information
beyond matched spatial appearance while preserving the established prediction when
new evidence is absent?** A useful answer requires measured correction quality,
missingness robustness, and transfer. A single image cannot guarantee identification
of true motion; consensus among incorrect pose hypotheses is not calibrated certainty.

The broad ingredients are established. This bounded primary-literature review is
current through 2026-09-06; it does not establish patentability or exhaustive novelty.
Published HOI/contact/skeleton scores are not comparable to our three-class endpoint.

| Primary work | Existing idea | Required distinction or control |
| --- | --- | --- |
| [ReZero, 2021](https://proceedings.mlr.press/v161/bachlechner21a.html) | Zero-initialized residual scaling | Exact baseline return is engineering correctness, not sufficient novelty. |
| [ControlNet, 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) | Frozen established model with zero-initialized conditional additions | A new task-specific mechanism must add evidence beyond this known recipe. |
| [ModDrop, 2015 version](https://arxiv.org/abs/1501.00102) | Missing/noisy-modality training | Explicit masks and modality dropout alone are established. |
| [PMFNet, 2019](https://arxiv.org/abs/1909.08453) | Pose-guided local appearance and interaction context | Joint-indexed DINO features need an equal-budget spatial control. |
| [PaStaNet, 2020](https://arxiv.org/abs/2004.00945) | Body-part states and activity reasoning | Show value without assuming extra part-state supervision. |
| [PoseConv3D, 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html) | Heatmap pose representation and robustness | Compare uncertainty propagation with simple heatmap/dispersion features. |
| [Skeleton-aware GCN for HOI, 2022](https://arxiv.org/abs/2207.05733) | Human/object keypoint graphs | Graph topology itself is not novel. |
| [HOT, 2023](https://arxiv.org/abs/2303.03373) | Body/scene contact inference | Its training images include V-COCO; a pretrained auxiliary cannot be assumed clean. |
| [UAHOI, 2024](https://arxiv.org/abs/2408.07430) | Prediction-uncertainty-aware HOI learning | Distinguish uncertainty in input evidence from generic confidence scores. |
| [GraphiContact, March 2026](https://arxiv.org/html/2603.20310v1) | Pose, contact, pretrained encoders and adaptive routing | A close competing direction. Its SIMU describes perturbation variability rather than calibrated uncertainty; our proposed calibration and activity endpoint still require evidence and novelty verification. |

If the grid control performs as well as the anatomical model, keep the grid model.
If average pose hypotheses perform as well as learned uncertainty weighting, keep
averaging. If the remaining errors require time, prioritize temporal evidence or
abstention. A simpler effective solution is a successful research outcome even when
it defeats the original architectural story.

## 7. Reproducible outputs and boundaries

Run the new [component analysis](../experiments/analyze_hac_component_relationships.py)
with `--output-dir .runs/research_20260906/component_relationships_final --resamples 10000`
only when that output does not exist; use a new directory for a replay. It consumes
four fixed, independently hashed development exports. It saves source hashes,
canonical group ordering, the actual resampling indices, RNG/version metadata,
aggregate tables, and a standalone PNG/PDF figure. It does not open original
manifests, feature stores, checkpoints, or new evaluation roles.

Final ignored analysis directory: `.runs/research_20260906/component_relationships_final`.

| Artifact | SHA-256 |
| --- | --- |
| `summary.json` | `d0d3f13b0ef38475b522d560d65daaa126181ff96d592cb713155ee711f057e4` |
| `descriptive_slices.csv` | `70e5389634e104a539d7d748d0c990660f908a8462afabb05f2c4059a38a038f` |
| `vcoco_resample_indices.npy` | `2dd8bd348dc1d1a38efbebda8c24ed3408eb9084a00b364fcd463561bc05da41` |
| `cptr_resample_indices.npy` | `5925747b057317a86e3b56af279e36cbf097a8f04fdd8455ea00314055c2da75` |
| Analysis source | `c69ab5b321d95419d0a20fde673d189de36657fa3db49929ed74653a6a4a08f3` |

The synthetic report at `.runs/research_20260906/cptr_initialization_contract/summary.json`
has SHA-256 `dd10f9ba20182923c8f77f00ab4a42bda5e50036c6af5a65f57dcee65a5fb19a`.
CPTR intervals in this new correlation study use seed 20260906, whereas the earlier
CPTR diagnostic used 20260919. The small numerical interval differences are from
that documented resampling stream; neither replaces the other as a confirmation test.

This review trained no model and opened no calibration/confirmation/test rows,
images, arrays, or checkpoints. One reviewer encountered a public README paragraph
containing old aggregate test metrics while checking its scope; those metrics were
excluded from all analysis, correlations, and decisions. The earlier receiving-session
calibration-manifest incident remains disclosed in
[the provenance record](HAC_CONTINUATION_PROVENANCE_20260906.md).

The historical document-lineage gap remains. It does not prevent this analysis,
synthetic checks, or preparation of concrete future protocols. New fitting must use
separate current protocols with explicit data roles and exact committed sources.
