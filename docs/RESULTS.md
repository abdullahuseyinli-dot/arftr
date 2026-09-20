# ARFTR and temporal-research result highlights

The retained result is **85.383648% macro-F1 / 85.895118% accuracy**, with **702
errors** on 4,977 adaptively reused Okutama centers. It is not an untouched test.
The [model card](MODEL_CARD.md) defines the inputs, population and limitations.

## Original architecture evidence

| System | Macro-F1 | Interpretation |
| --- | ---: | --- |
| Exact M4 actor-memory anchor | 84.811129% | Matched baseline |
| P6 restoration only | 85.035693% | Complementary posture and motion evidence |
| A3 motion residual only | 84.856492% | Restricted template correction |
| Temporal update only | 84.994543% | Same-track neighbor evidence |
| Factor residuals without temporal update | 85.102432% | Complementary residual sources |
| **Full ARFTR** | **85.383648%** | Retained combination |
| Shuffled-neighbor control | 80.921234% | Correspondence-destroying control |

![Original matched component evidence](../assets/arftr_architecture_results.png)

Full ARFTR yields **98 rescues, 65 harms, 33 net corrections** over M4. NLL improves
from 0.390042 to 0.377349 and summed multiclass Brier from 0.220730 to 0.213505.
The +0.572519-point macro-F1 interval spans zero (95% scenario bootstrap:
−0.122986 to +1.349570 points); the strict significance screen did not pass.
Component-removal arms reuse the full model's selected coefficients.
[Source export](../results/arftr_development/architecture_study.json).

![Retained ARFTR confusion matrix](../assets/arftr_confusion_matrix.png)

## Earlier sealed temporal confirmation — different models and population

On **1,771 separate confirmation examples**, the earlier temporal study established
useful gains. These must not be relabeled as ARFTR test results.

| System | Macro-F1 | Accuracy | Clip-model use |
| --- | ---: | ---: | ---: |
| Matched static model | 74.58% | 73.01% | 0% |
| Temporal teacher | **78.54%** | **77.08%** | 100% |
| Fixed-budget hybrid | **78.17%** | **76.79%** | 50% |

![Sealed temporal confirmation](../assets/vcoco_v3_confirmation_comparison.png)

The teacher gains **+3.96 pp [2.02, 5.68]** over static. The half-budget hybrid gains
**+3.60 pp [1.44, 6.04]**, retaining **90.7% of that macro-F1 gain** and matching
the teacher's locomotion F1. Clip-use fraction is not a measured latency saving.
[Metrics](../results/vcoco_v3/confirmation_metrics.csv),
[paired uncertainty](../results/vcoco_v3/confirmation_uncertainty.json),
[report and routing analysis](VCOCO_V3_MOTION_IDENTIFIABILITY.md).

## Development gains and their limits

The historical T2-to-ARFTR increase is **71.923768% → 85.383648%**, or **+13.46 pp**,
with 710 fewer errors. It combines representation, source, fusion and memory changes;
it is not the isolated effect of ARFTR's final layer. The
[knowledge graph](../results/arftr_development/knowledge_graph.json) and
[lineage review](HAC_EXPERIMENT_REVIEW_20260912.md) retain the links.

Higher exploratory point estimates also remain visible in the
[complete continuation ledger](../results/arftr_development/experiment_ledger.csv):
the crossing verifier reached 85.611648% but did not improve every outer fold;
the signed-motion control reached 85.467181% with inconsistent folds and worse
proper scores. Neither is a promoted improvement. The earlier CPTR validation gain
likewise failed grouped-OOF verification. See the [research history](RESEARCH_OVERVIEW.md)
and [CPTR report](OKUTAMA_CPTR_DEVELOPMENT.md), without treating diagnostic or oracle
scores as deployable achievements.

## Still-image contributions

The [companion benchmark](https://github.com/abdullahuseyinli-dot/polar-posture-recognition)
presents the 93.99% POLAR ensemble, 86.63% V-COCO held-out posture stack,
86.97% nested-development DINO/SigLIP stack, and DINOv2/DINOv3/SigLIP2 controls.
Their independent populations and evaluation boundaries remain explicit.
