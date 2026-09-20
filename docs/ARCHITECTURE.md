# ARFTR: architecture and component evidence

ARFTR is the retained fusion architecture from this project's person-centered
image/video and actor-memory research. Its design contribution is the combination
of **anchor-based restoration, factor-specific expert correction, and a
track-constrained temporal update**. It builds on frozen visual representations;
it is not a newly pretrained backbone. Posture/motion factorization already existed
in the earlier models—the contribution here is how it structures evidence fusion.

## What ARFTR actually computes

ARFTR means **Anchor-Restored Factorized Temporal Residual**. It operates on
upstream probabilities rather than learning a new video encoder:

- **M4 anchor:** the selected new-source actor-memory model supplies base probabilities.
- **P6 restoration:** the image/video/spatial consensus restores posture and motion log-odds.
- **A3 residual:** the unrestricted-template expert contributes only to upright motion.
- **Temporal step:** factor scores use exact +/-1-second neighbors from the same
  scenario, fold, recording and track, then decode back to three probabilities.

Posture is sitting versus upright; motion is walking/running versus standing
conditional on being upright. Four coefficients are selected from a fixed
300-candidate set using inner out-of-fold predictions inside each outer training
fold. The ARFTR layer adds no neural fits; its upstream predictions include
three-seed refits. These details follow the
[retained protocol](../experiments/okutama_arftr_protocol.json) and
[implementation](../src/hac/arftr.py).

```mermaid
flowchart TD
    M["M4 probabilities: anchor"] --> F["Convert to posture and motion log-odds"]
    P["P6 probabilities: restoration"] --> F
    A["A3 probabilities: unrestricted-template expert"] --> F
    F --> R["Restore P6 factors; add A3 motion residual"]
    S["Four coefficients selected from inner OOF predictions<br/>Outer-training data only"] -.-> R
    S -.-> T
    R --> T["One temporal averaging pass in factor space"]
    N["Exact -1 / +1 second neighbors<br/>Same scenario, fold, recording and track"] --> T
    T --> D["Decode sitting / standing / walking-running probabilities"]
```

For posture log-odds `s` and conditional-motion log-odds `m`, the residual step is:

```text
s = s_M4 + a * (s_P6 - s_M4)
m = m_M4 + b * (m_P6 - m_M4) + c * (m_A3 - m_M4)
```

For each factor `z`, the temporal pass uses
`z_final = z + beta * (mean(valid_neighbor_z) - z)`; a row without a valid neighbor
is unchanged by that pass. Decode `p_sitting = sigmoid(s_final)`, then split the
remaining upright probability with `sigmoid(m_final)` into walking/running and
standing. All-zero coefficients return the original M4 probabilities byte-for-byte.
The protocol applies each arm per aligned upstream seed, then averages probabilities.

No class labels enter these inference functions. The recorded setup **does require
person/track metadata and may use future (+1 second) evidence**; it is not a tested
causal, zero-latency raw-video classifier. See the [model card](MODEL_CARD.md).

## Original component study

All seven arms use the same 4,977 centers and five scenario-grouped outer folds.
The original three-seed prediction averages are preserved. Arms that remove a
component reuse the full arm's inner-selected coefficients, rather than being
separately tuned. This is a component-removal study, not seven independent searches.

| Arm | Macro-F1 | Role |
| --- | ---: | --- |
| Exact M4 anchor | 84.811129% | Byte-exact zero-correction control |
| P6 restoration only | 85.035693% | Restore posture and motion without A3 residual or temporal update |
| A3 motion only | 84.856492% | Add only the unrestricted-template expert's motion residual |
| Temporal only | 84.994543% | Same-track temporal update without factor residuals |
| Factor residuals, no temporal | 85.102432% | Combine P6 restoration and A3 motion |
| **Full ARFTR** | **85.383648%** | Retained combination |
| Shuffled-neighbor control | 80.921234% | Destroy temporal identity while preserving neighbor counts |

![Original ARFTR component comparison, including the anchor and shuffled-neighbor control.](../assets/arftr_architecture_results.png)

Versus M4, full ARFTR rescues 98 errors and harms 65 correct predictions: **33 net
corrections**, with errors falling from 735 to 702. NLL decreases from 0.390042 to
0.377349, and summed multiclass Brier decreases from 0.220730 to 0.213505.

The macro-F1 change is **+0.572519 pp**, with a 95% scenario-bootstrap interval of
**[−0.122986, +1.349570] pp** (11 groups, 10,000 resamples). The exact one-sided
scenario-swap test gives **p = 0.064941**, above the predeclared 0.05 threshold;
the original adaptive screen therefore did not pass in full. Retention is a
development decision, not established statistically significant superiority.

The shuffled control supports the relevance of the chosen temporal correspondence
in this recipe; it does not isolate pure motion from correlated appearance or
establish generalization. These distinctions also keep the **+13.459880 pp historical
T2-to-ARFTR gain** separate from the narrower layer comparison.

The [aggregate component export](../results/arftr_development/architecture_study.json)
and [source manifest](../results/arftr_development/architecture_manifest.json) retain
the original scores, confusion matrices, control contract, uncertainty and screen
decision. Public checks recompute confusion-based metrics and net corrections;
bootstrap intervals and proper scores require private row predictions for replay.

## Experimental knowledge map

This map connects ARFTR to its development lineage and subsequent research. The
arrows do not imply isolated causal gains for every stage. Detailed later trials
belong to the [research history](RESEARCH_OVERVIEW.md), not to the retained architecture.

```mermaid
flowchart TD
    A[Earlier T2: 71.923768%] -->|historical multi-change gain| B[Retained ARFTR: 85.383648%]
    C[Person-centric appearance and temporal evidence] --> B
    D[Actor memory and contextual evidence] --> B
    E[Anchored factorized residual correction] --> B
    B --> F[702 remaining errors]
    F --> G[Broad correction and sequence trials: rejected]
    F --> H[Crossing verifier: small gain, inconsistent folds]
    F --> I[RGB and native-motion residuals: not promoted]
    I --> J[Matched shared-null training: 85.311396%]
    J --> K[Exact nullness passes; task continuation fails]
    K --> L[Retain ARFTR; close this search cycle]
```

The [curated JSON knowledge graph](../results/arftr_development/knowledge_graph.json)
and [experiment ledger](../results/arftr_development/experiment_ledger.csv) connect
the public result nodes. The local full graph retains 793 nodes and 1,496 edges;
its digest is included in the public graph. The curated graph is not advertised
as the complete experimental history.

## Reading the implementation

| Responsibility | Implementation / contract |
| --- | --- |
| Retained factor residual | [arftr.py](../src/hac/arftr.py) |
| M4 actor-memory anchor | [actor_evidence_memory.py](../src/hac/actor_evidence_memory.py) |
| A3 unrestricted-template expert | [UnrestrictedTemplateControl](../src/hac/sear_controls.py) |
| Retained outer evaluation | [ARFTR runner](../experiments/run_okutama_arftr.py), [protocol](../experiments/okutama_arftr_protocol.json) |
| Motion input representation | [native_motion_data.py](../src/hac/native_motion_data.py) |
| Plain bounded motion residual / loss | [native_motion_innovation.py](../src/hac/native_motion_innovation.py) |
| Shared-null residual and exact retain | [motion_null_contrast.py](../src/hac/motion_null_contrast.py) |
| Final matched experimental contract | [motion-null protocol](../experiments/okutama_motion_null_contrast_protocol.json) |
| Saved-checkpoint replay | [layout-corrected auditor](../experiments/audit_okutama_motion_null_contrast_v2.py) |

## Engineering lessons

1. Improvements in reconstruction, tracking consistency, or calibration proxies
   are not automatically improvements in activity decisions.
2. Complementary information must produce more rescued errors than harmed correct
   predictions, consistently across outer groups.
3. A null invariant is an architectural property, not evidence of task utility.
4. Applying reference subtraction after training is a different experiment from
   training that contrast from initialization.
5. Floating-point layout is part of exact replay. Mathematical equivalence alone
   does not guarantee bit-identical GPU outputs.
6. Any future learned intervention policy must retain the original prediction as
   an available action and learn intervention decisions inside each outer fold.

See the [research overview](RESEARCH_OVERVIEW.md) for numerical qualifications and
the [model card](MODEL_CARD.md) for intended-use boundaries.
