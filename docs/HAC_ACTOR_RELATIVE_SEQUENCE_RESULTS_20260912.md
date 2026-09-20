# Actor-relative sequence screen — result

## Decision

**Stop this standalone sequence family.**  It does not offer a safe route to
improve the frozen ARFTR system, so it must not be fused with or used to
overwrite ARFTR predictions.

The 75-fit, five-fold, three-seed OOF screen was completed and independently
replayed.  ARFTR remains the retained model at **85.38% macro-F1**.  The
pre-registered primary arm (actor-relative TCN with the pairwise relation
objective) reached **64.22%**, or **-21.16 points** versus ARFTR.

| Arm | Macro-F1 | Accuracy | Interpretation |
|---|---:|---:|---|
| Frozen ARFTR reference | 85.38% | 85.90% | Retained reference |
| S0: independent all-frame classifier | 68.48% | 68.17% | Adding the frame supervision alone is inadequate |
| S1: unordered all-frame pool | 71.78% | 71.37% | Best new arm; neighboring visual evidence is useful, but still far below ARFTR |
| S2: timestamp TCN | 69.30% | 68.96% | Ordering/TCN is worse than simple pooling |
| S3: actor-relative TCN | 64.16% | 64.01% | Relative representation destroys useful absolute evidence here |
| S4: actor-relative TCN + relation loss | 64.22% | 64.09% | Relation objective recovers only 0.06 points over S3 |

## What the screen established

- The apparent temporal opportunity is **not** captured by this lightweight
  standalone representation: S1 is the best sequence result, while both
  timestamped and actor-relative variants regress.
- Actor-relative centering and pairwise same-action regularization are not
  useful on this cache/training regime.  Treat them as negative evidence,
  rather than as candidates for integration.
- The primary arm produced 140 ARFTR-error rescues but 1,106 harms: a net
  **-966** on the stable-support diagnostic.  Its total OOF net correction was
  **-1,085**, and no diagnostic scenario improved.
- Annotation-derived support strata were used only for post-hoc diagnostics;
  no unavailable annotation signal is an inference feature.

## Integrity

- 75 / 75 fits completed under the locked protocol
  `a704ec01cd91a93698f7d2c7eeb3fef2182bb4c7bae663b5e24ae42fee49eb4b`.
- A separate CPU replay verified all 75 checkpoints and all reported metrics.
- This is adaptive internal evidence, not an external-generalization claim.

## Next move

Do not widen this sequence family or run a fusion sweep.  The next high-value
branch is to work from ARFTR's preserved confidence/error structure: a strictly
nested, OOF-trained *abstaining correction* study that may only change a label
when an independently learned correctness predictor exceeds a high threshold.
It must be benchmarked against ARFTR, include reject/no-change as a valid
outcome, and be abandoned if it cannot make positive net corrections on every
outer fold.  This targets the real residual without allowing the weak sequence
model to corrupt high-confidence correct predictions.
