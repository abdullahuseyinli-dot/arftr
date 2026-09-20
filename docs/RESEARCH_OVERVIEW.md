# Research overview

Status: development search closed; retained result unchanged. Package version
`3.1.0.dev0` is an unreleased working revision, not a published release or DOI.

## Distinct evaluation tracks

| Track | Result | Evaluation boundary |
| --- | --- | --- |
| POLAR v1 | 94.0% macro-F1 | Source-audited four-class held-out test, 3,329 images |
| V-COCO v2 | 86.63% macro-F1 | Person-level official test, 6,077 people |
| Temporal v3 | 78.54% macro-F1 | Separate Okutama confirmation, 1,771 examples |
| ARFTR continuation | 85.383648% macro-F1 | Adaptive five-fold development, 4,977 centers / 11 scenarios |

These numbers are **not comparable rankings**: populations, labels, modalities and
selection histories differ. The [documentation index](README.md) links the original
reports for the first three tracks. The final track is backed by the
[portable metric export](../results/arftr_development/metrics.json).

## What the continuation established

The saved earliest T2 result was 71.923768% macro-F1; retained ARFTR reached
85.383648%, a **+13.459880 percentage-point historical gain**. This is a sequence of
data/representation, evidence fusion, actor-memory and factor-residual changes,
not an isolated causal estimate of one invention. The consolidated result ledger
records the same development population and explicitly distinguishes raw,
routed, and post-hoc results.

ARFTR leaves 702 errors and remains the retained system. Later point estimates can
be slightly higher without satisfying acceptance criteria: the crossing verifier
reached 85.611648%, but its fold nets were +9, +2, -1, 0, 0. It was not promoted.
Additional RGB and motion correction trials likewise failed to establish a robust
improvement. “Retained” therefore does not mean “highest exploratory point estimate.”
The [original architecture review](HAC_NEXT_ARCHITECTURE_MAP_20260912.md) also notes
that ARFTR's last incremental gain did not pass its strict scenario-significance
screen. Retention is a documented development decision, not a claim of statistically
established superiority for every component.

## Final optional mechanism test

The final matched study compared a plain motion residual with a shared-head
reference subtraction trained from initialization:

`delta = 0.5 * tanh(h(a, b, g) - h(a, 0, g))`

Both arms used the same five folds, seed 42, data, ancestry-safe priors, minibatch
schedule, 256 updates, optimizer and training objective. The paired arm enforces
zero residual when the explicit difference carrier is zero. Ordered means and
geometry still carry temporal information; the reference is not motion-free.

| System | Macro-F1 | Errors | Rescue / harm versus ARFTR |
| --- | ---: | ---: | ---: |
| Retained ARFTR | 85.383648% | 702 | — |
| Matched plain control | 85.467181% | 698 | 40 / 36 |
| Trained paired-null | 85.311396% | 707 | 40 / 45 |

The new arm lost 0.072252 pp versus ARFTR; its 95% scenario-bootstrap interval was
[-0.279515, +0.148178] pp. Fold nets were +1, +1, -1, -2, -4. NLL and Brier worsened.
It failed the fixed continuation gates. This is a negative result for the tested
recipe, not proof that all future motion models are unhelpful.

Numerical-only control exceptions were approved before opening new outcome scores;
class predictions, inputs and provenance still had to match. All ten saved
checkpoint outputs subsequently replayed bit-for-bit after correcting the audit's
reference tensor memory layout. No checkpoint was retrained or selected. The
separately written replay was by the same author, not an independent-person audit.

## Claims and limits

- No oracle score is presented as deployable performance.
- Annotation-derived support categories are diagnostic-only, not routing inputs.
- No outer-held labels trained a correction router in the final study; no router
  was fitted at all.
- The development set was repeatedly reused. Confidence intervals do not remove
  adaptive-selection bias or establish deployment performance.
- The final matched test used one seed. Neither seed robustness nor a universal
  architecture ceiling has been established.
- The public aggregate replay verifies arithmetic and gates, not feature extraction,
  optimizer execution, or the full transitive ancestry of private caches.

The evidence supports closing this search cycle, preserving the successful system
and unsuccessful branches, and making the work reproducible. Reopening requires a
new justified protocol, not a retrospective threshold change.
