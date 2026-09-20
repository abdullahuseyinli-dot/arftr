# Center-episode expectation: bounded evidence borrowing

Status: exploratory prototype and synthetic CPU tests only. No classifier was fit,
no live source was changed, and no classification improvement is claimed. The
trigger is the exposed M4-versus-M3 diagnostic: +0.482 macro-F1 points, but sampled
boundary harms increased from 75 to 83. This suggests testing whether usefulness
can overwhelm the survival prior; it does not prove that this is the only cause.

## Operator and bound

For each actual edge into observed node `j`, predict a hazard rate from a separate
small projection of the two frozen node features. Do not give this branch center
coordinates, base probabilities or contextual utility features. Convert rate to
boundary probability using elapsed physical time:

```text
m_j = softplus(hazard(endpoint_features)) * dt_j
s_j = exp(-m_j)                  # edge continuation probability
h_j = 1 - s_j                    # edge stopping probability
S(c,j) = product of s on the path from center c to node j
```

Missing slots are absent observations: the next observed edge bridges the gap.
The first observed node has no incoming edge. With at most five observations there
are at most nine contiguous observed segments `[l,r]` containing the center.

```text
P(left_endpoint=l)  = S(c,l) * h_l       # replace h_l by 1 at observed left edge
P(right_endpoint=r) = S(c,r) * h_next(r) # replace h_next by 1 at observed right edge
P(segment=[l,r]) = P(left=l) * P(right=r)
a_j = sum over segments containing j of P(segment) * softmax(utility within segment)_j
p_new = (1-g)*p_center + g*sum_j a_j*p_j
```

Each within-segment coefficient is at most one. Consequently:

```text
a_j <= sum_{segments containing j} P(segment) = S(c,j).
```

The final non-center probability coefficient `g*a_j` is bounded by the same value.
In contrast, `softmax(utility + log(S))` can put almost all mass on an arbitrarily
useful node even when `S` is tiny. The synthetic extreme test reaches attention
`exp(-5) = 0.006738` in the candidate while the matched control approaches one.

This is a bound against the *predicted* survival, not a guarantee that an edge is
truly state-continuous. False-positive boundaries can suppress useful evidence;
false-negative boundaries can still propagate wrong states. Edge independence is
an approximation. Contextual utility and the gate still see all permitted nodes,
so this is not strict feature-information isolation or a causal-influence bound.
It remains an offline, overlapping-clip, two-sided memory with unchanged pixels
and raw-frame support, not an online forecasting method.

## Matched comparison and learning

The two new arms have identical module shapes and **849,059 parameters** at the
original 3,072-dimensional visual-only input. Six quality fields remain diagnostics
only, matching M2-M5. Only the attention operator differs:

| Arm | Attention |
| --- | --- |
| Recalibrated control | `softmax(utility + detached_log_survival)` |
| Center-episode expectation | Segment-normalized utility marginalized over detached segment probabilities |

Both use the same independently projected hazards and unweighted proper boundary
BCE, masked to fully known training intervals. Classification gradients cannot
reach hazards, and boundary gradients cannot reach utility/gating or frozen input
features. Do **not** jointly clip all parameter gradients: clip the hazard and
utility/gate groups separately, otherwise classification magnitude affects hazard
updates despite graph detachment. Proper BCE alone does not ensure calibration;
report held boundary NLL, Brier and reliability by interval duration. Repeated
physical edges appear in several center windows: disclose occurrence-weighted and
deduplicated evaluation rather than treating them as independent evidence.

Keep the existing loss scales, gate initialization, nested grouping, four optimizer
settings and three outer seeds. Compare the two new arms directly; their difference
isolates marginalization more closely than either arm versus old M4, which also
changes hazard inputs, supervision weighting and gradient routing.

## Budget and next gate

The proposed matrix is `2 * 5 * (4 * 3 + 3) = 150` neural fits. Reuse the existing
deeply nested base cache only after exact population/configuration/provenance hash
checks; no new encoder extraction or base fit should be needed on matching inputs.
Completed M3 and M4 each required 75 fits, totaling about 1,053 seconds of recorded
fit time. With similar early stopping, budget approximately 25-45 minutes for the
new pair, but reserve up to 2.5 hours if fits run all 30 epochs. These are estimates,
not measurements of the new model. A synthetic GPU resource check and independent
runner review are required before a new execution lock and any classifier training.

The executable proposal is
[okutama_center_episode_protocol.json](../experiments/okutama_center_episode_protocol.json).
The prototype is [center_episode_memory.py](../src/hac/center_episode_memory.py);
[17 synthetic tests](../tests/test_center_episode_memory.py) pass. Tests cover all
80 valid mask/center combinations, normalized distributions, exact segment
membership marginals, missing-slot bridges, extreme usefulness, float64 singleton
and zero-gate fallback, finite extreme hazards, separate gradients, parameter
matching, actual-time scaling and unchanged physical-edge hazards after recentering
or shifting the same observations into different slots. No future result should
be tuned by changing the radius, hazard temperature or threshold on exposed OOF.

## Relation to existing ideas

Expected monotonic alignment is established in
[Raffel et al., monotonic attention](https://proceedings.mlr.press/v70/raffel17a.html).
Soft attention inside selected chunks is established in
[Chiu and Raffel, MoChA](https://arxiv.org/abs/1712.05382).
Marginalizing latent structured attention is established in
[Kim et al., Structured Attention Networks](https://arxiv.org/abs/1702.00887).
This HAC experiment tests a specific two-sided center-containing segment
distribution, supervised physical-duration hazards and a probability-contribution
bound. It does not claim invention of structured, latent-segment or chunkwise
attention generally, and global novelty has not been established.
