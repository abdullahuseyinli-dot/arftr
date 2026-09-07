# V-COCO continuation protocol — 2026-09-07

## Status and question

This protocol is declared before continuation fitting. Its machine-readable authority
is
[`experiments/vcoco_continuation_protocol.json`](../experiments/vcoco_continuation_protocol.json).
It creates a new dataset-specific contract and does not reuse the incomplete historical
external-CUDA document lock.

The research question is whether uncertainty over plausible anatomy and support
evidence supplies separate posture and motion corrections beyond matched mixed visual
features and ordinary spatial tokens, while returning the established prediction
exactly when new evidence is absent.

## Development role and folds

The historical V-COCO train and validation exports are jointly authorized as reused
development data: 6,640 people in 4,123 source images. Their old source tags remain in
reports but neither is a holdout. Historical official test, calibration, confirmation,
and future replication data are protected and have no role in fitting, selection,
calibration, thresholds, or routing.

All arms use one persisted image-group fold map in the shared pooled-feature row order.
The map has `row_index`, `person_id`, `image_id`, `label_index`, and one outer fold.
For every outer fold `k`, it also has `inner_fold_o{k}` and `stack_fold_o{k}`; outer-held
rows use `-1` for both. It then appends `selection_stack_fold_o{k}_i{j}` for every
outer fold `k` and inner held fold `j`, in outer-major/inner-major order. Such a column
is in `0..2` only for rows in the corresponding outer-train and inner-fit subset
(`outer_fold != k` and `inner_fold_o{k} != j`); every other row uses `-1`. No image may
cross any outer, inner, final-stack, or selection-stack boundary.

The splitter is shuffled `StratifiedGroupKFold`: five outer folds use seed 20260827;
the three inner folds for outer fold `k` use `20260827 + 10000 * (k + 1)`; and its three
stack folds use `20260827 + 100000 + k`. The exact canonical columns appear in the JSON
protocol. Inner-selection stack folds use the fixed seed
`20260827 + 200000 + 1000 * k + j`; they are shared by every family and candidate, so
candidate comparison cannot change its OOF partition. A residual feature, base score,
calibration fit, or model-selection operation may not use an outer-held label.

## V0: resolve the static attribution

V0 uses four already retained 6,640 × 768 development caches: tight and 25%-context
DINOv2 final-CLS features, plus tight and context SigLIP2 pooled features. All four row
maps have SHA-256
`fc0772cdf1f1bce3786cfdf3c9571bcf456cfaa00cb12ea0d26212a7ff0504cb`.
The source lock verifies each declaration, row map, and feature array after the new
protocol authorizes the combined development role. It does not open the historical
test manifest or any test feature.

Every V0 family also receives the same six deterministic geometry values from the
shared locked row sidecar: log box area, log aspect ratio, normalized box center x/y,
log person-pixel height, and normalized distance to the nearest image edge. Their
preprocessing is exactly `hac.vcoco_v3_models.geometry_features`; geometry is held
identical across families and is not counted as a representation-specific advantage.

| Family | Structure | Selection settings |
| --- | --- | ---: |
| Mixed flat | Four inputs, multiclass logistic probability stack | 8 |
| Mixed factorized | Four inputs, posture/motion logistic stack | 8 |
| Mixed factorized reliability | Four inputs, posture/motion stack with reliability | 8 |
| Mixed linear SVM | Four inputs, CUDA primal OVR squared-hinge plus grouped OOF calibration | 8 |

The logistic grid is `component_C ∈ {0.01, 0.1}`, `meta_C ∈ {0.1, 1.0}`, and class
weight in `{none, balanced}`. The SVM grid is `C ∈ {0.001, 0.01, 0.1, 1.0}` and the
same two class weights, with calibrator C=1, 2,000 maximum iterations, and tolerance
`1e-4`. Every calibration-fold and final SVM fit must stop before that ceiling; reaching
it fails closed and cannot produce a completed V0 result. V0 is one deterministic
nested run. Estimator seed is based at 20260907: inner
candidate base seed `20260907 + 10000 * outer_fold + candidate_index`, actual inner-fit
seed `20260907 + 10000 * outer_fold + candidate_index + 10000 * (inner_fold + 1)`, and
final outer-fit seed `20260907 + 100000 + 1000 * outer_fold + family_index` in protocol
order.
Candidates use only source-tag labels on the shared inner folds. Metrics are computed
once on concatenated inner-OOF predictions, not averaged across fold scores, with
fixed zero-division value zero. Candidates rank by macro-F1 descending,
locomotion-class F1 descending, log loss ascending, then stable candidate ID ascending.
Outer-held labels never enter this choice.

V0 determines whether factorization and reliability add value beyond complementary
features and a matched simple classifier. Its result remains reused-development
evidence and cannot establish external generalization.

## V1/V2: spatial correction architecture

V1 remains blocked. Existing caches are pooled representations and contain no DINO
patch grid. The current machine also lacks the rights-safe COCO train/validation source
images and the pinned DINOv2 weights needed for a fresh spatial extraction. The only
retained pose cache is a COCO ground-truth oracle and is ineligible for promotion. No
clean `K=4` predicted-pose hypotheses or auxiliary training-corpus overlap audit exists.

When those inputs are separately acquired and locked, V1 uses frozen
`facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`, 224-pixel inputs,
a 16 × 16 patch grid, 768 dimensions, tight and context views, and float16 storage.
All trained V1 arms use seeds 42–46 and one fixed head optimization configuration.

| Arm | Test |
| --- | --- |
| B0 | Matched established V0 classifier |
| B1 | Fixed-grid tokens at equal token, parameter, pixel, and hypothesis budget |
| B2 | Predicted-joint tokens without anatomy or support relations |
| B3-average | Full relations with fixed averaging over four hypotheses |
| B3-uncertainty | Full relations with separate posture/motion uncertainty weighting |
| Top-one | One pose repeated into the same aggregation layout |
| Anatomy/support deletions | Remove one relation family while matching nodes and capacity |

For head `h`, the invariant is
`z_h = z_h,anchor + m_h * g_h * Σ_k(w_h,k * r_h,k)`. The anchor coefficient is one.
Unavailable or disabled evidence returns the exact anchor. Residual output weights and
biases begin at zero while the available-evidence gate begins nonzero, and tests must
show nonzero residual gradients.

B3-uncertainty is the sole promotion candidate. It must improve macro-F1 over B0 by at
least 0.010, have a positive paired 95% interval and Holm-adjusted whole-image swap
test, and not worsen NLL. A structural or uncertainty claim additionally requires the
corresponding B1, B2, or B3-average contrast to have a positive interval and pass the
same declared family correction.

## Statistics, resources, and locks

All paired intervals use exactly 10,000 whole-image resamples with seed 20260906.
Null tests use 100,000 whole-image probability-vector swaps with plus-one correction.
Confusion counts are aggregated before macro-F1, loss sums before division, and fixed
three-class zero-division semantics apply. V-COCO is a separate multiplicity family
from Okutama.

Before any fitting, one `SeedSequence(20260906)` spawns two PCG64 streams. The first
persists a shared `uint16` `(10000, 4123)` matrix of group-bootstrap indices. The
second persists the shared swap signs as little-endian packed bits in a `uint8`
`(100000, 516)` array, decoded to the first 4,123 bits. A receipt binds the sorted
canonical image-group IDs, NumPy/bit-generator provenance, and both artifact hashes.
The V0 execution lock binds that receipt and both arrays; every declared comparison
must reuse them verbatim.

V0's resource scope begins at the four locked pooled feature caches and ends at the
nested heads. It records bounded outer-fold and complete-run wall time, peak allocated
and reserved CUDA memory, and every CUDA-SVM optimization outcome on the same cohort.
It does not authorize an end-to-end pipeline efficiency claim. Decode, pose and
spatial extraction, cold start, synchronized warm latency, throughput, host memory,
and energy remain deferred until V1's missing inputs exist and can be locked.

The fresh V0 process sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before its first CUDA
query, enables PyTorch deterministic algorithms and deterministic cuDNN behavior, and
disables cuDNN benchmarking. These settings are validated from the committed protocol
before source- or execution-lock comparison.

The source lock is
`.runs/research_20260907/protocol_locks/vcoco_source_lock.json`, with status
`VCOCO_CONTINUATION_SOURCE_LOCKED_BEFORE_V0_FITTING`. It binds clean committed source
blobs, both development manifests, the four pooled cache declarations/maps/arrays, the
complete installed Python distribution set, and CUDA/device details. The source lock
authorizes only construction of the shared fold map and statistics
randomization artifacts; it does not authorize fitting. The V0 execution
lock additionally binds the shared fold map, committed runner, and all three shared
randomization artifacts, has status
`VCOCO_CONTINUATION_EXECUTION_LOCKED_BEFORE_FITTING`, and alone authorizes V0 fitting.
V0 writes metrics, per-class metrics, the complete candidate-selection record, OOF
probabilities, paired statistics, CUDA-SVM optimization audit, fold-usage receipt, and
the final summary beneath `.runs/research_20260907/vcoco_v0`.
V1 cannot be authorized until every missing source, patch grid, pose hypothesis,
overlap audit, and fixed optimization field is present in a new lock.

The prior README encounter with historical aggregate test metrics and the receiving
calibration-manifest incident remain provenance disclosures. Neither metric nor data
enters this protocol. Every current-run protected access counter must be zero.
