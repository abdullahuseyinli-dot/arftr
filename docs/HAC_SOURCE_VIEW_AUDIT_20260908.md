# HAC richer-input execution audit

Date: 2026-09-08. Scope: the retained 4,977 development centers and their 21
authorized recordings. Acquisition, alignment and feature pilots use no action
values and read no protected scenario images. A separately authorized, prospectively
frozen source-effect classifier screen follows these audits. Acquisition of the
original training archive is opaque; only authorized recording payloads may
subsequently be decoded.

## Executed results

| Check | Measured result | Meaning |
| --- | --- | --- |
| Official 4K training source | HTTP 200; 14,985,760,603 bytes; range requests supported | The original training videos are accessible through the provider's updated Dropbox link |
| Fixed dense-token pilot | 128 complete long clips, selected by scenario round robin and sample-ID hash | Selection used no action values, predictions, or error status |
| Existing 3x3 reproduction | 128/128 rows bit-exact to retained P3 features | Original crop, checkpoint, encoder and inference recipe reproduced |
| Independent pilot replay | Both 3x3 and 12x12 cache SHA-256 values identical across runs | Finer feature extraction is reproducible on this hardware |
| Batch-one GPU memory | 549,969,920 bytes peak allocated; 671,088,640 reserved | Dense pooling requires approximately 0.512 GiB allocated GPU memory |
| First pilot / replay duration | 37.05 / 41.07 seconds including model loading | Full extraction is practical; runtime varies with concurrent workload |
| Replay median clip time | 0.2491 seconds end to end; 0.0901 seconds forward plus both pooling operations | Disk decoding and preprocessing are a substantial part of extraction cost |
| Fine-to-coarse consistency | Maximum float32 difference 0.0000114441 after pooling 12x12 back to 3x3 | Expected floating-point reduction-order difference, not changed model weights |
| Fine within-coarse-cell variation | Median RMS 0.40378 | The 12x12 cache retains variation discarded by coarse averaging; this label-blind pilot alone does not measure classification value |
| Full dense cache | 4,977/4,977 complete; 467 exact short-input fallbacks; 19.10 minutes | All retained rows now have both controlled 3x3 and 12x12 features |
| Independent full-cache verification | 4,977/4,977 historical 3x3 rows bit-exact; IDs, fallback flags and all source-frame indices verified | No cohort or temporal-input substitution was introduced |
| Full 12x12 storage | 8,806,662,272 bytes | Approximately 8.20 GiB; classification experiment can now consume the cache |
| Original 4K acquisition | Complete 14,985,760,603-byte archive; 21 permitted videos extracted and CRC/SHA checked | Only retained development recordings were decoded |
| Native-source fidelity | 128 center images; median native actor height 107 pixels; median detail-versus-exact-downsample MSE 129.63 | Measurable native image detail exists; no classification improvement established |
| Initial 4K timing audit | Best local offsets: 0 for 82 rows, -2 for 41, -1 for 2, +1 for 1, +2 for 2 | A naive equal-frame-index substitution fails; wider disjoint-fit/validation timing audit is required |
| Wider timing search | 125 distinct physical frames: 64 fit, 61 validation, including 12 label-blind supplemental views | Eight recordings have constant native/JPEG timeline offsets; frame-rate resampling is not the selected explanation |
| Fresh timing validation | 63 new disjoint early/middle/late frames; all 21 views pass; maximum JPEG/native MAE 1.755 at 320-pixel audit scale | Constant-offset rules generalize to fresh frames and agree with true PyAV timestamps |
| Exact native64 input gate | 64 centers, 7 historical short fallbacks, 1,024/1,024 frames pass; maximum MAE 1.791 | Every input frame was checked before model loading; physical crop boundaries match the existing720 arm |
| Three-arm native64 feature pilot | 64/64 existing720 coarse features bit-exact; native4K and exact-native-downsample controls cached at both grids | The native-resolution comparison is executable and controlled; the pilot itself trained no classifier |
| Native64 resource cost | 175.13 seconds CPU preparation; 47.38 seconds all three encoder arms; 565,174,272 bytes peak allocated GPU memory | Small-head jobs overlapped by explicit scheduling, so this is not an isolated hardware benchmark |
| Native feature change | Median coarse-feature cosine 0.99370 versus exact downsample; 0.98095 versus supplied720 | Both resolution and source/codec differences matter; feature change is not classification gain |
| Full-cohort native support | Default decoder supports 4,948 complete windows; 29 need whole-clip existing720 fallback | All 4,977 centers can be retained without fabricated, repeated or interpolated native frames |
| Edit-list metadata alternative | Ignoring edit lists exposes complete packet support for 4,965 centers; 12 windows still exceed native tails | Potentially recovers 17 windows, but this alternate decoding path is not yet pixel-certified |
| Shared chronological source work | 79,168 native actor/frame uses reduce to 14,463 unique source frames and 75,178 unique actor crops | One decoded native frame can serve multiple actors; all 29 whole-clip720 fallbacks remain unchanged |
| Bounded chronological benchmark | Two median-workload permitted views; 3,554 frames decoded, 1,462 required images pixel-checked, 6,561 actor crops shared | 1.1.10 took 69.52 seconds and 2.2.5 took 27.82 seconds; no full-archive decode sweep |
| Uint8 intermediate footprint | 5,639,225,040 bytes actual for both full-cohort crop arms | Approximately 5.25 GiB instead of approximately 280 GB float32 staging |
| Independent optimized-input replay | Ten tensors across five intersecting native64 clips bit-exact, maximum difference zero | Chronological decoding and uint8 storage preserve the earlier controlled pilot inputs |
| Full paired source extraction | All 4,977 centers; 4,948 native windows and 29 exact720 whole-clip fallbacks | Both native4K and exact-native-downsample are available at 3x3 and 12x12 grids |
| Full paired runtime | 958.62 seconds crops plus 1,685.80 seconds encoding = 44.07 minutes, excluding final hashing | Optimized full extraction met the bounded resource estimate under concurrent small-head workload |
| Full independent source verification | Ten final array hashes verified; all 128 pilot source-arm checks exact; 29 fallbacks exact at both grids | No timing-rule change, cohort loss or altered control values |
| Paired-view metadata | 10 paired scenarios; 3,676 rows at 327 shared scenario/frame keys | These are candidate matching opportunities, not verified actor pairs |
| Fixed paired-view geometry pilot | 30 scene pairs, 60 image decodes, 10 scenarios | Same-index scene geometry was tested without action labels |
| Geometry feasibility | 6 pairs passed the fixed scene-match gate; 13 failed held-match quality; 11 had insufficient matches | Ordinary ORB/homography matching is not a reliable automatic cross-view alignment solution here |
| Actor proposals | 6 unverified proposals; one track-pair proposal recurred at two times | No actor correspondence or frame synchronization is certified for training |

The full 4,977-row 12x12 cache and 128-center original-4K fidelity audit are
complete as separate measurements. The 720p dense cache never substitutes
4K pixels, so the separate source-resolution comparison remains interpretable.
The 4K timing gate and bounded matched-feature resource pilot are now complete.
The search initially used the first 64 pilot rows for fitting and disjoint physical
frames from the remaining 64 for validation. Duplicate actor centers do not count
as independent physical images; label-independent extra centers supplied sparse
views. One two-frame fit spuriously preferred a 29.97/30 slope by MAE 0.0101 and
failed validation. The revised fit-only rule requires a nonunit timebase to improve
fit MAE by at least 0.05; it selected constant offsets for all views. This rule was
frozen before 63 fresh early/middle/late images were accessed, and all 21 views
passed that new validation. Pixel alignment, not action labels, determined the map.

The first cross-decoder check was overly strict about RGB equality: OpenCV and
PyAV conversions differ by roughly subpixel intensity levels. That failed receipt
is retained. The corrected cross-decoder control requires the true-PTS image to
be the nearest nearby OpenCV frame (excess MAE below 0.05, absolute MAE below 2).
All 63 true-PTS images had zero excess over the nearest candidate; maximum absolute
cross-decoder MAE was 1.260. This is within-recording timing verification, not
cross-drone synchronization.

Default decoder native index is `JPEG source frame + offset`:

| Recording | Offset |
| --- | ---: |
| 1.2.11 | -12 |
| 1.2.5 | -2 |
| 1.2.7 | -9 |
| 2.1.5 | -26 |
| 2.2.11 | -23 |
| 2.2.5 | -21 |
| 2.2.7 | -26 |
| 2.2.8 | -2 |
| Remaining 13 permitted recordings | 0 |

Packet timestamps identify MP4 edit lists as the source of these offsets. For
example, 1.2.11 contains twelve negative-PTS packets; ordinary decoding hides them.
The [FFmpeg MOV demuxer documentation](https://ffmpeg.org/ffmpeg-formats.html#mov_002fmp4_002f3gp)
documents the option to ignore edit lists. A metadata-only comparison of default
and ignored-edit-list packet timelines was run for all 21 permitted recordings.
The safer immediately verified full-cache recipe is **4,948 native clips plus 29
whole-clip existing720 fallbacks**, retaining the exact 4,977 IDs and temporal inputs.

## Exact artifacts and commands

Metadata and the fixed pilot selection:
`.runs/research_20260908/source_view_audit/`.

The provider's page source was retained as `provider_index.md`, with SHA-256
`1cdf68b29c279040355a9758bb13f0333c11f38ffb28da01be587c3ef82159a7`.
The 128-ID selection SHA-256 is
`c512ba92d0db798b8ebd56d3e9c65889c8743fcff69a3f89f027010ec59c75f6`.

The independent dense replay, including exact executable source snapshots, is in
`.runs/research_20260908/dense_token_pilot_replay/`:

- `tokens_grid3.npy`: 14,155,904 bytes;
  `ab00a3261c248f10a85b4142560c213383359c4aca116825e8e9378e206660d6`.
- `tokens_grid12.npy`: 226,492,544 bytes;
  `b32100827ff142d30cc9edab43e23df3948119f07759f8760bc4d138e2e5139a`.

The paired-view pilot is in `.runs/research_20260908/paired_view_pilot/`.
Its fixed 30-pair selection hash is
`cda6cbf8b3c058b25ebf76dbdb4e298d2440c21895e0595dabca57bb546111c6`.

The full dense cache is in `.runs/research_20260908/dense_tokens_full/`, with
an independent `verification.json`. The 12x12 feature SHA-256 is
`f97d582ddfe0fcc42a7476aa1af60a6317d427bccf3ffcefb27dbb19e9e85f97`;
the bit-exact 3x3 control SHA-256 is
`da6af8f2fd879fe06dee36e9991f499ab6f6fac48b44f0f5679538d8d64687c5`.
Full extraction took 1,145.81 seconds, with median end-to-end clip time 0.22177
seconds and median forward-plus-pooling time 0.08445 seconds. Peak GPU memory
was unchanged from the bounded pilot. Exact executable snapshots are retained;
the later CPU-only verification mode was added after extraction began.

The original-source fidelity receipts and twelve preselected detail panels are
in `.runs/research_20260908/source_4k_fidelity/`. Wider timebase-search evidence
is in `.runs/research_20260908/source_4k_alignment_r1/`; the first attempt is retained
in `source_4k_alignment/` and stopped on an overestimated container frame count.
Fresh successful validation is in `source_4k_alignment_validation_r1/`.

The bounded matched-resolution pilot is in
`.runs/research_20260908/native4k_matched_pilot/`:

- `pixel_gate.json` binds all 1,024 exact image checks and hashes prepared tensors.
- `features/summary.json` reports the three-arm resource measurements.
- `features/native4k_grid12.npy` SHA-256:
  `f4826bff286cc3d62bd159b1844ea7fcc299f736de832eea05348202a04f8fb3`.
- `features/native4k_grid3.npy` SHA-256:
  `fbe0346b9ca11d725b13adb231dd64c56151ee40f3d1773014555ed1139792d3`.
- `features/request.json` records concurrent small-head jobs and all prepared-input
  hashes. No classifier was fitted and no DINO native-resolution pilot was run.

The all-window support census is in
`.runs/research_20260908/native4k_support_census/summary.json`. It reads packet
timestamps only, uses exact P3/P0 fallback support, and decodes no images. The 12
windows not covered even by the unedited timeline comprise ten short-fallback
centers and two long-input centers; native clips are never silently shortened.

The bounded chronological cache benchmark is in
`.runs/research_20260908/native4k_streaming_benchmark/`, with an independent
`verification.json`. Selection was committed from metadata before decoding:
the median unique-crop workload view among zero-offset recordings (1.1.10), and
the corresponding median among nonzero-offset recordings (2.2.5). Both default
decoder offsets remain unchanged. Every required image was compared to its exact
allowed JPEG; maximum image MAE was 1.806/255 at the 320-pixel audit scale.
Native and exact-downsample crops are stored as raw uint8 shards, with explicit
offsets and shapes, and are read back through the unchanged V-JEPA preprocessing.
Thirty-two fixed sample-ID-hash clips supplied the preprocessing timing pilot.

Metadata counts across all 4,977 exact short/long windows are:

- 4,948 native windows and 29 whole-clip existing720 fallbacks.
- 79,168 native actor/frame uses, 75,178 distinct actor crops and 14,463 unique
  requested video frames; total chronological decode span is 34,281 frames.
- 5,075,302,536 native uint8 crop bytes plus 563,922,504 exact-downsample crop bytes.
- Two native/downsample feature arms at both grids would add approximately
  18.7 GB, giving approximately 24.3 GB total new crop-plus-feature storage.

Weighted extrapolation of the two-view measurements estimates 987.78 seconds
(16.5 minutes) for full crop building, and 430.67 seconds (7.2 minutes) for reading
and preprocessing both crop arms. The previously measured 0.08445-second
V-JEPA forward-plus-pooling median projects approximately 13.9 minutes for two
new encoder arms on 4,948 native windows. The sum is approximately 37.6 minutes
before overhead; **40–55 minutes is a plausible, unverified full-run estimate**.
This is not a completed full extraction or an exclusive-hardware benchmark.

```powershell
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode prepare --output-dir .runs/research_20260908/source_view_audit
.\.venv\Scripts\python.exe experiments/pilot_okutama_dense_tokens.py --selection .runs/research_20260908/source_view_audit/pilot_selection.json --output-dir .runs/research_20260908/dense_token_pilot_replay
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode paired --output-dir .runs/research_20260908/paired_view_pilot
.\.venv\Scripts\python.exe experiments/pilot_okutama_dense_tokens.py --mode full --output-dir .runs/research_20260908/dense_tokens_full
.\.venv\Scripts\python.exe experiments/pilot_okutama_dense_tokens.py --mode verify --output-dir .runs/research_20260908/dense_tokens_full
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode align --output-dir .runs/research_20260908/source_4k_alignment --selection .runs/research_20260908/source_view_audit/pilot_selection.json --fidelity-summary .runs/research_20260908/source_4k_fidelity/summary.json
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode validate_alignment --output-dir .runs/research_20260908/source_4k_alignment_validation_r1 --alignment-search .runs/research_20260908/source_4k_alignment_r1/summary.json
.\.venv\Scripts\python.exe experiments/pilot_okutama_dense_tokens.py --mode native_prepare --output-dir .runs/research_20260908/native4k_matched_pilot --alignment-summary .runs/research_20260908/source_4k_alignment_validation_r1/summary.json --fidelity-summary .runs/research_20260908/source_4k_fidelity/summary.json
.\.venv\Scripts\python.exe experiments/pilot_okutama_dense_tokens.py --mode native_encode --output-dir .runs/research_20260908/native4k_matched_pilot
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode support_census --output-dir .runs/research_20260908/native4k_support_census --alignment-search .runs/research_20260908/source_4k_alignment_validation_r1/summary.json --fidelity-summary .runs/research_20260908/source_4k_fidelity/summary.json
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode streaming_benchmark --output-dir .runs/research_20260908/native4k_streaming_benchmark --alignment-search .runs/research_20260908/source_4k_alignment_validation_r1/summary.json --fidelity-summary .runs/research_20260908/source_4k_fidelity/summary.json --support-census .runs/research_20260908/native4k_support_census/summary.json
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode streaming_verify --output-dir .runs/research_20260908/native4k_streaming_benchmark
```

Use a fresh output directory for any independent replay; existing evidence is never
overwritten. The full utility records identity order, exact source frame numbers,
the 467 short-input fallback choices, progress, cached-feature reproduction, and
source snapshots. Incomplete long clips retain the original sixteen short frames.

## Source identity and access terms

The first-party [Okutama repository](https://github.com/miquelmarti/Okutama-Action)
and its [website source](https://github.com/miquelmarti/Okutama-Action/blob/master/index.md)
link the original 4K training archive and declare CC BY-NC-SA 3.0. They explicitly
describe two-drone scenarios and locally assigned tracking IDs.

The working [training archive link](https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/ADtwW9gmCdlhrvyaY-grV3A/TrainSetVideos.zip?dl=1&e=1&rlkey=7u7131amaul29amyr4jbnnu03)
was resolved from that provider repository. No test archive was requested. Download
resides under `C:\Users\DELL\hac_external_data\OkutamaAction\source4k`.
An acquisition SHA-256 is an observed digest, not a publisher-signed checksum.
The complete original archive SHA-256 is
`458d6cd705c452762db63f04d338c950788b704bdd4b3f0826072c034dbb0547`.

## Decision

Proceed with the 12x12 actor-local representation comparison. The pilot establishes
low enough extraction cost and exact baseline reproduction; it does not establish
a classification improvement. Native image detail and controlled feature changes
are established. The completed benchmark led to authorization of the full paired
extraction described below. The initial random-seek,
float32-staging design projected over four hours for paired arms. The newly executed
chronological shared-frame/uint8 benchmark reduces the estimate to approximately
40–55 minutes and 24.3 GB of new crops plus features, making a fully controlled
native/downsample trial practical within roughly one hour if the extrapolation holds.
Use that optimized path, not the original float32 staging. Preserve all centers with recorded
whole-clip existing720 fallback for absent native support.

## Authorized full paired run and frozen source screen

The full paired feature run is complete under
`.runs/research_20260908/native4k_paired_full/`. It used the same default decoder,
immutable offset map, chronological crop reuse, per-required-image pixel gate,
and 29 exact720 whole-clip fallbacks. All 14,463 required source images passed
the pixel gate (maximum 320-wide RGB MAE 1.82594), producing 75,178 shared crop
pairs in 958.62 seconds. Paired feature extraction for all 4,977 centers took
1,685.80 seconds including model load: 44.07 minutes combined, excluding final
hashing. File timestamps give 46 minutes 35 seconds from the retained crop-launch
snapshot to the final feature summary, including intermediate verification and
hashing gaps. This overlapped the team's small GPU heads; these are measured workload
times, not an isolated-GPU benchmark. Peak own allocation was 565,174,272 bytes
(0.526 GiB), reserved memory 700,448,768 bytes (0.652 GiB).

All 128 pilot source-arm comparisons reproduced both grids exactly. A separate
read-only verification recomputed all ten final array hashes, confirmed 4,977
unique IDs in historical order and complete flags, checked the explicit 29 source
and 467 historical-short fallback masks, and verified both grids of both source
arms equal existing720 exactly on every source-fallback row. The receipt is
`features/independent_post_extraction_verification.json`; the completed feature
summary SHA-256 is
`7fc53483f1df3478b77ae2e015294b4f780645845e40c26543289991957a1dce`.
The CPU source-effect screen below is complete. Its measured classification
results distinguish a source/rendering effect from a native-resolution effect.

Before any new accuracy, a small source-effect protocol was frozen at
`.runs/research_20260908/native4k_source_effect/protocol.json`, SHA-256
`e62c30fa2687f8762eefed32353a1c3f64ba9994f1c3037a94f3ad3601afd546`.
It tests all three fixed sources (existing720, native4K, exact-native-downsample)
with two fixed heads: 768-dimensional global mean and 7,680-dimensional global
mean plus time-averaged 3x3 spatial contrasts. It retains the original five outer
scenario folds, three inner grouped folds, four C values (1e-5, 1e-4, 1e-3, 1e-2),
balanced multinomial logistic regression, training-only standardization and the
original convergence ceiling. Every predeclared arm is reported; no source/head
selection on outer labels, learned OOF fusion, DINO extraction or duration change.

The two new sources require 260 estimator fits. Independently replaying both
existing720 controls adds 130, for 390 maximum total; this arithmetic is explicitly
recorded instead of treating control fits as free. This is a source-effect test,
not a new-architecture claim. GPU extraction is batch-one with a 1 GiB budget;
the small linear screen is CPU-only after completed feature verification.

Executable provenance is separately bound. The active paired encoder retained
`features/pilot_okutama_dense_tokens.py` at 02:13:21 UTC, before its request at
02:13:31 and model receipt at 02:13:38. That encoder snapshot SHA-256 is
`0b184fb45082a006a40a86350f476d83bbbdcee12de221343c4c6a5dc29d5533`.
The working script was subsequently changed at 02:16:19 UTC to harden only
`run_source_effect` resumption. An AST comparison confirmed every other function,
all imports and the remaining module body were unchanged. The later file is not
misrepresented as the encoder's executed source; `features/execution_source_binding.json`
binds the retained snapshot and original request without modifying either.

The crop phase likewise retained its launch snapshot at 01:55:24 UTC, SHA-256
`0c9760f76fe5a81b2a6ded468e6c793c8bb01d6ebcb94050289322166f51d1cc`.
The later 02:05:50 audit-script change affected only `verify_streaming_cache`,
adding metadata hash binding to the independent replay receipt. Every other AST
node, including the active chronological extraction function, remained equal.
`crop_execution_source_binding.json` distinguishes the executed crop snapshot
from the subsequently strengthened verifier; no crop recipe was changed.

The CPU classifier is frozen separately at SHA-256
`854b20f32d4918da7def5a7580871c126906702363a59d7f1813051526dc9b80`,
with `classifier_execution_lock.json` binding the protocol and imported helpers.
Its launcher will verify and execute those exact bytes. A pre-fit integrity
addendum requires existing720 global-mean probabilities to reproduce the historical
P3 arm within 1e-8 after checking exact IDs, labels, scenario IDs and folds; a
mismatch fails closed. Completed per-fold checkpoints can be resumed only after
request, source snapshot, checkpoint hash and partition identity checks.

```powershell
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode streaming_full --output-dir .runs/research_20260908/native4k_paired_full --alignment-search .runs/research_20260908/source_4k_alignment_validation_r1/summary.json --fidelity-summary .runs/research_20260908/source_4k_fidelity/summary.json --support-census .runs/research_20260908/native4k_support_census/summary.json
.\.venv\Scripts\python.exe experiments/audit_okutama_source_views.py --mode streaming_verify --output-dir .runs/research_20260908/native4k_paired_full
.\.venv\Scripts\python.exe experiments/pilot_okutama_dense_tokens.py --mode native_full_encode --output-dir .runs/research_20260908/native4k_paired_full
.\.venv\Scripts\python.exe .runs/research_20260908/native4k_source_effect/launch_frozen_classifier.py
```

## Completed fixed source-effect classification

All 390 nested logistic estimators completed in 950.48 seconds (15.84 minutes),
without convergence failures. Outer scores were calculated only after all six
predeclared arms finished. The historical existing720 global-mean probabilities
reproduced P3 **bit-exactly**, maximum absolute difference zero. An independent
read-only check using scikit-learn recomputed all six macro-F1/accuracy values,
all six paired rescue/harm counts, exact historical identities and scenario
partitions. No model, source or head was selected or fused using these outcomes.

| Fixed source | Global mean macro-F1 | Global + spatial contrast macro-F1 | Global mean accuracy | Spatial-contrast accuracy |
| --- | ---: | ---: | ---: | ---: |
| Supplied existing720 | 80.4254% | 79.2378% | 81.4145% | 80.4501% |
| Native4K | 82.1686% | 81.7266% | 82.7205% | 82.2785% |
| Exact-native-downsample720 | 82.5642% | 82.0302% | 83.1425% | 82.5598% |

| Predeclared paired comparison | Head | Macro-F1 change (percentage points) | Errors corrected | Previously correct rows harmed |
| --- | --- | ---: | ---: | ---: |
| Native4K minus exact-native-downsample | Global mean | -0.3956 | 144 | 165 |
| Native4K minus exact-native-downsample | Spatial contrast | -0.3036 | 137 | 151 |
| Native4K minus supplied existing720 | Global mean | +1.7432 | 274 | 209 |
| Native4K minus supplied existing720 | Spatial contrast | +2.4888 | 283 | 192 |
| Exact-native-downsample minus supplied existing720 | Global mean | +2.1389 | 239 | 153 |
| Exact-native-downsample minus supplied existing720 | Spatial contrast | +2.7924 | 252 | 147 |

The useful finding is **not that extra resolution improved this model**. In both
fixed heads, the controlled native-resolution contrast was slightly negative.
Instead, regenerating even a 720p input from the original source improved over
the supplied JPEG path. This supports a source/rendering-fidelity hypothesis;
it does not isolate codec, color conversion, resampling or their interaction.
Time-averaged spatial contrasts also underperformed global mean for every source,
so this simple extra spatial feature block is not a gain-producing architecture.

For global mean, regenerated downsample improved sitting F1 from 76.8106% to
80.4290%, standing from 79.1094% to 81.0544%, and walking/running from 85.3560%
to 86.2094%. The largest outer-fold improvement was fold 3 (scenarios 2.7 and 2.8),
from 67.7676% to 76.0597% macro-F1; all five folds improved for this particular
fixed comparison. These are evaluation summaries, not newly tuned routing rules.
The 29 source-fallback rows use identical features but can still receive different
predictions because each source arm fits different training-population weights.

The next defensible source-axis test is to separate JPEG/rendering effects from
spatial detail with a fixed recompression/resampling control, then cross a fixed
source recipe with the team's temporal architecture on the same folds. Do not
claim a new best system from this source screen alone: its 82.5642% is a standalone
frozen global-mean probe, not an automatically combined successor to the existing
multicomponent baseline. Richer 12x12 caches for both source arms are available
for a separately declared architecture comparison.

Results: `.runs/research_20260908/native4k_source_effect/results/summary.json`,
SHA-256 `796412e96290db8576298e1811a7b37d7fa2d813b32c963220a3926efe492fb1`.
The complete per-row probabilities, IDs, labels, scenarios and folds are in
`oof_probabilities.npz`, SHA-256
`d00b1a89d993ba561e3435a03ea86eccb0d60d1a30ab7d3f247973bb861c4ae6`.
`historical_replay_verification.json` and `independent_result_verification.json`
record the replay and independent arithmetic checks. Per-fold checkpoints retain
the grouped inner selection evidence and fitted model/scaler arrays.

Keep paired-view supervision behind
actor identity and synchronization verification, because ordinary scene matching
failed on most of this fixed pilot.
