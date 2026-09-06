# Okutama scenario grouping: source-lineage review, 2026-09-06

The 11 CPTR OOF groups named `recording_id` are **scenario identifiers in the
historical generation code**, not separate drone-video identifiers. The evidence
below supports that interpretation through matching source and summary hashes. This
review did not reconstruct the original row-level video-to-scenario mapping, replay
the data pipeline, or establish that different scenarios are statistically
independent. It is a provenance correction, not a new protocol or execution lock.

## Source semantics

The historical [development auditor](../tools/audit_okutama_development.py#L76)
parses an original video identifier as `drone.part_of_day.scenario` and returns
`part_of_day.scenario` as `scenario_id`, removing the drone component. Its audited
source bytes match the current file. The auditor invokes that mapping when
[constructing development rows](../tools/audit_okutama_development.py#L229).

The hash-matched [feature extractor](../experiments/cache_okutama_temporal_features.py#L401)
preserves the original identifier in `video_id`, assigns
`recording_id = scenario_id`, and constructs track identifiers from the original
video and provider track. Thus synchronized drone views of one declared scenario
receive the same downstream `recording_id`. The extractor explicitly reports
[scenario and video counts separately](../experiments/cache_okutama_temporal_features.py#L421).

The current [split builder](../tools/build_vcoco_v3_temporal_split.py#L126) preserves
one split per `recording_id` and track. Its source is recorded below for inspection,
but the historical split receipt does not bind that builder's source hash; this
review does not upgrade that omission into verified execution provenance.

## Observed file hashes

These SHA-256 values were recomputed from the source or JSON files themselves.
Paths are relative to the repository root. The JSON files contain provenance and
aggregate summaries; no metadata CSV or feature/prediction array was opened for
this lineage review.

| File | SHA-256 |
| --- | --- |
| `tools/audit_okutama_development.py` | `d8f533c485f87efb90b538f89aa3083567ce1e44636445e164b28cf430b16303` |
| `experiments/cache_okutama_temporal_features.py` | `6f860d2cbdf129b64679e87bff29817eb63b35b72c59055c4d6a0e262f0e3654` |
| `tools/build_vcoco_v3_temporal_split.py` | `42d4b28faa447a89fb7731ac15cf3566aa725cb77ae85a98a766dc9341070811` |
| `.runs/vcoco_v3/okutama/development_audit/summary.json` | `212942e26b2a66df46435b770458adbd6da7632a9c3ba7b8dce51db25cf6af0f` |
| `.runs/vcoco_v3/okutama/features/dinov2_base/summary.json` | `eba4490a9c31f9235d388eb631285821e575cad782fbfb17aaace5bcdc0d5a05` |
| `.runs/vcoco_v3/temporal/development_manifest.provenance.json` | `dee0abfe1a4f3c5f273b12fd32f2f974078aa061a7699b319e5f0a02d2d4bbc1` |
| `.runs/vcoco_v3/temporal/temporal_manifest_lock.json` | `47f956f99b291bc93c4825c45cc1f4e295cf92a1b437cf5b0e9f1ae645f54734` |
| `.runs/cptr/development_final/summary.json` | `4bfde80da8f0ab0ad3ebc95482e31d9df9fbfa5104b69c4831b72ab8468961c4` |

## Matching links between receipts

1. The development-audit summary's `source_sha256.auditor` equals the observed
   auditor hash above. Its `artifact_sha256.development_centres.csv` equals the
   feature summary's `source_sha256.centres`:
   `6a95f2d19c45e070e2d74bda4e09c629f875393df5febcb254fcd8d28646af5f`.
2. The feature summary's `source_sha256.audit_summary` equals the observed audit
   summary hash, and `source_sha256.extractor` equals the observed extractor hash.
   Its `artifact_sha256.development_metadata.csv` equals the split provenance's
   `source_sha256.provider_metadata`:
   `db6de8f77d9ed25c0db0df227d1fd009c3a9b844aade686a5733ff0f259b9cf4`.
3. The split provenance's `artifact_sha256.development_manifest.csv`, the temporal
   manifest lock's `source_sha256.manifest`, and the CPTR development summary's
   `source_sha256.manifest` all agree:
   `5997e412ea8823bff9787aea9b825f465096dfbc25ccebad0bfe97104a4ae4fa`.
   The manifest lock's `source_sha256.split_provenance` equals the observed split
   provenance hash above.
4. The CPTR development-summary hash is already independently bound by
   [the receiving diagnostic](../experiments/analyze_okutama_cptr_baseline_preservation.py)
   and recorded in its retained summary. That diagnostic validates the 11 eligible
   OOF group identifiers without consulting the mixed-role manifest.

The three CSV hashes in these links are **historical attestations read from JSON**;
their CSV bytes were not opened or rehashed here. The agreement identifies a
consistent historical lineage, not a fresh row-level audit or proof that every
historical execution followed its declared source. The extractor's feature-store
hash also agrees between its summary and the temporal manifest lock; no feature
store or array was opened to establish that summary-level equality.

## Correction and remaining work

The initial [continuation plan](HAC_CONTINUATION_EXPERIMENT_PLAN_20260906.md) treated
the meaning of the retained `recording_id` values as unresolved and entertained
grouping synchronized views again. That concern should be narrowed: the historical
source explicitly performs scenario grouping already. Calling these values
individual drone recordings, or treating the two components of an ID such as
`1.2` as drone and scenario, would be incorrect under this source.

Use the statement: "11 scenario identifiers stored under `recording_id`, supported
by hash-matched historical generation code and summary lineage; original
video-to-scenario assignments were not re-enumerated in this review." Inference
remains conditional on the retained folds, selected candidates, and those declared
groups. Repeated tracks, views, folds, and seeds do not create additional
independent scenarios.

Before new fitting, bind the eligible role-specific video, scenario, track, and fold
assignments and test their disjointness. Preserve the historical calibration access
disclosure in [the provenance record](HAC_CONTINUATION_PROVENANCE_20260906.md).
This receipt neither opens a protected role nor repairs the external-CUDA document
portability gap. No model fit, checkpoint access, metadata-row scan, or historical
artifact modification was performed for this review.
