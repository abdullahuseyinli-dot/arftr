# HAC continuation provenance record - 2026-09-06

## Status and scope

This is a continuation provenance record for the receiving workspace. It documents
what was verified after transfer and one historical portability defect discovered by
the initial readiness audit. It is **not** a repaired lock, an amendment to any
historical protocol, or authorization to fit, select, calibrate, or evaluate a model.

The reviewed and restored public base commit is
`2697126e3887f99d0b815ada68eb3c4a3c861822`. Receiving work proceeds on the local
continuation branch `research/continuation-20260906`. The supplied
`PACKAGE_MANIFEST.json` declares status
`COMPLETE`, 2,884 records (2,880 archived and four reconstructed from Git), and the
same reviewed commit. Its recorded SHA-256 is
`83ec460de5b273fc235e85434c4850755170f1f6f000229ceb4aa99f41786f18`.
The outer transfer ZIP is 11,363,833,970 bytes with SHA-256
`33b7cdc42b4fff2c8ba0d77d32e49da8616092551c0f4a89ee8b5915ed045238`.
Deep verification and restoration completed without a mismatched member. The ignored
receiving path map has SHA-256
`c504a51d66c9b0f2f14fceb5e56f5c242782e8fc375b2c67a65337c54b6edf9d`.

The receiving environment was established with Python 3.11.9, NumPy 2.4.6, Pandas
3.0.5, PyTorch `2.11.0+cu128`, Torchvision `0.26.0+cu128`, and the CUDA 12.8 wheel
runtime. CUDA was visible on an NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU.
The host is 64-bit Windows 11 Pro build 26200 with an Intel Core Ultra 9 285H
(16 cores/16 logical processors); NVIDIA driver 596.72 reports compute capability
12.0 and 12,227 MiB. The exact `requirements-v3-lock.txt` SHA-256 is
`2b57c7772b4cd3660f8ba9c5116e7c1f94c5af2f6a537d384c0241a818f26a1b`.
Package-pin, CUDA matrix-operation, and Torchvision NMS smoke checks passed. The
ignored environment record has SHA-256
`538f15cb4dec3f19e9f8ec4325c99bbbf9e1e505bffe47b1b4b56ece5636ddd7`.
No model fit or model inference was performed, so this receipt makes no model-level
determinism claim; every receiving resampling analysis records its own seed and method.
The transferred portable-evidence audit was reproduced in an ignored continuation
directory and passed all 77 declared manifest checks. This verifies the carried
portable evidence at its declared byte representations; it does not replay raw
experiments or restore eligibility of a consumed evaluation split.

The initial receiving-machine readiness record is intentionally outside Git at
`.runs/research_20260906/readiness_initial/summary.json`, SHA-256
`8ba09c491aac8cf77c05471c28685963dc9860c5f5b424c9f245d4d36bc347f5`.
It was run without `--probe-dinov3` and without opening NumPy prediction, test,
calibration, or confirmation arrays. The audit completed, but correctly reported
`next_gate: protocol` and `VCOCO_V3_PROTOCOL_OR_AMENDMENT_INVALID`.

After that audit, a receiving-machine search for stale absolute paths was scoped too
broadly and printed the first five data rows of
`.runs/vcoco_v3/temporal/development_manifest.csv`; those rows have the CPTR
`calibration` role. An interim CPTR diagnostic then streamed the same mixed-role
manifest line by line to route on its final `split` field. It did not CSV-parse or
retain protected fields, but it did read 1,979 calibration-row strings and recorded
that count. The focused integration test repeated that raw scan. These were process
errors, not planned calibration analyses, and “zero calibration reads” is not an
accurate description of the receiving session.

No calibration image, feature array, probability, or checkpoint and no confirmation
input was opened. The interim router used only each row's final `split` token to
exclude and count forbidden-role rows; no displayed or raw-row field contributed to a
candidate metric or selection decision. The replacement analyzer at
`experiments/analyze_okutama_cptr_baseline_preservation.py` has SHA-256
`8e433feb2453ef0340eb56255fd2078cffd79843a95348bba7339147c361299f`;
its focused test has SHA-256
`cdce5960a06dce99df7fcf87c4fe083399712fbb1bc6b97aa7b474101876ef9a`,
and its 10,000-resample ignored summary has SHA-256
`32dcf384865983651c317fe961cfe6187013b13f6910df83330f804f6ee85e73`.
It has no mixed-role manifest path or read and validates the two permitted scopes from
independently hash-bound development prediction arrays and anchors. Any future
continuation lock must disclose both exposures and must not describe the calibration
manifest as pristine or unseen.

## External-CUDA amendment portability defect

The restored historical lock at
`.runs/vcoco_v3/protocol/external_cuda_amendment_lock.json` has SHA-256
`c6b4d7c414104a29e515a016274f57d3186a1c2a848c38444ca1071de3a81b38`.
It binds document bytes that are not present in the reviewed Git history or transfer
payload:

| Document | Lock-time SHA-256 | First/current Git blob | Published SHA-256 |
| --- | --- | --- | --- |
| `docs/VCOCO_V3_EXTERNAL_CUDA_AMENDMENT.md` | `104a4f042c13249f737cace31eac2288e849fb186043d525bf53dd2f657b8ca1` | `1fac20d2ad9dcf57258d55c3136aed72759b3369` | `6ae422c0209c4c27aad7696877612345aab45cb3d14a639da6371f3bd744f7bb` |
| `docs/VCOCO_V3_RESEARCH_PROTOCOL.md` | `4e36667920481a5d4d60719ff101c6597225d494c6f62f8f64dd7a987129b308` | `b07bff9ddb0ceb5cd235b6096bb4c226b46c96e3` | `8f5bbab54dfdb9c9e329643219efa8171405ac7f9f1537f0d51b9d95a6e46ec1` |

The evidence supports the following diagnosis:

- The lock records `locked_at_utc: 2026-08-24T08:53:55.433695+00:00` and repository
  revision `11b7b9160785466841588b91ac815943dd94dad5`. Neither document path exists in
  that commit.
- Restored historical readiness records show that the expected bytes did exist in
  the original working tree. At `2026-08-24T09:10:35.099095+00:00`, the tree had 83
  changes; at `2026-08-24T13:30:14.858486+00:00`, it had 100. Both audits report exact
  expected/current document-hash matches and an accepted effective amendment.
- Both documents were first committed later that day: on the research line at
  `e93eaf57eff7c27cd112a230c1044cd22384ef55` and on the main line at
  `222d07e115eefd7ad9553dfd656be2819d4bf9d2`. Those commits contain the published
  blobs in the table, not the lock-time bytes.
- A read-only search covered all 60 commits reachable from local refs and reflogs and
  all 754 blob objects. Git reported no unreachable objects. Neither expected SHA-256
  was found, and LF/CRLF normalization does not explain either mismatch.
- Neither expected SHA-256 nor a target document member occurs in the 2,884-record
  transfer manifest or payload ZIP member lists. The transfer explicitly omitted
  repository source in favor of reconstructing it from the reviewed commit. That
  assumption is insufficient for these two never-committed lock-time documents.

The historical lock therefore has credible contemporaneous hash attestations, but
its exact source documents are no longer byte-replayable in this checkout. It was not
shown to be invalid when created; it is stale relative to the published source and
the transfer has a narrow source-lineage completeness gap. Updating the old hashes to
the published files would erase that distinction and must not be treated as a repair.
The preserved `VCOCO_V3_PORTABLE_PROTOCOL_LINEAGE_COMPLETE` status records the
historical export's own conclusion; it does not override this receiving-machine proof
that the two exact lock-time source documents are absent.

## Immutable boundaries

- Do not edit or regenerate the historical external-CUDA amendment lock, historical
  readiness records, exported evidence, labels, or consumed-run artifacts.
- Do not weaken the readiness integrity check, substitute the published documents for
  the missing bytes, or infer protocol acceptance from the readiness process exiting
  successfully.
- Previously used POLAR/V-COCO tests and Okutama confirmation remain consumed.
  Protected CPTR calibration/confirmation data and arrays are not development inputs.
- This note records provenance only. It does not authorize new fitting or opening an
  evaluation gate.

## Safest recovery and continuation action

1. If the original workstation, backups, Git index/object database, or editor local
   history remain available, recover the two lock-time files and accept them only
   after exact SHA-256 matches to the lock. Preserve recovered bytes as immutable
   archival source copies; do not overwrite the published documents.
2. If exact recovery is impossible, retain this explicit gap and the contemporaneous
   readiness attestations. Do not reconstruct document wording from hashes, reports,
   or later files.
3. Before any new fitting, establish separate V-COCO and Okutama continuation
   protocols/amendments at new paths. Commit their documents first, require a clean
   tree, bind the commit and Git blob IDs as well as SHA-256 values, and archive the
   exact bound source documents.
4. Restrict each continuation study to data permitted by its new protocol. Any new
   confirmatory claim requires a freshly locked independent holdout; historical
   test, calibration, and confirmation arrays remain outside the development loop.
