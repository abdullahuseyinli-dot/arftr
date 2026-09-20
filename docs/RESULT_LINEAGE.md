# Result lineage

The current September continuation is indexed in the
[research overview](RESEARCH_OVERVIEW.md) and its
[portable evidence](../results/arftr_development/README.md). The final paired-null
comparison is complete and ARFTR remains retained. The release chains below are
historical studies with different datasets and evaluation boundaries; their scores
must not be combined into a single leaderboard.

The promoted POLAR result has one evidence chain from the pre-fit data audit to the
portable release. Every selection decision was made on the clean development split.
The official test manifest was opened once, after all final fits and their artifact
hashes had been verified.

## Release chain

| Gate | Tracked evidence | Outcome |
| --- | --- | --- |
| Data lock | `polar_data_audit.json`, `polar_quarantine.csv` | 125 source-related images quarantined before fitting |
| Development protocol | `polar_study_protocol.json` | Models, searches, metrics, and label mappings declared |
| Development selection | `polar_validation_*`, `polar_classifier_*`, `polar_confirmation_*` | Candidates and ensemble weights fixed without test rows |
| Final selection lock | `polar_final_selection_lock.json` | Immutable SHA-256 `fa3fb7c80a073d29048afb8e0b8da1fb17f5ade9721630347c37523714cca187` |
| Final-fit gate | `polar_final_fit_manifest.json` | Nine neural fits and three probes hash-verified; zero test rows read |
| Test-access gate | `polar_test_access_gate.json` | Official test cache opened once after the lock |
| Locked evaluation | `polar_test_*` | 3,329 test images; no post-test model selection |
| External evaluation | `polar_external_*` | V-COCO evaluated without retuning after a clean overlap audit |
| Explanation audit | `polar_faithfulness_*` | Fixed 256-image cohort; no attribution-method selection on test |
| Fault audit | `polar_fault_*` | Separate input and classifier-weight bit-flip evaluation |
| Portable export | `polar_final_evidence_manifest.json` | Path-free tracked evidence with per-file hashes |

## Data fingerprints

| Artifact | SHA-256 |
| --- | --- |
| Clean POLAR development manifest | `2cc5bf62790f6064acc74f7113c341b5c2cd1124bcbfbc3d1e9eb19618a75b83` |
| Clean POLAR test manifest | `a8e01e1037e980edeb1886ef2bbbd22a07c316df044741c08e8e2afa8d32fce5` |
| Opened test-manifest cache | `35d2a73b33de866f52d857c266f57d3eee1259113f3da43b07bdd3e8dfa1142e` |
| Clean V-COCO person manifest | `0eb0afcc7e832babc495d3d9a74981077968f401097eedc34c6aeb9a0e644cb4` |
| POLAR/V-COCO overlap audit | `668e0951fc2d4128559a140a306e612fb32321e246617ca0923ee380fc1cc7d0` |

## Selection boundary

The following were fixed before the official test cache opened:

- label spaces and primary metric;
- image views, adaptation depths, augmentation, dropout, and fixed epoch counts;
- seeds 42, 52, and 62 for neural fits;
- the DINOv2 multilayer representation and linear/RBF classifier settings;
- seed averaging and the five component ensemble weights;
- bootstrap seed and resample count;
- the external mapping, attribution cohort, and fault-injection levels.

Test results did not affect model identity, weights, or tie-breaking.

## Person-level V-COCO transfer chain - study v2

| Gate | Tracked evidence | Outcome |
| --- | --- | --- |
| Protocol lock | `results/vcoco_v2/protocol_lock.json` | Model families, views, grouping, and evaluation policy fixed before new fitting |
| Development screen | `results/vcoco_v2/development_candidates.csv` | Candidate comparisons restricted to development data |
| Final selection | `results/vcoco_v2/final_selection_lock.json` | Scale-conditioned stack and all hyperparameters fixed before test access |
| Test-access gate | `results/vcoco_v2/test_access_gate.json` | Official test opened after the selection lock passed |
| Official evaluation | `results/vcoco_v2/official_test_summary.json` | One locked evaluation with paired uncertainty and per-class results |
| Portable export | `results/vcoco_v2/evidence_manifest.json` | Every published v2 table and JSON artifact inventoried by hash |

## Motion-identifiability chain - study v3

| Gate | Tracked evidence | Outcome |
| --- | --- | --- |
| Protocol and amendment | `results/vcoco_v3/protocol_lineage.json` | Original protocol, CUDA amendment, and source hashes retained in one portable lineage record |
| Human pilot audit | `results/vcoco_v3/annotation_summary.json` | 130 presentations audited; labels excluded from candidate selection |
| Representation and spatial development | `results/vcoco_v3/source_tag_development_metrics.csv` | Frozen representation, crop, and stacking screens completed on source-tag development data |
| Temporal development | `results/vcoco_v3/temporal_development_metrics.csv` | Clip span, temporal model, and routing candidates evaluated before confirmation |
| Grouped temporal cross-fit | `results/vcoco_v3/temporal_crossfit_summary.json` | Student targets and fixed pipeline selected from recording-grouped development evidence |
| Confirmation gate | `results/vcoco_v3/protocol_lineage.json` | Confirmation opened once; confirmation results were not used for selection |
| Sealed confirmation | `results/vcoco_v3/confirmation_summary.json` | Static, temporal, and budgeted-routing results recorded with paired uncertainty |
| Portable export | `results/vcoco_v3/evidence_manifest.json` | Confirmation tables, diagnostics, and protocol lineage inventoried by hash |

## Okutama CPTR development chain

| Gate | Tracked evidence | Outcome |
| --- | --- | --- |
| Protocol | `experiments/okutama_cptr_protocol.json` | Execution order, CUDA policy, counterfactuals, and promotion thresholds fixed |
| Feature stores | `results/okutama_cptr/provenance.json` | DINOv2 body regions, camera-compensated trajectories, and SigLIP control revision-bound |
| Component screens | `results/okutama_cptr/component_ablation.csv` | Fourteen retained and rejected development steps recorded with their source revisions |
| Five-seed evaluation | `results/okutama_cptr/headline_metrics.csv` | +0.0081 validation macro-F1 for center and parts |
| Grouped cross-fit | `results/okutama_cptr/fold_seed_metrics.csv` | 25 CUDA runs; -0.0020 OOF macro-F1 |
| Faithfulness | `results/okutama_cptr/faithfulness_summary.json` | Temporal null, order, part, geometry, visibility, and latency checks |
| Development lock | `results/okutama_cptr/development_decision.json` | Existing temporal ensemble retained; CPTR calibration left closed |
| Portable export | `results/okutama_cptr/evidence_manifest.json` | Path-sanitized tables and JSON files with per-file hashes |

The CPTR branch ends at its development gate. It does not reuse the previously opened
Okutama confirmation labels, and it does not promote a validation-only gain after the
recording-grouped OOF comparison moves in the opposite direction.

## Scientific validation boundary

The remaining evidence requirements are bound in
[`SCIENTIFIC_VALIDATION_PLAN.md`](SCIENTIFIC_VALIDATION_PLAN.md). The plan separates
measurement completeness from outcome direction: annotation reliability,
external-domain replication, grouped inference, matched controls, operational cost,
and fresh-environment reproduction must be reported whether their results are
positive, neutral, or negative.

## Local versus tracked artifacts

Local `.runs/` evidence retains checkpoints, fitted classifier binaries, local image
paths, full probability arrays, full-resolution attribution maps, failures, and
interrupted runs. These files are intentionally ignored rather than deleted.

The tracked `results/` export excludes reconstructive or machine-specific artifacts.
Its manifest records the hash of every exported table and JSON document. The final-fit
manifest publishes checkpoint hashes, sizes, configurations, and runtimes without
publishing the checkpoints themselves.

## Historical study

The original 285-image COCO experiment has a separate selection and test lineage. It is
preserved in the non-POLAR result files and summarized in
[`LEGACY_COCO_STUDY.md`](LEGACY_COCO_STUDY.md). Its already-inspected 43-image test set
was not reused to select the POLAR system, and its result is not the repository headline.
