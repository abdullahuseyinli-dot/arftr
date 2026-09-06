# Experiment runners

The September 2026 [team review](../docs/HAC_ENGINEERING_RESEARCH_REVIEW_20260906.md)
and [next-run plan](../docs/HAC_NEXT_RUN_PLAN_20260906.md) add two analysis tools:
`analyze_hac_component_relationships.py` compares immutable development exports with
shared grouped resampling and saves an aggregate figure; `audit_cptr_initialization_contract.py`
executes a synthetic no-data/no-fit check of legacy attenuation and short-frame sampling.
Neither command trains or promotes a model.

The repository contains four versioned experiment sequences. Each sequence separates
data preparation, development-only model selection, final fitting, test access, and
portable evidence export.

## POLAR v1 source benchmark

1. `tools/prepare_polar.py` builds the target manifest, audits exact and perceptual
   cross-split relationships, and writes the quarantine.
2. `cache_polar_features.py` and the probe runners screen views, label spaces, and
   frozen classifiers on development data.
3. `train_polar_candidate.py` runs adaptation and regularization candidates;
   `aggregate_polar_seed_predictions.py` and the confirmation runners establish
   multi-seed stability.
4. `analyze_polar_validation.py` measures complementarity and selects blend weights.
5. `lock_polar_final_selection.py` writes the pre-test selection lock.
6. `fit_polar_final_model.py` and `fit_polar_final_probe.py` fit the declared models on
   the combined clean development set and verify their artifacts.
7. `evaluate_polar_final.py` performs the single locked POLAR test evaluation.
8. `evaluate_vcoco_external.py` runs the original no-retuning V-COCO audit after
   `tools/audit_polar_vcoco_overlap.py` verifies the cross-dataset boundary.
9. `evaluate_polar_faithfulness.py` runs bbox-aware attribution, perturbation,
   target-sensitivity, and parameter-randomization checks on the fixed cohort.
10. `evaluate_polar_fault_robustness.py` evaluates exact input and classifier-weight
    bit flips as a separate robustness study.

## V-COCO v2 person-level study

1. `tools/build_vcoco_v2_manifests.py` creates factorized person-level manifests while
   retaining the official train, validation, and test memberships.
2. `tools/lock_vcoco_v2_protocol.py` fixes the target mapping, overlap policy,
   development boundary, test gate, metrics, and promotion rule.
3. `tools/build_vcoco_v2_training_manifest.py` materializes the label-bearing training
   view from the locked source manifest.
4. `cache_vcoco_v2_features.py` extracts revision-pinned DINOv2, ConvNeXt, and SigLIP2
   features for declared person views without reading test labels.
5. `screen_vcoco_v2_features.py`, `evaluate_vcoco_v2_source_model_views.py`, and
   `screen_vcoco_v2_pose_rbf.py` run the controlled representation, crop, geometry,
   classifier, and pose screens.
6. `evaluate_vcoco_v2_factorized_fusion.py` compares factorized posture-motion targets
   with flat classifiers under matched inputs.
7. `run_vcoco_v2_fewshot_curve.py` measures image-grouped target-label efficiency.
8. `train_polar_candidate.py` and `evaluate_vcoco_v2_neural_checkpoint.py` run the
   linear-probe-then-fine-tune experiments with mild and AugMix policies.
9. `analyze_vcoco_v2_mechanisms.py` and `analyze_vcoco_v2_prediction_shift.py` quantify
   person-scale, context, background, geometry, and error-transition effects.
10. `fit_vcoco_v2_final_stack.py` replays the selected development result and refits the
    stack on the combined train and validation rows.
11. `tools/lock_vcoco_v2_final_selection.py` binds the selected estimator, inference
    sources, historical baseline, metrics, and official-test contract.
12. `cache_vcoco_v2_final_test_features.py` performs label-free test feature extraction;
    `evaluate_vcoco_v2_final_test.py` then opens the test labels once and writes the
    locked evaluation.
13. `tools/export_vcoco_v2_results.py` creates the path-sanitized evidence package, and
    `tools/render_vcoco_v2_figures.py` renders the tracked figures from that package.

## Motion-identifiability and temporal extension

1. `tools/finalize_vcoco_v3_pilot.py` freezes the fixed 130-presentation annotation
   prefix; the resulting descriptive labels are excluded from model selection.
2. `evaluate_vcoco_v3_nested_stacks.py` compares factorized DINO and DINO/SigLIP
   probability stacks under nested image-grouped cross-validation.
3. `run_vcoco_v3_feature_queue.py`, `evaluate_vcoco_v3_spatial.py`, and
   `evaluate_vcoco_v3_representations.py` run the locked spatial and matched frozen
   representation screens.
4. `lock_vcoco_v3_neural.py` applies the predeclared promotion gate. The recorded run
   stops neural adaptation when no spatial family qualifies.
5. `audit_okutama_development.py` and `build_vcoco_v3_temporal_split.py` audit the
   provider train archive and create scenario-grouped train, validation, and
   calibration partitions.
6. `cache_okutama_temporal_features.py` extracts revision-bound DINOv2-B frame features
   on CUDA; the source-only and few-shot runners then measure the target-domain gap.
7. `run_vcoco_v3_temporal_queue.py` trains the matched static and short-clip candidates,
   recording-grouped cross-fits, and static students on CUDA.
8. `finalize_vcoco_v3_temporal_development.py` fixes the classification student and
   validates the temporal-benefit router before calibration data is read.
9. `calibrate_vcoco_v3_temporal_pipeline.py` binds the final five-seed models,
   temperatures, routing budgets, prediction-set thresholds, and artifact hashes.
10. `audit_okutama_confirmation.py` opens the provider test archive once;
    `evaluate_vcoco_v3_temporal_confirmation.py` evaluates the locked pipeline without
    changing it.
11. `tools/export_vcoco_v3_results.py` writes the path-sanitized evidence package, and
    `tools/render_vcoco_v3_figures.py` renders figures only from that package.

## Okutama CPTR architecture study

1. `tools/lock_okutama_cptr_protocol.py` binds the data boundary, architecture,
   execution order, regularization, counterfactuals, and promotion gates.
2. `audit_okutama_cptr_baseline.py` replays the frozen five-seed baseline and its
   temporal interventions before fitting a new component.
3. `cache_okutama_cptr_motion.py` builds raw and camera-compensated person trajectories;
   `cache_okutama_cptr_parts.py` and `cache_okutama_cptr_siglip.py` build the frozen
   part-region and center-frame specialist stores on CUDA.
4. `train_okutama_cptr_candidate.py` executes the locked raw trajectory, compensated
   trajectory, center-conditioned, dual-clock, part, counterfactual, masked,
   specialist, GroupDRO, and integrated screens. Each grid is hash-locked before its
   first fit.
5. `fit_okutama_cptr_router.py` evaluates the continuous utility router and required
   confidence heuristics; `pretrain_okutama_cptr_masked.py` evaluates label-free target
   video adaptation.
6. `train_okutama_cptr_lora_specialist.py` trains the declared top-block DINOv2 LoRA
   control with CUDA automatic mixed precision.
7. The center-and-parts component is rerun with five seeds. Its fixed epoch counts are
   recorded in `okutama_cptr_crossfit_plan.json` and bound by
   `lock_okutama_cptr_crossfit_plan.py`.
8. `crossfit_okutama_cptr.py` runs all 25 recording-grouped fold/seed fits on CUDA.
   `finalize_okutama_cptr_development.py` aggregates OOF predictions, exact group-swap
   tests, paired cluster intervals, class metrics, and visibility/transition subgroups.
9. `evaluate_okutama_cptr_faithfulness.py` measures temporal and modality interventions,
   learned gates, reliability, and cached-feature inference latency.
10. `lock_okutama_cptr_development.py` fixes the development decision before any
    calibration row can be opened. `export_okutama_cptr_results.py` produces the
    path-sanitized package in `results/okutama_cptr`.

## 2026-09-06 continuation diagnostics

The finalized receiving-workspace tools analyze retained development predictions
without fitting models or opening test, calibration, or confirmation arrays. The
continuation provenance record discloses an earlier mixed-manifest metadata exposure;
it was not an array or model-evaluation access.

1. `build_vcoco_development_evidence.py` binds the v2 train/val row manifests and joins
   them to byte-locked predictions from a run that attests v3 grouped OOF; the retained
   artifact does not permit independent fold reconstruction.
2. `analyze_vcoco_development_evidence.py` reports paired error directions, scale
   slices, proper losses, and source-image-cluster uncertainty.
3. `analyze_okutama_cptr_baseline_preservation.py` compares the retained CPTR candidate
   with its exact temporal anchor and a clearly labeled oracle occlusion fallback,
   using recording-cluster intervals and exact recording swaps.

The continuation provenance record and draft experiment plan are in `docs/`. The
draft grants no fitting authorization; a new committed protocol and fresh lock are
required before running any proposed intervention or candidate.

The commands remain explicit so each gate has a visible input and output. Long-running
jobs resume only when the recorded implementation and artifact hashes still match.
`pipeline_source.ipynb` belongs to the historical COCO benchmark; the root notebook is
the current executed evidence summary.
