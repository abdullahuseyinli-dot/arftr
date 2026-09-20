# Crossing-event trial: small gain, safety gate not met

The completed45-fit fixed primary achieved85.611648% macro-F1 (+0.228000percentage points),19rescues/9harms,692errors. Independent replay and recursive ancestry passed. Fold nets[+9,+2,-1,0,0] fail the required strictly positive result in every fold; total gain/net also fall short of the locked0.50point/25net thresholds. **Retain ARFTR85.383648%,702errors.** The new candidate and fitted gates remain saved, not deployed.

The initial attempt aborted numerically after8fits, without releasing held-out outcomes. A separately locked, complete rerun changed only L-BFGS curvature history10→108; same objective, tolerance and1,000-iteration ceiling. All45fits completed in2.98seconds excluding provenance verification. This was not a long neural-training run.

Current evidence:

- `.runs/research_20260919/crossing_event_verifier_numerical_v2/REPORT.md`: full9-arm table.
- `INTERPRETATION.md` in that directory: attribution, limitations and decision.
- `summary.json`: class/fold/scenario metrics, paired bootstrap, calibration and transitions.
- `knowledge_graph_corrected.json` / `.graphml`:705nodes,1365edges. A positive but insufficient effect supports conditional-event modeling; promotion failure is not a universal mechanism falsification.
- `experiment_ledger.json`:226records, including the separately marked invalid partial run.
- `release_receipt.json`: hashes and GraphML roundtrip verification.

Conditional next phase launched at23:56London: `.runs/research_20260919/learned_persistent_tracking_v2_smoke16`. It runs a frozen CoTracker3 in an isolated verified CUDA12.8 environment, not activity training. Synthetic known-motion controls precede the paired native/downsample16-center screen; failures stop it. The screen cannot automatically expand to128 or train a classifier. The main project environment and original evidence remain unchanged.

Run interpreter: `.venv-tracking-cu128/Scripts/python.exe`.
Launch receipt/logs: `.runs/research_20260919/learned_tracker_setup_v1/smoke16_launch.json`, `smoke16.stdout.log`, `smoke16.stderr.log`.

No additional human review is required for these tests. No tracking result or further classification gain is claimed before its recorded measurements complete.
