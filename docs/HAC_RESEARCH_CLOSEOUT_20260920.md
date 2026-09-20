# ARFTR improvement phase: closed

Date: 20 September 2026. The user accepted the recommendation to close this improvement phase. This is an in-place research closeout, not deletion of the repository, remote archival, or a claim that further improvement is impossible.

## Retained result

| Item | Preserved value |
|---|---|
| System | ARFTR, `r5_arftr_full` |
| Macro-F1 | 85.3836480737% |
| Accuracy | 85.8951175407% |
| Errors / evaluated centers | 702 / 4,977 |
| Evaluation scope | Adaptive internal development; not independent deployment confirmation |
| Result directory | `.runs/research_20260912/arftr_v1/results/v0001/` |
| Fold checkpoints and receipts | `.runs/research_20260912/arftr_v1/fold-0/` through `fold-4/` |

## Applied decision

- End the current architecture/correction search cycle. No automatic next trial, tracking fallback, sweep, extraction or training is planned.
- Preserve the retained model, predictions, dependencies, prior experimental evidence and negative results unchanged.
- Keep the distinction between measured results, post-hoc diagnostics, oracle bounds and untested hypotheses. No diagnostic result replaces ARFTR.
- Reopening requires an explicit user request and a newly justified, bounded protocol. Existing scripts remain executable; this is a workflow decision, not an operating-system execution lock.

No training or queue workers were found active during closeout, and no HAC-related Windows scheduled task was found. No unrelated processes were stopped. The machine-readable closeout receipt records the time and scope of these checks; this is not a claim about remote schedulers.

## Evidence and preservation

The [viability assessment](../.runs/research_20260920/viability_reassessment_v1/ASSESSMENT.md) explains the decision, including the recent plateau and the motion mechanism-to-code gap. Its [mind map](../.runs/research_20260920/viability_reassessment_v1/MIND_MAP.md), [JSON graph](../.runs/research_20260920/viability_reassessment_v1/knowledge_graph.json), [GraphML](../.runs/research_20260920/viability_reassessment_v1/knowledge_graph.graphml) and [fresh metric ledger](../.runs/research_20260920/viability_reassessment_v1/fresh_metric_ledger.csv) remain unchanged. The [consolidated historical ledger](../.runs/research_20260920/aerial_council_v1/experiment_ledger_consolidated.json) retains the broader lineage.

The [closeout receipt](../.runs/research_20260920/research_closeout_v1/closure.json) applies the recommendation recorded in that historical graph; it supersedes the recommendation's pending status without rewriting the graph. The [SHA-256 manifest](../.runs/research_20260920/research_closeout_v1/preservation_manifest.json) inventories the retained ARFTR files, its 182 directly locked dependencies, the assessment, the council ledger/graph, and closeout documentation. Fold checkpoints, predictions, receipts and the aggregate output are checked against their historical hashes. The manifest is an integrity record, not a backup or a full transitive packaging of raw data and dependencies.

The original continuation archive remains at `C:\Users\DELL\Downloads\HAC_Research_Continuation_20260906.zip`. It is neither modified nor copied by closeout. No historical files, caches or environments are deleted, moved, or made read-only.

**Backup limitation:** `.runs/` is Git-ignored. The evidence remains local; a Git commit alone would not back it up. No remote push, commit, new archive or independent backup is created here.

Conclusion: retain the working result and stop spending on this search cycle. Remaining hypotheses are unproven, not promises of future gains.
