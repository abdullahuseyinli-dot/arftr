# Repository maintenance and release boundary

## Public checkout versus research workstation

| Location | Role | Distribution policy |
| --- | --- | --- |
| `src/hac/` | Library and research components | Source retained; not all modules are production APIs |
| `experiments/` | Historical runners and locked protocols | Preserve paths; execution may require private caches |
| `tests/` | Synthetic/unit/evidence-contract checks | Must not depend on live queue progress |
| `docs/` | Current guides plus dated records | Current index distinguishes active guidance from history |
| `results/` | Compact public evidence | Aggregate exports and existing permitted historical tables |
| `.runs/` | Checkpoints, caches, receipts, full graphs | Local and Git-ignored; requires separate backup |
| `.venv*`, `data/` media | Local dependencies and source data | Not part of a release |

No dataset, checkpoint, cache or historical result was deleted or moved by this
cleanup. Named virtual environments are excluded from Git but still present locally.
The research continuation was committed and fast-forwarded to `main` at `a10075e`.
Before a future release, still inspect untracked files so intended source is not
silently omitted and local research data is not accidentally included.
The existing local `uv.lock` is not a validated project lock and is excluded; the
documented dependency declarations and historical environment snapshots remain
the explicit installation records. Do not infer a fully pinned environment from them.

## Evidence preservation

Pre-cleanup document snapshots live under
`.runs/research_20260920/repo_polish_20260920_v1/original_documents/` with an integrity
record. This is a local recovery copy, **not an independent backup**. All numerical
research artifacts and locked training implementations are preserved.

Current presentation edits must not be confused with rewriting an experimental
protocol. Historical manifests are not regenerated. The public evidence export
records source hashes and omits pixels, labels-by-row, identities and checkpoints.
The public result graph is a curated subset, not the complete local graph.

## Code quality policy

`tools/check_style.py` uses `results/quality/ruff_baseline.json` to retain a visible
inventory of pre-existing diagnostics in historical research code. It fails on new
diagnostics, including new files. Broad auto-formatting of hash-locked training code
is deliberately avoided. Future functional fixes should create a new versioned
recipe where existing execution locks bind the old source.

## Before publishing a release

1. Run current evidence checks, unit tests, style regression checks, and compilation.
2. Review `git diff` and the complete untracked-source inventory. Do not add `.runs/`,
   environments, credentials, model weights, or dataset media.
3. Back up local research evidence independently; a Git commit is not that backup.
4. Choose a release version and citation metadata. `3.1.0.dev0` currently means
   **unreleased**; no DOI, release date, or publication is claimed.
5. Build and inspect a new archive/manifest for that chosen version, rather than
   altering v1/v2/v3 manifests. Validate from a fresh checkout and install.
6. Review dataset/model redistribution boundaries and the existing third-party notices.
7. Publish or deposit only after explicit approval. Cleanup does not authorize a
   remote push, GitHub release, Zenodo upload, or modification of old tags.
