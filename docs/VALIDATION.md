# Validation record — 20 September 2026

This records the independent ARFTR 1.0.0 project package and earlier presentation
reviews. None is a new scientific evaluation. Local checks and remote CI are
distinguished below.

## ARFTR 1.0.0 project separation

| Check | Outcome on 20 September 2026 |
| --- | --- |
| Full workstation test suite | **1,535 passed, 6 CUDA-only skips**, 78.20 seconds |
| Public-only checkout | 1,004 permitted files; no private runs, environments or dataset media |
| Focused tests in that public copy | **11 passed**, including metadata/navigation and architecture figures |
| Public evidence | Three continuation systems and seven original component arms recomputed; 220 historical numerical/report hashes and six current figures verified |
| Local preservation | 218 protected records, 59 completed-experiment artifacts and 907 execution-locked dependencies unchanged |
| Style and compilation | Passed; no new Ruff findings (533 unchanged legacy findings) |
| Independent wheel | `arftr-1.0.0-py3-none-any.whl` built successfully |

The rename preserves Git history; the companion POLAR benchmark has its own
identity, source/evidence inventory, tests and figures. Project version 1.0.0 does
not imply independent confirmation of the adaptive ARFTR score. The current
[quality-gates workflow](https://github.com/abdullahuseyinli-dot/arftr/actions/workflows/ci.yml?query=branch%3Amain)
is the authoritative remote check; prior runs below refer to earlier commits.

## Original cleanup validation

| Check | Outcome |
| --- | --- |
| Public-only working-tree copy, without `.runs/`, datasets or environments | 1,523 tests passed; 13 skipped; zero failures |
| Skip breakdown | Seven private-artifact integration tests; six CUDA-dependent tests |
| Six explicitly registered private integration tests, run locally with required assets | All passed |
| Current documentation navigation | Validated; no broken checked local links |
| Public confusion-matrix replay | Three systems; 4,977 examples; macro-F1, accuracy and errors agree |
| Gate and rescue/harm arithmetic | Matches the locked final-study decision |
| Historical report/numerical-export hashes | 220 files verified |
| Original closeout preservation inventory | 218 accounted for; original README bytes retained in its snapshot |
| Final experiment artifacts | 59 files unchanged |
| Final experiment locked dependencies | 907 files unchanged |
| Ruff regression check | No new findings; 533 explicit legacy findings remain |
| Compilation | `src`, `experiments` and `tools` passed |
| Package build | `human_activity_classification-3.1.0.dev0-py3-none-any.whl` built successfully |

The test environment was the existing general-purpose Windows environment:
Python 3.11.9, CPU PyTorch 2.14.0, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1,
pytest 9.1.1 and Ruff 0.16.6. The public-only copy tests isolation from local data;
it was **not a fresh dependency installation**. Python 3.12 and remote Linux CI were
not executed during this cleanup. The archived GPU experiment used its separate
historical environment; no training environment was modified.

After the public-only full test run, architecture/validation documentation and the
navigation check's document list were updated. Evidence, style, navigation and the
nine focused project/queue tests were rerun successfully; no model code changed.
No publication, DOI, new training or promotion was performed during cleanup.
The subsequent authorized commit and main-branch push are recorded below.

Detailed inventories, JUnit records, original-document snapshots and the build
artifact are stored locally under
`.runs/research_20260920/repo_polish_20260920_v1/`. These are local recovery/validation
copies, not an independently backed-up research archive.

## Main-branch integration

The research continuation and cleanup were fast-forwarded into `main` at
`a10075ee03f5459930a6ee9a92fff7b4850251cb`. The
[Linux quality-gates run](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/runs/35525764564)
passed installation, style regression, compilation, tests and public evidence checks.
GitHub's default branch was verified as `main`. This CI result applies to that exact
commit; it does not validate revisions made after it. The README's main-specific
CI badge links to the current remote status.

## Comparison with the earlier main

The review compared previous `main` (`2697126`) with the integrated revision
(`a10075e`) and checked the landing page, documentation entry points, reports,
notebook, figures, public evidence, metadata, installation commands and CI.
It was not a new line-by-line audit of every historical training implementation.

| Area | Comparison finding | Follow-up |
| --- | --- | --- |
| Landing page | Newer scope and claim boundaries were clearer, but useful visuals had disappeared | Restored a selected confirmation figure; added development gain and fold-stability plots |
| Reports and notebook | Original artifacts were preserved, but harder to discover | Direct README/index links and explicit historical scope |
| Architecture | Newer research map lacked the actual computation | Factor-residual diagram, equations and look-ahead/metadata limitations |
| Latest results | Portable evidence existed but residual errors were not visualized | Row-normalized confusion matrix, counts and per-class metrics |
| Status | Cleanup-only statements predated the subsequent merge and remote CI | Separate dated validation stages and commit-specific CI evidence |
| Evidence integrity | Earlier numerical exports, figures and PDFs were unchanged | Retained hashes and added checks binding new figures to public inputs and renderer |

No model architecture, training recipe, checkpoint, raw data or scientific result
was changed by this review. A better repository presentation is not a claim of a
new model-performance gain.

## Presentation-review validation

| Check | Outcome |
| --- | --- |
| Public-only copy: 998 source/evidence files, no private runs or datasets | 1,527 passed, 13 skipped, zero failures; 81.77 seconds |
| Skip breakdown | Seven private-artifact integration tests; six CUDA-dependent tests |
| Required private integration checks on the research workstation | Six passed; 1,534 deselected |
| Expanded current documentation navigation | 132 local links checked |
| New figure tests | Input integrity, plotted values, confusion orientation, rendering and output tampering checked |
| Figure output integrity | Four PNG/SVG files bound to public inputs and renderer |
| Historical and local preservation | 220 public artifacts, 218 protected records, 59 final-study files and 907 locked dependencies verified |
| Style and compilation | No new Ruff findings; compilation passed; 533 legacy findings remain disclosed |
| Public-only package build | `human_activity_classification-3.1.0.dev0-py3-none-any.whl`, 402,879 bytes |

The public copy used the same existing Windows CPU environment described above,
not a fresh dependency installation. Both new figures were inspected visually.
The full suite included all four new figure tests; the six private integration checks
ran separately with `--require-local-artifacts`. No GPU training or model evaluation
was launched. The notebook and all historical figures/PDFs remain byte-preserved.

Local inventories, the public copy, JUnit results and wheel are under
`.runs/research_20260920/presentation_review_v1/`. The wheel's SHA-256 is
`2b479e1e6f486b344b47be3503edaae4cf964377cc3835816d294bf158bc36a5`.
This validation record and whitespace-only normalization of generated SVG paths
were completed after the full test run; the final navigation, evidence-preservation,
style and focused tests were rerun before committing.

## ARFTR-centered landing-page revision

Following the presentation review, the landing page was reorganized around ARFTR's
design, retained result and original component study. Earlier studies and detailed
follow-up corrections remain in linked supporting documentation. No scientific
result, historical report or original evidence export was removed or rewritten.

The new architecture supplement exports the seven original ARFTR arms from their
preserved September 12 summary, with a source hash and protocol binding. Its primary
confusion matrix, macro-F1, accuracy, NLL and Brier agree with retained ARFTR. The
original uncertain interval and unsuccessful strict significance screen remain
explicit; this export is not a new experiment or a stronger performance claim.

Validation used a new 1,002-file public-only copy in the existing Windows CPU
environment: **1,528 tests passed, 13 skipped, zero failures**, in 84.21 seconds.
The skips remain seven private-artifact and six CUDA-dependent tests. Checks also
passed for seven component confusion matrices, 142 current local links, six figure
files, metadata, style and compilation. The 220 historical public artifacts and
all previously protected local research files/dependencies remained intact.

The new component figure was inspected visually. Its values and interval are read
from the aggregate export. Local export provenance, public-copy inventory and test
results are under `.runs/research_20260920/arftr_landing_page_v1/`. This validation
record and the final display labels were refined after the full run. Source review
corrected the earlier A3 description: A3 is the unrestricted-template expert; M4 is
the actor-memory anchor. Numerical data did not change. Navigation, preservation,
style and focused checks were rerun before committing. The live main-branch CI badge identifies the
remote check for the published revision.

See [reproduction instructions](REPRODUCIBILITY.md) and
[release-readiness boundaries](REPOSITORY_MAINTENANCE.md).
