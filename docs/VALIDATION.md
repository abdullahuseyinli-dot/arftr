# Validation record — 20 September 2026

This records local validation of the portfolio cleanup, not a new scientific
evaluation or a remote CI run.

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
No publication, DOI, new training, promotion, commit or remote push was performed.

Detailed inventories, JUnit records, original-document snapshots and the build
artifact are stored locally under
`.runs/research_20260920/repo_polish_20260920_v1/`. These are local recovery/validation
copies, not an independently backed-up research archive.

See [reproduction instructions](REPRODUCIBILITY.md) and
[release-readiness boundaries](REPOSITORY_MAINTENANCE.md).
