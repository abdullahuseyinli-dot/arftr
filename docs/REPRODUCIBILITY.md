# Reproducibility

## Tier 1 — verify the public evidence, no GPU or dataset

From the repository root, with Python 3.11 or 3.12:

```bash
git clone https://github.com/abdullahuseyinli-dot/arftr.git
cd arftr
python tools/check_project.py
```

This standard-library check verifies current navigation, metadata consistency,
public evidence hashes, confusion-matrix macro-F1/accuracy/error counts, rescue/harm
arithmetic, current figure/report hashes and the fixed final-study decision. It
rejects incomplete evidence inventories and checks each class's support, precision
and recall against the confusion counts. It does
not install anything or download data. It cannot recompute NLL/Brier or scenario-bootstrap intervals without
the original per-example probabilities; those are explicitly exported quantities.

## Tier 2 — install and run code-level tests

Use a fresh environment, separate from historical training environments:

The companion POLAR project retains the historical `hac` import namespace. Install
the two projects in **separate virtual environments**, not together in one environment.

```bash
python -m venv .venv
# Activate: Windows .venv\Scripts\activate; Linux/macOS source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,report,research]"
python -m pytest
python tools/check_style.py
python -m compileall -q src experiments tools
python tools/check_project.py
```

Install the appropriate PyTorch build first if GPU execution is needed. The public
checks and unit tests do not require a GPU or permission to download model weights.
Historical integration tests that need non-distributed `.runs/` evidence are
explicitly marked `local_artifacts` and skip when their required files are absent;
CUDA-only tests also skip on CPU. These skips are not claimed as successful replay.
CI tests both supported Python versions (3.11 and 3.12) with CPU PyTorch.
On the research workstation, require those selected integration inputs with:

```bash
python -m pytest -m local_artifacts --require-local-artifacts
```

The optional `research` extra supplies graph tooling; notebook and PDF dependencies
are only required for their corresponding tooling.

To regenerate the new aggregate-only ARFTR figures (no model execution):

```bash
python tools/render_arftr_figures.py
python tools/build_study_papers.py docs/ARFTR_REPORT.md -o output/pdf/arftr_report_v1.0.0.pdf
python tools/seal_arftr_report.py
python tools/check_project.py
```

This updates only the named ARFTR figures, current report and their manifests.
Review intentional changes before refreshing the report seal: it binds the
Markdown, renderer, figure manifest and PDF. The original study figures, notebook
and PDFs remain untouched; see the [figure guide](../assets/README.md).

`check_style.py` compares Ruff diagnostics with an explicit legacy baseline. New
issues fail; an improvement is allowed. The baseline is not a claim that all
historical research code is style-clean. It avoids changing hash-locked source
solely for formatting. The baseline and its scope are documented in
[repository maintenance](REPOSITORY_MAINTENANCE.md).

## Tier 3 — full historical experiment replay

This requires licensed datasets, original feature caches, outer-fold ancestry
receipts, checkpoints and the matching environment. Those are not in Git. Saved
run artifacts are under `.runs/` on the research workstation. Existing preparation,
training and replay scripts remain in `experiments/`, but many intentionally target
their original local study directories and refuse overwrite or retraining.

The final optional study used PyTorch 2.11.0+cu128 on an NVIDIA RTX PRO 3000 Blackwell
Generation Laptop GPU. All ten saved head outputs replayed bit-exactly with the
layout-corrected auditor. This does not guarantee bitwise identity on another GPU,
PyTorch build, or memory layout. The old environment snapshots are not promises of
current cross-platform binary reproducibility.
The root `requirements-lock.txt` is an earlier study snapshot, not a lockfile for
the full ARFTR research continuation. The maintained package uses `pyproject.toml`;
its dependency ranges do not guarantee bitwise reproduction of historical fits.

The cleanup changes documentation referenced by historical hash manifests. Original
document bytes were copied into the local cleanup snapshot before editing. A legacy
validator that insists the *current* README has old bytes will correctly reject it;
this is not a checkpoint change. Use the preservation-aware local check:

```bash
python tools/check_project.py --local-preservation
```

That mode checks historical protected files either at their unchanged paths or at
their explicitly recorded pre-cleanup document snapshots, and reports every such
substitution. It does not silently rewrite old manifests or waive model/data hashes.

## Historical releases

Versioned v1/v2/v3 PDFs, manifests and checksum files remain historical evidence.
Their builders and exact Git-archive validators target those releases, not the
separately versioned ARFTR 1.0 tree. Do not regenerate an old release manifest to make
new content appear to have been part of the old study. Likewise, the historical
`tools/build_readme.py` generates the old v3 page and must not overwrite the current
hand-maintained overview.
