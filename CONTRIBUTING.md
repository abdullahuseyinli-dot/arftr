# Contributing

Start with the [documentation index](docs/README.md) and
[reproduction guide](docs/REPRODUCIBILITY.md). Small, independently verifiable
changes are easier to review than combined architecture and evaluation changes.

- Keep evaluation populations, labels and selection boundaries explicit. Do not
  compare development, validation, confirmation and official-test scores as if
  they were interchangeable.
- Add synthetic or temporary-directory tests; never depend on private files or
  the current progress of a local training queue.
- Preserve historical execution locks and results. Use a new versioned experiment
  rather than rewriting evidence to fit a new implementation.
- Do not commit datasets, checkpoints, caches, access tokens or environment folders.
- Do not use held-out labels to train routers or choose intervention thresholds.
- Document negative outcomes and distinguish oracle/diagnostic scores from usable
  model results. No performance improvement is assumed until a fixed gate passes.

Before submitting: run `python -m pytest`, `python tools/check_project.py`,
`python tools/check_style.py` and compilation as described in the reproduction guide.
Changing a hash-locked research implementation may require a separately versioned
recipe and updated evidence; merely updating a checksum is not sufficient.
