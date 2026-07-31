# Contributor Guide

Keep changes focused and testable. Prefer existing tmol patterns over new
abstractions unless the new abstraction removes real complexity.

## Documentation

The external docs are built with Sphinx, MyST Markdown, nbsphinx, and
sphinx-gallery.

```bash
pip install -r docs/docs_requirements.txt
docs/make
```

Rendered HTML is written to `docs/_build/html`.

Examples live in `docs/examples/` and are rendered by sphinx-gallery. Notebooks
live in `docs/notebooks/` and are rendered without execution during docs builds.

## Pull Requests

Before opening a PR:

```bash
pre-commit run --all-files
pytest tmol/tests/ -v -k "not cuda"
docs/make
```

If your change touches CUDA kernels, packing, scoring terms, or minimization,
include the relevant GPU tests or explain why they were not run locally.

## API Documentation

API pages are generated during `docs/make` with `sphinx-autoapi`. Public modules
should have useful module, class, and function docstrings because those
docstrings become the reference documentation.

Use Google or NumPy-style docstrings.
