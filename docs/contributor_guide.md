# Contributor Guide

Keep changes focused and testable. Prefer existing tmol patterns over new
abstractions unless the new abstraction removes real complexity.

## Documentation

The external docs are built with Sphinx, MyST Markdown, nbsphinx, autodoc, and
sphinx-gallery.

```bash
pip install -r docs/docs_requirements.txt
python .github/scripts/smoke_tutorial_notebooks.py --write
docs/make
```

Rendered HTML is written to `docs/_build/html`.

Examples live in `docs/examples/` and are rendered by sphinx-gallery with
committed thumbnails. nbsphinx does not execute notebooks during the Sphinx
phase; the smoke command above executes them first and writes the plots, tables,
and viewer HTML that Sphinx consumes. CI performs the same two-step build.

## Pull Requests

Before opening a PR:

```bash
pre-commit run --all-files
pytest tmol/tests/ -v -k "not cuda"
python .github/scripts/smoke_tutorial_notebooks.py --write
docs/make
```

If your change touches CUDA kernels, packing, scoring terms, or minimization,
include the relevant GPU tests or explain why they were not run locally.

## API Documentation

API pages under `docs/api/` are authored and use `sphinx.ext.autodoc`, matching
the AtomWorks and RFD4 docs setup. Public modules should have useful module,
class, and function docstrings because those docstrings become the reference
documentation.

Use Google or NumPy-style docstrings.

```{toctree}
:maxdepth: 2

user_guide/development
```
