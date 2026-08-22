# Contributor guide

Keep changes focused and testable. Prefer existing TMol patterns over new
abstractions unless the new abstraction removes real complexity.

> - **Prerequisites:** {doc}`Development setup </user_guide/development>`.
> - **Documentation entry points:** {doc}`Tutorials </examples_index>`,
>   {doc}`workflow recipes </workflows/index>`, and
>   {doc}`API reference </api_reference>`.

## Documentation

The external docs are built with Sphinx, MyST Markdown, nbsphinx, and autodoc.
Two documentation forms serve different purposes:

- `docs/workflows/` and selected `docs/user_guide/` pages are concise,
  reusable recipes that link to deeper material rather than reproducing it.
- The top-level **Tutorials** section contains the eight interactive notebooks
  in `docs/tutorial/`. They are deeper, executable walkthroughs rendered by
  nbsphinx.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
pip install scikit-build-core pybind11 ninja packaging "cmake>=3.18,<4"
TMOL_DISABLE_WHEEL_FETCH=1 \
  pip install --no-build-isolation -e ".[docs]" \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
python .github/scripts/smoke_tutorial_notebooks.py --write
make -C docs html
```

Rendered HTML is written to `docs/_build/html`.

The committed notebook thumbnails live under `docs/_static/tutorials/`.
nbsphinx does not execute notebooks during the Sphinx phase; the smoke command
above executes them first and writes the plots, tables, and viewer HTML that
Sphinx consumes. CI performs the same two-step notebook-and-Sphinx build.

## Pull Requests

Before opening a PR:

```bash
pre-commit run --all-files
pytest tmol/tests/ -v -k "not cuda"
python .github/scripts/smoke_tutorial_notebooks.py --write
make -C docs html
```

If your change touches CUDA kernels, packing, scoring terms, or minimization,
include the relevant GPU tests or explain why they were not run locally.

## API Documentation

API pages under `docs/api/` use `sphinx.ext.autodoc`. Public modules should have
useful module, class, and function docstrings because those docstrings become
the reference documentation.

Use Google or NumPy-style docstrings.
