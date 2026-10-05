# Contributor guide

Set up an editable installation using {doc}`Development </user_guide/development>`.

## Documentation

Sphinx builds the Markdown/RST guides and the notebooks in `docs/tutorial/`.
Execute the notebooks before building HTML:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.8"
pip install scikit-build-core pybind11 ninja packaging "cmake>=3.24,<4"
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

## Writing documentation

- Lead with the operation or result. Put prerequisites next to the example.
- Use descriptive headings, short paragraphs, and runnable code.
- Keep caveats where they affect a decision. Link to shared conventions instead
  of repeating them on every page.
- Link to a tutorial or API page when it adds detail; avoid introductory link
  checklists and descriptions of what the page is about to explain.
- Keep notebook code, fixtures, outputs, and Colab setup consistent.

## Pull requests

Before opening a PR:

```bash
pre-commit run --all-files
pytest tmol/tests/ -v -k "not cuda"
python .github/scripts/smoke_tutorial_notebooks.py --write
make -C docs html
```

If your change touches CUDA kernels, packing, scoring terms, or minimization,
include the relevant GPU tests or explain why they were not run locally.

## API documentation

API pages under `docs/api/` use `sphinx.ext.autodoc`. Public modules should have
useful module, class, and function docstrings because those docstrings become
the reference documentation.

Use Google or NumPy-style docstrings.

## Agent skills

Keep `skills/` instructions consistent with public commands and APIs. Link to
the docs for explanations. When behavior changes, update the skill, its card,
and its eval case together. See
{doc}`Agent skills </agent_skills>` for the catalog.

```{toctree}
:hidden:

Development <user_guide/development>
Agent skills <agent_skills>
```
