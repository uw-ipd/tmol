# Contributor guide

Set up an editable installation using {doc}`Development </user_guide/development>`.

## Documentation

Sphinx builds the Markdown/RST guides and the notebooks in `docs/tutorial/`.
Execute the notebooks before building HTML:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.14"
python -m pip install "scikit-build-core>=0.10" "pybind11>=2.12" ninja packaging "cmake>=3.24,<4"
python -m pip install --no-build-isolation -e ".[docs]" \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
python .github/scripts/smoke_tutorial_notebooks.py --write
make -C docs html
```

Rendered HTML is written to `docs/_build/html`.

nbsphinx does not execute notebooks during the Sphinx phase; the smoke command
above executes them first and writes the plots, tables, and viewer HTML that
Sphinx consumes. CI performs the same two-step notebook-and-Sphinx build.

### Adding a tutorial notebook

Put it in `docs/tutorial/` as `NN_name.ipynb` and add it to the gallery in
`docs/examples_index.rst`. CI finds notebooks by that pattern, so nothing else
needs registering. A tutorial must:

- start with the shared setup cell (copy it from an existing notebook):
  `from notebook_setup import *` then `device, param_db = setup()`.
  [`notebook_setup.py`](tutorial/notebook_setup.py) installs TMol on Colab and
  provides the common TMol names, `data_path()` for the checked-in example
  structures, and the device and seed. Add new example files to its `FIXTURES`
  so Colab downloads them;
- carry an Open In Colab badge linking to itself, and the `accelerator: GPU`
  notebook metadata. `smoke_tutorial_notebooks.py` rejects a notebook without
  them;
- set `"tmol": {"gpu_smoke": true}` in the notebook metadata. Every tutorial
  runs on CUDA in CI as well as on CPU, because CPU-only runs hide
  device-mismatch bugs; a test enforces it. Tag cells that need a GPU with
  `gpu-only`.

`docs/tutorial/tmol_tutorial.ipynb` combines all of the section notebooks
above into one notebook with a single shared Setup cell, for readers who want
the whole walkthrough in one place. It doesn't match the `NN_name.ipynb`
pattern, so it isn't part of CI's notebook discovery or GPU smoke -- update it
by hand whenever a section notebook changes, so the two don't drift.

## Writing documentation

- Lead with the operation or result. Put prerequisites next to the example.
- Use descriptive headings, short paragraphs, and runnable code.
- Keep caveats where they affect a decision. Link to shared conventions instead
  of repeating them on every page.
- Link to a tutorial or API page when it adds detail; avoid introductory link
  checklists and descriptions of what the page is about to explain.
- Keep notebook code, fixtures, outputs, and Colab setup consistent.

For tutorials, keep the explanation next to the operation it explains:

- Start with a concrete molecular question, the structure used to answer it,
  and the result the reader will produce. Give a short roadmap in execution order.
- Complete that story with a result check before optional extensions. Connect the
  final table or structure back to the opening question; do not promise a specific
  stochastic outcome or imply that a short run establishes convergence.
- Introduce each new object when it first appears. Distinguish residue labels
  from zero-based block indices, and state what a mask selects.
- Give each code cell one job: prepare inputs, calculate, or inspect a result.
  Prefer public APIs over manual edits to internal tensors.
- After a calculation, say what to look for in its table, plot, or viewer.
  Explain the reference and sign of a score difference before interpreting it.
- Keep modeling calls and movement controls visible. Tag lengthy plotting or
  measurement helpers with `collapse-code`; their outputs should remain open.
- Put advanced extensions after the first complete example. Keep necessary
  scientific limits beside the result rather than repeating them in every cell.
- Keep result tables focused on the comparison: use readable column labels and
  separate score summaries from geometry checks. State what the reader should
  compare before showing the table or plot.

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
