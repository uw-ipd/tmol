# tmol Docs

The documentation is built with Sphinx, MyST Markdown, nbsphinx, autodoc, and
sphinx-gallery. The source branch keeps authored documentation pages, notebooks,
examples, and static assets. CI builds rendered HTML and publishes that output
to `gh-pages`.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
pip install scikit-build-core pybind11 ninja packaging "cmake>=3.18,<4"
TMOL_DISABLE_WHEEL_FETCH=1 \
  pip install --no-build-isolation -e ".[docs]" \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
python .github/scripts/smoke_tutorial_notebooks.py --write
docs/make
```

The rendered site is written to `docs/_build/html`.

API pages are authored under `docs/api/` and use `sphinx.ext.autodoc`, matching
AtomWorks and RFD4. Gallery examples are sourced from `docs/examples/` and
rendered with committed thumbnails, matching the RFD4 gallery setup. Notebooks
are executed by the smoke command before nbsphinx renders their saved outputs.
CI follows the same sequence, so published plots, tables, and py3Dmol viewers do
not depend on committed notebook outputs.

GitHub Actions builds docs on pull requests and pushes to `master`. Pull
requests upload the rendered HTML as an artifact, and same-repository pull
requests also publish a preview under `previews/pr-<number>/` on the Pages site.
Pushes to `master` deploy the current docs to `latest/` and `vX.Y.Z/` on the
`gh-pages` branch and update `_static/switcher.json` for the version picker.
