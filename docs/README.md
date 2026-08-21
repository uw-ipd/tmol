# TMol documentation

The documentation is built with Sphinx, MyST Markdown, nbsphinx, autodoc, and
sphinx-gallery.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
pip install scikit-build-core pybind11 ninja packaging "cmake>=3.18,<4"
TMOL_DISABLE_WHEEL_FETCH=1 \
  pip install --no-build-isolation -e ".[docs]" \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
python .github/scripts/smoke_tutorial_notebooks.py --write
docs/make
python .github/scripts/check_api_docs.py
python .github/scripts/check_docs_navigation.py
```

The rendered site is written to `docs/_build/html`.

Task-oriented recipes are grouped by `docs/workflows/index.md`, even when an
existing source file remains under `docs/user_guide/`. The eight numbered
Tutorials live under `docs/tutorial/`; the smoke command executes them before
nbsphinx renders their saved outputs and interactive viewers.

API pages are authored under `docs/api/`. Separate Sphinx-Gallery scripts are
sourced from `docs/examples/`; their generated output is not part of the public
navigation.

GitHub Actions builds docs on pull requests, pushes to `master`, and
`kdidi/**` integration branches. Same-repository pull requests publish under
`previews/pr-<number>/`; integration branches publish under
`previews/branch-<branch-name>-<ref-hash>/`. Pushes to `master` deploy the
current docs to `latest/` and `vX.Y.Z/` on the `gh-pages` branch and update
`_static/switcher.json` for the version picker.
