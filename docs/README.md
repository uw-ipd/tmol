# tmol Docs

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
```

The rendered site is written to `docs/_build/html`.

API pages are authored under `docs/api/`. Gallery examples are sourced from
`docs/examples/` and use committed thumbnails. Notebooks are executed by the
smoke command before nbsphinx renders their saved outputs.

GitHub Actions builds docs on pull requests, pushes to `master`, and
`kdidi/**` integration branches. Same-repository pull requests publish under
`previews/pr-<number>/`; integration branches publish under
`previews/branch-<branch-name>-<ref-hash>/`. Pushes to `master` deploy the
current docs to `latest/` and `vX.Y.Z/` on the `gh-pages` branch and update
`_static/switcher.json` for the version picker.
