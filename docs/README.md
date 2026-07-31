# tmol Docs

The documentation is built with Sphinx, MyST Markdown, nbsphinx, and
sphinx-gallery.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
pip install scikit-build-core pybind11 ninja packaging "cmake>=3.18,<4"
TMOL_DISABLE_WHEEL_FETCH=1 \
  pip install --no-build-isolation -e ".[docs]" \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
docs/make
```

The rendered site is written to `docs/_build/html`.

`docs/make` runs `sphinx-build -W --keep-going`. API pages are generated from
source with `sphinx-autoapi`, so API reference generation does not import tmol
modules. Gallery examples are sourced from `docs/examples/`, are executed, and
fail the docs build on error. Notebooks are sourced from `docs/notebooks/` and
are rendered without execution.

GitHub Actions builds docs on pull requests and pushes to `master`. Pull
requests upload the rendered HTML as an artifact, and same-repository pull
requests also publish a preview under `previews/pr-<number>/` on the Pages site.
Pushes to `master` deploy the current docs to `latest/` and `vX.Y.Z/` on the
`gh-pages` branch and update `_static/switcher.json` for the version picker.
