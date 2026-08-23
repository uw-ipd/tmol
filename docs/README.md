# TMol documentation

The documentation is built with Sphinx, MyST Markdown, nbsphinx, and autodoc.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
pip install scikit-build-core pybind11 ninja packaging "cmake>=3.18,<4"
TMOL_DISABLE_WHEEL_FETCH=1 \
  pip install --no-build-isolation -e ".[docs]" \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
python .github/scripts/smoke_tutorial_notebooks.py --write
make -C docs html
```

The rendered site is written to `docs/_build/html`.

Task-oriented recipes are grouped by `docs/workflows/index.md`, even when an
existing source file remains under `docs/user_guide/`. The eight numbered
Tutorials live under `docs/tutorial/`; the smoke command executes them before
nbsphinx renders their saved outputs and interactive viewers.

API pages are authored under `docs/api/`.

GitHub Actions builds documentation once per pull request. Same-repository pull
requests publish under `previews/pr-<number>/`; pushes to `master` deploy the
current documentation to `latest/`.
