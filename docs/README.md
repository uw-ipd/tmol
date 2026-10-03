# TMol documentation

The documentation is built with Sphinx, MyST Markdown, nbsphinx, and autodoc.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
pip install scikit-build-core pybind11 ninja packaging "cmake>=3.24,<4"
TMOL_DISABLE_WHEEL_FETCH=1 \
  pip install --no-build-isolation -e ".[docs]" \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
python .github/scripts/smoke_tutorial_notebooks.py --write
make -C docs html
```

The rendered site is written to `docs/_build/html`.

## Branding and visual style

The [brand assets](_static/brand/README.md) include the editable SVG master,
light/dark logos, standalone marks, PNG sizes, monochrome artwork, and favicons.
Run `python scripts/generate_brand_assets.py` from the repository root to
regenerate exports after editing the master; its small standalone dependencies
are listed in `scripts/requirements-brand.txt`.

`_static/custom.css` defines the graphite/orange documentation palette and
typography. The logo orange is reserved for artwork; text links use darker
orange on white and lighter orange in dark mode for contrast. Use the theme's
semantic colors for notes, warnings, and errors. Check light and dark themes at
desktop and mobile widths when changing styles, including long code lines,
tables, keyboard focus, and the navigation menu. `_templates/layout.html` adds
the adaptive SVG favicon and Apple touch icon alongside Sphinx's ICO fallback.

Task-oriented recipes are grouped by `docs/workflows/index.md`, even when an
existing source file remains under `docs/user_guide/`. The ten numbered
notebooks live under `docs/tutorial/`; the smoke command executes them before
nbsphinx renders their saved outputs and interactive viewers.

API pages are authored under `docs/api/`.

GitHub Actions builds documentation once per pull request. Same-repository pull
requests publish under `previews/pr-<number>/`; pushes to `master` deploy the
current documentation to `latest/`. The stable public URL
<https://uw-ipd.github.io/tmol/> redirects to that deployment.
