# TMol documentation

The documentation is built with Sphinx, MyST Markdown, nbsphinx, and autodoc.

Follow the [contributor guide](contributor_guide.md#documentation) to install
build dependencies, execute notebooks, and render the site to `docs/_build/html`.

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

`quickstart.md` introduces the common workflows; `terminology.md` explains the
concepts with interactive diagrams. Detailed task references remain under the
Quickstart in `workflows/index.md`. Notebook tutorials live in `tutorial/` and
API pages in `api/`. The smoke command executes the notebooks
before nbsphinx renders their outputs and viewers. Follow the
[writing guidelines](contributor_guide.md#writing-documentation) when editing prose.

## Interactive explanations

The glossary and playground keep their accessible markup in their Markdown
pages; `_static/tmol-explorers.js` supplies the interactions. The pinned local
3Dmol viewer and its license support offline viewing. One light/dark structure
image pair covers missing JavaScript, data, or WebGL; score controls still work
without WebGL. Keep the article background, compact captions, and thin rules.

The playground contains actual TMol scores, not a browser approximation.
Regenerate its checked-in dataset after changes to the scoring model:

```bash
python scripts/generate_score_playground.py
python scripts/generate_score_playground.py --verify
```

The generator rotates both Phe45 side-chain bonds in the bundled ubiquitin
structure, checks fixed atoms and bond lengths, and verifies weighted totals
and finite gradients. `--verify` compares every scored sample with the committed
dataset. CI also runs `.github/scripts/check_docs_explorers.cjs` against the
built site using Playwright 1.62.1. It checks exact displayed scores, keyboard
controls, URL state, dragging, local assets, mobile layout, light/dark themes,
and data/viewer failure states. To repeat it locally, set `NODE_PATH` to a
Playwright installation and run:

```bash
node .github/scripts/check_docs_explorers.cjs docs/_build/html
```

Optionally set `TMOL_BROWSER_EXECUTABLE` to an existing Chromium executable.

GitHub Actions builds documentation once per pull request. Same-repository pull
requests publish under `previews/pr-<number>/`; pushes to `master` deploy the
current documentation to `latest/`. The stable public URL
<https://uw-ipd.github.io/tmol/> redirects to that deployment.
