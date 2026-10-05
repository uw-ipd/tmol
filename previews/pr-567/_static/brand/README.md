# TMol brand assets

The logo combines a graphite protein helix with a short orange arrow whose
straight shaft has a constant width and a square tail. It suggests force and
acceleration. An orange `t` continues that accent in the
wordmark. Keep the ribbon orientation; do not mirror the symbol. This is a
stylized identity mark, not a molecular structure diagram.

The square symbol omits text, labels, motion streaks, and atomic detail so the
same compact geometry can serve as a favicon.

## Choose an asset

| Use | Asset |
| --- | --- |
| White or pale background | `tmol-logo-light.svg` |
| Dark background | `tmol-logo-dark.svg` |
| Avatar or standalone symbol | `tmol-mark-light.svg` or `tmol-mark-dark.svg` |
| One-color print | `tmol-logo-black.svg` or `tmol-logo-white.svg` |
| One-color symbol | `tmol-mark-black.svg` or `tmol-mark-white.svg` |
| Browser favicon | `favicon.svg`, with `favicon.ico` as a fallback |
| Apple home-screen icon | `apple-touch-icon.png` (180 × 180) |

Names describe the **background**: “light” goes on light backgrounds.
Transparent logo PNGs are supplied at widths of 512, 1024, and 2048 pixels.
Transparent symbol PNGs are supplied at 64, 128, 256, and 512 pixels square.
The ICO contains 16, 32, and 48 pixel frames; matching PNGs are also supplied.
Legacy favicons and the Apple icon have a white backing to keep the graphite
visible in both light and dark browser chrome. The SVG favicon adapts to the
browser's color-scheme preference independently of the site's theme switcher.

## Palette and typography

- Orange: **#F46B35** for the force vector and `t`.
- Graphite: **#263238** for the light-background logo.
- Pale foreground: **#EDF1F5** for the dark-background logo.

Each color logo uses two solid foreground colors. Transparent antialiased
edges naturally contain intermediate pixel values in PNG exports. The wordmark
is outlined artwork: it does not require a font installation and should not
be retyped in a substitute font. Use the SVG for publication and resizing.

Preserve the aspect ratio and the built-in clear space. Prefer the full
wordmark at 128 CSS pixels wide or larger; use the symbol at smaller sizes.
Avoid placing the logo over detailed photographs. Use the monochrome variant
when only one ink is available.

## Edit and regenerate

`tmol-logo-light.svg` is the canonical vector master. Its `mark` and `wordmark`
groups contain only paths, with `ink` and `accent` classes. There are no linked
images, external fonts, scripts, or gradients. Derived SVGs and PNGs are
generated from that single geometry; they are not separate AI redraws.

The initial concept was created with OpenAI image generation, selected by the
maintainer, then traced and cleaned into this two-color vector master. The
short force arrow was subsequently drawn as a native SVG path. Assets are
distributed under the repository's Apache-2.0 license.

From the repository root, with Python and the Cairo shared library installed:

```bash
python -m venv .venv-brand
.venv-brand/bin/python -m pip install -r scripts/requirements-brand.txt
.venv-brand/bin/python scripts/generate_brand_assets.py
```

To inspect exports without updating tracked files, use
`--output-dir /path/to/preview`. Edit the master first, regenerate, then inspect
both backgrounds and the actual 16/32 pixel icons before committing.
