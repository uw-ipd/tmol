#!/usr/bin/env python3
"""Export TMol's SVG master into theme, print, raster, and favicon variants.

Install the small, independent tool environment in requirements-brand.txt.
No TMol/PyTorch installation or font files are needed. Edit the SVG master,
then run this script from any directory. See docs/_static/brand/README.md.
"""

from __future__ import annotations

import argparse
import copy
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "docs" / "_static" / "brand"
SOURCE = BRAND / "tmol-logo-light.svg"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)

INK = "#263238"
PALE = "#EDF1F5"
ORANGE = "#F46B35"
LOGO_WIDTHS = (512, 1024, 2048)
MARK_SIZES = (64, 128, 256, 512)
FAVICON_SIZES = (16, 32, 48)
# Center the compact helix and short arrow in a square with clear space.
MARK_VIEWBOX = "-43 65 590 590"


def variant(master: ET.Element, ink: str, accent: str, mark=False) -> ET.Element:
    svg = copy.deepcopy(master)
    if mark:
        svg.remove(svg.find(f"{{{NS}}}g[@id='wordmark']"))
        svg.set("viewBox", MARK_VIEWBOX)
        svg.set("width", "590")
        svg.set("height", "590")
        svg.find(f"{{{NS}}}desc").text = (
            "TMol protein helix with a straight, constant-width orange arrow."
        )
    for path in svg.iter(f"{{{NS}}}path"):
        path.set("fill", ink if path.get("class") == "ink" else accent)
    return svg


def svg_bytes(svg: ET.Element) -> bytes:
    ET.indent(svg)
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True) + b"\n"


def write_svg(svg: ET.Element, name: str, dest: Path) -> bytes:
    content = svg_bytes(svg)
    (dest / name).write_bytes(content)
    return content


def raster(content: bytes, width: int) -> Image.Image:
    # Render at 4x for crisp, antialiased edges even at small sizes. PNGs
    # have fully opaque interiors and genuine transparency outside paths.
    png = cairosvg.svg2png(bytestring=content, output_width=width * 4)
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def tile(content: bytes, size: int) -> Image.Image:
    """White backing makes legacy favicons legible in either tab theme."""
    image = raster(content, size)
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    return background


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BRAND)
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    master = ET.parse(SOURCE).getroot()

    for theme, ink, accent in (
        ("light", INK, ORANGE),
        ("dark", PALE, ORANGE),
        ("black", "#000000", "#000000"),
        ("white", "#FFFFFF", "#FFFFFF"),
    ):
        for mark in (False, True):
            stem = f"tmol-{'mark' if mark else 'logo'}-{theme}"
            svg = variant(master, ink, accent, mark=mark)
            # Never overwrite the hand-editable source during generation.
            if stem == "tmol-logo-light" and dest == BRAND.resolve():
                content = SOURCE.read_bytes()
            else:
                content = write_svg(svg, f"{stem}.svg", dest)
            if theme in ("light", "dark"):
                for width in MARK_SIZES if mark else LOGO_WIDTHS:
                    raster(content, width).save(dest / f"{stem}-{width}.png")

    # Modern browsers select foreground color using their OS/browser theme.
    # This is independent of the documentation's in-page theme switcher.
    adaptive = variant(master, INK, ORANGE, mark=True)
    style = ET.Element(f"{{{NS}}}style")
    style.text = (
        f".ink {{ fill: {INK}; }} "
        f"@media (prefers-color-scheme: dark) {{ .ink {{ fill: {PALE}; }} }}"
    )
    adaptive.insert(2, style)
    write_svg(adaptive, "favicon.svg", dest)

    static = svg_bytes(variant(master, INK, ORANGE, mark=True))
    for size in FAVICON_SIZES:
        tile(static, size).save(dest / f"favicon-{size}.png")
    tile(static, 180).save(dest / "apple-touch-icon.png")
    # Store individual hand-rendered frames rather than letting Pillow
    # downsample the largest ICO frame for every smaller size.
    frames = [tile(static, size) for size in FAVICON_SIZES]
    frames[-1].save(
        dest / "favicon.ico",
        sizes=[(size, size) for size in FAVICON_SIZES],
        append_images=frames[:-1],
    )
    print(f"Brand assets exported to {dest}")


if __name__ == "__main__":
    main()
