#!/usr/bin/env python3
"""Generate a PEP 503 index over tmol's GitHub Release wheels.

tmol's wheels carry PEP 440 local version segments (``+cu130torch2.13``) so a
binary records the CUDA and torch minor it was compiled against. PyPI rejects
local versions, which is why ``publish.yml`` uploads only the sdist there and
attaches the wheels to the GitHub Release instead.

A PEP 503 "simple" index has no such restriction -- ``download.pytorch.org``
serves local versions, and tmol's own build already consumes that index. Its
links may be absolute URLs, so this index costs no storage: it points straight
at the release assets.

Every release is indexed, not just the newest, so ``tmol==0.1.55`` keeps
resolving after 0.1.56 ships. The layout is one simple-index per variant::

    <root>/<variant>/index.html          the project list
    <root>/<variant>/tmol/index.html     that variant's wheels, every version

Install::

    pip install tmol --extra-index-url https://uw-ipd.github.io/tmol/whl/cu130torch2.13/
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

# A wheel filename is dist-version-python-abi-platform.whl; only the version,
# whose local segment names the variant, matters here.
WHEEL = re.compile(r"^(?P<dist>[^-]+)-(?P<version>[^-]+)-.+\.whl$")
NO_VARIANT = "any"


def release_assets(repo: str) -> list[dict]:
    """Every asset of every release, newest release first."""
    releases = json.loads(
        subprocess.check_output(
            ["gh", "api", "--paginate", f"repos/{repo}/releases"], text=True
        )
    )
    return [asset for release in releases for asset in release["assets"]]


def variant_of(filename: str) -> str | None:
    """The local version segment of a wheel, or None if not a wheel."""
    match = WHEEL.match(filename)
    if not match:
        return None
    _, _, local = match.group("version").partition("+")
    return local or NO_VARIANT


def link(asset: dict) -> str:
    """One PEP 503 anchor, carrying the hash when the API reports one."""
    url = asset["browser_download_url"]
    digest = asset.get("digest")
    if digest and digest.startswith("sha256:"):
        url = f"{url}#sha256={digest.removeprefix('sha256:')}"
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(asset["name"])}</a><br>\n'


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<!DOCTYPE html>\n<html><body>\n{body}</body></html>\n", "utf-8")


def build(assets: list[dict], out: Path, project: str) -> dict[str, int]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for asset in assets:
        variant = variant_of(asset["name"])
        if variant:
            by_variant[variant].append(asset)
    for variant, wheels in by_variant.items():
        wheels.sort(key=lambda asset: asset["name"])
        write(out / variant / project / "index.html", "".join(map(link, wheels)))
        write(out / variant / "index.html", f'<a href="{project}/">{project}</a><br>\n')
    write(
        out / "index.html",
        "".join(f'<a href="{name}/">{name}</a><br>\n' for name in sorted(by_variant)),
    )
    return {variant: len(wheels) for variant, wheels in by_variant.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="uw-ipd/tmol")
    parser.add_argument("--out", type=Path, required=True, help="index root")
    parser.add_argument("--project", default="tmol")
    args = parser.parse_args()

    counts = build(release_assets(args.repo), args.out, args.project)
    if not counts:
        print(f"no wheels on any {args.repo} release")
        return 1
    for variant in sorted(counts):
        print(f"{variant:<22} {counts[variant]:3d} wheels")
    print(f"\n{len(counts)} variants -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
