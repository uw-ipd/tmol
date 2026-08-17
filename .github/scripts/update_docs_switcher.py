#!/usr/bin/env python3
"""Create pydata-sphinx-theme switcher files for versioned Pages docs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")


def _version_key(name: str) -> tuple[int, ...]:
    if name == "latest":
        return (10**9,)
    if name.startswith("v"):
        return tuple(int(part) for part in name[1:].split("."))
    return (0,)


def _collect_entries(
    pages_dir: Path,
    base_url: str,
    current_version_dir: str | None,
    current_display_name: str | None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    version_dirs = [
        path.name
        for path in pages_dir.iterdir()
        if path.is_dir() and VERSION_RE.match(path.name)
    ]
    version_dirs.sort(key=_version_key, reverse=True)

    for dirname in version_dirs:
        version = dirname[1:]
        entries.append(
            {
                "version": version,
                "url": f"{base_url}/{dirname}/index.html",
                "name": dirname,
                "preferred": dirname == current_version_dir,
            }
        )

    latest_dir = pages_dir / "latest"
    if latest_dir.is_dir():
        entries.append(
            {
                "version": "latest",
                "url": f"{base_url}/latest/index.html",
                "name": "latest",
            }
        )

    if current_version_dir and current_version_dir.startswith("previews/"):
        entries.insert(
            0,
            {
                "version": current_version_dir.replace("/", "-"),
                "url": f"{base_url}/{current_version_dir}/index.html",
                "name": current_display_name or current_version_dir,
            },
        )

    return entries


def _write_switchers(pages_dir: Path, entries: list[dict[str, object]]) -> None:
    payload = json.dumps(entries, indent=2) + "\n"
    targets = [
        path
        for path in pages_dir.iterdir()
        if path.is_dir() and (path.name == "latest" or VERSION_RE.match(path.name))
    ]
    previews = pages_dir / "previews"
    if previews.exists():
        targets.extend(path for path in previews.iterdir() if path.is_dir())

    for target in targets:
        static_dir = target / "_static"
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / "switcher.json").write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-dir", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--current-version-dir")
    parser.add_argument("--current-display-name")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    entries = _collect_entries(
        args.pages_dir,
        base_url,
        args.current_version_dir,
        args.current_display_name,
    )
    _write_switchers(args.pages_dir, entries)
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
