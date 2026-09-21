#!/usr/bin/env python3
"""Add an ABI qualifier to the static project version for a wheel build."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
VERSION_RE = re.compile(r'(?m)^version = "([^"]+)"$')


def main() -> None:
    """Write the release wheel's PEP 440 local version into pyproject.toml."""
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} LOCAL_TAG")

    text = PYPROJECT.read_text(encoding="utf-8")
    matches = VERSION_RE.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"expected one project version in {PYPROJECT}")

    base_version = Version(matches[0])
    if base_version.local is not None:
        raise SystemExit(f"project version already has a local tag: {base_version}")
    wheel_version = Version(f"{base_version}+{sys.argv[1]}")

    PYPROJECT.write_text(
        VERSION_RE.sub(f'version = "{wheel_version}"', text), encoding="utf-8"
    )
    print(wheel_version)


if __name__ == "__main__":
    main()
