#!/usr/bin/env python3
"""Set a wheel's ABI qualifier and matching PyTorch dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
VERSION_RE = re.compile(r'(?m)^version = "([^"]+)"$')
TORCH_RE = re.compile(r'(?m)^(\s*)"torch[<>=!~][^"\n]*",$')


def main() -> None:
    """Write the release wheel's PEP 440 local version into pyproject.toml."""
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} LOCAL_TAG")
    abi = re.fullmatch(r"(?:cpu|cu\d+)torch(\d+\.\d+)", sys.argv[1])
    if abi is None:
        raise SystemExit("local tag must identify a CPU/CUDA PyTorch minor version")

    text = PYPROJECT.read_text(encoding="utf-8")
    matches = VERSION_RE.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"expected one project version in {PYPROJECT}")

    base_version = Version(matches[0])
    if base_version.local is not None:
        raise SystemExit(f"project version already has a local tag: {base_version}")
    wheel_version = Version(f"{base_version}+{sys.argv[1]}")
    text, count = TORCH_RE.subn(rf'\1"torch=={abi[1]}.*",', text)
    if count != 2:
        raise SystemExit("expected one runtime and one build PyTorch requirement")

    PYPROJECT.write_text(
        VERSION_RE.sub(f'version = "{wheel_version}"', text), encoding="utf-8"
    )
    print(wheel_version)


if __name__ == "__main__":
    main()
