#!/usr/bin/env python3
"""Reject PyTorch C++ imports and unversioned C++ runtime imports on Linux."""

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check(wheel: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary, zipfile.ZipFile(wheel) as archive:
        libraries = [name for name in archive.namelist() if name.endswith(".so")]
        if not libraries:
            raise ValueError(f"No native libraries in {wheel}")
        for name in libraries:
            library = archive.extract(name, temporary)
            nm = (
                ["nm", "-D", "-u"] if sys.platform.startswith("linux") else ["nm", "-u"]
            )
            symbols = subprocess.check_output([*nm, library], text=True)
            symbols = subprocess.check_output(["c++filt"], input=symbols, text=True)
            forbidden = [
                line
                for line in symbols.splitlines()
                if re.search(r"\b(?:at|c10|torch)::", line)
                or (
                    sys.platform.startswith("linux")
                    and "std::" in line
                    and not re.search(r"@(?:GLIBCXX|CXXABI)_", line)
                )
            ]
            if forbidden:
                raise ValueError(
                    f"{name} imports unstable Torch or C++ runtime symbols:\n"
                    + "\n".join(forbidden)
                )
    print(f"{wheel.name}: stable native ABI ({len(libraries)} libraries)")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        check(Path(argument))
