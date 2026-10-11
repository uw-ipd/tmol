#!/usr/bin/env python3
"""Reject version-specific PyTorch C++ imports in a built runtime wheel."""

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
            symbols = subprocess.check_output(["nm", "-u", library], text=True)
            symbols = subprocess.check_output(["c++filt"], input=symbols, text=True)
            forbidden = [
                line
                for line in symbols.splitlines()
                if re.search(r"\b(?:at|c10|torch)::", line)
            ]
            if forbidden:
                raise ValueError(
                    f"{name} imports the unstable PyTorch C++ ABI:\n"
                    + "\n".join(forbidden)
                )
    print(f"{wheel.name}: stable native ABI ({len(libraries)} libraries)")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        check(Path(argument))
