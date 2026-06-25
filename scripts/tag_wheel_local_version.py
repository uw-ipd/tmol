#!/usr/bin/env python3
"""Rename dist wheels with a +local version tag and patch METADATA Version."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tmol_build_backend import _align_wheel_metadata_from_filename  # noqa: E402


def main() -> None:
    dist_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    local_tag = sys.argv[2]
    for whl in sorted(dist_dir.glob("*.whl")):
        if "+" in whl.name:
            _align_wheel_metadata_from_filename(whl)
            continue
        new_name = re.sub(r"^(tmol-[^-]+)-", rf"\1+{local_tag}-", whl.name, count=1)
        target = dist_dir / new_name
        whl.rename(target)
        _align_wheel_metadata_from_filename(target)
        print(new_name)


if __name__ == "__main__":
    main()
