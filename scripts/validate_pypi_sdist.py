#!/usr/bin/env python3
"""Reject source distributions with dependencies or versions PyPI disallows."""

from __future__ import annotations

import argparse
import tarfile
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename
from packaging.version import Version


def validate_sdist(path: Path) -> None:
    name, version = parse_sdist_filename(path.name)
    if name != "tmol" or version.local:
        raise ValueError(f"expected a public tmol source version: {path.name}")
    with tarfile.open(path) as archive:
        member = archive.getmember(f"tmol-{version}/PKG-INFO")
        if not member.isfile():
            raise ValueError("PKG-INFO must be a regular file")
        metadata = BytesParser().parsebytes(archive.extractfile(member).read())
    if canonicalize_name(metadata["Name"] or "") != name:
        raise ValueError("source filename and metadata names differ")
    if Version(metadata["Version"] or "0") != version:
        raise ValueError("source filename and metadata versions differ")
    for value in metadata.get_all("Requires-Dist", []):
        if Requirement(value).url:
            raise ValueError(f"PyPI does not accept direct URL dependencies: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    validate_sdist(args.sdist)
    print(f"Validated PyPI dependency metadata: {args.sdist.name}")


if __name__ == "__main__":
    main()
