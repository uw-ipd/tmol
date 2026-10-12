#!/usr/bin/env python3
"""Validate Torch-free tmol-jax wheel and sdist metadata/content."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import parse_sdist_filename, parse_wheel_filename


def requirements(metadata):
    return tuple(
        Requirement(value).name.lower()
        for value in metadata.get_all("Requires-Dist", failobj=[])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()

    wheel_name, wheel_version, _, _ = parse_wheel_filename(args.wheel.name)
    sdist_name, sdist_version = parse_sdist_filename(args.sdist.name)
    assert wheel_name == sdist_name == "tmol-jax"
    assert wheel_version == sdist_version

    with ZipFile(args.wheel) as archive:
        names = archive.namelist()
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert "torch" not in requirements(metadata)
        assert any(
            name.endswith("database/default/scoring/dunbrack.npz") for name in names
        )
        assert any(
            name.endswith("database/default/chemical/chemical.yaml") for name in names
        )

    with tarfile.open(args.sdist, "r:gz") as archive:
        names = archive.getnames()
        assert any(name.endswith("/pyproject.toml") for name in names)
        assert any(
            name.endswith("/tmol_jax/database/default/scoring/dunbrack.npz")
            for name in names
        )
        assert not any("/tmol/" in name for name in names)

    print(f"validated Torch-free tmol-jax {wheel_version} wheel and sdist")


if __name__ == "__main__":
    main()
