"""Repackage the compact CUDA 13 / Torch 2.13 wheel for PyPI."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

from packaging.utils import parse_wheel_filename

from release_artifacts import validate_wheel

ABI = "cu130torch2.13"
PACKAGE = "tmol_cu130_torch213"
PYPI_FILE_LIMIT = 100_000_000
TAG = "cp312-cp312-manylinux_2_28_x86_64"


def pypi_gpu_wheel(source: Path, destination: Path) -> Path:
    """Keep the full native library and runtime data under one public version."""
    validate_wheel(source)
    name, version, _, tags = parse_wheel_filename(source.name)
    if name != "tmol" or version.local != ABI or {str(tag) for tag in tags} != {TAG}:
        raise ValueError("expected a CUDA 13 / Torch 2.13 Linux x86_64 wheel")

    destination.mkdir(parents=True, exist_ok=True)
    old_prefix = f"tmol-{version}"
    new_prefix = f"{PACKAGE}-{version.public}"
    target = destination / f"{new_prefix}-{TAG}.whl"
    if target.exists():
        raise ValueError(f"refusing to overwrite {target}")

    rows = []
    native_count = 0
    try:
        with (
            zipfile.ZipFile(source) as archive,
            zipfile.ZipFile(
                target, "w", zipfile.ZIP_DEFLATED, compresslevel=9
            ) as output,
        ):
            for info in archive.infolist():
                if (
                    info.is_dir()
                    or info.filename.startswith("tmol/tests/")
                    or info.filename == old_prefix + ".dist-info/RECORD"
                ):
                    continue
                data = archive.read(info)
                if info.filename.startswith("tmol/_C") and info.filename.endswith(
                    ".so"
                ):
                    native_count += 1
                renamed = copy.copy(info)
                for suffix in (".dist-info/", ".data/"):
                    if renamed.filename.startswith(old_prefix + suffix):
                        renamed.filename = (
                            new_prefix + renamed.filename[len(old_prefix) :]
                        )
                if info.filename == old_prefix + ".dist-info/METADATA":
                    data = re.sub(
                        rb"(?m)^Name: tmol$",
                        b"Name: tmol-cu130-torch213",
                        data,
                        count=1,
                    )
                    data = re.sub(
                        rb"(?m)^Version: [^\r\n]+$",
                        f"Version: {version.public}".encode(),
                        data,
                        count=1,
                    )
                output.writestr(renamed, data)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
                    b"="
                )
                rows.append((renamed.filename, "sha256=" + digest.decode(), len(data)))
            record = new_prefix + ".dist-info/RECORD"
            if native_count != 1:
                raise ValueError("expected exactly one TMol native library")
            rows.append((record, "", ""))
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            output.writestr(record, buffer.getvalue())
        if target.stat().st_size > PYPI_FILE_LIMIT:
            raise ValueError(f"wheel exceeds PyPI's default file limit: {target.name}")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(pypi_gpu_wheel(args.wheel, args.output))
