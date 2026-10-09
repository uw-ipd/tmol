"""Split the CUDA 13 / Torch 2.13 release wheel into PyPI-sized distributions.

Both distributions install disjoint files into ``tmol/``. The native library
stays at its original path, so its bytes and runtime search path are unchanged.
"""

from __future__ import annotations

import base64
import argparse
import csv
import hashlib
import io
import re
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import parse_wheel_filename

from release_artifacts import validate_wheel

ABI = "cu130torch2.13"
PACKAGES = ("tmol_cu130_torch213", "tmol_kernels_cu130_torch213")
NATIVE = re.compile(r"tmol/_C(?:\.[^/]+)?\.so$")
PYPI_FILE_LIMIT = 100_000_000


def _write_wheel(path: Path, contents: dict[str, bytes], record: str) -> None:
    rows = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name, data in contents.items():
            output.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
                b"="
            )
            rows.append((name, "sha256=" + digest.decode(), len(data)))
        rows.append((record, "", ""))
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        output.writestr(record, buffer.getvalue())
    if path.stat().st_size > PYPI_FILE_LIMIT:
        path.unlink()
        raise ValueError(f"wheel exceeds PyPI's default file limit: {path.name}")


def split_gpu_wheel(source: Path, destination: Path) -> tuple[Path, Path]:
    """Preserve the compiled library while separating it from runtime data."""
    validate_wheel(source)
    name, version, _, tags = parse_wheel_filename(source.name)
    if name != "tmol" or version.local != ABI or len(tags) != 1:
        raise ValueError("expected one CUDA 13 / Torch 2.13 TMol wheel")
    tag = str(next(iter(tags)))
    if tag != "cp312-cp312-manylinux_2_28_x86_64":
        raise ValueError(
            "the PyPI CUDA variant currently targets Linux x86_64 Python 3.12"
        )

    destination.mkdir(parents=True, exist_ok=True)
    old_info = f"tmol-{version}.dist-info/"
    outputs = tuple(
        destination / f"{package}-{version.public}-{tag}.whl" for package in PACKAGES
    )
    if any(path.exists() for path in outputs):
        raise ValueError("refusing to overwrite an existing wheel")

    core_info = f"{PACKAGES[0]}-{version.public}.dist-info/"
    native_info = f"{PACKAGES[1]}-{version.public}.dist-info/"
    core: dict[str, bytes] = {}
    native: dict[str, bytes] = {}
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            path = member.filename
            if (
                member.is_dir()
                or path.startswith("tmol/tests/")
                or path == old_info + "RECORD"
            ):
                continue
            data = archive.read(member)
            if NATIVE.fullmatch(path):
                native[path] = data
            elif path.startswith(old_info):
                renamed = core_info + path[len(old_info) :]
                if renamed.endswith("/METADATA"):
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
                    headers, separator, body = data.partition(b"\n\n")
                    if not separator:
                        raise ValueError("wheel metadata has no header/body separator")
                    data = (
                        headers
                        + f"\nRequires-Dist: tmol-kernels-cu130-torch213=={version.public}".encode()
                        + separator
                        + body
                    )
                core[renamed] = data
            else:
                core[path] = data
        if len(native) != 1 or core_info + "METADATA" not in core:
            raise ValueError(
                "expected one TMol native library and complete wheel metadata"
            )
        original_metadata = BytesParser().parsebytes(
            archive.read(old_info + "METADATA")
        )
        if "torch==2.13.*" not in original_metadata.get_all("Requires-Dist", []):
            raise ValueError("the compiled wheel must pin its Torch minor version")
        native[native_info + "METADATA"] = (
            "Metadata-Version: 2.3\n"
            "Name: tmol-kernels-cu130-torch213\n"
            f"Version: {version.public}\n"
            "Requires-Python: >=3.12,<3.13\n"
            "Requires-Dist: torch==2.13.*\n"
            "Description-Content-Type: text/plain\n"
            "\nNative CUDA 13 kernels for TMol and PyTorch 2.13.\n"
        ).encode()
        native[native_info + "WHEEL"] = (
            "Wheel-Version: 1.0\nGenerator: tmol release_artifacts\n"
            f"Root-Is-Purelib: false\nTag: {tag}\n"
        ).encode()

    try:
        _write_wheel(outputs[0], core, core_info + "RECORD")
        _write_wheel(outputs[1], native, native_info + "RECORD")
    except Exception:
        for path in outputs:
            path.unlink(missing_ok=True)
        raise
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in split_gpu_wheel(args.wheel, args.output):
        print(path)
