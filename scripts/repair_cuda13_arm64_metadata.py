#!/usr/bin/env python3
"""Correct the known cuSPARSELt 0.8.0/0.8.1 ARM64 WHEEL tag in this environment.

PyTorch pins these NVIDIA releases. Their aarch64 wheel filenames are correct,
but their internal tag says ``manylinux2014_sbsa``, which pip cannot recognize.
Only WHEEL and its RECORD entry are changed; version, requirements and native
libraries are preserved. Run ``python -m pip check`` afterwards as usual.

Fingerprints below come from the original wheels and their RECORD files at
https://pypi.nvidia.com/nvidia-cusparselt-cu13/ (also mirrored on PyPI).
"""

import base64
import csv
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import platform
import stat
import tempfile

PACKAGE = "nvidia-cusparselt-cu13"
BAD_TAG = b"Tag: py3-none-manylinux2014_sbsa\n"
GOOD_TAG = b"Tag: py3-none-manylinux2014_aarch64\n"
WHEEL_SHA256 = "f44130c350c4a5261dabead1793c18f3c1b9d54a075af0015ae445e92cbff01b"
LIBRARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"
# Full upstream wheel SHA-256, installed METADATA SHA-256, and library RECORD
# digest. Never extend this list merely to silence a new pip check failure.
KNOWN = {
    "0.8.0": (
        "400c6ed1cf6780fc6efedd64ec9f1345871767e6a1a0a552a1ea0578117ea77c",
        "72e9b5bb91a38047d99145a1a802ad947ff555431e3bf33751cb31a5fa91f6e2",
        "IN5rNsu_V1OyaCbux1OiAFYI4Uy8O_3UaBE4CYvoaUE",
    ),
    "0.8.1": (
        "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f",
        "353669bafb80cbf22f1324121ac05049d36a6b0fb44b1fed3fab00b513231171",
        "-7A7v40iOGcPJCVsVFIoHCpmAWKJm4OLtNDMBtrQJ98",
    ),
}


def record_digest(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode(
        "ascii"
    ).rstrip("=")


def atomic_write(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def repair_distribution(dist):
    info = f"nvidia_cusparselt_cu13-{dist.version}.dist-info"
    wheel_name = f"{info}/WHEEL"
    wheel_path = Path(dist.locate_file(wheel_name))
    record_path = Path(dist.locate_file(f"{info}/RECORD"))
    old_wheel = wheel_path.read_bytes()
    if BAD_TAG not in old_wheel:
        return False
    if dist.version not in KNOWN:
        raise ValueError(f"Refusing to repair an unrecognized {PACKAGE} version")
    _, metadata_hash, library_hash = KNOWN[dist.version]
    if hashlib.sha256(old_wheel).hexdigest() != WHEEL_SHA256:
        raise ValueError("Refusing to change unrecognized WHEEL metadata")
    metadata = Path(dist.locate_file(f"{info}/METADATA")).read_bytes()
    if hashlib.sha256(metadata).hexdigest() != metadata_hash:
        raise ValueError("Refusing to change a distribution with different METADATA")
    with Path(dist.locate_file(LIBRARY)).open("rb") as stream:
        header = stream.read(20)
        # ELF64, little endian, EM_AARCH64 (183). Never retag an x86 library.
        if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\xb7\x00":
            raise ValueError("cuSPARSELt is not an AArch64 ELF64 library")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").digest()
    if base64.urlsafe_b64encode(digest).decode().rstrip("=") != library_hash:
        raise ValueError("cuSPARSELt library differs from the original NVIDIA wheel")
    if wheel_path.is_symlink() or record_path.is_symlink():
        raise ValueError("Refusing to replace symlinked package metadata")
    old_record = record_path.read_bytes()
    rows = list(csv.reader(io.StringIO(old_record.decode("utf-8"), newline="")))
    entries = [row for row in rows if row and row[0] == wheel_name]
    if entries != [[wheel_name, record_digest(old_wheel), str(len(old_wheel))]]:
        raise ValueError("WHEEL does not match its unique RECORD entry")
    new_wheel = old_wheel.replace(BAD_TAG, GOOD_TAG)
    entries[0][1:] = [record_digest(new_wheel), str(len(new_wheel))]
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    atomic_write(wheel_path, new_wheel)
    try:
        atomic_write(record_path, buffer.getvalue().encode("utf-8"))
    except BaseException:
        atomic_write(wheel_path, old_wheel)
        raise
    print(
        f"Corrected {PACKAGE} {dist.version} platform metadata: sbsa -> aarch64; "
        "verified original NVIDIA library SHA-256, native bytes unchanged."
    )
    return True


def main():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        print("cuSPARSELt ARM64 metadata repair is not needed on this platform.")
        return
    try:
        dist = importlib.metadata.distribution(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        print(f"{PACKAGE} is not installed; no metadata repair needed.")
        return
    if not repair_distribution(dist):
        print(f"{PACKAGE} has no known incorrect SBSA tag; metadata unchanged.")


if __name__ == "__main__":
    main()
