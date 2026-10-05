#!/usr/bin/env python3
"""Prepare standard CPU wheels for PyPI and explicit ABI indexes for GitHub."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import html
import io
import re
import shutil
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import quote

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename

from release_matrix import CPU_TORCH_VERSIONS


def validate_wheel(source: Path) -> None:
    """Check identity, Torch ABI and RECORD hashes before publishing any variant."""
    name, version, _, tags = parse_wheel_filename(source.name)
    abi = re.fullmatch(r"(?:cpu|cu\d+)torch(\d+\.\d+)", version.local or "")
    if name != "tmol" or abi is None:
        raise ValueError(f"expected an ABI-qualified TMol wheel: {source.name}")
    prefix = f"tmol-{version}.dist-info/"
    with zipfile.ZipFile(source) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(names) != len(set(names)) or any(
            p.startswith("/") or ".." in Path(p).parts for p in names
        ):
            raise ValueError("unsafe or duplicate wheel paths")
        metadata = BytesParser().parsebytes(archive.read(prefix + "METADATA"))
        if metadata["Name"] != name or metadata["Version"] != str(version):
            raise ValueError("wheel filename and metadata disagree")
        requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist", [])]
        if [str(r.specifier) for r in requirements if r.name == "torch"] != [
            f"=={abi[1]}.*"
        ]:
            raise ValueError("wheel must constrain its compiled PyTorch minor")
        if any(r.url for r in requirements):
            raise ValueError("release dependencies must come from package indexes")
        wheel_metadata = BytesParser().parsebytes(archive.read(prefix + "WHEEL"))
        if set(wheel_metadata.get_all("Tag", [])) != {str(tag) for tag in tags}:
            raise ValueError("wheel filename and internal compatibility tags disagree")
        record = prefix + "RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        if len(rows) != len(names) or {row[0] for row in rows} != set(names):
            raise ValueError("wheel RECORD does not cover exactly its contents")
        for path, digest, size in rows:
            if path == record:
                if digest or size:
                    raise ValueError("RECORD must not hash itself")
                continue
            with archive.open(path) as stream:
                computed = hashlib.file_digest(stream, "sha256").digest()
            if digest != "sha256=" + base64.urlsafe_b64encode(computed).rstrip(
                b"="
            ).decode() or size != str(archive.getinfo(path).file_size):
                raise ValueError(f"invalid wheel RECORD: {path}")


def public_cpu_wheel(source: Path, destination: Path) -> Path:
    """Copy a CPU wheel under its public version, retaining its Torch constraint.

    CUDA variants keep their local versions and are distributed separately.
    Rebuild RECORD after changing the filename, dist-info path, and metadata.
    Native libraries and all other package files are copied byte for byte.
    """
    name, version, _, _ = parse_wheel_filename(source.name)
    if name != "tmol" or version.local not in {
        f"cputorch{torch}" for torch in CPU_TORCH_VERSIONS
    }:
        raise ValueError(f"not a supported default CPU wheel: {source.name}")
    public = version.public
    old_prefix, new_prefix = f"tmol-{version}", f"tmol-{public}"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name.replace(old_prefix, new_prefix, 1)
    if target.exists():
        raise ValueError(f"duplicate public wheel: {target.name}")
    with zipfile.ZipFile(source) as archive:
        metadata_path = old_prefix + ".dist-info/METADATA"
        metadata = archive.read(metadata_path)
        parsed = BytesParser().parsebytes(metadata)
        if parsed["Name"] != "tmol" or parsed["Version"] != str(version):
            raise ValueError("wheel filename and metadata disagree")
        torch = version.local.removeprefix("cputorch")
        requirements = [Requirement(r) for r in parsed.get_all("Requires-Dist", [])]
        if [str(r.specifier) for r in requirements if r.name == "torch"] != [
            f"=={torch}.*"
        ]:
            raise ValueError("CPU wheel must constrain its compiled PyTorch minor")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            p.startswith("/") or ".." in Path(p).parts for p in names
        ):
            raise ValueError("unsafe or duplicate wheel paths")
        if any(p.endswith(("/RECORD.jws", "/RECORD.p7s")) for p in names):
            raise ValueError("cannot rewrite a signed wheel")
        expected = {
            path: (digest, size)
            for path, digest, size in csv.reader(
                io.StringIO(archive.read(old_prefix + ".dist-info/RECORD").decode())
            )
        }
        rows = []
        try:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
                for info in archive.infolist():
                    if info.is_dir() or info.filename.endswith(".dist-info/RECORD"):
                        continue
                    data = archive.read(info)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    if expected.get(info.filename) != (
                        "sha256=" + digest.rstrip(b"=").decode(),
                        str(len(data)),
                    ):
                        raise ValueError(f"invalid wheel RECORD: {info.filename}")
                    if info.filename == metadata_path:
                        data = re.sub(
                            rb"(?m)^Version: [^\r\n]+",
                            f"Version: {public}".encode(),
                            data,
                            count=1,
                        )
                    renamed = copy.copy(info)
                    for suffix in (".dist-info/", ".data/"):
                        if renamed.filename.startswith(old_prefix + suffix):
                            renamed.filename = (
                                new_prefix + renamed.filename[len(old_prefix) :]
                            )
                    output.writestr(renamed, data)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    rows.append(
                        (
                            renamed.filename,
                            "sha256=" + digest.rstrip(b"=").decode(),
                            len(data),
                        )
                    )
                record = new_prefix + ".dist-info/RECORD"
                rows.append((record, "", ""))
                buffer = io.StringIO(newline="")
                csv.writer(buffer, lineterminator="\n").writerows(rows)
                output.writestr(record, buffer.getvalue())
        except Exception:
            target.unlink(missing_ok=True)
            raise
    return target


def write_index(files: list[Path], target: Path, base_url: str) -> None:
    """Write a static pip link page with content hashes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    links = []
    for path in sorted(files):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        url = base_url.rstrip("/") + "/" + quote(path.name) + "#sha256=" + digest
        links.append(
            f'<a href="{html.escape(url, quote=True)}">{html.escape(path.name)}</a><br>'
        )
    target.write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        "<title>TMol wheels</title></head><body>\n"
        + "\n".join(links)
        + "\n</body></html>\n",
        encoding="utf-8",
    )


def prepare(wheels: Path, sdist: Path, destination: Path) -> None:
    """Prepare public distributions and versioned indexes from validated assets."""
    paths = sorted(wheels.glob("*.whl"))
    versions = {parse_wheel_filename(p.name)[1].public for p in paths}
    if len(versions) != 1:
        raise ValueError("expected wheel assets for one release")
    version = versions.pop()
    if sdist.name != f"tmol-{version}.tar.gz":
        raise ValueError("source and wheel versions disagree")
    pypi = destination / "pypi"
    pypi.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sdist, pypi / sdist.name)
    variants: dict[str, list[Path]] = {}
    for path in paths:
        validate_wheel(path)
        local = parse_wheel_filename(path.name)[1].local
        if local is None:
            raise ValueError("GitHub wheels must identify their ABI")
        variants.setdefault(local, []).append(path)
        if local.startswith("cputorch"):
            public_cpu_wheel(path, pypi)
    for variant, files in variants.items():
        write_index(
            files,
            destination / "wheels" / f"v{version}" / variant / "index.html",
            f"https://github.com/uw-ipd/tmol/releases/download/v{version}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.wheels, args.sdist, args.output)
