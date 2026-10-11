#!/usr/bin/env python3
"""Prepare public CPU wheels, staged CUDA candidates, and explicit ABI indexes."""

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


def _public_wheel(source: Path, destination: Path, *, compact_cuda=False) -> Path:
    """Rewrite one validated wheel's identity and RECORD, retaining runtime bytes."""
    validate_wheel(source)
    _, version, _, _ = parse_wheel_filename(source.name)
    distribution = "tmol_cu130_torch213" if compact_cuda else "tmol"
    old_prefix = f"tmol-{version}"
    new_prefix = f"{distribution}-{version.public}"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name.replace(old_prefix, new_prefix, 1)
    if target.exists():
        raise ValueError(f"refusing to overwrite {target}")
    rows = []
    try:
        with (
            zipfile.ZipFile(source) as archive,
            zipfile.ZipFile(
                target,
                "w",
                zipfile.ZIP_DEFLATED,
                compresslevel=9 if compact_cuda else None,
            ) as output,
        ):
            names = archive.namelist()
            if any(p.endswith(("/RECORD.jws", "/RECORD.p7s")) for p in names):
                raise ValueError("cannot rewrite a signed wheel")
            if (
                compact_cuda
                and sum(p.startswith("tmol/_C") and p.endswith(".so") for p in names)
                != 1
            ):
                raise ValueError("expected exactly one TMol native library")
            for info in archive.infolist():
                if (
                    info.is_dir()
                    or info.filename == old_prefix + ".dist-info/RECORD"
                    or (compact_cuda and info.filename.startswith("tmol/tests/"))
                ):
                    continue
                data = archive.read(info)
                if info.filename == old_prefix + ".dist-info/METADATA":
                    for field, value in (
                        ("Name", distribution.replace("_", "-")),
                        ("Version", version.public),
                    ):
                        data = re.sub(
                            rf"(?m)^{field}: [^\r\n]+".encode(),
                            f"{field}: {value}".encode(),
                            data,
                            count=1,
                        )
                renamed = copy.copy(info)
                for suffix in (".dist-info/", ".data/"):
                    if renamed.filename.startswith(old_prefix + suffix):
                        renamed.filename = (
                            new_prefix + renamed.filename[len(old_prefix) :]
                        )
                output.writestr(
                    renamed,
                    data,
                    compress_type=(
                        zipfile.ZIP_DEFLATED if compact_cuda else info.compress_type
                    ),
                    compresslevel=9 if compact_cuda else None,
                )
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
                    b"="
                )
                rows.append((renamed.filename, "sha256=" + digest.decode(), len(data)))
            record = new_prefix + ".dist-info/RECORD"
            rows.append((record, "", ""))
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            output.writestr(record, buffer.getvalue())
        if compact_cuda and target.stat().st_size > 100_000_000:
            raise ValueError(f"wheel exceeds PyPI's default file limit: {target.name}")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def public_cpu_wheel(source: Path, destination: Path) -> Path:
    """Remove the CPU ABI version suffix, preserving all files and Torch pins."""
    name, version, _, _ = parse_wheel_filename(source.name)
    if name != "tmol" or version.local not in {
        f"cputorch{torch}" for torch in CPU_TORCH_VERSIONS
    }:
        raise ValueError(f"not a supported default CPU wheel: {source.name}")
    return _public_wheel(source, destination)


def pypi_gpu_wheel(source: Path, destination: Path) -> Path:
    """Stage the compact CUDA 13 / Torch 2.13 candidate under its own PyPI name."""
    name, version, _, tags = parse_wheel_filename(source.name)
    if (
        name != "tmol"
        or version.local != "cu130torch2.13"
        or {str(tag) for tag in tags} != {"cp312-cp312-manylinux_2_28_x86_64"}
    ):
        raise ValueError("expected a CUDA 13 / Torch 2.13 Linux x86_64 wheel")
    return _public_wheel(source, destination, compact_cuda=True)


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheels", type=Path)
    source.add_argument("--pypi-cuda-wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pypi_cuda_wheel:
        if args.sdist:
            parser.error("--sdist is only used with --wheels")
        print(pypi_gpu_wheel(args.pypi_cuda_wheel, args.output))
    else:
        if not args.sdist:
            parser.error("--wheels requires --sdist")
        prepare(args.wheels, args.sdist, args.output)
