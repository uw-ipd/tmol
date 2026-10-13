#!/usr/bin/env python3
"""Validate and stage the CPU/API package and optional CUDA operator wheels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import shutil
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import quote

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename

REQUIRED_NEUTRAL_MIGRATION_EVIDENCE = frozenset(
    {
        "neutral-wheel-contents",
        "torch-free-import",
        "torch-only-install",
        "jax-only-install",
        "coexistence-install",
        "uninstall-ownership",
    }
)
# Publication remains fail-closed until every item above has retained evidence.
DECLARED_NEUTRAL_MIGRATION_EVIDENCE = frozenset()


def require_neutral_migration_ready(
    evidence=DECLARED_NEUTRAL_MIGRATION_EVIDENCE,
) -> None:
    missing = REQUIRED_NEUTRAL_MIGRATION_EVIDENCE - frozenset(evidence)
    if missing:
        raise ValueError(
            "framework-neutral package migration is not publication-ready; "
            f"missing evidence: {', '.join(sorted(missing))}"
        )


def _only_for_extra(requirement: Requirement, extra: str) -> bool:
    if requirement.marker is None or not requirement.marker.evaluate({"extra": extra}):
        return False
    return not any(
        requirement.marker.evaluate({"extra": other})
        for other in ("", "torch", "jax")
        if other != extra
    )


def _validate_base_dependencies(metadata, requirements: list[Requirement]) -> None:
    extras = set(metadata.get_all("Provides-Extra", []))
    if not {"torch", "jax"}.issubset(extras):
        raise ValueError("base wheel must declare torch and jax extras")
    tensor_requirements = [
        requirement
        for requirement in requirements
        if requirement.name in {"torch", "jax", "tmol-jax"}
    ]
    torch_requirements = [
        requirement
        for requirement in tensor_requirements
        if requirement.name == "torch"
    ]
    jax_requirements = [
        requirement for requirement in tensor_requirements if requirement.name == "jax"
    ]
    if (
        len(torch_requirements) != 1
        or len(jax_requirements) != 1
        or any(requirement.name == "tmol-jax" for requirement in tensor_requirements)
    ):
        raise ValueError(
            "base wheel must declare exactly one Torch and one JAX requirement"
        )
    torch = torch_requirements[0]
    if str(torch.specifier) != ">=2.11" or not _only_for_extra(torch, "torch"):
        raise ValueError("torch extra must target the stable PyTorch ABI")
    jax = jax_requirements[0]
    if str(jax.specifier) != ">=0.11.2" or not _only_for_extra(jax, "jax"):
        raise ValueError("jax extra must require the supported JAX runtime")


def _validate_operator_dependencies(
    requirements: list[Requirement], names: list[str], version
) -> None:
    torch = [r for r in requirements if r.name == "torch"]
    if (
        len(torch) != 1
        or str(torch[0].specifier) != ">=2.11"
        or torch[0].marker is not None
    ):
        raise ValueError(
            "operator wheel must target the stable PyTorch ABI (torch>=2.11)"
        )
    tmol = [r for r in requirements if r.name == "tmol"]
    if (
        len(tmol) != 1
        or str(tmol[0].specifier) != f"=={version}"
        or tmol[0].marker is not None
    ):
        raise ValueError("operator package must require the matching TMol release")
    if any(path.startswith("tmol/") for path in names):
        raise ValueError("operator package must not overwrite the tmol package")


def validate_wheel(source: Path) -> None:
    """Check package identity, dependency ownership, and RECORD hashes."""
    name, version, _, tags = parse_wheel_filename(source.name)
    if name not in {"tmol", "tmol-ops-cu12", "tmol-ops-cu13"} or version.local:
        raise ValueError(f"unexpected native package wheel: {source.name}")
    prefix = f"{name.replace('-', '_')}-{version}.dist-info/"
    with zipfile.ZipFile(source) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(names) != len(set(names)) or any(
            p.startswith("/") or ".." in Path(p).parts for p in names
        ):
            raise ValueError("unsafe or duplicate wheel paths")
        metadata = BytesParser().parsebytes(archive.read(prefix + "METADATA"))
        if metadata["Name"].replace("_", "-") != name or metadata["Version"] != str(
            version
        ):
            raise ValueError("wheel filename and metadata disagree")
        requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist", [])]
        if name == "tmol":
            _validate_base_dependencies(metadata, requirements)
        else:
            _validate_operator_dependencies(requirements, names, version)
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
    require_neutral_migration_ready()
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
        package = parse_wheel_filename(path.name)[0]
        variants.setdefault(package, []).append(path)
        shutil.copy2(path, pypi / path.name)
    for variant, files in variants.items():
        write_index(
            files,
            destination / "wheels" / f"v{version}" / variant / "index.html",
            f"https://github.com/RosettaCommons/tmol/releases/download/v{version}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.wheels, args.sdist, args.output)
