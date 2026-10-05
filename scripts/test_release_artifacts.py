"""Exercise pip's index resolver and wheel integrity, without a native build."""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile
from email.parser import BytesParser

import pytest

from release_artifacts import public_cpu_wheel, validate_wheel, write_index
from staging_index import serve

ROOT = Path(__file__).resolve().parents[1]


def wheel(directory, version="9.9.9+cputorch2.14", torch="==2.14.*", name="tmol"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{version}-py3-none-any.whl"
    info = f"{name}-{version}.dist-info"
    contents = {
        "tmol/__init__.py": b"# fixture package\n",
        # Binary bytes stand in for the extension; repackaging must preserve them.
        "tmol/_C.so": bytes(range(256)),
        f"{info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + (f"Requires-Dist: torch{torch}\n" if name == "tmol" else "")
            + "\nFixture\n"
        ).encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    rows = [
        (
            name,
            "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode(),
            len(data),
        )
        for name, data in contents.items()
    ]
    record = f"{info}/RECORD"
    rows.append((record, "", ""))
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    contents[record] = buffer.getvalue().encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return path


def resolve(work, *args, success=True):
    report = work / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--no-cache-dir",
            "--no-deps",
            "--report",
            str(report),
            *args,
        ],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return json.loads(report.read_text()) if success else result.stdout + result.stderr


def test_public_cpu_wheel_preserves_native_bytes_constraints_and_record(tmp_path):
    original = wheel(tmp_path / "original")
    public = public_cpu_wheel(original, tmp_path / "public")
    assert public.name == "tmol-9.9.9-py3-none-any.whl"
    with zipfile.ZipFile(original) as old, zipfile.ZipFile(public) as new:
        assert new.read("tmol/_C.so") == old.read("tmol/_C.so")
        metadata = BytesParser().parsebytes(new.read("tmol-9.9.9.dist-info/METADATA"))
        assert metadata["Version"] == "9.9.9"
        assert metadata.get_all("Requires-Dist") == ["torch==2.14.*"]
        rows = list(
            csv.reader(io.StringIO(new.read("tmol-9.9.9.dist-info/RECORD").decode()))
        )
        assert {row[0] for row in rows} == set(new.namelist())
        for name, digest, size in rows[:-1]:
            data = new.read(name)
            assert int(size) == len(data)
            assert (
                digest
                == "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )


@pytest.mark.parametrize(
    "local, torch",
    [
        ("cu132torch2.14", "==2.14.*"),
        ("cputorch2.13", "==2.13.*"),
        ("cputorch2.14", ">=2.8"),
    ],
)
def test_public_wheel_rejects_wrong_cuda_or_torch(tmp_path, local, torch):
    with pytest.raises(ValueError):
        public_cpu_wheel(
            wheel(tmp_path / "files", f"9.9.9+{local}", torch), tmp_path / "out"
        )


def test_public_wheel_rejects_corrupt_record(tmp_path):
    original = wheel(tmp_path / "files")
    with zipfile.ZipFile(original) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents["tmol/_C.so"] = b"corrupted"
    with zipfile.ZipFile(original, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="RECORD"):
        public_cpu_wheel(original, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.whl"))


def test_pip_resolves_public_cpu_and_explicit_cuda_indexes(tmp_path):
    cpu = public_cpu_wheel(wheel(tmp_path / "original"), tmp_path / "files")
    cuda = wheel(tmp_path / "files", "9.9.9+cu132torch2.14")
    with serve(tmp_path) as url:
        write_index([cpu], tmp_path / "simple/tmol/index.html", f"{url}/files")
        write_index([cuda], tmp_path / "cu132torch2.14/index.html", f"{url}/files")
        ordinary = resolve(tmp_path, "--index-url", f"{url}/simple", "tmol==9.9.9")
        assert ordinary["install"][0]["metadata"]["version"] == "9.9.9"
        gpu = resolve(
            tmp_path,
            "--index-url",
            f"{url}/simple",
            "--find-links",
            f"{url}/cu132torch2.14/",
            "--only-binary=tmol",
            "tmol==9.9.9+cu132torch2.14",
        )
        assert gpu["install"][0]["metadata"]["version"] == "9.9.9+cu132torch2.14"
        # Exact variant and binary-only selection must fail, not fall back to CPU.
        failure = resolve(
            tmp_path,
            "--index-url",
            f"{url}/simple",
            "--find-links",
            f"{url}/cu132torch2.14/",
            "--only-binary=tmol",
            "tmol==9.9.9+cu130torch2.13",
            success=False,
        )
        assert "No matching distribution" in failure
        cpu.write_bytes(cpu.read_bytes() + b"tampered")
        failure = resolve(
            tmp_path, "--index-url", f"{url}/simple", "tmol==9.9.9", success=False
        )
        assert "HASHES" in failure


def test_pip_rejects_torch_abi_conflict(tmp_path):
    fake_torch = wheel(tmp_path / "files", "2.13.0", name="torch")
    cpu = public_cpu_wheel(wheel(tmp_path / "original"), tmp_path / "files")
    with serve(tmp_path) as url:
        write_index([cpu], tmp_path / "simple/tmol/index.html", f"{url}/files")
        write_index([fake_torch], tmp_path / "simple/torch/index.html", f"{url}/files")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--no-cache-dir",
                "--index-url",
                f"{url}/simple",
                "tmol==9.9.9",
                "torch==2.13.0",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "ResolutionImpossible" in result.stderr
        assert "torch==2.14.*" in result.stdout


def test_indexed_sdist_metadata_retains_public_candidate_version(tmp_path):
    # This is the real source tree and real backend, not a mock hook. Pip must
    # accept an sdist discovered via a package index before starting compilation.
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--skip-dependency-check",
            "--outdir",
            str(tmp_path / "files"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    archive = next((tmp_path / "files").glob("*.tar.gz"))
    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    with serve(tmp_path) as url:
        write_index([archive], tmp_path / "simple/tmol/index.html", f"{url}/files")
        result = resolve(
            tmp_path,
            "--index-url",
            f"{url}/simple",
            "--no-binary=tmol",
            "--no-build-isolation",
            f"tmol=={expected}",
        )
        assert result["install"][0]["metadata"]["version"] == expected
        assert result["install"][0]["download_info"]["url"].endswith(archive.name)


def test_validates_cuda_identity_and_metadata_before_publication(tmp_path):
    valid = wheel(tmp_path, "9.9.9+cu132torch2.14")
    validate_wheel(valid)
    with zipfile.ZipFile(valid) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    metadata = "tmol-9.9.9+cu132torch2.14.dist-info/METADATA"
    entries[metadata] = entries[metadata].replace(
        b"Version: 9.9.9+cu132torch2.14", b"Version: 9.9.9"
    )
    with zipfile.ZipFile(valid, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="filename and metadata"):
        validate_wheel(valid)
