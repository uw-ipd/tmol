"""Check real pip resolution and release wheel integrity."""

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

import pytest
from release_artifacts import validate_wheel, write_index
from staging_index import serve

ROOT = Path(__file__).resolve().parents[1]


def wheel(
    directory,
    version="9.9.9",
    torch=">=2.11",
    name="tmol",
    tag="py3-none-any",
    newline="\n",
):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{version}-{tag}.whl"
    info = f"{name}-{version}.dist-info"
    contents = {
        f"{name}/__init__.py": b"# fixture package\n",
        # Binary bytes stand in for the extension; repackaging must preserve them.
        f"{name}/_C.so": bytes(range(256)),
        f"{name}-{version}.data/data/share/tmol/params.bin": b"shared data",
        f"{info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + (f"Requires-Dist: torch{torch}\n" if name.startswith("tmol") else "")
            + (
                f"Requires-Dist: tmol=={version}\n"
                if name.startswith("tmol_ops")
                else ""
            )
            + "\nFixture\n"
        )
        .replace("\n", newline)
        .encode(),
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n".encode(),
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


@pytest.mark.parametrize("name", ["tmol", "tmol_ops_cu12", "tmol_ops_cu13"])
def test_validates_identity_constraints_and_record(tmp_path, name):
    valid = wheel(tmp_path, name=name)
    validate_wheel(valid)
    with zipfile.ZipFile(valid) as archive:
        entries = {path: archive.read(path) for path in archive.namelist()}
    entries[f"{name}/_C.so"] = b"corrupted"
    with zipfile.ZipFile(valid, "w") as archive:
        for path, data in entries.items():
            archive.writestr(path, data)
    with pytest.raises(ValueError, match="RECORD"):
        validate_wheel(valid)


@pytest.mark.parametrize("torch", ["==2.14.*", ">=2.8"])
def test_rejects_unstable_torch_constraints(tmp_path, torch):
    with pytest.raises(ValueError, match="stable PyTorch ABI"):
        validate_wheel(wheel(tmp_path, torch=torch))


def test_pip_resolves_packages_and_verifies_hashes(tmp_path):
    with serve(tmp_path) as url:
        for name in ("tmol", "tmol_ops_cu12", "tmol_ops_cu13"):
            candidate = wheel(tmp_path / "files", name=name)
            project = name.replace("_", "-")
            write_index(
                [candidate], tmp_path / f"simple/{project}/index.html", f"{url}/files"
            )
            report = resolve(
                tmp_path, "--index-url", f"{url}/simple", f"{project}==9.9.9"
            )
            assert report["install"][0]["metadata"]["version"] == "9.9.9"
        candidate.write_bytes(candidate.read_bytes() + b"tampered")
        failure = resolve(
            tmp_path,
            "--index-url",
            f"{url}/simple",
            "tmol-ops-cu13==9.9.9",
            success=False,
        )
        assert "HASHES" in failure


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
