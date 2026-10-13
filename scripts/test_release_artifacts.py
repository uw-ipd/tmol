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
from release_artifacts import (
    REQUIRED_NEUTRAL_MIGRATION_EVIDENCE,
    prepare,
    require_neutral_migration_ready,
    validate_wheel,
    write_index,
)
from staging_index import serve

ROOT = Path(__file__).resolve().parents[1]


def wheel(
    directory,
    version="9.9.9",
    torch=">=2.11",
    name="tmol",
    tag="py3-none-any",
    newline="\n",
    root_torch_marker='; extra == "torch"',
    root_jax_marker='; extra == "jax"',
    duplicate_torch_marker=None,
    duplicate_jax_marker=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{version}-{tag}.whl"
    info = f"{name}-{version}.dist-info"
    metadata = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    if name == "tmol":
        metadata.extend(
            [
                "Provides-Extra: torch",
                "Provides-Extra: jax",
                f"Requires-Dist: torch{torch}{root_torch_marker}",
                f"Requires-Dist: jax>=0.11.2{root_jax_marker}",
            ]
        )
        if duplicate_torch_marker is not None:
            metadata.append(f"Requires-Dist: torch{torch}{duplicate_torch_marker}")
        if duplicate_jax_marker is not None:
            metadata.append(f"Requires-Dist: jax>=0.11.2{duplicate_jax_marker}")
    elif name.startswith("tmol_ops"):
        metadata.extend(
            [
                f"Requires-Dist: torch{torch}",
                f"Requires-Dist: tmol=={version}",
            ]
        )
    contents = {
        f"{name}/__init__.py": b"# fixture package\n",
        # Binary bytes stand in for the extension; repackaging must preserve them.
        f"{name}/_C.so": bytes(range(256)),
        f"{name}-{version}.data/data/share/tmol/params.bin": b"shared data",
        f"{info}/METADATA": ("\n".join(metadata) + "\n\nFixture\n")
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


def test_rejects_unconditional_root_tensor_dependencies(tmp_path):
    with pytest.raises(ValueError, match="torch extra"):
        validate_wheel(wheel(tmp_path, root_torch_marker=""))
    with pytest.raises(ValueError, match="jax extra"):
        validate_wheel(wheel(tmp_path, root_jax_marker=""))


@pytest.mark.parametrize(
    "duplicate",
    [
        {"duplicate_torch_marker": '; extra == "torch"'},
        {"duplicate_torch_marker": ""},
        {"duplicate_jax_marker": '; extra == "jax"'},
        {"duplicate_jax_marker": ""},
    ],
)
def test_rejects_duplicate_root_tensor_requirements(tmp_path, duplicate):
    with pytest.raises(ValueError, match="exactly one Torch and one JAX"):
        validate_wheel(wheel(tmp_path, **duplicate))


def test_neutral_migration_publication_gate_is_fail_closed():
    with pytest.raises(ValueError, match="not publication-ready") as error:
        require_neutral_migration_ready()
    for item in REQUIRED_NEUTRAL_MIGRATION_EVIDENCE:
        assert item in str(error.value)
    require_neutral_migration_ready(REQUIRED_NEUTRAL_MIGRATION_EVIDENCE)


def test_publication_preparation_uses_fail_closed_gate(tmp_path):
    with pytest.raises(ValueError, match="not publication-ready"):
        prepare(tmp_path / "wheels", tmp_path / "tmol-9.9.9.tar.gz", tmp_path / "out")


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
