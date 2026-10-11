#!/usr/bin/env python3
"""Install actual release candidates through pip, then score in a fresh venv.

Used by both pull-request and tag workflows. Never installs TMol by a local
wheel path: that would bypass the index candidate/metadata consistency check.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
import venv

from packaging.requirements import Requirement
from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)

from release_artifacts import (
    pypi_gpu_wheel,
    public_cpu_wheel,
    validate_wheel,
    write_index,
)
from staging_index import serve

ROOT = Path(__file__).resolve().parents[1]


def check(*command, cwd, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheel", type=Path)
    source.add_argument("--sdist", type=Path)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torch-index", default="https://pypi.org/simple")
    parser.add_argument("--default-torch", action="store_true")
    parser.add_argument(
        "--pypi-cuda",
        action="store_true",
        help="Repackage and test the renamed CUDA 13 / Torch 2.13 PyPI candidate",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.pypi_cuda and not args.wheel:
        parser.error("--pypi-cuda requires --wheel")
    with tempfile.TemporaryDirectory(prefix="tmol-install-") as temp:
        work = Path(temp)
        files = work / "files"
        files.mkdir()
        if args.wheel:
            validate_wheel(args.wheel)
            _, version, _, _ = parse_wheel_filename(args.wheel.name)
            local = version.local
            expected = {
                canonicalize_name(Requirement(r).name)
                for r in tomllib.loads((ROOT / "pyproject.toml").read_text())[
                    "project"
                ]["dependencies"]
            }
            with zipfile.ZipFile(args.wheel) as archive:
                metadata = BytesParser().parsebytes(
                    archive.read(f"tmol-{version}.dist-info/METADATA")
                )
            actual = {
                canonicalize_name(Requirement(r).name)
                for r in metadata.get_all("Requires-Dist", [])
                if "extra ==" not in r
            }
            assert actual == expected, actual ^ expected
            if args.pypi_cuda:
                artifact = pypi_gpu_wheel(args.wheel, files)
            elif local and local.startswith("cpu"):
                artifact = public_cpu_wheel(args.wheel, files)
            else:
                artifact = files / args.wheel.name
                shutil.copy2(args.wheel, artifact)
            distribution, version, _, _ = parse_wheel_filename(artifact.name)
        else:
            artifact = files / args.sdist.name
            shutil.copy2(args.sdist, artifact)
            distribution, version = parse_sdist_filename(artifact.name)
            local = None
        venv.EnvBuilder(with_pip=True).create(work / "env")
        python = work / "env/bin/python"
        pip = [python, "-m", "pip", "--isolated"]
        check(*pip, "install", "--upgrade", "pip", cwd=work)
        if not args.default_torch:
            check(
                *pip,
                "install",
                f"torch=={args.torch_version}",
                "--index-url",
                args.torch_index,
                cwd=work,
            )
        if args.sdist:
            check(
                *pip,
                "install",
                "scikit-build-core>=0.10",
                "cmake>=3.24,<4",
                "ninja",
                "packaging",
                "pybind11>=2.12",
                cwd=work,
            )
        with serve(work) as url:
            page = work / (
                "variant/index.html"
                if version.local
                else f"simple/{distribution}/index.html"
            )
            write_index([artifact], page, f"{url}/files")
            selection = (
                ["--find-links", f"{url}/variant/"]
                if version.local
                else ["--extra-index-url", f"{url}/simple/"]
            )
            build = (
                [
                    "--no-binary=tmol",
                    "--no-build-isolation",
                    "-Ccmake.define.TMOL_ENABLE_CUDA=OFF",
                ]
                if args.sdist
                else [f"--only-binary={distribution}"]
            )
            environment = dict(os.environ, MAX_JOBS=os.environ.get("MAX_JOBS", "2"))
            check(
                *pip,
                "install",
                "--no-cache-dir",
                "--index-url",
                "https://pypi.org/simple",
                *selection,
                *build,
                "--report",
                work / "install.json",
                f"{distribution}=={version}",
                cwd=work,
                env=environment,
            )
            report = json.loads((work / "install.json").read_text())
            installed = next(
                item
                for item in report["install"]
                if canonicalize_name(item["metadata"]["name"]) == distribution
            )
            assert installed["download_info"]["url"].startswith(
                f"{url}/files/"
            ), installed
            assert installed["metadata"]["version"] == str(version)
        # Use the same documented repair available to ARM64 CUDA 13 users.
        # It verifies the original NVIDIA library and changes only its bad tag;
        # pip check and the native scoring checks below remain mandatory.
        check(python, "-I", ROOT / "scripts/repair_cuda13_arm64_metadata.py", cwd=work)
        check(*pip, "check", cwd=work)
        if local and not args.default_torch:
            cuda = None if local.startswith("cpu") else local.split("torch")[0][2:]
            check(
                python,
                "-I",
                "-c",
                "import torch; "
                f"assert torch.__version__.split('+')[0] == {args.torch_version!r}; "
                f"assert (torch.version.cuda or '').replace('.', '') == {(cuda or '')!r}",
                cwd=work,
            )
        check(
            python,
            "-I",
            ROOT / "scripts/smoke_installed_package.py",
            "--distribution",
            distribution,
            "--version",
            version,
            "--pdb",
            ROOT / "tmol/tests/data/pdb/1ubq.pdb",
            "--device",
            args.device,
            cwd=work,
        )


if __name__ == "__main__":
    main()
