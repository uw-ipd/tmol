#!/usr/bin/env python3
"""Install release candidates through a temporary package index and score."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv

from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from release_artifacts import validate_wheel, write_index
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
    parser.add_argument(
        "--base-wheel",
        type=Path,
        help="Matching tmol wheel for a CUDA operator candidate",
    )
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torch-index", default="https://pypi.org/simple")
    parser.add_argument("--default-torch", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tmol-install-") as temporary:
        work = Path(temporary)
        files = work / "files"
        files.mkdir()
        candidates = [args.wheel or args.sdist]
        if args.base_wheel:
            candidates.append(args.base_wheel)
        packages = {}
        for candidate in candidates:
            if candidate.suffix == ".whl":
                validate_wheel(candidate)
                name, version, _, _ = parse_wheel_filename(candidate.name)
            else:
                name, version = parse_sdist_filename(candidate.name)
            artifact = files / candidate.name
            shutil.copy2(candidate, artifact)
            packages[name] = (version, artifact)
        distribution = next(iter(packages))
        version = packages[distribution][0]
        if distribution != "tmol" and (
            "tmol" not in packages or packages["tmol"][0] != version
        ):
            parser.error("CUDA operators require --base-wheel at the same version")
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
            for name, (_, artifact) in packages.items():
                write_index(
                    [artifact], work / f"simple/{name}/index.html", f"{url}/files"
                )
            build = (
                ["--no-binary=tmol", "--no-build-isolation"]
                if args.sdist
                else ["--only-binary=" + ",".join(packages)]
            )
            check(
                *pip,
                "install",
                "--no-cache-dir",
                "--index-url",
                "https://pypi.org/simple",
                "--extra-index-url",
                f"{url}/simple/",
                *build,
                "--report",
                work / "install.json",
                f"{distribution}=={version}",
                cwd=work,
                env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL="2"),
            )
            installed = {
                canonicalize_name(item["metadata"]["name"]): item
                for item in json.loads((work / "install.json").read_text())["install"]
            }
            for name, (expected, _) in packages.items():
                assert installed[name]["download_info"]["url"].startswith(
                    f"{url}/files/"
                )
                assert installed[name]["metadata"]["version"] == str(expected)
        check(python, "-I", ROOT / "scripts/repair_cuda13_arm64_metadata.py", cwd=work)
        check(*pip, "check", cwd=work)
        if not args.default_torch:
            check(
                python,
                "-I",
                "-c",
                f'import torch; assert torch.__version__.split("+")[0] == {args.torch_version!r}',
                cwd=work,
            )
        check(
            python,
            "-I",
            "-c",
            f'import tmol; assert tmol.has_cuda() == {distribution != "tmol"}',
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
