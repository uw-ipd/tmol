#!/usr/bin/env python3
"""Install JAX release candidates in isolation and exercise native CUDA."""

import argparse
from pathlib import Path
import subprocess
import tempfile
import tomllib
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check(*command, cwd):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def owned_packages(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return {
            name.partition("/")[0]
            for name in archive.namelist()
            if "/" in name and not name.partition("/")[0].endswith(".dist-info")
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--native-wheel", type=Path, required=True)
    parser.add_argument("--backend", choices=("cu12", "cu13"), required=True)
    args = parser.parse_args()

    pure_owned = owned_packages(args.wheel)
    native_owned = owned_packages(args.native_wheel)
    assert pure_owned == {"tmol_jax"}, pure_owned
    assert native_owned == {"tmol_jax_cuda"}, native_owned
    assert pure_owned.isdisjoint(native_owned)

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    with tempfile.TemporaryDirectory(prefix=f"tmol-jax-{args.backend}-") as temporary:
        work = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(work / "env")
        python = work / "env/bin/python"
        pip = [python, "-m", "pip", "--isolated"]
        extra = "cuda12" if args.backend == "cu12" else "cuda13"
        check(*pip, "install", "--upgrade", "pip", cwd=work)
        check(*pip, "install", f"jax[{extra}]==0.11.2", cwd=work)
        check(*pip, "install", args.wheel.resolve(), cwd=work)
        check(
            *pip,
            "install",
            "--no-index",
            "--no-deps",
            args.native_wheel.resolve(),
            cwd=work,
        )
        check(*pip, "check", cwd=work)
        check(*pip, "uninstall", "-y", "tmol-jax-cuda", cwd=work)
        check(python, "-I", "-c", "import tmol_jax", cwd=work)
        check(
            *pip,
            "install",
            "--no-index",
            "--no-deps",
            args.native_wheel.resolve(),
            cwd=work,
        )
        check(
            python,
            "-I",
            ROOT / "scripts/smoke_installed_jax.py",
            "--pdb",
            ROOT / "tmol/tests/data/pdb/1ubq.pdb",
            "--version",
            version,
            "--device",
            "cuda",
            "--expect-no-torch",
            "--require-native-cuda",
            cwd=work,
        )


if __name__ == "__main__":
    main()
