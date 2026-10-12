#!/usr/bin/env python3
"""Build the optional tmol-jax CUDA wheel from canonical TMol headers."""

from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist/jax-cuda"))
    parser.add_argument(
        "--cuda-architectures",
        help="Semicolon-separated CUDA SM values; defaults to nvcc native",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "packages/jax-cuda"
    with tempfile.TemporaryDirectory(prefix="tmol-jax-cuda-build-") as temporary:
        stage = Path(temporary)
        shutil.copyfile(package / "pyproject.toml", stage / "pyproject.toml")
        shutil.copyfile(package / "setup.py", stage / "setup.py")
        shutil.copytree(package / "tmol_jax_cuda", stage / "tmol_jax_cuda")
        shutil.copyfile(package / "ljlk_cuda.cu", stage / "ljlk_cuda.cu")
        shutil.copytree(root / "tmol/score/common", stage / "include/tmol/score/common")
        shutil.copytree(
            root / "tmol/score/ljlk/potentials",
            stage / "include/tmol/score/ljlk/potentials",
        )
        shutil.copytree(root / "tmol/extern/Eigen", stage / "include/tmol/extern/Eigen")
        shutil.copytree(
            root / "tmol/extern/moderngpu",
            stage / "include/tmol/extern/moderngpu",
        )
        shutil.copyfile(root / "LICENSE", stage / "LICENSE")
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        (stage / "VERSION").write_text(metadata["project"]["version"] + "\n")
        environment = os.environ.copy()
        if args.cuda_architectures:
            environment["CMAKE_CUDA_ARCHITECTURES"] = args.cuda_architectures
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(args.outdir.resolve()),
                str(stage),
            ],
            check=True,
            env=environment,
        )


if __name__ == "__main__":
    main()
