"""Build the optional tmol-jax CUDA extension with nvcc."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import jax
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CudaBuildExt(build_ext):
    def build_extension(self, extension):
        compiler = shutil.which("nvcc")
        if compiler is None:
            raise RuntimeError("nvcc is required to build tmol-jax-cuda")
        output = Path(self.get_ext_fullpath(extension.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        architectures = os.environ.get("CMAKE_CUDA_ARCHITECTURES", "native")
        architecture_flags = []
        for architecture in architectures.split(";"):
            if architecture == "native":
                architecture_flags.append("-arch=native")
            elif architecture:
                architecture_flags.append(
                    f"-gencode=arch=compute_{architecture},code=sm_{architecture}"
                )
        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            f"-I{jax.ffi.include_dir()}",
            "-Iinclude",
            "-Iinclude/tmol/extern",
            *architecture_flags,
            *extension.sources,
            "-o",
            str(output),
        ]
        subprocess.run(command, check=True)


setup(
    ext_modules=[
        Extension("tmol_jax_cuda._ljlk_cuda", ["ljlk_cuda.cu", "lkball_cuda.cu"])
    ],
    cmdclass={"build_ext": CudaBuildExt},
    install_requires=[
        "jax>=0.11.2",
        f"tmol-jax=={Path('VERSION').read_text().strip()}",
    ],
)
