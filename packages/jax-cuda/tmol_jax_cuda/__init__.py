"""Location helper for the optional tmol-jax CUDA library."""

from pathlib import Path


def library_path() -> Path:
    libraries = tuple(Path(__file__).parent.glob("_ljlk_cuda*.so"))
    if len(libraries) != 1:
        raise RuntimeError("tmol-jax-cuda does not contain exactly one CUDA library")
    return libraries[0]


__all__ = ["library_path"]
