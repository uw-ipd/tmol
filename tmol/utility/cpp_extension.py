from ..extern import include_paths as extern_include_paths
from .. import include_paths as tmol_include_paths

import torch.utils.cpp_extension


_default_include_paths = tmol_include_paths() + extern_include_paths()


def load(
    name,
    sources,
    extra_cflags=("-O3",),
    extra_cuda_cflags=(),
    extra_ldflags=(),
    extra_include_paths=(),
    build_directory=None,
    verbose=False,
):
    """Jit-compile torch cpp_extension with tmol paths."""

    return torch.utils.cpp_extension.load(
        name=name,
        sources=sources,
        extra_cflags=list(extra_cflags),
        extra_cuda_cflags=list(extra_cuda_cflags),
        extra_ldflags=list(extra_ldflags),
        extra_include_paths=list(extra_include_paths) + _default_include_paths,
        build_directory=build_directory,
        verbose=verbose,
    )
