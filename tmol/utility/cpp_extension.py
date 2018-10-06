from functools import wraps
import warnings

from ..extern import include_paths as extern_include_paths
from .. import include_paths as tmol_include_paths

import torch.utils.cpp_extension


warnings.filterwarnings(
    "ignore", message=r"(\n|.)*\(c\+\+\) may be ABI-incompatible with PyTorch"
)

_default_include_paths = tmol_include_paths() + extern_include_paths()


@wraps(torch.utils.cpp_extension.load)
def load(*args, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""

    kwargs["extra_cflags"] = kwargs.get("extra_cflags", ["-O3"])
    kwargs["extra_include_paths"] = (
        kwargs.get("extra_include_flags", []) + _default_include_paths
    )

    return torch.utils.cpp_extension.load(*args, **kwargs)


@wraps(torch.utils.cpp_extension.load_inline)
def load_inline(*args, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""

    kwargs["extra_cflags"] = kwargs.get("extra_cflags", ["-O3"])
    kwargs["extra_include_paths"] = (
        kwargs.get("extra_include_flags", []) + _default_include_paths
    )

    return torch.utils.cpp_extension.load_inline(*args, **kwargs)
