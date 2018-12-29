from functools import wraps
import warnings

from ..extern import include_paths as extern_include_paths
from .. import include_paths as tmol_include_paths

import torch.utils.cpp_extension


# Add warning filter for use of c++ (rather than g++) for extension
# compilation. c++ is provided by g++ on our platform.
warnings.filterwarnings(
    "ignore",
    message=r"(\n|.)*"
    r"x86_64-conda_cos6-linux-gnu-c\+\+.*"
    r"may be ABI-incompatible with PyTorch(\n|.)*",
)

_default_include_paths = tmol_include_paths() + extern_include_paths()


@wraps(torch.utils.cpp_extension.load)
def load(*args, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""

    kwargs["extra_cflags"] = kwargs.get("extra_cflags", ["-O3", "--std=c++14"])
    kwargs["extra_include_paths"] = (
        kwargs.get("extra_include_flags", []) + _default_include_paths
    )

    return torch.utils.cpp_extension.load(*args, **kwargs)


@wraps(torch.utils.cpp_extension.load_inline)
def load_inline(*args, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""

    kwargs["extra_cflags"] = kwargs.get("extra_cflags", ["-O3", "--std=c++14"])
    kwargs["extra_include_paths"] = (
        kwargs.get("extra_include_flags", []) + _default_include_paths
    )

    return torch.utils.cpp_extension.load_inline(*args, **kwargs)
