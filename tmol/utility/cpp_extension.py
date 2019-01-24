import pathlib
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

_default_include_paths = list(tmol_include_paths() + extern_include_paths())

_required_flags = ["--std=c++14"]
# _default_flags = ["-O3"]
_default_flags = ["-g", "-Og"]

_required_cuda_flags = ["--expt-extended-lambda", "--expt-relaxed-constexpr"]
_default_cuda_flags = []


def _augment_kwargs(kwargs):
    kwargs["extra_cflags"] = (
        list(kwargs.get("extra_cflags", _default_flags)) + _required_flags
    )
    kwargs["extra_cuda_cflags"] = (
        list(kwargs.get("extra_cuda_cflags", _default_cuda_flags))
        + _required_cuda_flags
    )
    kwargs["extra_include_paths"] = (
        list(kwargs.get("extra_include_flags", [])) + _default_include_paths
    )

    return kwargs


@wraps(torch.utils.cpp_extension.load)
def load(*args, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""

    kwargs = _augment_kwargs(kwargs)
    return torch.utils.cpp_extension.load(*args, **kwargs)


@wraps(torch.utils.cpp_extension.load_inline)
def load_inline(*args, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""

    kwargs = _augment_kwargs(kwargs)
    return torch.utils.cpp_extension.load_inline(*args, **kwargs)


def relpaths(src_path, paths):
    """Paths relative to the parent of given src file.

    Used to indiciate paths relative to a module's __file__.

    Example:
        srcs = relpaths(__file__, ["sibling.cpp", "sibling.cu"])
    """

    if isinstance(paths, (str, bytes)):
        paths = [paths]

    return [str(pathlib.Path(src_path).parent / s) for s in paths]


def modulename(src_name):
    """Adapt module name to valid cpp extension name.

    Used to adapt a module __name__ to a valid extension name.

    Example:
        name = modulename(__name__)
    """

    return src_name.replace(".", "_")
