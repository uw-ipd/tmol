import pathlib
import os
from functools import wraps
import warnings

from ..extern import include_paths as extern_include_paths
from .. import include_paths as tmol_include_paths

import torch.utils.cpp_extension
from torch.utils.cpp_extension import _is_cuda_file


# Add warning filter for use of c++ (rather than g++) for extension
# compilation. c++ is provided by g++ on our platform.
warnings.filterwarnings(
    "ignore",
    message=r"(\n|.)*"
    r"x86_64-conda_cos6-linux-gnu-c\+\+.*"
    r"is not compatible with the compiler Pytorch(\n|.)*",
)

_default_include_paths = list(tmol_include_paths() + extern_include_paths())

_required_flags = ["--std=c++14", "-DWITH_NVTX", "-w"]

if os.environ.get("DEBUG"):
    _default_flags = ["-O3", "-DDEBUG"]
    # _default_flags = ["-g", "-Og", "-DDEBUG"]
else:
    _default_flags = ["-O3"]


# TO DO! Look at what OS we're running on
# only add the -ccbin gcc-8 flag if we're on ubuntu 20.04 or higher
#
#
# which version of torch are we compiling against?
def get_torch_version():
    return torch.__version__.split(".")[0:2]


torch_major, torch_minor = get_torch_version()

_required_cuda_flags = [
    "-std=c++14",
    "--expt-extended-lambda",
    "-DWITH_NVTX",
    "-w",
    # "-G",
    "-DTORCH_VERSION_MAJOR=%s" % torch_major,
    "-DTORCH_VERSION_MINOR=%s" % torch_minor,
]


# Add an additional --gpu-architecture flag to the nvcc command:
# The flag we pass to nvcc should be controllable from the TORCH_CUDA_ARCH_LIST
# environment variable; if this is set, then use that. If it is not set, then
# we will query the device. This lets us use an older version of torch with a
# more recent GPU (e.g. an A100 with cuda10.1)
if torch.cuda.is_available():
    import os

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    # If not given, determine what's needed for the GPU that can be found
    if not arch_list:
        _major, _minor = torch.cuda.get_device_capability(0)
    else:
        # Take the first architecture listed.
        # Deal with lists that are ' ' or ';' separated
        _major, _minor = arch_list.replace(" ", ";").split(";")[0].split(".")
    _required_cuda_flags.append(f"--gpu-architecture=sm_{_major}{_minor}")


_default_cuda_flags = []


def _augment_kwargs(name, sources, **kwargs):
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

    if kwargs.get("with_cuda", None) is None:
        with_cuda = any(map(_is_cuda_file, sources))
        kwargs["with_cuda"] = with_cuda

    if kwargs["with_cuda"]:
        kwargs["extra_cflags"] += ["-DWITH_CUDA"]
        kwargs["extra_cuda_cflags"] += ["-DWITH_CUDA"]

    return kwargs


def cuda_if_available(sources):
    """Filter cuda sources if cuda is not available."""
    if torch.cuda.is_available():
        return sources
    else:
        return [s for s in sources if not _is_cuda_file(s)]


@wraps(torch.utils.cpp_extension.load)
def load(name, sources, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""
    kwargs = _augment_kwargs(name, sources, **kwargs)
    return torch.utils.cpp_extension.load(name, sources, **kwargs)


@wraps(torch.utils.cpp_extension.load_inline)
def load_inline(name, sources, **kwargs):
    """Jit-compile torch cpp_extension with tmol paths."""
    kwargs = _augment_kwargs(name, sources, **kwargs)
    return torch.utils.cpp_extension.load_inline(name, sources, **kwargs)


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
