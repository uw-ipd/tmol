import os
import pathlib
import warnings
from functools import wraps

from .. import include_paths as tmol_include_paths
from .._cuda_env import get_cccl_include as _get_cccl_include

# ---------------------------------------------------------------------------
# Auto-configure CUDA for pip-installed toolkit BEFORE torch reads CUDA_HOME.
# This must happen before `import torch.utils.cpp_extension` because PyTorch
# evaluates CUDA_HOME at module-load time.
# ---------------------------------------------------------------------------
from .._cuda_env import setup as _cuda_env_setup
from ..extern import include_paths as extern_include_paths

_cuda_env_setup()

import torch  # noqa: E402
import torch.utils.cpp_extension  # noqa: E402
from torch.utils.cpp_extension import _is_cuda_file  # noqa: E402

# Add warning filter for use of c++ (rather than g++) for extension
# compilation. c++ is provided by g++ on our platform.
warnings.filterwarnings(
    "ignore",
    message=r"(\n|.)*" r"x86_64-conda_cos6-linux-gnu-c\+\+.*" r"is not compatible with the compiler Pytorch(\n|.)*",
)

_default_include_paths = list(tmol_include_paths() + extern_include_paths())

# Add CCCL include path (nv/target, cub/, thrust/) from pip-installed nvidia-cuda-cccl
_cccl_include = _get_cccl_include()
if _cccl_include:
    _default_include_paths.append(_cccl_include)

_required_flags = ["--std=c++17", "-DWITH_NVTX", "-w"]

if os.environ.get("DEBUG"):
    _default_flags = ["-O3", "-DDEBUG"]
    # _default_flags = ["-g", "-Og", "-DDEBUG"]
else:
    _default_flags = ["-O3"]


def get_torch_version():
    return torch.__version__.split(".")[0:2]


torch_major, torch_minor = get_torch_version()

_required_cuda_flags = [
    "-std=c++17",
    "--expt-extended-lambda",
    "-DWITH_NVTX",
    "-w",
    f"-DTORCH_VERSION_MAJOR={torch_major}",
    f"-DTORCH_VERSION_MINOR={torch_minor}",
]

# GPU architecture flags (--gpu-architecture / -gencode) are NOT added here.
# PyTorch's cpp_extension.load() reads TORCH_CUDA_ARCH_LIST natively and
# produces proper multi-arch -gencode flags.  If TORCH_CUDA_ARCH_LIST is
# unset, PyTorch auto-detects the local GPU's compute capability.

# Find NVTX include path for -DWITH_NVTX support.
# Try the pip-installed nvidia.nvtx package first, then fall back to CUDA_HOME.
try:
    import nvidia.nvtx as _nvtx

    _nvtx_include = os.path.join(_nvtx.__path__[0], "include")
    if os.path.isdir(_nvtx_include):
        _default_include_paths.append(_nvtx_include)
except ImportError:
    _cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if _cuda_home:
        _nvtx_candidate = os.path.join(_cuda_home, "include")
        if os.path.isdir(_nvtx_candidate):
            _default_include_paths.append(_nvtx_candidate)

_default_cuda_flags = []


# Add additional flags.
# The "verbose" flag can be controlled by adding the environment variable
# "TMOL_TORCH_EXTENSIONS_VERBOSE" which will ask ninja to print all compiler
# commands to the terminal
def _augment_kwargs(name, sources, **kwargs):
    kwargs["extra_cflags"] = list(kwargs.get("extra_cflags", _default_flags)) + _required_flags
    kwargs["extra_cuda_cflags"] = list(kwargs.get("extra_cuda_cflags", _default_cuda_flags)) + _required_cuda_flags
    kwargs["extra_include_paths"] = list(kwargs.get("extra_include_flags", [])) + _default_include_paths

    if kwargs.get("with_cuda") is None:
        with_cuda = any(map(_is_cuda_file, sources))
        kwargs["with_cuda"] = with_cuda

    if kwargs["with_cuda"]:
        kwargs["extra_cflags"] += ["-DWITH_CUDA"]
        kwargs["extra_cuda_cflags"] += ["-DWITH_CUDA"]

    if os.environ.get("TMOL_TORCH_EXTENSIONS_VERBOSE"):
        kwargs["verbose"] = True

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

    if isinstance(paths, str | bytes):
        paths = [paths]

    return [str(pathlib.Path(src_path).parent / s) for s in paths]


def modulename(src_name):
    """Adapt module name to valid cpp extension name.

    Used to adapt a module __name__ to a valid extension name.

    Example:
        name = modulename(__name__)
    """

    return src_name.replace(".", "_")

