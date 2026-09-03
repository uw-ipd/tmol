import os
import pathlib
import warnings
from functools import wraps

from .._cuda_env import (  # noqa: F401
    get_cccl_include as _get_cccl_include,
    setup as _cuda_env_setup,
)


# Avoid importing from parent package (..) to prevent circular import issues
# when this module is imported during package initialization.
# Compute tmol include paths directly.
def _tmol_include_paths():
    """C++/CUDA include paths for tmol components."""
    return [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))]


# ---------------------------------------------------------------------------
# Auto-configure CUDA for pip-installed toolkit BEFORE torch reads CUDA_HOME.
# This must happen before `import torch.utils.cpp_extension` because PyTorch
# evaluates CUDA_HOME at module-load time.
# ---------------------------------------------------------------------------
from ..extern import include_paths as extern_include_paths  # noqa: E402

_cuda_env_setup()

import torch.utils.cpp_extension  # noqa: E402
from torch.utils.cpp_extension import _is_cuda_file  # noqa: E402

# Add warning filter for use of c++ (rather than g++) for extension
# compilation. c++ is provided by g++ on our platform.
warnings.filterwarnings(
    "ignore",
    message=r"(\n|.)*"
    r"x86_64-conda_cos6-linux-gnu-c\+\+.*"
    r"is not compatible with the compiler Pytorch(\n|.)*",
)

_default_include_paths = list(_tmol_include_paths() + extern_include_paths())

# Add CCCL include path (nv/target, cub/, thrust/) from pip-installed nvidia-cuda-cccl
_cccl_include = _get_cccl_include()
if _cccl_include:
    _default_include_paths.append(_cccl_include)

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


def _required_cxx_standard(torch_major, torch_minor):
    """Return the language standard required by this PyTorch release."""
    version = int(torch_major), int(torch_minor)
    return 20 if version >= (2, 13) else 17


_cxx_standard = _required_cxx_standard(torch_major, torch_minor)
_required_flags = [f"--std=c++{_cxx_standard}", "-DWITH_NVTX", "-w"]

_required_cuda_flags = [
    f"-std=c++{_cxx_standard}",
    "--expt-extended-lambda",
    "-DWITH_NVTX",
    "-w",
    # "-G",
    f"-DTORCH_VERSION_MAJOR={torch_major}",
    f"-DTORCH_VERSION_MINOR={torch_minor}",
]


def _select_cuda_architecture(arch_list, device_capability):
    """Choose the active device from a possibly multi-architecture setting."""
    current = ".".join(str(part) for part in device_capability)
    if not arch_list:
        return current
    requested = arch_list.replace(" ", ";").split(";")
    requested = [arch.removesuffix("+PTX") for arch in requested if arch]
    if not requested:
        return current
    # A multi-architecture value commonly comes from a wheel/container build
    # environment and can lag newly installed hardware. A local JIT build is
    # for the active device, so do not silently compile its first (often
    # oldest) entry when the device is absent. Preserve a single explicit
    # architecture as the user's deliberate cross-compilation request.
    return current if len(requested) > 1 else requested[0]


# Add an additional --gpu-architecture flag to the nvcc command:
# The flag we pass to nvcc should be controllable from the TORCH_CUDA_ARCH_LIST
# environment variable; if this is set, then use that. If it is not set, then
# we will query the device. This lets us use an older version of torch with a
# more recent GPU (e.g. an A100 with cuda10.1)
if torch.cuda.is_available():
    import os

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    # If not given, determine what's needed for the GPU that can be found
    _major, _minor = _select_cuda_architecture(
        arch_list, torch.cuda.get_device_capability()
    ).split(".")
    _required_cuda_flags.append(f"--gpu-architecture=sm_{_major}{_minor}")

    # we need to add the search path for nvtx3
    # which should be installed relative to nvcc

    import subprocess
    import sys

    # Find nvtx include path — works for both pip and conda layouts
    try:
        import nvidia.nvtx as _nvtx

        _nvtx_include = os.path.join(_nvtx.__path__[0], "include")
        if os.path.isdir(_nvtx_include):
            _default_include_paths.append(_nvtx_include)
    except ImportError:
        # Fallback: guess relative to nvcc location (conda layout)
        path = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
        nvcc_dir = os.path.dirname(path.stdout.strip())
        ver_info = sys.version_info
        _default_include_paths.append(
            f"{nvcc_dir}/../lib/python{ver_info.major}.{ver_info.minor}"
            f"/site-packages/nvidia/nvtx/include"
        )

# Match the Release AOT build. Without an explicit optimization level, local
# JIT extensions can run materially slower than the packaged CUDA kernels.
_default_cuda_flags = ["-O3"]


# Add additional flags.
# The "verbose" flag can be controlled by adding the environment variable
# "TMOL_TORCH_EXTENSIONS_VERBOSE" which will ask ninja to print all compiler
# commands to the terminal
def _augment_kwargs(name, sources, **kwargs):
    kwargs["extra_cflags"] = (
        _default_flags + list(kwargs.get("extra_cflags", [])) + _required_flags
    )
    kwargs["extra_cuda_cflags"] = (
        _default_cuda_flags
        + list(kwargs.get("extra_cuda_cflags", []))
        + _required_cuda_flags
    )
    kwargs["extra_include_paths"] = (
        list(kwargs.get("extra_include_flags", [])) + _default_include_paths
    )

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
