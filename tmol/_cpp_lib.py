"""
Runtime loader for tmol's pre-compiled C++/CUDA extensions.

Loads a CPU or optional CUDA backend and registers its stable native bindings
in the existing ``torch.ops.tmol_*`` namespaces.

Usage::

    from tmol._cpp_lib import _ensure_loaded

    _ensure_loaded()
    # Now torch.ops.tmol_ljlk.ljlk_pose_scores etc. are available
"""

from __future__ import annotations

import importlib
import logging
import re
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from threading import RLock

import torch

logger = logging.getLogger(__name__)

_loaded = False
_libraries = []
_load_lock = RLock()
_native = None


class TmolExtensionNotBuiltError(Exception):
    """Raised when tmol's C++/CUDA extensions are not available."""

    def __str__(self) -> str:
        return (
            "tmol C++/CUDA extensions are not built.\n"
            "  Install tmol from a pre-built wheel:\n"
            "    pip install tmol\n"
            "  Or build from source (CPU-only builds need no CUDA toolkit):\n"
            "    pip install -e .\n"
            "  On older Linux clusters, prefer:\n"
            "    TMOL_DISABLE_WHEEL_FETCH=1 pip install -e .\n"
            "  or set TMOL_JIT_FALLBACK=1 (CPU-only needs a C++ compiler and "
            "ninja; CUDA builds also need nvcc).\n"
        )


class TmolExtensionIncompatibleError(Exception):
    """Raised when the compiled extension is incompatible with the current environment."""

    def __init__(self, details: str = "") -> None:
        self.details = details

    def __str__(self) -> str:
        return (
            "tmol C++/CUDA extensions could not be loaded in this environment.\n"
            f"  {self.details}\n"
        )


def extension_load_error_details(exc: OSError) -> str:
    """Turn a dynamic-loader failure into actionable guidance."""
    msg = str(exc).strip()
    lower = msg.lower()

    if "glibcxx_" in lower or "libstdc++.so" in lower:
        return (
            f"{msg}\n"
            "  System libstdc++ is too old for this pre-built wheel "
            "(C++ runtime ABI mismatch).\n"
            "  Fixes:\n"
            "    - Load a newer GCC module and export LD_LIBRARY_PATH to its libstdc++\n"
            "    - conda install -c conda-forge libstdcxx-ng (then set LD_LIBRARY_PATH)\n"
            "    - Use a container with a recent base image\n"
            "    - Build tmol on this machine:\n"
            "        TMOL_DISABLE_WHEEL_FETCH=1 pip install -e .\n"
            "    - Or set TMOL_JIT_FALLBACK=1 (CPU-only needs a C++ compiler and "
            "ninja; CUDA also needs nvcc; slower first import)"
        )

    if "glibc_" in lower and "libc.so" in lower:
        return (
            f"{msg}\n"
            "  System glibc is too old for this pre-built wheel.\n"
            "  Fixes: use a newer OS/container, or build from source on this machine:\n"
            "    TMOL_DISABLE_WHEEL_FETCH=1 pip install -e ."
        )

    if re.search(r"\bcuda\b|cudnn|cublas", lower):
        return (
            f"{msg}\n"
            "  CUDA runtime libraries may be missing or mismatched.\n"
            "  Fixes: install a PyTorch build with matching CUDA, set CUDA_HOME/LD_LIBRARY_PATH,\n"
            "  or build from source on this machine."
        )

    return (
        f"{msg}\n"
        "  If you installed a pre-built wheel, verify Python/platform tags, PyTorch >=2.11, and the CUDA major.\n"
        "  Otherwise build on this machine: TMOL_DISABLE_WHEEL_FETCH=1 pip install -e ."
    )


def _call_native(function, arguments, *args, **kwargs):
    if len(args) == len(arguments) and not kwargs:
        return function(*args)
    values = list(args)
    for argument in arguments[len(args) :]:
        values.append(
            kwargs.pop(argument.name)
            if argument.name in kwargs
            else argument.default_value
        )
    if kwargs:
        raise TypeError(f"Unknown native arguments: {tuple(kwargs)}")
    return function(*values)


def _native_module_name():
    cuda = torch.version.cuda
    if cuda:
        package = f"tmol_ops_cu{cuda.split('.')[0]}"
        if importlib.util.find_spec(package) is not None:
            try:
                if version(package) != version("tmol"):
                    raise TmolExtensionIncompatibleError(
                        f"{package} and tmol must have the same version; reinstall tmol[cu{cuda.split('.')[0]}]."
                    )
            except PackageNotFoundError:  # A developer source build.
                pass
            return f"{package}._C"
        other = "tmol_ops_cu12" if package.endswith("13") else "tmol_ops_cu13"
        if importlib.util.find_spec(other) is not None:
            raise TmolExtensionIncompatibleError(
                f"Installed {other} does not match PyTorch's CUDA {cuda}. "
                f"Install tmol[cu{cuda.split('.')[0]}] or a matching CUDA build of PyTorch."
            )
    return "tmol._C"


def _find_extension_library() -> str | None:
    """Locate the backend selected for the installed PyTorch CUDA major."""
    try:
        spec = importlib.util.find_spec(_native_module_name())
        if spec is not None:
            return spec.origin
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def _ensure_loaded() -> None:
    """Load one native backend and register its operators."""
    global _loaded, _native
    with _load_lock:
        if _loaded:
            return
        module_name = _native_module_name()
        if _find_extension_library() is None:
            raise TmolExtensionNotBuiltError()
        libraries = []
        try:
            native = importlib.import_module(module_name)
            for name in dir(native):
                module = getattr(native, name)
                if not hasattr(module, "_schemas"):
                    continue
                library = torch.library.Library(name, "FRAGMENT")
                libraries.append(library)
                for schema in module._schemas:
                    operator = schema.split("(", 1)[0]
                    library.define(schema)
                    if hasattr(module, operator):
                        arguments = getattr(
                            getattr(torch.ops, name), operator
                        ).default._schema.arguments
                        implementation = partial(
                            _call_native, getattr(module, operator), arguments
                        )
                        library.impl(
                            operator, implementation, "CompositeImplicitAutograd"
                        )
        except (ImportError, OSError) as exc:
            raise TmolExtensionIncompatibleError(
                extension_load_error_details(exc)
            ) from exc
        _libraries.extend(libraries)
        _native = native
        _loaded = True
        logger.debug("Loaded native backend %s", module_name)


def has_cuda() -> bool:
    """Whether the loaded TMol backend includes CUDA kernels (independent of GPU availability)."""
    _ensure_loaded()
    return _native.has_cuda
