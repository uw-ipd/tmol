"""
Runtime loader for tmol's pre-compiled C++/CUDA extensions.

Adapted from xformers/_cpp_lib.py. Loads the ``tmol._C`` shared library
which registers all TORCH_LIBRARY ops into ``torch.ops.tmol_*`` namespaces.

Usage::

    from tmol._cpp_lib import _ensure_loaded

    _ensure_loaded()
    # Now torch.ops.tmol_ljlk.ljlk_pose_scores etc. are available
"""

from __future__ import annotations

import importlib
import logging
import re

import torch

logger = logging.getLogger(__name__)

_loaded = False


class TmolExtensionNotBuiltError(Exception):
    """Raised when tmol's C++/CUDA extensions are not available."""

    def __str__(self) -> str:
        return (
            "tmol C++/CUDA extensions are not built.\n"
            "  Install tmol from a pre-built wheel:\n"
            "    pip install tmol\n"
            "  Or build from source (requires CUDA toolkit):\n"
            "    pip install -e .\n"
            "  On older Linux clusters, prefer:\n"
            "    TMOL_DISABLE_WHEEL_FETCH=1 pip install -e .\n"
            "  or set TMOL_JIT_FALLBACK=1 if nvcc is available.\n"
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
            "    - Or set TMOL_JIT_FALLBACK=1 (requires nvcc; slower first import)"
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
        "  If you installed a pre-built wheel, verify Python, PyTorch, and CUDA tags match.\n"
        "  Otherwise build on this machine: TMOL_DISABLE_WHEEL_FETCH=1 pip install -e ."
    )


def _find_extension_library() -> str | None:
    """Locate the _C shared library in tmol's package directory."""
    try:
        spec = importlib.util.find_spec("tmol._C")
        if spec is not None and spec.origin is not None:
            return spec.origin
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def _ensure_loaded() -> None:
    """
    Load the tmol._C shared library, registering all TORCH_LIBRARY ops.

    This function is idempotent — calling it multiple times is safe.
    Raises TmolExtensionNotBuiltError if the library is not found.
    """
    global _loaded
    if _loaded:
        return

    lib_path = _find_extension_library()
    if lib_path is None:
        raise TmolExtensionNotBuiltError()

    try:
        torch.ops.load_library(lib_path)
    except OSError as exc:
        raise TmolExtensionIncompatibleError(extension_load_error_details(exc)) from exc

    _loaded = True
    logger.debug("Loaded tmol._C from %s", lib_path)
