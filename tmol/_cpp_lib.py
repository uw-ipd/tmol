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
        )


class TmolExtensionIncompatibleError(Exception):
    """Raised when the compiled extension is incompatible with the current environment."""

    def __init__(self, details: str = "") -> None:
        self.details = details

    def __str__(self) -> str:
        return (
            f"tmol C++/CUDA extensions are incompatible with this environment.\n"
            f"  {self.details}\n"
            "  Please reinstall tmol for your current PyTorch/CUDA/Python version.\n"
        )


def _find_extension_library() -> str | None:
    """Locate the _C shared library path without loading it."""
    import glob
    import os

    # Search by filename pattern — avoids triggering importlib to load the .so
    # (which would register TORCH_LIBRARY ops and cause a double-registration
    # crash if load_library is later called for the same file).
    package_dir = os.path.dirname(__file__)
    matches = glob.glob(os.path.join(package_dir, "_C.*.so"))
    if matches:
        return matches[0]
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

    # Check whether the .so is already in the process (loaded by Python's own
    # import machinery, e.g. via an editable-install .pth that puts the package
    # directory on sys.path and a subsequent `import tmol._C`).
    # In that case all TORCH_LIBRARY ops are already registered; calling
    # load_library again causes a "Key already registered" C++ abort.
    if "tmol._C" in __import__("sys").modules:
        _loaded = True
        logger.debug("tmol._C already in sys.modules; skipping load_library")
        return

    lib_path = _find_extension_library()
    if lib_path is None:
        raise TmolExtensionNotBuiltError()

    try:
        torch.ops.load_library(lib_path)
    except OSError as exc:
        raise TmolExtensionIncompatibleError(str(exc)) from exc

    _loaded = True
    logger.debug("Loaded tmol._C from %s", lib_path)
