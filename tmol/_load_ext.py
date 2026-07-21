"""Helpers to choose between precompiled extensions and JIT dev mode."""

from __future__ import annotations

import logging
import os

from tmol._cpp_lib import (
    TmolExtensionIncompatibleError,
    TmolExtensionNotBuiltError,
    _ensure_loaded,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def ensure_compiled_or_jit() -> bool:
    """Return True if JIT should be used; False if precompiled is loaded.

    Behavior:
    - If TMOL_USE_JIT=1 -> returns True (no precompiled load attempt)
    - Else tries to load precompiled _C and returns False if successful
    - If load fails and TMOL_JIT_FALLBACK=1 -> returns True
    - Otherwise re-raises the load error
    """
    if _env_flag("TMOL_USE_JIT"):
        logger.info("tmol: TMOL_USE_JIT=1; using JIT extensions.")
        return True

    try:
        _ensure_loaded()
        return False
    except (TmolExtensionNotBuiltError, TmolExtensionIncompatibleError) as exc:
        if _env_flag("TMOL_JIT_FALLBACK"):
            logger.warning(
                "tmol: precompiled extensions unavailable (%s); falling back to JIT.",
                exc,
            )
            return True
        raise


def load_ops(module_name: str, file: str, sources, ops_name: str):
    """JIT-compile and register torch ops if needed, then return the torch.ops namespace.

    In JIT mode, compiles the given sources and registers them as torch ops.
    In AOT mode, the ops are already registered by the precompiled library.
    Either way, returns ``torch.ops.<ops_name>``.

    Args:
        module_name: Pass ``__name__`` from the calling module.
        file: Pass ``__file__`` from the calling module.
        sources: Source filenames relative to ``file``'s directory.
        ops_name: The ``torch.ops`` namespace (e.g. ``"tmol_ljlk"``).
    """
    if ensure_compiled_or_jit():
        from tmol.utility.cpp_extension import (
            load,
            relpaths,
            modulename,
            cuda_if_available,
        )

        load(
            modulename(module_name),
            cuda_if_available(relpaths(file, sources)),
            is_python_module=False,
        )
    import torch

    return getattr(torch.ops, ops_name)


def load_module(module_name: str, file: str, sources, precompiled_path: str):
    """JIT-compile or import a precompiled pybind11 module.

    In JIT mode, compiles the given sources and returns the resulting module.
    In AOT mode, imports and returns ``precompiled_path``.

    ``cuda_if_available`` is applied automatically, so CUDA source files are
    silently dropped when CUDA is unavailable.

    Args:
        module_name: Pass ``__name__`` from the calling module.
        file: Pass ``__file__`` from the calling module.
        sources: Source filename(s) relative to ``file``'s directory.
        precompiled_path: Dotted import path to the precompiled ``_ext`` module
            (e.g. ``"tmol.tests.score.common.geom._ext"``).
    """
    if ensure_compiled_or_jit():
        from tmol.utility.cpp_extension import (
            load,
            relpaths,
            modulename,
            cuda_if_available,
        )

        return load(modulename(module_name), cuda_if_available(relpaths(file, sources)))
    else:
        import importlib

        return importlib.import_module(precompiled_path)
