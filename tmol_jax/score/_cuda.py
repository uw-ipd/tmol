"""Versioned loading for the optional no-Torch CUDA adapter."""

from __future__ import annotations

import ctypes
import importlib
from functools import lru_cache

import jax

ABI_VERSION = 1


@lru_cache(maxsize=1)
def load_library():
    """Load a compatible adapter, or return ``None`` when it is not installed."""
    try:
        plugin = importlib.import_module("tmol_jax_cuda")
        path = plugin.library_path()
    except (AttributeError, ImportError, OSError, RuntimeError):
        return None
    try:
        library = ctypes.cdll.LoadLibrary(str(path))
    except OSError:
        return None
    try:
        abi_version = library.tmol_jax_cuda_abi_version
    except AttributeError as error:
        raise RuntimeError(
            "installed tmol-jax-cuda has no ABI marker; reinstall a matching wheel"
        ) from error
    abi_version.argtypes = []
    abi_version.restype = ctypes.c_int
    actual = abi_version()
    if actual != ABI_VERSION:
        raise RuntimeError(
            f"tmol-jax-cuda ABI {actual} does not match frontend ABI {ABI_VERSION}; "
            "reinstall matching tmol-jax and tmol-jax-cuda wheels"
        )
    return library


@lru_cache(maxsize=None)
def register_ffi_targets(prefix: str, suffixes=("forward", "backward")) -> bool:
    """Register one adapter operation after checking the shared ABI marker."""
    library = load_library()
    if library is None:
        return False
    try:
        symbols = {
            suffix: getattr(library, f"{prefix}_{suffix}") for suffix in suffixes
        }
    except AttributeError:
        return False
    for suffix, symbol in symbols.items():
        name = f"{prefix}_{suffix}"
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(symbol), platform="CUDA")
    return True
