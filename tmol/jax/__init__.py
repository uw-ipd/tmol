"""Lazy JAX adapters for TMol's shared runtime.

Importing :mod:`tmol.jax` does not import JAX. Backend modules are loaded only
when an exported attribute is requested.
"""

from importlib import import_module

__all__ = ["DISULFIDE_CPU_FFI_TARGETS", "register_disulfide_cpu_ffi_targets", "score"]

_EXPORTS = {
    "DISULFIDE_CPU_FFI_TARGETS": "DISULFIDE_CPU_FFI_TARGETS",
    "register_disulfide_cpu_ffi_targets": "register_ffi_targets",
    "score": "score",
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("tmol.jax.disulfide"), target)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
