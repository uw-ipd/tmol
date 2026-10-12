"""Optional CUDA FFI implementation of the electrostatics pair sum."""

from __future__ import annotations

import ctypes
import importlib
from functools import lru_cache

import jax
import jax.numpy as jnp

_GLOBAL_PARAMETER_COUNT = 11
_C1 = jnp.float32(322.0637)


@lru_cache(maxsize=1)
def _register_library() -> bool:
    try:
        plugin = importlib.import_module("tmol_jax_cuda")
        path = plugin.library_path()
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False
    try:
        library = ctypes.cdll.LoadLibrary(str(path))
        symbols = {
            name: getattr(library, f"tmol_jax_elec_{name}")
            for name in ("forward", "backward")
        }
    except (AttributeError, OSError):
        return False
    for name, symbol in symbols.items():
        jax.ffi.register_ffi_target(
            f"tmol_jax_elec_{name}",
            jax.ffi.pycapsule(symbol),
            platform="CUDA",
        )
    _register_library.library = library
    return True


def available() -> bool:
    """Return whether this installation contains its optional CUDA kernel."""
    return _register_library()


def _pack_global_parameters(state):
    def eps(distance):
        distance_s = distance * state.S
        return state.D - jnp.float32(0.5) * (state.D - state.D0) * (
            jnp.float32(2) + jnp.float32(2) * distance_s + distance_s * distance_s
        ) * jnp.exp(-distance_s)

    def deps(distance):
        return (
            jnp.float32(0.5)
            * (state.D - state.D0)
            * distance
            * distance
            * state.S
            * state.S
            * state.S
            * jnp.exp(-distance * state.S)
        )

    cutoff_offset = _C1 / (state.max_dis * eps(state.max_dis))
    min_score = _C1 / (state.min_dis * eps(state.min_dis)) - cutoff_offset
    low_end = state.min_dis + jnp.float32(0.25)
    low_eps = eps(low_end)
    low_score = _C1 / (low_end * low_eps) - cutoff_offset
    low_deriv = (
        -_C1
        * (low_eps + low_end * deps(low_end))
        / (low_end * low_end * low_eps * low_eps)
    )
    high_start = state.max_dis - jnp.float32(1)
    high_eps = eps(high_start)
    high_score = _C1 / (high_start * high_eps) - cutoff_offset
    high_deriv = (
        -_C1
        * (high_eps + high_start * deps(high_start))
        / (high_start * high_start * high_eps * high_eps)
    )
    values = jnp.stack(
        (
            state.D,
            state.D0,
            state.S,
            state.min_dis,
            state.max_dis,
            cutoff_offset,
            min_score,
            low_score,
            low_deriv,
            high_score,
            high_deriv,
        ),
        dtype=jnp.float32,
    )
    assert values.shape == (_GLOBAL_PARAMETER_COUNT,)
    return values


def score(state, coordinates):
    """Evaluate the electrostatics channel with the installed CUDA kernel."""
    charges = jnp.asarray(state.charges, dtype=jnp.float32)
    bonded_paths = jnp.asarray(state.bonded_path_length, dtype=jnp.int32)
    atom_is_real = jnp.asarray(state.atom_is_real, dtype=jnp.bool_)
    global_parameters = state.cuda_global_parameters

    @jax.custom_vjp
    def call(current_coordinates):
        return jax.ffi.ffi_call(
            "tmol_jax_elec_forward",
            jax.ShapeDtypeStruct((current_coordinates.shape[0],), jnp.float32),
        )(
            current_coordinates,
            charges,
            bonded_paths,
            atom_is_real,
            global_parameters,
        )

    def forward(current_coordinates):
        return call(current_coordinates), current_coordinates

    def backward(current_coordinates, cotangent):
        gradient = jax.ffi.ffi_call(
            "tmol_jax_elec_backward",
            jax.ShapeDtypeStruct(current_coordinates.shape, jnp.float32),
        )(
            current_coordinates,
            charges,
            bonded_paths,
            atom_is_real,
            global_parameters,
            cotangent,
        )
        return (gradient,)

    call.defvjp(forward, backward)
    return call(coordinates)


__all__ = ["available", "score"]
