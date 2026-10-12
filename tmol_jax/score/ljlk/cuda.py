"""Optional CUDA FFI implementation of the LJ/LK pair sum."""

from __future__ import annotations

import ctypes
import importlib
from functools import lru_cache

import jax
import jax.numpy as jnp

_PARAMETER_COUNT = 13


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
            name: getattr(library, f"tmol_jax_ljlk_{name}")
            for name in ("forward", "backward")
        }
    except (AttributeError, OSError):
        return False
    for name, symbol in symbols.items():
        jax.ffi.register_ffi_target(
            f"tmol_jax_ljlk_{name}",
            jax.ffi.pycapsule(symbol),
            platform="CUDA",
        )
    # The capsule does not own the dynamic library.
    _register_library.library = library
    return True


def available() -> bool:
    """Return whether this installation contains its optional CUDA kernel."""
    return _register_library()


def _pack_parameters(state):
    parameters = state.parameters
    coefficient = -parameters.lk_dgfree / (
        jnp.float32(2 * 5.56832799683) * parameters.lk_lambda
    )
    inverse_lambda_squared = 1 / (parameters.lk_lambda * parameters.lk_lambda)
    packed = jnp.stack(
        (
            parameters.lj_radius,
            jnp.sqrt(parameters.lj_wdepth),
            parameters.lk_dgfree,
            parameters.lk_lambda,
            parameters.lk_volume,
            parameters.is_donor,
            parameters.is_hydroxyl,
            parameters.is_polarh,
            parameters.is_acceptor,
            parameters.is_carbon_lk,
            jnp.zeros_like(parameters.is_polarh),
            coefficient,
            inverse_lambda_squared,
        ),
        axis=-1,
        dtype=jnp.float32,
    )
    assert packed.shape[-1] == _PARAMETER_COUNT
    global_parameters = state.global_parameters
    global_values = jnp.stack(
        (
            global_parameters.max_dis,
            global_parameters.lj_dlin_sigma_factor,
            global_parameters.lj_hbond_dis,
            global_parameters.lj_hbond_OH_donor_dis,
            global_parameters.lj_hbond_hdis,
        ),
        dtype=jnp.float32,
    )
    return packed, global_values


def score(state, coords):
    """Evaluate three LJ/LK channels with the installed CUDA kernel."""
    parameters, global_parameters = _pack_parameters(state)
    bonded_path_length = jnp.asarray(state.bonded_path_length, dtype=jnp.int32)
    atom_is_real = jnp.asarray(state.atom_is_real, dtype=jnp.bool_)

    @jax.custom_vjp
    def call(coordinates):
        return jax.ffi.ffi_call(
            "tmol_jax_ljlk_forward",
            jax.ShapeDtypeStruct((coordinates.shape[0], 3), jnp.float32),
        )(
            coordinates,
            parameters,
            bonded_path_length,
            atom_is_real,
            global_parameters,
        )

    def forward(coordinates):
        return call(coordinates), coordinates

    def backward(coordinates, cotangent):
        gradient = jax.ffi.ffi_call(
            "tmol_jax_ljlk_backward",
            jax.ShapeDtypeStruct(coordinates.shape, jnp.float32),
        )(
            coordinates,
            parameters,
            bonded_path_length,
            atom_is_real,
            global_parameters,
            cotangent,
        )
        return (gradient,)

    call.defvjp(forward, backward)
    return call(coords)


__all__ = ["available", "score"]
