"""Optional CUDA FFI implementation of the LK-ball pair sum."""

from __future__ import annotations

import ctypes
import importlib
from functools import lru_cache

import jax
import jax.numpy as jnp

_PARAMETER_COUNT = 9


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
            name: getattr(library, f"tmol_jax_lkball_{name}")
            for name in ("forward", "backward")
        }
    except (AttributeError, OSError):
        return False
    for name, symbol in symbols.items():
        jax.ffi.register_ffi_target(
            f"tmol_jax_lkball_{name}",
            jax.ffi.pycapsule(symbol),
            platform="CUDA",
        )
    _register_library.library = library
    return True


def available() -> bool:
    """Return whether this installation contains its optional CUDA kernel."""
    return _register_library()


def _pack_parameters(term, state):
    coefficient = -state.lk_dgfree / (jnp.float32(2 * 5.56832799683) * state.lk_lambda)
    packed = jnp.stack(
        (
            state.lj_radius,
            coefficient,
            1 / (state.lk_lambda * state.lk_lambda),
            state.lk_volume,
            state.is_donor,
            state.is_hydroxyl,
            state.is_polarh,
            state.is_acceptor,
            state.is_carbon_lk,
        ),
        axis=-1,
        dtype=jnp.float32,
    )
    assert packed.shape[-1] == _PARAMETER_COUNT
    defaults = term.defaults
    global_values = jnp.asarray(
        [
            defaults.lj_hbond_dis,
            defaults.lj_hbond_OH_donor_dis,
            defaults.lj_hbond_hdis,
            defaults.lkb_water_dist,
            defaults.distance_threshold,
        ],
        dtype=jnp.float32,
    )
    return packed, global_values


def _forward_ffi(
    coordinates, waters, parameters, bonded_paths, polars, occluders, globals_
):
    return jax.ffi.ffi_call(
        "tmol_jax_lkball_forward",
        jax.ShapeDtypeStruct((coordinates.shape[0], 4), jnp.float32),
    )(coordinates, waters, parameters, bonded_paths, polars, occluders, globals_)


@jax.custom_vjp
def _call(coordinates, waters, parameters, bonded_paths, polars, occluders, globals_):
    return _forward_ffi(
        coordinates, waters, parameters, bonded_paths, polars, occluders, globals_
    )


def _call_forward(
    coordinates, waters, parameters, bonded_paths, polars, occluders, globals_
):
    result = _forward_ffi(
        coordinates, waters, parameters, bonded_paths, polars, occluders, globals_
    )
    return result, (
        coordinates,
        waters,
        parameters,
        bonded_paths,
        polars,
        occluders,
        globals_,
    )


def _call_backward(residual, cotangent):
    coordinates, waters, parameters, bonded_paths, polars, occluders, globals_ = (
        residual
    )
    coordinate_gradient, water_gradient = jax.ffi.ffi_call(
        "tmol_jax_lkball_backward",
        (
            jax.ShapeDtypeStruct(coordinates.shape, jnp.float32),
            jax.ShapeDtypeStruct(waters.shape, jnp.float32),
        ),
    )(
        coordinates,
        waters,
        parameters,
        bonded_paths,
        polars,
        occluders,
        globals_,
        cotangent,
    )
    return coordinate_gradient, water_gradient, None, None, None, None, None


_call.defvjp(_call_forward, _call_backward)


def score(term, state, coordinates):
    """Evaluate four LK-ball channels with the installed CUDA kernel."""

    def place_one(x, donor, acceptor, polar, base, base0, hybrid, hydrogens, mask, n):
        return term._place_all_waters(
            x, donor, acceptor, polar, base, base0, hybrid, hydrogens, mask, n
        )

    def place(x):
        return jax.vmap(place_one)(
            x,
            state.is_donor,
            state.is_acceptor,
            state.is_polar,
            state.B_idx,
            state.B0_idx,
            state.acc_hybridization,
            state.attached_H_idx,
            state.attached_H_mask,
            state.n_attached_H,
        )

    parameters, global_values = _pack_parameters(term, state)
    return _call(
        coordinates,
        place(coordinates),
        parameters,
        jnp.asarray(state.bonded_path_length, dtype=jnp.int32),
        jnp.asarray(state.polar_indices, dtype=jnp.int32),
        jnp.asarray(state.occluder_indices, dtype=jnp.int32),
        global_values,
    )


__all__ = ["available", "score"]
