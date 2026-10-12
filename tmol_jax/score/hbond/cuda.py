"""Optional CUDA FFI implementation of the hydrogen-bond pair sum."""

from __future__ import annotations

import ctypes
import importlib
from functools import lru_cache

import jax
import jax.numpy as jnp


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
            name: getattr(library, f"tmol_jax_hbond_{name}")
            for name in ("forward", "backward")
        }
    except (AttributeError, OSError):
        return False
    for name, symbol in symbols.items():
        jax.ffi.register_ffi_target(
            f"tmol_jax_hbond_{name}",
            jax.ffi.pycapsule(symbol),
            platform="CUDA",
        )
    _register_library.library = library
    return True


def available() -> bool:
    """Return whether this installation contains its optional CUDA kernel."""
    return _register_library()


def _pack_parameters(state):
    atoms = state.atom_params
    donors = jnp.stack(
        (atoms.donor_H_atom, atoms.donor_heavy_atom, atoms.donor_type),
        axis=-1,
        dtype=jnp.int32,
    )
    acceptors = jnp.stack(
        (
            atoms.acceptor_atom,
            atoms.acceptor_base,
            atoms.acceptor_base0,
            atoms.acceptor_type,
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    pairs = jnp.stack(
        (
            state.pair_params.weight,
            state.pair_params.acceptor_hybridization.astype(jnp.float32),
        ),
        axis=-1,
        dtype=jnp.float32,
    )
    globals_ = state.global_params
    global_values = jnp.stack(
        (
            globals_.hb_sp2_range_span,
            globals_.hb_sp2_BAH180_rise,
            globals_.hb_sp2_outer_width,
            globals_.hb_sp3_softmax_fade,
            globals_.threshold_distance,
            globals_.max_ah_dist,
        ),
        dtype=jnp.float32,
    )
    return donors, acceptors, pairs, global_values


def _forward_ffi(
    coordinates, donors, acceptors, separations, pairs, polynomial_words, globals_
):
    return jax.ffi.ffi_call(
        "tmol_jax_hbond_forward",
        jax.ShapeDtypeStruct((coordinates.shape[0],), jnp.float32),
    )(
        coordinates,
        donors,
        acceptors,
        separations,
        pairs,
        polynomial_words,
        globals_,
    )


@jax.custom_vjp
def _call(
    coordinates, donors, acceptors, separations, pairs, polynomial_words, globals_
):
    return _forward_ffi(
        coordinates,
        donors,
        acceptors,
        separations,
        pairs,
        polynomial_words,
        globals_,
    )


def _call_forward(
    coordinates, donors, acceptors, separations, pairs, polynomial_words, globals_
):
    result = _forward_ffi(
        coordinates,
        donors,
        acceptors,
        separations,
        pairs,
        polynomial_words,
        globals_,
    )
    return result, (
        coordinates,
        donors,
        acceptors,
        separations,
        pairs,
        polynomial_words,
        globals_,
    )


def _call_backward(residual, cotangent):
    coordinates, donors, acceptors, separations, pairs, polynomial_words, globals_ = (
        residual
    )
    gradient = jax.ffi.ffi_call(
        "tmol_jax_hbond_backward",
        jax.ShapeDtypeStruct(coordinates.shape, jnp.float32),
    )(
        coordinates,
        donors,
        acceptors,
        separations,
        pairs,
        polynomial_words,
        globals_,
        cotangent,
    )
    return gradient, None, None, None, None, None, None


_call.defvjp(_call_forward, _call_backward)


def score(state, coordinates):
    """Evaluate the hydrogen-bond channel with the installed CUDA kernel."""
    donors, acceptors, pairs, global_values = _pack_parameters(state)
    return _call(
        coordinates,
        donors,
        acceptors,
        jnp.asarray(state.donor_acceptor_bond_sep, dtype=jnp.int32),
        pairs,
        jnp.asarray(state.polynomials.native_poly_words, dtype=jnp.uint32),
        global_values,
    )


__all__ = ["available", "score"]
