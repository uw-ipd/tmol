"""Optional CUDA FFI implementation of the LJ/LK pair sum."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tmol_jax.score._cuda import register_ffi_targets

_PARAMETER_COUNT = 13


def available() -> bool:
    """Return whether this installation contains its optional CUDA kernel."""
    return register_ffi_targets("tmol_jax_ljlk")


def fused_available() -> bool:
    """Return whether the optional library contains the fused pair kernel."""
    return available() and register_ffi_targets("tmol_jax_ljlk_elec")


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
    bonded_path_length = jnp.asarray(state.bonded_path_length, dtype=jnp.uint8)
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


def score_fused(ljlk_state, elec_state, coords):
    """Evaluate LJ/LK and electrostatics in one exact pair traversal."""
    parameters, ljlk_global = _pack_parameters(ljlk_state)
    ljlk_bonded_paths = jnp.asarray(ljlk_state.bonded_path_length, dtype=jnp.uint8)
    charges = jnp.asarray(elec_state.charges, dtype=jnp.float32)
    elec_bonded_paths = jnp.asarray(elec_state.bonded_path_length, dtype=jnp.uint8)
    atom_is_real = jnp.asarray(ljlk_state.atom_is_real, dtype=jnp.bool_)
    elec_global = elec_state.cuda_global_parameters

    @jax.custom_vjp
    def call(coordinates):
        return jax.ffi.ffi_call(
            "tmol_jax_ljlk_elec_forward",
            jax.ShapeDtypeStruct((coordinates.shape[0], 4), jnp.float32),
        )(
            coordinates,
            parameters,
            ljlk_bonded_paths,
            charges,
            elec_bonded_paths,
            atom_is_real,
            ljlk_global,
            elec_global,
        )

    def forward(coordinates):
        return call(coordinates), coordinates

    def backward(coordinates, cotangent):
        gradient = jax.ffi.ffi_call(
            "tmol_jax_ljlk_elec_backward",
            jax.ShapeDtypeStruct(coordinates.shape, jnp.float32),
        )(
            coordinates,
            parameters,
            ljlk_bonded_paths,
            charges,
            elec_bonded_paths,
            atom_is_real,
            ljlk_global,
            elec_global,
            cotangent,
        )
        return (gradient,)

    call.defvjp(forward, backward)
    return call(coords)


__all__ = ["available", "fused_available", "score", "score_fused"]
