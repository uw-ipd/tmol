"""Dormant JAX contract for the framework-neutral CPU disulfide core.

No native FFI handler is installed by this module. A future native package must
register all handlers explicitly before :func:`score` can dispatch them.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import MappingProxyType

DISULFIDE_CPU_FFI_TARGETS = MappingProxyType(
    {
        "forward": "tmol_disulfide_cpu_ffi_forward",
        "tracked_forward": "tmol_disulfide_cpu_ffi_forward_tracked",
        "vjp": "tmol_disulfide_cpu_ffi_vjp",
        "recompute_vjp": "tmol_disulfide_cpu_ffi_recompute_vjp",
    }
)

_REGISTERED_TARGETS = {}


def _jax():
    try:
        return import_module("jax")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'TMol JAX scoring requires the "jax" extra: pip install "tmol[jax]"',
            name="jax",
        ) from error


def register_ffi_targets(capsules):
    """Register a complete or partial role-to-capsule mapping on CPU.

    Repeating a registration with the identical capsule object is a no-op.
    Reusing a role with a different capsule is rejected before any new target
    is registered, so independently loaded adapters cannot silently replace an
    existing handler.
    """

    unknown = set(capsules) - set(DISULFIDE_CPU_FFI_TARGETS)
    if unknown:
        raise ValueError(f"unknown disulfide CPU FFI roles: {sorted(unknown)}")
    conflicts = [
        role
        for role, capsule in capsules.items()
        if role in _REGISTERED_TARGETS and _REGISTERED_TARGETS[role] is not capsule
    ]
    if conflicts:
        raise ValueError(
            "conflicting disulfide CPU FFI capsules for roles: "
            + ", ".join(sorted(conflicts))
        )

    jax = _jax()
    for role, capsule in capsules.items():
        if role in _REGISTERED_TARGETS:
            continue
        jax.ffi.register_ffi_target(
            DISULFIDE_CPU_FFI_TARGETS[role], capsule, platform="CPU"
        )
        _REGISTERED_TARGETS[role] = capsule


def _require_targets(recompute_backward):
    roles = {"forward", "recompute_vjp"}
    if not recompute_backward:
        roles = {"forward", "tracked_forward", "vjp"}
    missing = roles - set(_REGISTERED_TARGETS)
    if missing:
        names = ", ".join(DISULFIDE_CPU_FFI_TARGETS[role] for role in sorted(missing))
        raise RuntimeError(
            "TMol JAX disulfide CPU FFI handlers are not installed/registered: "
            f"{names}"
        )


def _validate_inputs(coords, pairs, params, recompute_backward):
    if type(recompute_backward) is not bool:
        raise TypeError("recompute_backward must be a concrete Python bool")
    if getattr(coords, "ndim", None) != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [n_atoms, 3]")
    if getattr(pairs, "ndim", None) != 2 or pairs.shape[1] != 8:
        raise ValueError("pairs must have shape [n_pairs, 8]")
    if getattr(params, "ndim", None) != 2 or params.shape[1] != 33:
        raise ValueError("params must have shape [n_params, 33]")

    coord_dtype = str(coords.dtype)
    if coord_dtype not in {"float32", "float64"}:
        raise TypeError("coords must have dtype float32 or float64")
    if str(params.dtype) != coord_dtype:
        raise TypeError("params must have the same dtype as coords")
    if str(pairs.dtype) != "int32":
        raise TypeError("pairs must have dtype int32")


def _result(jax, shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _call(role, result, *operands):
    jax = _jax()
    return jax.ffi.ffi_call(DISULFIDE_CPU_FFI_TARGETS[role], result)(*operands)


@lru_cache(maxsize=None)
def _score_call(coord_shape, dtype_name, recompute_backward):
    jax = _jax()
    dtype = getattr(jax.numpy, dtype_name)

    def n_pairs_shape(pairs):
        return (pairs.shape[0],)

    @jax.custom_vjp
    def call(coords, pairs, params):
        return _call(
            "forward", _result(jax, n_pairs_shape(pairs), dtype), coords, pairs, params
        )

    def forward(coords, pairs, params):
        energy_shape = _result(jax, n_pairs_shape(pairs), dtype)
        if recompute_backward:
            energies = _call("forward", energy_shape, coords, pairs, params)
            return energies, (coords, pairs, params)
        derivative_shape = _result(jax, (pairs.shape[0], 6, 3), dtype)
        energies, derivatives = _call(
            "tracked_forward",
            (energy_shape, derivative_shape),
            coords,
            pairs,
            params,
        )
        return energies, (pairs, derivatives)

    def backward(residual, cotangents):
        gradient_shape = _result(jax, coord_shape, dtype)
        if recompute_backward:
            coords, pairs, params = residual
            gradient = _call(
                "recompute_vjp",
                gradient_shape,
                coords,
                pairs,
                params,
                cotangents,
            )
        else:
            pairs, derivatives = residual
            gradient = _call("vjp", gradient_shape, pairs, derivatives, cotangents)
        return gradient, None, None

    call.defvjp(forward, backward)
    return call


def score(coords, pairs, params, *, recompute_backward=False):
    """Return one CPU disulfide energy per pair through future XLA FFI handlers.

    The static ``recompute_backward`` policy must be a concrete Python bool.
    The default differentiated forward retains pair derivatives; recompute mode
    instead retains the three inputs and delegates derivative scratch ownership
    to the future XLA FFI handler.
    """

    _validate_inputs(coords, pairs, params, recompute_backward)
    _require_targets(recompute_backward)
    return _score_call(tuple(coords.shape), str(coords.dtype), recompute_backward)(
        coords, pairs, params
    )


__all__ = ["DISULFIDE_CPU_FFI_TARGETS", "register_ffi_targets", "score"]
