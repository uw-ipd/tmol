"""Explicit JAX contract for the framework-neutral disulfide core.

The base wheel owns the CPU adapter and a matching CUDA companion may own the
CUDA adapter. Callers register handlers explicitly before :func:`score` can
dispatch them.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType

# Target names are intentionally identical across XLA platforms.
DISULFIDE_CPU_FFI_TARGETS = MappingProxyType(
    {
        "forward": "tmol_disulfide_cpu_ffi_forward",
        "tracked_forward": "tmol_disulfide_cpu_ffi_forward_tracked",
        "vjp": "tmol_disulfide_cpu_ffi_vjp",
        "recompute_vjp": "tmol_disulfide_cpu_ffi_recompute_vjp",
    }
)

_REGISTERED_TARGETS = {}
_NATIVE_LIBRARIES = {}


def _jax():
    try:
        return import_module("jax")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'TMol JAX scoring requires the "jax" extra: pip install "tmol[jax]"',
            name="jax",
        ) from error


@lru_cache(maxsize=1)
def _installed_cpu_capsules():
    """Load the base-wheel-owned CPU adapter without importing Torch."""

    if "cpu" not in _NATIVE_LIBRARIES:
        package_dir = Path(__file__).resolve().parent.parent
        candidates = sorted(package_dir.glob("_shared_disulfide_ffi.*"))
        if len(candidates) != 1:
            raise RuntimeError(
                "TMol JAX disulfide CPU adapter is missing or ambiguous; "
                "reinstall the matching tmol wheel"
            )
        _NATIVE_LIBRARIES["cpu"] = ctypes.CDLL(str(candidates[0]))
    jax = _jax()
    return {
        role: jax.ffi.pycapsule(getattr(_NATIVE_LIBRARIES["cpu"], target))
        for role, target in DISULFIDE_CPU_FFI_TARGETS.items()
    }


def _cuda_adapter_candidates():
    candidates = []
    for package_name in ("tmol_ops_cu13", "tmol_ops_cu12"):
        spec = find_spec(package_name)
        if spec is None:
            continue
        for location in spec.submodule_search_locations or ():
            candidates.extend(
                sorted(Path(location).glob("_shared_disulfide_cuda_ffi.*"))
            )
    return tuple(dict.fromkeys(candidates))


@lru_cache(maxsize=1)
def _installed_cuda_capsules():
    """Return a matching CUDA-companion adapter, or ``None`` when absent."""

    candidates = _cuda_adapter_candidates()
    if not candidates:
        return None
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            "TMol JAX disulfide CUDA adapter is ambiguous; uninstall the "
            f"nonmatching CUDA companion and reinstall one version: {paths}"
        )
    try:
        library = ctypes.CDLL(str(candidates[0]))
    except OSError as error:
        raise RuntimeError(
            "TMol JAX disulfide CUDA adapter could not load; reinstall matching "
            "tmol and tmol-ops-cu12/cu13 wheels"
        ) from error
    _NATIVE_LIBRARIES["CUDA"] = library
    jax = _jax()
    try:
        return {
            role: jax.ffi.pycapsule(getattr(library, target))
            for role, target in DISULFIDE_CPU_FFI_TARGETS.items()
        }
    except AttributeError as error:
        raise RuntimeError(
            "TMol JAX disulfide CUDA adapter is incomplete; reinstall matching "
            "tmol and CUDA companion wheels"
        ) from error


def register_ffi_targets(capsules=None, *, platform="cpu"):
    """Register role-to-capsule mappings and discover an optional companion.

    Repeating a registration with the identical capsule object is a no-op.
    Reusing a role with a different capsule is rejected before any new target
    is registered. Calling with no arguments registers CPU and, when exactly
    one matching CUDA companion is installed, CUDA without importing Torch.
    """

    discover_cuda = capsules is None and platform == "cpu"
    if capsules is None:
        capsules = _installed_cpu_capsules()
    if platform not in {"cpu", "CUDA"}:
        raise ValueError(f"unsupported disulfide FFI platform: {platform!r}")
    unknown = set(capsules) - set(DISULFIDE_CPU_FFI_TARGETS)
    if unknown:
        raise ValueError(f"unknown disulfide FFI roles: {sorted(unknown)}")
    conflicts = [
        role
        for role, capsule in capsules.items()
        if (platform, role) in _REGISTERED_TARGETS
        and _REGISTERED_TARGETS[platform, role] is not capsule
    ]
    if conflicts:
        raise ValueError(
            f"conflicting disulfide {platform} FFI capsules for roles: "
            + ", ".join(sorted(conflicts))
        )

    jax = _jax()
    for role, capsule in capsules.items():
        key = (platform, role)
        if key in _REGISTERED_TARGETS:
            continue
        jax.ffi.register_ffi_target(
            DISULFIDE_CPU_FFI_TARGETS[role], capsule, platform=platform
        )
        _REGISTERED_TARGETS[key] = capsule
    if discover_cuda:
        cuda_capsules = _installed_cuda_capsules()
        if cuda_capsules is not None:
            register_ffi_targets(cuda_capsules, platform="CUDA")


def _require_targets(recompute_backward):
    roles = {"forward", "recompute_vjp"}
    if not recompute_backward:
        roles = {"forward", "tracked_forward", "vjp"}
    missing = {
        role
        for role in roles
        if not any(
            (platform, role) in _REGISTERED_TARGETS for platform in ("cpu", "CUDA")
        )
    }
    if missing:
        names = ", ".join(DISULFIDE_CPU_FFI_TARGETS[role] for role in sorted(missing))
        raise RuntimeError(
            "TMol JAX disulfide FFI handlers are not installed/registered: " f"{names}"
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
            workspace_shape = _result(jax, (pairs.shape[0], 6, 3), dtype)
            gradient, _ = _call(
                "recompute_vjp",
                (gradient_shape, workspace_shape),
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
    """Return one disulfide energy per pair through shared XLA FFI handlers.

    The static ``recompute_backward`` policy must be a concrete Python bool.
    The default differentiated forward retains pair derivatives; recompute mode
    instead retains the three inputs and delegates derivative scratch ownership
    to the XLA FFI handler. Autodiff is defined only for coordinates. Pairs and
    parameters are nondifferentiable constants. CUDA calls require immutable
    pairs prepared and index-validated on the host before device upload.
    """

    _validate_inputs(coords, pairs, params, recompute_backward)
    _require_targets(recompute_backward)
    return _score_call(tuple(coords.shape), str(coords.dtype), recompute_backward)(
        coords, pairs, params
    )


__all__ = ["DISULFIDE_CPU_FFI_TARGETS", "register_ffi_targets", "score"]
