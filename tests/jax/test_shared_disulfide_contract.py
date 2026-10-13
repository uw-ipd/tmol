"""Contract tests for the dormant shared-core JAX disulfide wrapper."""

import os
from pathlib import Path
import subprocess
import inspect
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol.jax import disulfide

TARGETS = {
    "forward": "tmol_disulfide_cpu_ffi_forward",
    "tracked_forward": "tmol_disulfide_cpu_ffi_forward_tracked",
    "vjp": "tmol_disulfide_cpu_ffi_vjp",
    "recompute_vjp": "tmol_disulfide_cpu_ffi_recompute_vjp",
}


@pytest.fixture(autouse=True)
def clean_registration():
    disulfide._REGISTERED_TARGETS.clear()
    disulfide._score_call.cache_clear()
    yield
    disulfide._REGISTERED_TARGETS.clear()
    disulfide._score_call.cache_clear()


def arrays(dtype=jnp.float32):
    return (
        jnp.zeros((7, 3), dtype=dtype),
        jnp.zeros((2, 8), dtype=jnp.int32),
        jnp.zeros((3, 33), dtype=dtype),
    )


def register_mock_targets(monkeypatch):
    registered = []
    monkeypatch.setattr(
        jax.ffi,
        "register_ffi_target",
        lambda name, capsule, platform: registered.append((name, capsule, platform)),
    )
    capsules = {role: object() for role in TARGETS}
    disulfide.register_ffi_targets(capsules)
    return capsules, registered


def install_mock_calls(monkeypatch):
    calls = []

    def call(role, result, *operands):
        calls.append((role, tuple(operand.shape for operand in operands)))
        if isinstance(result, tuple):
            return tuple(jnp.zeros(spec.shape, spec.dtype) for spec in result)
        return jnp.zeros(result.shape, result.dtype)

    monkeypatch.setattr(disulfide, "_call", call)
    return calls


def test_target_names_and_registration_are_stable(monkeypatch):
    assert dict(disulfide.DISULFIDE_CPU_FFI_TARGETS) == TARGETS
    capsules, registered = register_mock_targets(monkeypatch)
    assert [(name, platform) for name, _, platform in registered] == [
        (name, "CPU") for name in TARGETS.values()
    ]

    disulfide.register_ffi_targets(capsules)
    assert len(registered) == 4
    with pytest.raises(ValueError, match="conflicting.*forward"):
        disulfide.register_ffi_targets({"forward": object()})
    assert len(registered) == 4


def test_score_validates_static_mode_shapes_and_dtypes():
    coords, pairs, params = arrays()
    with pytest.raises(TypeError, match="concrete Python bool"):
        disulfide.score(coords, pairs, params, recompute_backward=np.bool_(False))
    with pytest.raises(ValueError, match=r"coords.*\[n_atoms, 3\]"):
        disulfide.score(coords[:, :2], pairs, params)
    with pytest.raises(ValueError, match=r"pairs.*\[n_pairs, 8\]"):
        disulfide.score(coords, pairs[:, :7], params)
    with pytest.raises(ValueError, match=r"params.*\[n_params, 33\]"):
        disulfide.score(coords, pairs, params[:, :32])
    with pytest.raises(TypeError, match="coords must have dtype"):
        disulfide.score(coords.astype(jnp.int32), pairs, params)
    with pytest.raises(TypeError, match="same dtype"):
        disulfide.score(coords, pairs, np.zeros((3, 33), dtype=np.float64))
    with pytest.raises(TypeError, match="pairs must have dtype int32"):
        disulfide.score(coords, np.zeros((2, 8), dtype=np.int64), params)


def test_missing_targets_fail_before_dispatch():
    with pytest.raises(RuntimeError, match="not installed/registered") as error:
        disulfide.score(*arrays())
    assert TARGETS["forward"] in str(error.value)


def test_default_vjp_retains_pair_derivatives(monkeypatch):
    register_mock_targets(monkeypatch)
    calls = install_mock_calls(monkeypatch)
    coords, pairs, params = arrays()

    assert disulfide.score(coords, pairs, params).shape == (2,)
    gradient = jax.grad(lambda x: disulfide.score(x, pairs, params).sum())(coords)

    assert gradient.shape == coords.shape
    assert [role for role, _ in calls] == ["forward", "tracked_forward", "vjp"]
    assert calls[-1][1] == ((2, 8), (2, 6, 3), (2,))


def test_recompute_vjp_saves_inputs_and_uses_recompute_target(monkeypatch):
    register_mock_targets(monkeypatch)
    calls = install_mock_calls(monkeypatch)
    coords, pairs, params = arrays()

    gradient = jax.grad(
        lambda x: disulfide.score(x, pairs, params, recompute_backward=True).sum()
    )(coords)

    assert gradient.shape == coords.shape
    assert [role for role, _ in calls] == ["forward", "recompute_vjp"]
    assert calls[-1][1] == ((7, 3), (2, 8), (3, 33), (2,))


def test_dispatch_source_has_no_host_callback_path():
    source = inspect.getsource(disulfide)
    assert "pure_callback" not in source
    assert "device_get" not in source
    assert "np.asarray" not in source


def test_importing_package_does_not_require_jax():
    repository = Path(__file__).parents[2]
    script = r"""
import builtins
import sys
real_import = builtins.__import__

def block_jax(name, *args, **kwargs):
    if name.partition(".")[0] == "jax":
        raise ModuleNotFoundError("No module named 'jax'", name="jax")
    return real_import(name, *args, **kwargs)

builtins.__import__ = block_jax
import tmol.jax
assert "jax" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        check=True,
        env={**os.environ, "PYTHONPATH": str(repository)},
    )
