import ctypes
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from tmol.jax import disulfide

TARGETS = {
    "forward": "tmol_disulfide_cpu_ffi_forward",
    "tracked_forward": "tmol_disulfide_cpu_ffi_forward_tracked",
    "vjp": "tmol_disulfide_cpu_ffi_vjp",
    "recompute_vjp": "tmol_disulfide_cpu_ffi_recompute_vjp",
}


def inputs(dtype):
    coords = jnp.asarray(
        [
            [-1.0, 1.0, 0.2],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.05, 0.1, 0.1],
            [2.2, 1.0, 0.3],
            [3.0, 1.3, 0.9],
        ],
        dtype=dtype,
    )
    pairs = jnp.asarray([[0, 1, 2, 3, 4, 5, 0, 1]], dtype=jnp.int32)
    values = np.asarray(
        [
            2.03,
            0.15,
            -0.2,
            0.1,
            1.2,
            1.8,
            0.2,
            1.1,
            -1.5,
            -0.1,
            0.8,
            1.4,
            0.1,
            -1.2,
            0.7,
            -0.2,
            0.1,
            0.9,
            0.05,
            1.3,
            1.0,
            0.6,
            0.4,
            0.8,
            1.1,
            0.3,
            0.15,
            0.9,
            -1.0,
            -0.05,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.dtype(dtype),
    )
    params = np.stack((values, values))
    params[1, [13, 16, 19]] *= -1
    params[1, 32] = -1
    return coords, pairs, jnp.asarray(params)


def value_and_gradient(coords, pairs, params, recompute):
    cotangent = jnp.full((pairs.shape[0],), 0.7, dtype=coords.dtype)

    def objective(x):
        return jnp.vdot(
            disulfide.score(x, pairs, params, recompute_backward=recompute),
            cotangent,
        )

    return disulfide.score(
        coords, pairs, params, recompute_backward=recompute
    ), jax.grad(objective)(coords)


def exported_symbols(path):
    output = subprocess.check_output(
        ["nm", "-D", "--defined-only", "--format=posix", path], text=True
    )
    entries = [line.split() for line in output.splitlines() if line]
    return {
        fields[0]
        for fields in entries
        if len(fields) >= 2 and fields[1] not in {"W", "w", "V", "v"}
    }


def main():
    adapter_path = Path(sys.argv[1])
    common_path = Path(sys.argv[2])
    platform = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    library = ctypes.cdll.LoadLibrary(str(adapter_path))
    capsules = {
        role: jax.ffi.pycapsule(getattr(library, name))
        for role, name in TARGETS.items()
    }
    disulfide.register_ffi_targets(capsules, platform=platform)

    for dtype, tolerance in ((jnp.float32, 2e-5), (jnp.float64, 1e-12)):
        coords, pairs, params = inputs(dtype)
        retained_value, retained_gradient = value_and_gradient(
            coords, pairs, params, False
        )
        recompute_value, recompute_gradient = value_and_gradient(
            coords, pairs, params, True
        )
        np.testing.assert_allclose(
            np.asarray(recompute_value),
            np.asarray(retained_value),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            np.asarray(recompute_gradient),
            np.asarray(retained_gradient),
            rtol=tolerance,
            atol=tolerance,
        )
        assert np.isfinite(np.asarray(retained_value)).all()
        assert np.isfinite(np.asarray(retained_gradient)).all()

        epsilon = 1e-3 if dtype == jnp.float32 else 1e-6
        plus = coords.at[2, 0].add(epsilon)
        minus = coords.at[2, 0].add(-epsilon)
        cotangent = jnp.full((pairs.shape[0],), 0.7, dtype=dtype)
        finite_difference = (
            jnp.vdot(disulfide.score(plus, pairs, params), cotangent)
            - jnp.vdot(disulfide.score(minus, pairs, params), cotangent)
        ) / (2 * epsilon)
        np.testing.assert_allclose(
            np.asarray(retained_gradient[2, 0]),
            np.asarray(finite_difference),
            rtol=3e-2 if dtype == jnp.float32 else 1e-5,
            atol=3e-3 if dtype == jnp.float32 else 2e-5,
        )

        overlapping_pairs = jnp.concatenate((pairs, pairs), axis=0)
        overlap_value, overlap_gradient = value_and_gradient(
            coords, overlapping_pairs, params, False
        )
        overlap_recompute_value, overlap_recompute_gradient = value_and_gradient(
            coords, overlapping_pairs, params, True
        )
        np.testing.assert_allclose(
            np.asarray(overlap_value),
            np.repeat(np.asarray(retained_value), 2),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            np.asarray(overlap_gradient),
            2 * np.asarray(retained_gradient),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            np.asarray(overlap_recompute_value),
            np.asarray(overlap_value),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            np.asarray(overlap_recompute_gradient),
            np.asarray(overlap_gradient),
            rtol=tolerance,
            atol=tolerance,
        )

        empty_coords = jnp.empty((0, 3), dtype=dtype)
        empty_pairs = jnp.empty((0, 8), dtype=jnp.int32)
        empty_params = jnp.empty((0, 33), dtype=dtype)
        for recompute in (False, True):
            value, gradient = value_and_gradient(
                empty_coords, empty_pairs, empty_params, recompute
            )
            assert value.shape == (0,)
            assert gradient.shape == (0, 3)

    if platform == "cpu":
        coords, pairs, params = inputs(jnp.float64)
        invalid_pairs = pairs.at[0, 2].set(coords.shape[0])
        try:
            np.asarray(disulfide.score(coords, invalid_pairs, params))
        except Exception as error:
            assert "pair index out of bounds" in str(error).lower()
        else:
            raise AssertionError("invalid pair atom index unexpectedly succeeded")

    assert exported_symbols(adapter_path) == set(TARGETS.values())
    common_symbols = exported_symbols(common_path)
    assert common_symbols
    common_prefix = (
        "tmol_disulfide_cpu_" if platform == "cpu" else "tmol_disulfide_cuda_"
    )
    assert all(symbol.startswith(common_prefix) for symbol in common_symbols)
    common_linkage = subprocess.check_output(["ldd", common_path], text=True)
    assert "libtorch" not in common_linkage
    assert "jax" not in common_linkage


if __name__ == "__main__":
    main()
