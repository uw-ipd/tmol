"""Near-planar torsions retain accurate values and finite derivatives."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.common.geom import dihedral_angle


def test_planar_torsion_gradient_and_degenerate_mask():
    angles = np.array([0, 1e-5, np.pi - 1e-5, np.pi], dtype=np.float32)
    coords = np.tile(
        np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]]), (4, 1, 1)
    ).astype(np.float32)
    coords[:, 3, 1] = np.cos(angles)
    coords[:, 3, 2] = np.sin(angles)

    def energy(x):
        phi = dihedral_angle(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
        return jnp.cos(3 * phi - 0.4).sum()

    gradient = jax.jit(jax.grad(energy))
    expected = -3 * np.sin(3 * angles.astype(np.float64) - 0.4) * np.cos(angles)
    np.testing.assert_allclose(
        gradient(jnp.asarray(coords))[:, 3, 2], expected, atol=3e-6
    )
    np.testing.assert_array_equal(gradient(jnp.zeros_like(coords)), 0)
