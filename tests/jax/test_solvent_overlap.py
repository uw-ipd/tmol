"""Water overlap saturates at one and absent waters have finite zero gradients."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.lk_ball.potentials import LK_BALL_DEFAULTS, lk_bridge_fraction


def test_water_bridge_saturation_and_empty_water_mask():
    first = jnp.zeros(3)
    second = jnp.array([np.sqrt(8 / 3) * LK_BALL_DEFAULTS.lkb_water_dist, 0, 0])
    waters = jnp.zeros((2, 3))

    def score(w, mask):
        return lk_bridge_fraction(first, second, w, waters, waters_i_mask=mask)

    full = jnp.ones(2, dtype=bool)
    empty = jnp.zeros(2, dtype=bool)
    np.testing.assert_allclose(score(waters, full), 1.0, atol=1e-6)
    np.testing.assert_array_equal(score(waters, empty), 0.0)
    np.testing.assert_array_equal(jax.grad(score)(waters, empty), 0.0)
    assert np.isfinite(jax.grad(score)(waters, full)).all()
