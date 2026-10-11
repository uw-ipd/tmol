"""Validate the stable basis against every canonical source polynomial."""

from pathlib import Path
from tmol_jax.database.scoring.hbond import PolynomialParameters

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from tmol_jax.score.hbond.params import _poly_to_array
from tmol_jax.score.hbond.potentials import _eval_packed_poly


def test_all_hbond_polynomials():
    source = Path(__file__).parents[2] / "tmol/database/default/scoring/hbond.yaml"
    rows = yaml.safe_load(source.read_text())["polynomial_parameters"]
    max_value_error = max_gradient_error = 0.0
    evaluate = jax.jit(jax.vmap(_eval_packed_poly, in_axes=(0, None)))
    differentiate = jax.jit(jax.vmap(jax.grad(_eval_packed_poly), in_axes=(0, None)))
    for row in rows:
        packed = jnp.asarray(_poly_to_array(PolynomialParameters(**row)))
        # Use matching floating inputs and stay away from discontinuous bounds.
        x = np.linspace(row["xmin"], row["xmax"], 513, dtype=np.float32)[1:-1]
        coefficients = np.array([row.get(f"c_{i}", 0.0) for i in range(10, -1, -1)])
        expected = np.polyval(coefficients, x.astype(np.float64))
        actual = np.asarray(evaluate(x, packed))
        expected_grad = np.polyval(np.polyder(coefficients), x.astype(np.float64))
        actual_grad = np.asarray(differentiate(x, packed))
        max_value_error = max(max_value_error, float(np.max(abs(actual - expected))))
        max_gradient_error = max(
            max_gradient_error, float(np.max(abs(actual_grad - expected_grad)))
        )
        np.testing.assert_allclose(
            actual, expected, rtol=3e-5, atol=3e-5, err_msg=row["name"]
        )
        np.testing.assert_allclose(
            actual_grad, expected_grad, rtol=5e-5, atol=5e-5, err_msg=row["name"]
        )
    print("maximum value/gradient errors", max_value_error, max_gradient_error)


def test_empty_polynomial_is_finite_and_zero():
    values = jnp.array([-100.0, 0.0, 100.0])
    poly = jnp.zeros(15)
    np.testing.assert_array_equal(
        jax.vmap(_eval_packed_poly, in_axes=(0, None))(values, poly), 0
    )
    np.testing.assert_array_equal(
        jax.vmap(jax.grad(_eval_packed_poly), in_axes=(0, None))(values, poly), 0
    )
