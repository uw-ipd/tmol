"""Bounded polynomial evaluation (used by hbond scoring).

Port of the ``bound_poly`` struct from
``tmol/score/common/polynomial.hh``. Outside the bracketed x-range the
polynomial is clamped to a constant, otherwise the coefficients are
evaluated by Horner's method.

The canonical hbond usage is an 11-coefficient polynomial (degree 10)
though nothing in the implementation hard-codes that length.
"""

from __future__ import annotations

import jax.numpy as jnp

from tmol_jax.types import Array


def poly_v(x: Array, coeffs: Array) -> Array:
    """Evaluate ``sum(coeffs[i] * x**(N-1-i))`` via Horner.

    This matches ``poly<POrd, ...>::V`` in ``polynomial.hh``: coefficients
    are given in descending order of power.

    Works for both a simple 1-D ``coeffs`` array (``jnp.polyval`` form)
    and a batched ``[..., n_coeffs]`` array broadcast-compatible with
    ``x``. The hbond scoring pipeline uses the batched form: coefficients
    are pre-gathered to the ``[P, D, A, n_coeffs]`` pair grid and
    evaluated once against the matching ``[P, D, A]`` geometry scalar.
    """
    coeffs = jnp.asarray(coeffs)
    if coeffs.ndim <= 1:
        # Fast path: scalar / 1-D coefficient vector; delegate to numpy.
        return jnp.polyval(coeffs, x)

    # Batched Horner. Coefficients are in descending power order, so the
    # leading coefficient drives ``y`` first and we iterate down to the
    # constant term. ``coeffs[..., i]`` has the same batch shape as the
    # leading axes of ``x`` (up to broadcasting).
    n = coeffs.shape[-1]
    y = coeffs[..., 0]
    for i in range(1, n):
        y = y * x + coeffs[..., i]
    return y


def bound_poly(
    x: Array,
    coeffs: Array,
    xrange: tuple | Array,
    ybound: tuple | Array,
) -> Array:
    """Evaluate a bounded polynomial.

    Parameters
    ----------
    x:
        Point(s) at which to evaluate. Scalar or array.
    coeffs:
        Polynomial coefficients in descending-power order. Shape is
        either ``[n_coeffs]`` (simple univariate polynomial) or
        ``[..., n_coeffs]`` broadcast-compatible with ``x`` (the hbond
        use case: one polynomial per pair gathered to a ``[P, D, A]``
        grid).
    xrange:
        ``(x_lo, x_hi)`` — bounds of the polynomial's support. Scalar or
        broadcast-compatible ``[..., 2]`` array.
    ybound:
        ``(y_lo, y_hi)`` — values to return when ``x`` falls outside
        ``xrange``.

    Returns
    -------
    ``y_lo`` if ``x < x_lo``, ``y_hi`` if ``x > x_hi``, else
    ``sum(coeffs[i] * x**(N-1-i))``.
    """
    # Support both tuple-style (x_lo, x_hi) and array-style [..., 2] inputs.
    if isinstance(xrange, (tuple, list)):
        x_lo, x_hi = xrange[0], xrange[1]
    else:
        xrange_arr = jnp.asarray(xrange)
        x_lo = xrange_arr[..., 0]
        x_hi = xrange_arr[..., 1]

    if isinstance(ybound, (tuple, list)):
        y_lo, y_hi = ybound[0], ybound[1]
    else:
        ybound_arr = jnp.asarray(ybound)
        y_lo = ybound_arr[..., 0]
        y_hi = ybound_arr[..., 1]

    poly_val = poly_v(x, coeffs)
    # Cascade: low bound wins, then high bound, else polynomial.
    value = jnp.where(x > x_hi, y_hi, poly_val)
    value = jnp.where(x < x_lo, y_lo, value)
    return value


__all__ = ["bound_poly", "poly_v"]
