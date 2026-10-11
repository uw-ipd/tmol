"""Cubic-Hermite polynomial interpolation in pure JAX.

Port of ``tmol/score/common/cubic_hermite_polynomial.hh``. We only port
the ``V`` (value) variants — JAX's ``jax.grad`` handles derivatives, so
the C++ ``_dx`` / ``_V_dV`` helpers are unnecessary.

A cubic Hermite polynomial is defined by the value and first derivative
at the two endpoints ``(x0, y0, dy0)`` and ``(x1, y1, dy1)``. The
implementation maps ``x`` onto the unit interval ``t = (x - x0) / (x1 - x0)``
and evaluates the standard cubic Hermite form.
"""

from __future__ import annotations

from tmol_jax.types import Array


def interpolate_t(t: Array, p0: Array, dp0: Array, p1: Array, dp1: Array) -> Array:
    """Cubic Hermite interpolation of ``p`` on ``t`` in ``[0, 1]``.

    Upstream formula from ``cubic_hermite_polynomial.hh``.
    """
    return p0 + t * (
        dp0 + t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 + t * (dp0 + dp1 + 2 * p0 - 2 * p1))
    )


def interpolate_to_zero_t(t: Array, p0: Array, dp0: Array) -> Array:
    """Cubic Hermite interpolation of ``p`` on ``t`` in ``[0, 1]`` with
    ``(p1, dp1) == (0, 0)`` at ``t = 1``.
    """
    return p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0)))


def interpolate(
    x: Array,
    x0: Array,
    y0: Array,
    dy0: Array,
    x1: Array,
    y1: Array,
    dy1: Array,
) -> Array:
    """Value of the cubic Hermite polynomial on ``[x0, x1]`` at ``x``.

    ``dy0`` and ``dy1`` are derivatives with respect to ``x`` at the two
    endpoints; they are rescaled to the unit interval internally.
    """
    dx = x1 - x0
    t = (x - x0) / dx
    dp0 = dy0 * dx
    dp1 = dy1 * dx
    return interpolate_t(t, y0, dp0, y1, dp1)


def interpolate_to_zero(x: Array, x0: Array, y0: Array, dy0: Array, x1: Array) -> Array:
    """Cubic Hermite interpolation of ``y`` on ``[x0, x1]`` to
    ``(y1, dy1) == (0, 0)`` at ``x1``.
    """
    dx = x1 - x0
    t = (x - x0) / dx
    dp0 = dy0 * dx
    return interpolate_to_zero_t(t, y0, dp0)


__all__ = [
    "interpolate",
    "interpolate_to_zero",
    "interpolate_t",
    "interpolate_to_zero_t",
]
