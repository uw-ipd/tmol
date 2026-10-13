"""Cubic Hermite polynomial helpers (direct port of
``tmol.numeric.interpolation.cubic_hermite_polynomial``).

Each pair consists of a *value* function (``interpolate*``) and a
*derivative* function (``interpolate*_dx`` / ``interpolate*_dt``) with
the same endpoint constraints. The ``interpolate_to_zero*`` variants
anchor the right endpoint at ``(p1, dp1) = (0, 0)`` and are used by
smooth knotted potentials that fade to zero.

All functions work elementwise on scalars or arrays and are
differentiable by :func:`jax.grad` / :func:`jax.jacfwd`.
"""

from __future__ import annotations


def interpolate_t(t, p0, dp0, p1, dp1):
    """Cubic interpolation of ``p`` on ``t in [0, 1]``."""
    return p0 + t * (
        dp0
        + t
        * (
            -2.0 * dp0
            - dp1
            - 3.0 * p0
            + 3.0 * p1
            + t * (dp0 + dp1 + 2.0 * p0 - 2.0 * p1)
        )
    )


def interpolate_dt(t, p0, dp0, p1, dp1):
    """Cubic interpolation of ``dp/dt`` on ``t in [0, 1]``."""
    return dp0 + t * (
        -4.0 * dp0
        - 2.0 * dp1
        - 6.0 * p0
        + 6.0 * p1
        + t * (3.0 * dp0 + 3.0 * dp1 + 6.0 * p0 - 6.0 * p1)
    )


def interpolate(x, x0, p0, dpdx0, x1, p1, dpdx1):
    """Cubic interpolation of ``p`` on ``x in [x0, x1]``."""
    dx = x1 - x0
    t = (x - x0) / dx
    dp0 = dpdx0 * dx
    dp1 = dpdx1 * dx
    return p0 + t * (
        dp0
        + t
        * (
            -2.0 * dp0
            - dp1
            - 3.0 * p0
            + 3.0 * p1
            + t * (dp0 + dp1 + 2.0 * p0 - 2.0 * p1)
        )
    )


def interpolate_dx(x, x0, p0, dpdx0, x1, p1, dpdx1):
    """Cubic interpolation of ``dp/dx`` on ``x in [x0, x1]``."""
    dx = x1 - x0
    t = (x - x0) / dx
    dp0 = dpdx0 * dx
    dp1 = dpdx1 * dx
    dp = dp0 + t * (
        -4.0 * dp0
        - 2.0 * dp1
        - 6.0 * p0
        + 6.0 * p1
        + t * (3.0 * dp0 + 3.0 * dp1 + 6.0 * p0 - 6.0 * p1)
    )
    return dp / dx


def interpolate_to_zero_t(t, p0, dp0):
    """Cubic interpolation of ``p`` on ``t in [0, 1]`` to ``(p1, dp1) == 0``."""
    return p0 + t * (dp0 + t * (-2.0 * dp0 - 3.0 * p0 + t * (dp0 + 2.0 * p0)))


def interpolate_to_zero_dt(t, p0, dp0):
    """Cubic interpolation of ``dp/dt`` on ``t in [0, 1]`` to ``(p1, dp1) == 0``."""
    return dp0 + t * (-4.0 * dp0 - 6.0 * p0 + t * (3.0 * dp0 + 6.0 * p0))


def interpolate_to_zero(x, x0, p0, dpdx0, x1):
    """Cubic interpolation of ``p`` on ``x in [x0, x1]`` to zero at ``x1``."""
    dx = x1 - x0
    t = (x - x0) / dx
    dp0 = dpdx0 * dx
    return p0 + t * (dp0 + t * (-2.0 * dp0 - 3.0 * p0 + t * (dp0 + 2.0 * p0)))


def interpolate_to_zero_dx(x, x0, p0, dpdx0, x1):
    """Cubic interpolation of ``dp/dx`` on ``x in [x0, x1]`` to zero at ``x1``."""
    dx = x1 - x0
    t = (x - x0) / dx
    dp0 = dpdx0 * dx
    dp = dp0 + t * (-4.0 * dp0 - 6.0 * p0 + t * (3.0 * dp0 + 6.0 * p0))
    return dp / dx


__all__ = [
    "interpolate_t",
    "interpolate_dt",
    "interpolate",
    "interpolate_dx",
    "interpolate_to_zero_t",
    "interpolate_to_zero_dt",
    "interpolate_to_zero",
    "interpolate_to_zero_dx",
]
