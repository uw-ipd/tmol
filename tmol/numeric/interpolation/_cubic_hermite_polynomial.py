def interpolate_t(t, p0, dp0, p1, dp1):
    """Cubic interpolation of p on t in [0, 1]."""
    return p0 + t * (
        dp0 + t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 + t * (dp0 + dp1 + 2 * p0 - 2 * p1))
    )


def interpolate(x, x0, p0, dpdx0, x1, p1, dpdx1):
    """Cubic interpolation of p on x in [x0, x1]."""
    t = (x - x0) / (x1 - x0)
    dp0 = dpdx0 * (x1 - x0)
    dp1 = dpdx1 * (x1 - x0)

    return p0 + t * (
        dp0 + t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 + t * (dp0 + dp1 + 2 * p0 - 2 * p1))
    )


def interpolate_dt(t, p0, dp0, p1, dp1):
    """Cubic interpolation of dp/dt on t in [0, 1]."""
    return dp0 + t * (
        -4 * dp0 - 2 * dp1 - 6 * p0 + 6 * p1 + t * (3 * dp0 + 3 * dp1 + 6 * p0 - 6 * p1)
    )


def interpolate_dx(x, x0, p0, dpdx0, x1, p1, dpdx1):
    """Cubic interpolation of dp/dx on x in [x0, x1]."""
    t = (x - x0) / (x1 - x0)
    dp0 = dpdx0 * (x1 - x0)
    dp1 = dpdx1 * (x1 - x0)

    dp = dp0 + t * (
        -4 * dp0 - 2 * dp1 - 6 * p0 + 6 * p1 + t * (3 * dp0 + 3 * dp1 + 6 * p0 - 6 * p1)
    )

    return dp / (x1 - x0)


def interpolate_to_zero_t(t, p0, dp0):
    """Cubic interpolation of p on t in [0, 1] to (p1, dp1) == 0."""
    return p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0)))


def interpolate_to_zero(x, x0, p0, dpdx0, x1):
    """Cubic interpolation of p on x in [x0, x1] to (p1, dpdx1) == 0 at x1."""
    t = (x - x0) / (x1 - x0)
    dp0 = dpdx0 * (x1 - x0)

    return p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0)))


def interpolate_to_zero_dt(t, p0, dp0):
    """Cubic interpolation of dp/dt on t in [0, 1] to (p1, dp1) == 0."""
    return dp0 + t * (-4 * dp0 - 6 * p0 + t * (3 * dp0 + 6 * p0))


def interpolate_to_zero_dx(x, x0, p0, dpdx0, x1):
    """Cubic interpolation of dp/dx on x in [x0, x1] to (p1, dpdx1) == 0 at x1."""
    t = (x - x0) / (x1 - x0)
    dp0 = dpdx0 * (x1 - x0)

    dp = dp0 + t * (-4 * dp0 - 6 * p0 + t * (3 * dp0 + 6 * p0))

    return dp / (x1 - x0)
