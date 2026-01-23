import pytest

import hypothesis
import hypothesis.strategies
import toolz


import math
from math import nan
import numpy


def approx_for(values):
    values = numpy.array(values)
    atol = numpy.max(numpy.abs(values)) * 1e-5

    return toolz.partial(pytest.approx, nan_ok=True, rel=1e-5, abs=atol)


real = (
    hypothesis.strategies.decimals(
        allow_infinity=False, places=3, max_value=1e5, min_value=-1e5
    )
    .filter(lambda v: not v.is_snan())
    .map(float)
)


@pytest.fixture(scope="session")
def cubic_hermite_polynomial():
    """Compile within fixture to prevent hypothesis runtime warnings."""
    import tmol.score.common.cubic_hermite_polynomial as cubic_hermite_polynomial

    return cubic_hermite_polynomial


@hypothesis.given(real, real, real, real)
def test_unit_interpolate(cubic_hermite_polynomial, p0, dp0, p1, dp1):
    from tmol.score.common.cubic_hermite_polynomial import interpolate_t, interpolate_dt

    params = (p0, dp0, p1, dp1)
    approx = approx_for(params)

    if any(map(math.isnan, params)):
        assert interpolate_t(0.0, p0, dp0, p1, dp1) == approx(nan)
        assert interpolate_dt(0.0, p0, dp0, p1, dp1) == approx(nan)
        assert interpolate_t(1.0, p0, dp0, p1, dp1) == approx(nan)
        assert interpolate_dt(1.0, p0, dp0, p1, dp1) == approx(nan)
    else:
        assert interpolate_t(0.0, p0, dp0, p1, dp1) == approx(p0)
        assert interpolate_dt(0.0, p0, dp0, p1, dp1) == approx(dp0)
        assert interpolate_t(1.0, p0, dp0, p1, dp1) == approx(p1)
        assert interpolate_dt(1.0, p0, dp0, p1, dp1) == approx(dp1)


@hypothesis.given(real, real)
def test_unit_interpolate_to_zero(cubic_hermite_polynomial, p0, dp0):
    from tmol.score.common.cubic_hermite_polynomial import (
        interpolate_to_zero_t,
        interpolate_to_zero_dt,
    )

    params = (p0, dp0)
    approx = approx_for(params)

    if any(map(math.isnan, params)):
        assert interpolate_to_zero_t(0.0, p0, dp0) == approx(nan)
        assert interpolate_to_zero_dt(0.0, p0, dp0) == approx(nan)
        assert interpolate_to_zero_t(1.0, p0, dp0) == approx(nan)
        assert interpolate_to_zero_dt(1.0, p0, dp0) == approx(nan)
    else:
        assert interpolate_to_zero_t(0.0, p0, dp0) == approx(p0)
        assert interpolate_to_zero_dt(0.0, p0, dp0) == approx(dp0)
        assert interpolate_to_zero_t(1.0, p0, dp0) == approx(0.0)
        assert interpolate_to_zero_dt(1.0, p0, dp0) == approx(0.0)


@hypothesis.given(real, real, real, real, real, real)
def test_interpolate(cubic_hermite_polynomial, x0, p0, dpdx0, x1, p1, dpdx1):
    from tmol.score.common.cubic_hermite_polynomial import interpolate, interpolate_dx

    params = (x0, p0, dpdx0, x1, p1, dpdx1)
    approx = approx_for(params)
    if x0 == x1:
        assert interpolate(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
        assert interpolate_dx(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
        assert interpolate(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
        assert interpolate_dx(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
    elif any(map(math.isnan, params)):
        assert interpolate(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
        assert interpolate_dx(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
        assert interpolate(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
        assert interpolate_dx(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(nan)
    else:
        assert interpolate(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(p0)
        assert interpolate_dx(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(dpdx0)
        assert interpolate(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(p1)
        assert interpolate_dx(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(dpdx1)


@hypothesis.given(real, real, real, real, real, real)
def test_interpolate_to_zero(cubic_hermite_polynomial, x0, p0, dpdx0, x1, p1, dpdx1):
    from tmol.score.common.cubic_hermite_polynomial import (
        interpolate_to_zero,
        interpolate_to_zero_dx,
        interpolate_to_zero_V_dV,
    )

    params = (x0, p0, dpdx0, x1)
    approx = approx_for(params)

    if x0 == x1:
        assert interpolate_to_zero(x0, x0, p0, dpdx0, x1) == approx(nan)
        assert interpolate_to_zero_dx(x0, x0, p0, dpdx0, x1) == approx(nan)
        assert interpolate_to_zero(x1, x0, p0, dpdx0, x1) == approx(nan)
        assert interpolate_to_zero_dx(x1, x0, p0, dpdx0, x1) == approx(nan)
    elif any(map(math.isnan, params)):
        assert math.isnan(interpolate_to_zero(x0, x0, p0, dpdx0, x1))
        assert math.isnan(interpolate_to_zero_dx(x0, x0, p0, dpdx0, x1))
        assert math.isnan(interpolate_to_zero(x1, x0, p0, dpdx0, x1))
        assert math.isnan(interpolate_to_zero_dx(x1, x0, p0, dpdx0, x1))
    else:
        assert interpolate_to_zero(x0, x0, p0, dpdx0, x1) == approx(p0)
        assert interpolate_to_zero_dx(x0, x0, p0, dpdx0, x1) == approx(dpdx0)
        assert interpolate_to_zero_V_dV(x0, x0, p0, dpdx0, x1) == approx((p0, dpdx0))

        assert interpolate_to_zero(x1, x0, p0, dpdx0, x1) == approx(0.0)
        assert interpolate_to_zero_dx(x1, x0, p0, dpdx0, x1) == approx(0.0)
        assert interpolate_to_zero_V_dV(x1, x0, p0, dpdx0, x1) == approx((0.0, 0.0))
