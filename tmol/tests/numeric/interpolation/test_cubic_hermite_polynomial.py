import pytest
from pytest import approx

import hypothesis
import hypothesis.strategies


import math
import numba

import tmol.numeric.interpolation.cubic_hermite_polynomial as cubic_hermite_polynomial

interpolate_t = numba.jit(cubic_hermite_polynomial.interpolate_t, nopython=True)
interpolate_dt = numba.jit(cubic_hermite_polynomial.interpolate_dt, nopython=True)
interpolate_to_zero_t = numba.jit(
    cubic_hermite_polynomial.interpolate_to_zero_t, nopython=True
)
interpolate_to_zero_dt = numba.jit(
    cubic_hermite_polynomial.interpolate_to_zero_dt, nopython=True
)
interpolate = numba.jit(cubic_hermite_polynomial.interpolate, nopython=True)
interpolate_dx = numba.jit(cubic_hermite_polynomial.interpolate_dx, nopython=True)
interpolate_to_zero = numba.jit(
    cubic_hermite_polynomial.interpolate_to_zero, nopython=True
)
interpolate_to_zero_dx = numba.jit(
    cubic_hermite_polynomial.interpolate_to_zero_dx, nopython=True
)

# Use width=16 restricting test values to "reasonable" precision (e > -8)
real = hypothesis.strategies.floats(allow_infinity=False, width=16)


@hypothesis.given(real, real, real, real)
@hypothesis.settings(deadline=None, derandomize=True)
def test_unit_interpolate(p0, dp0, p1, dp1):
    params = (p0, dp0, p1, dp1)
    if any(map(math.isnan, params)):
        assert math.isnan(interpolate_t(0.0, p0, dp0, p1, dp1))
        assert math.isnan(interpolate_dt(0.0, p0, dp0, p1, dp1))
        assert math.isnan(interpolate_t(1.0, p0, dp0, p1, dp1))
        assert math.isnan(interpolate_dt(1.0, p0, dp0, p1, dp1))
    else:
        assert interpolate_t(0.0, p0, dp0, p1, dp1) == approx(p0)
        assert interpolate_dt(0.0, p0, dp0, p1, dp1) == approx(dp0)
        assert interpolate_t(1.0, p0, dp0, p1, dp1) == approx(p1)
        assert interpolate_dt(1.0, p0, dp0, p1, dp1) == approx(dp1)


@hypothesis.given(real, real)
@hypothesis.settings(deadline=None, derandomize=True)
def test_unit_interpolate_to_zero(p0, dp0):
    params = (p0, dp0)
    if any(map(math.isnan, params)):
        assert math.isnan(interpolate_to_zero_t(0.0, p0, dp0))
        assert math.isnan(interpolate_to_zero_dt(0.0, p0, dp0))
        assert math.isnan(interpolate_to_zero_t(1.0, p0, dp0))
        assert math.isnan(interpolate_to_zero_dt(1.0, p0, dp0))
    else:
        assert interpolate_to_zero_t(0.0, p0, dp0) == approx(p0)
        assert interpolate_to_zero_dt(0.0, p0, dp0) == approx(dp0)
        assert interpolate_to_zero_t(1.0, p0, dp0) == approx(0.0)
        assert interpolate_to_zero_dt(1.0, p0, dp0) == approx(0.0)


@hypothesis.given(real, real, real, real, real, real)
@hypothesis.settings(deadline=None, derandomize=True)
def test_interpolate(x0, p0, dpdx0, x1, p1, dpdx1):
    params = (x0, p0, dpdx0, x1, p1, dpdx1)
    if x0 == x1:
        with pytest.raises(ZeroDivisionError):
            interpolate(x0, x0, p0, dpdx0, x1, p1, dpdx1)
        with pytest.raises(ZeroDivisionError):
            interpolate_dx(x0, x0, p0, dpdx0, x1, p1, dpdx1)
        with pytest.raises(ZeroDivisionError):
            interpolate(x1, x0, p0, dpdx0, x1, p1, dpdx1)
        with pytest.raises(ZeroDivisionError):
            interpolate_dx(x1, x0, p0, dpdx0, x1, p1, dpdx1)
    elif any(map(math.isnan, params)):
        assert math.isnan(interpolate(x0, x0, p0, dpdx0, x1, p1, dpdx1))
        assert math.isnan(interpolate_dx(x0, x0, p0, dpdx0, x1, p1, dpdx1))
        assert math.isnan(interpolate(x1, x0, p0, dpdx0, x1, p1, dpdx1))
        assert math.isnan(interpolate_dx(x1, x0, p0, dpdx0, x1, p1, dpdx1))
    else:
        assert interpolate(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(p0)
        assert interpolate_dx(x0, x0, p0, dpdx0, x1, p1, dpdx1) == approx(dpdx0)
        assert interpolate(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(p1)
        assert interpolate_dx(x1, x0, p0, dpdx0, x1, p1, dpdx1) == approx(dpdx1)


@hypothesis.given(real, real, real, real, real, real)
@hypothesis.settings(deadline=None, derandomize=True)
def test_interpolate_to_zero(x0, p0, dpdx0, x1, p1, dpdx1):
    params = (x0, p0, dpdx0, x1)
    if x0 == x1:
        with pytest.raises(ZeroDivisionError):
            interpolate_to_zero(x0, x0, p0, dpdx0, x1)
        with pytest.raises(ZeroDivisionError):
            interpolate_to_zero_dx(x0, x0, p0, dpdx0, x1)
        with pytest.raises(ZeroDivisionError):
            interpolate_to_zero(x1, x0, p0, dpdx0, x1)
        with pytest.raises(ZeroDivisionError):
            interpolate_to_zero_dx(x1, x0, p0, dpdx0, x1)
    elif any(map(math.isnan, params)):
        assert math.isnan(interpolate_to_zero(x0, x0, p0, dpdx0, x1))
        assert math.isnan(interpolate_to_zero_dx(x0, x0, p0, dpdx0, x1))
        assert math.isnan(interpolate_to_zero(x1, x0, p0, dpdx0, x1))
        assert math.isnan(interpolate_to_zero_dx(x1, x0, p0, dpdx0, x1))
    else:
        assert interpolate_to_zero(x0, x0, p0, dpdx0, x1) == approx(p0)
        assert interpolate_to_zero_dx(x0, x0, p0, dpdx0, x1) == approx(dpdx0)
        assert interpolate_to_zero(x1, x0, p0, dpdx0, x1) == approx(0.0)
        assert interpolate_to_zero_dx(x1, x0, p0, dpdx0, x1) == approx(0.0)
