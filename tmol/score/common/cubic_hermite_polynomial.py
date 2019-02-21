from tmol.utility.cpp_extension import load, relpaths, modulename

_cubic_hermite_polynomial = load(
    modulename(__name__), relpaths(__file__, "_cubic_hermite_polynomial.cpp")
)

interpolate = _cubic_hermite_polynomial.interpolate
interpolate_dx = _cubic_hermite_polynomial.interpolate_dx

interpolate_t = _cubic_hermite_polynomial.interpolate_t
interpolate_dt = _cubic_hermite_polynomial.interpolate_dt

interpolate_to_zero = _cubic_hermite_polynomial.interpolate_to_zero
interpolate_to_zero_dx = _cubic_hermite_polynomial.interpolate_to_zero_dx
interpolate_to_zero_V_dV = _cubic_hermite_polynomial.interpolate_to_zero_V_dV

interpolate_to_zero_t = _cubic_hermite_polynomial.interpolate_to_zero_t
interpolate_to_zero_dt = _cubic_hermite_polynomial.interpolate_to_zero_dt
