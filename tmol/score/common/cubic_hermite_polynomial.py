from tmol._load_ext import load_module

_m = load_module(
    __name__,
    __file__,
    "_cubic_hermite_polynomial.cpp",
    "tmol.score.common._cubic_hermite_polynomial",
)

interpolate = _m.interpolate
interpolate_dx = _m.interpolate_dx
interpolate_t = _m.interpolate_t
interpolate_dt = _m.interpolate_dt
interpolate_to_zero = _m.interpolate_to_zero
interpolate_to_zero_dx = _m.interpolate_to_zero_dx
interpolate_to_zero_V_dV = _m.interpolate_to_zero_V_dV
interpolate_to_zero_t = _m.interpolate_to_zero_t
interpolate_to_zero_dt = _m.interpolate_to_zero_dt

__all__ = [
    "interpolate",
    "interpolate_dt",
    "interpolate_dx",
    "interpolate_t",
    "interpolate_to_zero",
    "interpolate_to_zero_V_dV",
    "interpolate_to_zero_dt",
    "interpolate_to_zero_dx",
    "interpolate_to_zero_t",
]
