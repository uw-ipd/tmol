from tmol._load_ext import _load_module

_m = _load_module(
    __name__,
    __file__,
    "_cubic_hermite_polynomial.cpp",
    "tmol.score.common._cubic_hermite_polynomial",
)

_interpolate = _m.interpolate
_interpolate_dx = _m.interpolate_dx
_interpolate_t = _m.interpolate_t
_interpolate_dt = _m.interpolate_dt
_interpolate_to_zero = _m.interpolate_to_zero
_interpolate_to_zero_dx = _m.interpolate_to_zero_dx
_interpolate_to_zero_V_dV = _m.interpolate_to_zero_V_dV
_interpolate_to_zero_t = _m.interpolate_to_zero_t
_interpolate_to_zero_dt = _m.interpolate_to_zero_dt

__all__ = [
    "_interpolate",
    "_interpolate_dt",
    "_interpolate_dx",
    "_interpolate_t",
    "_interpolate_to_zero",
    "_interpolate_to_zero_V_dV",
    "_interpolate_to_zero_dt",
    "_interpolate_to_zero_dx",
    "_interpolate_to_zero_t",
]
