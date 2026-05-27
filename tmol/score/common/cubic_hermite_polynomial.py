from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, relpaths, modulename

    _mod = load(
        modulename(__name__),
        relpaths(__file__, ["_cubic_hermite_polynomial.cpp"]),
        is_python_module=True,
    )
    interpolate = _mod.interpolate
    interpolate_dt = _mod.interpolate_dt
    interpolate_dx = _mod.interpolate_dx
    interpolate_t = _mod.interpolate_t
    interpolate_to_zero = _mod.interpolate_to_zero
    interpolate_to_zero_dt = _mod.interpolate_to_zero_dt
    interpolate_to_zero_dx = _mod.interpolate_to_zero_dx
    interpolate_to_zero_t = _mod.interpolate_to_zero_t
    interpolate_to_zero_V_dV = _mod.interpolate_to_zero_V_dV
else:
    from tmol.score.common._cubic_hermite_polynomial import (
        interpolate,
        interpolate_dt,
        interpolate_dx,
        interpolate_t,
        interpolate_to_zero,
        interpolate_to_zero_dt,
        interpolate_to_zero_dx,
        interpolate_to_zero_t,
        interpolate_to_zero_V_dV,
    )

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
