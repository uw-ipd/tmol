from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, relpaths, modulename

    _mod = load(
        modulename(__name__),
        relpaths(__file__, ["bspline.pybind.cpp"]),
        is_python_module=True,
    )
    computeCoeffs2 = _mod.computeCoeffs2
    computeCoeffs3 = _mod.computeCoeffs3
    computeCoeffs4 = _mod.computeCoeffs4
    interpolate2 = _mod.interpolate2
    interpolate3 = _mod.interpolate3
    interpolate4 = _mod.interpolate4
else:
    from tmol.numeric.bspline_compiled._compiled import (
        computeCoeffs2,
        computeCoeffs3,
        computeCoeffs4,
        interpolate2,
        interpolate3,
        interpolate4,
    )

__all__ = [
    "computeCoeffs2",
    "computeCoeffs3",
    "computeCoeffs4",
    "interpolate2",
    "interpolate3",
    "interpolate4",
]
