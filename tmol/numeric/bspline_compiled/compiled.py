from tmol._load_ext import load_module

_mod = load_module(
    __name__,
    __file__,
    ["bspline.pybind.cpp"],
    "tmol.numeric.bspline_compiled._compiled",
)

computeCoeffs2 = _mod.computeCoeffs2
computeCoeffs3 = _mod.computeCoeffs3
computeCoeffs4 = _mod.computeCoeffs4
interpolate2 = _mod.interpolate2
interpolate3 = _mod.interpolate3
interpolate4 = _mod.interpolate4

__all__ = [
    "computeCoeffs2",
    "computeCoeffs3",
    "computeCoeffs4",
    "interpolate2",
    "interpolate3",
    "interpolate4",
]
