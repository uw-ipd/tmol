from tmol._load_ext import _load_module

_mod = _load_module(
    __name__,
    __file__,
    ["bspline.pybind.cpp"],
    "tmol.numeric.bspline_compiled._compiled",
)

_computeCoeffs2 = _mod.computeCoeffs2
_computeCoeffs3 = _mod.computeCoeffs3
_computeCoeffs4 = _mod.computeCoeffs4
_computeCoeffs2Batch = _mod.computeCoeffs2Batch
_computeCoeffs3Batch = _mod.computeCoeffs3Batch
_computeCoeffs4Batch = _mod.computeCoeffs4Batch
_interpolate2 = _mod.interpolate2
_interpolate3 = _mod.interpolate3
_interpolate4 = _mod.interpolate4

__all__ = [
    "_computeCoeffs2",
    "_computeCoeffs3",
    "_computeCoeffs4",
    "_computeCoeffs2Batch",
    "_computeCoeffs3Batch",
    "_computeCoeffs4Batch",
    "_interpolate2",
    "_interpolate3",
    "_interpolate4",
]
