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
computeCoeffs2Batch = _mod.computeCoeffs2Batch
computeCoeffs3Batch = _mod.computeCoeffs3Batch
computeCoeffs4Batch = _mod.computeCoeffs4Batch
interpolate2 = _mod.interpolate2
interpolate3 = _mod.interpolate3
interpolate4 = _mod.interpolate4

__all__ = [
    "computeCoeffs2",
    "computeCoeffs3",
    "computeCoeffs4",
    "computeCoeffs2Batch",
    "computeCoeffs3Batch",
    "computeCoeffs4Batch",
    "interpolate2",
    "interpolate3",
    "interpolate4",
]
