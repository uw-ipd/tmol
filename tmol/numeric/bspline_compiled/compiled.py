from tmol.utility.cpp_extension import load, relpaths, modulename

# only CPU is exposed via pybind
_compiled = load(modulename(__name__), relpaths(__file__, ["bspline.pybind.cpp"]))

computeCoeffs2 = _compiled.computeCoeffs2
interpolate2 = _compiled.interpolate2
computeCoeffs3 = _compiled.computeCoeffs3
interpolate3 = _compiled.interpolate3
computeCoeffs4 = _compiled.computeCoeffs4
interpolate4 = _compiled.interpolate4
