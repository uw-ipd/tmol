from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(__file__, ["bspline.pybind.cpp"])

functions = [
    "computeCoeffs2",
    "interpolate2",
    "computeCoeffs3",
    "interpolate3",
    "computeCoeffs4",
    "interpolate4",
]

loader = TorchOpLoader(__name__, sources, functions)

computeCoeffs2 = loader.computeCoeffs2
interpolate2 = loader.interpolate2
computeCoeffs3 = loader.computeCoeffs3
interpolate3 = loader.interpolate3
computeCoeffs4 = loader.computeCoeffs4
interpolate4 = loader.interpolate4
