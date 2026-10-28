import numpy

from tmol._load_ext import _load_module

_polynomial = _load_module(
    __name__,
    __file__,
    "polynomial.pybind.cpp",
    "tmol.tests.score.common.polynomial._ext",
)

poly_v = numpy.vectorize(_polynomial.poly_v, signature="(),(n)->()")
poly_v_d = numpy.vectorize(_polynomial.poly_v_d, signature="(),(n)->(),()")
