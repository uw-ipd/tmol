import numpy
from tmol.utility.cpp_extension import load, relpaths, modulename

_polynomial = load(modulename(__name__), relpaths(__file__, "_polynomial.cpp"))

poly_v = numpy.vectorize(_polynomial.poly_v, signature="(),(n)->()")
poly_v_d = numpy.vectorize(_polynomial.poly_v_d, signature="(),(n)->(),()")
