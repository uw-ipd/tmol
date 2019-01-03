import numpy
from pathlib import Path
from tmol.utility.cpp_extension import load

_polynomial = load(
    "_polynomial", [str(Path(__file__).parent / s) for s in ("_polynomial.cpp",)]
)

poly_v = numpy.vectorize(_polynomial.poly_v, signature="(),(n)->()")
poly_v_d = numpy.vectorize(_polynomial.poly_v_d, signature="(),(n)->(),()")
