import numpy

from tmol.tests.score.common.polynomial._ext import (
    poly_v as _poly_v,
    poly_v_d as _poly_v_d,
)

poly_v = numpy.vectorize(_poly_v, signature="(),(n)->()")
poly_v_d = numpy.vectorize(_poly_v_d, signature="(),(n)->(),()")
