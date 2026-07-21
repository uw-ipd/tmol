import numpy

from tmol._load_ext import load_module

_geom = load_module(
    __name__,
    __file__,
    "geom.pybind.cpp",
    "tmol.tests.score.common.geom._ext",
)

# Wrap pybind11 scalar functions with numpy.vectorize so they broadcast
# over arrays of inputs and provide the gufunc signature needed by
# VectorizedOp in the test autograd harness.

distance_V = numpy.vectorize(_geom.distance_V, signature="(3),(3)->()")
distance_V_dV = numpy.vectorize(_geom.distance_V_dV, signature="(3),(3)->(),(3),(3)")

interior_angle_V = numpy.vectorize(_geom.interior_angle_V, signature="(3),(3)->()")
interior_angle_V_dV = numpy.vectorize(
    _geom.interior_angle_V_dV, signature="(3),(3)->(),(3),(3)"
)

cos_interior_angle_V = numpy.vectorize(
    _geom.cos_interior_angle_V, signature="(3),(3)->()"
)
cos_interior_angle_V_dV = numpy.vectorize(
    _geom.cos_interior_angle_V_dV, signature="(3),(3)->(),(3),(3)"
)

dihedral_angle_V = numpy.vectorize(
    _geom.dihedral_angle_V, signature="(3),(3),(3),(3)->()"
)
dihedral_angle_V_dV = numpy.vectorize(
    _geom.dihedral_angle_V_dV, signature="(3),(3),(3),(3)->(),(3),(3),(3),(3)"
)
