import numpy
from tmol.utility.cpp_extension import load, relpaths, modulename

_geom = load(modulename(__name__), relpaths(__file__, "geom.pybind.cpp"))

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
