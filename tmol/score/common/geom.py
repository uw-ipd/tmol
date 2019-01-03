import numpy
from pathlib import Path
from tmol.utility.cpp_extension import load

_geom = load("_geom", [str(Path(__file__).parent / s) for s in ("_geom.cpp",)])

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
