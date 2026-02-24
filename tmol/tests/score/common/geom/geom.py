import numpy

from tmol.tests.score.common.geom._ext import (
    cos_interior_angle_V,
    dihedral_angle_V,
    distance_V,
    interior_angle_V,
)
from tmol.tests.score.common.geom._ext import (
    cos_interior_angle_V_dV as _cos_interior_angle_V_dV,
)
from tmol.tests.score.common.geom._ext import (
    dihedral_angle_V_dV as _dihedral_angle_V_dV,
)
from tmol.tests.score.common.geom._ext import (
    distance_V_dV as _distance_V_dV,
)
from tmol.tests.score.common.geom._ext import (
    interior_angle_V_dV as _interior_angle_V_dV,
)

# Wrap pybind11 scalar functions with numpy.vectorize so they broadcast
# over arrays of inputs and provide the gufunc signature needed by
# VectorizedOp in the test autograd harness.

distance_V = numpy.vectorize(distance_V, signature="(3),(3)->()")
distance_V_dV = numpy.vectorize(_distance_V_dV, signature="(3),(3)->(),(3),(3)")

interior_angle_V = numpy.vectorize(interior_angle_V, signature="(3),(3)->()")
interior_angle_V_dV = numpy.vectorize(_interior_angle_V_dV, signature="(3),(3)->(),(3),(3)")

cos_interior_angle_V = numpy.vectorize(cos_interior_angle_V, signature="(3),(3)->()")
cos_interior_angle_V_dV = numpy.vectorize(_cos_interior_angle_V_dV, signature="(3),(3)->(),(3),(3)")

dihedral_angle_V = numpy.vectorize(dihedral_angle_V, signature="(3),(3),(3),(3)->()")
dihedral_angle_V_dV = numpy.vectorize(_dihedral_angle_V_dV, signature="(3),(3),(3),(3)->(),(3),(3),(3),(3)")
