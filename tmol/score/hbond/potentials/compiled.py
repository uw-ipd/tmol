import numpy
from pathlib import Path
from tmol.utility.cpp_extension import load
from enum import IntEnum

_compiled_sources = [str(Path(__file__).parent / s) for s in ("_compiled.cpp",)]
_compiled = load("_compiled", _compiled_sources)

hbond_score = _compiled.hbond_score
AH_dist_V_dV = numpy.vectorize(
    _compiled.AH_dist_V_dV, signature="(3),(3),(11),(2),(2)->(),(3),(3)"
)
AHD_angle_V_dV = numpy.vectorize(
    _compiled.AHD_angle_V_dV, signature="(3),(3),(3),(11),(2),(2)->(),(3),(3),(3)"
)
BAH_angle_V_dV = numpy.vectorize(
    _compiled.BAH_angle_V_dV,
    signature="(3),(3),(3),(3),(),(11),(2),(2),()->(),(3),(3),(3),(3)",
)


class AcceptorType(IntEnum):
    sp2 = 0
    sp3 = 1
    ring = 2
