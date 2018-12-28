import numpy
from pathlib import Path
from tmol.utility.cpp_extension import load

_compiled_sources = [str(Path(__file__).parent / s) for s in ("_compiled.cpp",)]
_compiled = load("_compiled", _compiled_sources)

hbond_score = _compiled.hbond_score
AH_dist_V_dV = numpy.vectorize(
    _compiled.AH_dist_V_dV, signature="(3),(3),(11),(2),(2)->(),(3),(3)"
)
AHD_angle_V_dV = numpy.vectorize(
    _compiled.AHD_angle_V_dV, signature="(3),(3),(3),(11),(2),(2)->(),(3),(3),(3)"
)

AcceptorType = _compiled.AcceptorType
