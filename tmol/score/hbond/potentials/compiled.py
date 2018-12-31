import numpy
from pathlib import Path
from tmol.utility.cpp_extension import load
from enum import IntEnum

_compiled_sources = [str(Path(__file__).parent / s) for s in ("_compiled.cpp",)]
_compiled = load("_compiled", _compiled_sources)


hbond_score_V_dV = numpy.vectorize(
    _compiled.hbond_score_V_dV,
    signature="(3),(3),(3),(3),(3),"  # D,H,A,B,B0
    "(),(),(),"  # acceptor_type, accwt, donwt
    "(11),(2),(2),"  # AHdist
    "(11),(2),(2),"  # cosBAH
    "(11),(2),(2),"  # cosAHD
    "(),(),(),()"  # params
    "->"
    "(),"  # E
    "(3),(3),(3),(3),(3)",  # dE_d[D,H,A,B,B0]
)

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

sp2chi_energy_V_dV = numpy.vectorize(
    _compiled.sp2chi_energy_V_dV, signature="(),(),(),(),()->(),(),()"
)

hbond_pair_score = _compiled.hbond_pair_score
