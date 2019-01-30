import numpy
from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(
    modulename(__name__),
    relpaths(__file__, ["compiled.pybind.cpp", "compiled.cpu.cpp"]),
)


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
