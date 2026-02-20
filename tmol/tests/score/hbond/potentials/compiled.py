import numpy

from tmol.tests.score.hbond.potentials._ext import (
    hbond_score_V_dV as _hbond_score_V_dV,
    AH_dist_V_dV as _AH_dist_V_dV,
    AHD_angle_V_dV as _AHD_angle_V_dV,
    BAH_angle_V_dV as _BAH_angle_V_dV,
    sp2chi_energy_V_dV as _sp2chi_energy_V_dV,
)

hbond_score_V_dV = numpy.vectorize(
    _hbond_score_V_dV,
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
    _AH_dist_V_dV, signature="(3),(3),(11),(2),(2)->(),(3),(3)"
)
AHD_angle_V_dV = numpy.vectorize(
    _AHD_angle_V_dV, signature="(3),(3),(3),(11),(2),(2)->(),(3),(3),(3)"
)
BAH_angle_V_dV = numpy.vectorize(
    _BAH_angle_V_dV,
    signature="(3),(3),(3),(3),(),(11),(2),(2),()->(),(3),(3),(3),(3)",
)

sp2chi_energy_V_dV = numpy.vectorize(
    _sp2chi_energy_V_dV, signature="(),(),(),(),()->(),(),()"
)
