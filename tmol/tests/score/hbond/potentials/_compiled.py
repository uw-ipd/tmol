import numpy

from tmol._load_ext import load_module

_compiled = load_module(
    __name__,
    __file__,
    ["compiled.pybind.cpp"],
    "tmol.tests.score.hbond.potentials._ext",
)

# hbond_score_V_dV takes struct arguments — export raw (used with kwargs).
hbond_score_V_dV = _compiled.hbond_score_V_dV

# The remaining functions take simple array/scalar args and are used
# through VectorizedOp in gradcheck tests, so wrap with numpy.vectorize.
AH_dist_V_dV = numpy.vectorize(
    _compiled.AH_dist_V_dV, signature="(3),(3),(n)->(),(3),(3)"
)
AHD_angle_V_dV = numpy.vectorize(
    _compiled.AHD_angle_V_dV, signature="(3),(3),(3),(n)->(),(3),(3),(3)"
)
BAH_angle_V_dV = numpy.vectorize(
    _compiled.BAH_angle_V_dV,
    signature="(3),(3),(3),(3),(),(n),()->(),(3),(3),(3),(3)",
)
sp2chi_energy_V_dV = numpy.vectorize(
    _compiled.sp2chi_energy_V_dV, signature="(),(),(),(),()->(),(),()"
)
