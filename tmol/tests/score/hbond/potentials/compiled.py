import numpy

from tmol.tests.score.hbond.potentials._ext import (
    AH_dist_V_dV as _AH_dist_V_dV,
)
from tmol.tests.score.hbond.potentials._ext import (
    AHD_angle_V_dV as _AHD_angle_V_dV,
)
from tmol.tests.score.hbond.potentials._ext import (
    BAH_angle_V_dV as _BAH_angle_V_dV,
)
from tmol.tests.score.hbond.potentials._ext import (
    hbond_score_V_dV as _hbond_score_V_dV,
)
from tmol.tests.score.hbond.potentials._ext import (
    sp2chi_energy_V_dV as _sp2chi_energy_V_dV,
)

# hbond_score_V_dV takes struct arguments — export raw (used with kwargs).
hbond_score_V_dV = _hbond_score_V_dV

# The remaining functions take simple array/scalar args and are used
# through VectorizedOp in gradcheck tests, so wrap with numpy.vectorize.
AH_dist_V_dV = numpy.vectorize(_AH_dist_V_dV, signature="(3),(3),(n)->(),(3),(3)")
AHD_angle_V_dV = numpy.vectorize(_AHD_angle_V_dV, signature="(3),(3),(3),(n)->(),(3),(3),(3)")
BAH_angle_V_dV = numpy.vectorize(
    _BAH_angle_V_dV,
    signature="(3),(3),(3),(3),(),(n),()->(),(3),(3),(3),(3)",
)
sp2chi_energy_V_dV = numpy.vectorize(_sp2chi_energy_V_dV, signature="(),(),(),(),()->(),(),()")
