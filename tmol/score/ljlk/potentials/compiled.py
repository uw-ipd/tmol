import numpy
from pathlib import Path
from tmol.utility.cpp_extension import load

_compiled_sources = [str(Path(__file__).parent / s) for s in ("_compiled.cpp",)]
_compiled = load(__name__.replace(".", "_"), _compiled_sources)

lj_score_V = numpy.vectorize(
    _compiled.lj_score_V,
    signature="(),()," "(),(),(),(),(),(),"  # dist, bonded_path_length
    # i : lj_radius, lj_wdepth, is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),"
    # j : lj_radius, lj_wdepth, is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),()" "->" "()",  # params  # E
)

lj_score_V_dV = numpy.vectorize(
    _compiled.lj_score_V_dV,
    signature=""
    # dist, bonded_path_length
    "(),(),"
    # i : lj_radius, lj_wdepth, is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),"
    # j : lj_radius, lj_wdepth, is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),"
    # params
    "(),(),()"
    # E, dE_dD
    "->(),()",
)

vdw_V_dV = numpy.vectorize(_compiled.vdw_V_dV, signature="(),(),()->(),()")

vdw_V = numpy.vectorize(_compiled.vdw_V, signature="(),(),()->()")
