import numpy
from pathlib import Path
from tmol.utility.cpp_extension import load

_compiled_sources = [str(Path(__file__).parent / s) for s in ("_compiled.cpp",)]
_compiled = load(__name__.replace(".", "_"), _compiled_sources)

lj_sigma = _compiled.lj_sigma

lj_score_V = numpy.vectorize(
    _compiled.lj_score_V,
    signature=""
    # dist, bonded_path_length
    "(),(),"
    # i : lj_radius, lj_wdepth, is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),"
    # j : lj_radius, lj_wdepth, is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),"
    # params
    "(),(),()"
    # E
    "->()",
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

lk_isotropic_score_V_dV = numpy.vectorize(
    _compiled.lk_isotropic_score_V_dV,
    signature=""
    # dist, bonded_path_length
    "(),(),"
    # i :
    # lj_radius, lk_dgfree, lk_lambda, lk_volume,
    # is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),(),(),"
    # j :
    # lj_radius, lk_dgfree, lk_lambda, lk_volume,
    # is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),(),(),"
    # params
    "(),(),()"
    # E dE_dD
    "->(),()",
)

lk_isotropic_score_V = numpy.vectorize(
    _compiled.lk_isotropic_score_V,
    signature=""
    # dist, bonded_path_length
    "(),(),"
    # i :
    # lj_radius, lk_dgfree, lk_lambda, lk_volume,
    # is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),(),(),"
    # j :
    # lj_radius, lk_dgfree, lk_lambda, lk_volume,
    # is_donor, is_hydroxyl, is_polarh, is_acceptor
    "(),(),(),(),(),(),(),(),"
    # params
    "(),(),()"
    # E
    "->()",
)

vdw_V_dV = numpy.vectorize(_compiled.vdw_V_dV, signature="(),(),()->(),()")
vdw_V = numpy.vectorize(_compiled.vdw_V, signature="(),(),()->()")

f_desolv_V_dV = numpy.vectorize(
    _compiled.f_desolv_V_dV, signature="(),(),(),(),()->(),()"
)
f_desolv_V = numpy.vectorize(_compiled.f_desolv_V, signature="(),(),(),(),()->()")

lk_isotropic = _compiled.lk_isotropic
lk_isotropic_triu = _compiled.lk_isotropic_triu

lj = _compiled.lj
lj_triu = _compiled.lj_triu
