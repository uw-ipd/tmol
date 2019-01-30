import numpy
from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(modulename(__name__), relpaths(__file__, ["compiled.pybind.cpp"]))

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
