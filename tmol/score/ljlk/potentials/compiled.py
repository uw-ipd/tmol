from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(
    modulename(__name__),
    relpaths(__file__, ["compiled.cuda.cu", "compiled.cpu.cpp", "compiled.pybind.cpp"]),
)

lk_isotropic = _compiled.lk_isotropic
lk_isotropic_triu = _compiled.lk_isotropic_triu

lj = _compiled.lj
lj_triu = _compiled.lj_triu
