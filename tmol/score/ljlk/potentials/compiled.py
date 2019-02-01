from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)

lk_isotropic = _compiled.lk_isotropic
lk_isotropic_triu = _compiled.lk_isotropic_triu

lj = _compiled.lj
lj_triu = _compiled.lj_triu
