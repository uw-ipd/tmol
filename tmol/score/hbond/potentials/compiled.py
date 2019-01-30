from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(
    modulename(__name__),
    relpaths(__file__, ["compiled.pybind.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]),
)

hbond_pair_score = _compiled.hbond_pair_score
