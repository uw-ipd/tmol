from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)


def sample_chi_for_rotamers(*args, **kwargs):
    return _compiled.sample_chi_for_rotamers[
        (args[0][0].device.type, args[0][0].dtype)
    ](*args, **kwargs)
