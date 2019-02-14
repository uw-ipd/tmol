from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)


def elec(*args, **kwargs):
    return _compiled.elec[(args[0].device.type, args[0].dtype)](*args, **kwargs)


def elec_triu(*args, **kwargs):
    return _compiled.elec_triu[(args[0].device.type, args[0].dtype)](*args, **kwargs)
