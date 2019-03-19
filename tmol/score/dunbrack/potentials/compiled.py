from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            ["compiled.pybind.cpp", "compiled.cpu.cpp"],  # temp, "compiled.cuda.cu"]
        )
    ),
)


def dunbrack_energy(*args, **kwargs):
    return _compiled.dunbrack[(args[0][0].device.type, args[0][0].dtype)](
        *args, **kwargs
    )


def dunbrack_deriv(*args, **kwargs):
    return _compiled.dunbrack_deriv[(args[0][0].device.type, args[0][0].dtype)](
        *args, **kwargs
    )
