from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            ["compiled_inverse_kin.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"],
        )
    ),
    is_python_module=True,
)


def inverse_kin(*args, **kwargs):
    return _compiled.inverse_kin[(args[0].device.type, args[0].dtype)](*args, **kwargs)
