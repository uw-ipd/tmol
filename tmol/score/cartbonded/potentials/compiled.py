from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)


def cartbonded_length(*args, **kwargs):
    return _compiled.cartbonded_length[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )


def cartbonded_angle(*args, **kwargs):
    return _compiled.cartbonded_angle[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )


def cartbonded_torsion(*args, **kwargs):
    return _compiled.cartbonded_torsion[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )


def cartbonded_hxltorsion(*args, **kwargs):
    return _compiled.cartbonded_hxltorsion[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )
