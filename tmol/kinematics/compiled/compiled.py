from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)


def forward_kin(*args, **kwargs):
    return _compiled.forward_kin[(args[0].device.type, args[0].dtype)](*args, **kwargs)


def dof_transforms(*args, **kwargs):
    return _compiled.dof_transforms[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )


def backward_kin(*args, **kwargs):
    return _compiled.backward_kin[(args[0].device.type, args[0].dtype)](*args, **kwargs)


def f1f2_to_deriv(*args, **kwargs):
    return _compiled.f1f2_to_deriv[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )


def segscan_f1f2s(*args, **kwargs):
    return _compiled.segscan_f1f2s[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )
