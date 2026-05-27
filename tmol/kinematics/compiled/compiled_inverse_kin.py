from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

    _mod = load(
        modulename(__name__),
        cuda_if_available(
            relpaths(
                __file__,
                ["compiled_inverse_kin.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"],
            )
        ),
        is_python_module=True,
    )
    _inverse_kin_dispatch = _mod.inverse_kin
else:
    from tmol.kinematics.compiled._compiled_inverse_kin import (
        inverse_kin as _inverse_kin_dispatch,
    )


def inverse_kin(*args, **kwargs):
    return _inverse_kin_dispatch[(args[0].device.type, args[0].dtype)](*args, **kwargs)
