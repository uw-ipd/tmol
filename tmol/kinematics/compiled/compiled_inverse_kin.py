from tmol._load_ext import load_module

_mod = load_module(
    __name__,
    __file__,
    ["compiled_inverse_kin.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"],
    "tmol.kinematics.compiled._compiled_inverse_kin",
)
_inverse_kin_dispatch = _mod.inverse_kin


def inverse_kin(*args, **kwargs):
    return _inverse_kin_dispatch[(args[0].device.type, args[0].dtype)](*args, **kwargs)
