import torch
from tmol._load_ext import load_module

_mod = load_module(
    __name__,
    __file__,
    ["compiled_inverse_kin.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"],
    "tmol.kinematics.compiled._compiled_inverse_kin",
)
_inverse_kin_dispatch = _mod.inverse_kin


def inverse_kin(*args, **kwargs):
    # If any tensor arg is on MPS, run on CPU (MPS has no compiled kernel).
    any_mps = any(isinstance(a, torch.Tensor) and a.device.type == "mps" for a in args)
    if any_mps:
        cpu_args = tuple(a.to("cpu") if isinstance(a, torch.Tensor) else a for a in args)
        cpu_kwargs = {
            k: v.to("cpu") if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        dtype = cpu_args[0].dtype
        result = _inverse_kin_dispatch[("cpu", dtype)](*cpu_args, **cpu_kwargs)
        # float64 cannot live on MPS — return on CPU; float32 can be moved back.
        if dtype == torch.float64:
            return result
        # Find the original MPS device from args
        mps_device = next(
            a.device for a in args if isinstance(a, torch.Tensor) and a.device.type == "mps"
        )
        return result.to(mps_device)
    device = args[0].device
    return _inverse_kin_dispatch[(device.type, args[0].dtype)](*args, **kwargs)
