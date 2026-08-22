import torch

from .compiled_ops import forward_kin_op, inverse_kin as _inverse_kin_dispatch

__all__ = ["forward_kin_op", "inverse_kin"]


def inverse_kin(*args, **kwargs):
    # inverse_kin has no compiled MPS kernel; run on CPU and move the result back.
    any_mps = any(isinstance(a, torch.Tensor) and a.device.type == "mps" for a in args)
    if not any_mps:
        return _inverse_kin_dispatch(*args, **kwargs)

    cpu_args = tuple(a.to("cpu") if isinstance(a, torch.Tensor) else a for a in args)
    cpu_kwargs = {
        k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
    }
    dtype = cpu_args[0].dtype
    result = _inverse_kin_dispatch(*cpu_args, **cpu_kwargs)
    # float64 cannot live on MPS — return on CPU; float32 can be moved back.
    if dtype == torch.float64:
        return result
    mps_device = next(
        a.device for a in args if isinstance(a, torch.Tensor) and a.device.type == "mps"
    )
    return result.to(mps_device)
