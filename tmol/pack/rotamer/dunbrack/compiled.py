import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


load(
    modulename(__name__),
    cuda_if_available(
        relpaths(__file__, ["compiled.ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"])
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

dun_sample_chi = _ops.dun_sample_chi
