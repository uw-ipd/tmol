import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(__file__, ["apsp_vestibule.ops.cpp", "apsp.cpu.cpp", "apsp.cuda.cu"])
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))
stacked_apsp = _ops.apsp_op
