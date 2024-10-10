import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "constraint_score.cpu.cpp",
                "constraint_score.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))
get_torsion_angle = _ops.get_torsion_angle
