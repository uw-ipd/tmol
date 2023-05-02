import torch
from tmol.utility.cpp_extension import load, modulename, relpaths, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "gen_pose_hydrogens.cpu.cpp",
                "gen_pose_hydrogens.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))
gen_pose_hydrogens = _ops.gen_pose_hydrogens
