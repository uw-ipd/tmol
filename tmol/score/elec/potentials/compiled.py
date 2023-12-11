import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "elec_pose_score.cpu.cpp",
                "elec_pose_score.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

elec_pose_scores = _ops.elec_pose_scores
