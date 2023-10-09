import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "backbone_torsion_pose_score.cpu.cpp",
                "backbone_torsion_pose_score.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

backbone_torsion_pose_score = _ops.backbone_torsion_pose_score
