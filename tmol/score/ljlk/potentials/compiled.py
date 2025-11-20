import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "ljlk_pose_score.cpu.cpp",
                "ljlk_pose_score.cuda.cu",
                "ljlk_fusion_module.cpu.cpp",
                "ljlk_fusion_module.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

ljlk_pose_scores = _ops.ljlk_pose_scores
ljlk_rotamer_scores = _ops.ljlk_rotamer_scores

create_ljlk_fusion_module = _ops.create_ljlk_fusion_module
free_fusion_module = _ops.free_fusion_module
test_run_forward = _ops.test_run_forward
