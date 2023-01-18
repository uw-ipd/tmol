import torch
from tmol.utility.cpp_extension import load, modulename, relpaths, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "compiled.cpu.cpp",
                "compiled.cuda.cu",
                "lk_ball_pose_score.cpu.cpp",
                "lk_ball_pose_score.cuda.cu",
                "rotamer_pair_energy_lkball.cpu.cpp",
                "rotamer_pair_energy_lkball.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))
score_lkball = _ops.score_lkball
watergen_lkball = _ops.watergen_lkball
