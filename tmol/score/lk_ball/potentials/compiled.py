import torch
from tmol.utility.cpp_extension import load, modulename, relpaths, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "lk_ball_pose_score.cpu.cpp",
                "lk_ball_pose_score.cuda.cu",
                "rotamer_pair_energy_lkball.cpu.cpp",
                "rotamer_pair_energy_lkball.cuda.cu",
                "gen_pose_waters.cpu.cpp",
                "gen_pose_waters.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))
gen_pose_waters = _ops.gen_pose_waters
pose_score_lk_ball = _ops.lk_ball_pose_score
