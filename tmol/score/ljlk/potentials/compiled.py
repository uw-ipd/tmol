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
                "rotamer_pair_energy_lj.cpu.cpp",
                "rotamer_pair_energy_lj.cuda.cu",
                # "rotamer_pair_energy_lk.cpu.cpp",
                # "rotamer_pair_energy_lk.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

ljlk_pose_scores = _ops.ljlk_pose_scores
score_ljlk_inter_system_scores = _ops.score_ljlk_inter_system_scores
register_lj_lk_rotamer_pair_energy_eval = _ops.register_lj_lk_rotamer_pair_energy_eval
