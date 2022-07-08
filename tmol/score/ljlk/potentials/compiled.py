import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "lj.compiled.cpu.cpp",
                "lj.compiled.cuda.cu",
                "lk_isotropic.compiled.cpu.cpp",
                "lk_isotropic.compiled.cuda.cu",
                "ljlk_pose_score.cpu.cpp",
                "ljlk_pose_score.cuda.cu",
                # "rotamer_pair_energy_lj.cpu.cpp",
                # "rotamer_pair_energy_lj.cuda.cu",
                # "rotamer_pair_energy_lk.cpu.cpp",
                # "rotamer_pair_energy_lk.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

score_ljlk_lj = _ops.score_ljlk_lj
score_ljlk_lj_triu = _ops.score_ljlk_lj_triu
score_ljlk_lk_isotropic = _ops.score_ljlk_lk_isotropic
score_ljlk_lk_isotropic_triu = _ops.score_ljlk_lk_isotropic_triu
ljlk_pose_scores = _ops.ljlk_pose_scores
# temp score_ljlk_inter_system_scores = _ops.score_ljlk_inter_system_scores
# temp register_lj_lk_rotamer_pair_energy_eval = _ops.register_lj_lk_rotamer_pair_energy_eval
