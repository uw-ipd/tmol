import torch

from tmol._load_ext import ensure_compiled_or_jit

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "ljlk_pose_score.cpu.cpp",
                "ljlk_pose_score.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

    load(
        modulename(__name__),
        cuda_if_available(
            relpaths(
                __file__,
                [
                    "compiled.ops.cpp",
                    "ljlk_pose_score.cpu.cpp",
                    "ljlk_pose_score.cuda.cu",
                    # "rotamer_pair_energy_lk.cpu.cpp",
                    # "rotamer_pair_energy_lk.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

# Ops registered under TORCH_LIBRARY(tmol_ljlk, ...) in C++
_ops = torch.ops.tmol_ljlk

ljlk_pose_scores = _ops.ljlk_pose_scores
ljlk_rotamer_scores = _ops.ljlk_rotamer_scores
