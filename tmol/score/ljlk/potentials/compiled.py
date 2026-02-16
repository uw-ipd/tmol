import torch

from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
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
                    # "rotamer_pair_energy_lk.cpu.cpp",
                    # "rotamer_pair_energy_lk.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

    _ops = getattr(torch.ops, modulename(__name__))
else:
    _ops = torch.ops.tmol_ljlk

ljlk_pose_scores = _ops.ljlk_pose_scores
ljlk_rotamer_scores = _ops.ljlk_rotamer_scores
