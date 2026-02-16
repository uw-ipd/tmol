import torch

from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
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
                    # "rotamer_pair_energy_lkball.cpu.cpp",
                    # "rotamer_pair_energy_lkball.cuda.cu",
                    "gen_pose_waters.cpu.cpp",
                    "gen_pose_waters.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

    _ops = getattr(torch.ops, modulename(__name__))
else:
    _ops = torch.ops.tmol_lk_ball

gen_pose_waters = _ops.gen_pose_waters
lk_ball_pose_score = _ops.lk_ball_pose_score
lk_ball_rotamer_score = _ops.lk_ball_rotamer_score
