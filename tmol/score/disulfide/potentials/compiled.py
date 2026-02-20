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
                    "disulfide_pose_score.cpu.cpp",
                    "disulfide_pose_score.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

# Ops registered under TORCH_LIBRARY(tmol_disulfide, ...) in C++
_ops = torch.ops.tmol_disulfide

disulfide_pose_scores = _ops.disulfide_pose_scores
disulfide_rotamer_scores = _ops.disulfide_rotamer_scores
