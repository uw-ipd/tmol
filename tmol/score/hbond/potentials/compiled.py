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
                    "hbond_pose_score.cpu.cpp",
                    "hbond_pose_score.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

# Ops registered under TORCH_LIBRARY(tmol_hbond, ...) in C++
_ops = torch.ops.tmol_hbond

hbond_pose_scores = _ops.hbond_pose_scores
hbond_rotamer_scores = _ops.hbond_rotamer_scores
