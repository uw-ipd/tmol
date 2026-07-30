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
                    "genbonded_pose_score.cpu.cpp",
                    "genbonded_pose_score.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

    _ops = torch.ops.tmol_genbonded
else:
    _ops = torch.ops.tmol_genbonded

genbonded_pose_scores = _ops.genbonded_pose_scores
genbonded_rotamer_scores = _ops.genbonded_rotamer_scores
