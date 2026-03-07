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
                    "backbone_torsion_pose_score.cpu.cpp",
                    "backbone_torsion_pose_score.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

    _ops = torch.ops.tmol_bb_torsion
else:
    _ops = torch.ops.tmol_bb_torsion

backbone_torsion_pose_score = _ops.backbone_torsion_pose_score
backbone_torsion_rotamer_score = _ops.backbone_torsion_rotamer_score
