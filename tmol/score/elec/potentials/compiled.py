import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "compiled.cpu.cpp",
                "compiled.cuda.cu",
                "elec_pose_score.cpu.cpp",
                "elec_pose_score.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

score_elec = _ops.score_elec
score_elec_triu = _ops.score_elec_triu
elec_pose_scores = _ops.elec_pose_scores
