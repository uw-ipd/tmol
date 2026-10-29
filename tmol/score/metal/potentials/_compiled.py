from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "../../common/whole_pose_scoring.cuda.cu",
        "metal_coordination_pose_score.cpu.cpp",
        "metal_coordination_pose_score.cuda.cu",
    ],
    "tmol_metal",
)

_metal_coordination_pose_scores = _ops.metal_coordination_pose_scores
_metal_coordination_rotamer_scores = _ops.metal_coordination_rotamer_scores
