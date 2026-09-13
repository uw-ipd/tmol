from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "../../common/whole_pose_scoring.cuda.cu",
        "disulfide_pose_score.cpu.cpp",
        "disulfide_pose_score.cuda.cu",
    ],
    "tmol_disulfide",
)

disulfide_pose_scores = _ops.disulfide_pose_scores
disulfide_rotamer_scores = _ops.disulfide_rotamer_scores
