from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "dunbrack_pose_score.cpu.cpp",
        "dunbrack_pose_score.cuda.cu",
    ],
    "tmol_dunbrack",
)

dunbrack_pose_scores = _ops.dunbrack_pose_scores
dunbrack_rotamer_scores = _ops.dunbrack_rotamer_scores
