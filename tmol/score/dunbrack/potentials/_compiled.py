from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "../../common/whole_pose_scoring.cuda.cu",
        "dunbrack_pose_score.cpu.cpp",
        "dunbrack_pose_score.cuda.cu",
    ],
    "tmol_dunbrack",
)

_dunbrack_pose_scores = _ops.dunbrack_pose_scores
_dunbrack_rotamer_scores = _ops.dunbrack_rotamer_scores
