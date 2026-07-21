from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "elec_pose_score.cpu.cpp",
        "elec_pose_score.cuda.cu",
    ],
    "tmol_elec",
)

elec_pose_scores = _ops.elec_pose_scores
elec_rotamer_scores = _ops.elec_rotamer_scores
