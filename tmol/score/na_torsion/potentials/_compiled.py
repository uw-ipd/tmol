from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    ["compiled.ops.cpp", "na_torsion_pose_score.cuda.cu"],
    "tmol_na_torsion",
)

na_torsion_pose_score = _ops.na_torsion_pose_score
