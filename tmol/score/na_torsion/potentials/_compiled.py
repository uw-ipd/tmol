from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    ["compiled.ops.cpp", "na_torsion_pose_score.cuda.cu"],
    "tmol_na_torsion",
)

_na_torsion_pose_score = _ops.na_torsion_pose_score
