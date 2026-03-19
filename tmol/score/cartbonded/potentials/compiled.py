from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "cartbonded_pose_score.cpu.cpp",
        "cartbonded_pose_score.cuda.cu",
    ],
    "tmol_cartbonded",
)

cartbonded_pose_scores = _ops.cartbonded_pose_scores
cartbonded_rotamer_scores = _ops.cartbonded_rotamer_scores
