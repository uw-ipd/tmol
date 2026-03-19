from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "ljlk_pose_score.cpu.cpp",
        "ljlk_pose_score.cuda.cu",
        # "rotamer_pair_energy_lk.cpu.cpp",
        # "rotamer_pair_energy_lk.cuda.cu",
    ],
    "tmol_ljlk",
)

ljlk_pose_scores = _ops.ljlk_pose_scores
ljlk_rotamer_scores = _ops.ljlk_rotamer_scores
