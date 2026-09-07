from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "ljlk_pose_score.cpu.cpp",
        "ljlk_pose_score.cuda.cu",
        "ljlk_elec_pose_score.cpu.cpp",
        "ljlk_elec_pose_score.cuda.cu",
        # "rotamer_pair_energy_lk.cpu.cpp",
        # "rotamer_pair_energy_lk.cuda.cu",
    ],
    "tmol_ljlk",
)

ljlk_pose_scores = _ops.ljlk_pose_scores
ljlk_elec_pose_scores = _ops.ljlk_elec_pose_scores
ljlk_elec_weighted_pose_scores = _ops.ljlk_elec_weighted_pose_scores
weighted_fused_score_sum = _ops.weighted_fused_score_sum
ljlk_rotamer_scores = _ops.ljlk_rotamer_scores
build_compact_block_neighbors = _ops.build_compact_block_neighbors
