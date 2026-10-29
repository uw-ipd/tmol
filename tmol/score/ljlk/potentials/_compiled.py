from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "../../common/whole_pose_scoring.cuda.cu",
        "ljlk_pose_score.cpu.cpp",
        "ljlk_pose_score.cuda.cu",
        "ljlk_elec_pose_score.cpu.cpp",
        "ljlk_elec_pose_score.cuda.cu",
        # "rotamer_pair_energy_lk.cpu.cpp",
        # "rotamer_pair_energy_lk.cuda.cu",
    ],
    "tmol_ljlk",
)

_ljlk_pose_scores = _ops.ljlk_pose_scores
_ljlk_elec_pose_scores = _ops.ljlk_elec_pose_scores
_ljlk_elec_weighted_pose_scores = _ops.ljlk_elec_weighted_pose_scores
_weighted_fused_score_sum = _ops.weighted_fused_score_sum
_ljlk_rotamer_scores = _ops.ljlk_rotamer_scores
_ljlk_elec_rotamer_dispatch = _ops.ljlk_elec_rotamer_dispatch
_ljlk_elec_weighted_rotamer_scores = _ops.ljlk_elec_weighted_rotamer_scores
_build_compact_block_neighbors = _ops.build_compact_block_neighbors
