from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "../../common/whole_pose_scoring.cuda.cu",
        "hbond_pose_score.cpu.cpp",
        "hbond_pose_score.cuda.cu",
        "gen_hbond_bases.cpu.cpp",
        "gen_hbond_bases.cuda.cu",
    ],
    "tmol_hbond",
)

hbond_pose_scores = _ops.hbond_pose_scores
hbond_rotamer_scores = _ops.hbond_rotamer_scores
hbond_rotamer_scores_shared = _ops.hbond_rotamer_scores_shared
gen_hbond_bases = _ops.gen_hbond_bases
