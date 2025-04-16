from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    [
        "compiled.ops.cpp",
        "ljlk_pose_score.cpu.cpp",
        "ljlk_pose_score.cuda.cu",
        "rotamer_pair_energy_lj.cpu.cpp",
        "rotamer_pair_energy_lj.cuda.cu",
    ],
)

functions = [
    "ljlk_pose_scores",
    "score_ljlk_inter_system_scores",
    "register_lj_lk_rotamer_pair_energy_eval",
]

loader = TorchOpLoader(__name__, sources, functions)

ljlk_pose_scores = loader.ljlk_pose_scores
score_ljlk_inter_system_scores = loader.score_ljlk_inter_system_scores
register_lj_lk_rotamer_pair_energy_eval = loader.register_lj_lk_rotamer_pair_energy_eval
