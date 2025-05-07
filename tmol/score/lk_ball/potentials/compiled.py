from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    [
        "compiled.ops.cpp",
        "lk_ball_pose_score.cpu.cpp",
        "lk_ball_pose_score.cuda.cu",
        "rotamer_pair_energy_lkball.cpu.cpp",
        "rotamer_pair_energy_lkball.cuda.cu",
        "gen_pose_waters.cpu.cpp",
        "gen_pose_waters.cuda.cu",
    ],
)

functions = ["gen_pose_waters", "lk_ball_pose_score"]

loader = TorchOpLoader(__name__, sources, functions)

gen_pose_waters = loader.gen_pose_waters
pose_score_lk_ball = loader.lk_ball_pose_score
