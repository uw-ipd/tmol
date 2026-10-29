from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "grid_build.cpu.cpp",
        "grid_build.cuda.cu",
        "grid_interp_pose_score.cpu.cpp",
        "grid_interp_pose_score.cuda.cu",
        "grid_score.cpu.cpp",
        "grid_score.cuda.cu",
        "live_hbond.cpu.cpp",
        "live_hbond.cuda.cu",
        "live_lk_ball.cpu.cpp",
        "live_lk_ball.cuda.cu",
    ],
    "tmol_gridscore",
)

build_grid = _ops.build_grid
interpolate_with_derivatives = _ops.interpolate_with_derivatives
live_hbond = _ops.live_hbond
live_lk_ball = _ops.live_lk_ball
grid_score_atoms = _ops.grid_score_atoms
grid_score_waters = _ops.grid_score_waters
