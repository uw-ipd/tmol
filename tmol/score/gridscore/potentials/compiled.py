from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "grid_interp_pose_score.cpu.cpp",
        "grid_interp_pose_score.cuda.cu",
        "live_hbond.cpu.cpp",
        "live_hbond.cuda.cu",
        "live_lk_ball.cpu.cpp",
        "live_lk_ball.cuda.cu",
    ],
    "tmol_gridscore",
)

build_grid = _ops.build_grid
evaluate_pairs = _ops.evaluate_pairs
interpolate = _ops.interpolate
interpolate_with_derivatives = _ops.interpolate_with_derivatives
live_hbond = _ops.live_hbond
live_lk_ball = _ops.live_lk_ball
