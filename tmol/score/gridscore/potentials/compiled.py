from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "grid_interp_pose_score.cpu.cpp",
        "grid_interp_pose_score.cuda.cu",
    ],
    "tmol_gridscore",
)

build_grid = _ops.build_grid
evaluate_pairs = _ops.evaluate_pairs
interpolate = _ops.interpolate
