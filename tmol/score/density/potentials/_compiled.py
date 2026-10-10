from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "density_score.cpu.cpp",
        "density_score.cuda.cu",
    ],
    "tmol_density",
)

density_score = _ops.density_score
