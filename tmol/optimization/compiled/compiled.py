from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    ["lbfgs.ops.cpp", "lbfgs.cpu.cpp", "lbfgs.cuda.cu"],
    "tmol_optimization",
)

lbfgs_two_loop = _ops.lbfgs_two_loop
