"""Fused state transitions of the pose-batched L-BFGS (see ``_batched_lbfgs``)."""

from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "lbfgs_compiled.ops.cpp",
        "lbfgs_compiled.cpu.cpp",
        "lbfgs_compiled.cuda.cu",
    ],
    "tmol_dock",
)

lbfgs_direction = _ops.lbfgs_direction
lbfgs_update = _ops.lbfgs_update
