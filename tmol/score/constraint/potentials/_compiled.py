from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "constraint_score.cpu.cpp",
        "constraint_score.cuda.cu",
    ],
    "tmol_constraint",
)

_get_torsion_angle = _ops.get_torsion_angle
