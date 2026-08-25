from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "constraint_score.cpu.cpp",
        "constraint_score.cuda.cu",
    ],
    "tmol_constraint",
)

get_torsion_angle = _ops.get_torsion_angle
