from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "gen_pose_leaf_atoms.cpu.cpp",
        "gen_pose_leaf_atoms.cuda.cu",
        "resolve_his_taut.cpu.cpp",
        "resolve_his_taut.cuda.cu",
    ],
    "tmol_io",
)

gen_pose_leaf_atoms = _ops.gen_pose_leaf_atoms
resolve_his_taut = _ops.resolve_his_taut
