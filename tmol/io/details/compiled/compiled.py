from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    [
        "compiled.ops.cpp",
        "gen_pose_leaf_atoms.cpu.cpp",
        "gen_pose_leaf_atoms.cuda.cu",
        "resolve_his_taut.cpu.cpp",
        "resolve_his_taut.cuda.cu",
    ],
)

functions = ["gen_pose_leaf_atoms", "resolve_his_taut"]

loader = TorchOpLoader(__name__, sources, functions)

gen_pose_leaf_atoms = loader.gen_pose_leaf_atoms
resolve_his_taut = loader.resolve_his_taut
