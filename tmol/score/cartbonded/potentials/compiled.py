from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    [
        "compiled.ops.cpp",
        "cartbonded_pose_score.cpu.cpp",
        "cartbonded_pose_score.cuda.cu",
    ],
)

functions = ["cartbonded_pose_scores"]

loader = TorchOpLoader(__name__, sources, functions)

cartbonded_pose_scores = loader.cartbonded_pose_scores
