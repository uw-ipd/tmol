from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    [
        "compiled.ops.cpp",
        "disulfide_pose_score.cpu.cpp",
        "disulfide_pose_score.cuda.cu",
    ],
)

functions = ["disulfide_pose_scores"]

loader = TorchOpLoader(__name__, sources, functions)

disulfide_pose_scores = loader.disulfide_pose_scores
