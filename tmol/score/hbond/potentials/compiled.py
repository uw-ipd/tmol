from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    ["compiled.ops.cpp", "hbond_pose_score.cpu.cpp", "hbond_pose_score.cuda.cu"],
)

functions = ["hbond_pose_scores"]

loader = TorchOpLoader(__name__, sources, functions)

hbond_pose_scores = loader.hbond_pose_scores
