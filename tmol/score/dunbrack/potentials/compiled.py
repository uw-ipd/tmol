from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    ["compiled.ops.cpp", "dunbrack_pose_score.cpu.cpp", "dunbrack_pose_score.cuda.cu"],
)

functions = ["dunbrack_pose_scores"]

loader = TorchOpLoader(__name__, sources, functions)

dunbrack_pose_scores = loader.dunbrack_pose_scores
