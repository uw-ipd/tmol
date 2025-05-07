from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__, ["compiled.ops.cpp", "elec_pose_score.cpu.cpp", "elec_pose_score.cuda.cu"]
)

functions = ["elec_pose_scores"]

loader = TorchOpLoader(__name__, sources, functions)

elec_pose_scores = loader.elec_pose_scores
