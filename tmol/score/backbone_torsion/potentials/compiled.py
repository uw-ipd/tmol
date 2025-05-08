from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    [
        "compiled.ops.cpp",
        "backbone_torsion_pose_score.cpu.cpp",
        "backbone_torsion_pose_score.cuda.cu",
    ],
)

functions = ["backbone_torsion_pose_score"]

loader = TorchOpLoader(__name__, sources, functions)

backbone_torsion_pose_score = loader.backbone_torsion_pose_score
