from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    ["compiled.ops.cpp", "constraint_score.cpu.cpp", "constraint_score.cuda.cu"],
)

functions = ["get_torsion_angle"]

loader = TorchOpLoader(__name__, sources, functions)

get_torsion_angle = loader.get_torsion_angle
