from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__, ["compiled_ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
)

functions = ["forward_kin_op", "forward_only_op"]

loader = TorchOpLoader(__name__, sources, functions)

forward_kin_op = loader.forward_kin_op
forward_only_op = loader.forward_only_op
