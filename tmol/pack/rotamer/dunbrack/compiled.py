from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__, ["compiled.ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
)

functions = ["dun_sample_chi"]

loader = TorchOpLoader(__name__, sources, functions)

dun_sample_chi = loader.dun_sample_chi
