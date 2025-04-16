from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__, ["compiled_inverse_kin.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
)

functions = ["inverse_kin"]

loader = TorchOpLoader(__name__, sources, functions)


def inverse_kin(*args, **kwargs):
    return loader.inverse_kin[(args[0].device.type, args[0].dtype)](*args, **kwargs)
