from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp"]  # "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)


def create_tensor_collection1(*args, **kwargs):
    return _compiled.create_tensor_collection1[
        (args[0][0].device.type, args[0][0].dtype)
    ](*args, **kwargs)


def create_tensor_collection2(*args, **kwargs):
    return _compiled.create_tensor_collection2[
        (args[0][0].device.type, args[0][0].dtype)
    ](*args, **kwargs)


def create_tensor_collection3(*args, **kwargs):
    return _compiled.create_tensor_collection3[
        (args[0][0].device.type, args[0][0].dtype)
    ](*args, **kwargs)


def create_tensor_collection4(*args, **kwargs):
    return _compiled.create_tensor_collection4[
        (args[0][0].device.type, args[0][0].dtype)
    ](*args, **kwargs)
