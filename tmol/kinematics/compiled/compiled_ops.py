from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_ops = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(__file__, ["compiled_ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"])
    ),
    is_python_module=False,
)
