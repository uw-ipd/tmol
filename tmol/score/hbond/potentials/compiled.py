from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(__file__, ["compiled.ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"])
    ),
    is_python_module=False,
)


def hbond_pair_score(*args, **kwargs):
    return _compiled.hbond_pair_score[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )
