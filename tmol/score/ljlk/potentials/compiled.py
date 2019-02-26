from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.pybind.cpp",
                "lj.compiled.cpu.cpp",
                "lj.compiled.cuda.cu",
                "lk_isotropic.compiled.cpu.cpp",
                "lk_isotropic.compiled.cuda.cu",
            ],
        )
    ),
)


def lk_isotropic(*args, **kwargs):
    return _compiled.lk_isotropic[(args[0].device.type, args[0].dtype)](*args, **kwargs)


def lk_isotropic_triu(*args, **kwargs):
    return _compiled.lk_isotropic_triu[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )


def lj(*args, **kwargs):
    return _compiled.lj[(args[0].device.type, args[0].dtype)](*args, **kwargs)


def lj_triu(*args, **kwargs):
    return _compiled.lj_triu[(args[0].device.type, args[0].dtype)](*args, **kwargs)
