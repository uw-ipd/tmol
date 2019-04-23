from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "lj.compiled.cpu.cpp",
                "lj.compiled.cuda.cu",
                "lk_isotropic.compiled.cpu.cpp",
                "lk_isotropic.compiled.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)


def lk_isotropic(*args, **kwargs):
    raise NotImplementedError()


def lk_isotropic_triu(*args, **kwargs):
    raise NotImplementedError()


def lj(*args, **kwargs):
    raise NotImplementedError()


def lj_triu(*args, **kwargs):
    raise NotImplementedError()
