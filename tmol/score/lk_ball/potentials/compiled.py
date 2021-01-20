from tmol.utility.cpp_extension import load, modulename, relpaths, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "compiled.cpu.cpp",
                "compiled.cuda.cu",
                "rotamer_pair_energy_lkball.cpu.cpp",
                "rotamer_pair_energy_lkball.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)
