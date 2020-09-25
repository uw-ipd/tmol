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
                "rotamer_pair_energy_lj.cpu.cpp",
                "rotamer_pair_energy_lj.cuda.cu",
                "rotamer_pair_energy_lk.cpu.cpp",
                "rotamer_pair_energy_lk.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)
