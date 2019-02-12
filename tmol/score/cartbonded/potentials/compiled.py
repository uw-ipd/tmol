from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)

cartbonded_length = _compiled.cartbonded_length
cartbonded_angle = _compiled.cartbonded_angle
cartbonded_torsion = _compiled.cartbonded_torsion
cartbonded_hxltorsion = _compiled.cartbonded_hxltorsion
