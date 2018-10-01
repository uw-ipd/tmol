import os.path
import tmol.utility.cpp_extension

cpu = tmol.utility.cpp_extension.load(
    (__name__ + ".cpu").replace(".", "_"),
    [os.path.dirname(__file__) + "/cpp_potential.cpu.cpp"],
)
