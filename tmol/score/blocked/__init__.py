import os
import tmol.utility.cpp_extension

cuda = tmol.utility.cpp_extension.load(
    (__name__ + ".cuda").replace(".", "_"),
    [os.path.join(os.path.dirname(__file__), f) for f in ("cuda.cpp", "cuda.cu")],
)

cpu = tmol.utility.cpp_extension.load(
    (__name__ + ".cpu").replace(".", "_"),
    [os.path.join(os.path.dirname(__file__), f) for f in ("cpu.cpp",)],
)
