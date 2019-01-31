from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(modulename(__name__), relpaths(__file__, ["lk_ball.pybind.cpp"]))

build_acc_waters = _compiled.build_acc_waters
build_don_water = _compiled.build_don_water
