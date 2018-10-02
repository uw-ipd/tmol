import os.path
import tmol.utility.cpp_extension

cpu = tmol.utility.cpp_extension.load(
    (__name__ + ".cpu").replace(".", "_"),
    [os.path.join(os.path.dirname(__file__), f) for f in ("cpp_potential.cpu.cpp",)],
)

cuda = tmol.utility.cpp_extension.load(
    (__name__ + ".cuda").replace(".", "_"),
    [
        os.path.join(os.path.dirname(__file__), f)
        for f in ("cpp_potential.cuda.cpp", "cpp_potential.cuda.cu")
    ],
)


def lj_intra(
    coords,
    atom_types,
    bonded_path_length,
    # Pair score parameters
    lj_sigma,
    lj_switch_slope,
    lj_switch_intercept,
    lj_coeff_sigma12,
    lj_coeff_sigma6,
    lj_spline_y0,
    lj_spline_dy0,
    # Global score parameters
    lj_switch_dis2sigma,
    spline_start,
    max_dis,
):
    if coords.device.type == "cpu":
        kernel = cpu.lj_intra
    else:
        kernel = cuda.lj_intra

    return kernel(
        coords,
        atom_types,
        bonded_path_length,
        # Pair score parameters
        lj_sigma,
        lj_switch_slope,
        lj_switch_intercept,
        lj_coeff_sigma12,
        lj_coeff_sigma6,
        lj_spline_y0,
        lj_spline_dy0,
        # Global score parameters
        lj_switch_dis2sigma,
        spline_start,
        max_dis,
    )
