import torch
import os.path
import tmol.utility.cpp_extension

cpu = tmol.utility.cpp_extension.load(
    (__name__ + ".cpu").replace(".", "_"),
    [os.path.join(os.path.dirname(__file__), f) for f in ("cpp_potential.cpu.cpp",)],
)

if torch.cuda.is_available():
    cuda = tmol.utility.cpp_extension.load(
        (__name__ + ".cuda").replace(".", "_"),
        [
            os.path.join(os.path.dirname(__file__), f)
            for f in ("cpp_potential.cuda.cpp", "cpp_potential.cuda.cu")
        ],
    )
else:
    cuda = None

potentials = {
    "blocked": {
        "cpu": cpu.lj_intra,
        # "cuda" : cuda.lj_intra,
    },
    "naive": {
        "cpu": cpu.lj_intra_naive,
        # "cuda" : cuda.lj_intra,
    },
}

POTENTIAL_SET = "blocked"


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
    kernel = potentials[POTENTIAL_SET][coords.device.type]

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
