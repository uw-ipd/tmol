import numpy
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


def _lj_intra_blocked(coords, **kwargs):
    # Output allocation is _far_ faster via numpy for empty allocations,
    # presumably performing a calloc-based allocation, rather than zeroing
    # for large allocs
    result = torch.from_numpy(numpy.zeros((coords.shape[0],) * 2, dtype="f4"))
    block_table = cpu.block_interaction_table(coords, kwargs["max_dis"])
    return cpu.lj_intra_block(coords, result, block_table, **kwargs)


potentials = {
    "blocked": {
        "cpu": _lj_intra_blocked
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
        types=atom_types,
        bonded_path_length=bonded_path_length,
        # Pair score parameters
        lj_sigma=lj_sigma,
        lj_switch_slope=lj_switch_slope,
        lj_switch_intercept=lj_switch_intercept,
        lj_coeff_sigma12=lj_coeff_sigma12,
        lj_coeff_sigma6=lj_coeff_sigma6,
        lj_spline_y0=lj_spline_y0,
        lj_spline_dy0=lj_spline_dy0,
        # Global score parameters
        lj_switch_dis2sigma=lj_switch_dis2sigma,
        spline_start=spline_start,
        max_dis=max_dis,
    )
