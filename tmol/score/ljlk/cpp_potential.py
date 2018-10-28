import torch
import os.path
import tmol.utility.cpp_extension
import typing

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

POTENTIAL_SET = "blocked"
BLOCK_SIZE = 8


def _lj_intra_blocked_cpu(coords, **kwargs):
    block_pairs = cpu.block_interaction_lists(coords, kwargs["max_dis"], BLOCK_SIZE)
    block_scores = cpu.lj_intra_block(coords, block_pairs, BLOCK_SIZE, **kwargs)

    return (block_pairs.t(), block_scores)


def _lj_intra_blocked_cuda(coords, **kwargs):
    assert BLOCK_SIZE == 8
    rsize, block_pairs, block_scores = cuda.lj_intra_block(coords, **kwargs)

    return (block_pairs[:rsize].t(), block_scores[:rsize])


potentials = {"blocked": {"cpu": _lj_intra_blocked_cpu, "cuda": _lj_intra_blocked_cuda}}


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
) -> typing.Tuple[torch.tensor, torch.tensor]:
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
