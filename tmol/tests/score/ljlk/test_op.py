import math

import torch
import tmol.database

from tmol.utility.reactive import reactive_attrs

from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph

import tmol.score.ljlk.numba_potential as numba_potential
import tmol.score.ljlk.torch_op as torch_op


@reactive_attrs
class DataGraph(
    BondedAtomScoreGraph,
    BlockedInteratomicDistanceGraph,
    CartesianAtomicCoordinateProvider,
):
    pass


def test_op_device(torch_device, ubq_system):
    """Op executes lj potential evaluation via configured params.

    Op is initialized over a fixed parameter resolver, and is affinitized to
    that device. Op pre-loads parameter tensors as array views for later reuse.
    Op accepts input coordinate locations and atom types (resolved as type
    numbers) and executes the forward pass, returning a triu pairwise
    potential value.
    """

    params: LJLKParamResolver = LJLKParamResolver.from_database(
        tmol.database.ParameterDatabase.get_default().scoring.ljlk, device=torch_device
    )

    dg = DataGraph.build_for(ubq_system, device=torch_device, requires_grad=False)
    raw_bpl = dg.bonded_path_length[0]
    raw_bpl[raw_bpl == math.inf] = 64
    bonded_path_length = torch.tensor(raw_bpl).to(dg.device, dtype=torch.uint8)
    coords = dg.coords[0]
    atom_types = params.type_idx(dg.atom_types[0])
    atom_types[dg.atom_types[0] == None] = -1  # noqa
    atom_types = torch.tensor(atom_types).to(device=torch_device)
    interblock_distance = dg.interblock_distance.min_dist[0]

    numba_op = torch_op.LJOp.from_params(params, jit_type="numba")
    assert numba_op.device == torch_device
    assert numba_op.jit_type == "numba"

    numba_pscore = numba_op.intra(coords, atom_types, bonded_path_length)
    numba_blocked_pscore = numba_op.intra(
        coords, atom_types, bonded_path_length, interblock_distance
    )

    assert numba_pscore.shape == (coords.shape[0], coords.shape[0])

    cpp_op = torch_op.LJOp.from_params(params, jit_type="cpp")
    assert cpp_op.device == torch_device
    assert cpp_op.jit_type == "cpp"

    cpp_pscore = cpp_op.intra(coords, atom_types, bonded_path_length)
    cpp_blocked_pscore = cpp_op.intra(
        coords, atom_types, bonded_path_length, interblock_distance
    )

    assert cpp_pscore.shape == (coords.shape[0], coords.shape[0])

    # old kernel impl
    kernel_result = numba_potential.lj_intra_kernel(
        coords,
        atom_types,
        bonded_path_length,
        lj_sigma=params.pair_params.lj_sigma,
        lj_switch_slope=params.pair_params.lj_switch_slope,
        lj_switch_intercept=params.pair_params.lj_switch_intercept,
        lj_coeff_sigma12=params.pair_params.lj_coeff_sigma12,
        lj_coeff_sigma6=params.pair_params.lj_coeff_sigma6,
        lj_spline_y0=params.pair_params.lj_spline_y0,
        lj_spline_dy0=params.pair_params.lj_spline_dy0,
        # Global params
        lj_switch_dis2sigma=params.global_params.lj_switch_dis2sigma,
        spline_start=params.global_params.spline_start,
        max_dis=params.global_params.max_dis,
    )

    torch.testing.assert_allclose(numba_pscore, kernel_result)
    torch.testing.assert_allclose(numba_blocked_pscore, kernel_result)

    torch.testing.assert_allclose(cpp_pscore, kernel_result)
    torch.testing.assert_allclose(cpp_blocked_pscore, kernel_result)
