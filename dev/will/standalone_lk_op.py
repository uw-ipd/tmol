"""
builds a copy of the jk cuda kernel from my dev directory and compares to the original. just an exercise to start to unravel how the cuda extensions work
"""

import math

import torch

import tmol.database

from tmol.utility.reactive import reactive_attrs

from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph

import tmol.score.ljlk.torch_op as torch_op

from tmol.system.io import read_pdb
import tmol.tests.data.pdb


@reactive_attrs
class DataGraph(
        BondedAtomScoreGraph,
        BlockedInteratomicDistanceGraph,
        CartesianAtomicCoordinateProvider,
):
    pass


__cuda_kernel_copy = tmol.utility.cpp_extension.load(
    'will_lk_copy', ['./lk_copy.cuda.cpp', './lk_copy.cuda.cu'])


def cuda_kernel_copy(coords, **kwargs):
    rsize, block_pairs, block_scores = __cuda_kernel_copy.lj_intra_block(
        coords, **kwargs)
    return (block_pairs[:rsize].t(), block_scores[:rsize])


def main(torch_device, ubq_system):

    print('build params')
    params = LJLKParamResolver.from_database(
        tmol.database.ParameterDatabase.get_default().scoring.ljlk,
        device=torch_device)

    print('build DataGraph')
    dg = DataGraph.build_for(
        ubq_system, device=torch_device, requires_grad=False)
    raw_bpl = dg.bonded_path_length[0]
    raw_bpl[raw_bpl == math.inf] = 64
    bonded_path_length = torch.tensor(raw_bpl).to(dg.device, dtype=torch.uint8)
    coords = dg.coords[0]
    atom_types = params.type_idx(dg.atom_types[0])
    atom_types[dg.atom_types[0] == None] = -1  # noqa
    atom_types = torch.tensor(atom_types).to(device=torch_device)

    print('build LJOp')
    cpp_op = torch_op.LJOp.from_params(params)

    assert cpp_op.device == torch_device

    print('call cpp_op.intra')
    block_inds_op, block_scores_op = cpp_op.intra(coords, atom_types,
                                                  bonded_path_length)

    print('call kernel')
    kernel_result = cuda_kernel_copy(
        coords,
        types=atom_types,
        bonded_path_length=bonded_path_length,
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

    print('collate scores and compare')
    block_inds_kr, block_scores_kr = kernel_result
    assert block_inds_op.max() == block_inds_kr.max()

    nblocks = block_inds_op.max() + 1
    ordcpu = (block_inds_op[0] * nblocks + block_inds_op[1]).sort()[1]
    ordkrn = (block_inds_kr[0] * nblocks + block_inds_kr[1]).sort()[1]

    assert (block_inds_op[:, ordcpu] == block_inds_kr[:, ordkrn]).all()
    torch.testing.assert_allclose(block_scores_op[ordcpu],
                                  block_scores_kr[ordkrn])
    print('kernel and cpp_op agree')

    block_all_zero = block_scores_op.sum(2).sum(1) == 0
    print('fraction nonzero blocks:',
          1.0 - float(block_all_zero.sum()) / len(block_all_zero))

    print('DONE')


if __name__ == '__main__':

    print('create ubq_system')
    ubq = read_pdb(tmol.tests.data.pdb.data["1ubq"])
    main(torch.device('cuda:0'), ubq)