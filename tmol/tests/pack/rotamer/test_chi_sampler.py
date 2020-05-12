import torch
import numpy

import tmol.pack.rotamer.compiled

from tmol.system.packed import PackedResidueSystem
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.score_graph import score_graph
from tmol.score.dunbrack.params import DunbrackParamResolver


def test_sample_chi_for_rotamers_smoke(ubq_system, default_database, torch_device):
    print("starting test sample chi for rotamers smoke")

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.sampling_db
    dun_params_aux = resolver.scoring_db_aux

    @score_graph
    class CartDunbrackGraph(
        CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
    ):
        pass

    dun_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )

    # resolver = dun_graph.dun_resolve_indices
    # dun_params = resolver.sampling_db #

    def _ti32(l):
        return torch.tensor(l, dtype=torch.int32, device=torch_device)

    def _tf32(l):
        return torch.tensor(l, dtype=torch.float32, device=torch_device)

    ndihe_for_res = _ti32([2, 2, 2])
    dihedral_offset_for_res = _ti32([0, 2, 4])
    dihedral_atom_inds = _ti32(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
        ]
    )

    rottable_set_for_buildable_restype = _ti32(
        [[0, 6], [0, 12], [1, 2], [1, 4], [1, 15], [2, 0]]
    )
    chi_expansion_for_buildable_restype = _ti32(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    nans_for_expansion = numpy.empty((6, 4, 1), dtype=float)
    nans_for_expansion[:] = numpy.nan
    non_dunbrack_expansion_for_buildable_restype = _tf32(nans_for_expansion)
    non_dunbrack_expansion_counts_for_buildable_restype = _ti32(numpy.zeros((6, 4)))
    prob_cumsum_limit_for_buildable_restype = _tf32(numpy.full((6,), 0.95, dtype=float))
    dihedrals = _tf32(numpy.zeros((6), dtype=float))

    torch.ops.tmol.dun_sample_chi(
        dun_graph.coords[0, :],
        dun_params.rotameric_prob_tables,
        dun_params.rotprob_table_sizes,
        dun_params.rotprob_table_strides,
        dun_params.rotameric_mean_tables,
        dun_params.rotameric_sdev_tables,
        dun_params.rotmean_table_sizes,
        dun_params.rotmean_table_strides,
        dun_params.rotameric_bb_start,
        dun_params.rotameric_bb_step,
        dun_params.rotameric_bb_periodicity,
        dun_params.semirotameric_tables,
        dun_params.semirot_table_sizes,
        dun_params.semirot_table_strides,
        dun_params.semirot_start,
        dun_params.semirot_step,
        dun_params.semirot_periodicity,
        dun_params.rotameric_rotind2tableind,
        dun_params.semirotameric_rotind2tableind,
        dun_params.n_rotamers_for_tableset,
        dun_params.n_rotamers_for_tableset_offsets,
        dun_params.sorted_rotamer_2_rotamer,
        ndihe_for_res,
        dihedral_offset_for_res,
        dihedral_atom_inds,
        rottable_set_for_buildable_restype,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_for_buildable_restype,
        non_dunbrack_expansion_counts_for_buildable_restype,
        prob_cumsum_limit_for_buildable_restype,
        dihedrals,
    )
