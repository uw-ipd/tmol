import attr
import pandas

import numpy
import torch

import itertools

from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.types.torch import Tensor

# @attr.s(auto_attribs=True)
# class ScoreSetup:
#     param_resolver: RamaParamResolver
#     tcoords: torch.Tensor
#     tphi_atom_indices: torch.Tensor
#     tpsi_atom_indices: torch.Tensor
#     tramatable_indices: torch.Tensor
#
#     @classmethod
#     def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
#         coords = system.coords
#         tcoords = (
#             torch.from_numpy(coords)
#             .to(device=torch_device, dtype=torch.float)
#             .requires_grad_(True)
#         )
#         res_names = system.atom_metadata["residue_name"].copy()
#
#         rama_database = database.scoring.rama
#         param_resolver = RamaParamResolver.from_database(
#             database.scoring.rama, torch_device
#         )
#
#         phis = numpy.array(
#             [
#                 [
#                     x["residue_index"],
#                     x["atom_index_a"],
#                     x["atom_index_b"],
#                     x["atom_index_c"],
#                     x["atom_index_d"],
#                 ]
#                 for x in system.torsion_metadata[
#                     system.torsion_metadata["name"] == "phi"
#                 ]
#             ]
#         )
#         psis = numpy.array(
#             [
#                 [
#                     x["residue_index"],
#                     x["atom_index_a"],
#                     x["atom_index_b"],
#                     x["atom_index_c"],
#                     x["atom_index_d"],
#                 ]
#                 for x in system.torsion_metadata[
#                     system.torsion_metadata["name"] == "psi"
#                 ]
#             ]
#         )
#         dfphis = pandas.DataFrame(phis)
#         dfpsis = pandas.DataFrame(psis)
#         phipsis = dfphis.merge(
#             dfpsis, left_on=0, right_on=0, suffixes=("_phi", "_psi")
#         ).values[:, 1:]
#
#         ramatable_indices = param_resolver.resolve_ramatables(
#             res_names[phipsis[:, 5]],  # psi atom 'b'
#             res_names[phipsis[:, 7]],  # psi atom 'd'
#         )
#
#         rama_defined = numpy.all(phipsis != -1, axis=1)
#         tphi_atom_indices = torch.from_numpy(phipsis[rama_defined, :4]).to(
#             device=param_resolver.device, dtype=torch.int32
#         )
#         tpsi_atom_indices = torch.from_numpy(phipsis[rama_defined, 4:]).to(
#             device=param_resolver.device, dtype=torch.int32
#         )
#         tramatable_indices = torch.from_numpy(ramatable_indices[rama_defined]).to(
#             device=param_resolver.device, dtype=torch.int32
#         )
#
#         return cls(
#             param_resolver=param_resolver,
#             tcoords=tcoords,
#             tphi_atom_indices=tphi_atom_indices,
#             tpsi_atom_indices=tpsi_atom_indices,
#             tramatable_indices=tramatable_indices,
#         )


def test_dun_param_resolver_construction(default_database, torch_device):
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.packed_db
    dun_params_aux = resolver.packed_db_aux

    # properties that are independent of the library, except for the
    # fact that it represents alpha amino acids
    assert dun_params.rotameric_bb_start.shape[1] == 2
    assert dun_params.rotameric_bb_step.shape[1] == 2
    assert dun_params.rotameric_bb_periodicity.shape[1] == 2
    assert dun_params.semirot_start.shape[1] == 3
    assert dun_params.semirot_step.shape[1] == 3
    assert dun_params.semirot_periodicity.shape[1] == 3

    # properties that will depend on the libraries that are read in
    dunlib = default_database.scoring.dun
    nlibs = len(dunlib.rotameric_libraries) + len(dunlib.semi_rotameric_libraries)
    assert dun_params_aux.rotameric_prob_tableset_offsets.shape[0] == nlibs
    assert dun_params_aux.rotameric_meansdev_tableset_offsets.shape[0] == nlibs
    assert dun_params.rotameric_bb_start.shape[0] == nlibs
    assert dun_params.rotameric_bb_step.shape[0] == nlibs
    assert dun_params.rotameric_bb_periodicity.shape[0] == nlibs
    assert len(resolver.all_table_indices) == nlibs
    assert dun_params_aux.nchi_for_table_set.shape[0] == nlibs

    nrotameric_rots = sum(
        (
            rotlib.rotameric_data.rotamers.shape[0]
            for rotlib in itertools.chain(
                dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries
            )
        )
    )
    assert len(dun_params.rotameric_prob_tables) == nrotameric_rots

    nchitot = sum(
        (
            rotlib.rotameric_data.rotamers.shape[0]
            * rotlib.rotameric_data.rotamers.shape[1]
            for rotlib in itertools.chain(
                dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries
            )
        )
    )
    assert len(dun_params.rotameric_mean_tables) == nchitot
    assert len(dun_params.rotameric_sdev_tables) == nchitot

    max_nrots = sum(
        3 ** rotlib.rotameric_data.rotamers.shape[1]
        for rotlib in dunlib.rotameric_libraries
    ) + sum(
        3 ** rotlib.rotameric_chi_rotamers.shape[1]
        for rotlib in dunlib.semi_rotameric_libraries
    )
    assert dun_params.rotind2tableind.shape[0] == max_nrots

    # ok, this kinda makes assumptions about the step + periodicity; I ought to fix this
    for table in dun_params.rotameric_mean_tables:
        assert len(table.shape) == 2 and table.shape[0] == 36 and table.shape[1] == 36
    for table in dun_params.rotameric_sdev_tables:
        assert len(table.shape) == 2 and table.shape[0] == 36 and table.shape[1] == 36

    nrotameric_libs = len(dunlib.rotameric_libraries)
    assert len(resolver.rotameric_table_indices) == nrotameric_libs

    nsemirotameric_libs = len(dunlib.semi_rotameric_libraries)
    assert len(resolver.semirotameric_table_indices) == nsemirotameric_libs
    assert dun_params.semirot_start.shape[0] == nsemirotameric_libs
    assert dun_params.semirot_step.shape[0] == nsemirotameric_libs
    assert dun_params.semirot_periodicity.shape[0] == nsemirotameric_libs
    assert dun_params.semirotameric_tableset_offsets.shape[0] == nsemirotameric_libs

    n_rotameric_rots_of_semirot_tables = sum(
        rotlib.rotameric_chi_rotamers.shape[0]
        for rotlib in dunlib.semi_rotameric_libraries
    )
    assert len(dun_params.semirotameric_tables) == n_rotameric_rots_of_semirot_tables

    # now let's make sure that everything lives on the proper device
    for table in itertools.chain(
        dun_params.rotameric_prob_tables,
        dun_params.rotameric_mean_tables,
        dun_params.rotameric_sdev_tables,
        dun_params.semirotameric_tables,
    ):
        assert table.device == torch_device

    assert dun_params.rotameric_prob_tableset_offsets.device == torch_device
    assert dun_params.rotameric_meansdev_tableset_offsets.device == torch_device
    assert dun_params.rotameric_bb_start.device == torch_device
    assert dun_params.rotameric_bb_step.device == torch_device
    assert dun_params.rotameric_bb_periodicity.device == torch_device
    assert dun_params.nchi_for_table_set.device == torch_device
    assert dun_params.rotind2tableind.device == torch_device
    assert dun_params.rotind2tableind_offsets.device == torch_device
    assert dun_params.semirotameric_tableset_offsets.device == torch_device
    assert dun_params.semirot_start.device == torch_device
    assert dun_params.semirot_step.device == torch_device
    assert dun_params.semirot_periodicity.device == torch_device

    # ok; everything is the right size.
    # make sure that the data has been properly initialized.

    rotprob_offsets = dun_params.rotameric_prob_tableset_offsets.cpu().numpy()
    assert rotprob_offsets[0] == 0
    all_rotdat = [
        rotlib.rotameric_data
        for rotlib in itertools.chain(
            dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries
        )
    ]
    for i in range(1, rotprob_offsets.shape[0]):
        assert (
            rotprob_offsets[i - 1] + all_rotdat[i - 1].rotamers.shape[0]
            == rotprob_offsets[i]
        )

    rotmean_offsets = dun_params.rotameric_meansdev_tableset_offsets.cpu().numpy()
    rotsandchi = [
        rotdat.rotamers.shape[0] * rotdat.rotamers.shape[1] for rotdat in all_rotdat
    ]
    assert rotmean_offsets[0] == 0
    for i in range(1, rotmean_offsets.shape[0]):
        assert rotmean_offsets[i - 1] + rotsandchi[i - 1] == rotmean_offsets[i]

    nchi = [rotdat.rotamers.shape[1] for rotdat in all_rotdat]
    nchi_for_set = dun_params.nchi_for_table_set.cpu().numpy()
    numpy.testing.assert_array_equal(nchi, nchi_for_set)

    ri2ti_offsets = dun_params.rotind2tableind_offsets.cpu().numpy()
    ri2ti = dun_params.rotind2tableind.cpu().numpy()
    rotamers = [
        rotlib.rotameric_data.rotamers for rotlib in dunlib.rotameric_libraries
    ] + [rotlib.rotameric_chi_rotamers for rotlib in dunlib.semi_rotameric_libraries]
    assert ri2ti_offsets[0] == 0
    for i in range(ri2ti_offsets.shape[0]):
        assert (
            i == 0
            or ri2ti_offsets[i - 1] + 3 ** rotamers[i - 1].shape[1] == ri2ti_offsets[i]
        )

        dim_prods = numpy.power(3, numpy.flip(numpy.arange(rotamers[i].shape[1]), 0))
        for j in range(rotamers[i].shape[0]):
            rotind = ri2ti_offsets[i] + numpy.sum(
                (rotamers[i].numpy()[j, :] - 1) * dim_prods, 0
            )
            assert ri2ti[rotind] == j

    semirot_offsets = dun_params.semirotameric_tableset_offsets.cpu().numpy()
    semirot_rotamers = [
        rotlib.rotameric_chi_rotamers for rotlib in dunlib.semi_rotameric_libraries
    ]
    assert semirot_offsets[0] == 0
    for i in range(1, len(semirot_rotamers)):
        assert (
            semirot_offsets[i - 1] + 3 ** semirot_rotamers[i - 1].shape[1]
            == semirot_offsets[i]
        )


def test_dun_param_resolver_construction(default_database, torch_device):

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )

    example_names = numpy.array(["ALA", "PHE", "ARG", "LEU", "GLY", "GLU", "MET"])

    phis = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10],
        ],
        dtype=torch.long,
        device=torch_device,
    )
    psis = torch.tensor(
        [
            [0, 2, 2, 3, 4],
            [1, 3, 3, 4, 5],
            [2, 4, 4, 5, 6],
            [3, 5, 5, 6, 7],
            [4, 6, 6, 7, 8],
            [5, 7, 7, 8, 9],
            [6, 8, 8, 9, 10],
        ],
        dtype=torch.long,
        device=torch_device,
    )
    chi = torch.tensor(
        [
            [1, 0, 3, 5, 7, 9],
            [1, 1, 5, 7, 9, 11],
            [2, 0, 9, 11, 13, 15],
            [2, 1, 11, 13, 15, 17],
            [2, 2, 13, 15, 17, 19],
            [2, 3, 15, 17, 19, 21],
            [3, 0, 17, 19, 21, 23],
            [3, 1, 19, 21, 23, 25],
            [5, 0, 31, 33, 35, 37],
            [5, 1, 33, 35, 37, 39],
            [5, 2, 35, 36, 37, 39],
            [6, 0, 41, 42, 43, 44],
            [6, 1, 42, 43, 44, 45],
            [6, 2, 43, 44, 45, 46],
        ],
        dtype=torch.long,
        device=torch_device,
    )

    dun_params = resolver.resolve_dunbrack_parameters(
        example_names, phis, psis, chi, torch_device
    )

    ndihe_for_res_gold = numpy.array([4, 6, 4, 5, 5], dtype=int)
    numpy.testing.assert_array_equal(
        ndihe_for_res_gold, dun_params.ndihe_for_res.cpu().numpy()
    )

    dihedral_offsets_gold = numpy.array([0, 4, 10, 14, 19], dtype=int)
    numpy.testing.assert_array_equal(
        dihedral_offsets_gold, dun_params.dihedral_offsets.cpu().numpy()
    )

    dihedral_atom_indices_gold = numpy.array(
        [
            [2, 3, 4, 5],
            [3, 3, 4, 5],
            [3, 5, 7, 9],
            [5, 7, 9, 11],
            [3, 4, 5, 6],
            [4, 4, 5, 6],
            [9, 11, 13, 15],
            [11, 13, 15, 17],
            [13, 15, 17, 19],
            [15, 17, 19, 21],
            [4, 5, 6, 7],
            [5, 5, 6, 7],
            [17, 19, 21, 23],
            [19, 21, 23, 25],
            [6, 7, 8, 9],
            [7, 7, 8, 9],
            [31, 33, 35, 37],
            [33, 35, 37, 39],
            [35, 36, 37, 39],
            [7, 8, 9, 10],
            [8, 8, 9, 10],
            [41, 42, 43, 44],
            [42, 43, 44, 45],
            [43, 44, 45, 46],
        ]
    )
    numpy.testing.assert_array_equal(
        dihedral_atom_indices_gold, dun_params.dihedral_atom_indices.cpu().numpy()
    )

    rns_inds = resolver.all_table_indices.get_indexer(example_names)
    rottable_set_for_res_gold = rns_inds[rns_inds != -1]
    numpy.testing.assert_array_equal(
        rottable_set_for_res_gold, dun_params.rottable_set_for_res.cpu().numpy()
    )

    nchi_for_res_gold = numpy.array([2, 4, 2, 3, 3], dtype=int)
    numpy.testing.assert_array_equal(
        nchi_for_res_gold, dun_params.nchi_for_res.cpu().numpy()
    )

    nrotameric_chi_for_res_gold = numpy.array([1, 4, 2, 2, 3])
    numpy.testing.assert_array_equal(
        nrotameric_chi_for_res_gold, dun_params.nrotameric_chi_for_res
    )

    rotres2resid_gold = numpy.array([1, 2, 4])
    numpy.testing.assert_array_equal(
        rotres2resid_gold, dun_params.rotres2resid.cpu().numpy()
    )

    prob_table_offset_for_rotresidue_gold = resolver.packed_db_aux.rotameric_prob_tableset_offsets[
        rottable_set_for_res_gold[dun_params.rotres2resid]
    ]
    numpy.testing.assert_array_equal(
        prob_table_offset_for_rotresidue_gold,
        dun_params.prob_table_offset_for_rotresidue.cpu().numpy(),
    )

    rotmean_table_offset_for_residue_gold = resolver.packed_db_aux.rotameric_meansdev_tableset_offsets[
        rottable_set_for_res_gold
    ]
    numpy.testing.assert_array_equal(
        rotmean_table_offset_for_residue_gold,
        dun_params.rotmean_table_offset_for_residue.cpu().numpy(),
    )

    rotind2tableind_offset_for_res_gold = resolver.packed_db_aux.rotind2tableind_offsets[
        rottable_set_for_res_gold
    ]
    numpy.testing.assert_array_equal(
        rotind2tableind_offset_for_res_gold,
        dun_params.rotind2tableind_offset_for_res.cpu().numpy(),
    )

    rotameric_chi_desc_gold = numpy.array(
        [[1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [4, 0], [4, 1], [4, 2]],
        dtype=int,
    )
    numpy.testing.assert_array_equal(
        rotameric_chi_desc_gold, dun_params.rotameric_chi_desc.cpu().numpy()
    )

    s_inds = resolver.semirotameric_table_indices.get_indexer(example_names)
    semirot_res_inds = s_inds[s_inds != -1]

    semirotameric_chi_desc_gold = numpy.array([[0, 3, 0, 0], [3, 18, 0, 0]], dtype=int)
    semirotameric_chi_desc_gold[:, 3] = semirot_res_inds
    semirotameric_chi_desc_gold[
        :, 2
    ] = resolver.packed_db_aux.semirotameric_tableset_offsets[semirot_res_inds]
    numpy.testing.assert_array_equal(
        semirotameric_chi_desc_gold, dun_params.semirotameric_chi_desc.cpu().numpy()
    )
