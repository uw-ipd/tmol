import attr
import pandas

import numpy
import torch

import itertools

from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.types.torch import Tensor, TensorCollection


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Dummy:
    foo: TensorCollection(torch.float)[2]


def test_tensor_collection_validation():
    import tmol.utility.tensor.compiled as tutc

    tensor = torch.zeros([2, 3], dtype=torch.float)
    tensor_list = [tensor]
    tcollection = tutc.create_tensor_collection2(tensor_list)
    d = Dummy(foo=tcollection)


def test_packed_dun_database_construction(default_database, torch_device):
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    packed_db = resolver.packed_db
    packed_db_aux = resolver.packed_db_aux

    # properties that are independent of the library, except for the
    # fact that it represents alpha amino acids
    assert packed_db.rotameric_bb_start.shape[1] == 2
    assert packed_db.rotameric_bb_step.shape[1] == 2
    assert packed_db.rotameric_bb_periodicity.shape[1] == 2
    assert packed_db.semirot_start.shape[1] == 3
    assert packed_db.semirot_step.shape[1] == 3
    assert packed_db.semirot_periodicity.shape[1] == 3

    # properties that will depend on the libraries that are read in
    dunlib = default_database.scoring.dun
    nlibs = len(dunlib.rotameric_libraries) + len(dunlib.semi_rotameric_libraries)
    assert packed_db.rotameric_bb_start.shape[0] == nlibs
    assert packed_db.rotameric_bb_step.shape[0] == nlibs
    assert packed_db.rotameric_bb_periodicity.shape[0] == nlibs
    assert len(resolver.all_table_indices) == nlibs
    assert packed_db_aux.nchi_for_table_set.shape[0] == nlibs
    assert len(packed_db.rotameric_prob_tables) == nlibs
    assert len(packed_db.rotameric_neglnprob_tables) == nlibs
    assert len(packed_db.rotameric_mean_tables) == nlibs
    assert len(packed_db.rotameric_sdev_tables) == nlibs

    # nrotameric_rots = sum(
    #     (
    #         rotlib.rotameric_data.rotamers.shape[0]
    #         for rotlib in itertools.chain(
    #             dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries
    #         )
    #     )
    # )
    # assert len(packed_db.rotameric_prob_tables) == nrotameric_rots
    for i, rotlib in enumerate(
        itertools.chain(dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries)
    ):
        assert (
            rotlib.rotameric_data.rotamers.shape[0]
            == packed_db.rotameric_prob_tables[i].shape[0]
        )

    # nchitot = sum(
    #    (
    #        rotlib.rotameric_data.rotamers.shape[0]
    #        * rotlib.rotameric_data.rotamers.shape[1]
    #        for rotlib in itertools.chain(
    #            dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries
    #        )
    #    )
    # )
    # assert len(packed_db.rotameric_mean_tables) == nchitot
    # assert len(packed_db.rotameric_sdev_tables) == nchitot
    for i, rotlib in enumerate(
        itertools.chain(dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries)
    ):
        assert (
            rotlib.rotameric_data.rotamers.shape[0]
            * rotlib.rotameric_data.rotamers.shape[1]
            == packed_db.rotameric_mean_tables[i].shape[0]
        )

    max_nrots = sum(
        3 ** rotlib.rotameric_data.rotamers.shape[1]
        for rotlib in dunlib.rotameric_libraries
    ) + sum(
        3 ** rotlib.rotameric_chi_rotamers.shape[1]
        for rotlib in dunlib.semi_rotameric_libraries
    )
    assert packed_db.rotameric_rotind2tableind.shape[0] == max_nrots

    # ok, this kinda makes assumptions about the step + periodicity; I ought to fix this
    for table in packed_db.rotameric_mean_tables:
        assert len(table.shape) == 3 and table.shape[1] == 36 and table.shape[2] == 36
    for table in packed_db.rotameric_sdev_tables:
        assert len(table.shape) == 3 and table.shape[1] == 36 and table.shape[2] == 36

    nrotameric_libs = len(dunlib.rotameric_libraries)
    assert len(resolver.rotameric_table_indices) == nrotameric_libs

    nsemirotameric_libs = len(dunlib.semi_rotameric_libraries)
    assert len(resolver.semirotameric_table_indices) == nsemirotameric_libs
    assert packed_db.semirot_start.shape[0] == nsemirotameric_libs
    assert packed_db.semirot_step.shape[0] == nsemirotameric_libs
    assert packed_db.semirot_periodicity.shape[0] == nsemirotameric_libs
    assert len(packed_db.semirotameric_tables) == nsemirotameric_libs

    for i, rotlib in enumerate(dunlib.semi_rotameric_libraries):
        assert (
            rotlib.rotameric_chi_rotamers.shape[0]
            == packed_db.semirotameric_tables[i].shape[0]
        )

    # now let's make sure that everything lives on the proper device
    for table in itertools.chain(
        packed_db.rotameric_prob_tables,
        packed_db.rotameric_mean_tables,
        packed_db.rotameric_sdev_tables,
        packed_db.semirotameric_tables,
    ):
        assert table.device == torch_device

    # assert packed_db.rotameric_prob_tableset_offsets.device == torch_device
    # assert packed_db.rotameric_meansdev_tableset_offsets.device == torch_device
    assert packed_db.rotameric_bb_start.device == torch_device
    assert packed_db.rotameric_bb_step.device == torch_device
    assert packed_db.rotameric_bb_periodicity.device == torch_device
    assert packed_db.rotameric_rotind2tableind.device == torch_device
    assert packed_db.semirotameric_rotind2tableind.device == torch_device
    # assert packed_db.semirotameric_tableset_offsets.device == torch_device
    assert packed_db.semirot_start.device == torch_device
    assert packed_db.semirot_step.device == torch_device
    assert packed_db.semirot_periodicity.device == torch_device

    assert packed_db_aux.nchi_for_table_set.device == torch_device
    assert packed_db_aux.rotind2tableind_offsets.device == torch_device

    # ok; everything is the right size.
    # make sure that the data has been properly initialized.

    # gone rotprob_offsets = packed_db.rotameric_prob_tableset_offsets.cpu().numpy()
    # gone assert rotprob_offsets[0] == 0
    all_rotdat = [
        rotlib.rotameric_data
        for rotlib in itertools.chain(
            dunlib.rotameric_libraries, dunlib.semi_rotameric_libraries
        )
    ]
    # gone for i in range(1, rotprob_offsets.shape[0]):
    # gone     assert (
    # gone         rotprob_offsets[i - 1] + all_rotdat[i - 1].rotamers.shape[0]
    # gone         == rotprob_offsets[i]
    # gone     )

    # gone rotmean_offsets = packed_db.rotameric_meansdev_tableset_offsets.cpu().numpy()
    # gone rotsandchi = [
    # gone     rotdat.rotamers.shape[0] * rotdat.rotamers.shape[1] for rotdat in all_rotdat
    # gone ]
    # gone assert rotmean_offsets[0] == 0
    # gone for i in range(1, rotmean_offsets.shape[0]):
    # gone     assert rotmean_offsets[i - 1] + rotsandchi[i - 1] == rotmean_offsets[i]

    nchi = [rotdat.rotamers.shape[1] for rotdat in all_rotdat]
    nchi_for_set = packed_db_aux.nchi_for_table_set.cpu().numpy()
    numpy.testing.assert_array_equal(nchi, nchi_for_set)

    rotameric_ri2ti_offsets = packed_db_aux.rotind2tableind_offsets.cpu().numpy()
    rotameric_ri2ti = packed_db.rotameric_rotind2tableind.cpu().numpy()
    rotamers = [
        rotlib.rotameric_data.rotamers for rotlib in dunlib.rotameric_libraries
    ] + [rotlib.rotameric_chi_rotamers for rotlib in dunlib.semi_rotameric_libraries]
    assert rotameric_ri2ti_offsets[0] == 0

    for i in range(rotameric_ri2ti_offsets.shape[0]):
        assert (
            i == 0
            or rotameric_ri2ti_offsets[i - 1] + 3 ** rotamers[i - 1].shape[1]
            == rotameric_ri2ti_offsets[i]
        )

        dim_prods = numpy.power(3, numpy.flip(numpy.arange(rotamers[i].shape[1]), 0))
        n_nonrot_bins = (
            0
            if i < nrotameric_libs
            else dunlib.semi_rotameric_libraries[
                i - nrotameric_libs
            ].rotameric_data.rotamers.shape[0]
            / 3 ** rotamers[i].shape[1]
        )
        for j in range(rotamers[i].shape[0]):
            rotind = rotameric_ri2ti_offsets[i] + numpy.sum(
                (rotamers[i].numpy()[j, :] - 1) * dim_prods, 0
            )
            tableind = j if i < nrotameric_libs else n_nonrot_bins * j
            assert rotameric_ri2ti[rotind] == tableind

    # gone semirot_offsets = packed_db.semirotameric_tableset_offsets.cpu().numpy()
    # gone semirot_rotamers = [
    # gone     rotlib.rotameric_chi_rotamers for rotlib in dunlib.semi_rotameric_libraries
    # gone ]
    # gone assert semirot_offsets[0] == 0
    # gone for i in range(1, len(semirot_rotamers)):
    # gone     assert (
    # gone         semirot_offsets[i - 1] + 3 ** semirot_rotamers[i - 1].shape[1]
    # gone         == semirot_offsets[i]
    # gone     )


def skip_test_dun_param_resolver_construction(default_database, torch_device):

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
        dtype=torch.int32,
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
        dtype=torch.int32,
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
        dtype=torch.int32,
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
        dihedral_offsets_gold, dun_params.dihedral_offset_for_res.cpu().numpy()
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
        dihedral_atom_indices_gold, dun_params.dihedral_atom_inds.cpu().numpy()
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

    # prob_table_offset_for_rotresidue_gold = resolver.packed_db_aux.rotameric_prob_tableset_offsets[
    #     rottable_set_for_res_gold[dun_params.rotres2resid]
    # ]
    # numpy.testing.assert_array_equal(
    #     prob_table_offset_for_rotresidue_gold,
    #     dun_params.prob_table_offset_for_rotresidue.cpu().numpy(),
    # )
    #
    # rotmean_table_offset_for_residue_gold = resolver.packed_db_aux.rotameric_meansdev_tableset_offsets[
    #     rottable_set_for_res_gold
    # ]
    # numpy.testing.assert_array_equal(
    #     rotmean_table_offset_for_residue_gold,
    #     dun_params.rotmean_table_offset_for_residue.cpu().numpy(),
    # )

    rotind2tableind_offset_for_res_gold = resolver.packed_db_aux.rotind2tableind_offsets[
        rottable_set_for_res_gold
    ]
    numpy.testing.assert_array_equal(
        rotind2tableind_offset_for_res_gold,
        dun_params.rotind2tableind_offset_for_res.cpu().numpy(),
    )

    rotameric_chi_desc_gold = numpy.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [4, 0],
            [4, 1],
            [4, 2],
        ],
        # [[1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [4, 0], [4, 1], [4, 2]],
        dtype=int,
    )
    numpy.testing.assert_array_equal(
        rotameric_chi_desc_gold, dun_params.rotameric_chi_desc.cpu().numpy()
    )

    s_inds = resolver.semirotameric_table_indices.get_indexer(example_names)
    semirot_res_inds = s_inds[s_inds != -1]

    semirotameric_chi_desc_gold = numpy.array(
        [[0, 3, -1, 0], [3, 18, -1, 0]], dtype=int
    )
    semirotameric_chi_desc_gold[:, 3] = semirot_res_inds
    # semirotameric_chi_desc_gold[
    #     :, 2
    # ] = resolver.packed_db_aux.semirotameric_tableset_offsets[semirot_res_inds]
    numpy.testing.assert_array_equal(
        semirotameric_chi_desc_gold, dun_params.semirotameric_chi_desc.cpu().numpy()
    )
