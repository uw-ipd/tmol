import numpy
import torch

import itertools

from tmol.score.dunbrack.params import DunbrackParamResolver, nplus1d_tensor_from_list


def test_nplus1d_tensor_from_list():
    ts = [
        torch.ones([4, 4], dtype=torch.int32),
        2 * torch.ones([3, 4], dtype=torch.int32),
        3 * torch.ones([5, 2], dtype=torch.int32),
        4 * torch.ones([5, 5], dtype=torch.int32),
    ]
    joined, sizes, strides = nplus1d_tensor_from_list(ts)

    gold_sizes = numpy.array([[4, 4], [3, 4], [5, 2], [5, 5]], dtype=numpy.int64)
    numpy.testing.assert_equal(sizes.cpu().numpy(), gold_sizes)
    for i in range(4):
        for j in range(5):
            for k in range(5):
                assert joined[i, j, k] == (
                    (i + 1) if (j < gold_sizes[i, 0] and k < gold_sizes[i, 1]) else 0
                )


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
    # assert len(resolver.all_table_indices) == nlibs
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
    assert dun_params.rotameric_rotind2tableind.shape[0] == max_nrots

    # ok, this kinda makes assumptions about the step + periodicity; I ought to fix this
    for table in dun_params.rotameric_mean_tables:
        assert len(table.shape) == 2 and table.shape[0] == 36 and table.shape[1] == 36
    for table in dun_params.rotameric_sdev_tables:
        assert len(table.shape) == 2 and table.shape[0] == 36 and table.shape[1] == 36

    nrotameric_libs = len(dunlib.rotameric_libraries)
    # assert len(resolver.rotameric_table_indices) == nrotameric_libs

    nsemirotameric_libs = len(dunlib.semi_rotameric_libraries)
    # assert len(resolver.semirotameric_table_indices) == nsemirotameric_libs
    assert dun_params.semirot_start.shape[0] == nsemirotameric_libs
    assert dun_params.semirot_step.shape[0] == nsemirotameric_libs
    assert dun_params.semirot_periodicity.shape[0] == nsemirotameric_libs
    assert dun_params_aux.semirotameric_tableset_offsets.shape[0] == nsemirotameric_libs

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

    assert dun_params_aux.rotameric_prob_tableset_offsets.device == torch_device
    assert dun_params_aux.rotameric_meansdev_tableset_offsets.device == torch_device
    assert dun_params.rotameric_bb_start.device == torch_device
    assert dun_params.rotameric_bb_step.device == torch_device
    assert dun_params.rotameric_bb_periodicity.device == torch_device
    assert dun_params_aux.nchi_for_table_set.device == torch_device
    assert dun_params.rotameric_rotind2tableind.device == torch_device
    assert dun_params.semirotameric_rotind2tableind.device == torch_device
    assert dun_params_aux.rotind2tableind_offsets.device == torch_device
    assert dun_params_aux.semirotameric_tableset_offsets.device == torch_device
    assert dun_params.semirot_start.device == torch_device
    assert dun_params.semirot_step.device == torch_device
    assert dun_params.semirot_periodicity.device == torch_device

    # ok; everything is the right size.
    # make sure that the data has been properly initialized.

    rotprob_offsets = dun_params_aux.rotameric_prob_tableset_offsets.cpu().numpy()
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

    rotmean_offsets = dun_params_aux.rotameric_meansdev_tableset_offsets.cpu().numpy()
    rotsandchi = [
        rotdat.rotamers.shape[0] * rotdat.rotamers.shape[1] for rotdat in all_rotdat
    ]
    assert rotmean_offsets[0] == 0
    for i in range(1, rotmean_offsets.shape[0]):
        assert rotmean_offsets[i - 1] + rotsandchi[i - 1] == rotmean_offsets[i]

    nchi = [rotdat.rotamers.shape[1] for rotdat in all_rotdat]
    nchi_for_set = dun_params_aux.nchi_for_table_set.cpu().numpy()
    numpy.testing.assert_array_equal(nchi, nchi_for_set)

    rotameric_ri2ti_offsets = dun_params_aux.rotind2tableind_offsets.cpu().numpy()
    rotameric_ri2ti = dun_params.rotameric_rotind2tableind.cpu().numpy()
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

    semirot_offsets = dun_params_aux.semirotameric_tableset_offsets.cpu().numpy()
    semirot_rotamers = [
        rotlib.rotameric_chi_rotamers for rotlib in dunlib.semi_rotameric_libraries
    ]
    assert semirot_offsets[0] == 0
    for i in range(1, len(semirot_rotamers)):
        assert (
            semirot_offsets[i - 1] + 3 ** semirot_rotamers[i - 1].shape[1]
            == semirot_offsets[i]
        )


def test_dun_param_resolver_construction2(default_database, torch_device):

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )

    example_names = numpy.array(
        [["ALA", "PHE", "ARG", "LEU", "GLY", "GLU", "MET"]], dtype=object
    )

    phis = torch.tensor(
        [
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
                [6, 7, 8, 9, 10],
            ]
        ],
        dtype=torch.int32,
        device=torch_device,
    )
    psis = torch.tensor(
        [
            [
                [0, 2, 2, 3, 4],
                [1, 3, 3, 4, 5],
                [2, 4, 4, 5, 6],
                [3, 5, 5, 6, 7],
                [4, 6, 6, 7, 8],
                [5, 7, 7, 8, 9],
                [6, 8, 8, 9, 10],
            ]
        ],
        dtype=torch.int32,
        device=torch_device,
    )
    chi = torch.tensor(
        [
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
            ]
        ],
        dtype=torch.int32,
        device=torch_device,
    )

    dun_params = resolver.resolve_dunbrack_parameters(
        example_names, phis, psis, chi, torch_device
    )

    ndihe_for_res_gold = numpy.array([[4, 6, 4, 5, 5]], dtype=int)
    numpy.testing.assert_array_equal(
        ndihe_for_res_gold, dun_params.ndihe_for_res.cpu().numpy()
    )

    dihedral_offsets_gold = numpy.array([[0, 4, 10, 14, 19]], dtype=int)
    numpy.testing.assert_array_equal(
        dihedral_offsets_gold, dun_params.dihedral_offset_for_res.cpu().numpy()
    )

    dihedral_atom_indices_gold = numpy.array(
        [
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
        ]
    )
    numpy.testing.assert_array_equal(
        dihedral_atom_indices_gold, dun_params.dihedral_atom_inds.cpu().numpy()
    )

    rns_inds = resolver.all_table_indices.index.get_indexer(example_names.ravel())
    rns_inds[rns_inds != -1] = resolver.all_table_indices.iloc[
        rns_inds[rns_inds != -1]
    ]["dun_table_name"].values
    rottable_set_for_res_gold = rns_inds[rns_inds != -1].reshape(1, -1)
    numpy.testing.assert_array_equal(
        rottable_set_for_res_gold, dun_params.rottable_set_for_res.cpu().numpy()
    )

    nchi_for_res_gold = numpy.array([[2, 4, 2, 3, 3]], dtype=int)
    numpy.testing.assert_array_equal(
        nchi_for_res_gold, dun_params.nchi_for_res.cpu().numpy()
    )

    nrotameric_chi_for_res_gold = numpy.array([[1, 4, 2, 2, 3]], dtype=int)
    numpy.testing.assert_array_equal(
        nrotameric_chi_for_res_gold, dun_params.nrotameric_chi_for_res.cpu().numpy()
    )

    rotres2resid_gold = numpy.array([[1, 2, 4]])
    numpy.testing.assert_array_equal(
        rotres2resid_gold, dun_params.rotres2resid.cpu().numpy()
    )

    # prob_table_offset_for_rotresidue_gold => ptofrr_gold
    # annoyingly had to be renamed because flake8 and black couldn't
    # agree on how to treat a long line
    ptofrr_gold = (
        resolver.packed_db_aux.rotameric_prob_tableset_offsets[
            rottable_set_for_res_gold[0, dun_params.rotres2resid.cpu().numpy()]
        ]
        .cpu()
        .numpy()
        .reshape(1, -1)
    )
    numpy.testing.assert_array_equal(
        ptofrr_gold, dun_params.prob_table_offset_for_rotresidue.cpu().numpy()
    )

    # rotmean_table_offset_for_residue_gold
    rmtofr_gold = (
        resolver.packed_db_aux.rotameric_meansdev_tableset_offsets[
            rottable_set_for_res_gold
        ]
        .cpu()
        .numpy()
        .reshape(1, -1)
    )
    numpy.testing.assert_array_equal(
        rmtofr_gold, dun_params.rotmean_table_offset_for_residue.cpu().numpy()
    )

    # rotind2tableind_offset_for_res_gold =
    ri2tiofr_gold = (
        resolver.packed_db_aux.rotind2tableind_offsets[rottable_set_for_res_gold]
        .cpu()
        .numpy()
        .reshape(1, -1)
    )
    numpy.testing.assert_array_equal(
        ri2tiofr_gold, dun_params.rotind2tableind_offset_for_res.cpu().numpy()
    )

    rotameric_chi_desc_gold = numpy.array(
        [
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
            ]
        ],
        dtype=int,
    )
    numpy.testing.assert_array_equal(
        rotameric_chi_desc_gold, dun_params.rotameric_chi_desc.cpu().numpy()
    )

    s_inds = resolver.semirotameric_table_indices.index.get_indexer(
        example_names.ravel()
    )
    s_inds[s_inds != -1] = resolver.semirotameric_table_indices.iloc[
        s_inds[s_inds != -1]
    ]["dun_table_name"].values
    semirot_res_inds = s_inds[s_inds != -1]

    semirotameric_chi_desc_gold = numpy.array(
        [[[0, 3, 0, 0], [3, 18, 0, 0]]], dtype=int
    )
    semirotameric_chi_desc_gold[:, :, 3] = semirot_res_inds
    semirotameric_chi_desc_gold[:, :, 2] = (
        resolver.packed_db_aux.semirotameric_tableset_offsets[semirot_res_inds]
        .cpu()
        .numpy()
    )
    numpy.testing.assert_array_equal(
        semirotameric_chi_desc_gold, dun_params.semirotameric_chi_desc.cpu().numpy()
    )


def stack_system_depth2(torch_device):
    example_names = numpy.array(
        [
            ["ALA", "PHE", "ARG", "LEU", "GLY", "GLU", "MET"],
            ["ALA", "PHE", "ARG", "LEU", "THR", None, None],
        ]
    )

    phis = torch.tensor(
        [
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
                [6, 7, 8, 9, 10],
            ],
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
        ],
        dtype=torch.int32,
        device=torch_device,
    )
    psis = torch.tensor(
        [
            [
                [0, 2, 2, 3, 4],
                [1, 3, 3, 4, 5],
                [2, 4, 4, 5, 6],
                [3, 5, 5, 6, 7],
                [4, 6, 6, 7, 8],
                [5, 7, 7, 8, 9],
                [6, 8, 8, 9, 10],
            ],
            [
                [0, 2, 2, 3, 4],
                [1, 3, 3, 4, 5],
                [2, 4, 4, 5, 6],
                [3, 5, 5, 6, 7],
                [4, 6, 6, 7, 8],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
        ],
        dtype=torch.int32,
        device=torch_device,
    )
    chi = torch.tensor(
        [
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
            [
                [1, 0, 3, 5, 7, 9],
                [1, 1, 5, 7, 9, 11],
                [2, 0, 9, 11, 13, 15],
                [2, 1, 11, 13, 15, 17],
                [2, 2, 13, 15, 17, 19],
                [2, 3, 15, 17, 19, 21],
                [3, 0, 17, 19, 21, 23],
                [3, 1, 19, 21, 23, 25],
                [4, 0, 44, 45, 46, 47],
                [4, 1, 45, 46, 47, 48],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
        ],
        dtype=torch.int32,
        device=torch_device,
    )
    return example_names, phis, psis, chi


def expected_stacked_parameters():
    ndihe_for_res = torch.tensor([[4, 6, 4, 5, 5], [4, 6, 4, 3, -1]], dtype=torch.int32)
    dihedral_offset_for_res = torch.tensor(
        [[0, 4, 10, 14, 19], [0, 4, 10, 14, 17]], dtype=torch.int32
    )
    dihedral_atom_inds = torch.tensor(
        [
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
            ],
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
                [5, 6, 7, 8],
                [6, 6, 7, 8],
                [44, 45, 46, 47],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
        ],
        dtype=torch.int32,
    )
    rottable_set_for_res = torch.tensor(
        [[12, 6, 3, 11, 4], [12, 6, 3, 8, -1]], dtype=torch.int32
    )
    nchi_for_res = torch.tensor([[2, 4, 2, 3, 3], [2, 4, 2, 1, -1]], dtype=torch.int32)
    nrotameric_chi_for_res = torch.tensor(
        [[1, 4, 2, 2, 3], [1, 4, 2, 1, -1]], dtype=torch.int32
    )
    rotres2resid = torch.tensor([[1, 2, 4], [1, 2, 3]], dtype=torch.int32)
    prob_table_offset_for_rotresidue = torch.tensor(
        [[123, 85, 94], [123, 85, 201]], dtype=torch.int32
    )
    rotmean_table_offset_for_residue = torch.tensor(
        [[925, 418, 313, 763, 331], [925, 418, 313, 721, -1]], dtype=torch.int32
    )
    rotind2tableind_offset_for_res = torch.tensor(
        [[258, 156, 93, 249, 102], [258, 156, 93, 240, -1]], dtype=torch.int32
    )
    rotameric_chi_desc = torch.tensor(
        [
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
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ],
        ],
        dtype=torch.int32,
    )
    semirotameric_chi_desc = torch.tensor(
        [[[0, 3, 12, 2], [3, 18, 3, 1]], [[0, 3, 12, 2], [-1, -1, -1, -1]]],
        dtype=torch.int32,
    )
    return dict(
        ndihe_for_res=ndihe_for_res,
        dihedral_offset_for_res=dihedral_offset_for_res,
        dihedral_atom_inds=dihedral_atom_inds,
        rottable_set_for_res=rottable_set_for_res,
        nchi_for_res=nchi_for_res,
        nrotameric_chi_for_res=nrotameric_chi_for_res,
        rotres2resid=rotres2resid,
        prob_table_offset_for_rotresidue=prob_table_offset_for_rotresidue,
        rotmean_table_offset_for_residue=rotmean_table_offset_for_residue,
        rotind2tableind_offset_for_res=rotind2tableind_offset_for_res,
        rotameric_chi_desc=rotameric_chi_desc,
        semirotameric_chi_desc=semirotameric_chi_desc,
    )


def test_stacked_dun_param_resolver_construction(default_database, torch_device):

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    example_names, phis, psis, chi = stack_system_depth2(torch_device)

    dun_params = resolver.resolve_dunbrack_parameters(
        example_names, phis, psis, chi, torch_device
    )

    expected = expected_stacked_parameters()

    torch.testing.assert_allclose(
        expected["ndihe_for_res"], dun_params.ndihe_for_res.to("cpu")
    )
    torch.testing.assert_allclose(
        expected["dihedral_offset_for_res"],
        dun_params.dihedral_offset_for_res.to("cpu"),
    )
    torch.testing.assert_allclose(
        expected["dihedral_atom_inds"], dun_params.dihedral_atom_inds.to("cpu")
    )
    torch.testing.assert_allclose(
        expected["rottable_set_for_res"], dun_params.rottable_set_for_res.to("cpu")
    )
    torch.testing.assert_allclose(
        expected["nchi_for_res"], dun_params.nchi_for_res.to("cpu")
    )
    torch.testing.assert_allclose(
        expected["nrotameric_chi_for_res"], dun_params.nrotameric_chi_for_res.to("cpu")
    )
    torch.testing.assert_allclose(
        expected["rotres2resid"], dun_params.rotres2resid.to("cpu")
    )
    torch.testing.assert_allclose(
        expected["prob_table_offset_for_rotresidue"],
        dun_params.prob_table_offset_for_rotresidue.to("cpu"),
    )
    torch.testing.assert_allclose(
        expected["rotmean_table_offset_for_residue"],
        dun_params.rotmean_table_offset_for_residue.to("cpu"),
    )
    torch.testing.assert_allclose(
        expected["rotind2tableind_offset_for_res"],
        dun_params.rotind2tableind_offset_for_res.to("cpu"),
    )
    torch.testing.assert_allclose(
        expected["rotameric_chi_desc"], dun_params.rotameric_chi_desc.to("cpu")
    )
    torch.testing.assert_allclose(
        expected["semirotameric_chi_desc"], dun_params.semirotameric_chi_desc.to("cpu")
    )
        
