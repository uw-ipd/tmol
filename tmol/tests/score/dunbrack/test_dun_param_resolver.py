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
    dun_params = resolver.dun_params

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
    assert dun_params.rotameric_prob_tableset_offsets.shape[0] == nlibs
    assert dun_params.rotameric_meansdev_tableset_offsets.shape[0] == nlibs
    assert dun_params.rotameric_bb_start.shape[0] == nlibs
    assert dun_params.rotameric_bb_step.shape[0] == nlibs
    assert dun_params.rotameric_bb_periodicity.shape[0] == nlibs
    assert len(resolver.all_table_indices) == nlibs
    assert dun_params.nchi_for_table_set.shape[0] == nlibs

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
    def exclusive_cumsum(inds: Tensor(torch.long)[:]) -> Tensor(torch.long)[:]:
        return torch.cat(
            (
                torch.tensor([0], dtype=torch.long, device=inds.device),
                torch.cumsum(inds, 0).narrow(0, 0, inds.shape[0] - 1),
            )
        )

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )

    example_names = numpy.array(["ALA", "PHE", "ARG", "LEU", "GLY", "GLU", "MET"])
    rns_inds, r_inds, s_inds = resolver.resolve_dun_indices(example_names, torch_device)
    print(rns_inds)

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

    nchi_for_pose_res = -1 * torch.ones(
        (len(example_names),), dtype=torch.long, device=torch_device
    )
    # DunbrackParams #5
    nchi_for_res = resolver.dun_params.nchi_for_table_set[rns_inds[rns_inds != -1]]

    nchi_for_pose_res[rns_inds != -1] = nchi_for_res
    print("nchi_for_pose_res", nchi_for_pose_res)

    chi_selected = chi[chi[:, 1] < nchi_for_pose_res[chi[:, 0]], :]
    chi_sel_res = chi_selected[:, 0:1]
    chi_sel_ats = chi_selected[:, 2:]
    print("chi_sel_res.shape", chi_sel_res.shape)
    print("chi_sel_ats.shape", chi_sel_ats.shape)
    chi_wanted = torch.cat((chi_sel_res, chi_sel_ats), 1)
    print("chi_wanted", chi_wanted)
    phi_wanted = phis[rns_inds[phis[:, 0]] != -1]
    print("phi_wanted", phi_wanted)
    psi_wanted = psis[rns_inds[psis[:, 0]] != -1]
    print("psi_wanted", psi_wanted)

    dihedrals = torch.cat((phi_wanted, psi_wanted, chi_wanted), 0)
    if dihedrals.shape[0] < 2049:
        dihedral_res = (example_names.shape[0] + 1) * torch.ones(
            (2049,), dtype=torch.long, device=torch_device
        )
        dihedral_res[: dihedrals.shape[0]] = dihedrals[:, 0]
    else:
        dihedral_res = dihedrals[:, 0]

    dihedral_res_sorted, sort_inds = torch.sort(dihedral_res, 0)

    if dihedrals.shape[0] < 2049:
        sort_inds = sort_inds[: dihedrals.shape[0]]

    # DunbrackParams #3
    dihedral_indices = dihedrals[sort_inds, :]

    # ok, at this point a subset of the residues in the Pose are
    # going to be scored by the dunbrack score. This subset
    # is what we're going to consider when we talk about residues
    # by index. So, e.g., if the first residue to be scored is
    # pose-residue 1, then we'll treat that as dunbrack-residue 0.
    # So we need to remap pose-residue indices into
    # dunbrack-residue indices. With that restricted subset, we'll
    # talk about which residues are rotameric and which residues
    # are semi-rotameric.

    dun_residues = torch.unique(dihedrals[:, 0], sorted=True)
    n_dun_residues = dun_residues.shape[0]
    r_inds = r_inds[dun_residues]
    s_inds = s_inds[dun_residues]

    print("r_inds", r_inds)
    print("s_inds", s_inds)

    # DunbrackParams #1
    ndihe_for_res = 2 + nchi_for_res
    # DunbrackParmas #2
    dihedral_offsets = exclusive_cumsum(ndihe_for_res)

    print("ndihe_for_res", ndihe_for_res)
    print("dihedral_offsets", dihedral_offsets)

    # DunbrackParams #4
    rottable_set_for_res = rns_inds[dun_residues]
    print("rottable_set_for_res", rottable_set_for_res)

    # DunbrackParams #6
    nrotameric_chi_for_res = nchi_for_res
    nrotameric_chi_for_res[s_inds != -1] = nrotameric_chi_for_res[s_inds != -1] - 1
    print("nrotameric_chi_for_res", nrotameric_chi_for_res)

    # DunbrackParams #7
    rotres2resid = torch.arange(
        ndihe_for_res.shape[0], dtype=torch.long, device=torch_device
    )[r_inds != -1]
    print("rotres2resid", rotres2resid)

    dun_rotres = r_inds[r_inds != -1]
    n_rotameric_res = dun_rotres.shape[0]
    # DunbrackParams #8
    prob_table_offset_for_rotresidue = resolver.dun_params.rotameric_prob_tableset_offsets[
        dun_rotres
    ]
    print("prob_table_offset_for_rotresidue", prob_table_offset_for_rotresidue)
    # DunbrackParams #9
    rotmean_table_offset_for_residue = resolver.dun_params.rotameric_prob_tableset_offsets[
        rottable_set_for_res
    ]
    print("rotmean_table_offset_for_residue", rotmean_table_offset_for_residue)

    # DunbrackParams #10
    rotind2tableind_offset_for_res = resolver.dun_params.rotind2tableind_offsets[
        rottable_set_for_res
    ]
    print("rotind2tableind_offset_for_res", rotind2tableind_offset_for_res)

    # DunbrackParams #11
    n_rotameric_chi = torch.sum(nchi_for_res[r_inds != -1])
    n_chi_for_rotameric_res = nchi_for_res[r_inds != -1]
    n_chi_for_rotameric_res_offsets = exclusive_cumsum(n_chi_for_rotameric_res)
    print("n_rotameric_chi", n_rotameric_chi)
    print("n_chi_for_rotameric_res", n_chi_for_rotameric_res)
    print("n_chi_for_rotameric_res_offsets", n_chi_for_rotameric_res_offsets)

    rotameric_chi_desc = torch.zeros(
        [n_rotameric_chi, 2], dtype=torch.long, device=torch_device
    )

    chi_is_first = torch.zeros(
        (n_rotameric_chi,), dtype=torch.long, device=torch_device
    )
    chi_is_first[n_chi_for_rotameric_res_offsets] = torch.ones(
        (n_rotameric_res), dtype=torch.long, device=torch_device
    )
    print("chi_is_first", chi_is_first)
    rotres_for_chi = torch.cumsum(chi_is_first, 0) - 1
    print("rotres_for_chi", rotres_for_chi)
    rotameric_chi_desc[:, 0] = torch.arange(
        n_dun_residues, dtype=torch.long, device=torch_device
    )[rotres_for_chi]

    offsets_for_chi = n_chi_for_rotameric_res_offsets[rotres_for_chi]
    print("offsets_for_chi", offsets_for_chi)

    rotameric_chi_desc[:, 1] = (
        torch.arange(n_rotameric_chi, dtype=torch.long, device=torch_device)
        - offsets_for_chi
    )
    print("rotameric_chi_desc", rotameric_chi_desc)

    # DunbrackParams #11
    n_semirotameric_res = torch.sum(s_inds != -1)
    semirotameric_chi_desc = torch.zeros(
        (n_semirotameric_res, 4), dtype=torch.long, device=torch_device
    )
    semirotameric_chi_desc[:, 0] = torch.arange(
        s_inds.shape[0], dtype=torch.long, device=torch_device
    )[s_inds != -1]

    semirotameric_chi_desc[:, 1] = (
        dihedral_offsets[s_inds != -1] + nchi_for_res[s_inds != -1]
    )
    semirotameric_chi_desc[:, 2] = resolver.dun_params.semirotameric_tableset_offsets[
        s_inds[s_inds != -1]
    ]
    semirotameric_chi_desc[:, 3] = s_inds[s_inds != -1]

    print("semirotameric_chi_desc", semirotameric_chi_desc)
