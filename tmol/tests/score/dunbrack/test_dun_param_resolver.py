import attr
import pandas

import numpy
import torch

import itertools

from tmol.score.dunbrack.params import DunbrackParamResolver


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
    # that doesn't mean it has been properly initialized.

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
