import attr
import pandas

import numpy
import torch

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


def test_dun_param_resolver(default_database, torch_device):
    blah = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
