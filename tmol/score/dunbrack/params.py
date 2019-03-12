import attr
import cattr

import numpy
import pandas
import torch

import toolz.functoolz
import itertools

from typing import List, Tuple

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.numeric.bspline import BSplineInterpolation

from tmol.database.scoring.dunbrack_libraries import (
    RotamericDataForAA,
    RotamericAADunbrackLibrary,
    SemiRotamericAADunbrackLibrary,
    DunMappingParams,
    DunbrackRotamerLibrary,
)


def exclusive_cumsum(inds: Tensor(torch.long)[:]) -> Tensor(torch.long)[:]:
    return torch.cat(
        (
            torch.tensor([0], dtype=torch.long, device=inds.device),
            torch.cumsum(inds, 0).narrow(0, 0, inds.shape[0] - 1),
        )
    )


@attr.s(auto_attribs=True)
class DunbrackParams(TensorGroup):
    ndihe_for_res: Tensor(torch.int32)[:]
    dihedral_offsets: Tensor(torch.int32)[:]
    dihedral_atom_indices: Tensor(torch.int32)[..., 4]
    rottable_set_for_res: Tensor(torch.int32)[:]
    nchi_for_res: Tensor(torch.int32)[:]
    nrotameric_chi_for_res: Tensor(torch.int32)[:]  # ??needed??
    rotres2resid: Tensor(torch.int32)[:]
    prob_table_offset_for_rotresidue: Tensor(torch.int32)[:]
    rotmean_table_offset_for_residue: Tensor(torch.int32)[:]
    rotind2tableind_offset_for_res: Tensor(torch.int32)[:]
    rotameric_chi_desc: Tensor(torch.int32)[:, 2]
    semirotameric_chi_desc: Tensor(torch.int32)[:, 4]


# the rama database on the device
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedDunbrackDatabase(ConvertAttrs):

    rotameric_prob_tables: List
    rotameric_mean_tables: List
    rotameric_sdev_tables: List

    rotameric_prob_tableset_offsets: Tensor(torch.long)[:]
    rotameric_meansdev_tableset_offsets: Tensor(torch.long)[:]

    rotameric_bb_start: Tensor(torch.float)[:, :]
    rotameric_bb_step: Tensor(torch.float)[:, :]
    rotameric_bb_periodicity: Tensor(torch.float)[:, :]

    nchi_for_table_set: Tensor(torch.long)[:]
    rotind2tableind: Tensor(torch.long)[:]
    rotind2tableind_offsets: Tensor(torch.long)[:]

    semirotameric_tables: List
    semirotameric_tableset_offsets: Tensor(torch.long)[:]

    semirot_start: Tensor(torch.float)[:, :]
    semirot_step: Tensor(torch.float)[:, :]
    semirot_periodicity: Tensor(torch.float)[:, :]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DunbrackParamResolver(ValidateAttrs):
    _from_dun_db_cache = {}

    # This lives on the device
    dun_params: PackedDunbrackDatabase

    # This will live on the CPU
    all_table_indices: pandas.Index
    rotameric_table_indices: pandas.Index
    semirotameric_table_indices: pandas.Index

    device: torch.device

    @classmethod
    @validate_args
    @toolz.functoolz.memoize(
        cache=_from_dun_db_cache,
        key=lambda args, kwargs: (args[1], args[2].type, args[2].index),
    )
    def from_database(cls, dun_database: DunbrackRotamerLibrary, device: torch.device):
        all_rotlibs = [
            rotlib
            for rotlib in itertools.chain(
                dun_database.rotameric_libraries, dun_database.semi_rotameric_libraries
            )
        ]

        all_table_names = [x.table_name for x in all_rotlibs]
        all_table_lookup = (
            pandas.DataFrame.from_records(cattr.unstructure(dun_database.dun_lookup))
            .set_index("dun_table_name")
            .reindex(all_table_names)
        )
        all_table_indices = pandas.Index(all_table_lookup["residue_name"])

        rotameric_table_names = [x.table_name for x in dun_database.rotameric_libraries]
        rotameric_table_lookup = (
            pandas.DataFrame.from_records(cattr.unstructure(dun_database.dun_lookup))
            .set_index("dun_table_name")
            .reindex(rotameric_table_names)
        )
        rotameric_table_indices = pandas.Index(rotameric_table_lookup["residue_name"])

        semirotameric_table_names = [
            x.table_name for x in dun_database.semi_rotameric_libraries
        ]
        semirotameric_table_lookup = (
            pandas.DataFrame.from_records(cattr.unstructure(dun_database.dun_lookup))
            .set_index("dun_table_name")
            .reindex(semirotameric_table_names)
        )
        semirotameric_table_indices = pandas.Index(
            semirotameric_table_lookup["residue_name"]
        )

        rotameric_prob_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_probabilities[i,])
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_probabilities.shape[0])
        ]

        prob_table_name_and_nrots = [
            (rotlib.table_name, rotlib.rotameric_data.rotamer_probabilities.shape[0])
            for rotlib in all_rotlibs
        ]

        prob_table_nrots = [0] + [
            rotlib.rotameric_data.rotamer_probabilities.shape[0]
            for rotlib in all_rotlibs
        ][:-1]
        nchi_for_table_set = torch.tensor(
            [rotlib.rotameric_data.rotamers.shape[1] for rotlib in all_rotlibs],
            dtype=torch.long,
            device=device,
        )

        prob_table_offsets = torch.cumsum(
            torch.tensor(prob_table_nrots, dtype=torch.long, device=device), 0
        )

        prob_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_prob_tables
        ]

        rotameric_mean_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_means[i, :, :, j])
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_means.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_means.shape[3])
        ]

        mean_table_n_entries = [0] + [
            rotlib.rotameric_data.rotamer_means.shape[0]
            * rotlib.rotameric_data.rotamer_means.shape[3]
            for rotlib in all_rotlibs
        ][:-1]
        rotameric_mean_offsets = torch.cumsum(
            torch.tensor(mean_table_n_entries, dtype=torch.long, device=device), 0
        )

        rotameric_sdev_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_stdvs[i, :, :, j])
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_stdvs.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_stdvs.shape[3])
        ]

        mean_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_mean_tables
        ]

        sdev_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_sdev_tables
        ]

        rotameric_bb_start = torch.tensor(
            [
                list(rotlib.rotameric_data.backbone_dihedral_start)
                for rotlib in all_rotlibs
            ],
            dtype=torch.long,
            device=device,
        )
        rotameric_bb_step = torch.tensor(
            [
                list(rotlib.rotameric_data.backbone_dihedral_step)
                for rotlib in all_rotlibs
            ],
            dtype=torch.long,
            device=device,
        )
        rotameric_bb_periodicity = (
            rotameric_bb_step.new_ones(rotameric_bb_step.shape) * 2 * numpy.pi
        )

        rotind2tableind = []
        rotamer_sets = [
            rotlib.rotameric_data.rotamers
            for rotlib in dun_database.rotameric_libraries
        ] + [
            rotlib.rotameric_chi_rotamers
            for rotlib in dun_database.semi_rotameric_libraries
        ]
        ntablerots = [0]
        for rotamers in rotamer_sets:
            exponents = [x for x in range(rotamers.shape[1])]
            exponents.reverse()
            exponents = torch.tensor(exponents, dtype=torch.long)
            prods = torch.pow(3, exponents)
            rotinds = torch.sum((rotamers - 1) * prods, 1)
            ri2ti = -1 * torch.ones([3 ** rotamers.shape[1]], dtype=torch.long)
            ri2ti[rotinds] = torch.arange(rotamers.shape[0], dtype=torch.long)
            rotind2tableind.extend(list(ri2ti))
            ntablerots.append(rotinds.shape[0])
        rotind2tableind = torch.tensor(
            rotind2tableind, dtype=torch.long, device=device
        ).reshape((-1,))

        rotind2tableind_offsets = torch.cumsum(
            torch.tensor(
                [0] + [3 ** rotamers.shape[1] for rotamers in rotamer_sets][:-1],
                dtype=torch.long,
                device=device,
            ),
            0,
        )

        # ======

        nsemirot_rotamers = [0] + [
            rotlib.nonrotameric_chi_probabilities.shape[0]
            for rotlib in dun_database.semi_rotameric_libraries
        ][:-1]
        semirotameric_tableset_offsets = torch.cumsum(
            torch.tensor(nsemirot_rotamers, dtype=torch.long, device=device), 0
        )
        semirotameric_prob_tables = [
            torch.tensor(rotlib.nonrotameric_chi_probabilities[i,])
            for rotlib in dun_database.semi_rotameric_libraries
            for i in range(rotlib.nonrotameric_chi_probabilities.shape[0])
        ]

        semirot_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in semirotameric_prob_tables
        ]

        semirot_start = torch.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=torch.float,
            device=device,
        )
        semirot_start[:, 0] = -1 * numpy.pi
        semirot_start[:, 1] = -1 * numpy.pi
        semirot_start[:, 2] = torch.tensor(
            [x.non_rot_chi_start for x in dun_database.semi_rotameric_libraries],
            dtype=torch.float,
            device=device,
        )

        semirot_step = torch.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=torch.float,
            device=device,
        )
        semirot_step[:, 0] = 10 * numpy.pi / 180
        semirot_step[:, 1] = 10 * numpy.pi / 180
        semirot_step[:, 2] = torch.tensor(
            [x.non_rot_chi_step for x in dun_database.semi_rotameric_libraries],
            dtype=torch.float,
            device=device,
        )

        semirot_periodicity = torch.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=torch.float,
            device=device,
        )
        semirot_periodicity[:, 0] = 2 * numpy.pi
        semirot_periodicity[:, 1] = 2 * numpy.pi
        semirot_periodicity[:, 2] = torch.tensor(
            [x.non_rot_chi_period for x in dun_database.semi_rotameric_libraries],
            dtype=torch.float,
            device=device,
        )

        dun_params = PackedDunbrackDatabase(
            rotameric_prob_tables=prob_coeffs,
            rotameric_mean_tables=mean_coeffs,
            rotameric_sdev_tables=sdev_coeffs,
            rotameric_prob_tableset_offsets=prob_table_offsets,
            rotameric_meansdev_tableset_offsets=rotameric_mean_offsets,
            rotameric_bb_start=rotameric_bb_start,
            rotameric_bb_step=rotameric_bb_step,
            rotameric_bb_periodicity=rotameric_bb_periodicity,
            nchi_for_table_set=nchi_for_table_set,
            rotind2tableind=rotind2tableind,
            rotind2tableind_offsets=rotind2tableind_offsets,
            semirotameric_tables=semirot_coeffs,
            semirotameric_tableset_offsets=semirotameric_tableset_offsets,
            semirot_start=semirot_start,
            semirot_step=semirot_step,
            semirot_periodicity=semirot_periodicity,
        )

        # rama_params = PackedRamaDatabase(
        #     tables=[
        #         BSplineInterpolation.from_coordinates(
        #             torch.tensor(f.table, dtype=torch.float, device=device)
        #         )
        #         for f in rama_database.rama_tables
        #     ],
        #     bbsteps=[
        #         torch.tensor(f.bbstep, dtype=torch.float, device=device)
        #         for f in rama_database.rama_tables
        #     ],
        #     bbstarts=[
        #         torch.tensor(f.bbstart, dtype=torch.float, device=device)
        #         for f in rama_database.rama_tables
        #     ],
        # )
        #

        return cls(
            dun_params=dun_params,
            all_table_indices=all_table_indices,
            rotameric_table_indices=rotameric_table_indices,
            semirotameric_table_indices=semirotameric_table_indices,
            device=device,
        )

    def resolve_dunbrack_parameters(
        self,
        res_names: NDArray(object)[:],
        phi: Tensor(torch.long)[:, 5],
        psi: Tensor(torch.long)[:, 5],
        chi: Tensor(torch.long)[:, 6],
        torch_device: torch.device,
    ) -> DunbrackParams:
        # soon dfphi = pandas.DataFrame(phi.cpu().numpy())
        # soon dfpsi = pandas.DataFrame(psi.cpu().numpy())
        # soon phipsi = dfphi.merge(dfpsi, left_on=0, right_on=0, suffixes=("_phi","_psi")).values[:,:]
        # soon all_defined = numpy.all(phipsi != -1)

        rns_inds, r_inds, s_inds = self.resolve_dun_indices(res_names, torch_device)

        # the "pose" residues are indexed by the info in phi/psi/chi tensors
        nchi_for_pose_res, nchi_for_res = self.determine_nchi_for_res(
            len(res_names), rns_inds, torch_device
        )

        chi_selected = self.select_chi(chi, nchi_for_pose_res)
        phi_wanted = phi[rns_inds[phi[:, 0]] != -1][:, 1:]
        psi_wanted = psi[rns_inds[psi[:, 0]] != -1][:, 1:]

        # ok, at this point a subset of the residues in the Pose are
        # going to be scored by the dunbrack score. This subset
        # is what we're going to consider when we talk about residues
        # by index. So, e.g., if the first residue to be scored is
        # pose-residue 1, then we'll treat that as dunbrack-residue 0.
        # So we need to remap pose-residue indices into
        # dunbrack-residue indices. With that restricted subset, we'll
        # talk about which residues are rotameric and which residues
        # are semi-rotameric.

        dun_residues = torch.unique(chi_selected[:, 0], sorted=True)
        n_dun_residues = dun_residues.shape[0]
        r_inds = r_inds[dun_residues]
        s_inds = s_inds[dun_residues]

        ndihe_for_res = 2 + nchi_for_res
        dihedral_offsets = exclusive_cumsum(ndihe_for_res)
        dihedral_atom_indices = self.merge_dihedral_atom_indices(
            ndihe_for_res,
            dihedral_offsets,
            nchi_for_res,
            phi_wanted,
            psi_wanted,
            chi_selected,
            torch_device,
        )

        rottable_set_for_res = rns_inds[dun_residues]

        nrotameric_chi_for_res = self.get_nrotameric_chi_for_res(nchi_for_res, s_inds)
        rotres2resid = self.find_rotres2resid(
            ndihe_for_res.shape[0], r_inds, torch_device
        )

        dun_rotres = r_inds[r_inds != -1]
        n_rotameric_res = dun_rotres.shape[0]
        prob_table_offset_for_rotresidue = self.dun_params.rotameric_prob_tableset_offsets[
            dun_rotres
        ]
        rotmean_table_offset_for_residue = self.dun_params.rotameric_meansdev_tableset_offsets[
            rottable_set_for_res
        ]

        rotind2tableind_offset_for_res = self.dun_params.rotind2tableind_offsets[
            rottable_set_for_res
        ]

        rotameric_chi_desc = self.create_rotameric_chi_descriptors(
            nchi_for_res, r_inds, rotres2resid, torch_device
        )

        semirotameric_chi_desc = self.create_semirotameric_chi_descriptors(
            s_inds, dihedral_offsets, nchi_for_res, torch_device
        )

        return DunbrackParams(
            ndihe_for_res=ndihe_for_res,
            dihedral_offsets=dihedral_offsets,
            dihedral_atom_indices=dihedral_atom_indices,
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

    def resolve_dun_indices(
        self, resnames: NDArray(object), device: torch.device
    ) -> Tuple[Tensor(torch.long)[:], Tensor(torch.long)[:], Tensor(torch.long)[:]]:
        rns_inds = self.all_table_indices.get_indexer(resnames)
        r_inds = self.rotameric_table_indices.get_indexer(resnames)
        s_inds = self.semirotameric_table_indices.get_indexer(resnames)

        rns_inds = torch.tensor(rns_inds, dtype=torch.long, device=device)
        r_inds = torch.tensor(r_inds, dtype=torch.long, device=device)
        s_inds = torch.tensor(s_inds, dtype=torch.long, device=device)

        return rns_inds, r_inds, s_inds

    def determine_nchi_for_res(self, nres, rns_inds, torch_device):
        nchi_for_pose_res = -1 * torch.ones(
            (nres,), dtype=torch.long, device=torch_device
        )
        # DunbrackParams #5
        nchi_for_res = self.dun_params.nchi_for_table_set[rns_inds[rns_inds != -1]]

        nchi_for_pose_res[rns_inds != -1] = nchi_for_res
        return nchi_for_pose_res, nchi_for_res

    def select_chi(self, chi, nchi_for_pose_res):
        return chi[chi[:, 1] < nchi_for_pose_res[chi[:, 0]], :]

    def merge_dihedral_atom_indices(
        self,
        ndihe_for_res,
        dihedral_offsets,
        nchi_for_res,
        phi_wanted,
        psi_wanted,
        chi_selected,
        torch_device,
    ):
        n_dun_residues = ndihe_for_res.shape[0]
        dihedral_atom_indices = torch.zeros(
            [torch.sum(ndihe_for_res), 4], dtype=torch.long, device=torch_device
        )
        dihedral_atom_indices[dihedral_offsets, :] = phi_wanted
        dihedral_atom_indices[dihedral_offsets + 1, :] = psi_wanted

        nchi_offsets = exclusive_cumsum(nchi_for_res)
        chi_is_first = torch.zeros(
            (chi_selected.shape[0],), dtype=torch.long, device=torch_device
        )
        chi_is_first[nchi_offsets] = torch.ones(
            (n_dun_residues,), dtype=torch.long, device=torch_device
        )
        res_for_chi = torch.cumsum(chi_is_first, 0) - 1
        offsets_for_chi = dihedral_offsets[res_for_chi]

        chi_sel_ats = chi_selected[:, 2:]
        dihedral_atom_indices[chi_selected[:, 1] + offsets_for_chi + 2] = chi_sel_ats

        return dihedral_atom_indices

    def get_nrotameric_chi_for_res(self, nchi_for_res, s_inds):
        nrotameric_chi_for_res = nchi_for_res.clone()
        nrotameric_chi_for_res[s_inds != -1] = nrotameric_chi_for_res[s_inds != -1] - 1
        return nrotameric_chi_for_res

    def find_rotres2resid(self, nres, r_inds, torch_device):
        return torch.arange(nres, dtype=torch.long, device=torch_device)[r_inds != -1]

    def create_rotameric_chi_descriptors(
        self, nchi_for_res, r_inds, rotres2resid, torch_device
    ):
        n_rotameric_res = torch.sum(r_inds != -1)
        n_rotameric_chi = torch.sum(nchi_for_res[r_inds != -1])
        n_chi_for_rotameric_res = nchi_for_res[r_inds != -1]
        n_chi_for_rotameric_res_offsets = exclusive_cumsum(n_chi_for_rotameric_res)

        rotameric_chi_desc = torch.zeros(
            [n_rotameric_chi, 2], dtype=torch.long, device=torch_device
        )

        rotamericres_chi_is_first = torch.zeros(
            (n_rotameric_chi,), dtype=torch.long, device=torch_device
        )
        rotamericres_chi_is_first[n_chi_for_rotameric_res_offsets] = torch.ones(
            (n_rotameric_res), dtype=torch.long, device=torch_device
        )
        rotres_for_chi = torch.cumsum(rotamericres_chi_is_first, 0) - 1
        rotameric_chi_desc[:, 0] = rotres2resid[rotres_for_chi]

        offsets_for_chi = n_chi_for_rotameric_res_offsets[rotres_for_chi]

        rotameric_chi_desc[:, 1] = (
            torch.arange(n_rotameric_chi, dtype=torch.long, device=torch_device)
            - offsets_for_chi
        )
        return rotameric_chi_desc

    def create_semirotameric_chi_descriptors(
        self, s_inds, dihedral_offsets, nchi_for_res, torch_device
    ):

        n_semirotameric_res = torch.sum(s_inds != -1)
        semirotameric_chi_desc = torch.zeros(
            (n_semirotameric_res, 4), dtype=torch.long, device=torch_device
        )
        semirotameric_chi_desc[:, 0] = torch.arange(
            s_inds.shape[0], dtype=torch.long, device=torch_device
        )[s_inds != -1]

        semirotameric_chi_desc[:, 1] = (
            dihedral_offsets[s_inds != -1] + 1 + nchi_for_res[s_inds != -1]
        )
        semirotameric_chi_desc[:, 2] = self.dun_params.semirotameric_tableset_offsets[
            s_inds[s_inds != -1]
        ]
        semirotameric_chi_desc[:, 3] = s_inds[s_inds != -1]
        return semirotameric_chi_desc
