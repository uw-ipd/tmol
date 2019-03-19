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


def exclusive_cumsum(inds: Tensor(torch.int32)[:]) -> Tensor(torch.int32)[:]:
    return torch.cat(
        (
            torch.tensor([0], dtype=torch.int32, device=inds.device),
            torch.cumsum(inds, 0, dtype=torch.int32).narrow(0, 0, inds.shape[0] - 1),
        )
    )


def print_row_numbered_tensor(tensor):
    if len(tensor.shape) == 1:
        print(
            torch.cat(
                (
                    torch.arange(tensor.shape[0], dtype=tensor.dtype).reshape(-1, 1),
                    tensor.reshape(-1, 1),
                ),
                1,
            )
        )
    else:
        print(
            torch.cat(
                (
                    torch.arange(tensor.shape[0], dtype=tensor.dtype).reshape(-1, 1),
                    tensor,
                ),
                1,
            )
        )


@attr.s(auto_attribs=True)
class DunbrackParams(TensorGroup):
    ndihe_for_res: Tensor(torch.int32)[:]
    dihedral_offset_for_res: Tensor(torch.int32)[:]  # prev dihedral_offsets
    dihedral_atom_inds: Tensor(torch.int32)[..., 4]  # prev dihedral_atom_indices
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
    rotameric_neglnprob_tables: List
    rotameric_mean_tables: List
    rotameric_sdev_tables: List

    rotameric_bb_start: Tensor(torch.float)[:, :]
    rotameric_bb_step: Tensor(torch.float)[:, :]
    rotameric_bb_periodicity: Tensor(torch.float)[:, :]

    rotameric_rotind2tableind: Tensor(torch.int32)[:]
    semirotameric_rotind2tableind: Tensor(torch.int32)[:]

    semirotameric_tables: List
    semirot_start: Tensor(torch.float)[:, :]
    semirot_step: Tensor(torch.float)[:, :]
    semirot_periodicity: Tensor(torch.float)[:, :]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedDunbrackDatabaseAux(ConvertAttrs):
    rotameric_prob_tableset_offsets: Tensor(torch.int32)[:]
    rotameric_meansdev_tableset_offsets: Tensor(torch.int32)[:]
    nchi_for_table_set: Tensor(torch.int32)[:]
    rotind2tableind_offsets: Tensor(torch.int32)[:]
    semirotameric_tableset_offsets: Tensor(torch.int32)[:]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DunbrackParamResolver(ValidateAttrs):
    _from_dun_db_cache = {}

    # These live on the device
    packed_db: PackedDunbrackDatabase
    packed_db_aux: PackedDunbrackDatabaseAux

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

        for rotlib in all_rotlibs:
            rotamers = rotlib.rotameric_data.rotamers
            rotmeans = rotlib.rotameric_data.rotamer_means
            # print("Table", rotlib.table_name)
            # for i in range(rotamers.shape[0]):
            #     for j in range(rotamers.shape[1]):
            #         print(" (" + str(rotamers[i,j]) + ", " + str(rotmeans[i,0,0,j]) + ")", end="")
            #     print()

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
        for table in rotameric_prob_tables:
            table[table == 0] = 1e-6
        rotameric_neglnprob_tables = [
            -1 * torch.log(table) for table in rotameric_prob_tables
        ]

        prob_table_name_and_nrots = [
            (rotlib.table_name, rotlib.rotameric_data.rotamer_probabilities.shape[0])
            for rotlib in all_rotlibs
        ]
        # print("prob_table_name_and_nrots")
        # print(prob_table_name_and_nrots)

        nchi_for_table_set = torch.tensor(
            [rotlib.rotameric_data.rotamers.shape[1] for rotlib in all_rotlibs],
            dtype=torch.int32,
            device=device,
        )

        prob_table_nrots = torch.tensor(
            [
                rotlib.rotameric_data.rotamer_probabilities.shape[0]
                for rotlib in all_rotlibs
            ],
            dtype=torch.int32,
            device=device,
        )
        prob_table_offsets = exclusive_cumsum(prob_table_nrots)
        # print("prob_table_offsets")
        # print_row_numbered_tensor(prob_table_offsets)

        prob_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_prob_tables
        ]

        neglnprob_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_neglnprob_tables
        ]

        rotameric_mean_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_means[i, :, :, j])
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_means.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_means.shape[3])
        ]

        # if the mean is near -180, wrap it towards +180
        for x in rotameric_mean_tables:
            x[x < -120] = x[x < -120] + 360
            x *= numpy.pi / 180

        mean_table_n_entries = [0] + [
            rotlib.rotameric_data.rotamer_means.shape[0]
            * rotlib.rotameric_data.rotamer_means.shape[3]
            for rotlib in all_rotlibs
        ][:-1]
        rotameric_mean_offsets = torch.cumsum(
            torch.tensor(mean_table_n_entries, dtype=torch.int32, device=device), 0
        )

        rotameric_sdev_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_stdvs[i, :, :, j])
            * numpy.pi
            / 180
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
            dtype=torch.float,
            device=device,
        )
        rotameric_bb_start *= numpy.pi / 180
        rotameric_bb_step = torch.tensor(
            [
                list(rotlib.rotameric_data.backbone_dihedral_step)
                for rotlib in all_rotlibs
            ],
            dtype=torch.float,
            device=device,
        )
        rotameric_bb_step *= numpy.pi / 180
        rotameric_bb_periodicity = (
            rotameric_bb_step.new_ones(rotameric_bb_step.shape) * 2 * numpy.pi
        )

        rotameric_rotind2tableind = []
        semirotameric_rotind2tableind = []
        ntablerots = [0]
        for rotlib in dun_database.rotameric_libraries:
            rotamers = rotlib.rotameric_data.rotamers
            exponents = [x for x in range(rotamers.shape[1])]
            exponents.reverse()
            exponents = torch.tensor(exponents, dtype=torch.int32)
            prods = torch.pow(3, exponents)
            rotinds = torch.sum((rotamers - 1) * prods, 1)
            ri2ti = -1 * torch.ones([3 ** rotamers.shape[1]], dtype=torch.int32)
            ri2ti[rotinds] = torch.arange(rotamers.shape[0], dtype=torch.int32)
            rotameric_rotind2tableind.extend(list(ri2ti))
            semirotameric_rotind2tableind.extend([0] * len(ri2ti))
            ntablerots.append(rotinds.shape[0])

        for rotlib in dun_database.semi_rotameric_libraries:
            rotameric_rotamers = rotlib.rotameric_data.rotamers
            semirotameric_rotamers = rotlib.rotameric_chi_rotamers
            exponents = [x for x in range(semirotameric_rotamers.shape[1])]
            exponents.reverse()
            exponents = torch.tensor(exponents, dtype=torch.int32)
            prods = torch.pow(3, exponents)
            rotinds = torch.sum((semirotameric_rotamers - 1) * prods, 1)
            sr_ri2ti = -1 * torch.ones(
                [3 ** semirotameric_rotamers.shape[1]], dtype=torch.int32
            )
            sr_ri2ti[rotinds] = torch.arange(
                semirotameric_rotamers.shape[0], dtype=torch.int32
            )
            semirotameric_rotind2tableind.extend(list(sr_ri2ti))

            # OK: this code assumes that a) for the rotameric data, all combinations of rotameric-chi rotamers +
            # binned-non-rotameric-chi are defined (an assumption not needed for rotameric residues) so that
            # 3^(n-rotameric-chi) divides n-rotameric-rotamers cleanly , and
            # b) the rotamers stored in the rotameric_data are in sorted order

            r_ri2ti = -1 * torch.ones(
                [3 ** semirotameric_rotamers.shape[1]], dtype=torch.int32
            )
            n_nonrotameric_chi_rotamers = (
                rotameric_rotamers.shape[0] / 3 ** semirotameric_rotamers.shape[1]
            )
            r_ri2ti[rotinds] = n_nonrotameric_chi_rotamers * torch.arange(
                semirotameric_rotamers.shape[0], dtype=torch.int32
            )
            rotameric_rotind2tableind.extend(list(r_ri2ti))

            # print(rotlib.table_name, "rotinds & rotamers")
            # print(r_ri2ti)
            # print(sr_ri2ti)

        rotameric_rotind2tableind = torch.tensor(
            rotameric_rotind2tableind, dtype=torch.int32, device=device
        ).reshape((-1,))

        semirotameric_rotind2tableind = torch.tensor(
            semirotameric_rotind2tableind, dtype=torch.int32, device=device
        ).reshape((-1,))

        rotamer_sets = [
            rotlib.rotameric_data.rotamers
            for rotlib in dun_database.rotameric_libraries
        ] + [
            rotlib.rotameric_chi_rotamers
            for rotlib in dun_database.semi_rotameric_libraries
        ]

        # same for both rotameric and semi-rotameric rotind2tableind tables
        rotind2tableind_offsets = torch.cumsum(
            torch.tensor(
                [0] + [3 ** rotamers.shape[1] for rotamers in rotamer_sets][:-1],
                dtype=torch.int32,
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
            torch.tensor(nsemirot_rotamers, dtype=torch.int32, device=device), 0
        )
        semirotameric_prob_tables = [
            torch.tensor(rotlib.nonrotameric_chi_probabilities[i,])
            for rotlib in dun_database.semi_rotameric_libraries
            for i in range(rotlib.nonrotameric_chi_probabilities.shape[0])
        ]
        # these aren't used for rotamer building, so we'll just use this for the neglnprobs
        for table in semirotameric_prob_tables:
            table[table == 0] = 1e-6
            table[:] = -1 * torch.log(table)

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
        semirot_start[:, 2] = (
            torch.tensor(
                [x.non_rot_chi_start for x in dun_database.semi_rotameric_libraries],
                dtype=torch.float,
                device=device,
            )
            * numpy.pi
            / 180
        )

        semirot_step = torch.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=torch.float,
            device=device,
        )
        semirot_step[:, 0] = 10 * numpy.pi / 180
        semirot_step[:, 1] = 10 * numpy.pi / 180
        semirot_step[:, 2] = (
            torch.tensor(
                [x.non_rot_chi_step for x in dun_database.semi_rotameric_libraries],
                dtype=torch.float,
                device=device,
            )
            * numpy.pi
            / 180
        )

        semirot_periodicity = torch.zeros(
            (len(dun_database.semi_rotameric_libraries), 3),
            dtype=torch.float,
            device=device,
        )
        semirot_periodicity[:, 0] = 2 * numpy.pi
        semirot_periodicity[:, 1] = 2 * numpy.pi
        semirot_periodicity[:, 2] = (
            torch.tensor(
                [x.non_rot_chi_period for x in dun_database.semi_rotameric_libraries],
                dtype=torch.float,
                device=device,
            )
            * numpy.pi
            / 180
        )

        packed_db = PackedDunbrackDatabase(
            rotameric_prob_tables=prob_coeffs,
            rotameric_neglnprob_tables=neglnprob_coeffs,
            rotameric_mean_tables=mean_coeffs,
            rotameric_sdev_tables=sdev_coeffs,
            rotameric_bb_start=rotameric_bb_start,
            rotameric_bb_step=rotameric_bb_step,
            rotameric_bb_periodicity=rotameric_bb_periodicity,
            rotameric_rotind2tableind=rotameric_rotind2tableind,
            semirotameric_rotind2tableind=semirotameric_rotind2tableind,
            semirotameric_tables=semirot_coeffs,
            semirot_start=semirot_start,
            semirot_step=semirot_step,
            semirot_periodicity=semirot_periodicity,
        )
        packed_db_aux = PackedDunbrackDatabaseAux(
            rotameric_prob_tableset_offsets=prob_table_offsets,
            rotameric_meansdev_tableset_offsets=rotameric_mean_offsets,
            nchi_for_table_set=nchi_for_table_set,
            rotind2tableind_offsets=rotind2tableind_offsets,
            semirotameric_tableset_offsets=semirotameric_tableset_offsets,
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
            packed_db=packed_db,
            packed_db_aux=packed_db_aux,
            all_table_indices=all_table_indices,
            rotameric_table_indices=rotameric_table_indices,
            semirotameric_table_indices=semirotameric_table_indices,
            device=device,
        )

    # @validate_args
    def resolve_dunbrack_parameters(
        self,
        res_names: NDArray(object)[:],
        phi: Tensor(torch.int32)[:, 5],
        psi: Tensor(torch.int32)[:, 5],
        chi: Tensor(torch.int32)[:, 6],
        torch_device: torch.device,
    ) -> DunbrackParams:
        # soon dfphi = pandas.DataFrame(phi.cpu().numpy())
        # soon dfpsi = pandas.DataFrame(psi.cpu().numpy())
        # soon phipsi = dfphi.merge(dfpsi, left_on=0, right_on=0, suffixes=("_phi","_psi")).values[:,:]
        # soon all_defined = numpy.all(phipsi != -1)

        torch.set_printoptions(threshold=5000)
        # print("phi")
        # print(phi)
        # print("psi")
        # print(psi)
        # print("chi")
        # print(chi)

        rns_inds, r_inds, s_inds = self.resolve_dun_indices(res_names, torch_device)

        # the "pose" residues are indexed by the info in phi/psi/chi tensors
        nchi_for_pose_res, nchi_for_res = self.determine_nchi_for_res(
            len(res_names), rns_inds, torch_device
        )

        dun_res_names = res_names[rns_inds.numpy() != -1]
        # print("dun res names")
        # print(dun_res_names.shape)
        # print(dun_res_names)

        chi_selected = self.select_chi(chi, nchi_for_pose_res)
        chi_selected = chi_selected.type(torch.int32)
        phi_wanted = phi[rns_inds[phi[:, 0].type(torch.int64)] != -1][:, 1:]
        psi_wanted = psi[rns_inds[psi[:, 0].type(torch.int64)] != -1][:, 1:]

        # ok, at this point a subset of the residues in the Pose are
        # going to be scored by the dunbrack score. This subset
        # is what we're going to consider when we talk about residues
        # by index. So, e.g., if the first residue to be scored is
        # pose-residue 1, then we'll treat that as dunbrack-residue 0.
        # So we need to remap pose-residue indices into
        # dunbrack-residue indices. With that restricted subset, we'll
        # talk about which residues are rotameric and which residues
        # are semi-rotameric.

        dun_residues = torch.unique(chi_selected[:, 0], sorted=True).type(torch.int64)
        # print("dun_residues")
        # print(dun_residues)
        n_dun_residues = dun_residues.shape[0]
        # print("r_inds before")
        # print(r_inds)
        r_inds = r_inds[dun_residues]
        # print("r_inds after")
        # print(r_inds)
        # print("s_inds before")
        # print(s_inds)
        s_inds = s_inds[dun_residues]
        # print("s_inds after")
        # print(s_inds)

        # print("rotameric residues")
        rotres_inds = numpy.arange(r_inds.shape[0], dtype=int)[r_inds.numpy() != -1]
        # print("rotres_inds")
        # print(rotres_inds)
        # print(dun_res_names[rotres_inds].shape)
        # print(dun_res_names[rotres_inds])

        # print("semirotameric residues")
        semirotres_inds = numpy.arange(s_inds.shape[0], dtype=int)[s_inds.numpy() != -1]
        # print(dun_res_names[semirotres_inds].shape)
        # print(dun_res_names[semirotres_inds])

        ndihe_for_res = 2 + nchi_for_res
        # print("3 nchi_for_res.type()", nchi_for_res.type())
        # print("ndihe_for_res.type()", ndihe_for_res.type())
        dihedral_offset_for_res = exclusive_cumsum(ndihe_for_res)
        dihedral_atom_inds = self.merge_dihedral_atom_indices(
            ndihe_for_res,
            dihedral_offset_for_res,
            nchi_for_res,
            phi_wanted,
            psi_wanted,
            chi_selected,
            torch_device,
        )

        rottable_set_for_res64 = rns_inds[dun_residues]
        rottable_set_for_res = rottable_set_for_res64.type(torch.int32)

        nrotameric_chi_for_res = self.get_nrotameric_chi_for_res(nchi_for_res, s_inds)
        rotres2resid = self.find_rotres2resid(
            ndihe_for_res.shape[0], r_inds, torch_device
        )

        dun_rotres = r_inds[r_inds != -1]
        n_rotameric_res = dun_rotres.shape[0]
        prob_table_offset_for_rotresidue = self.packed_db_aux.rotameric_prob_tableset_offsets[
            dun_rotres
        ]
        # print("prob_table_offset_for_rotresidue")
        # print_row_numbered_tensor(prob_table_offset_for_rotresidue)
        rotmean_table_offset_for_residue = self.packed_db_aux.rotameric_meansdev_tableset_offsets[
            rottable_set_for_res64
        ]

        rotind2tableind_offset_for_res = self.packed_db_aux.rotind2tableind_offsets[
            rottable_set_for_res64
        ]

        rotameric_chi_desc = self.create_rotameric_chi_descriptors(
            nrotameric_chi_for_res, rns_inds, rotres2resid, torch_device
        )

        semirotameric_chi_desc = self.create_semirotameric_chi_descriptors(
            s_inds, dihedral_offset_for_res, nchi_for_res, torch_device, dun_res_names
        )

        return DunbrackParams(
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

    def resolve_dun_indices(
        self, resnames: NDArray(object), device: torch.device
    ) -> Tuple[Tensor(torch.int32)[:], Tensor(torch.int32)[:], Tensor(torch.int32)[:]]:
        # print("resnames", resnames)
        rns_inds = self.all_table_indices.get_indexer(resnames)
        r_inds = self.rotameric_table_indices.get_indexer(resnames)
        s_inds = self.semirotameric_table_indices.get_indexer(resnames)

        rns_inds = torch.tensor(rns_inds, dtype=torch.int64, device=device)
        r_inds = torch.tensor(r_inds, dtype=torch.int64, device=device)
        s_inds = torch.tensor(s_inds, dtype=torch.int64, device=device)

        return rns_inds, r_inds, s_inds

    def determine_nchi_for_res(self, nres, rns_inds, torch_device):
        nchi_for_pose_res = -1 * torch.ones(
            (nres,), dtype=torch.int64, device=torch_device
        )
        # DunbrackParams #5
        nchi_for_res = self.packed_db_aux.nchi_for_table_set[rns_inds[rns_inds != -1]]
        # print("nchi_for_pose_res.type()", nchi_for_pose_res.type())
        # print("1 nchi_for_res.type()", nchi_for_res.type())

        nchi_for_pose_res[rns_inds != -1] = nchi_for_res.type(torch.int64)
        return nchi_for_pose_res, nchi_for_res

    def select_chi(self, chi, nchi_for_pose_res):
        chi64 = chi.type(torch.int64)
        return chi[chi64[:, 1] < nchi_for_pose_res[chi64[:, 0]], :]

    def merge_dihedral_atom_indices(
        self,
        ndihe_for_res,
        dihedral_offset_for_res,
        nchi_for_res,
        phi_wanted,
        psi_wanted,
        chi_selected,
        torch_device,
    ):
        dihedral_offset_for_res64 = dihedral_offset_for_res.type(torch.int64)
        n_dun_residues = ndihe_for_res.shape[0]
        dihedral_atom_inds = torch.zeros(
            [torch.sum(ndihe_for_res), 4], dtype=torch.int32, device=torch_device
        )
        dihedral_atom_inds[dihedral_offset_for_res64, :] = phi_wanted.type(torch.int32)
        dihedral_atom_inds[dihedral_offset_for_res64 + 1, :] = psi_wanted.type(
            torch.int32
        )
        # print("nchi_for_res")
        # print_row_numbered_tensor(nchi_for_res)

        nchi_offsets = exclusive_cumsum(nchi_for_res).type(torch.int64)
        # print("nchi_offsets")
        # print_row_numbered_tensor(nchi_offsets)
        chi_is_first = torch.zeros(
            (chi_selected.shape[0],), dtype=torch.int32, device=torch_device
        )
        chi_is_first[nchi_offsets] = torch.ones(
            (n_dun_residues,), dtype=torch.int32, device=torch_device
        )
        # print("chi_is_first")
        # print_row_numbered_tensor(chi_is_first)
        res_for_chi = torch.cumsum(chi_is_first, 0) - 1
        res_for_chi64 = res_for_chi.type(torch.int64)
        # print("res_for_chi64")
        # print_row_numbered_tensor(res_for_chi64)
        offsets_for_chi = dihedral_offset_for_res64[res_for_chi64]

        chi_sel_ats = chi_selected[:, 2:]
        chi_selected64 = chi_selected.type(torch.int64)
        # print("chi_selected64")
        # print_row_numbered_tensor(chi_selected64)
        # print("offsets_for_chi")
        # print_row_numbered_tensor(offsets_for_chi)
        # print("chi_sel_ats")
        # print_row_numbered_tensor(chi_sel_ats)
        dihedral_atom_inds[chi_selected64[:, 1] + offsets_for_chi + 2] = chi_sel_ats
        # print("dihedral_atom_inds")
        # print_row_numbered_tensor(dihedral_atom_inds)

        return dihedral_atom_inds

    def get_nrotameric_chi_for_res(self, nchi_for_res, s_inds):
        nrotameric_chi_for_res = nchi_for_res.clone()
        nrotameric_chi_for_res[s_inds != -1] = nrotameric_chi_for_res[s_inds != -1] - 1
        return nrotameric_chi_for_res

    def find_rotres2resid(self, nres, r_inds, torch_device):
        return torch.arange(nres, dtype=torch.int32, device=torch_device)[r_inds != -1]

    def create_rotameric_chi_descriptors(
        self, nrotameric_chi_for_res, rns_inds, rotres2resid, torch_device
    ):
        nrotameric_chi = torch.sum(nrotameric_chi_for_res)
        nrotameric_chi_for_res_offsets = exclusive_cumsum(nrotameric_chi_for_res)
        # print("nrotameric_chi_for_res_offsets")
        # print_row_numbered_tensor(nrotameric_chi_for_res_offsets)

        rotameric_chi_desc = torch.zeros(
            [nrotameric_chi, 2], dtype=torch.int32, device=torch_device
        )

        chi_is_first = torch.zeros(
            (nrotameric_chi,), dtype=torch.int32, device=torch_device
        )
        chi_is_first[nrotameric_chi_for_res_offsets.type(torch.int64)] = torch.ones(
            (nrotameric_chi_for_res.shape[0],), dtype=torch.int32, device=torch_device
        )
        res_for_chi = torch.cumsum(chi_is_first, 0) - 1
        rotameric_chi_desc[:, 0] = torch.arange(
            rns_inds.shape[0], dtype=torch.int32, device=torch_device
        )[res_for_chi]

        offsets_for_chi = nrotameric_chi_for_res_offsets[res_for_chi]

        rotameric_chi_desc[:, 1] = (
            torch.arange(nrotameric_chi, dtype=torch.int32, device=torch_device)
            - offsets_for_chi
        )
        # print("rotameric_chi_desc")
        # print_row_numbered_tensor(rotameric_chi_desc)
        return rotameric_chi_desc

    def create_semirotameric_chi_descriptors(
        self, s_inds, dihedral_offset_for_res, nchi_for_res, torch_device, res_names
    ):
        # semirotchi_desc[:,0] == residue index
        # semirotchi_desc[:,1] == semirotchi_dihedral_index res
        # semirotchi_desc[:,2] == semirot_table_offset
        # semirotchi_desc[:,3] == semirot_table_set (in the range 0-7 for the 8 semirot aas)

        n_semirotameric_res = torch.sum(s_inds != -1)
        semirotameric_chi_desc = torch.zeros(
            (n_semirotameric_res, 4), dtype=torch.int32, device=torch_device
        )
        semirotameric_chi_desc[:, 0] = torch.arange(
            s_inds.shape[0], dtype=torch.int32, device=torch_device
        )[s_inds != -1]

        # the semirotameric chi is the last chi, so, from the residue's dihedral offset
        # add 2 for the two backbone dihedrals and (the number of chi - 1)
        semirotameric_chi_desc[:, 1] = (
            dihedral_offset_for_res[s_inds != -1] + 1 + nchi_for_res[s_inds != -1]
        )
        semirotameric_chi_desc[
            :, 2
        ] = self.packed_db_aux.semirotameric_tableset_offsets[s_inds[s_inds != -1]]
        semirotameric_chi_desc[:, 3] = s_inds[s_inds != -1]

        return semirotameric_chi_desc
