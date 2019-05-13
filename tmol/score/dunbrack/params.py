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

from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary


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


# @validate_args
def nplus1d_tensor_from_list(
    tensors: List
):  # -> Tuple[Tensor, Tensor(torch.long)[:,:], Tensor(torch.long)[:,:]] :
    assert len(tensors) > 0

    for tensor in tensors:
        assert len(tensor.shape) == len(tensors[0].shape)
        assert tensor.dtype == tensors[0].dtype
        assert tensor.device == tensors[0].device

    max_sizes = [max(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape))]
    newdimsizes = [len(tensors)] + max_sizes

    newt = torch.zeros(newdimsizes, dtype=tensors[0].dtype, device=tensors[0].device)
    sizes = torch.zeros(
        (len(tensors), tensors[0].dim()), dtype=torch.int64, device=tensors[0].device
    )
    strides = torch.zeros(
        (len(tensors), tensors[0].dim()), dtype=torch.int64, device=tensors[0].device
    )

    for i, t in enumerate(tensors):
        ti = newt[i, :]
        for j in range(t.dim()):
            ti = ti.narrow(j, 0, t.shape[j])
        ti[:] = t
        sizes[i, :] = torch.tensor(t.shape, dtype=torch.int64, device=t.device)
        strides[i, :] = torch.tensor(t.stride(), dtype=torch.int64, device=t.device)
    return newt, sizes, strides


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


@attr.s(auto_attribs=True)
class DunbrackScratch(TensorGroup):
    dihedrals: Tensor(torch.float)[:]
    ddihe_dxyz: Tensor(torch.float)[:, 4, 3]
    rotameric_rottable_assignment: Tensor(torch.int32)[:]
    semirotameric_rottable_assignment: Tensor(torch.int32)[:]


# the rama database on the device
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedDunbrackDatabase(ConvertAttrs):

    rotameric_prob_tables: Tensor(torch.float)[:, :, :]
    rotameric_neglnprob_tables: Tensor(torch.float)[:, :, :]
    rotprob_table_sizes: Tensor(torch.long)[:, 2]
    rotprob_table_strides: Tensor(torch.long)[:, 2]
    rotameric_mean_tables: Tensor(torch.float)[:, :, :]
    rotameric_sdev_tables: Tensor(torch.float)[:, :, :]
    rotmean_table_sizes: Tensor(torch.long)[:, 2]
    rotmean_table_strides: Tensor(torch.long)[:, 2]

    rotameric_bb_start: Tensor(torch.float)[:, :]
    rotameric_bb_step: Tensor(torch.float)[:, :]
    rotameric_bb_periodicity: Tensor(torch.float)[:, :]

    rotameric_rotind2tableind: Tensor(torch.int32)[:]
    semirotameric_rotind2tableind: Tensor(torch.int32)[:]

    semirotameric_tables: Tensor(torch.float)[:, :, :, :]
    semirot_table_sizes: Tensor(torch.long)[:, 3]
    semirot_table_strides: Tensor(torch.long)[:, 3]
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
    all_table_indices: pandas.DataFrame
    rotameric_table_indices: pandas.DataFrame
    semirotameric_table_indices: pandas.DataFrame

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

        all_table_indices = cls.create_all_table_indices(
            [x.table_name for x in all_rotlibs], dun_database.dun_lookup
        )
        rotameric_table_indices = cls.create_rotameric_indices(dun_database)
        semirotameric_table_indices = cls.create_semirotameric_indices(dun_database)
        nchi_for_table_set = cls.create_nchi_for_table_set(all_rotlibs, device)

        prob_table_offsets = cls.create_prob_table_offsets(all_rotlibs, device)
        p_coeffs, pc_sizes, pc_strides, nlp_coeffs = cls.compute_rotprob_coeffs(
            all_rotlibs, device
        )

        rotameric_mean_offsets = cls.create_rot_mean_offsets(all_rotlibs, device)

        mean_coeffs, mc_sizes, mc_strides = cls.calculate_rot_mean_coeffs(
            all_rotlibs, device
        )
        sdev_coeffs = cls.calculate_rot_sdev_coeffs(all_rotlibs, device)

        rot_bb_start, rot_bb_step, rot_bb_per = cls.create_rot_periodicities(
            all_rotlibs, device
        )

        rot_ri2ti, semirot_ri2ti = cls.create_rotind2tableinds(dun_database, device)

        rotind2tableind_offsets = cls.create_rotameric_rotind2tableind_offsets(
            dun_database, device
        )

        sr_coeffs, sr_sizes, sr_strides = cls.calc_semirot_coeffs(dun_database, device)

        sr_start, sr_step, sr_periodicity = cls.create_semirot_periodicity(
            dun_database, device
        )
        sr_tableset_offsets = cls.create_semirot_offsets(dun_database, device)

        packed_db = PackedDunbrackDatabase(
            rotameric_prob_tables=p_coeffs,
            rotameric_neglnprob_tables=nlp_coeffs,
            rotprob_table_sizes=pc_sizes,
            rotprob_table_strides=pc_strides,
            rotameric_mean_tables=mean_coeffs,
            rotameric_sdev_tables=sdev_coeffs,
            rotmean_table_sizes=mc_sizes,
            rotmean_table_strides=mc_strides,
            rotameric_bb_start=rot_bb_start,
            rotameric_bb_step=rot_bb_step,
            rotameric_bb_periodicity=rot_bb_per,
            rotameric_rotind2tableind=rot_ri2ti,
            semirotameric_rotind2tableind=semirot_ri2ti,
            semirotameric_tables=sr_coeffs,
            semirot_table_sizes=sr_sizes,
            semirot_table_strides=sr_strides,
            semirot_start=sr_start,
            semirot_step=sr_step,
            semirot_periodicity=sr_periodicity,
        )
        packed_db_aux = PackedDunbrackDatabaseAux(
            rotameric_prob_tableset_offsets=prob_table_offsets,
            rotameric_meansdev_tableset_offsets=rotameric_mean_offsets,
            nchi_for_table_set=nchi_for_table_set,
            rotind2tableind_offsets=rotind2tableind_offsets,
            semirotameric_tableset_offsets=sr_tableset_offsets,
        )

        return cls(
            packed_db=packed_db,
            packed_db_aux=packed_db_aux,
            all_table_indices=all_table_indices,
            rotameric_table_indices=rotameric_table_indices,
            semirotameric_table_indices=semirotameric_table_indices,
            device=device,
        )

    @classmethod
    def create_all_table_indices(cls, all_table_names, dun_lookup):
        # all_table_names = [x.table_name for x in all_rotlibs]
        all_table_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(dun_lookup)
        ).set_index("residue_name")
        dun_indices = pandas.Index(all_table_names)
        all_table_lookup.dun_table_name = dun_indices.get_indexer(
            all_table_lookup.dun_table_name
        )
        return all_table_lookup

    @classmethod
    def create_rotameric_indices(cls, dun_database):
        rotameric_table_names = [x.table_name for x in dun_database.rotameric_libraries]
        rotameric_table_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(dun_database.dun_lookup)
        ).set_index("residue_name")
        indices = pandas.Index(rotameric_table_names)
        rotameric_table_lookup.dun_table_name = indices.get_indexer(
            rotameric_table_lookup.dun_table_name
        )
        return rotameric_table_lookup

    @classmethod
    def create_semirotameric_indices(cls, dun_database):
        semirotameric_table_names = [
            x.table_name for x in dun_database.semi_rotameric_libraries
        ]
        semirotameric_table_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(dun_database.dun_lookup)
        ).set_index("residue_name")
        indices = pandas.Index(semirotameric_table_names)
        semirotameric_table_lookup.dun_table_name = indices.get_indexer(
            semirotameric_table_lookup.dun_table_name
        )
        return semirotameric_table_lookup

    @classmethod
    def create_nchi_for_table_set(cls, all_rotlibs, device):
        return torch.tensor(
            [rotlib.rotameric_data.rotamers.shape[1] for rotlib in all_rotlibs],
            dtype=torch.int32,
            device=device,
        )

    @classmethod
    def create_prob_table_offsets(cls, all_rotlibs, device):
        prob_table_nrots = torch.tensor(
            [
                rotlib.rotameric_data.rotamer_probabilities.shape[0]
                for rotlib in all_rotlibs
            ],
            dtype=torch.int32,
            device=device,
        )
        return exclusive_cumsum(prob_table_nrots)

    @classmethod
    def compute_rotprob_coeffs(cls, all_rotlibs, device):
        rotameric_prob_tables = [
            rotlib.rotameric_data.rotamer_probabilities[i, :, :].clone().detach()
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_probabilities.shape[0])
        ]
        for table in rotameric_prob_tables:
            table[table == 0] = 1e-6
        rotameric_neglnprob_tables = [
            -1 * torch.log(table) for table in rotameric_prob_tables
        ]

        prob_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_prob_tables
        ]
        prob_coeffs, prob_coeffs_sizes, prob_coeffs_strides = nplus1d_tensor_from_list(
            prob_coeffs
        )

        neglnprob_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_neglnprob_tables
        ]
        neglnprob_coeffs, _, _2 = nplus1d_tensor_from_list(neglnprob_coeffs)
        return prob_coeffs, prob_coeffs_sizes, prob_coeffs_strides, neglnprob_coeffs

    @classmethod
    def create_rot_mean_offsets(cls, all_rotlibs, device):
        mean_table_n_entries = [0] + [
            rotlib.rotameric_data.rotamer_means.shape[0]
            * rotlib.rotameric_data.rotamer_means.shape[3]
            for rotlib in all_rotlibs
        ][:-1]
        return torch.cumsum(
            torch.tensor(mean_table_n_entries, dtype=torch.int32, device=device), 0
        )

    @classmethod
    def calculate_rot_mean_coeffs(cls, all_rotlibs, device):
        rotameric_mean_tables = [
            rotlib.rotameric_data.rotamer_means[i, :, :, j].clone().detach()
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_means.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_means.shape[3])
        ]

        # if the mean is near -180, wrap it towards +180
        for x in rotameric_mean_tables:
            x[x < -120] = x[x < -120] + 360
            x *= numpy.pi / 180

        mean_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_mean_tables
        ]
        mean_coeffs, mean_coeffs_sizes, mean_coeffs_strides = nplus1d_tensor_from_list(
            mean_coeffs
        )

        return mean_coeffs, mean_coeffs_sizes, mean_coeffs_strides

    @classmethod
    def calculate_rot_sdev_coeffs(cls, all_rotlibs, device):
        rotameric_sdev_tables = [
            rotlib.rotameric_data.rotamer_stdvs[i, :, :, j].clone().detach()
            * numpy.pi
            / 180
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_stdvs.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_stdvs.shape[3])
        ]

        sdev_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in rotameric_sdev_tables
        ]
        sdev_coeffs, _, _2 = nplus1d_tensor_from_list(sdev_coeffs)
        return sdev_coeffs

    @classmethod
    def create_rot_periodicities(cls, all_rotlibs, device):
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
        return rotameric_bb_start, rotameric_bb_step, rotameric_bb_periodicity

    @classmethod
    def create_rotind2tableinds(cls, dun_database, device):
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

            if (
                len(rotlib.rotameric_data.rotamer_alias.shape) == 2
                and rotlib.rotameric_data.rotamer_alias.shape[0] > 0
            ):
                orig_rotids = (
                    rotlib.rotameric_data.rotamer_alias[:, 0 : rotamers.shape[1]]
                    .clone()
                    .detach()
                )
                alt_rotids = (
                    rotlib.rotameric_data.rotamer_alias[:, rotamers.shape[1] :]
                    .clone()
                    .detach()
                )
                orig_inds = torch.sum((orig_rotids - 1) * prods, 1)
                alt_inds = torch.sum((alt_rotids - 1) * prods, 1)
                ri2ti[orig_inds.type(torch.long)] = ri2ti[alt_inds.type(torch.long)]

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

            # OK: this code assumes that a) for the rotameric data, all combinations
            # of rotameric-chi rotamers + binned-non-rotameric-chi are defined
            # (an assumption not needed for rotameric residues) so that
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

        rotameric_rotind2tableind = torch.tensor(
            rotameric_rotind2tableind, dtype=torch.int32, device=device
        ).reshape((-1,))

        semirotameric_rotind2tableind = torch.tensor(
            semirotameric_rotind2tableind, dtype=torch.int32, device=device
        ).reshape((-1,))
        return rotameric_rotind2tableind, semirotameric_rotind2tableind

    @classmethod
    def create_rotameric_rotind2tableind_offsets(cls, dun_database, device):
        rotamer_sets = [
            rotlib.rotameric_data.rotamers
            for rotlib in dun_database.rotameric_libraries
        ] + [
            rotlib.rotameric_chi_rotamers
            for rotlib in dun_database.semi_rotameric_libraries
        ]

        # same for both rotameric and semi-rotameric rotind2tableind tables
        return torch.cumsum(
            torch.tensor(
                [0] + [3 ** rotamers.shape[1] for rotamers in rotamer_sets][:-1],
                dtype=torch.int32,
                device=device,
            ),
            0,
        )

    @classmethod
    def calc_semirot_coeffs(cls, dun_database, device):
        semirotameric_prob_tables = [
            rotlib.nonrotameric_chi_probabilities[i, :, :, :].clone().detach()
            for rotlib in dun_database.semi_rotameric_libraries
            for i in range(rotlib.nonrotameric_chi_probabilities.shape[0])
        ]
        # these aren't used for rotamer building, so we'll just use this for
        # the neglnprobs
        for table in semirotameric_prob_tables:
            table[table == 0] = 1e-6
            table[:] = -1 * torch.log(table)

        semirot_coeffs = [
            BSplineInterpolation.from_coordinates(t).coeffs.to(device)
            for t in semirotameric_prob_tables
        ]
        return nplus1d_tensor_from_list(semirot_coeffs)

    @classmethod
    def create_semirot_periodicity(cls, dun_database, device):
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
        return semirot_start, semirot_step, semirot_periodicity

    @classmethod
    def create_semirot_offsets(cls, dun_database, device):
        nsemirot_rotamers = [0] + [
            rotlib.nonrotameric_chi_probabilities.shape[0]
            for rotlib in dun_database.semi_rotameric_libraries
        ][:-1]
        return torch.cumsum(
            torch.tensor(nsemirot_rotamers, dtype=torch.int32, device=device), 0
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

        phi = phi.clone()
        psi = psi.clone()
        phi_not_defined = (phi == -1).byte().any(1)
        phi[phi_not_defined, 1:] = -1
        psi_not_defined = (psi == -1).byte().any(1)
        psi[psi_not_defined, 1:] = -2

        rns_inds, r_inds, s_inds = self.resolve_dun_indices(res_names, torch_device)

        # the "pose" residues are indexed by the info in phi/psi/chi tensors
        nchi_for_pose_res, nchi_for_res = self.determine_nchi_for_res(
            len(res_names), rns_inds, torch_device
        )

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
        # n_dun_residues = dun_residues.shape[0]
        r_inds = r_inds[dun_residues]
        s_inds = s_inds[dun_residues]

        ndihe_for_res = 2 + nchi_for_res
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
        # n_rotameric_res = dun_rotres.shape[0]
        db_aux = self.packed_db_aux
        prob_table_offset_for_rotresidue = db_aux.rotameric_prob_tableset_offsets[
            dun_rotres
        ]
        rotmean_table_offset_for_residue = db_aux.rotameric_meansdev_tableset_offsets[
            rottable_set_for_res64
        ]

        rotind2tableind_offset_for_res = db_aux.rotind2tableind_offsets[
            rottable_set_for_res64
        ]

        rotameric_chi_desc = self.create_rotameric_chi_descriptors(
            nrotameric_chi_for_res, rns_inds, rotres2resid, torch_device
        )

        semirotameric_chi_desc = self.create_semirotameric_chi_descriptors(
            s_inds, dihedral_offset_for_res, nchi_for_res, torch_device
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

    @validate_args
    def allocate_dunbrack_scratch_space(
        self, params: DunbrackParams
    ) -> DunbrackScratch:
        ndihe = params.dihedral_atom_inds.shape[0]
        nres = params.ndihe_for_res.shape[0]
        device = params.dihedral_atom_inds.device

        dihedrals = torch.zeros((ndihe,), dtype=torch.float, device=device)
        ddihe_dxyz = torch.zeros((ndihe, 4, 3), dtype=torch.float, device=device)
        rotameric_rottable_assignment = torch.zeros(
            (nres,), dtype=torch.int32, device=device
        )
        semirotameric_rottable_assignment = torch.zeros(
            (nres,), dtype=torch.int32, device=device
        )

        return DunbrackScratch(
            dihedrals=dihedrals,
            ddihe_dxyz=ddihe_dxyz,
            rotameric_rottable_assignment=rotameric_rottable_assignment,
            semirotameric_rottable_assignment=semirotameric_rottable_assignment,
        )

    def indices_from_names(
        self, dataframe: pandas.DataFrame, names: NDArray(object), device: torch.device
    ):
        inds = dataframe.index.get_indexer(names)
        inds[inds != -1] = dataframe.iloc[inds[inds != -1]]["dun_table_name"].values
        return torch.tensor(inds, dtype=torch.int64, device=device)

    def resolve_dun_indices(
        self, resnames: NDArray(object), device: torch.device
    ) -> Tuple[Tensor(torch.int32)[:], Tensor(torch.int32)[:], Tensor(torch.int32)[:]]:

        rns_inds = self.indices_from_names(self.all_table_indices, resnames, device)
        r_inds = self.indices_from_names(self.rotameric_table_indices, resnames, device)
        s_inds = self.indices_from_names(
            self.semirotameric_table_indices, resnames, device
        )
        return rns_inds, r_inds, s_inds

    def determine_nchi_for_res(self, nres, rns_inds, torch_device):
        nchi_for_pose_res = -1 * torch.ones(
            (nres,), dtype=torch.int64, device=torch_device
        )
        # DunbrackParams #5
        nchi_for_res = self.packed_db_aux.nchi_for_table_set[rns_inds[rns_inds != -1]]

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

        nchi_offsets = exclusive_cumsum(nchi_for_res).type(torch.int64)
        chi_is_first = torch.zeros(
            (chi_selected.shape[0],), dtype=torch.int32, device=torch_device
        )
        chi_is_first[nchi_offsets] = torch.ones(
            (n_dun_residues,), dtype=torch.int32, device=torch_device
        )
        res_for_chi = torch.cumsum(chi_is_first, 0) - 1
        res_for_chi64 = res_for_chi.type(torch.int64)
        offsets_for_chi = dihedral_offset_for_res64[res_for_chi64]

        chi_sel_ats = chi_selected[:, 2:]
        chi_selected64 = chi_selected.type(torch.int64)
        dihedral_atom_inds[chi_selected64[:, 1] + offsets_for_chi + 2] = chi_sel_ats

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
        return rotameric_chi_desc

    def create_semirotameric_chi_descriptors(
        self, s_inds, dihedral_offset_for_res, nchi_for_res, torch_device
    ):
        # semirotchi_desc[:,0] == residue index
        # semirotchi_desc[:,1] == semirotchi_dihedral_index res
        # semirotchi_desc[:,2] == semirot_table_offset
        # semirotchi_desc[:,3] == semirot_table_set (in the range 0-7
        #                         for the 8 semirot aas)

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
