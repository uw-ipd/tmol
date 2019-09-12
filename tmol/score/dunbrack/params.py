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

from tmol.score.common.stack_condense import (
    condense_torch_inds,
    condense_subset,
    take_condensed_3d_subset,
    take_values_w_sentineled_index,
    take_values_w_sentineled_index_and_dest,
    take_values_w_sentineled_dest,
)


@validate_args
def exclusive_cumsum1d(inds: Tensor(torch.int32)[:]) -> Tensor(torch.int32)[:]:
    return torch.cat(
        (
            torch.tensor([0], dtype=torch.int32, device=inds.device),
            torch.cumsum(inds, 0, dtype=torch.int32).narrow(0, 0, inds.shape[0] - 1),
        )
    )


@validate_args
def exclusive_cumsum2d(inds: Tensor(torch.int32)[:, :]) -> Tensor(torch.int32)[:, :]:
    return torch.cat(
        (
            torch.zeros((inds.shape[0], 1), dtype=torch.int32, device=inds.device),
            torch.cumsum(inds, dim=1, dtype=torch.int32)[:, :-1],
        ),
        dim=1,
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
    ndihe_for_res: Tensor(torch.int32)[:, :]
    dihedral_offset_for_res: Tensor(torch.int32)[:, :]  # prev dihedral_offsets
    dihedral_atom_inds: Tensor(torch.int32)[:, :, 4]  # prev dihedral_atom_indices
    rottable_set_for_res: Tensor(torch.int32)[:, :]
    nchi_for_res: Tensor(torch.int32)[:, :]
    nrotameric_chi_for_res: Tensor(torch.int32)[:, :]  # ??needed??
    rotres2resid: Tensor(torch.int32)[:, :]
    prob_table_offset_for_rotresidue: Tensor(torch.int32)[:, :]
    rotmean_table_offset_for_residue: Tensor(torch.int32)[:, :]
    rotind2tableind_offset_for_res: Tensor(torch.int32)[:, :]
    rotameric_chi_desc: Tensor(torch.int32)[:, :, 2]
    semirotameric_chi_desc: Tensor(torch.int32)[:, :, 4]


@attr.s(auto_attribs=True)
class DunbrackScratch(TensorGroup):
    dihedrals: Tensor(torch.float)[:, :]
    ddihe_dxyz: Tensor(torch.float)[:, :, 4, 3]
    rotameric_rottable_assignment: Tensor(torch.int32)[:, :]
    semirotameric_rottable_assignment: Tensor(torch.int32)[:, :]


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

        all_table_indices = cls._create_all_table_indices(
            [x.table_name for x in all_rotlibs], dun_database.dun_lookup
        )
        rotameric_table_indices = cls._create_rotameric_indices(dun_database)
        semirotameric_table_indices = cls._create_semirotameric_indices(dun_database)
        nchi_for_table_set = cls._create_nchi_for_table_set(all_rotlibs, device)

        prob_table_offsets = cls._create_prob_table_offsets(all_rotlibs, device)
        p_coeffs, pc_sizes, pc_strides, nlp_coeffs = cls._compute_rotprob_coeffs(
            all_rotlibs, device
        )

        rotameric_mean_offsets = cls._create_rot_mean_offsets(all_rotlibs, device)

        mean_coeffs, mc_sizes, mc_strides = cls._calculate_rot_mean_coeffs(
            all_rotlibs, device
        )
        sdev_coeffs = cls._calculate_rot_sdev_coeffs(all_rotlibs, device)

        rot_bb_start, rot_bb_step, rot_bb_per = cls._create_rot_periodicities(
            all_rotlibs, device
        )

        rot_ri2ti, semirot_ri2ti = cls._create_rotind2tableinds(dun_database, device)

        rotind2tableind_offsets = cls._create_rotameric_rotind2tableind_offsets(
            dun_database, device
        )

        sr_coeffs, sr_sizes, sr_strides = cls._calc_semirot_coeffs(dun_database, device)

        sr_start, sr_step, sr_periodicity = cls._create_semirot_periodicity(
            dun_database, device
        )
        sr_tableset_offsets = cls._create_semirot_offsets(dun_database, device)

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
    def _create_all_table_indices(cls, all_table_names, dun_lookup):
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
    def _create_rotameric_indices(cls, dun_database):
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
    def _create_semirotameric_indices(cls, dun_database):
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
    def _create_nchi_for_table_set(cls, all_rotlibs, device):
        return torch.tensor(
            [rotlib.rotameric_data.rotamers.shape[1] for rotlib in all_rotlibs],
            dtype=torch.int32,
            device=device,
        )

    @classmethod
    def _create_prob_table_offsets(cls, all_rotlibs, device):
        prob_table_nrots = torch.tensor(
            [
                rotlib.rotameric_data.rotamer_probabilities.shape[0]
                for rotlib in all_rotlibs
            ],
            dtype=torch.int32,
            device=device,
        )
        return exclusive_cumsum1d(prob_table_nrots)

    @classmethod
    def _compute_rotprob_coeffs(cls, all_rotlibs, device):
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
    def _create_rot_mean_offsets(cls, all_rotlibs, device):
        mean_table_n_entries = [0] + [
            rotlib.rotameric_data.rotamer_means.shape[0]
            * rotlib.rotameric_data.rotamer_means.shape[3]
            for rotlib in all_rotlibs
        ][:-1]
        return torch.cumsum(
            torch.tensor(mean_table_n_entries, dtype=torch.int32, device=device), 0
        )

    @classmethod
    def _calculate_rot_mean_coeffs(cls, all_rotlibs, device):
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
    def _calculate_rot_sdev_coeffs(cls, all_rotlibs, device):
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
    def _create_rot_periodicities(cls, all_rotlibs, device):
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
    def _create_rotind2tableinds(cls, dun_database, device):
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
    def _create_rotameric_rotind2tableind_offsets(cls, dun_database, device):
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
    def _calc_semirot_coeffs(cls, dun_database, device):
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
    def _create_semirot_periodicity(cls, dun_database, device):
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
    def _create_semirot_offsets(cls, dun_database, device):
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
        res_names: NDArray(object)[:, :],
        phi: Tensor(torch.int32)[:, :, 5],
        psi: Tensor(torch.int32)[:, :, 5],
        chi: Tensor(torch.int32)[:, :, 6],
        torch_device: torch.device,
    ) -> DunbrackParams:

        nstacks = res_names.shape[0]
        assert phi.shape[0] == nstacks
        assert psi.shape[0] == nstacks
        assert chi.shape[0] == nstacks

        rns_inds, r_inds, s_inds = self._resolve_dun_indices(res_names, torch_device)

        rns_inds_real = rns_inds != -1
        rns_inds_to_keep = condense_torch_inds(rns_inds_real, torch_device)
        nz_rns_inds = torch.nonzero(rns_inds_to_keep != -1)

        # rottable_set_for_res64 represents the table indices of residues
        # that will be scored by the dunbrack library; we will hold on to
        # `rns_inds` as these are the indices of the pose-residues
        rottable_set_for_res64 = take_values_w_sentineled_dest(
            rns_inds, rns_inds_real, rns_inds_to_keep
        )
        rottable_set_for_res = rottable_set_for_res64.type(torch.int32)

        # the "pose" residues are indexed by the info in phi/psi/chi tensors
        nchi_for_pose_res = take_values_w_sentineled_index(
            self.packed_db_aux.nchi_for_table_set, rns_inds
        ).type(torch.int64)
        nchi_for_res = take_values_w_sentineled_index(
            self.packed_db_aux.nchi_for_table_set, rottable_set_for_res64
        )

        chi_selected = self._select_chi(chi, nchi_for_pose_res).type(torch.int32)

        phi = self._clone_and_mark_missing_bb_atoms(phi, -1)
        psi = self._clone_and_mark_missing_bb_atoms(psi, -2)

        phi_wanted = take_condensed_3d_subset(
            phi[:, :, 1:], rns_inds_to_keep, nz_rns_inds
        )
        psi_wanted = take_condensed_3d_subset(
            psi[:, :, 1:], rns_inds_to_keep, nz_rns_inds
        )

        # ok, at this point a subset of the residues in the Pose are
        # going to be scored by the dunbrack score. This subset
        # is what we're going to consider when we talk about "residues"
        # by index. So, e.g., if the first residue to be scored is
        # pose-residue 1, then we'll treat that as dunbrack-residue 0.
        # So we need to remap pose-residue indices into
        # dunbrack-residue indices. With that restricted subset, we'll
        # talk about which residues are rotameric and which residues
        # are semi-rotameric.

        r_inds = take_values_w_sentineled_dest(r_inds, rns_inds_real, rns_inds_to_keep)
        s_inds = take_values_w_sentineled_dest(s_inds, rns_inds_real, rns_inds_to_keep)

        ndihe_for_res = self._calc_ndihe_for_res(nchi_for_res, torch_device)
        dihedral_offset_for_res = exclusive_cumsum2d(ndihe_for_res)

        dihedral_atom_inds = self._merge_dihedral_atom_indices(
            ndihe_for_res,
            dihedral_offset_for_res,
            nchi_for_res,
            phi_wanted,
            psi_wanted,
            chi_selected,
            torch_device,
        )

        nrotameric_chi_for_res = self._get_nrotameric_chi_for_res(nchi_for_res, s_inds)
        rotres2resid = self._find_rotres2resid(r_inds, torch_device)

        db_aux = self.packed_db_aux
        prob_table_offset_for_rotresidue = self._get_prob_table_offsets_for_rotresidues(
            db_aux, rotres2resid, r_inds
        )

        rotmean_table_offset_for_residue = take_values_w_sentineled_index(
            db_aux.rotameric_meansdev_tableset_offsets, rottable_set_for_res64
        )
        rotind2tableind_offset_for_res = take_values_w_sentineled_index(
            db_aux.rotind2tableind_offsets, rottable_set_for_res64
        )

        rotameric_chi_desc = self._create_rotameric_chi_descriptors(
            nrotameric_chi_for_res,
            rns_inds,
            rns_inds_to_keep,
            nz_rns_inds,
            rotres2resid,
            torch_device,
        )

        semirotameric_chi_desc = self._create_semirotameric_chi_descriptors(
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
        nstacks = params.ndihe_for_res.shape[0]
        ndihe = params.dihedral_atom_inds.shape[1]
        nres = params.ndihe_for_res.shape[1]
        device = params.dihedral_atom_inds.device

        dihedrals = torch.zeros((nstacks, ndihe), dtype=torch.float, device=device)
        ddihe_dxyz = torch.zeros(
            (nstacks, ndihe, 4, 3), dtype=torch.float, device=device
        )
        rotameric_rottable_assignment = torch.zeros(
            (nstacks, nres), dtype=torch.int32, device=device
        )
        semirotameric_rottable_assignment = torch.zeros(
            (nstacks, nres), dtype=torch.int32, device=device
        )

        return DunbrackScratch(
            dihedrals=dihedrals,
            ddihe_dxyz=ddihe_dxyz,
            rotameric_rottable_assignment=rotameric_rottable_assignment,
            semirotameric_rottable_assignment=semirotameric_rottable_assignment,
        )

    @validate_args
    def _indices_from_names(
        self,
        dataframe: pandas.DataFrame,
        names: NDArray(object)[:, :],
        device: torch.device,
    ) -> Tensor(torch.int64)[:, :]:
        names_flat = names.ravel()
        inds = dataframe.index.get_indexer(names_flat)
        inds[inds != -1] = dataframe.iloc[inds[inds != -1]]["dun_table_name"].values
        inds = inds.reshape(names.shape)
        return torch.tensor(inds, dtype=torch.int64, device=device)

    @validate_args
    def _resolve_dun_indices(
        self, resnames: NDArray(object)[:, :], device: torch.device
    ) -> Tuple[
        Tensor(torch.int64)[:, :], Tensor(torch.int64)[:, :], Tensor(torch.int64)[:, :]
    ]:

        rns_inds = self._indices_from_names(self.all_table_indices, resnames, device)
        r_inds = self._indices_from_names(
            self.rotameric_table_indices, resnames, device
        )
        s_inds = self._indices_from_names(
            self.semirotameric_table_indices, resnames, device
        )
        return rns_inds, r_inds, s_inds

    @validate_args
    def _select_chi(
        self,
        chi: Tensor(torch.int32)[:, :, 6],
        nchi_for_pose_res: Tensor(torch.int64)[:, :],
    ) -> Tensor(torch.int64)[:, :, 6]:
        """Not all chi in a residue are used in the dunbrack energy
        calculation, e.g. THR chi2. So, select the subset of chi
        that will be used. The nchi_for_pose array already contains
        the number of chi used by each residue; we compare the index
        of each chi dihedral (column 1) against the number stored
        in the nchi_for_pose array using the residue index (column 0)
        """
        chi64 = chi.type(torch.int64)
        # get the stack indices for each of the chi once we have
        # made a 1-dimensional view.
        stack_inds = (
            torch.arange(chi.shape[0] * chi.shape[1], dtype=torch.int64) / chi.shape[1]
        )
        chi64_res = chi64[:, :, 0].view(-1)

        chi64_in_range = (
            chi64[:, :, 1].view(-1) < nchi_for_pose_res[stack_inds, chi64_res]
        ).view((chi.shape[0], chi.shape[1]))

        # now take a subset of the chi and condense them
        return condense_subset(chi64, chi64_in_range)

    @validate_args
    def _clone_and_mark_missing_bb_atoms(
        self, bb_ats: Tensor(torch.int32)[:, :, 5], undefined_val: int
    ) -> Tensor(torch.int32)[:, :, 5]:
        bb_ats = bb_ats.clone()
        ats_not_defined = (bb_ats == -1).byte().any(2)
        nz_not_defined = torch.nonzero(ats_not_defined)
        bb_ats[nz_not_defined[:, 0], nz_not_defined[:, 1], :] = undefined_val
        return bb_ats

    @validate_args
    def _calc_ndihe_for_res(
        self, nchi_for_res: Tensor(torch.int32)[:, :], torch_device: torch.device
    ) -> Tensor(torch.int32)[:, :]:
        """There are two more dihedrals for each residue than there are chi; these
        are the backbone phi/psi dihedrals"""
        ndihe_for_res = torch.full(
            nchi_for_res.shape, -1, dtype=torch.int32, device=torch_device
        )
        ndihe_for_res[nchi_for_res != -1] = 2 + nchi_for_res[nchi_for_res != -1]
        return ndihe_for_res

    @validate_args
    def _merge_dihedral_atom_indices(
        self,
        ndihe_for_res: Tensor(torch.int32)[:, :],
        dihedral_offset_for_res: Tensor(torch.int32)[:, :],
        nchi_for_res: Tensor(torch.int32)[:, :],
        phi_wanted: Tensor(torch.int32)[:, :, 4],
        psi_wanted: Tensor(torch.int32)[:, :, 4],
        chi_selected: Tensor(torch.int32)[:, :, 6],
        torch_device: torch.device,
    ) -> Tensor(torch.int32)[:, :, 4]:
        dihedral_offset_for_res64 = dihedral_offset_for_res.type(torch.int64)

        nstacks = ndihe_for_res.shape[0]

        ndihe_for_res_w_zeros = ndihe_for_res.clone()
        ndihe_for_res_w_zeros[ndihe_for_res_w_zeros == -1] = 0
        max_ndihe = torch.max(torch.sum(ndihe_for_res_w_zeros, dim=1))
        dihedral_atom_inds = torch.full(
            (nstacks, max_ndihe, 4), -1, dtype=torch.int32, device=torch_device
        )

        real_res = ndihe_for_res != -1
        nz_real_res = torch.nonzero(real_res)

        dihedral_atom_inds[
            nz_real_res[:, 0], dihedral_offset_for_res64[real_res], :
        ] = phi_wanted.type(torch.int32)[real_res]
        dihedral_atom_inds[
            nz_real_res[:, 0], dihedral_offset_for_res64[real_res] + 1, :
        ] = psi_wanted.type(torch.int32)[real_res]

        nchi_offsets = exclusive_cumsum2d(nchi_for_res).type(torch.int64)
        chi_is_first = torch.zeros(
            (nstacks, chi_selected.shape[1]), dtype=torch.int32, device=torch_device
        )
        chi_is_first[nz_real_res[:, 0], nchi_offsets[real_res]] = 1
        res_for_chi64 = torch.cumsum(chi_is_first, dim=1, dtype=torch.int64) - 1

        offsets_for_chi = torch.full(
            (nstacks, chi_selected.shape[1]), -1, dtype=torch.int64, device=torch_device
        )

        real_chi = chi_selected[:, :, 0] >= 0
        nz_real_chi = torch.nonzero(real_chi)
        offsets_for_chi[
            nz_real_chi[:, 0], nz_real_chi[:, 1]
        ] = dihedral_offset_for_res64[nz_real_chi[:, 0], res_for_chi64[real_chi]]

        chi_sel_ats = chi_selected[:, :, 2:]
        chi_selected64 = chi_selected.type(torch.int64)
        dihedral_atom_inds[
            nz_real_chi[:, 0],
            chi_selected64[nz_real_chi[:, 0], nz_real_chi[:, 1], 1]
            + offsets_for_chi[real_chi]
            + 2,
        ] = chi_sel_ats[real_chi]

        return dihedral_atom_inds

    @validate_args
    def _get_nrotameric_chi_for_res(
        self, nchi_for_res: Tensor(torch.int32)[:, :], s_inds: Tensor(torch.int64)[:, :]
    ) -> Tensor(torch.int32)[:, :]:
        nrotameric_chi_for_res = nchi_for_res.clone()
        nrotameric_chi_for_res[s_inds != -1] = nrotameric_chi_for_res[s_inds != -1] - 1
        return nrotameric_chi_for_res

    @validate_args
    def _find_rotres2resid(
        self, r_inds: Tensor(torch.int64)[:, :], torch_device: torch.device
    ) -> Tensor(torch.int32)[:, :]:
        r_inds_condensed = condense_torch_inds(r_inds != -1, torch_device)
        rotres2resid = torch.full(
            r_inds_condensed.shape, -1, dtype=torch.int32, device=torch_device
        )
        rotres2resid[r_inds_condensed != -1] = torch.nonzero(r_inds != -1)[:, 1].type(
            torch.int32
        )
        return rotres2resid

    @validate_args
    def _get_prob_table_offsets_for_rotresidues(
        self,
        db_aux: PackedDunbrackDatabaseAux,
        rotres2resid: Tensor(torch.int32)[:, :],
        r_inds: Tensor(torch.int64)[:, :],
    ) -> Tensor(torch.int32):
        return take_values_w_sentineled_index_and_dest(
            db_aux.rotameric_prob_tableset_offsets, r_inds, rotres2resid
        )

    @validate_args
    def _create_rotameric_chi_descriptors(
        self,
        nrotameric_chi_for_res: Tensor(torch.int32)[:, :],
        rns_inds: Tensor(torch.int64)[:, :],
        rns_inds_to_keep: Tensor(torch.int64)[:, :],
        nz_rns_inds: Tensor(torch.int64)[:, :],
        rotres2resid: Tensor(torch.int32)[:, :],
        torch_device: torch.device,
    ) -> Tensor(torch.int32)[:, :, 2]:
        """Create the array that says for each of the rotameric chi
        0: what (dunbrack) residue (i.e. not Pose residue) does it belong to, and
        1: which chi for that residue is it (i.e. the 1st chi (0), 2nd chi (1) etc).
        A sentinel value of -1 is used for chi that are not real.
        """

        nstacks = rns_inds.shape[0]
        nrotameric_chi_for_res_w_zeros = nrotameric_chi_for_res.clone()
        nrotameric_chi_for_res_w_zeros[nrotameric_chi_for_res < 0] = 0

        # for each stack, how many rotameric chi are there to score?
        nrotameric_chi = torch.sum(nrotameric_chi_for_res_w_zeros, dim=1)

        # what is the largest number of rotameric chi across all stacks?
        max_nrotameric_chi = torch.max(nrotameric_chi)
        nrotameric_chi_for_res_offsets = exclusive_cumsum2d(nrotameric_chi_for_res)

        # the tensor returned by this function
        rotameric_chi_desc = torch.full(
            (nstacks, max_nrotameric_chi, 2), -1, dtype=torch.int32, device=torch_device
        )

        nrotchi_offsets64 = nrotameric_chi_for_res_offsets.type(torch.int64)
        real_res = rns_inds_to_keep != -1
        nz_real_res = nz_rns_inds

        # the subset of the output tensor that are real; e.g. if stack 0
        # has 4 rotameric chi and stack 1 has 5, then the entry
        # rotameric_chi_desc[0,4,:] will be [-1,-1]
        # the "real_chi" tensor is a boolean mask of the chi that are real
        # compare the count from 0..max_nrotameric_chi-1 to the number of chi
        # for the corresponding stack
        real_chi = torch.arange(
            max_nrotameric_chi.item(), dtype=torch.int64, device=torch_device
        ).repeat(nstacks).view((nstacks, max_nrotameric_chi)) < nrotameric_chi.view(
            (-1, 1)
        )
        nz_real_chi = torch.nonzero(real_chi)

        chi_is_first = torch.zeros(
            (rns_inds.shape[0], max_nrotameric_chi),
            dtype=torch.int32,
            device=torch_device,
        )
        chi_is_first[nz_real_res[:, 0], nrotchi_offsets64[real_res]] = 1
        res_for_chi = torch.cumsum(chi_is_first, dim=1) - 1

        rotameric_chi_desc[nz_real_chi[:, 0], nz_real_chi[:, 1], 0] = res_for_chi[
            real_chi
        ].type(torch.int32)

        # 1D array for the real chi of the offset for their residue
        offsets_for_chi = nrotameric_chi_for_res_offsets[
            nz_real_chi[:, 0], res_for_chi[real_chi]
        ]

        # now, store the per-residue index for each of the rotameric chi
        rotameric_chi_desc[nz_real_chi[:, 0], nz_real_chi[:, 1], 1] = (
            nz_real_chi[:, 1].type(torch.int32) - offsets_for_chi
        )

        return rotameric_chi_desc

    @validate_args
    def _create_semirotameric_chi_descriptors(
        self,
        s_inds: Tensor(torch.int64)[:, :],
        dihedral_offset_for_res: Tensor(torch.int32)[:, :],
        nchi_for_res: Tensor(torch.int32),
        torch_device: torch.device,
    ) -> Tensor(torch.int32)[:, :, 4]:
        # semirotchi_desc[:, :,0] == residue index
        # semirotchi_desc[:, :,1] == semirotchi_dihedral_index res
        # semirotchi_desc[:, :,2] == semirot_table_offset
        # semirotchi_desc[:, :,3] == semirot_table_set (in the range 0-7
        #                            for the 8 semirot aas)

        # s_inds.shape[1] is the number of dunbrack residues
        assert s_inds.shape[1] == dihedral_offset_for_res.shape[1]

        nstacks = s_inds.shape[0]
        real_sres = s_inds != -1
        sres_keep = condense_torch_inds(real_sres, torch_device)
        nz_sres_keep = torch.nonzero(sres_keep != -1)

        semirotameric_chi_desc = torch.full(
            (nstacks, sres_keep.shape[1], 4), -1, dtype=torch.int32, device=torch_device
        )
        semirotameric_chi_desc[:, :, 0] = sres_keep.type(torch.int32)

        # the semirotameric chi is the last chi, so, from the residue's dihedral offset
        # add 2 for the two backbone dihedrals and (the number of chi - 1)
        semirotameric_chi_desc[nz_sres_keep[:, 0], nz_sres_keep[:, 1], 1] = (
            dihedral_offset_for_res[s_inds != -1] + 1 + nchi_for_res[s_inds != -1]
        )
        semirotameric_chi_desc[
            nz_sres_keep[:, 0], nz_sres_keep[:, 1], 2
        ] = self.packed_db_aux.semirotameric_tableset_offsets[s_inds[s_inds != -1]]
        semirotameric_chi_desc[nz_sres_keep[:, 0], nz_sres_keep[:, 1], 3] = s_inds[
            s_inds != -1
        ].type(torch.int32)

        return semirotameric_chi_desc
