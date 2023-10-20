import attr
import cattr

import numpy
import pandas
import torch

import toolz.functoolz
import itertools

from typing import Tuple

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.numeric.bspline import BSplineInterpolation

from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

from tmol.utility.tensor.common_operations import (
    exclusive_cumsum1d,
    exclusive_cumsum2d,
    # print_row_numbered_tensor,
    nplus1d_tensor_from_list,
    cat_differently_sized_tensors,
)

from tmol.score.common.stack_condense import (
    condense_torch_inds,
    condense_subset,
    take_condensed_3d_subset,
    take_values_w_sentineled_index,
    take_values_w_sentineled_index_and_dest,
    take_values_w_sentineled_dest,
)


@attr.s(auto_attribs=True)
class DunbrackParams(TensorGroup):
    ndihe_for_res: Tensor[torch.int32][:, :]
    dihedral_offset_for_res: Tensor[torch.int32][:, :]  # prev dihedral_offsets
    dihedral_atom_inds: Tensor[torch.int32][:, :, 4]  # prev dihedral_atom_indices
    rottable_set_for_res: Tensor[torch.int32][:, :]
    nchi_for_res: Tensor[torch.int32][:, :]
    nrotameric_chi_for_res: Tensor[torch.int32][:, :]  # ??needed??
    rotres2resid: Tensor[torch.int32][:, :]
    prob_table_offset_for_rotresidue: Tensor[torch.int32][:, :]
    rotmean_table_offset_for_residue: Tensor[torch.int32][:, :]
    rotind2tableind_offset_for_res: Tensor[torch.int32][:, :]
    rotameric_chi_desc: Tensor[torch.int32][:, :, 2]
    semirotameric_chi_desc: Tensor[torch.int32][:, :, 4]


@attr.s(auto_attribs=True)
class DunbrackScratch(TensorGroup):
    dihedrals: Tensor[torch.float][:, :]
    ddihe_dxyz: Tensor[torch.float][:, :, 4, 3]
    rotameric_rottable_assignment: Tensor[torch.int32][:, :]
    semirotameric_rottable_assignment: Tensor[torch.int32][:, :]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDunbrackDatabaseView(ConvertAttrs):
    """The tables for the dunbrack database needed for scoring
    stored on the device
    """

    rotameric_neglnprob_tables: Tensor[torch.float][:, :, :]
    rotprob_table_sizes: Tensor[torch.long][:, 2]
    rotprob_table_strides: Tensor[torch.long][:, 2]
    rotameric_mean_tables: Tensor[torch.float][:, :, :]
    rotameric_sdev_tables: Tensor[torch.float][:, :, :]
    rotmean_table_sizes: Tensor[torch.long][:, 2]
    rotmean_table_strides: Tensor[torch.long][:, 2]

    rotameric_bb_start: Tensor[torch.float][:, :]
    rotameric_bb_step: Tensor[torch.float][:, :]
    rotameric_bb_periodicity: Tensor[torch.float][:, :]

    rotameric_rotind2tableind: Tensor[torch.int32][:]
    semirotameric_rotind2tableind: Tensor[torch.int32][:]

    semirotameric_tables: Tensor[torch.float][:, :, :, :]
    semirot_table_sizes: Tensor[torch.long][:, 3]
    semirot_table_strides: Tensor[torch.long][:, 3]
    semirot_start: Tensor[torch.float][:, :]
    semirot_step: Tensor[torch.float][:, :]
    semirot_periodicity: Tensor[torch.float][:, :]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDunbrackDatabaseAux(ConvertAttrs):
    rotameric_prob_tableset_offsets: Tensor[torch.int32][:]
    rotameric_meansdev_tableset_offsets: Tensor[torch.int32][:]
    nchi_for_table_set: Tensor[torch.int32][:]
    rotameric_chi_ri2ti_offsets: Tensor[torch.int32][:]
    semirotameric_tableset_offsets: Tensor[torch.int32][:]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SamplingDunbrackDatabaseView(ConvertAttrs):
    """The tables that are needed in order to sample
    side-chain conformations.
    """

    rotameric_prob_tables: Tensor[torch.float][:, :, :]
    rotprob_table_sizes: Tensor[torch.long][:, 2]
    rotprob_table_strides: Tensor[torch.long][:, 2]
    rotameric_mean_tables: Tensor[torch.float][:, :, :]
    rotameric_sdev_tables: Tensor[torch.float][:, :, :]
    rotmean_table_sizes: Tensor[torch.long][:, 2]
    rotmean_table_strides: Tensor[torch.long][:, 2]
    rotameric_meansdev_tableset_offsets: Tensor[torch.int32][:]

    n_rotamers_for_tableset: Tensor[torch.long][:]
    n_rotamers_for_tableset_offsets: Tensor[torch.int32][:]
    sorted_rotamer_2_rotamer: Tensor[torch.long][:, :, :]

    rotameric_bb_start: Tensor[torch.float][:, :]
    rotameric_bb_step: Tensor[torch.float][:, :]
    rotameric_bb_periodicity: Tensor[torch.float][:, :]

    rotameric_rotind2tableind: Tensor[torch.int32][:]
    semirotameric_rotind2tableind: Tensor[torch.int32][:]
    all_chi_rotind2tableind: Tensor[torch.int32][:]
    all_chi_rotind2tableind_offsets: Tensor[torch.int32][:]

    semirotameric_tables: Tensor[torch.float][:, :, :, :]
    semirot_table_sizes: Tensor[torch.long][:, 3]
    semirot_table_strides: Tensor[torch.long][:, 3]
    semirot_start: Tensor[torch.float][:, :]
    semirot_step: Tensor[torch.float][:, :]
    semirot_periodicity: Tensor[torch.float][:, :]

    nchi_for_table_set: Tensor[torch.int32][:]
    rotwells: Tensor[torch.int32][:, :]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DunbrackParamResolver(ValidateAttrs):
    _from_dun_db_cache = {}

    # These live on the device
    scoring_db: ScoringDunbrackDatabaseView
    scoring_db_aux: ScoringDunbrackDatabaseAux
    sampling_db: SamplingDunbrackDatabaseView

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

        prob_table_nrots, prob_table_offsets = cls._create_prob_table_offsets(
            all_rotlibs, device
        )
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

        rot_ri2ti, semirot_ri2ti, allchi_ri2ti = cls._create_rotind2tableinds(
            dun_database, device
        )

        rotameric_chi_ri2ti_offsets = cls._create_rotameric_rotind2tableind_offsets(
            dun_database, device
        )
        all_chi_rotind2tableind_offsets = cls._create_all_chi_rotind2tableind_offsets(
            dun_database, device
        )

        sr_coeffs, sr_sizes, sr_strides = cls._calc_semirot_coeffs(dun_database, device)

        sr_start, sr_step, sr_periodicity = cls._create_semirot_periodicity(
            dun_database, device
        )
        sr_tableset_offsets = cls._create_semirot_offsets(dun_database, device)

        sorted_rotamer_2_rotamer = cls.create_sorted_rot_2_rot(all_rotlibs, device)

        rotwells = cls.create_rotamer_well_table(all_rotlibs, device)

        scoring_db = ScoringDunbrackDatabaseView(
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
        scoring_db_aux = ScoringDunbrackDatabaseAux(
            rotameric_prob_tableset_offsets=prob_table_offsets,
            rotameric_meansdev_tableset_offsets=rotameric_mean_offsets,
            nchi_for_table_set=nchi_for_table_set,
            rotameric_chi_ri2ti_offsets=rotameric_chi_ri2ti_offsets,
            semirotameric_tableset_offsets=sr_tableset_offsets,
        )

        sampling_db = SamplingDunbrackDatabaseView(
            rotameric_prob_tables=p_coeffs,
            rotprob_table_sizes=pc_sizes,
            rotprob_table_strides=pc_strides,
            rotameric_mean_tables=mean_coeffs,
            rotameric_sdev_tables=sdev_coeffs,
            rotmean_table_sizes=mc_sizes,
            rotmean_table_strides=mc_strides,
            rotameric_meansdev_tableset_offsets=rotameric_mean_offsets,
            n_rotamers_for_tableset=prob_table_nrots,
            n_rotamers_for_tableset_offsets=prob_table_offsets,
            sorted_rotamer_2_rotamer=sorted_rotamer_2_rotamer,
            rotameric_bb_start=rot_bb_start,
            rotameric_bb_step=rot_bb_step,
            rotameric_bb_periodicity=rot_bb_per,
            rotameric_rotind2tableind=rot_ri2ti,
            semirotameric_rotind2tableind=semirot_ri2ti,
            all_chi_rotind2tableind=allchi_ri2ti,
            all_chi_rotind2tableind_offsets=all_chi_rotind2tableind_offsets,
            semirotameric_tables=sr_coeffs,
            semirot_table_sizes=sr_sizes,
            semirot_table_strides=sr_strides,
            semirot_start=sr_start,
            semirot_step=sr_step,
            semirot_periodicity=sr_periodicity,
            nchi_for_table_set=nchi_for_table_set,
            rotwells=rotwells,
        )

        return cls(
            scoring_db=scoring_db,
            scoring_db_aux=scoring_db_aux,
            sampling_db=sampling_db,
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
        return prob_table_nrots, exclusive_cumsum1d(prob_table_nrots)

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
        """
        rotameric_rotind2tableind: a mapping based on the rotameric chi for a residue
           to the table index for that rotamer; the table index can then be used to
           access the rotamer probability, rotamer chi mean, and rotamer sdev
           tables
        semirotameric_rotind2tableind: a mapping based on the rotameric chi for a
           residue to the table index for that rotamer; the table index can then be
           used to access the semirotameric_tables table. Only valid for
           semirotameric residue types
        all_chi_rotind2tableind: a mapping based on both rotameric and non-rotameric
           chi for a residue to the table index for that rotamer; the table index
           can then be used to access the rotamer probability, rotamer chi mean,
           and rotamer sdev tables
        """

        rotameric_rotind2tableind = []
        semirotameric_rotind2tableind = []
        all_chi_rotind2tableind = []
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
            all_chi_rotind2tableind.extend(list(ri2ti))
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
            r_ri2ti[rotinds] = (
                n_nonrotameric_chi_rotamers
                * torch.arange(semirotameric_rotamers.shape[0])
            ).to(dtype=torch.int32)
            rotameric_rotind2tableind.extend(list(r_ri2ti))

            ac_ri2ti = torch.arange(rotameric_rotamers.shape[0], dtype=torch.int32)
            all_chi_rotind2tableind.extend(list(ac_ri2ti))

        rotameric_rotind2tableind = torch.tensor(
            rotameric_rotind2tableind, dtype=torch.int32, device=device
        ).reshape((-1,))

        semirotameric_rotind2tableind = torch.tensor(
            semirotameric_rotind2tableind, dtype=torch.int32, device=device
        ).reshape((-1,))

        all_chi_rotind2tableind = torch.tensor(
            all_chi_rotind2tableind, dtype=torch.int32, device=device
        )
        return (
            rotameric_rotind2tableind,
            semirotameric_rotind2tableind,
            all_chi_rotind2tableind,
        )

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
    def _create_all_chi_rotind2tableind_offsets(cls, dun_database, device):
        """
        Offsets into the all-chi rotind2tableind table
        """
        rotamer_counts_sets = [
            3 ** rotlib.rotameric_data.rotamers.shape[1]
            for rotlib in dun_database.rotameric_libraries
        ] + [
            rotlib.rotameric_data.rotamers.shape[0]
            for rotlib in dun_database.semi_rotameric_libraries
        ]

        return torch.cumsum(
            torch.tensor(
                [0] + rotamer_counts_sets[:-1], dtype=torch.int32, device=device
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

    @classmethod
    def create_sorted_rot_2_rot(cls, all_rotlibs, device):
        sorted_2_rotinds, _1, _2 = cat_differently_sized_tensors(
            [
                rot.rotameric_data.prob_sorted_rot_inds.permute(2, 0, 1).to(device)
                for rot in all_rotlibs
            ]
        )

        return sorted_2_rotinds.permute(1, 2, 0)

    @classmethod
    def create_rotamer_well_table(cls, all_rotlibs, device):
        rotwells, _1, _2 = cat_differently_sized_tensors(
            [rot.rotameric_data.rotamers.to(device) for rot in all_rotlibs]
        )
        return rotwells

    def _indices_from_names(
        self,
        dataframe: pandas.DataFrame,
        names: NDArray[object][:, :],
        device: torch.device,
    ) -> Tensor[torch.int64][:, :]:
        names_flat = names.ravel()
        inds = dataframe.index.get_indexer(names_flat)
        inds[inds != -1] = dataframe.iloc[inds[inds != -1]]["dun_table_name"].values
        inds = inds.reshape(names.shape)
        return torch.tensor(inds, dtype=torch.int64, device=device)

