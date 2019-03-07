import attr
import cattr

import numpy
import pandas
import torch

import toolz.functoolz
import itertools

from typing import List

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

    def resolve_ramatables(
        self, resnames0: NDArray(object), resnames1: NDArray(object)
    ) -> NDArray("i8")[...]:
        indices = rama_index.get_indexer(resnames0, resnames1)
        wildcard = numpy.full_like(resnames1, "_")
        indices[indices == -1] = rama_index.get_indexer(
            [resnames0[indices == -1], wildcard]
        )
        return indices

    @classmethod
    # @validate_args
    # @toolz.functoolz.memoize(
    #    cache=_from_dun_db_cache,
    #    key=lambda args, kwargs: (args[1], args[2].type, args[2].index),
    # )
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

        # print("all_table_names", all_table_names)
        # print("rotameric_table_names", rotameric_table_names)
        # print("semirotameric_table_names", semirotameric_table_names)
        # example_residues = ["ARG", "PHE", "ALA", "LEU"]
        # print("all", all_table_indices.get_indexer(example_residues))
        # print("rot", rotameric_table_indices.get_indexer(example_residues))
        # print("sem", semirotameric_table_indices.get_indexer(example_residues))

        rotameric_prob_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_probabilities[i,])
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_probabilities.shape[0])
        ]

        prob_table_name_and_nrots = [
            (rotlib.table_name, rotlib.rotameric_data.rotamer_probabilities.shape[0])
            for rotlib in all_rotlibs
        ]
        print(prob_table_name_and_nrots)

        prob_table_nrots = [0] + [
            rotlib.rotameric_data.rotamer_probabilities.shape[0]
            for rotlib in all_rotlibs
        ][:-1]
        print("prob_table_nrots", prob_table_nrots)
        nchi_for_table_set = torch.tensor(
            [rotlib.rotameric_data.rotamers.shape[1] for rotlib in all_rotlibs],
            dtype=torch.long,
            device=device,
        )

        prob_table_offsets = torch.cumsum(
            torch.tensor(prob_table_nrots, dtype=torch.long, device=device), 0
        )
        print("prob_table_offsets", prob_table_offsets)

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
        print(rotameric_mean_offsets)

        for rotlib in all_rotlibs:
            print(
                "rotlib.rotameric_data.rotamer_stdvs.shape",
                rotlib.rotameric_data.rotamer_stdvs.shape,
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

        print("len(prob_coeffs)", len(prob_coeffs))
        print("len(mean_coeffs)", len(mean_coeffs))
        print("len(sdev_coeffs)", len(sdev_coeffs))

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
        print("rotind2tableind", rotind2tableind)

        rotind2tableind_offsets = torch.cumsum(
            torch.tensor(
                [0] + [3 ** rotamers.shape[1] for rotamers in rotamer_sets][:-1],
                dtype=torch.long,
                device=device,
            ),
            0,
        )
        print("rotind2tableind_offsets", rotind2tableind_offsets)

        # ======

        print(
            "len(dun_database.semi_rotameric_libraries)",
            len(dun_database.semi_rotameric_libraries),
        )
        nsemirot_rotamers = [0] + [
            rotlib.nonrotameric_chi_probabilities.shape[0]
            for rotlib in dun_database.semi_rotameric_libraries
        ][:-1]
        print("nsemirot_rotamers", nsemirot_rotamers)
        semirotameric_tableset_offsets = torch.cumsum(
            torch.tensor(nsemirot_rotamers, dtype=torch.long, device=device), 0
        )
        semirotameric_prob_tables = [
            torch.tensor(rotlib.nonrotameric_chi_probabilities[i,])
            for rotlib in dun_database.semi_rotameric_libraries
            for i in range(rotlib.nonrotameric_chi_probabilities.shape[0])
        ]
        print("len(semirotameric_prob_tables)", len(semirotameric_prob_tables))

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
