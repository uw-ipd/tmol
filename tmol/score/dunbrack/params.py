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

    rotameric_prob_tableset_offsets: List
    rotameric_meansdev_tableset_offsets: List

    rotameric_bb_start: List
    rotameric_bb_step: List
    rotameric_bb_periodicity: List

    semirotameric_tables: List
    semirotameric_tableeset_offsets: List

    semirot_start: List
    semirot_step: List
    semirot_periodicity: List

    rotind2tableind: List


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DunbrackParamResolver(ValidateAttrs):
    _from_dun_db_cache = {}

    dun_indices: pandas.Index
    dun_params: PackedDunbrackDatabase

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
    def from_database(
        cls, dun_database: DunbrackRotamerLibrary
    ):  # , device: torch.device
        all_rotlibs = [
            rotlib
            for rotlib in itertools.chain(
                dun_database.rotameric_libraries, dun_database.semi_rotameric_libraries
            )
        ]

        names = [x.table_name for x in all_rotlibs]
        print(names)
        dun_records = pandas.DataFrame.from_records(
            cattr.unstructure(dun_database.dun_lookup)
        ).set_index("dun_table_name")
        # .reindex( names )
        print()
        print("names", names)
        print("dun_records", dun_records)

        # dun_indices = pandas.Index(rama_records[["res_middle", "res_upper"]])???

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
        prob_table_offsets = torch.cumsum(
            torch.tensor(prob_table_nrots, dtype=torch.long), 0
        )
        print("prob_table_offsets", prob_table_offsets)

        prob_coeffs = [
            BSplineInterpolation.from_coordinates(t) for t in rotameric_prob_tables
        ]

        rotameric_mean_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_means[i, j])
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_means.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_means.shape[1])
        ]

        mean_table_n_entries = [0] + [
            rotlib.rotameric_data.rotamer_means.shape[0]
            * rotlib.rotameric_data.rotamer_means.shape[1]
            for rotlib in all_rotlibs
        ][:-1]
        rotameric_mean_offsets = torch.cumsum(
            torch.tensor(mean_table_n_entries, dtype=torch.long), 0
        )
        print(rotameric_mean_offsets)

        rotameric_sdev_tables = [
            torch.tensor(rotlib.rotameric_data.rotamer_probabilities[i,])
            for rotlib in all_rotlibs
            for i in range(rotlib.rotameric_data.rotamer_stdvs.shape[0])
            for j in range(rotlib.rotameric_data.rotamer_stdvs.shape[1])
        ]

        mean_coeffs = [
            BSplineInterpolation.from_coordinates(t) for t in rotameric_mean_tables
        ]

        sdev_coeffs = [
            BSplineInterpolation.from_coordinates(t) for t in rotameric_sdev_tables
        ]

        print("len(prob_coeffs)", len(prob_coeffs))
        print("len(mean_coeffs)", len(mean_coeffs))
        print("len(sdev_coeffs)", len(sdev_coeffs))

        rotameric_bb_start = [
            rotlib.rotameric_data.backbone_dihedral_start for rotlib in all_rotlibs
        ]
        rotameric_bb_step = [
            rotlib.rotameric_data.backbone_dihedral_step for rotlib in all_rotlibs
        ]

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
            torch.tensor(nsemirot_rotamers, dtype=torch.long), 0
        )
        semirotameric_prob_tables = [
            torch.tensor(rotlib.nonrotameric_chi_probabilities[i,])
            for rotlib in dun_database.semi_rotameric_libraries
            for i in range(rotlib.nonrotameric_chi_probabilities.shape[0])
        ]
        print("len(semirotameric_prob_tables)", len(semirotameric_prob_tables))

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
        # return cls(rama_indices=rama_indices, rama_params=rama_params, device=device)
