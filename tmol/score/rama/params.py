import attr
import cattr

import numpy
import pandas
import torch

import toolz.functoolz

from typing import List

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.numeric.bspline import BSplineInterpolation

from tmol.database.scoring.rama import RamaDatabase


# the rama database on the device
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedRamaDatabase(ConvertAttrs):
    tables: List  # mapped to C++ via TCollection
    bbsteps: Tensor(torch.float)[...]
    bbstarts: Tensor(torch.float)[...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class RamaParamResolver(ValidateAttrs):
    _from_rama_db_cache = {}

    # respair -> table index mapping
    rama_lookup: pandas.DataFrame

    # array of tables
    rama_params: PackedRamaDatabase

    device: torch.device

    def resolve_ramatables(
        self, r1: NDArray(object), r2: NDArray(object)
    ) -> NDArray("i8")[...]:
        l_idx = self.rama_lookup.index.get_indexer([r1, r2])
        wildcard = numpy.full_like(r1, "_")
        l_idx[l_idx == -1] = self.rama_lookup.index.get_indexer(
            [r1[l_idx == -1], wildcard[l_idx == -1]]
        )
        t_idx = self.rama_lookup.iloc[l_idx, :]["table_id"].values
        return t_idx

    @classmethod
    @validate_args
    @toolz.functoolz.memoize(
        cache=_from_rama_db_cache,
        key=lambda args, kwargs: (args[1].uniq_id, args[2].type, args[2].index),
    )
    def from_database(cls, rama_database: RamaDatabase, device: torch.device):
        # setup name to index mapping
        rama_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(rama_database.rama_lookup)
        ).set_index(["res_middle", "res_upper"])
        tindices = pandas.Index([f.table_id for f in rama_database.rama_tables])

        # map table names to indices
        rama_lookup.table_id = tindices.get_indexer(rama_lookup.table_id)

        rama_params = PackedRamaDatabase(
            # interpolate on CPU then move coeffs to CUDA
            tables=[
                BSplineInterpolation.from_coordinates(
                    torch.tensor(f.table, dtype=torch.float)
                ).coeffs.to(device=device)
                for f in rama_database.rama_tables
            ],
            bbsteps=torch.tensor(
                [f.bbstep for f in rama_database.rama_tables],
                dtype=torch.float,
                device=device,
            ),
            bbstarts=torch.tensor(
                [f.bbstart for f in rama_database.rama_tables],
                dtype=torch.float,
                device=device,
            ),
        )

        return cls(rama_lookup=rama_lookup, rama_params=rama_params, device=device)
