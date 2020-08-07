import attr
import cattr

import numpy
import pandas
import torch

import toolz.functoolz

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.numeric.bspline import BSplineInterpolation

from tmol.database.scoring.rama import RamaDatabase
from tmol.types.tensor import TensorGroup


# rama parameters
@attr.s(auto_attribs=True)
class RamaParams(TensorGroup):
    phi_indices: Tensor(torch.int32)[..., 4]
    psi_indices: Tensor(torch.int32)[..., 4]
    param_indices: Tensor(torch.int32)[...]


# the rama database on the device
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedRamaDatabase(ConvertAttrs):
    tables: Tensor(torch.float)[:, :, :]
    bbsteps: Tensor(torch.float)[:, :]
    bbstarts: Tensor(torch.float)[:, :]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class RamaParamResolver(ValidateAttrs):
    _from_rama_db_cache = {}

    # respair -> table index mapping
    rama_lookup: pandas.DataFrame

    # rama tables (spline coeffs)
    rama_params: PackedRamaDatabase

    device: torch.device

    def resolve_ramatables(
        self, r1: NDArray(object), r2: NDArray(object)
    ) -> NDArray(numpy.long)[...]:
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

        # interpolate spline tables
        ntables = len(rama_database.rama_tables)
        assert ntables > 0
        tablesize = rama_database.rama_tables[0].table.shape
        tables = torch.empty((ntables, *tablesize))
        for i, t_i in enumerate(rama_database.rama_tables):
            tables[i, ...] = BSplineInterpolation.from_coordinates(
                torch.tensor(t_i.table, dtype=torch.float)
            ).coeffs

        rama_params = PackedRamaDatabase(
            # interpolate on CPU then move coeffs to GPU
            tables=tables.to(device=device),
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
