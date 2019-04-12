import attr
import cattr

import numpy
import pandas
import torch

import toolz.functoolz

from tmol.types.array import NDArray
from tmol.types.torch import Tensor, TensorCollection
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.numeric.bspline import BSplineInterpolation

from tmol.database.scoring.rama import RamaDatabase
from tmol.utility.tensor.compiled import create_tensor_collection

# the rama database on the device
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedRamaDatabase(ConvertAttrs):
    tables: TensorCollection(torch.float)[:, :]
    bbsteps: Tensor(torch.float)[...]
    bbstarts: Tensor(torch.float)[...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class RamaParamResolver(ValidateAttrs):
    _from_rama_db_cache = {}

    rama_indices: pandas.Index
    rama_params: PackedRamaDatabase

    device: torch.device

    def resolve_ramatables(
        self, resnames0: NDArray(object), resnames1: NDArray(object)
    ) -> NDArray("i8")[...]:
        indices = self.rama_indices.get_indexer([resnames0, resnames1])
        wildcard = numpy.full_like(resnames1, "_")
        indices[indices == -1] = self.rama_indices.get_indexer(
            [resnames0[indices == -1], wildcard[indices == -1]]
        )
        return indices

    @classmethod
    @validate_args
    @toolz.functoolz.memoize(
        cache=_from_rama_db_cache,
        key=lambda args, kwargs: (args[1].uniq_id, args[2].type, args[2].index),
    )
    def from_database(cls, rama_database: RamaDatabase, device: torch.device):
        # build name->index mapping
        rama_records = (
            pandas.DataFrame.from_records(cattr.unstructure(rama_database.rama_lookup))
            .set_index("name")
            .reindex([x.name for x in rama_database.rama_tables])
        )
        rama_indices = pandas.Index(rama_records[["res_middle", "res_upper"]])

        rama_params = PackedRamaDatabase(
            # interpolate on CPU then move coeffs to CUDA
            tables=create_tensor_collection(
                [
                    BSplineInterpolation.from_coordinates(
                        torch.tensor(f.table, dtype=torch.float)
                    ).coeffs.to(device=device)
                    for f in rama_database.rama_tables
                ]
            ),
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

        return cls(rama_indices=rama_indices, rama_params=rama_params, device=device)
