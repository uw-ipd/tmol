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
from tmol.database.scoring.omega_bbdep import OmegaBBDepDatabase
from tmol.types.tensor import TensorGroup
from tmol.chemical.restypes import uaid_t


# the rama database packed into a single tensor
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedRamaDatabase(ConvertAttrs):
    tables: Tensor[torch.float][:, :, :]
    bbsteps: Tensor[torch.float][:, :]
    bbstarts: Tensor[torch.float][:, :]


# the omega bb-dep database packed into a single tensor
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedOmegaDatabase(ConvertAttrs):
    tables: Tensor[torch.float][:, 2, :, :]
    bbsteps: Tensor[torch.float][:, :]
    bbstarts: Tensor[torch.float][:, :]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class BackboneTorsionParamResolver(ValidateAttrs):
    _from_rama_db_cache = {}

    # respair -> table index mapping
    rama_lookup: pandas.DataFrame
    omega_lookup: pandas.DataFrame

    # rama tables (spline coeffs)
    rama_params: PackedRamaDatabase
    omega_params: PackedOmegaDatabase

    device: torch.device

    @classmethod
    @validate_args
    @toolz.functoolz.memoize(
        cache=_from_rama_db_cache,
        key=lambda args, kwargs: (
            args[1].uniq_id,
            args[2].uniq_id,
            args[3].type,
            args[3].index,
        ),
    )
    def from_database(
        cls,
        rama_database: RamaDatabase,
        bbdep_omega_database: OmegaBBDepDatabase,
        device: torch.device,
    ):
        ## RAMA
        # setup name to index mapping
        rama_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(rama_database.rama_lookup)
        ).set_index(["res_middle", "res_upper"])
        tindices = pandas.Index([f.table_id for f in rama_database.rama_tables])

        # map table names to indices
        rama_lookup.table_id = tindices.get_indexer(rama_lookup.table_id)

        # interpolate spline tables
        ntables = len(rama_database.rama_tables)
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

        ## OMEGA
        omega_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(bbdep_omega_database.bbdep_omega_lookup)
        ).set_index(["res_middle", "res_upper"])
        tindices = pandas.Index(
            [f.table_id for f in bbdep_omega_database.bbdep_omega_tables]
        )

        # map table names to indices
        omega_lookup.table_id = tindices.get_indexer(omega_lookup.table_id)

        # interpolate spline tables
        ntables = len(bbdep_omega_database.bbdep_omega_tables)
        tablesize = rama_database.rama_tables[0].table.shape
        tables = torch.empty((ntables, 2, *tablesize))

        for i, t_i in enumerate(bbdep_omega_database.bbdep_omega_tables):
            tables[i, 0, ...] = BSplineInterpolation.from_coordinates(
                torch.tensor(t_i.mu, dtype=torch.float)
            ).coeffs
            tables[i, 1, ...] = BSplineInterpolation.from_coordinates(
                torch.tensor(t_i.sigma, dtype=torch.float)
            ).coeffs

        # assumes bbstep is the same for both tables
        omega_params = PackedOmegaDatabase(
            tables=tables.to(device=device),
            bbsteps=torch.tensor(
                [f.bbstep for f in bbdep_omega_database.bbdep_omega_tables],
                dtype=torch.float,
                device=device,
            ),
            bbstarts=torch.tensor(
                [f.bbstart for f in bbdep_omega_database.bbdep_omega_tables],
                dtype=torch.float,
                device=device,
            ),
        )

        return cls(
            rama_lookup=rama_lookup,
            rama_params=rama_params,
            omega_lookup=omega_lookup,
            omega_params=omega_params,
            device=device,
        )
