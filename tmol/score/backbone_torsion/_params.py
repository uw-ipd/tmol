import attr
import cattr

import pandas
import torch

import toolz.functoolz

from tmol.types import (
    Tensor,
    ValidateAttrs,
    ConvertAttrs,
    validate_args,
)
from tmol.numeric import BSplineInterpolation

from tmol.database.scoring import (
    RamaDatabase,
    OmegaBBDepDatabase,
)


def mirror_grid(table, bbstart, bbstep):
    """Reflect a periodic 2-D torsion grid through the origin.

    Bin k covers start + k*step, so the bin holding -x is (c - k) modulo the
    number of bins, with c = -2*start/step. The two backbone grids are
    registered differently -- rama starts at -180 and omega at -175 -- so the
    reflection is derived from the grid rather than assumed.
    """
    out = table
    for axis, (start, step) in enumerate(zip(bbstart, bbstep)):
        n = out.shape[axis]
        c = int(round(-2.0 * float(start) / float(step)))
        out = torch.roll(torch.flip(out, (axis,)), (c - n + 1) % n, dims=axis)
    return out


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

    # tables before the mirrored copies; table i has its mirror at i + n
    n_rama_tables: int
    n_omega_tables: int

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

        # interpolate spline tables; every table is followed by its mirror
        #    image at index i + ntables, which a d-amino acid selects so that
        #    its l counterpart's statistics are read at negated phi/psi
        ntables = len(rama_database.rama_tables)
        tablesize = rama_database.rama_tables[0].table.shape
        tables = torch.empty((2 * ntables, *tablesize))

        for i, t_i in enumerate(rama_database.rama_tables):
            raw = t_i.table.detach().clone().to(dtype=torch.float)
            tables[i, ...] = BSplineInterpolation.from_coordinates(raw).coeffs
            tables[i + ntables, ...] = BSplineInterpolation.from_coordinates(
                mirror_grid(raw, t_i.bbstart, t_i.bbstep)
            ).coeffs

        rama_params = PackedRamaDatabase(
            # interpolate on CPU then move coeffs to GPU
            tables=tables.to(device=device),
            bbsteps=torch.tensor(
                [f.bbstep for f in rama_database.rama_tables] * 2,
                dtype=torch.float,
                device=device,
            ),
            bbstarts=torch.tensor(
                [f.bbstart for f in rama_database.rama_tables] * 2,
                dtype=torch.float,
                device=device,
            ),
        )
        n_rama_tables = ntables

        ## OMEGA
        omega_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(bbdep_omega_database.bbdep_omega_lookup)
        ).set_index(["res_middle", "res_upper"])
        tindices = pandas.Index(
            [f.table_id for f in bbdep_omega_database.bbdep_omega_tables]
        )

        # map table names to indices
        omega_lookup.table_id = tindices.get_indexer(omega_lookup.table_id)

        # interpolate spline tables, mirrored copies second as for rama.
        #    omega is a gaussian about mu, and the mirror sends omega -> -omega,
        #    so the mirrored mean is -mu read at negated phi/psi
        ntables = len(bbdep_omega_database.bbdep_omega_tables)
        tablesize = rama_database.rama_tables[0].table.shape
        tables = torch.empty((2 * ntables, 2, *tablesize))

        for i, t_i in enumerate(bbdep_omega_database.bbdep_omega_tables):
            mu = t_i.mu.detach().clone().to(dtype=torch.float)
            sigma = t_i.sigma.detach().clone().to(dtype=torch.float)
            tables[i, 0, ...] = BSplineInterpolation.from_coordinates(mu).coeffs
            tables[i, 1, ...] = BSplineInterpolation.from_coordinates(sigma).coeffs
            tables[i + ntables, 0, ...] = BSplineInterpolation.from_coordinates(
                360.0 - mirror_grid(mu, t_i.bbstart, t_i.bbstep)
            ).coeffs
            tables[i + ntables, 1, ...] = BSplineInterpolation.from_coordinates(
                mirror_grid(sigma, t_i.bbstart, t_i.bbstep)
            ).coeffs

        # assumes bbstep is the same for both tables
        omega_params = PackedOmegaDatabase(
            tables=tables.to(device=device),
            bbsteps=torch.tensor(
                [f.bbstep for f in bbdep_omega_database.bbdep_omega_tables] * 2,
                dtype=torch.float,
                device=device,
            ),
            bbstarts=torch.tensor(
                [f.bbstart for f in bbdep_omega_database.bbdep_omega_tables] * 2,
                dtype=torch.float,
                device=device,
            ),
        )

        return cls(
            rama_lookup=rama_lookup,
            rama_params=rama_params,
            omega_lookup=omega_lookup,
            omega_params=omega_params,
            n_rama_tables=n_rama_tables,
            n_omega_tables=ntables,
            device=device,
        )
