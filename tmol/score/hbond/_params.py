import attr
from typing import Union

import numpy
import pandas
import torch

import toolz

from tmol.types import (
    Tensor,
    TensorGroup,
    ValidateAttrs,
    ConvertAttrs,
    validate_args,
)
from tmol.database.scoring import HBondDatabase
from tmol.database.chemical import ChemicalDatabase
from tmol.database import PatchedChemicalDatabase

from .._chemical_database import AcceptorHybridization
from tmol.utility.weak_identity_cache import WeakIdentityLRU


def _resolved_device(device):
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu") if device.type == "cpu" else device


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HBondPolyParams(TensorGroup, ConvertAttrs):
    range: Tensor[torch.double][..., 2]
    bound: Tensor[torch.double][..., 2]
    coeffs: Tensor[torch.double][..., 11]

    def __setitem__(self, idx, value):
        self.range[idx] = value.range[idx]
        self.bound[idx] = value.bound[idx]
        self.coeffs[idx] = value.coeffs[idx]

    def to(self, device: torch.device):
        return type(self)(
            **toolz.valmap(lambda t: t.to(device), attr.asdict(self, recurse=False))
        )

    @classmethod
    def full(cls, shape, fill_value):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls(
            range=torch.full(shape + (2,), fill_value),
            bound=torch.full(shape + (2,), fill_value),
            coeffs=torch.full(shape + (11,), fill_value),
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondPairParams(TensorGroup, ValidateAttrs):
    donor_weight: Tensor[torch.float32][...]
    acceptor_weight: Tensor[torch.float32][...]
    acceptor_hybridization: Tensor[torch.int32][...]
    AHdist: HBondPolyParams
    cosBAH: HBondPolyParams
    cosAHD: HBondPolyParams

    def to(self, device: torch.device):
        return type(self)(
            **toolz.valmap(lambda t: t.to(device), attr.asdict(self, recurse=False))
        )

    @classmethod
    def full(cls, shape, fill_value):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls(
            donor_weight=torch.full(shape, fill_value, dtype=torch.float),
            acceptor_weight=torch.full(shape, fill_value, dtype=torch.float),
            acceptor_hybridization=torch.full(
                shape, numpy.nan_to_num(fill_value), dtype=torch.int32
            ),  # nan_to_num fill value for integer dtype
            AHdist=HBondPolyParams.full(shape, fill_value),
            cosBAH=HBondPolyParams.full(shape, fill_value),
            cosAHD=HBondPolyParams.full(shape, fill_value),
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondParamResolver(ValidateAttrs):
    _from_db_cache = WeakIdentityLRU()

    donor_type_index: pandas.Index = attr.ib()
    acceptor_type_index: pandas.Index = attr.ib()

    pair_params: HBondPairParams = attr.ib()
    device: torch.device = attr.ib()

    @classmethod
    @validate_args
    def from_database(
        cls,
        chemical_database: Union[ChemicalDatabase, PatchedChemicalDatabase],
        hbond_database: HBondDatabase,
        device: torch.device,
    ):
        device = _resolved_device(device)
        return cls._from_db_cache.get_or_create_many(
            (chemical_database, hbond_database),
            device,
            lambda: cls._from_database(chemical_database, hbond_database, device),
        )

    @classmethod
    def _from_database(cls, chemical_database, hbond_database, device):
        donors = {g.name: g for g in hbond_database.donor_type_params}
        donor_type_index = pandas.Index(list(donors))

        acceptors = {g.name: g for g in hbond_database.acceptor_type_params}
        acceptor_type_index = pandas.Index(list(acceptors))

        atom_type_hybridization = {
            a.name: a.acceptor_hybridization for a in chemical_database.atom_types
        }
        acceptor_type_hybridization = {
            g.acceptor_type: atom_type_hybridization[g.a]
            for g in hbond_database.acceptor_atom_types
        }

        # Preserve the historical last-definition-wins rules while ordering the
        # dense table once. Repeated advanced-index assignment is unnecessary.
        poly_values = {
            p.name: [p.xmin, p.xmax, p.min_val, p.max_val]
            + [getattr(p, f"c_{n}") for n in range(10, -1, -1)]
            for p in hbond_database.polynomial_parameters
        }
        kinds = ("AHdist", "cosBAH", "cosAHD")
        pairs = {}
        for row in hbond_database.pair_parameters:
            if row.donor_type not in donors or row.acceptor_type not in acceptors:
                raise ValueError(
                    f"HBond pair references unknown family: "
                    f"{row.donor_type}, {row.acceptor_type}"
                )
            for kind in kinds:
                name = getattr(row, kind)
                if name not in poly_values:
                    raise ValueError(
                        f"HBond pair references unknown polynomial {name!r}"
                    )
            pairs[row.donor_type, row.acceptor_type] = row
        rows = []
        for donor in donors:
            for acceptor in acceptors:
                row = pairs.get((donor, acceptor))
                if row is None:
                    raise ValueError(
                        f"Missing HBond pair parameters: {donor}, {acceptor}"
                    )
                rows.append(row)
        shape = (len(donors), len(acceptors))

        def weights(records, kind):
            with numpy.errstate(over="ignore"):
                values = numpy.array(
                    [g.weight for g in records.values()], dtype=numpy.float32
                )
            invalid = ~numpy.isfinite(values)
            if invalid.any():
                names = [name for name, bad in zip(records, invalid) if bad]
                raise ValueError(
                    f"HBond {kind} weights {names} are non-finite in float32"
                )
            return values

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=device)

        # Gather on the host, then copy each final tensor once. Polynomial
        # coefficients retain double precision and descending-power order.
        def pack_polynomial(kind):
            names = [getattr(row, kind) for row in rows]
            values = numpy.array(
                [poly_values[name] for name in names], dtype=numpy.float64
            ).reshape(shape + (15,))
            invalid = ~numpy.isfinite(values).all(axis=-1).ravel()
            if invalid.any():
                names = list(
                    dict.fromkeys(name for name, bad in zip(names, invalid) if bad)
                )
                raise ValueError(
                    f"HBond polynomials {names} contain non-finite parameters"
                )
            return HBondPolyParams(
                range=tensor(values[..., :2], torch.float64),
                bound=tensor(values[..., 2:4], torch.float64),
                coeffs=tensor(values[..., 4:], torch.float64),
            )

        hybridization = AcceptorHybridization._index.get_indexer(
            [acceptor_type_hybridization[name] for name in acceptors]
        )
        if (hybridization < 0).any():
            raise ValueError("HBond acceptor family has unknown hybridization")
        pair_params = HBondPairParams(
            donor_weight=tensor(
                numpy.broadcast_to(weights(donors, "donor")[:, None], shape),
                torch.float32,
            ),
            acceptor_weight=tensor(
                numpy.broadcast_to(weights(acceptors, "acceptor")[None, :], shape),
                torch.float32,
            ),
            acceptor_hybridization=tensor(
                numpy.broadcast_to(hybridization[None, :], shape), torch.int32
            ),
            **{kind: pack_polynomial(kind) for kind in kinds},
        )

        return cls(
            donor_type_index=donor_type_index,
            acceptor_type_index=acceptor_type_index,
            pair_params=pair_params,
            device=device,
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CompactedHBondDatabase(ValidateAttrs):
    """Store the hbond evaluation parameters in a compact form"""

    _from_db_cache = WeakIdentityLRU()

    global_param_table: Tensor[torch.float32][:, :]
    pair_param_table: Tensor[torch.float32][:, :, :]
    pair_poly_table: Tensor[torch.float64][:, :, :]

    @classmethod
    @validate_args
    def from_database(
        cls,
        chemical_database: Union[ChemicalDatabase, PatchedChemicalDatabase],
        hbond_database: HBondDatabase,
        device: torch.device,
    ):
        device = _resolved_device(device)
        return cls._from_db_cache.get_or_create_many(
            (chemical_database, hbond_database),
            device,
            lambda: cls._from_database(chemical_database, hbond_database, device),
        )

    @classmethod
    def _from_database(cls, chemical_database, hbond_database, device):
        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _stack(ts):
            return torch.cat([t.unsqueeze(-1) for t in ts], -1)

        def _f32(ts):
            return [t.to(torch.float32) for t in ts]

        global_params = {
            n: torch.tensor(v, device=device).expand(1).to(dtype=torch.float32)
            for n, v in attr.asdict(hbond_database.global_parameters).items()
        }
        resolver = HBondParamResolver.from_database(
            chemical_database, hbond_database, device
        )
        distances = [
            p.xmax
            for p in hbond_database.polynomial_parameters
            if p.dimension == "hbgd_AHdist"
        ]
        if not distances and resolver.pair_params.donor_weight.numel():
            raise ValueError("HBond pair table has no AHdist polynomial range")
        max_ahdis = max(distances, default=0.0)

        # also store the distance of the longest possible hbond
        # by reading the set of hbond polynomials
        global_param_table = _p(
            (
                torch.cat(
                    [
                        global_params["hb_sp2_range_span"],
                        global_params["hb_sp2_BAH180_rise"],
                        global_params["hb_sp2_outer_width"],
                        global_params["hb_sp3_softmax_fade"],
                        global_params["threshold_distance"],
                        torch.tensor([max_ahdis], device=device, dtype=torch.float32),
                    ],
                    0,
                )
            ).unsqueeze(0)
        )

        pp = resolver.pair_params

        # Note: acceptor_hybridization is an integer, but can be exactly represented
        # as a float (since it is small). Pack it with the donor and acceptor weights
        # (which themselves could just be pre-multiplied?!) to reduce the number of
        # arguments to the hbond evaluation function
        pair_param_table = _p(
            _stack(
                _f32([pp.acceptor_hybridization, pp.acceptor_weight, pp.donor_weight])
            )
        )

        # Eigen allocates space for 12 Reals for an 11x1 matrix
        # so we need to pad out with 0s in between the coefficients
        # and the ranges.
        pad = torch.zeros(
            [pp.AHdist.coeffs.shape[0], pp.AHdist.coeffs.shape[1], 1],
            device=device,
            dtype=torch.float64,
        )

        pair_poly_table = _p(
            torch.cat(
                [
                    pp.AHdist.coeffs,
                    pad,
                    pp.AHdist.range,
                    pp.AHdist.bound,
                    pp.cosBAH.coeffs,
                    pad,
                    pp.cosBAH.range,
                    pp.cosBAH.bound,
                    pp.cosAHD.coeffs,
                    pad,
                    pp.cosAHD.range,
                    pp.cosAHD.bound,
                ],
                2,
            )
        )

        return cls(
            global_param_table=global_param_table,
            pair_param_table=pair_param_table,
            pair_poly_table=pair_poly_table,
        )
