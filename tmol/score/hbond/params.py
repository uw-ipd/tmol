import attr

from typing import Sequence

import numpy
import pandas
import torch

import toolz

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.hbond import HBondDatabase
from tmol.database.chemical import ChemicalDatabase

from ..chemical_database import AcceptorHybridization


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HBondPolyParams(TensorGroup, ConvertAttrs):
    range: Tensor(torch.double)[..., 2]
    bound: Tensor(torch.double)[..., 2]
    coeffs: Tensor(torch.double)[..., 11]

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
    donor_weight: Tensor("f")[...]
    acceptor_weight: Tensor("f")[...]
    acceptor_hybridization: Tensor("i4")[...]
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
    _from_db_cache = {}

    donor_type_index: pandas.Index = attr.ib()
    acceptor_type_index: pandas.Index = attr.ib()

    pair_params: HBondPairParams = attr.ib()
    device: torch.device = attr.ib()

    def resolve_donor_type(self, donor_types: Sequence[str]) -> torch.Tensor:
        """Resolve string donor type name into integer type index."""
        i = self.donor_type_index.get_indexer(donor_types)
        assert not numpy.any(i == -1), "donor type not present in index"
        return torch.from_numpy(i).to(device=self.device)

    def resolve_acceptor_type(self, acceptor_types: Sequence[str]) -> torch.Tensor:
        """Resolve string acceptor type name into integer type index."""
        i = self.acceptor_type_index.get_indexer(acceptor_types)
        assert not numpy.any(i == -1), "acceptor type not present in index"
        return torch.from_numpy(i).to(device=self.device)

    @classmethod
    @validate_args
    @toolz.functoolz.memoize(
        cache=_from_db_cache,
        key=lambda args, kwargs: (
            id(args[1]),
            id(args[2]),
            args[3].type,
            args[3].index,
        ),
    )
    def from_database(
        cls,
        chemical_database: ChemicalDatabase,
        hbond_database: HBondDatabase,
        device: torch.device,
    ):

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

        pair_params = HBondPairParams.full((len(donors), len(acceptors)), numpy.nan)

        # Denormalize donor/acceptor weight and class into pair parameter table
        for name, g in donors.items():
            i, = donor_type_index.get_indexer([name])
            pair_params.donor_weight[i, :] = g.weight

        for name, g in acceptors.items():
            i, = acceptor_type_index.get_indexer([name])
            pair_params.acceptor_weight[:, i] = g.weight
            pair_params.acceptor_hybridization[:, i] = int(
                AcceptorHybridization._index.get_indexer_for(
                    [acceptor_type_hybridization[name]]
                )
            )

        # Get polynomial parameters indexed by polynomial name
        poly_params = HBondPolyParams(
            **toolz.merge_with(
                numpy.vstack,
                [
                    {
                        "range": [p.xmin, p.xmax],
                        "bound": [p.min_val, p.max_val],
                        "coeffs": [getattr(p, "c_" + i) for i in "abcdefghijk"],
                    }
                    for p in hbond_database.polynomial_parameters
                ],
            )
        )

        poly_params = {
            p.name: poly_params[i]
            for i, p in enumerate(hbond_database.polynomial_parameters)
        }

        # Denormalize polynomial parameters into pair parameter table
        for pp in hbond_database.pair_parameters:
            di, = donor_type_index.get_indexer([pp.donor_type])
            assert di >= 0

            ai, = acceptor_type_index.get_indexer([pp.acceptor_type])
            assert ai >= 0

            pair_params[di, ai].AHdist[:] = poly_params[pp.AHdist]
            pair_params[di, ai].cosBAH[:] = poly_params[pp.cosBAH]
            pair_params[di, ai].cosAHD[:] = poly_params[pp.cosAHD]

        return cls(
            donor_type_index=donor_type_index,
            acceptor_type_index=acceptor_type_index,
            pair_params=pair_params.to(device),
            device=device,
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CompactedHBondDatabase(ValidateAttrs):
    _from_db_cache = {}

    global_param_table: Tensor(torch.float32)[:, :]
    pair_param_table: Tensor(torch.float32)[:, :, :]
    pair_poly_table: Tensor(torch.float64)[:, :, :]

    @classmethod
    @validate_args
    @toolz.functoolz.memoize(
        cache=_from_db_cache,
        key=lambda args, kwargs: (
            id(args[1]),
            id(args[2]),
            args[3].type,
            args[3].index,
        ),
    )
    def from_database(
        cls,
        chemical_database: ChemicalDatabase,
        hbond_database: HBondDatabase,
        device: torch.device,
    ):
        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tcat(ts, dim):
            return torch.cat(ts, dim)

        def _stack(ts):
            return _tcat([t.unsqueeze(-1) for t in ts], -1)

        def _f32(ts):
            return [t.to(torch.float32) for t in ts]

        global_params = {
            n: torch.tensor(v, device=device).expand(1).to(dtype=torch.float32)
            for n, v in attr.asdict(hbond_database.global_parameters).items()
        }

        global_param_table = _p(
            (
                torch.cat(
                    [
                        global_params["hb_sp2_range_span"],
                        global_params["hb_sp2_BAH180_rise"],
                        global_params["hb_sp2_outer_width"],
                        global_params["hb_sp3_softmax_fade"],
                        global_params["threshold_distance"],
                    ],
                    0,
                )
            ).unsqueeze(0)
        )
        # print()
        # print("global_param_table", global_param_table.size())

        resolver = HBondParamResolver.from_database(
            chemical_database, hbond_database, device
        )
        pp = resolver.pair_params

        pair_param_table = _p(
            _stack(
                _f32([pp.acceptor_hybridization, pp.acceptor_weight, pp.donor_weight])
            )
        )
        # print("Pair param table", pair_param_table.shape)
        pad = torch.zeros(
            [pp.AHdist.coeffs.shape[0], pp.AHdist.coeffs.shape[1], 1],
            device=device,
            dtype=torch.float64,
        )

        pair_poly_table = _p(
            _tcat(
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
        # print("Pair poly table", pair_poly_table.shape)

        return cls(
            global_param_table=global_param_table,
            pair_param_table=pair_param_table,
            pair_poly_table=pair_poly_table,
        )
