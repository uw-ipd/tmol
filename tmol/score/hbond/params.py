import attr

from typing import Sequence

import numpy
import pandas
import torch

import toolz

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs

from tmol.database.scoring.hbond import HBondDatabase


# static mapping to equivalent index in compiled potentials.
acceptor_class_index: pandas.Index = pandas.Index(["sp2", "sp3", "ring"])


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
    acceptor_class: Tensor("i4")[...]
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
            acceptor_class=torch.full(
                shape, numpy.nan_to_num(fill_value), dtype=torch.int32
            ),  # nan_to_num fill value for integer dtype
            AHdist=HBondPolyParams.full(shape, fill_value),
            cosBAH=HBondPolyParams.full(shape, fill_value),
            cosAHD=HBondPolyParams.full(shape, fill_value),
        )


@attr.s(frozen=True, slots=True)
class HBondParamResolver(ValidateAttrs):
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
    def from_database(cls, hbond_database: HBondDatabase, device: torch.device):

        donors = {g.name: g for g in hbond_database.donor_type_params}
        donor_type_index = pandas.Index(list(donors))

        acceptors = {g.name: g for g in hbond_database.acceptor_type_params}
        acceptor_type_index = pandas.Index(list(acceptors))

        pair_params = HBondPairParams.full((len(donors), len(acceptors)), numpy.nan)

        # Denormalize donor/acceptor weight and class into pair parameter table
        for name, g in donors.items():
            i, = donor_type_index.get_indexer([name])
            pair_params.donor_weight[i, :] = g.weight

        for name, g in acceptors.items():
            i, = acceptor_type_index.get_indexer([name])
            pair_params.acceptor_weight[:, i] = g.weight
            pair_params.acceptor_class[:, i] = int(
                acceptor_class_index.get_indexer_for([g.hybridization])
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
