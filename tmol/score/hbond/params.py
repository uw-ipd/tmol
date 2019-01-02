import attr
import cattr

from typing import Sequence

from enum import IntEnum

import numpy
import pandas
import torch

import toolz

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs

from tmol.database.scoring.hbond import HBondDatabase


class AcceptorClass(IntEnum):
    # Integer mapping from symbolic acceptor types int flag.
    # TODO resolve from compiled layer.
    sp2 = 0
    sp3 = 1
    ring = 2


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HBondPolyParams(TensorGroup, ConvertAttrs):
    range: Tensor(torch.double)[..., 2]
    bound: Tensor(torch.double)[..., 2]
    coeffs: Tensor(torch.double)[..., 11]

    def __setitem__(self, idx, value):
        self.range[idx] = value.range[idx]
        self.bound[idx] = value.bound[idx]
        self.coeffs[idx] = value.coeffs[idx]

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

    def resolve_donor_type(self, donor_types: Sequence[str]) -> torch.Tensor:
        """Resolve string donor type name into integer type index."""
        i = self.donor_type_index.get_indexer(donor_types)
        assert not numpy.any(i == -1), "donor type not present in index"
        return torch.from_numpy(i).to(device=self.pair_params.acceptor_class.device)

    def resolve_acceptor_type(self, acceptor_types: Sequence[str]) -> torch.Tensor:
        """Resolve string acceptor type name into integer type index."""
        i = self.acceptor_type_index.get_indexer(acceptor_types)
        assert not numpy.any(i == -1), "acceptor type not present in index"
        return torch.from_numpy(i).to(device=self.pair_params.acceptor_class.device)

    @classmethod
    def from_database(cls, hbond_database: HBondDatabase):
        atom_groups = hbond_database.atom_groups

        donors = list(set(g.donor_type for g in atom_groups.donors))

        acceptor_classes = (
            [(g.acceptor_type, AcceptorClass.sp2) for g in atom_groups.sp2_acceptors]
            + [(g.acceptor_type, AcceptorClass.sp3) for g in atom_groups.sp3_acceptors]
            + [
                (g.acceptor_type, AcceptorClass.ring)
                for g in atom_groups.ring_acceptors
            ]
        )
        assert len(set(acceptor_classes)) == len(
            set(dict(acceptor_classes).keys())
        ), "acceptor type with multiple hybridization classes defined."

        acceptor_classes = dict(acceptor_classes)
        acceptors = list(acceptor_classes.keys())

        donor_type_index = pandas.Index(donors)

        donor_weights = torch.Tensor(
            (
                pandas.DataFrame.from_records(
                    cattr.unstructure(hbond_database.don_weights)
                )
                .set_index("name")["weight"]
                .reindex(donor_type_index)
                .values
            )
        )

        acceptor_type_index = pandas.Index(acceptors)

        acceptor_weights = torch.Tensor(
            (
                pandas.DataFrame.from_records(
                    cattr.unstructure(hbond_database.acc_weights)
                )
                .set_index("name")["weight"]
                .reindex(acceptor_type_index)
            )
        )

        # Get polynomial parameters index by polynomial name
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

        pair_params = HBondPairParams.full((len(donors), len(acceptors)), numpy.nan)

        for pp in hbond_database.pair_parameters:
            di, = donor_type_index.get_indexer([pp.don_chem_type])
            assert di >= 0

            ai, = acceptor_type_index.get_indexer([pp.acc_chem_type])
            assert ai >= 0

            pair_params[di, ai].AHdist[:] = poly_params[pp.AHdist]
            pair_params[di, ai].cosBAH[:] = poly_params[pp.cosBAH]
            pair_params[di, ai].cosAHD[:] = poly_params[pp.cosAHD]
            pair_params[di, ai].donor_weight.reshape(1)[:] = donor_weights[di]
            pair_params[di, ai].acceptor_weight.reshape(1)[:] = acceptor_weights[ai]
            pair_params[di, ai].acceptor_class.reshape(1)[:] = acceptor_classes[
                pp.acc_chem_type
            ]

        return cls(
            donor_type_index=donor_type_index,
            acceptor_type_index=acceptor_type_index,
            pair_params=pair_params,
        )
