import attr
from attr import asdict
from typing import Mapping, Callable

from .params import RamaDatabase, CartBondedParamResolver

import torch


@attr.s(auto_attribs=True, frozen=True)
class RamaOp:
    """torch.autograd CartBondedLength operator."""

    params: Mapping[str, torch.Tensor]
    device: torch.device

    f: Callable = attr.ib()

    @f.default
    def _load_potential(self):
        from .potentials.compiled import rama

        return rama

    @classmethod
    def from_param_resolver(cls, param_resolver: RamaParamResolver):
        res = cls(
            params=asdict(param_resolver.bondlength_params),
            device=param_resolver.device,
        )
        assert all(res.device == t.device for t in res.params.values())
        return res

    @classmethod
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        return cls.from_param_resolver(
            param_resolver=CartBondedParamResolver.from_database(cb_database, device)
        )

    def score(self, atmpair_indices, parameter_indices, coords):
        E = CartBondedLengthFun(self)(atmpair_indices, parameter_indices, coords)
        return E
