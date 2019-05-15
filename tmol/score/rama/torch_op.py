import attr
from attr import asdict
from typing import Mapping, Callable

from .params import RamaDatabase, RamaParamResolver

import torch


@attr.s(auto_attribs=True, frozen=True)
class RamaOp:
    device: torch.device
    params: Mapping[str, torch.Tensor]
    param_resolver: RamaParamResolver

    f: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return compiled.rama

    @classmethod
    def from_param_resolver(cls, param_resolver: RamaParamResolver):
        res = cls(
            param_resolver=param_resolver,
            params=asdict(param_resolver.rama_params),
            device=param_resolver.rama_params.tables.device,  # param_resolver.device,
        )
        assert all(res.device == t.device for t in res.params.values())
        return res

    @classmethod
    def from_database(cls, rama_database: RamaDatabase, device: torch.device):
        return cls.from_param_resolver(
            param_resolver=RamaParamResolver.from_database(rama_database, device)
        )

    def intra(self, coords, phi_indices, psi_indices, parameter_indices):
        E = RamaScoreFun(self)(coords, phi_indices, psi_indices, parameter_indices)
        return E


class RamaScoreFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, coords, phi_indices, psi_indices, parameter_indices):
        # dE_dphi/psi are returned as ntors x 12 arrays
        E, dE_dx = ctx.op.f(
            coords, phi_indices, psi_indices, parameter_indices, **ctx.op.params
        )

        ctx.save_for_backward(dE_dx)

        return E

    def backward(ctx, dV_dE):
        dE_dx, = ctx.saved_tensors

        return (dV_dE * dE_dx, None, None, None)
