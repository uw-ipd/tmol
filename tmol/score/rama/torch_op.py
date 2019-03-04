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
            device=param_resolver.device,
        )
        assert all(
            res.device == t.device
            for t in res.params.values()
            if not isinstance(t, list)
        )
        assert all(
            all(res.device == t.device for t in l)
            for l in res.params.values()
            if isinstance(l, list)
        )
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
        assert phi_indices.dim() == 2
        assert phi_indices.shape[1] == 4
        assert psi_indices.dim() == 2
        assert psi_indices.shape[1] == 4
        assert parameter_indices.dim() == 1
        assert parameter_indices.shape[0] == phi_indices.shape[0]
        assert parameter_indices.shape[0] == psi_indices.shape[0]

        ctx.coords_shape = coords.size()

        # dE_dphi/psi are returned as ntors x 12 arrays
        E, dE_dphis, dE_dpsis = ctx.op.f(
            coords, phi_indices, psi_indices, parameter_indices, **ctx.op.params
        )

        phi_indices = phi_indices.transpose(0, 1)  # coo_tensor wants this
        psi_indices = psi_indices.transpose(0, 1)  # coo_tensor wants this
        ctx.save_for_backward(phi_indices, psi_indices, dE_dphis, dE_dpsis)

        return E

    def backward(ctx, dV_dE):
        phi_indices, psi_indices, dE_dphis, dE_dpsis = ctx.saved_tensors

        dVdA = torch.zeros(ctx.coords_shape, dtype=torch.float, device=dE_dphis.device)
        for i in range(4):
            dVdA += torch.sparse_coo_tensor(
                phi_indices[i, None, :],
                dV_dE[..., None] * dE_dphis[..., 3 * i : (3 * i + 3)],
                (ctx.coords_shape),
            ).to_dense()
            dVdA += torch.sparse_coo_tensor(
                psi_indices[i, None, :],
                dV_dE[..., None] * dE_dpsis[..., 3 * i : (3 * i + 3)],
                (ctx.coords_shape),
            ).to_dense()

        return (dVdA, None, None, None)
