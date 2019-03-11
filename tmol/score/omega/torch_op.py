import attr
from typing import Callable

import torch


@attr.s(auto_attribs=True, frozen=True)
class OmegaOp:
    device: torch.device
    f: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return compiled.omega

    @classmethod
    def from_device(cls, device: torch.device):
        res = cls(device=device)
        return res

    def intra(self, coords, omega_indices, K):
        E = OmegaScoreFun(self)(coords, omega_indices, K)
        return E


class OmegaScoreFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, coords, omega_indices, K):
        assert omega_indices.dim() == 2
        assert omega_indices.shape[1] == 4

        ctx.coords_shape = coords.size()

        # dE_dphi/psi are returned as ntors x 12 arrays
        E, dE_domegas = ctx.op.f(coords, omega_indices, K)

        omega_indices = omega_indices.transpose(0, 1)  # coo_tensor wants this
        dE_domegas = dE_domegas.reshape([-1, 3, 4])
        ctx.save_for_backward(omega_indices, dE_domegas)

        return E

    def backward(ctx, dV_dE):
        omega_indices, dE_domegas = ctx.saved_tensors

        dVdA = torch.zeros(
            ctx.coords_shape, dtype=torch.float, device=dE_domegas.device
        )
        for i in range(4):
            dVdA += torch.sparse_coo_tensor(
                omega_indices[i, None, :],
                dV_dE[..., None] * dE_domegas[..., i],
                (ctx.coords_shape),
            ).to_dense()

        print(dVdA[:10])
        return (dVdA, None, None)
