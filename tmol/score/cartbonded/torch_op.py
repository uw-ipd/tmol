import attr
from attr import asdict
from typing import Optional, Mapping, Callable

from .params import CartBondedDatabase, CartBondedParamResolver

import torch
import numpy


@attr.s(auto_attribs=True, frozen=True)
class CartBondedLengthOp:
    """torch.autograd CartBondedLength operator."""

    params: Mapping[str, torch.Tensor]
    device: torch.device

    f: Callable = attr.ib()

    @f.default
    def _load_potential(self):
        from .potentials.compiled import cartbonded_length

        return cartbonded_length

    @classmethod
    def from_param_resolver(cls, param_resolver: CartBondedParamResolver):
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


class CartBondedLengthFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, atmpair_indices, parameter_indices, coords):
        assert atmpair_indices.dim() == 2
        assert atmpair_indices.shape[1] == 2
        assert parameter_indices.dim() == 1
        assert parameter_indices.shape[0] == atmpair_indices.shape[0]

        ctx.coords_shape = coords.size()
        print("in", ctx.coords_shape)
        E, dE_dA, dE_dB = ctx.op.f(
            atmpair_indices, parameter_indices, coords, **ctx.op.params
        )

        atmpair_indices = atmpair_indices.transpose(0, 1)  # coo_tensor wants this
        ctx.save_for_backward(atmpair_indices, dE_dA, dE_dB)

        return E

    def backward(ctx, dV_dE):
        ind_ij, dE_dI, dE_dJ = ctx.saved_tensors

        dV_dA = torch.sparse_coo_tensor(
            ind_ij[0, None, :], dV_dE[..., None] * dE_dI, (ctx.coords_shape)
        ).to_dense()
        dV_dB = torch.sparse_coo_tensor(
            ind_ij[1, None, :], dV_dE[..., None] * dE_dJ, (ctx.coords_shape)
        ).to_dense()

        return (None, None, (dV_dA + dV_dB))
