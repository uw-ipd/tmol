import attr
from attr import asdict
from typing import Mapping, Callable

from .params import CartBondedDatabase, CartBondedParamResolver

import torch


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

    def score(self, coords, atmpair_indices, parameter_indices):
        E = CartBondedLengthFun(self)(coords, atmpair_indices, parameter_indices)
        return E


@attr.s(auto_attribs=True, frozen=True)
class CartBondedAngleOp:
    """torch.autograd CartBondedAngle operator."""

    params: Mapping[str, torch.Tensor]
    device: torch.device

    f: Callable = attr.ib()

    @f.default
    def _load_potential(self):
        from .potentials.compiled import cartbonded_angle

        return cartbonded_angle

    @classmethod
    def from_param_resolver(cls, param_resolver: CartBondedParamResolver):
        res = cls(
            params=asdict(param_resolver.bondangle_params), device=param_resolver.device
        )
        assert all(res.device == t.device for t in res.params.values())
        return res

    @classmethod
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        return cls.from_param_resolver(
            param_resolver=CartBondedParamResolver.from_database(cb_database, device)
        )

    def score(self, coords, atmtriple_indices, parameter_indices):
        E = CartBondedAngleFun(self)(coords, atmtriple_indices, parameter_indices)
        return E


@attr.s(auto_attribs=True, frozen=True)
class CartBondedTorsionOp:
    """torch.autograd CartBondedTorsion operator."""

    params: Mapping[str, torch.Tensor]
    device: torch.device

    f: Callable = attr.ib()

    @f.default
    def _load_potential(self):
        from .potentials.compiled import cartbonded_torsion

        return cartbonded_torsion

    @classmethod
    def from_param_resolver(cls, param_resolver: CartBondedParamResolver):
        res = cls(
            params=asdict(param_resolver.torsion_params), device=param_resolver.device
        )
        assert all(res.device == t.device for t in res.params.values())
        return res

    @classmethod
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        return cls.from_param_resolver(
            param_resolver=CartBondedParamResolver.from_database(cb_database, device)
        )

    def score(self, coords, atmquad_indices, parameter_indices):
        E = CartBondedTorsionFun(self)(coords, atmquad_indices, parameter_indices)
        return E


@attr.s(auto_attribs=True, frozen=True)
class CartBondedImproperOp:
    """torch.autograd CartBondedImproper operator."""

    params: Mapping[str, torch.Tensor]
    device: torch.device

    f: Callable = attr.ib()

    @f.default
    def _load_potential(self):
        from .potentials.compiled import cartbonded_torsion

        return cartbonded_torsion

    @classmethod
    def from_param_resolver(cls, param_resolver: CartBondedParamResolver):
        res = cls(
            params=asdict(param_resolver.improper_params), device=param_resolver.device
        )
        assert all(res.device == t.device for t in res.params.values())
        return res

    @classmethod
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        return cls.from_param_resolver(
            param_resolver=CartBondedParamResolver.from_database(cb_database, device)
        )

    def score(self, coords, atmquad_indices, parameter_indices):
        E = CartBondedTorsionFun(self)(coords, atmquad_indices, parameter_indices)
        return E


@attr.s(auto_attribs=True, frozen=True)
class CartBondedHxlTorsionOp:
    """torch.autograd CartBondedHxlTorsion operator."""

    params: Mapping[str, torch.Tensor]
    device: torch.device

    f: Callable = attr.ib()

    @f.default
    def _load_potential(self):
        from .potentials.compiled import cartbonded_hxltorsion

        return cartbonded_hxltorsion

    @classmethod
    def from_param_resolver(cls, param_resolver: CartBondedParamResolver):
        res = cls(
            params=asdict(param_resolver.hxltorsion_params),
            device=param_resolver.device,
        )
        assert all(res.device == t.device for t in res.params.values())
        return res

    @classmethod
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        return cls.from_param_resolver(
            param_resolver=CartBondedParamResolver.from_database(cb_database, device)
        )

    def score(self, coords, atmquad_indices, parameter_indices):
        E = CartBondedTorsionFun(self)(coords, atmquad_indices, parameter_indices)
        return E


class CartBondedLengthFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, coords, atmpair_indices, parameter_indices):
        assert atmpair_indices.dim() == 2
        assert atmpair_indices.shape[1] == 2
        assert parameter_indices.dim() == 1
        assert parameter_indices.shape[0] == atmpair_indices.shape[0]

        ctx.coords_shape = coords.size()
        E, dE_dA, dE_dB = ctx.op.f(
            coords, atmpair_indices, parameter_indices, **ctx.op.params
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

        return ((dV_dA + dV_dB), None, None)


class CartBondedAngleFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, coords, atmtriple_indices, parameter_indices):
        assert atmtriple_indices.dim() == 2
        assert atmtriple_indices.shape[1] == 3
        assert parameter_indices.dim() == 1
        assert parameter_indices.shape[0] == atmtriple_indices.shape[0]

        ctx.coords_shape = coords.size()
        E, dE_dA, dE_dB, dE_dC = ctx.op.f(
            coords, atmtriple_indices, parameter_indices, **ctx.op.params
        )

        atmtriple_indices = atmtriple_indices.transpose(0, 1)  # coo_tensor wants this
        ctx.save_for_backward(atmtriple_indices, dE_dA, dE_dB, dE_dC)

        return E

    def backward(ctx, dV_dE):
        ind_ijk, dE_dI, dE_dJ, dE_dK = ctx.saved_tensors

        dV_dA = torch.sparse_coo_tensor(
            ind_ijk[0, None, :], dV_dE[..., None] * dE_dI, (ctx.coords_shape)
        ).to_dense()
        dV_dB = torch.sparse_coo_tensor(
            ind_ijk[1, None, :], dV_dE[..., None] * dE_dJ, (ctx.coords_shape)
        ).to_dense()
        dV_dC = torch.sparse_coo_tensor(
            ind_ijk[2, None, :], dV_dE[..., None] * dE_dK, (ctx.coords_shape)
        ).to_dense()

        return ((dV_dA + dV_dB + dV_dC), None, None)


class CartBondedTorsionFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, coords, atmquad_indices, parameter_indices):
        assert atmquad_indices.dim() == 2
        assert atmquad_indices.shape[1] == 4
        assert parameter_indices.dim() == 1
        assert parameter_indices.shape[0] == atmquad_indices.shape[0]

        ctx.coords_shape = coords.size()
        E, dE_dA, dE_dB, dE_dC, dE_dD = ctx.op.f(
            coords, atmquad_indices, parameter_indices, **ctx.op.params
        )

        atmquad_indices = atmquad_indices.transpose(0, 1)  # coo_tensor wants this
        ctx.save_for_backward(atmquad_indices, dE_dA, dE_dB, dE_dC, dE_dD)

        return E

    def backward(ctx, dV_dE):
        ind_ijkl, dE_dI, dE_dJ, dE_dK, dE_dL = ctx.saved_tensors

        dV_dA = torch.sparse_coo_tensor(
            ind_ijkl[0, None, :], dV_dE[..., None] * dE_dI, (ctx.coords_shape)
        ).to_dense()
        dV_dB = torch.sparse_coo_tensor(
            ind_ijkl[1, None, :], dV_dE[..., None] * dE_dJ, (ctx.coords_shape)
        ).to_dense()
        dV_dC = torch.sparse_coo_tensor(
            ind_ijkl[2, None, :], dV_dE[..., None] * dE_dK, (ctx.coords_shape)
        ).to_dense()
        dV_dD = torch.sparse_coo_tensor(
            ind_ijkl[3, None, :], dV_dE[..., None] * dE_dL, (ctx.coords_shape)
        ).to_dense()

        return ((dV_dA + dV_dB + dV_dC + dV_dD), None, None)
