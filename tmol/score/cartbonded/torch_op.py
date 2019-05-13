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
        E, dE_dx = ctx.op.f(coords, atmpair_indices, parameter_indices, **ctx.op.params)
        ctx.save_for_backward(dE_dx)

        return E

    def backward(ctx, dV_dE):
        dE_dx, = ctx.saved_tensors
        return (dV_dE * dE_dx, None, None)


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
        E, dE_dx = ctx.op.f(
            coords, atmtriple_indices, parameter_indices, **ctx.op.params
        )
        ctx.save_for_backward(dE_dx)

        return E

    def backward(ctx, dV_dE):
        dE_dx, = ctx.saved_tensors
        return (dV_dE * dE_dx, None, None)


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
        E, dE_dx = ctx.op.f(coords, atmquad_indices, parameter_indices, **ctx.op.params)
        ctx.save_for_backward(dE_dx)

        return E

    def backward(ctx, dV_dE):
        dE_dx, = ctx.saved_tensors
        return (dV_dE * dE_dx, None, None)
