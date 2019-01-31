import attr
from attr import asdict
from toolz import merge, valmap
from typing import Callable, Mapping

import torch

from tmol.types.functional import validate_args
from tmol.utility.args import ignore_unused_kwargs

from .params import LJLKDatabase, LJLKParamResolver


@attr.s(auto_attribs=True, frozen=True)
class AtomOp:
    """torch.autograd atom pair score baseline operator."""

    device: torch.device
    param_resolver: LJLKParamResolver
    params: Mapping[str, torch.Tensor]

    @classmethod
    @validate_args
    def from_param_resolver(cls, param_resolver: LJLKParamResolver):
        res = cls(
            param_resolver=param_resolver,
            params=merge(
                asdict(param_resolver.global_params), asdict(param_resolver.type_params)
            ),
            device=param_resolver.device,
        )

        assert all(res.device == t.device for t in res.params.values())

        return res

    @classmethod
    @validate_args
    def from_database(cls, ljlk_database: LJLKDatabase, device: torch.device):

        return cls.from_param_resolver(
            param_resolver=LJLKParamResolver.from_database(ljlk_database, device)
        )

    def inter(
        self, coords_a, atom_types_a, coords_b, atom_types_b, bonded_path_lengths
    ):
        # Detach grad from output indicies, which are int-valued
        i, v = _AtomScoreFun(self, self.f)(
            coords_a, atom_types_a, coords_b, atom_types_b, bonded_path_lengths
        )

        return (i.detach(), v)

    def intra(self, coords, atom_types, bonded_path_lengths):
        # Call triu score function on coords, results in idependent autograd
        # paths for i and j axis
        i, v = _AtomScoreFun(self, self.f_triu)(
            coords, atom_types, coords, atom_types, bonded_path_lengths
        )

        # Detach grad from output indicies, which are int-valued
        return (i.detach(), v)


class _AtomScoreFun(torch.autograd.Function):
    def __init__(self, op, f):
        self.op = op
        self.f = f
        super().__init__()

    def forward(ctx, I, atom_type_I, J, atom_type_J, bonded_path_lengths):
        assert I.dim() == 2
        assert I.shape[1] == 3
        assert I.shape[:1] == atom_type_I.shape
        assert not atom_type_I.requires_grad

        assert J.dim() == 2
        assert J.shape[1] == 3
        assert J.shape[:1] == atom_type_J.shape
        assert not atom_type_J.requires_grad

        params = valmap(
            lambda t: t.to(I.dtype)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
            else t,
            ctx.op.params,
        )

        inds, E, *dE_dC = ctx.f(
            I, atom_type_I, J, atom_type_J, bonded_path_lengths, **params
        )

        # Assert of returned shape of indicies and scores. Seeing strange
        # results w/ reversed ordering if mgpu::tuple converted std::tuple
        assert inds.dim() == 2
        assert inds.shape[1] == 2
        assert inds.shape[0] == E.shape[0]

        inds = inds.transpose(0, 1)

        ctx.shape_I = atom_type_I.shape
        ctx.shape_J = atom_type_J.shape

        ctx.save_for_backward(*([inds] + dE_dC))

        return (inds, E)

    def backward(ctx, _ind_grads, dV_dE):
        ind, dE_dI, dE_dJ = ctx.saved_tensors
        ind_I, ind_J = ind

        dV_dI = torch.sparse_coo_tensor(
            ind_I[None, :], dV_dE[..., None] * dE_dI, ctx.shape_I + (3,)
        ).to_dense()

        dV_dJ = torch.sparse_coo_tensor(
            ind_J[None, :], dV_dE[..., None] * dE_dJ, ctx.shape_J + (3,)
        ).to_dense()

        return (dV_dI, None, dV_dJ, None, None)


@attr.s(auto_attribs=True, frozen=True)
class LJOp(AtomOp):
    f: Callable = attr.ib()
    f_triu: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return ignore_unused_kwargs(compiled.lj)

    @f_triu.default
    def _load_f_triu(self):
        from .potentials import compiled

        return ignore_unused_kwargs(compiled.lj_triu)


@attr.s(auto_attribs=True, frozen=True)
class LKOp(AtomOp):
    """torch.autograd hbond baseline operator."""

    f: Callable = attr.ib()
    f_triu: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return ignore_unused_kwargs(compiled.lk_isotropic)

    @f_triu.default
    def _load_f_triu(self):
        from .potentials import compiled

        return ignore_unused_kwargs(compiled.lk_isotropic_triu)
