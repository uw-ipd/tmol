import attr
from attr import asdict
from toolz import merge, valmap
from typing import Callable, Mapping

import torch

from tmol.types.functional import validate_args
from tmol.utility.args import ignore_unused_kwargs
from .params import ElecDatabase, ElecParamResolver


@attr.s(auto_attribs=True, frozen=True)
class ElecOp:
    device: torch.device
    params: Mapping[str, torch.Tensor]
    param_resolver: ElecParamResolver

    f: Callable = attr.ib()
    f_triu: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return ignore_unused_kwargs(compiled.elec)

    @f_triu.default
    def _load_f_triu(self):
        from .potentials import compiled

        return ignore_unused_kwargs(compiled.elec_triu)

    @classmethod
    def from_param_resolver(cls, param_resolver: ElecParamResolver):
        res = cls(
            param_resolver=param_resolver,
            params=asdict(param_resolver.global_params),
            device=param_resolver.device,
        )
        assert all(res.device == t.device for t in res.params.values())
        return res

    @classmethod
    @validate_args
    def from_database(cls, elec_database: ElecDatabase, device: torch.device):
        return cls.from_param_resolver(
            param_resolver=ElecParamResolver.from_database(elec_database, device)
        )

    def inter(self, coords_a, pcs_a, coords_b, pcs_b, rep_bpls):
        i, v = _ElecScoreFun(self, self.f)(coords_a, pcs_a, coords_b, pcs_b, rep_bpls)
        return (i.detach(), v)

    def intra(self, coords, pcs, rep_bpls):
        i, v = _ElecScoreFun(self, self.f_triu)(coords, pcs, coords, pcs, rep_bpls)
        return (i.detach(), v)


class _ElecScoreFun(torch.autograd.Function):
    def __init__(self, op, f):
        self.op = op
        self.f = f

    def forward(ctx, I, pc_I, J, pc_J, cp_bonded_path_lengths):
        assert I.dim() == 2
        assert I.shape[1] == 3
        assert I.shape[:1] == pc_I.shape
        assert not pc_I.requires_grad

        assert J.dim() == 2
        assert J.shape[1] == 3
        assert J.shape[:1] == pc_J.shape
        assert not pc_J.requires_grad

        params = valmap(
            lambda t: t.to(I.dtype)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
            else t,
            ctx.op.params,
        )

        inds, E, *dE_dC = ctx.f(I, pc_I, J, pc_J, cp_bonded_path_lengths, **params)

        # Assert of returned shape of indicies and scores. Seeing strange
        # results w/ reversed ordering if mgpu::tuple converted std::tuple
        assert inds.dim() == 2
        assert inds.shape[1] == 2
        assert inds.shape[0] == E.shape[0]

        inds = inds.transpose(0, 1)

        ctx.shape_I = pc_I.shape
        ctx.shape_J = pc_J.shape

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
