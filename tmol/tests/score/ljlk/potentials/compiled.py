import torch

from tmol.tests.score.ljlk.potentials._ext import (  # noqa: F401
    lj_score_V,
    lj_score_V_dV,
    lk_isotropic_score_V,
    lk_isotropic_score_V_dV,
    lj_sigma,
    vdw_V,
    vdw_V_dV,
    f_desolv_V,
    f_desolv_V_dV,
)


class LJScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist, *args):
        if dist.requires_grad:
            Vatr, Vrep, dVatr, dVrep = torch.tensor(
                [lj_score_V_dV(d, *args) for d in dist.reshape(-1)]
            ).transpose(0, 1)
            V = Vatr + Vrep
            dV = dVatr + dVrep

            V = V.to(dist.dtype).reshape(dist.shape)
            dV = dV.to(dist.dtype).reshape(dist.shape)

            ctx.save_for_backward(dV)
        else:
            Vatr, Vrep = torch.tensor(
                [lj_score_V(d, *args) for d in dist.reshape(-1)]
            ).transpose(0, 1)
            V = Vatr + Vrep

            V = V.to(dist.dtype).reshape(dist.shape)
        return V

    @staticmethod
    def backward(ctx, dE_dV):
        (dV,) = ctx.saved_tensors
        return dE_dV * dV, None, None, None, None


class LKScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist, *args):
        if dist.requires_grad:
            V, dV = torch.tensor(
                [lk_isotropic_score_V_dV(d, *args) for d in dist.reshape(-1)]
            ).transpose(0, 1)

            V = V.to(dist.dtype).reshape(dist.shape)
            dV = dV.to(dist.dtype).reshape(dist.shape)

            ctx.save_for_backward(dV)
        else:
            V = torch.tensor([lk_isotropic_score_V(d, *args) for d in dist.reshape(-1)])

            V = V.to(dist.dtype).reshape(dist.shape)
        return V

    @staticmethod
    def backward(ctx, dE_dV):
        (dV,) = ctx.saved_tensors
        return dE_dV * dV, None, None, None, None
