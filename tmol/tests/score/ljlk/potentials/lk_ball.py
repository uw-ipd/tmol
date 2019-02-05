from typing import Tuple

from tmol.types.torch import Tensor

import torch

from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(modulename(__name__), relpaths(__file__, ["lk_ball.pybind.cpp"]))


def detach_maybe_requires_grad(
    *inputs: torch.tensor
) -> Tuple[bool, Tuple[torch.tensor, ...]]:
    requires_grad = any(t.requires_grad for t in inputs)

    if requires_grad:
        return requires_grad, tuple(t.detach() for t in inputs)
    else:
        return requires_grad, inputs


class BuildAcceptorWater(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: Tensor(float)[:, 3],
        B: Tensor(float)[:, 3],
        B0: Tensor(float)[:, 3],
        dist: float,
        angle: float,
        torsion: float,
    ) -> Tensor(float)[:, 3]:

        rgrad, (A, B, B0) = detach_maybe_requires_grad(A, B, B0)
        inputs = (A, B, B0, dist, angle, torsion)

        if rgrad:
            ctx.inputs = inputs

        return torch.from_numpy(_compiled.build_acc_water_V(*inputs))

    @staticmethod
    def backward(ctx, dE_dW: Tensor(float)[:, 3]):
        inputs = ctx.inputs
        dW_dA, dW_dB, dW_dB0 = map(
            torch.from_numpy, _compiled.build_acc_water_dV(*inputs)
        )

        return dW_dA @ dE_dW, dW_dB @ dE_dW, dW_dB0 @ dE_dW, None, None, None


class BuildDonorWater(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, D: Tensor(float)[:, 3], H: Tensor(float)[:, 3], dist: float
    ) -> Tensor(float)[:, 3]:

        rgrad, (D, H) = detach_maybe_requires_grad(D, H)
        inputs = (D, H, dist)

        if rgrad:
            ctx.inputs = inputs

        return torch.from_numpy(_compiled.build_don_water_V(*inputs))

    @staticmethod
    def backward(ctx, dE_dW: Tensor(float)[:, 3]):
        inputs = ctx.inputs

        dW_dD, dW_dH = map(torch.from_numpy, _compiled.build_don_water_dV(*inputs))

        return dW_dD @ dE_dW, dW_dH @ dE_dW, None


class GetLKFraction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coord_i: Tensor(float)[3],
        waters_j: Tensor(float)[2, 3],
        lj_radius_i: Tensor(float),
    ) -> Tensor(float):

        rgrad, (coord_i, waters_j) = detach_maybe_requires_grad(coord_i, waters_j)
        inputs = (coord_i, waters_j, lj_radius_i)

        if rgrad:
            ctx.inputs = inputs

        return torch.tensor(_compiled.lk_fraction_V(*inputs)).to(coord_i.dtype)

    @staticmethod
    def backward(ctx, dE_dF: Tensor(float)):
        inputs = ctx.inputs

        dF_dCI, dF_dWJ = map(torch.from_numpy, _compiled.lk_fraction_dV(*inputs))

        return dF_dCI * dE_dF, dF_dWJ * dE_dF, None
