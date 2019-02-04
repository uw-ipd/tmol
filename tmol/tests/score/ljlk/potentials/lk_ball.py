from typing import Tuple

from tmol.types.torch import Tensor

import torch

from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(modulename(__name__), relpaths(__file__, ["lk_ball.pybind.cpp"]))

build_acc_waters = _compiled.build_acc_waters


def detach_maybe_requires_grad(
    *inputs: torch.tensor
) -> Tuple[bool, Tuple[torch.tensor, ...]]:
    requires_grad = any(t.requires_grad for t in inputs)

    if requires_grad:
        return requires_grad, tuple(t.detach() for t in inputs)
    else:
        return requires_grad, inputs


class BuildDonorWater(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, D: Tensor(float)[:, 3], H: Tensor(float)[:, 3], dist: float
    ) -> Tensor(float)[:, 3]:

        rgrad, (D, H) = detach_maybe_requires_grad(D, H)
        if rgrad:
            ctx.inputs = (D, H, dist)

        return torch.from_numpy(_compiled.build_don_water_V(D, H, dist))

    @staticmethod
    def backward(ctx, dE_dV: Tensor(float)[:, 3]):
        D, H, dist = ctx.inputs
        dV_dD, dV_dH = map(torch.from_numpy, _compiled.build_don_water_dV(D, H, dist))

        return dV_dD @ dE_dV, dV_dH @ dE_dV, None
